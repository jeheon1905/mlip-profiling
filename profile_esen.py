"""
Profiling script for eSEN model with Chrome Trace export for Perfetto visualization.

Supports various system sizes from small molecules to large bulk structures (up to 1000+ atoms).

Usage:
    # Profile with default test cases
    python profile_with_trace.py --device cuda
    
    # Profile specific atom counts
    python profile_with_trace.py --device cuda --natoms 100 200 500 1000
    
    # Profile with custom bulk structure
    python profile_with_trace.py --device cuda --bulk-element Cu --bulk-sizes 108 256 500

After running:
    1. Open https://ui.perfetto.dev
    2. Click "Open trace file"
    3. Upload the generated .json trace file
    4. Analyze the timeline visualization

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from ase import Atoms
from ase.build import molecule, bulk, make_supercell
from ase.io import read as ase_read
from torch.profiler import ProfilerActivity, profile, schedule

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_default,
)


# Available models for profiling
AVAILABLE_MODELS = [
    "uma-s-1",
    "uma-s-1p1",
    "uma-m-1p1",
    "esen-md-direct-all-omol",
    "esen-sm-conserving-all-omol",
    "esen-sm-direct-all-omol",
    "esen-sm-conserving-all-oc25",
    "esen-md-direct-all-oc25",
    "esen-sm-filtered-odac25",
    "esen-sm-full-odac25",
]

DEFAULT_MODEL = "esen-sm-conserving-all-omol"


# =============================================================================
# System Builders
# =============================================================================

def build_molecule(name: str) -> Atoms:
    """Build a molecule from ASE's database."""
    atoms = molecule(name)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def build_bulk_supercell(element: str, target_natoms: int, structure: str = "fcc") -> Atoms:
    """
    Build a bulk supercell with approximately target_natoms atoms.
    
    Args:
        element: Element symbol (e.g., "Cu", "Al", "Pt")
        target_natoms: Target number of atoms
        structure: Crystal structure ("fcc", "bcc", "hcp", etc.)
    
    Returns:
        ASE Atoms object with approximately target_natoms atoms
    """
    # Create unit cell
    unit = bulk(element, structure)
    atoms_per_unit = len(unit)
    
    # Calculate supercell size to get close to target
    # For cubic cells: n^3 * atoms_per_unit ≈ target_natoms
    n = int(np.ceil((target_natoms / atoms_per_unit) ** (1/3)))
    
    # Create supercell matrix
    supercell_matrix = [[n, 0, 0], [0, n, 0], [0, 0, n]]
    atoms = make_supercell(unit, supercell_matrix)
    
    # Trim to exact target if needed (optional)
    if len(atoms) > target_natoms:
        atoms = atoms[:target_natoms]
    
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0  # Bulk typically has spin 0
    
    return atoms


def build_random_cluster(element: str, natoms: int, spacing: float = 2.5) -> Atoms:
    """
    Build a random atomic cluster with specified number of atoms.
    
    Args:
        element: Element symbol
        natoms: Number of atoms
        spacing: Approximate spacing between atoms in Angstroms
    
    Returns:
        ASE Atoms object
    """
    # Create random positions in a spherical volume
    # Volume ~ natoms * spacing^3, so radius ~ (natoms)^(1/3) * spacing
    radius = (natoms ** (1/3)) * spacing
    
    positions = []
    np.random.seed(42)  # Reproducibility
    
    for _ in range(natoms):
        # Random point in sphere
        while True:
            pos = np.random.uniform(-radius, radius, 3)
            if np.linalg.norm(pos) <= radius:
                positions.append(pos)
                break
    
    atoms = Atoms(
        symbols=[element] * natoms,
        positions=positions,
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    
    return atoms


def atoms_to_batch(
    atoms: Atoms,
    cutoff: float,
    max_neighbors: int,
    external_graph_gen: bool,
) -> AtomicData:
    """Convert ASE Atoms to AtomicData batch."""
    data = AtomicData.from_ase(
        atoms,
        r_edges=external_graph_gen,
        radius=cutoff,
        max_neigh=max_neighbors if external_graph_gen else None,
        r_data_keys=["charge", "spin"],
        task_name="omol",
    )
    data.pos.requires_grad_(True)
    return atomicdata_list_to_batch([data])


def atoms_list_to_batch(
    atoms_list: list[Atoms],
    cutoff: float,
    max_neighbors: int,
    external_graph_gen: bool,
) -> AtomicData:
    """Convert a list of ASE Atoms to a batched AtomicData."""
    data_list = []
    for atoms in atoms_list:
        data = AtomicData.from_ase(
            atoms,
            r_edges=external_graph_gen,
            radius=cutoff,
            max_neigh=max_neighbors if external_graph_gen else None,
            r_data_keys=["charge", "spin"],
            task_name="omol",
        )
        data.pos.requires_grad_(True)
        data_list.append(data)
    return atomicdata_list_to_batch(data_list)


# =============================================================================
# Test Case Generators
# =============================================================================

def get_molecule_test_cases(
    molecules: list[str] | None = None,
) -> list[tuple[str, Atoms]]:
    """Get molecule test cases from ASE's database.
    
    Args:
        molecules: List of molecule names (e.g., ['H2O', 'CH4', 'C6H6'])
                   If None, uses default set.
    """
    if molecules is None:
        molecules = ["H2O", "CH4", "C6H6", "C60"]
    return [(name, build_molecule(name)) for name in molecules]


def get_bulk_test_cases(
    element: str = "Cu",
    sizes: list[int] | None = None,
) -> list[tuple[str, Atoms]]:
    """Get bulk supercell test cases with various sizes."""
    if sizes is None:
        sizes = [32, 108, 256, 500, 864]
    
    test_cases = []
    for n in sizes:
        atoms = build_bulk_supercell(element, n)
        actual_n = len(atoms)
        name = f"{element}_{actual_n}atoms"
        test_cases.append((name, atoms))
    
    return test_cases


def get_cluster_test_cases(
    element: str = "Cu",
    sizes: list[int] | None = None,
) -> list[tuple[str, Atoms]]:
    """Get random cluster test cases with various sizes."""
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000]
    
    return [
        (f"cluster_{element}_{n}", build_random_cluster(element, n))
        for n in sizes
    ]


def get_structure_file_test_cases(
    file_paths: list[str],
) -> list[tuple[str, Atoms]]:
    """
    Load structures from XYZ, CIF, or other ASE-supported files.
    
    Args:
        file_paths: List of paths to structure files (xyz, cif, etc.)
    
    Returns:
        List of (name, Atoms) tuples
    """
    test_cases = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        try:
            atoms = ase_read(file_path)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
        
        # Set default charge/spin if not present
        if "charge" not in atoms.info:
            atoms.info["charge"] = 0
        if "spin" not in atoms.info:
            atoms.info["spin"] = 0
        
        natoms = len(atoms)
        base_name = path.stem
        name = f"{base_name}_{natoms}atoms"
        test_cases.append((name, atoms))
    
    return test_cases


def apply_batching(
    test_cases: list[tuple[str, Atoms]],
    batch_sizes: list[int],
    copy_fn: callable | None = None,
) -> list[tuple[str, list[Atoms]]]:
    """
    Convert single-structure test cases to batched test cases.
    
    Args:
        test_cases: List of (name, Atoms) tuples
        batch_sizes: List of batch sizes to create
        copy_fn: Optional function to create copies of atoms.
                 If None, uses atoms.copy()
    
    Returns:
        List of (name, list[Atoms]) tuples for batched inference
    """
    batched_cases = []
    
    for name, atoms in test_cases:
        natoms = len(atoms)
        
        for bs in batch_sizes:
            # Create copies for batch
            if copy_fn:
                atoms_list = [copy_fn() for _ in range(bs)]
            else:
                atoms_list = [atoms.copy() for _ in range(bs)]
                # Ensure charge/spin are set on copies
                for a in atoms_list:
                    if "charge" not in a.info:
                        a.info["charge"] = atoms.info.get("charge", 0)
                    if "spin" not in a.info:
                        a.info["spin"] = atoms.info.get("spin", 0)
            
            total_atoms = natoms * bs
            batch_name = f"{name}_x{bs}_batch_{total_atoms}total"
            batched_cases.append((batch_name, atoms_list))
    
    return batched_cases


# =============================================================================
# Profiling Functions
# =============================================================================

# Key operations to track (record_function tags)
# These are the actual tag names found in the eSEN model code
# From escn_md.py, escn_md_block.py, and mlip_unit.py
TRACKED_OPERATIONS = [
    # mlip_unit.py (top-level)
    "element_refs",
    "forward",
    # escn_md.py (backbone)
    "get_displacement_and_cell",
    "generate_graph",
    "charge spin dataset embeddings",
    "obtain wigner",
    "obtain rotmat wigner original",
    "atom embedding",
    "edge embedding",
    "message passing 0",
    "message passing 1",
    "message passing 2",
    "message passing 3",
    # escn_md_block.py (message passing block)
    "SO2Conv",
    "edgewise",
    "atomwise",
]


def extract_operation_times_from_trace(trace_path: Path, active_steps: int) -> dict:
    """
    Extract timing directly from Chrome Trace file (same as Perfetto).
    
    Extracts both CPU and GPU times:
    - 'user_annotation': CPU time (kernel launch overhead)
    - 'gpu_user_annotation': GPU time (actual kernel execution)
    
    Args:
        trace_path: Path to the Chrome Trace JSON file
        active_steps: Number of active profiling steps
    
    Returns:
        Dictionary with operation names as keys and timing info as values.
        Each entry contains cpu_time_ms and gpu_time_ms.
    """
    import json
    
    with open(trace_path) as f:
        trace_data = json.load(f)
    
    # Collect durations for each operation, separated by CPU/GPU
    cpu_durations = {}  # name -> list of durations in ms
    gpu_durations = {}  # name -> list of durations in ms
    
    for event in trace_data.get("traceEvents", []):
        name = event.get("name", "")
        phase = event.get("ph", "")
        category = event.get("cat", "")
        
        # Only count complete events (X) from tracked operations
        if phase == "X" and name in TRACKED_OPERATIONS:
            dur_ms = event.get("dur", 0) / 1000.0  # μs to ms
            
            if category == "user_annotation":
                # CPU time (kernel launch)
                if name not in cpu_durations:
                    cpu_durations[name] = []
                cpu_durations[name].append(dur_ms)
            elif category == "gpu_user_annotation":
                # GPU time (actual kernel execution)
                if name not in gpu_durations:
                    gpu_durations[name] = []
                gpu_durations[name].append(dur_ms)
    
    # Calculate per-step averages
    all_names = set(cpu_durations.keys()) | set(gpu_durations.keys())
    operation_times = {}
    
    for name in all_names:
        # CPU stats
        cpu_durs = cpu_durations.get(name, [])
        cpu_total_ms = sum(cpu_durs)
        cpu_count = len(cpu_durs)
        
        # GPU stats
        gpu_durs = gpu_durations.get(name, [])
        gpu_total_ms = sum(gpu_durs)
        gpu_count = len(gpu_durs)
        
        # Use GPU count for calls_per_step if available, else CPU
        count = gpu_count if gpu_count > 0 else cpu_count
        calls_per_step = count // active_steps if active_steps > 0 else count
        
        operation_times[name] = {
            # GPU time (what Perfetto shows prominently)
            "gpu_time_ms": gpu_total_ms / active_steps if active_steps > 0 else gpu_total_ms,
            "gpu_total_ms": gpu_total_ms,
            "gpu_per_call_ms": gpu_total_ms / gpu_count if gpu_count > 0 else 0,
            # CPU time (kernel launch overhead)
            "cpu_time_ms": cpu_total_ms / active_steps if active_steps > 0 else cpu_total_ms,
            "cpu_total_ms": cpu_total_ms,
            "cpu_per_call_ms": cpu_total_ms / cpu_count if cpu_count > 0 else 0,
            # Call counts
            "count": calls_per_step,
            "total_count": count,
            # Legacy field (now points to GPU time for backward compatibility with Perfetto)
            "wall_time_ms": gpu_total_ms / active_steps if gpu_count > 0 else cpu_total_ms / active_steps if active_steps > 0 else 0,
        }
    
    print(f"  Extracted operation times from trace:")
    for name, times in operation_times.items():
        print(f"    {name}: GPU={times['gpu_time_ms']:.2f}ms, CPU={times['cpu_time_ms']:.2f}ms, calls={times['count']}/step")
    return operation_times


def trace_handler(output_dir: Path, name: str):
    """Create a trace handler that saves Chrome trace files."""
    def handler(prof):
        trace_path = output_dir / f"{name}.trace.json"
        print(f"  Saving trace: {trace_path}")
        prof.export_chrome_trace(str(trace_path))
    return handler


def run_profiling(
    device: str,
    output_dir: Path,
    test_cases: list[tuple[str, Atoms | list[Atoms]]],
    model_name: str = DEFAULT_MODEL,
    wait_steps: int = 2,
    warmup_steps: int = 3,
    active_steps: int = 5,
) -> dict:
    """
    Run profiling with Chrome trace export.
    
    Profiler schedule explanation:
    ┌─────────────────────────────────────────────────────────────┐
    │  wait_steps  │  warmup_steps  │      active_steps          │
    │  (ignored)   │  (not saved)   │  ★ RECORDED TO TRACE ★     │
    └─────────────────────────────────────────────────────────────┘
    
    - wait: Skip initial unstable iterations (JIT compilation, etc.)
    - warmup: Profiler runs but doesn't save (GPU cache warming)
    - active: Actually recorded to the trace file
    
    Args:
        device: "cpu" or "cuda"
        output_dir: Directory to save trace files
        test_cases: List of (name, Atoms) or (name, list[Atoms]) tuples.
                    If list[Atoms], they will be batched together.
        model_name: Name of the model to profile
        wait_steps: Steps to skip
        warmup_steps: Warmup steps (not recorded)
        active_steps: Steps to record
    
    Returns:
        Dictionary with profiling results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Schedule: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}")
    print(f"Test cases: {len(test_cases)}")
    print("-" * 60)
    
    # Load model
    settings = inference_settings_default()
    predictor = pretrained_mlip.get_predict_unit(
        model_name=model_name,
        inference_settings=settings,
        device=device,
    )
    
    cutoff = predictor.model.module.backbone.cutoff
    max_neighbors = predictor.model.module.backbone.max_neighbors
    external_graph_gen = bool(predictor.inference_settings.external_graph_gen)
    
    total_steps = wait_steps + warmup_steps + active_steps
    
    # Configure profiler activities
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    results = {}
    
    for name, atoms_or_list in test_cases:
        # Handle both single Atoms and list of Atoms (batched)
        if isinstance(atoms_or_list, list):
            # Batched case: list of Atoms
            atoms_list = atoms_or_list
            batch_size = len(atoms_list)
            natoms = sum(len(a) for a in atoms_list)
            is_batched = True
            
            # Convert list to batch
            data = atoms_list_to_batch(
                atoms_list=atoms_list,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                external_graph_gen=external_graph_gen,
            )
        else:
            # Single Atoms case
            atoms = atoms_or_list
            batch_size = 1
            natoms = len(atoms)
            is_batched = False
            
            # Convert to batch
            data = atoms_to_batch(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                external_graph_gen=external_graph_gen,
            )
        
        data = data.to(device)
        
        # Get edge count (after first forward pass for internal graph gen)
        with torch.no_grad():
            predictor.predict(data)
            num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        
        print(f"\n[{name}] Profiling...")
        print(f"  atoms={natoms}, batch_size={batch_size}, edges={num_edges}")
        
        # Create profiler schedule
        prof_schedule = schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        )
        
        # Profile with trace export
        latencies_ms = []
        
        with profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=trace_handler(output_dir, name),
            record_shapes=True,
            with_stack=True,
            profile_memory=False,
        ) as prof:
            for step in range(total_steps):
                t0 = time.perf_counter()
                predictor.predict(data)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                prof.step()
        
        # Extract per-operation timing from Chrome Trace (matches Perfetto exactly)
        trace_path = output_dir / f"{name}.trace.json"
        operation_times = extract_operation_times_from_trace(trace_path, active_steps)
        
        # Calculate stats (only from active steps)
        active_latencies = latencies_ms[wait_steps + warmup_steps:]
        mean_latency = sum(active_latencies) / len(active_latencies)
        
        results[name] = {
            "natoms": natoms,
            "batch_size": batch_size,
            "num_edges": num_edges,
            "is_batched": is_batched,
            "mean_latency_ms": mean_latency,
            "min_latency_ms": min(active_latencies),
            "max_latency_ms": max(active_latencies),
            "all_latencies_ms": active_latencies,
            "trace_file": str(output_dir / f"{name}.trace.json"),
            "operations": operation_times,
        }
        
        print(f"  mean_latency={mean_latency:.2f}ms")
        print(f"  Trace saved: {output_dir / name}.trace.json")
        
        # Print operation breakdown (GPU time matches Perfetto)
        print(f"  Operation breakdown (GPU time per inference):")
        for op_name in TRACKED_OPERATIONS:
            if op_name in operation_times:
                op = operation_times[op_name]
                gpu_ms = op['gpu_time_ms']
                cpu_ms = op['cpu_time_ms']
                print(f"    {op_name}: GPU={gpu_ms:.2f}ms, CPU={cpu_ms:.2f}ms ({op['count']} calls)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile ML interatomic potential models with Chrome Trace export for Perfetto visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Profile default molecules (H2O, CH4, C6H6, C60)
    python profile_with_trace.py --device cuda
    
    # Profile specific molecules
    python profile_with_trace.py --device cuda --molecules H2O NH3 CO2
    
    # Profile with a specific model
    python profile_with_trace.py --device cuda --model uma-s-1
    
    # Profile bulk Cu supercells
    python profile_with_trace.py --device cuda --bulk-element Cu --bulk-sizes 108 256 500
    
    # Profile random clusters
    python profile_with_trace.py --device cuda --cluster-element Al --cluster-sizes 100 500 1000
    
    # Load structures from XYZ/CIF files
    python profile_with_trace.py --device cuda --structure-files water_100.xyz cluster.cif
    
    # Apply batching to ANY test case type
    python profile_with_trace.py --device cuda --molecules H2O --batch-sizes 10 50 100
    python profile_with_trace.py --device cuda --cluster-element Al --cluster-sizes 100 --batch-sizes 10 50
    python profile_with_trace.py --device cuda --structure-files water.xyz --batch-sizes 10 50 100

Available models:
    {', '.join(AVAILABLE_MODELS)}

Batching:
    --batch-sizes applies to ALL test cases (molecules, bulk, clusters, structure files).
    Each test case is converted to batched versions with the specified batch sizes.
    
Schedule explanation:
    wait_steps:   Initial steps skipped (JIT compilation, unstable)
    warmup_steps: Profiler warmup (not recorded, GPU cache warming)
    active_steps: Actually recorded to trace file
    
    Example with wait=2, warmup=3, active=5:
    ├── step 0-1: ignored (wait)
    ├── step 2-4: warmup (not saved)
    └── step 5-9: ★ RECORDED ★
        """
    )
    
    # Model selection
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=AVAILABLE_MODELS,
                        help=f"Model to profile (default: {DEFAULT_MODEL})")
    
    # Device and output
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, 
                        default=Path("./profile_traces"),)
    
    # Profiler schedule
    parser.add_argument("--wait-steps", type=int, default=1,
                        help="Steps to skip before profiling (default: 1)")
    parser.add_argument("--warmup-steps", type=int, default=2,
                        help="Warmup steps, not recorded (default: 2)")
    parser.add_argument("--active-steps", type=int, default=3,
                        help="Steps to record (default: 3)")
    
    # ==========================================================================
    # Test case selection
    # ==========================================================================
    
    # Molecules from ASE database
    parser.add_argument("--molecules", type=str, nargs="+", default=None,
                        help="Molecule names from ASE database (e.g., H2O CH4 C6H6 NH3)")
    parser.add_argument("--include-molecules", action="store_true",
                        help="Include default molecules (H2O, CH4, C6H6, C60)")
    
    # Bulk supercells
    parser.add_argument("--bulk-element", type=str, default=None,
                        help="Element for bulk supercells (e.g., Cu, Al, Pt)")
    parser.add_argument("--bulk-sizes", type=int, nargs="+", default=None,
                        help="Atom counts for bulk supercells (e.g., 108 256 500)")
    
    # Random clusters
    parser.add_argument("--cluster-element", type=str, default=None,
                        help="Element for random clusters (e.g., Cu, Pt)")
    parser.add_argument("--cluster-sizes", type=int, nargs="+", default=None,
                        help="Atom counts for clusters (e.g., 100 200 500 1000)")
    
    # Structure files (XYZ, CIF, etc.)
    parser.add_argument("--structure-files", type=str, nargs="+", default=None,
                        help="Paths to structure files (xyz, cif, etc.)")
    
    # Universal batch sizes (applies to ALL test case types)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                        help="Batch sizes for batched inference (e.g., 10 50 100). "
                             "Applies to all test cases: molecules, bulk, clusters, structure files")
    
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    
    # =========================================================================
    # Build test cases (single structures first, then apply batching if needed)
    # =========================================================================
    test_cases = []
    
    # Default: if nothing specified, use default molecules
    nothing_specified = (
        args.molecules is None
        and not args.include_molecules 
        and args.bulk_element is None 
        and args.cluster_element is None
        and args.structure_files is None
    )
    
    # Molecules from ASE database
    if args.molecules:
        test_cases.extend(get_molecule_test_cases(molecules=args.molecules))
    
    if args.include_molecules or nothing_specified:
        test_cases.extend(get_molecule_test_cases())
    
    # Bulk supercells
    if args.bulk_element:
        test_cases.extend(get_bulk_test_cases(
            element=args.bulk_element,
            sizes=args.bulk_sizes,
        ))
    
    # Random clusters
    if args.cluster_element:
        test_cases.extend(get_cluster_test_cases(
            element=args.cluster_element,
            sizes=args.cluster_sizes,
        ))
    
    # Structure files (XYZ, CIF, etc.)
    if args.structure_files:
        test_cases.extend(get_structure_file_test_cases(
            file_paths=args.structure_files,
        ))
    
    if not test_cases:
        print("No test cases specified. Use --help for options.")
        return
    
    # =========================================================================
    # Apply batching if --batch-sizes is specified
    # =========================================================================
    if args.batch_sizes:
        test_cases = apply_batching(test_cases, args.batch_sizes)
    
    # =========================================================================
    # Sort and display
    # =========================================================================
    def get_total_atoms(item):
        name, atoms_or_list = item
        if isinstance(atoms_or_list, list):
            return sum(len(a) for a in atoms_or_list)
        return len(atoms_or_list)
    
    test_cases.sort(key=get_total_atoms)
    
    # Print test case summary
    print("Test cases to profile:")
    for name, atoms_or_list in test_cases:
        if isinstance(atoms_or_list, list):
            total = sum(len(a) for a in atoms_or_list)
            print(f"  - {name}: {total} atoms (batch of {len(atoms_or_list)})")
        else:
            print(f"  - {name}: {len(atoms_or_list)} atoms")
    print()
    
    # Run profiling
    results = run_profiling(
        device=args.device,
        output_dir=args.output_dir,
        test_cases=test_cases,
        model_name=args.model,
        wait_steps=args.wait_steps,
        warmup_steps=args.warmup_steps,
        active_steps=args.active_steps,
    )
    
    # Save summary JSON
    summary_path = args.output_dir / "summary.json"
    summary = {
        "model": args.model,
        "device": args.device,
        "profiler_schedule": {
            "wait_steps": args.wait_steps,
            "warmup_steps": args.warmup_steps,
            "active_steps": args.active_steps,
        },
        "results": results,
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save CSV table for easy analysis
    csv_path = args.output_dir / "timing_table.csv"
    save_timing_table_csv(results, csv_path, args.device)
    
    # Save markdown table
    md_path = args.output_dir / "timing_table.md"
    save_timing_table_markdown(results, md_path, args.device)
    
    # Print summary table
    print(f"\n{'=' * 80}")
    print("PROFILING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n{'Name':<30} {'Atoms':>8} {'Batch':>6} {'Edges':>10} {'Mean (ms)':>12}")
    print("-" * 80)
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        batch_size = data.get('batch_size', 1)
        num_edges = data.get('num_edges', 0)
        print(f"{name:<30} {data['natoms']:>8} {batch_size:>6} {num_edges:>10} {data['mean_latency_ms']:>12.2f}")
    
    print(f"\nTrace files saved to: {args.output_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Timing table CSV: {csv_path}")
    print(f"Timing table MD: {md_path}")
    print("\nNext steps:")
    print("  1. Open https://ui.perfetto.dev")
    print("  2. Upload a trace file (e.g., Cu_500atoms.trace.json)")
    print("  3. Analyze the timeline")
    print("  4. Open timing_table.csv in Excel or timing_table.md for quick view")


def save_timing_table_csv(results: dict, csv_path: Path, device: str):
    """
    Save timing results as CSV table.
    
    Columns: System, Atoms, Batch, Edges, Total_ms, then GPU and CPU times for each operation.
    GPU time is what Perfetto shows (actual kernel execution time).
    CPU time is kernel launch overhead.
    """
    # Build header
    header = ["System", "Atoms", "Batch", "Edges", "Total_ms"]
    for op in TRACKED_OPERATIONS:
        header.append(f"GPU_{op}")
        header.append(f"CPU_{op}")
    
    rows = []
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        row = [
            name,
            data["natoms"],
            data.get("batch_size", 1),
            data.get("num_edges", 0),
            f"{data['mean_latency_ms']:.2f}",
        ]
        
        operations = data.get("operations", {})
        for op_name in TRACKED_OPERATIONS:
            if op_name in operations:
                row.append(f"{operations[op_name]['gpu_time_ms']:.2f}")
                row.append(f"{operations[op_name]['cpu_time_ms']:.2f}")
            else:
                row.append("0.00")
                row.append("0.00")
        
        rows.append(row)
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"  CSV table saved: {csv_path}")


def save_timing_table_markdown(results: dict, md_path: Path, device: str):
    """
    Save timing results as Markdown table with GPU times (matches Perfetto).
    """
    # Shortened column names for readability
    short_names = {
        # mlip_unit.py
        "element_refs": "elem_ref",
        "forward": "forward",
        # escn_md.py
        "get_displacement_and_cell": "disp_cell",
        "generate_graph": "graph_gen",
        "charge spin dataset embeddings": "embed",
        "obtain wigner": "wigner",
        "obtain rotmat wigner original": "rotmat",
        "atom embedding": "atom_emb",
        "edge embedding": "edge_emb",
        "message passing 0": "msg_0",
        "message passing 1": "msg_1",
        "message passing 2": "msg_2",
        "message passing 3": "msg_3",
        # escn_md_block.py
        "SO2Conv": "SO2Conv",
        "edgewise": "edgewise",
        "atomwise": "atomwise",
    }
    
    lines = [
        f"# Profiling Results (GPU Time)",
        f"",
        f"Device: {device}",
        f"",
        f"**Note**: Times shown are GPU kernel execution times, matching Perfetto visualization.",
        f"",
        f"## Timing Table (ms per inference)",
        f"",
    ]
    
    # Header
    header_cols = ["System", "Atoms", "Batch", "Edges", "Total"] + [short_names.get(op, op) for op in TRACKED_OPERATIONS]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
    
    # Data rows
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        batch_size = data.get("batch_size", 1)
        num_edges = data.get("num_edges", 0)
        row = [
            name,
            str(data["natoms"]),
            str(batch_size),
            str(num_edges),
            f"{data['mean_latency_ms']:.1f}",
        ]
        
        operations = data.get("operations", {})
        for op_name in TRACKED_OPERATIONS:
            if op_name in operations:
                # Use GPU time (matches Perfetto)
                row.append(f"{operations[op_name]['gpu_time_ms']:.1f}")
            else:
                row.append("-")
        
        lines.append("| " + " | ".join(row) + " |")
    
    # Add legend
    lines.extend([
        f"",
        f"## Column Legend",
        f"",
    ])
    for op_name in TRACKED_OPERATIONS:
        short = short_names.get(op_name, op_name)
        lines.append(f"- **{short}**: {op_name}")
    
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"  Markdown table saved: {md_path}")


if __name__ == "__main__":
    main()
