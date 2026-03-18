"""
Profiling script for eSEN model with Chrome Trace export for Perfetto visualization.

Usage:
    # Profile with structure files
    python profile_esen.py --device cuda --structure-files structures/*.xyz
    
    # With batching
    python profile_esen.py --device cuda --structure-files water.xyz --batch-sizes 10 50 100

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
import timeit
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, inference_settings_default

from structure_builders import load_structures_from_files, apply_batching

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================================================
# Distributed / Multi-GPU Helpers
# =============================================================================

def _is_distributed() -> bool:
    """Check if running in distributed mode."""
    return torch.distributed.is_initialized()


def _synchronize(device: str = "cuda") -> None:
    """Synchronize across devices.
    
    Uses torch.distributed.barrier() if distributed mode is active,
    otherwise falls back to torch.cuda.synchronize() for single-GPU.
    """
    if _is_distributed():
        torch.distributed.barrier()
    elif device == "cuda":
        torch.cuda.synchronize()


def _get_rank() -> int:
    """Get the current rank (0 if not distributed)."""
    if _is_distributed():
        return torch.distributed.get_rank()
    return 0


def _get_world_size() -> int:
    """Get the world size (1 if not distributed)."""
    if _is_distributed():
        return torch.distributed.get_world_size()
    return 1


def _log_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logging.info(f"{prefix}Memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")


# =============================================================================
# QPS Measurement (fairchem-style)
# =============================================================================

def get_qps(
    data,
    predictor,
    device: str = "cuda",
    warmups: int = 10,
    timeiters: int = 10,
    repeats: int = 5,
) -> tuple[float, float, float, float]:
    """Measure QPS (Queries Per Second) and ns/day.
    
    This follows fairchem's uma_speed_benchmark.py methodology:
    - Use barrier for multi-GPU synchronization
    - Warmup iterations before timing
    - Use timeit.repeat for accurate measurements
    
    Args:
        data: Input data for prediction
        predictor: The predictor model
        device: Device type ("cuda" or "cpu")
        warmups: Number of warmup iterations
        timeiters: Number of iterations per timing repeat
        repeats: Number of timing repeats
        
    Returns:
        Tuple of (qps, ns_per_day, mean_ms, std_ms)
    """
    def timefunc():
        predictor.predict(data)
        _synchronize(device)
    
    # Warmup
    for i in range(warmups):
        timefunc()
        if i == 0:  # Log memory after first warmup
            _log_memory("After warmup: ")
    
    # Timing
    result = timeit.repeat(timefunc, number=timeiters, repeat=repeats)
    
    mean_time = np.mean(result)
    std_time = np.std(result)
    
    logging.info(
        f"Timing results over {repeats} repeats: {[f'{r:.4f}' for r in result]}, "
        f"mean: {mean_time:.4f}, std: {std_time:.4f}"
    )
    
    qps = timeiters / mean_time
    ns_per_day = qps * 24 * 3600 / 1e6
    mean_ms = (mean_time / timeiters) * 1000
    std_ms = (std_time / timeiters) * 1000
    
    return qps, ns_per_day, mean_ms, std_ms


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


# Key operations to track (record_function tags from eSEN model)
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


# =============================================================================
# Data Conversion
# =============================================================================

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
# Trace Analysis
# =============================================================================

def extract_operation_times_from_trace(trace_path: Path, active_steps: int) -> dict:
    """
    Extract timing directly from Chrome Trace file (same as Perfetto).
    
    Extracts both CPU and GPU times:
    - 'user_annotation': CPU time (kernel launch overhead)
    - 'gpu_user_annotation': GPU time (actual kernel execution)
    """
    with open(trace_path) as f:
        trace_data = json.load(f)
    
    cpu_durations = {}
    gpu_durations = {}
    
    for event in trace_data.get("traceEvents", []):
        name = event.get("name", "")
        phase = event.get("ph", "")
        category = event.get("cat", "")
        
        if phase == "X" and name in TRACKED_OPERATIONS:
            dur_ms = event.get("dur", 0) / 1000.0  # μs to ms
            
            if category == "user_annotation":
                if name not in cpu_durations:
                    cpu_durations[name] = []
                cpu_durations[name].append(dur_ms)
            elif category == "gpu_user_annotation":
                if name not in gpu_durations:
                    gpu_durations[name] = []
                gpu_durations[name].append(dur_ms)
    
    all_names = set(cpu_durations.keys()) | set(gpu_durations.keys())
    operation_times = {}
    
    for name in all_names:
        cpu_durs = cpu_durations.get(name, [])
        cpu_total_ms = sum(cpu_durs)
        cpu_count = len(cpu_durs)
        
        gpu_durs = gpu_durations.get(name, [])
        gpu_total_ms = sum(gpu_durs)
        gpu_count = len(gpu_durs)
        
        count = gpu_count if gpu_count > 0 else cpu_count
        calls_per_step = count // active_steps if active_steps > 0 else count
        
        operation_times[name] = {
            "gpu_time_ms": gpu_total_ms / active_steps if active_steps > 0 else gpu_total_ms,
            "cpu_time_ms": cpu_total_ms / active_steps if active_steps > 0 else cpu_total_ms,
            "count": calls_per_step,
        }
    
    return operation_times


def trace_handler(output_dir: Path, name: str):
    """Create a trace handler that saves Chrome trace files."""
    def handler(prof):
        trace_path = output_dir / f"{name}.trace.json"
        print(f"  Saving trace: {trace_path}")
        prof.export_chrome_trace(str(trace_path))
    return handler


# =============================================================================
# Profiling
# =============================================================================

def run_profiling(
    device: str,
    output_dir: Path,
    test_cases: list[tuple[str, Atoms | list[Atoms]]],
    model_name: str = DEFAULT_MODEL,
    inference_settings: InferenceSettings | None = None,
    wait_steps: int = 5,
    warmup_steps: int = 5,
    active_steps: int = 5,
    timeit_number: int = 10,
    timeit_repeat: int = 5,
) -> dict:
    """
    Run profiling with Chrome trace export.
    
    Schedule:
    ├── wait_steps:   Skipped (JIT compilation, unstable)
    ├── warmup_steps: Warmup (not recorded)
    └── active_steps: ★ RECORDED ★
    
    Timing uses timeit.repeat() for accurate measurements.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if inference_settings is None:
        inference_settings = inference_settings_default()
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"InferenceSettings:")
    print(f"  tf32={inference_settings.tf32}")
    print(f"  compile={inference_settings.compile}")
    print(f"  activation_checkpointing={inference_settings.activation_checkpointing}")
    print(f"  external_graph_gen={inference_settings.external_graph_gen}")
    print(f"Schedule: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}")
    print(f"Timing: timeit_number={timeit_number}, timeit_repeat={timeit_repeat}")
    print(f"Test cases: {len(test_cases)}")
    print("-" * 60)
    
    predictor = pretrained_mlip.get_predict_unit(
        model_name=model_name,
        inference_settings=inference_settings,
        device=device,
    )
    
    cutoff = predictor.model.module.backbone.cutoff
    max_neighbors = predictor.model.module.backbone.max_neighbors
    external_graph_gen = bool(predictor.inference_settings.external_graph_gen)
    
    total_steps = wait_steps + warmup_steps + active_steps
    
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    results = {}
    
    for name, atoms_or_list in test_cases:
        if isinstance(atoms_or_list, list):
            atoms_list = atoms_or_list
            batch_size = len(atoms_list)
            natoms = sum(len(a) for a in atoms_list)
            data = atoms_list_to_batch(atoms_list, cutoff, max_neighbors, external_graph_gen)
        else:
            atoms = atoms_or_list
            batch_size = 1
            natoms = len(atoms)
            data = atoms_to_batch(atoms, cutoff, max_neighbors, external_graph_gen)
        
        data = data.to(device)
        
        with torch.no_grad():
            predictor.predict(data)
            num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
        
        print(f"\n[{name}] Profiling...")
        print(f"  atoms={natoms}, batch_size={batch_size}, edges={num_edges}")
        
        prof_schedule = schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        )
        
        latencies_ms = []
        
        with profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=trace_handler(output_dir, name),
        ) as prof:
            # Initial sync before profiling loop (fairchem-style)
            _synchronize(device)
            
            for step in range(total_steps):
                t0 = time.perf_counter()
                predictor.predict(data)
                with record_function(f"barrier_{step}"):
                    _synchronize(device)  # barrier for multi-GPU or cuda.synchronize for single-GPU
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                prof.step()
        
        trace_path = output_dir / f"{name}.trace.json"
        operation_times = extract_operation_times_from_trace(trace_path, active_steps)
        
        active_latencies = latencies_ms[wait_steps + warmup_steps:]
        mean_latency = sum(active_latencies) / len(active_latencies)
        
        # QPS measurement using fairchem-style get_qps function
        qps, ns_per_day, timeit_mean_ms, timeit_std_ms = get_qps(
            data=data,
            predictor=predictor,
            device=device,
            warmups=10,
            timeiters=timeit_number,
            repeats=timeit_repeat,
        )
        
        results[name] = {
            "natoms": natoms,
            "batch_size": batch_size,
            "num_edges": num_edges,
            "mean_latency_ms": mean_latency,
            "min_latency_ms": min(active_latencies),
            "max_latency_ms": max(active_latencies),
            # timeit results
            "timeit_mean_ms": timeit_mean_ms,
            "timeit_std_ms": timeit_std_ms,
            "qps": qps,
            "ns_per_day": ns_per_day,
            "trace_file": str(trace_path),
            "operations": operation_times,
        }
        
        print(f"  timeit: {timeit_mean_ms:.2f} ± {timeit_std_ms:.2f} ms | QPS: {qps:.1f} | ns/day: {ns_per_day:.2f}")
    
    return results


# =============================================================================
# Output Formatters
# =============================================================================

def save_timing_table_csv(results: dict, csv_path: Path):
    """Save timing results as CSV table."""
    header = ["System", "Atoms", "Batch", "Edges", "Total_ms", "Timeit_ms", "Std_ms", "QPS", "ns_per_day"]
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
            f"{data.get('timeit_mean_ms', 0):.2f}",
            f"{data.get('timeit_std_ms', 0):.2f}",
            f"{data.get('qps', 0):.1f}",
            f"{data.get('ns_per_day', 0):.2f}",
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


def save_timing_table_markdown(results: dict, md_path: Path, device: str):
    """Save timing results as Markdown table."""
    short_names = {
        "element_refs": "elem_ref",
        "forward": "forward",
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
        "SO2Conv": "SO2Conv",
        "edgewise": "edgewise",
        "atomwise": "atomwise",
    }
    
    lines = [
        f"# Profiling Results (GPU Time)",
        f"",
        f"Device: {device}",
        f"",
        f"## Summary Table",
        f"",
    ]
    
    # Summary table with QPS and ns/day
    summary_header = ["System", "Atoms", "Batch", "Latency (ms)", "QPS", "ns/day"]
    lines.append("| " + " | ".join(summary_header) + " |")
    lines.append("|" + "|".join(["---"] * len(summary_header)) + "|")
    
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        row = [
            name,
            str(data["natoms"]),
            str(data.get("batch_size", 1)),
            f"{data.get('timeit_mean_ms', data['mean_latency_ms']):.2f}",
            f"{data.get('qps', 0):.1f}",
            f"{data.get('ns_per_day', 0):.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    lines.append("## Operation Timing Table (ms per inference)")
    lines.append("")
    
    header_cols = ["System", "Atoms", "Batch", "Edges", "Total"] + [short_names.get(op, op) for op in TRACKED_OPERATIONS]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
    
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        row = [
            name,
            str(data["natoms"]),
            str(data.get("batch_size", 1)),
            str(data.get("num_edges", 0)),
            f"{data['mean_latency_ms']:.1f}",
        ]
        
        operations = data.get("operations", {})
        for op_name in TRACKED_OPERATIONS:
            if op_name in operations:
                row.append(f"{operations[op_name]['gpu_time_ms']:.1f}")
            else:
                row.append("-")
        
        lines.append("| " + " | ".join(row) + " |")
    
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Profile eSEN model with Chrome Trace export for Perfetto visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Profile structures from files
    python profile_esen.py --device cuda --structure-files water.xyz cluster.cif
    
    # Profile with batching
    python profile_esen.py --device cuda --structure-files water.xyz --batch-sizes 10 50 100
    
    # Profile with a specific model
    python profile_esen.py --device cuda --model uma-s-1 --structure-files *.xyz

Available models:
    {', '.join(AVAILABLE_MODELS)}

To generate structure files, use structure_builders.py:
    python structure_builders.py --bulk-element Cu --bulk-sizes 108 256 500 --output-dir structures/
    python structure_builders.py --molecules H2O CH4 C6H6 --output-dir structures/
        """
    )
    
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=AVAILABLE_MODELS,
                        help=f"Model to profile (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("./profile_traces"))
    
    parser.add_argument("--wait-steps", type=int, default=5,
                        help="Steps to skip before profiling (default: 5)")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps, not recorded (default: 5)")
    parser.add_argument("--active-steps", type=int, default=5,
                        help="Steps to record (default: 5)")
    
    parser.add_argument("--structure-files", type=str, nargs="+", required=True,
                        help="Paths to structure files (xyz, cif, etc.)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                        help="Batch sizes for batched inference (e.g., 10 50 100)")
    
    # InferenceSettings arguments
    parser.add_argument("--tf32", action="store_true", default=True,
                        help="Enable TF32 mode (default: True)")
    parser.add_argument("--no-tf32", action="store_true",
                        help="Disable TF32 mode")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (default: False)")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Enable activation checkpointing (default: False)")
    parser.add_argument("--external-graph-gen", action="store_true",
                        help="Enable external graph generation (default: False)")
    
    # timeit arguments
    parser.add_argument("--timeit-number", type=int, default=10,
                        help="Number of iterations per timeit repeat (default: 10)")
    parser.add_argument("--timeit-repeat", type=int, default=5,
                        help="Number of timeit repeats (default: 5)")
    
    args = parser.parse_args()
    
    # Handle tf32 flag
    use_tf32 = args.tf32 and not args.no_tf32
    
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    
    # Load structures
    test_cases = load_structures_from_files(args.structure_files)
    
    if not test_cases:
        print("No valid structures found. Check file paths.")
        return
    
    # Apply batching if specified
    if args.batch_sizes:
        test_cases = apply_batching(test_cases, args.batch_sizes)
    
    # Sort by atom count
    def get_total_atoms(item):
        name, atoms_or_list = item
        if isinstance(atoms_or_list, list):
            return sum(len(a) for a in atoms_or_list)
        return len(atoms_or_list)
    
    test_cases.sort(key=get_total_atoms)
    
    print("Test cases to profile:")
    for name, atoms_or_list in test_cases:
        if isinstance(atoms_or_list, list):
            total = sum(len(a) for a in atoms_or_list)
            print(f"  - {name}: {total} atoms (batch of {len(atoms_or_list)})")
        else:
            print(f"  - {name}: {len(atoms_or_list)} atoms")
    print()
    
    # Create InferenceSettings
    inference_settings = InferenceSettings(
        tf32=use_tf32,
        compile=args.compile,
        activation_checkpointing=args.activation_checkpointing,
        external_graph_gen=args.external_graph_gen,
    )
    
    results = run_profiling(
        device=args.device,
        output_dir=args.output_dir,
        test_cases=test_cases,
        model_name=args.model,
        wait_steps=args.wait_steps,
        warmup_steps=args.warmup_steps,
        active_steps=args.active_steps,
        inference_settings=inference_settings,
        timeit_number=args.timeit_number,
        timeit_repeat=args.timeit_repeat,
    )
    
    # Save outputs
    summary_path = args.output_dir / "summary.json"
    summary = {
        "model": args.model,
        "device": args.device,
        "inference_settings": {
            "tf32": use_tf32,
            "compile": args.compile,
            "activation_checkpointing": args.activation_checkpointing,
            "external_graph_gen": args.external_graph_gen,
        },
        "profiler_schedule": {
            "wait_steps": args.wait_steps,
            "warmup_steps": args.warmup_steps,
            "active_steps": args.active_steps,
        },
        "timeit_settings": {
            "number": args.timeit_number,
            "repeat": args.timeit_repeat,
        },
        "results": results,
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    csv_path = args.output_dir / "timing_table.csv"
    save_timing_table_csv(results, csv_path)
    
    md_path = args.output_dir / "timing_table.md"
    save_timing_table_markdown(results, md_path, args.device)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("PROFILING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n{'Name':<25} {'Atoms':>7} {'Batch':>5} {'Latency (ms)':>12} {'QPS':>10} {'ns/day':>10}")
    print("-" * 80)
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        print(f"{name:<25} {data['natoms']:>7} {data.get('batch_size', 1):>5} {data['timeit_mean_ms']:>12.2f} {data.get('qps', 0):>10.1f} {data.get('ns_per_day', 0):>10.2f}")
    
    print(f"\nTrace files saved to: {args.output_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Timing table: {csv_path}")
    print("\nNext steps:")
    print("  1. Open https://ui.perfetto.dev")
    print("  2. Upload a trace file")
    print("  3. Analyze the timeline")


if __name__ == "__main__":
    main()
