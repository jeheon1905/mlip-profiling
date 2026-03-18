"""
Unified profiling script for MLIP models with Chrome Trace export for Perfetto visualization.

Supported models:
  - eSEN (fairchem): uma-s-1, esen-sm-conserving-all-omol, etc.
  - MACE: Any .pt model file

Usage:
    # eSEN model (by name)
    python profile_mlip.py --model-type esen --model-name esen-sm-conserving-all-omol \
        --structure-files structures/*.xyz

    # MACE model (by path)
    python profile_mlip.py --model-type mace --model-path model.pt \
        --structure-files structures/*.xyz

After running:
    1. Open https://ui.perfetto.dev
    2. Upload the generated .json trace file
    3. Analyze the timeline visualization

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from structure_builders import load_structures_from_files
from profile_utils import (
    synchronize,
    get_qps,
    extract_operation_times_from_trace,
    trace_handler,
)


# =============================================================================
# Model Adapter Interface
# =============================================================================

class ModelAdapter(ABC):
    """Abstract base class for MLIP model adapters."""
    
    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def run_inference(self, atoms: Atoms) -> Any:
        """Run inference on atoms. Includes graph generation."""
        pass
    
    @property
    @abstractmethod
    def tracked_operations(self) -> list[str]:
        """List of operation names to track in profiling."""
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> dict:
        """Model configuration info for logging."""
        pass
    
    def set_profiling_enabled(self, enabled: bool) -> None:
        """Enable/disable internal profiling (if supported)."""
        pass


# =============================================================================
# eSEN Adapter
# =============================================================================

class ESENAdapter(ModelAdapter):
    """Adapter for eSEN/UMA models from fairchem."""
    
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
    
    def __init__(self):
        self.predictor = None
        self.model_name = None
        self.device = None
        self.inference_settings = None
        self._cutoff = None
        self._max_neighbors = None
        self._external_graph_gen = None
    
    def load(
        self,
        model_name: str,
        device: str,
        inference_settings=None,
        **kwargs,
    ) -> None:
        from fairchem.core.calculate import pretrained_mlip
        from fairchem.core.units.mlip_unit.api.inference import inference_settings_default
        
        self.model_name = model_name
        self.device = device
        self.inference_settings = inference_settings or inference_settings_default()
        
        self.predictor = pretrained_mlip.get_predict_unit(
            model_name=model_name,
            inference_settings=self.inference_settings,
            device=device,
        )
        
        self._cutoff = self.predictor.model.module.backbone.cutoff
        self._max_neighbors = self.predictor.model.module.backbone.max_neighbors
        self._external_graph_gen = bool(self.inference_settings.external_graph_gen)
    
    def run_inference(self, atoms: Atoms) -> Any:
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
        
        # Create batch (graph generation included if external_graph_gen=True)
        data = AtomicData.from_ase(
            atoms,
            r_edges=self._external_graph_gen,
            radius=self._cutoff,
            max_neigh=self._max_neighbors if self._external_graph_gen else None,
            r_data_keys=["charge", "spin"],
            task_name="omol",
        )
        data.pos.requires_grad_(True)
        batch = atomicdata_list_to_batch([data]).to(self.device)
        
        # Inference (graph generated inside if external_graph_gen=False)
        return self.predictor.predict(batch)
    
    @property
    def tracked_operations(self) -> list[str]:
        return self.TRACKED_OPERATIONS
    
    @property
    def model_info(self) -> dict:
        return {
            "type": "esen",
            "name": self.model_name,
            "cutoff": self._cutoff,
            "max_neighbors": self._max_neighbors,
            "external_graph_gen": self._external_graph_gen,
            "tf32": self.inference_settings.tf32 if self.inference_settings else None,
            "compile": self.inference_settings.compile if self.inference_settings else None,
        }


# =============================================================================
# MACE Adapter
# =============================================================================

class MACEAdapter(ModelAdapter):
    """Adapter for MACE models."""
    
    TRACKED_OPERATIONS = [
        # Top-level profiling
        "forward",
        "generate_graph",
        # MACE.forward() operations (from models.py)
        "MACE::prepare_graph",
        "MACE::atomic_energies",
        "MACE::embeddings",
        "MACE::interaction_0",
        "MACE::interaction_1",
        "MACE::product_0",
        "MACE::product_1",
        "MACE::readouts",
        "MACE::get_outputs",
        # Block-level operations (from blocks.py)
        "MACE::ProductBasis",
        "MACE::SymmetricContraction",
        "MACE::Interaction::forward",
        "MACE::Interaction::skip_tp",
    ]
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.device = None
        self.z_table = None
        self.cutoff = None
        self.heads = None
    
    def load(
        self,
        model_path: str,
        device: str,
        **kwargs,
    ) -> None:
        from mace.tools import AtomicNumberTable
        
        self.model_path = model_path
        self.device = device
        
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        
        self.z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.cutoff = float(self.model.r_max)
        self.heads = getattr(self.model, 'heads', ['Default'])
    
    def run_inference(self, atoms: Atoms) -> Any:
        from mace.data import AtomicData
        from mace.data.utils import config_from_atoms
        from mace.tools.torch_geometric.dataloader import DataLoader
        
        # Graph generation
        with record_function("generate_graph"):
            config = config_from_atoms(atoms, head_name=self.heads[0])
            data = AtomicData.from_config(
                config,
                z_table=self.z_table,
                cutoff=self.cutoff,
                heads=self.heads,
            )
            loader = DataLoader([data], batch_size=1, shuffle=False, drop_last=False)
            batch = next(iter(loader)).to(self.device)
        
        # Model forward
        with record_function("forward"):
            out = self.model(batch.to_dict(), compute_force=True)
        
        return out
    
    @property
    def tracked_operations(self) -> list[str]:
        return self.TRACKED_OPERATIONS
    
    @property
    def model_info(self) -> dict:
        return {
            "type": "mace",
            "path": str(self.model_path),
            "cutoff": self.cutoff,
            "heads": self.heads,
            "num_elements": len(self.z_table) if self.z_table else None,
        }
    
    def set_profiling_enabled(self, enabled: bool) -> None:
        from mace.modules.profiling import set_profiling_enabled
        set_profiling_enabled(enabled)


# =============================================================================
# Adapter Factory
# =============================================================================

def get_adapter(model_type: str) -> ModelAdapter:
    """Get the appropriate adapter for the model type."""
    adapters = {
        "esen": ESENAdapter,
        "mace": MACEAdapter,
    }
    
    if model_type not in adapters:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(adapters.keys())}")
    
    return adapters[model_type]()


# =============================================================================
# Profiling
# =============================================================================

def run_profiling(
    adapter: ModelAdapter,
    device: str,
    output_dir: Path,
    test_cases: list[tuple[str, Atoms]],
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
    
    Graph generation is included in each inference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_info = adapter.model_info
    print(f"Model: {model_info}")
    print(f"Device: {device}")
    print(f"Schedule: wait={wait_steps}, warmup={warmup_steps}, active={active_steps}")
    print(f"Timing: timeit_number={timeit_number}, timeit_repeat={timeit_repeat}")
    print(f"Test cases: {len(test_cases)}")
    print("-" * 60)
    
    total_steps = wait_steps + warmup_steps + active_steps
    
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    results = {}
    
    for name, atoms in test_cases:
        atoms = atoms.copy()
        natoms = len(atoms)
        
        print(f"\n[{name}] Profiling...")
        print(f"  atoms={natoms}")
        
        prof_schedule = schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        )
        
        latencies_ms = []
        
        def run_inference():
            return adapter.run_inference(atoms)
        
        adapter.set_profiling_enabled(True)
        with profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=trace_handler(output_dir, name),
        ) as prof:
            synchronize(device)
            
            for step in range(total_steps):
                t0 = time.perf_counter()
                run_inference()
                with record_function(f"barrier_{step}"):
                    synchronize(device)
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                prof.step()
        adapter.set_profiling_enabled(False)
        
        trace_path = output_dir / f"{name}.trace.json"
        operation_times = extract_operation_times_from_trace(
            trace_path, active_steps, adapter.tracked_operations
        )
        
        active_latencies = latencies_ms[wait_steps + warmup_steps:]
        mean_latency = sum(active_latencies) / len(active_latencies)
        
        qps, ns_per_day, timeit_mean_ms, timeit_std_ms = get_qps(
            inference_fn=run_inference,
            device=device,
            warmups=10,
            timeiters=timeit_number,
            repeats=timeit_repeat,
        )
        
        results[name] = {
            "natoms": natoms,
            "mean_latency_ms": mean_latency,
            "min_latency_ms": min(active_latencies),
            "max_latency_ms": max(active_latencies),
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

def save_timing_table_csv(results: dict, csv_path: Path, tracked_ops: list[str]):
    """Save timing results as CSV table."""
    header = ["System", "Atoms", "Time_ms", "Std_ms", "QPS", "ns_per_day"]
    for op in tracked_ops:
        header.append(f"GPU_{op}")
        header.append(f"CPU_{op}")
    
    rows = []
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        row = [
            name,
            data["natoms"],
            f"{data['timeit_mean_ms']:.2f}",
            f"{data['timeit_std_ms']:.2f}",
            f"{data['qps']:.1f}",
            f"{data['ns_per_day']:.2f}",
        ]
        
        operations = data.get("operations", {})
        for op_name in tracked_ops:
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


def save_timing_table_markdown(results: dict, md_path: Path, model_info: dict):
    """Save timing results as Markdown table."""
    lines = [
        f"# MLIP Profiling Results",
        f"",
        f"Model: {model_info}",
        f"",
        f"## Performance Summary",
        f"",
        f"| System | Atoms | Time (ms) | QPS | ns/day |",
        f"|--------|-------|-----------|-----|--------|",
    ]
    
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        lines.append(
            f"| {name} | {data['natoms']} | "
            f"{data['timeit_mean_ms']:.2f} ± {data['timeit_std_ms']:.2f} | "
            f"{data['qps']:.1f} | {data['ns_per_day']:.2f} |"
        )
    
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified MLIP profiling with Chrome Trace export for Perfetto visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # eSEN model
    python profile_mlip.py --model-type esen --model-name esen-sm-conserving-all-omol \\
        --structure-files cu_fcc_*.xyz

    # MACE model
    python profile_mlip.py --model-type mace --model-path mace_model.pt \\
        --structure-files water_*.xyz

    # Custom timing settings
    python profile_mlip.py --model-type esen --model-name uma-s-1 \\
        --structure-files *.xyz --timeit-number 20 --timeit-repeat 5

To generate structure files, use structure_builders.py:
    python structure_builders.py --fcc Cu 32 108 500 --output-dir structures/
    python structure_builders.py --water 64 512 --output-dir structures/

After profiling:
    1. Open https://ui.perfetto.dev
    2. Upload a trace file from output_dir
    3. Analyze the timeline
        """
    )
    
    # Model selection
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["esen", "mace"],
                        help="Type of model to profile")
    parser.add_argument("--model-name", type=str,
                        help="Model name for eSEN (e.g., esen-sm-conserving-all-omol)")
    parser.add_argument("--model-path", type=str,
                        help="Path to model file for MACE")
    
    # Device and output
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("./profile_traces"))
    
    # Profiler schedule
    parser.add_argument("--wait-steps", type=int, default=5,
                        help="Steps to skip before profiling (default: 5)")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps, not recorded (default: 5)")
    parser.add_argument("--active-steps", type=int, default=5,
                        help="Steps to record (default: 5)")
    
    # Timing settings
    parser.add_argument("--timeit-number", type=int, default=10,
                        help="Number of iterations per timeit measurement (default: 10)")
    parser.add_argument("--timeit-repeat", type=int, default=5,
                        help="Number of timeit repetitions (default: 5)")
    
    # eSEN-specific options
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 for eSEN (default: False)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for eSEN (default: False)")
    parser.add_argument("--external-graph-gen", action="store_true",
                        help="Use external graph generation for eSEN (default: False)")
    
    # Structure files
    parser.add_argument("--structure-files", type=str, nargs="+", required=True,
                        help="Paths to structure files (xyz, cif, etc.)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type == "esen" and not args.model_name:
        parser.error("--model-name required for eSEN models")
    if args.model_type == "mace" and not args.model_path:
        parser.error("--model-path required for MACE models")
    
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    
    # Load structures
    test_cases = load_structures_from_files(args.structure_files)
    if not test_cases:
        print("No valid structures found. Check file paths.")
        return
    
    test_cases.sort(key=lambda x: len(x[1]))
    
    print("Test cases to profile:")
    for name, atoms in test_cases:
        print(f"  - {name}: {len(atoms)} atoms")
    print()
    
    # Create adapter and load model
    adapter = get_adapter(args.model_type)
    
    if args.model_type == "esen":
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        inference_settings = InferenceSettings(
            tf32=args.tf32,
            compile=args.compile,
            external_graph_gen=args.external_graph_gen,
        )
        adapter.load(
            model_name=args.model_name,
            device=args.device,
            inference_settings=inference_settings,
        )
    elif args.model_type == "mace":
        adapter.load(
            model_path=args.model_path,
            device=args.device,
        )
    
    # Run profiling
    results = run_profiling(
        adapter=adapter,
        device=args.device,
        output_dir=args.output_dir,
        test_cases=test_cases,
        wait_steps=args.wait_steps,
        warmup_steps=args.warmup_steps,
        active_steps=args.active_steps,
        timeit_number=args.timeit_number,
        timeit_repeat=args.timeit_repeat,
    )
    
    # Save outputs
    model_info = adapter.model_info
    
    summary_path = args.output_dir / "summary.json"
    summary = {
        "model": model_info,
        "device": args.device,
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
    save_timing_table_csv(results, csv_path, adapter.tracked_operations)
    
    md_path = args.output_dir / "timing_table.md"
    save_timing_table_markdown(results, md_path, model_info)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("PROFILING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n{'Name':<20} {'Atoms':>8} {'Time (ms)':>12} {'QPS':>10} {'ns/day':>10}")
    print("-" * 64)
    for name, data in sorted(results.items(), key=lambda x: x[1]["natoms"]):
        print(f"{name:<20} {data['natoms']:>8} {data['timeit_mean_ms']:>9.2f} ± {data['timeit_std_ms']:<4.2f} {data['qps']:>8.1f} {data['ns_per_day']:>10.2f}")
    
    print(f"\nTrace files saved to: {args.output_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Timing table: {csv_path}")
    print("\nNext steps:")
    print("  1. Open https://ui.perfetto.dev")
    print("  2. Upload a trace file")
    print("  3. Analyze the timeline")


if __name__ == "__main__":
    main()
