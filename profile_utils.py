"""
Common profiling utilities for MLIP Profiling.

Shared utilities used by profile_mlip.py.
Follows fairchem's uma_speed_benchmark.py methodology.
"""

from __future__ import annotations

import json
import logging
import timeit
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================================================
# Distributed / Multi-GPU Helpers
# =============================================================================

def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return torch.distributed.is_initialized()


def synchronize(device: str = "cuda") -> None:
    """Synchronize across devices.
    
    Uses torch.distributed.barrier() if distributed mode is active,
    otherwise falls back to torch.cuda.synchronize() for single-GPU.
    """
    if is_distributed():
        torch.distributed.barrier()
    elif device == "cuda":
        torch.cuda.synchronize()


def get_rank() -> int:
    """Get the current rank (0 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Get the world size (1 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def log_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logging.info(f"{prefix}Memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")


# =============================================================================
# QPS Measurement (fairchem-style)
# =============================================================================

def get_qps(
    inference_fn,
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
        inference_fn: Callable that performs one inference step
        device: Device type ("cuda" or "cpu")
        warmups: Number of warmup iterations
        timeiters: Number of iterations per timing repeat
        repeats: Number of timing repeats
        
    Returns:
        Tuple of (qps, ns_per_day, mean_ms, std_ms)
    """
    def timefunc():
        inference_fn()
        synchronize(device)
    
    # Warmup
    for i in range(warmups):
        timefunc()
        if i == 0:  # Log memory after first warmup
            log_memory("After warmup: ")
    
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


# =============================================================================
# Trace Analysis
# =============================================================================

def extract_operation_times_from_trace(
    trace_path: Path,
    active_steps: int,
    tracked_operations: list[str],
) -> dict:
    """
    Extract timing directly from Chrome Trace file (same as Perfetto).
    
    Extracts both CPU and GPU times:
    - 'user_annotation': CPU time (kernel launch overhead)
    - 'gpu_user_annotation': GPU time (actual kernel execution)
    
    Args:
        trace_path: Path to Chrome trace JSON file
        active_steps: Number of active profiling steps
        tracked_operations: List of operation names to track
        
    Returns:
        Dictionary of operation timing data
        
    Raises:
        FileNotFoundError: If trace file does not exist
        ValueError: If trace file is not valid JSON
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    
    try:
        with open(trace_path) as f:
            trace_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in trace file {trace_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read trace file {trace_path}: {e}")
    
    cpu_durations = {}
    gpu_durations = {}
    
    for event in trace_data.get("traceEvents", []):
        name = event.get("name", "")
        phase = event.get("ph", "")
        category = event.get("cat", "")
        
        if phase == "X" and name in tracked_operations:
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
