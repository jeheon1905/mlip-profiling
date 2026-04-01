#!/usr/bin/env python3
"""
Generate operation breakdown and kernel analysis plots from profiling results.

Usage:
    python scripts/generate_plots.py results/YYYY-MM-DD_HHMMSS_<GPU_TYPE>

This generates 4 plots per model configuration:
    1. {model}_breakdown.png      - Operation breakdown (all operations)
    2. {model}_breakdown_leaf.png - Operation breakdown (leaf only, no wrappers)
    3. {model}_pie.png            - Pie chart (leaf only)
    4. {model}_kernels.png        - Kernel categories breakdown
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# Global font size settings
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})

# Breakdown plots use larger fonts
BREAKDOWN_FONTSIZES = {
    'ytick': 16,
    'xlabel': 18,
    'title': 19,
    'legend': 16,
    'bar_label': 14,
}


# =============================================================================
# Configuration
# =============================================================================

# Operations that wrap other operations (not leaf nodes)
# Note: eSEN "message passing N" entries are matched dynamically via regex below
WRAPPER_OPERATIONS = {
    "eSEN::model_forward",
    "forward",
    # MACE intermediate wrappers (GPU time attributed to children)
    "MACE::Interaction::forward",
    "MACE::interaction_0",
    "MACE::interaction_1",
    "MACE::product_0",
    "MACE::product_1",
    "MACE::get_outputs",
    "MACE::ProductBasis",
}

# Regex pattern for dynamically matching wrapper operations
WRAPPER_PATTERNS = [
    r"^message passing \d+$",  # eSEN message passing layers (variable count)
]

# Patterns for grouping similar operations
GROUPING_PATTERNS = [
    # SevenNet patterns
    (r"SevenNet::(\d+)_convolution", "SevenNet::convolution"),
    (r"SevenNet::(\d+)_self_interaction_1", "SevenNet::self_interaction_1"),
    (r"SevenNet::(\d+)_self_interaction_2", "SevenNet::self_interaction_2"),
    (r"SevenNet::(\d+)_equivariant_gate", "SevenNet::equivariant_gate"),
    (r"SevenNet::(\d+)_self_connection_intro", "SevenNet::self_connection_intro"),
    (r"SevenNet::(\d+)_self_connection_outro", "SevenNet::self_connection_outro"),
    # MACE patterns
    (r"message passing (\d+)", "eSEN::message_passing"),
    (r"MACE::interaction_(\d+)", "MACE::interaction"),
    (r"MACE::product_(\d+)", "MACE::product"),
]

# Kernel category patterns
KERNEL_CATEGORIES = {
    r"sgemm|gemm|gemv|cublas|cutlass.*Kernel": "GEMM",
    r"segmented_polynomial": "cuEq TP",
    r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise": "Elementwise",
    r"reduce_kernel|Reduce": "Reduction",
    r"scatter.*gather|index.*kernel|gather_kernel": "Scatter/Gather",
    r"CatArray|concat": "Concat",
}


# =============================================================================
# Helper Functions
# =============================================================================


def is_leaf_operation(op_name: str) -> bool:
    """Check if operation is a leaf (not a wrapper)."""
    if op_name in WRAPPER_OPERATIONS:
        return False
    for pattern in WRAPPER_PATTERNS:
        if re.match(pattern, op_name):
            return False
    return True


def group_operation_name(name: str) -> str:
    """Group similar operations into one category."""
    for pattern, replacement in GROUPING_PATTERNS:
        if re.match(pattern, name):
            return replacement
    return name


def aggregate_operations(operations: dict, model_type: str, leaf_only: bool = True) -> dict:
    """Aggregate operations by grouped name, computing effective time."""
    aggregated = defaultdict(
        lambda: {"gpu_time_ms": 0, "cpu_time_ms": 0, "effective_time_ms": 0, "is_cpu_bound": False, "count": 0}
    )

    for name, stats in operations.items():
        if leaf_only and not is_leaf_operation(name):
            continue
        grouped_name = group_operation_name(name)
        aggregated[grouped_name]["gpu_time_ms"] += stats.get("gpu_time_ms", 0)
        aggregated[grouped_name]["cpu_time_ms"] += stats.get("cpu_time_ms", 0)
        aggregated[grouped_name]["count"] += 1

    # Compute effective time using explicit CPU operation list
    for name, stats in aggregated.items():
        if is_cpu_operation(name, model_type):
            stats["effective_time_ms"] = stats["cpu_time_ms"]
            stats["is_cpu_bound"] = True
        else:
            stats["effective_time_ms"] = stats["gpu_time_ms"]
            stats["is_cpu_bound"] = False

    return dict(aggregated)


def simplify_kernel_name(name: str, max_len: int = 40) -> str:
    """Simplify CUDA kernel name for display."""
    if "<" in name:
        name = name.split("<")[0]
    name = re.sub(r"^void\s+", "", name)
    name = re.sub(r"^at::native::", "", name)
    name = re.sub(r"^\(anonymous namespace\)::", "", name)
    return name[:max_len]


def categorize_kernel(kernel_name: str) -> str:
    """Categorize kernel by name pattern."""
    # Handle memcpy/memset pseudo-categories from GPU memory operations
    if kernel_name in ("Memcpy", "Memset"):
        return "Memcpy/Memset"
    for pattern, category in KERNEL_CATEGORIES.items():
        if re.search(pattern, kernel_name, re.IGNORECASE):
            return category
    return "Other"


# Operations that are genuinely CPU-bound, keyed by model_type.
# These use cpu_time as effective_time; everything else uses gpu_time.
#
# - MACE/SevenNet generate_graph: CPU neighbor list (matscipy / ase)
#   (eSEN generate_graph is GPU via nvalchemiops, so NOT listed here)
# - eSEN::data_preparation: CPU tensor creation (0 GPU kernels)
# - eSEN::data_to_device: CPU-to-GPU memcpy
CPU_OPERATIONS = {
    "mace": {"generate_graph"},
    "sevenn": {"generate_graph"},
    "esen": {"eSEN::data_preparation", "eSEN::data_to_device"},
}


def is_cpu_operation(op_name: str, model_type: str) -> bool:
    """Check if an operation should use CPU time as its effective time."""
    cpu_ops = CPU_OPERATIONS.get(model_type, set())
    return op_name in cpu_ops


# =============================================================================
# Plot Functions
# =============================================================================


def plot_operation_breakdown(
    operations: dict, title: str, output_path: Path, model_type: str, leaf_only: bool = False
):
    """Plot operation-level time breakdown using effective time.

    Each operation gets a single bar showing the meaningful time metric:
    - GPU time for GPU-bound operations (green)
    - CPU time for CPU-bound operations (orange)
    """
    title_suffix = " (Leaf Only)" if leaf_only else ""

    aggregated = aggregate_operations(operations, model_type, leaf_only=leaf_only)
    sorted_ops = sorted(
        aggregated.items(),
        key=lambda x: x[1]["effective_time_ms"],
        reverse=True,
    )[:15]

    if not sorted_ops:
        return

    names = []
    for op in sorted_ops:
        name = op[0]
        if op[1]["count"] > 1:
            name = f"{name} (x{op[1]['count']})"
        names.append(name)
    eff_times = [op[1]["effective_time_ms"] for op in sorted_ops]
    is_cpu = [op[1]["is_cpu_bound"] for op in sorted_ops]
    colors = ["#e67e22" if c else "#2ecc71" for c in is_cpu]

    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, eff_times, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=BREAKDOWN_FONTSIZES['ytick'])
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)", fontsize=BREAKDOWN_FONTSIZES['xlabel'])
    ax.set_title(f"{title}\nOperation Breakdown{title_suffix}", fontsize=BREAKDOWN_FONTSIZES['title'])

    # Legend
    legend_elements = [
        Patch(facecolor="#2ecc71", label="GPU time"),
        Patch(facecolor="#e67e22", label="CPU time"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=BREAKDOWN_FONTSIZES['legend'])

    for bar, t in zip(bars, eff_times):
        if t > 0.1:
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}",
                va="center",
                fontsize=BREAKDOWN_FONTSIZES['bar_label'],
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_pie_chart(operations: dict, title: str, output_path: Path, model_type: str):
    """Plot pie chart of effective time distribution (leaf only).

    Uses effective_time (GPU time for GPU ops, CPU time for CPU ops).
    CPU-bound operations are highlighted with [CPU] label.
    """
    aggregated = aggregate_operations(operations, model_type, leaf_only=True)
    sorted_ops = sorted(
        aggregated.items(),
        key=lambda x: x[1]["effective_time_ms"],
        reverse=True,
    )

    top_ops = sorted_ops[:8]
    other_eff = sum(op[1]["effective_time_ms"] for op in sorted_ops[8:])

    labels = []
    sizes = []
    is_cpu_flags = []
    for op_name, stats in top_ops:
        name = op_name
        if stats["count"] > 1:
            name = f"{name} (x{stats['count']})"
        if stats["is_cpu_bound"]:
            name = f"{name} [CPU]"
        labels.append(name)
        sizes.append(stats["effective_time_ms"])
        is_cpu_flags.append(stats["is_cpu_bound"])

    if other_eff > 0:
        labels.append("Other")
        sizes.append(other_eff)
        is_cpu_flags.append(False)

    # Filter out zeros
    non_zero_idx = [i for i in range(len(labels)) if sizes[i] > 0]
    if not non_zero_idx:
        return
    labels = [labels[i] for i in non_zero_idx]
    sizes = [sizes[i] for i in non_zero_idx]
    is_cpu_flags = [is_cpu_flags[i] for i in non_zero_idx]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    total_ms = sum(sizes)

    # Merge small slices (<3%) into "Other" to avoid label overlap
    merged_labels = []
    merged_sizes = []
    merged_colors = []
    small_total = 0
    for i, (label, size) in enumerate(zip(labels, sizes)):
        pct = size / total_ms * 100 if total_ms > 0 else 0
        if pct < 3 and label != "Other":
            small_total += size
        else:
            merged_labels.append(label)
            merged_sizes.append(size)
            merged_colors.append(colors[i])
    # Combine small slices with existing "Other" or create new
    if small_total > 0:
        if merged_labels and merged_labels[-1] == "Other":
            merged_sizes[-1] += small_total
        else:
            merged_labels.append("Other")
            merged_sizes.append(small_total)
            merged_colors.append((0.85, 0.85, 0.85, 1.0))

    # Format: percentage inside pie, labels in legend
    wedges, _, autotexts = ax.pie(
        merged_sizes,
        autopct="%1.1f%%",
        pctdistance=0.75,
        colors=merged_colors,
        startangle=90,
        textprops={"fontsize": 13},
    )
    ax.legend(
        wedges,
        merged_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=12,
    )
    ax.set_title(
        f"{title}\nEffective Time Distribution ({total_ms:.1f} ms total, Leaf Only)"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_kernel_breakdown(trace_path: Path, title: str, output_path: Path):
    """Plot kernel-level breakdown from trace file using gpu_user_annotation for accurate mapping."""
    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    # Use gpu_user_annotation (GPU-projected annotations) for accurate kernel mapping.
    # Unlike CPU-side user_annotation, gpu_user_annotation correctly attributes
    # async GPU kernels (e.g., backward pass kernels from autograd.grad) to the
    # record_function annotation that launched them, via correlation IDs.
    gpu_annotations = []
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X":
            gpu_annotations.append(
                {
                    "name": e.get("name", ""),
                    "ts": e.get("ts", 0),
                    "dur": e.get("dur", 0),
                    "end": e.get("ts", 0) + e.get("dur", 0),
                }
            )

    # Count profiler steps from CPU-side annotations (gpu_user_annotation doesn't have ProfilerStep)
    cpu_annotations = [
        e for e in events
        if e.get("cat") == "user_annotation" and e.get("ph") == "X"
    ]
    profiler_steps = sum(1 for a in cpu_annotations if "ProfilerStep" in a.get("name", ""))

    # Map kernels to operations using gpu_user_annotation containment
    ops_to_kernels = defaultdict(lambda: {"total_us": 0, "kernels": defaultdict(float), "count": 0})

    for e in events:
        if e.get("cat") == "kernel":
            k_start = e["ts"]
            k_end = k_start + e.get("dur", 0)
            k_name = simplify_kernel_name(e.get("name", ""))
            k_dur = e.get("dur", 0)

            containing = [
                a for a in gpu_annotations if a["ts"] <= k_start and a["end"] >= k_end
            ]
            if containing:
                smallest = min(containing, key=lambda x: x["dur"])
                # Apply grouping to operation name
                grouped_op = group_operation_name(smallest["name"])
                ops_to_kernels[grouped_op]["total_us"] += k_dur
                ops_to_kernels[grouped_op]["kernels"][k_name] += k_dur

    # Also map memcpy/memset operations
    for e in events:
        if e.get("cat") in ("gpu_memcpy", "gpu_memset"):
            m_start = e["ts"]
            m_end = m_start + e.get("dur", 0)
            m_dur = e.get("dur", 0)
            m_cat_name = "Memcpy" if "memcpy" in e.get("cat", "") else "Memset"

            containing = [
                a for a in gpu_annotations if a["ts"] <= m_start and a["end"] >= m_end
            ]
            if containing:
                smallest = min(containing, key=lambda x: x["dur"])
                grouped_op = group_operation_name(smallest["name"])
                ops_to_kernels[grouped_op]["total_us"] += m_dur
                ops_to_kernels[grouped_op]["kernels"][m_cat_name] += m_dur

    # Compute gpu_user_annotation total duration per grouped operation
    ops_ann_dur = defaultdict(float)
    for a in gpu_annotations:
        if is_leaf_operation(a["name"]) and "ProfilerStep" not in a["name"]:
            grouped = group_operation_name(a["name"])
            ops_ann_dur[grouped] += a["dur"]

    # Count how many original operations were grouped (from gpu_user_annotation)
    op_counts = defaultdict(int)
    for a in gpu_annotations:
        if is_leaf_operation(a["name"]) and "ProfilerStep" not in a["name"]:
            grouped = group_operation_name(a["name"])
            op_counts[grouped] += 1
    
    # Normalize counts (divide by number of profiler steps)
    if profiler_steps > 0:
        for op in op_counts:
            op_counts[op] = op_counts[op] // profiler_steps

    # Normalize kernel durations to per-iteration averages
    if profiler_steps > 0:
        for op in ops_to_kernels:
            ops_to_kernels[op]["total_us"] /= profiler_steps
            for k_name in ops_to_kernels[op]["kernels"]:
                ops_to_kernels[op]["kernels"][k_name] /= profiler_steps
        for op in ops_ann_dur:
            ops_ann_dur[op] /= profiler_steps

    # Get top operations sorted by gpu_user_annotation duration (matches summary.json)
    filtered_ops = {k: v for k, v in ops_to_kernels.items() 
                    if is_leaf_operation(k)
                    and "ProfilerStep" not in k
                    and not k.startswith("barrier_")}
    # Sort by annotation duration (= summary.json gpu_time_ms) for consistency
    sorted_ops = sorted(
        filtered_ops.items(),
        key=lambda x: ops_ann_dur.get(x[0], x[1]["total_us"]),
        reverse=True,
    )[:12]

    if not sorted_ops:
        return

    # Collect kernel categories
    all_categories = set()
    for _, stats in sorted_ops:
        for k_name in stats["kernels"].keys():
            all_categories.add(categorize_kernel(k_name))
    all_categories = sorted(all_categories)
    # Add Idle/Overhead as last category
    all_categories.append("Idle/Overhead")

    # Build data with grouped names
    op_names = []
    for op, _ in sorted_ops:
        name = op
        if op_counts.get(op, 1) > 1:
            name = f"{op} (x{op_counts[op]})"
        op_names.append(name)
    
    data_matrix = []
    for cat in all_categories:
        row = []
        for op_name, stats in sorted_ops:
            if cat == "Idle/Overhead":
                # Idle = annotation duration - total active kernel/memcpy time
                ann_dur_ms = ops_ann_dur.get(op_name, 0) / 1000
                active_ms = stats["total_us"] / 1000
                idle = max(0, ann_dur_ms - active_ms)
                row.append(idle)
            else:
                cat_time = sum(
                    dur
                    for k_name, dur in stats["kernels"].items()
                    if categorize_kernel(k_name) == cat
                )
                row.append(cat_time / 1000)  # to ms
        data_matrix.append(row)

    # Stacked bar
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(op_names))
    # Use Set2 for kernel categories, light gray for Idle
    n_cats = len(all_categories)
    colors = list(plt.cm.Set2(np.linspace(0, 1, max(n_cats - 1, 1))))
    colors.append((0.85, 0.85, 0.85, 1.0))  # light gray for Idle/Overhead

    bottom = np.zeros(len(op_names))
    for i, (cat, row_data) in enumerate(zip(all_categories, data_matrix)):
        ax.barh(x, row_data, left=bottom, label=cat, color=colors[i])
        bottom += np.array(row_data)

    ax.set_yticks(x)
    ax.set_yticklabels(op_names, fontsize=BREAKDOWN_FONTSIZES['ytick'])
    ax.invert_yaxis()
    ax.set_xlabel("GPU Time (ms)", fontsize=BREAKDOWN_FONTSIZES['xlabel'])
    ax.set_title(f"{title}\nOperation → Kernel Categories (bar total = GPU time from profiler)", fontsize=BREAKDOWN_FONTSIZES['title'])
    ax.legend(loc="lower right", fontsize=BREAKDOWN_FONTSIZES['legend'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def extract_atom_count(structure_name: str) -> int:
    """Extract atom count from structure name for sorting."""
    match = re.search(r"(\d+)atoms", structure_name)
    return int(match.group(1)) if match else 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate profiling visualization plots"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to results directory (e.g., results/2026-03-30_NVIDIA_A100-PCIE-40GB)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: {results_dir}/plots)",
    )
    parser.add_argument(
        "--structure",
        type=str,
        default=None,
        help="Generate plots for a specific structure (e.g., Cu_fcc_5x5x5_500atoms)",
    )
    args = parser.parse_args()

    base = args.results_dir
    outdir = args.output_dir or (base / "plots")
    outdir.mkdir(exist_ok=True)

    print(f"Results directory: {base}")
    print(f"Output directory: {outdir}")
    print()

    # Process each model configuration
    for summary_file in sorted(base.glob("*/*/*/summary.json")):
        model_type = summary_file.parts[-4]
        backend = summary_file.parts[-3]
        model_name = summary_file.parts[-2]

        label = f"{model_type}/{backend}/{model_name}"
        safe_label = f"{model_type}_{backend}_{model_name}"

        print(f"Processing: {label}")

        with open(summary_file) as f:
            summary = json.load(f)

        results = summary.get("results", {})
        if not results:
            print(f"  No results found, skipping")
            continue

        # Determine which structures to process
        if args.structure:
            # Specific structure
            matching = [k for k in results.keys() if args.structure in k]
            if not matching:
                print(f"  Structure '{args.structure}' not found, skipping")
                continue
            structure_keys = matching
        else:
            # Default: all structures, sorted by atom count
            structure_keys = sorted(results.keys(), key=extract_atom_count)

        for struct_key in structure_keys:
            result = results[struct_key]
            operations = result.get("operations", {})

            if not operations:
                print(f"  No operations data for {struct_key}, skipping")
                continue

            # Extract atom count for label
            atom_count = extract_atom_count(struct_key)
            struct_suffix = f"_{atom_count}atoms"
            struct_label = f"{label} ({atom_count} atoms)" if atom_count else label

            print(f"  Structure: {struct_key}")

            # 1. Operation breakdown (leaf only)
            plot_operation_breakdown(
                operations,
                struct_label,
                outdir / f"{safe_label}{struct_suffix}_breakdown.png",
                model_type=model_type,
                leaf_only=True,
            )

            # 2. Pie chart (leaf only)
            plot_pie_chart(
                operations,
                struct_label,
                outdir / f"{safe_label}{struct_suffix}_pie.png",
                model_type=model_type,
            )

            # 3. Kernel breakdown
            trace_file = result.get("trace_file")
            if trace_file and Path(trace_file).exists():
                plot_kernel_breakdown(
                    Path(trace_file),
                    struct_label,
                    outdir / f"{safe_label}{struct_suffix}_kernels.png",
                )
            else:
                print(f"    No trace file found for kernel analysis")

        print()

    # =========================================================================
    # Cross-model comparison plot
    # =========================================================================
    plot_model_comparison(base, outdir)

    print("Done!")


def plot_model_comparison(base: Path, outdir: Path):
    """Plot cross-model latency comparison across system sizes."""
    # Display name mapping: (model_type, backend) -> label
    MODEL_LABELS = {
        ("mace", "e3nn"): "MACE (e3nn)",
        ("mace", "cueq"): "MACE (cueq)",
        ("sevenn", "e3nn"): "SevenNet (e3nn)",
        ("sevenn", "cueq"): "SevenNet (cueq)",
        ("esen", "default"): "eSEN",
        ("esen", "e3nn"): "eSEN",
    }

    MODEL_COLORS = {
        "MACE (e3nn)": "#e74c3c",
        "MACE (cueq)": "#e74c3c",
        "SevenNet (e3nn)": "#3498db",
        "SevenNet (cueq)": "#3498db",
        "eSEN": "#2ecc71",
    }

    MODEL_STYLES = {
        "MACE (e3nn)": {"linestyle": "--", "marker": "o"},
        "MACE (cueq)": {"linestyle": "-", "marker": "o"},
        "SevenNet (e3nn)": {"linestyle": "--", "marker": "s"},
        "SevenNet (cueq)": {"linestyle": "-", "marker": "s"},
        "eSEN": {"linestyle": "-", "marker": "^"},
    }

    # Collect data: {label: {atoms: latency_ms}}
    model_data = {}

    for summary_file in sorted(base.glob("*/*/*/summary.json")):
        model_type = summary_file.parts[-4]
        backend = summary_file.parts[-3]

        label = MODEL_LABELS.get((model_type, backend))
        if label is None:
            continue

        with open(summary_file) as f:
            summary = json.load(f)

        results = summary.get("results", {})
        latencies = {}
        for struct_key, result in results.items():
            atoms = extract_atom_count(struct_key)
            if atoms > 0:
                # Prefer timeit (no profiler overhead) over profiler latency
                latency = result.get("timeit_mean_ms") or result.get("mean_latency_ms")
                if latency is not None:
                    latencies[atoms] = latency

        if latencies:
            model_data[label] = dict(sorted(latencies.items()))

    if not model_data:
        print("  No cross-model data found, skipping comparison plot")
        return

    print("Generating cross-model comparison plots...")

    # --- Plot 1: Latency vs System Size ---
    fig, ax = plt.subplots(figsize=(10, 7))

    for label, latencies in sorted(model_data.items()):
        atoms_list = list(latencies.keys())
        times = list(latencies.values())
        style = MODEL_STYLES.get(label, {"linestyle": "-", "marker": "x"})
        color = MODEL_COLORS.get(label, "#999999")
        ax.plot(
            atoms_list,
            times,
            label=label,
            color=color,
            linewidth=2,
            markersize=8,
            **style,
        )

    ax.set_xlabel("Number of Atoms", fontsize=14)
    ax.set_ylabel("Inference Latency (ms)", fontsize=14)
    ax.set_title("Model Comparison: Inference Latency vs System Size")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = outdir / "comparison_latency.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # --- Plot 2: Speedup (e3nn → cueq) ---
    speedup_data = {}
    for model_base in ["MACE", "SevenNet"]:
        e3nn_label = f"{model_base} (e3nn)"
        cueq_label = f"{model_base} (cueq)"
        if e3nn_label in model_data and cueq_label in model_data:
            speedups = {}
            for atoms in model_data[e3nn_label]:
                if atoms in model_data[cueq_label]:
                    speedups[atoms] = (
                        model_data[e3nn_label][atoms] / model_data[cueq_label][atoms]
                    )
            if speedups:
                speedup_data[model_base] = speedups

    if speedup_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_speedup = {"MACE": "#e74c3c", "SevenNet": "#3498db"}
        markers_speedup = {"MACE": "o", "SevenNet": "s"}

        for model_name, speedups in sorted(speedup_data.items()):
            atoms_list = list(speedups.keys())
            values = list(speedups.values())
            ax.plot(
                atoms_list,
                values,
                label=model_name,
                color=colors_speedup.get(model_name, "#999"),
                marker=markers_speedup.get(model_name, "x"),
                linewidth=2,
                markersize=8,
            )
            # Annotate values
            for a, v in zip(atoms_list, values):
                ax.annotate(
                    f"{v:.2f}x",
                    (a, v),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=11,
                )

        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Number of Atoms", fontsize=14)
        ax.set_ylabel("Speedup (e3nn \u2192 cueq)", fontsize=14)
        ax.set_title("cuEquivariance Speedup vs System Size")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = outdir / "comparison_speedup.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


if __name__ == "__main__":
    main()
