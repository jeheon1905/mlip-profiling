#!/usr/bin/env python3
"""
Visualize profiling results from summary.json files.

Usage:
    # Single summary file
    python scripts/visualize_results.py results/2026-03-27_NVIDIA_A100/esen/e3nn/esen-sm/summary.json

    # Compare multiple models
    python scripts/visualize_results.py results/2026-03-27_NVIDIA_A100/*/e3nn/*/summary.json

    # Compare backends
    python scripts/visualize_results.py results/2026-03-27_NVIDIA_A100/mace/*/mace-mp-small/summary.json
    
    # Show only leaf operations (exclude wrappers)
    python scripts/visualize_results.py --leaf-only results/.../summary.json
    
    # CUDA kernel analysis (most actionable for optimization)
    python scripts/visualize_results.py --kernels results/.../summary.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CUDA Kernel Categories for Grouping
# =============================================================================
# Map kernel name patterns to semantic categories for easier understanding

KERNEL_CATEGORIES = {
    # Matrix operations
    r'sgemm|gemm|gemv|cublas|cutlass.*Kernel': 'Matrix Multiply (GEMM)',
    r'magma_': 'MAGMA Linear Algebra',
    
    # cuEquivariance tensor products
    r'segmented_polynomial': 'cuEq Tensor Product',
    r'kernelcatcher.*tensor_product': 'cuEq Tensor Product',
    
    # Element-wise operations
    r'elementwise_kernel|vectorized_elementwise': 'Element-wise Ops',
    r'unrolled_elementwise': 'Element-wise Ops',
    
    # Reduction operations  
    r'reduce_kernel|Reduce': 'Reduction',
    
    # Memory operations
    r'scatter.*gather|index.*kernel|gather_kernel': 'Scatter/Gather',
    r'CatArray|concat': 'Concatenation',
    r'copy|memcpy|memset': 'Memory Copy',
    
    # Sorting
    r'RadixSort|sort': 'Sorting',
    
    # Attention (for transformer models)
    r'attention|softmax': 'Attention',
    
    # Activation functions
    r'sigmoid|relu|gelu|silu|swish|tanh': 'Activations',
}


def categorize_kernel(kernel_name: str) -> str:
    """Categorize a CUDA kernel name into a semantic group."""
    for pattern, category in KERNEL_CATEGORIES.items():
        if re.search(pattern, kernel_name, re.IGNORECASE):
            return category
    return 'Other'


def simplify_kernel_name(name: str) -> str:
    """Simplify CUDA kernel name by removing template parameters."""
    # Remove template parameters (everything between < and >)
    simplified = re.sub(r'<[^>]*>', '', name)
    # Remove function parameters
    simplified = re.sub(r'\([^)]*\)', '', simplified)
    # Clean up namespace prefixes
    simplified = re.sub(r'^void\s+', '', simplified)
    simplified = re.sub(r'^at::native::', '', simplified)
    simplified = re.sub(r'^\(anonymous namespace\)::', '', simplified)
    return simplified.strip()


# =============================================================================
# Operation Hierarchy Definitions
# =============================================================================
# Wrapper operations that contain child operations (should be excluded in leaf-only mode)

WRAPPER_OPERATIONS = {
    # eSEN wrappers (contain child operations)
    "eSEN::model_forward",      # Contains forward, compute_forces, etc.
    "forward",                  # Contains generate_graph, message passing, etc.
    "message passing 0",        # Contains SO2Conv, edgewise, atomwise
    "message passing 1",
    "message passing 2", 
    "message passing 3",
    
    # MACE wrappers
    "MACE::Interaction::forward",  # Contains skip_tp, linear_up, etc.
    
    # SevenNet - mostly leaf operations, few wrappers
}

# Leaf operations by model type (GPU time is meaningful for these)
LEAF_OPERATIONS = {
    "esen": [
        # Graph generation
        "generate_graph",
        "obtain wigner",
        "obtain rotmat wigner original",
        # Embeddings
        "atom embedding",
        "edge embedding",
        "charge spin dataset embeddings",
        # Core operations (inside message passing)
        "SO2Conv",
        "edgewise",
        "atomwise",
        # Output
        "balance_channels",
        "final_norm",
        # Force computation
        "eSEN::compute_forces",
        "eSEN::compute_forces_stress",
        # Data handling
        "eSEN::data_preparation",
        "eSEN::data_to_device",
        "eSEN::process_outputs",
    ],
    "mace": [
        # Graph generation
        "generate_graph",
        # Model operations
        "MACE::prepare_graph",
        "MACE::atomic_energies",
        "MACE::embeddings",
        "MACE::interaction_0",
        "MACE::interaction_1",
        "MACE::product_0",
        "MACE::product_1",
        "MACE::ProductBasis",
        "MACE::SymmetricContraction",
        "MACE::readouts",
        "MACE::get_outputs",
        # Force computation
        "MACE::compute_forces",
        "MACE::compute_forces_virials",
        # Block internals (if needed)
        "MACE::Interaction::skip_tp",
    ],
    "sevenn": [
        # Graph generation
        "generate_graph",
        # Input processing
        "SevenNet::edge_embedding",
        "SevenNet::onehot_idx_to_onehot",
        "SevenNet::onehot_to_feature_x",
        # Layer operations (all layers 0-4)
        *[f"SevenNet::{i}_{op}" for i in range(5) for op in [
            "self_connection_intro",
            "self_interaction_1",
            "convolution",
            "self_interaction_2",
            "self_connection_outro",
            "equivariant_gate",
        ]],
        # Output
        "SevenNet::reduce_output",
        "SevenNet::force_output",
    ],
}


def is_leaf_operation(op_name: str, model_type: str = None) -> bool:
    """Check if operation is a leaf (not a wrapper)."""
    # If in wrapper list, not a leaf
    if op_name in WRAPPER_OPERATIONS:
        return False
    
    # If model type specified, check against known leaves
    if model_type and model_type in LEAF_OPERATIONS:
        return op_name in LEAF_OPERATIONS[model_type]
    
    # Default: not a known wrapper = treat as leaf
    return True


def filter_leaf_operations(operations: dict, model_type: str = None) -> dict:
    """Filter to only leaf operations."""
    return {
        name: data for name, data in operations.items()
        if is_leaf_operation(name, model_type)
    }


def detect_model_type(summary: dict) -> str:
    """Detect model type from summary."""
    model_info = summary.get("model", {})
    return model_info.get("type", "unknown")


def load_summary(path: Path) -> dict:
    """Load summary.json file."""
    with open(path) as f:
        return json.load(f)


def get_label_from_path(path: Path) -> str:
    """Extract label from summary.json path.
    
    Expected structure: .../model_type/backend/model_name/summary.json
    Returns: model_type/backend/model_name
    """
    parts = path.parts
    # Find summary.json index and go back
    try:
        model_name = parts[-2]
        backend = parts[-3]
        model_type = parts[-4]
        return f"{model_type}/{backend}/{model_name}"
    except IndexError:
        return path.parent.name


def plot_operation_breakdown(summary: dict, title: str, output_path: Path = None, 
                            leaf_only: bool = False):
    """Plot operation-level time breakdown for a single model."""
    # Get first result (assuming single structure)
    result_key = list(summary["results"].keys())[0]
    result = summary["results"][result_key]
    
    operations = result.get("operations", {})
    if not operations:
        print(f"No operation data found in {title}")
        return
    
    # Filter to leaf operations if requested
    model_type = detect_model_type(summary)
    if leaf_only:
        operations = filter_leaf_operations(operations, model_type)
        title_suffix = " (Leaf Operations Only)"
    else:
        title_suffix = ""
    
    # Sort by GPU time
    sorted_ops = sorted(
        operations.items(),
        key=lambda x: x[1].get("gpu_time_ms", 0),
        reverse=True
    )
    
    # Take top 15 operations
    top_ops = sorted_ops[:15]
    
    names = [op[0] for op in top_ops]
    gpu_times = [op[1].get("gpu_time_ms", 0) for op in top_ops]
    cpu_times = [op[1].get("cpu_time_ms", 0) for op in top_ops]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(names))
    bar_height = 0.35
    
    bars1 = ax.barh(y_pos - bar_height/2, gpu_times, bar_height, label='GPU Time', color='#2ecc71')
    bars2 = ax.barh(y_pos + bar_height/2, cpu_times, bar_height, label='CPU Time', color='#3498db', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{title}\nOperation Breakdown (Top 15 by GPU time){title_suffix}')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        if width > 0.1:
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_comparison(summaries: list[tuple[str, dict]], output_path: Path = None):
    """Compare total latency across models."""
    labels = []
    latencies = []
    stds = []
    
    for label, summary in summaries:
        result_key = list(summary["results"].keys())[0]
        result = summary["results"][result_key]
        
        labels.append(label.replace("/", "\n"))
        latencies.append(result["timeit_mean_ms"])
        stds.append(result.get("timeit_std_ms", 0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, latencies, yerr=stds, capsize=5, color='#3498db', alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Model Latency Comparison')
    
    # Add value labels
    for bar, lat, std in zip(bars, latencies, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
               f'{lat:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_operation_comparison(summaries: list[tuple[str, dict]], output_path: Path = None):
    """Compare operation times across models."""
    # Collect all operations
    all_ops = set()
    for _, summary in summaries:
        result_key = list(summary["results"].keys())[0]
        operations = summary["results"][result_key].get("operations", {})
        all_ops.update(operations.keys())
    
    # Common high-level operations to compare
    key_ops = [
        "generate_graph", "forward",
        "eSEN::compute_forces", "MACE::compute_forces", 
        "message passing 0", "MACE::interaction_0", "SevenNet::0_convolution",
    ]
    
    # Filter to existing operations
    ops_to_plot = [op for op in key_ops if op in all_ops]
    
    if not ops_to_plot:
        # Fallback: use top operations from first model
        _, first_summary = summaries[0]
        result_key = list(first_summary["results"].keys())[0]
        operations = first_summary["results"][result_key].get("operations", {})
        ops_to_plot = sorted(operations.keys(), 
                            key=lambda x: operations[x].get("gpu_time_ms", 0),
                            reverse=True)[:8]
    
    # Build data matrix
    data = defaultdict(dict)
    for label, summary in summaries:
        result_key = list(summary["results"].keys())[0]
        operations = summary["results"][result_key].get("operations", {})
        for op in ops_to_plot:
            data[label][op] = operations.get(op, {}).get("gpu_time_ms", 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(ops_to_plot))
    width = 0.8 / len(summaries)
    colors = plt.cm.Set2(np.linspace(0, 1, len(summaries)))
    
    for i, (label, _) in enumerate(summaries):
        values = [data[label].get(op, 0) for op in ops_to_plot]
        offset = (i - len(summaries)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(ops_to_plot, rotation=45, ha='right')
    ax.set_ylabel('GPU Time (ms)')
    ax.set_title('Operation Time Comparison (GPU)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pie_chart(summary: dict, title: str, output_path: Path = None,
                   leaf_only: bool = False):
    """Plot pie chart of operation breakdown."""
    result_key = list(summary["results"].keys())[0]
    result = summary["results"][result_key]
    
    operations = result.get("operations", {})
    if not operations:
        return
    
    # Filter to leaf operations if requested
    model_type = detect_model_type(summary)
    if leaf_only:
        operations = filter_leaf_operations(operations, model_type)
        title_suffix = " (Leaf Only)"
    else:
        title_suffix = ""
    
    # Sort and get top operations
    sorted_ops = sorted(
        operations.items(),
        key=lambda x: x[1].get("gpu_time_ms", 0),
        reverse=True
    )
    
    # Take top 8 and group rest as "Other"
    top_ops = sorted_ops[:8]
    other_time = sum(op[1].get("gpu_time_ms", 0) for op in sorted_ops[8:])
    
    labels = [op[0] for op in top_ops]
    sizes = [op[1].get("gpu_time_ms", 0) for op in top_ops]
    
    if other_time > 0:
        labels.append("Other")
        sizes.append(other_time)
    
    # Remove zero values
    non_zero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    if not non_zero:
        return
    labels, sizes = zip(*non_zero)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    
    ax.set_title(f'{title}\nGPU Time Distribution{title_suffix}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_forward_vs_backward(summaries: list[tuple[str, dict]], output_path: Path = None):
    """Compare forward vs backward (force computation) time."""
    labels = []
    forward_times = []
    backward_times = []
    
    # Operation names for forward and backward
    forward_ops = ["forward"]
    backward_ops = ["eSEN::compute_forces", "MACE::compute_forces", "MACE::compute_forces_virials"]
    
    for label, summary in summaries:
        result_key = list(summary["results"].keys())[0]
        operations = summary["results"][result_key].get("operations", {})
        
        # Sum forward times
        fwd = sum(operations.get(op, {}).get("gpu_time_ms", 0) for op in forward_ops)
        
        # Sum backward times
        bwd = sum(operations.get(op, {}).get("gpu_time_ms", 0) for op in backward_ops)
        
        if fwd > 0 or bwd > 0:
            labels.append(label.replace("/", "\n"))
            forward_times.append(fwd)
            backward_times.append(bwd)
    
    if not labels:
        print("No forward/backward data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, forward_times, width, label='Forward', color='#3498db')
    ax.bar(x + width/2, backward_times, width, label='Backward (Forces)', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('GPU Time (ms)')
    ax.set_title('Forward vs Backward Time Comparison')
    ax.legend()
    
    # Add percentage labels
    for i, (fwd, bwd) in enumerate(zip(forward_times, backward_times)):
        total = fwd + bwd
        if total > 0:
            fwd_pct = fwd / total * 100
            bwd_pct = bwd / total * 100
            ax.text(i, max(fwd, bwd) + 2, f'F:{fwd_pct:.0f}% B:{bwd_pct:.0f}%', 
                   ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# CUDA Kernel Analysis Functions
# =============================================================================

def find_trace_file(summary_path: Path) -> Path:
    """Find the trace.json file associated with a summary.json."""
    # summary.json is in the same directory as trace files
    summary_dir = summary_path.parent
    
    # Look for trace files
    trace_files = list(summary_dir.glob("*.trace.json"))
    if trace_files:
        return trace_files[0]
    
    return None


def extract_cuda_kernels(trace_path: Path) -> dict:
    """Extract CUDA kernel timing information from a Chrome trace file.
    
    Returns dict with:
        - kernels_by_name: {kernel_name: {total_us, count}}
        - kernels_by_category: {category: {total_us, count}}
        - total_kernel_time_us: total GPU kernel execution time
    """
    with open(trace_path) as f:
        data = json.load(f)
    
    events = data.get('traceEvents', [])
    
    # Aggregate kernels by name
    kernels_by_name = defaultdict(lambda: {'total_us': 0, 'count': 0})
    kernels_by_category = defaultdict(lambda: {'total_us': 0, 'count': 0})
    
    for event in events:
        if event.get('cat') == 'kernel':
            name = event.get('name', 'Unknown')
            duration = event.get('dur', 0)
            
            # Simplify name for aggregation
            simple_name = simplify_kernel_name(name)
            
            kernels_by_name[simple_name]['total_us'] += duration
            kernels_by_name[simple_name]['count'] += 1
            
            # Categorize
            category = categorize_kernel(name)
            kernels_by_category[category]['total_us'] += duration
            kernels_by_category[category]['count'] += 1
    
    total_kernel_time = sum(k['total_us'] for k in kernels_by_name.values())
    
    return {
        'kernels_by_name': dict(kernels_by_name),
        'kernels_by_category': dict(kernels_by_category),
        'total_kernel_time_us': total_kernel_time,
    }


def plot_kernel_breakdown(kernel_data: dict, title: str, output_path: Path = None):
    """Plot CUDA kernel breakdown by category."""
    categories = kernel_data['kernels_by_category']
    total_time = kernel_data['total_kernel_time_us']
    
    if not categories:
        print(f"No kernel data found for {title}")
        return
    
    # Sort by total time
    sorted_cats = sorted(
        categories.items(),
        key=lambda x: x[1]['total_us'],
        reverse=True
    )
    
    labels = [cat for cat, _ in sorted_cats]
    times_ms = [data['total_us'] / 1000 for _, data in sorted_cats]
    counts = [data['count'] for _, data in sorted_cats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, times_ms, color=colors)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(f'{title}\nCUDA Kernel Breakdown by Category')
    
    # Add labels with percentage and count
    for i, (bar, t, c) in enumerate(zip(bars, times_ms, counts)):
        pct = t * 1000 / total_time * 100 if total_time > 0 else 0
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{t:.1f}ms ({pct:.1f}%, {c} calls)', va='center', fontsize=9)
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        times_ms, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90
    )
    ax2.set_title(f'GPU Time Distribution\n(Total: {total_time/1000:.1f}ms)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_kernel_detail(kernel_data: dict, title: str, output_path: Path = None, top_n: int = 15):
    """Plot detailed CUDA kernel breakdown by individual kernel."""
    kernels = kernel_data['kernels_by_name']
    total_time = kernel_data['total_kernel_time_us']
    
    if not kernels:
        return
    
    # Sort by total time
    sorted_kernels = sorted(
        kernels.items(),
        key=lambda x: x[1]['total_us'],
        reverse=True
    )[:top_n]
    
    names = [name[:50] for name, _ in sorted_kernels]
    times_ms = [data['total_us'] / 1000 for _, data in sorted_kernels]
    counts = [data['count'] for _, data in sorted_kernels]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(y_pos, times_ms, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{title}\nTop {top_n} CUDA Kernels by Time')
    
    # Add labels
    for bar, t, c in zip(bars, times_ms, counts):
        pct = t * 1000 / total_time * 100 if total_time > 0 else 0
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
               f'{t:.2f}ms ({pct:.1f}%), {c}x', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_kernel_comparison(kernel_data_list: list[tuple[str, dict]], output_path: Path = None):
    """Compare CUDA kernel categories across models."""
    # Collect all categories
    all_categories = set()
    for _, data in kernel_data_list:
        all_categories.update(data['kernels_by_category'].keys())
    
    # Sort categories by total time across all models
    cat_totals = defaultdict(float)
    for _, data in kernel_data_list:
        for cat, stats in data['kernels_by_category'].items():
            cat_totals[cat] += stats['total_us']
    
    sorted_categories = sorted(cat_totals.keys(), key=lambda x: cat_totals[x], reverse=True)[:10]
    
    # Build data matrix
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(sorted_categories))
    width = 0.8 / len(kernel_data_list)
    colors = plt.cm.Set1(np.linspace(0, 1, len(kernel_data_list)))
    
    for i, (label, data) in enumerate(kernel_data_list):
        values = [data['kernels_by_category'].get(cat, {}).get('total_us', 0) / 1000 
                  for cat in sorted_categories]
        offset = (i - len(kernel_data_list)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_categories, rotation=45, ha='right')
    ax.set_ylabel('Time (ms)')
    ax.set_title('CUDA Kernel Category Comparison\n(Lower is Better for Optimization)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_kernel_summary(kernel_data_list: list[tuple[str, dict]]):
    """Print a text summary of kernel analysis."""
    print("\n" + "=" * 80)
    print("CUDA KERNEL ANALYSIS SUMMARY")
    print("=" * 80)
    print("\nOptimization Targets (by category):")
    print("-" * 80)
    
    for label, data in kernel_data_list:
        total_ms = data['total_kernel_time_us'] / 1000
        print(f"\n{label} (Total: {total_ms:.1f}ms)")
        
        sorted_cats = sorted(
            data['kernels_by_category'].items(),
            key=lambda x: x[1]['total_us'],
            reverse=True
        )
        
        cumulative = 0
        for cat, stats in sorted_cats[:8]:
            time_ms = stats['total_us'] / 1000
            pct = stats['total_us'] / data['total_kernel_time_us'] * 100
            cumulative += pct
            print(f"  {cat:25s}: {time_ms:7.2f}ms ({pct:5.1f}%) - {stats['count']:4d} calls - {cumulative:5.1f}% cumulative")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    # Analyze common patterns
    for label, data in kernel_data_list:
        cats = data['kernels_by_category']
        total = data['total_kernel_time_us']
        
        print(f"\n{label}:")
        
        # Element-wise ops
        elem_pct = cats.get('Element-wise Ops', {}).get('total_us', 0) / total * 100 if total > 0 else 0
        if elem_pct > 25:
            print(f"  - Element-wise ops take {elem_pct:.1f}% → Consider kernel fusion or operator fusion")
        
        # GEMM
        gemm_pct = cats.get('Matrix Multiply (GEMM)', {}).get('total_us', 0) / total * 100 if total > 0 else 0
        if gemm_pct > 20:
            print(f"  - GEMM takes {gemm_pct:.1f}% → Consider Tensor Cores (FP16/BF16) or Flash-style algorithms")
        
        # cuEq
        cueq_pct = cats.get('cuEq Tensor Product', {}).get('total_us', 0) / total * 100 if total > 0 else 0
        if cueq_pct > 5:
            print(f"  - cuEquivariance tensor products: {cueq_pct:.1f}% (optimized)")
        
        # Scatter/Gather
        scatter_pct = cats.get('Scatter/Gather', {}).get('total_us', 0) / total * 100 if total > 0 else 0
        if scatter_pct > 10:
            print(f"  - Scatter/Gather takes {scatter_pct:.1f}% → Memory-bound; consider data layout optimization")
        
        # Reduction
        reduce_pct = cats.get('Reduction', {}).get('total_us', 0) / total * 100 if total > 0 else 0
        if reduce_pct > 10:
            print(f"  - Reduction takes {reduce_pct:.1f}% → Consider hierarchical reduction or segment reduction")


def main():
    parser = argparse.ArgumentParser(description="Visualize profiling results")
    parser.add_argument("summary_files", nargs="+", type=Path,
                       help="Path(s) to summary.json file(s)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                       help="Output directory for plots (default: display)")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png",
                       help="Output format (default: png)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots, only save")
    parser.add_argument("--leaf-only", "-l", action="store_true",
                       help="Show only leaf operations (exclude wrappers like 'forward')")
    parser.add_argument("--kernels", "-k", action="store_true",
                       help="Analyze CUDA kernels from trace files (most actionable for optimization)")
    
    args = parser.parse_args()
    
    # Expand glob patterns and load summaries
    summaries = []
    summary_paths = []
    for pattern in args.summary_files:
        if "*" in str(pattern):
            files = list(Path(".").glob(str(pattern)))
        else:
            files = [pattern]
        
        for f in files:
            if f.exists():
                label = get_label_from_path(f)
                summary = load_summary(f)
                summaries.append((label, summary))
                summary_paths.append((label, f))
                print(f"Loaded: {label}")
    
    if not summaries:
        print("No summary files found")
        return
    
    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # CUDA Kernel Analysis (if requested)
    # ==========================================================================
    if args.kernels:
        print("\n" + "=" * 80)
        print("CUDA KERNEL ANALYSIS MODE")
        print("=" * 80)
        
        kernel_data_list = []
        for label, summary_path in summary_paths:
            trace_path = find_trace_file(summary_path)
            if trace_path and trace_path.exists():
                print(f"Analyzing: {trace_path.name}")
                kernel_data = extract_cuda_kernels(trace_path)
                kernel_data_list.append((label, kernel_data))
                
                # Individual kernel plots
                if args.output_dir:
                    safe_label = label.replace("/", "_").replace(" ", "_")
                    plot_kernel_breakdown(
                        kernel_data, label,
                        args.output_dir / f"{safe_label}_kernels.{args.format}"
                    )
                    plot_kernel_detail(
                        kernel_data, label,
                        args.output_dir / f"{safe_label}_kernels_detail.{args.format}"
                    )
                else:
                    plot_kernel_breakdown(kernel_data, label)
                    plot_kernel_detail(kernel_data, label)
            else:
                print(f"  Warning: No trace file found for {label}")
        
        # Comparison plot if multiple models
        if len(kernel_data_list) > 1:
            plot_kernel_comparison(
                kernel_data_list,
                args.output_dir / f"kernel_comparison.{args.format}" if args.output_dir else None
            )
        
        # Print text summary with recommendations
        if kernel_data_list:
            print_kernel_summary(kernel_data_list)
        
        print("\nKernel analysis complete!")
        return
    
    # ==========================================================================
    # Standard Operation-level Analysis
    # ==========================================================================
    
    # Generate plots
    if len(summaries) == 1:
        # Single model: detailed breakdown
        label, summary = summaries[0]
        safe_label = label.replace("/", "_").replace(" ", "_")
        
        plot_operation_breakdown(
            summary, label,
            args.output_dir / f"{safe_label}_breakdown.{args.format}" if args.output_dir else None,
            leaf_only=args.leaf_only
        )
        
        plot_pie_chart(
            summary, label,
            args.output_dir / f"{safe_label}_pie.{args.format}" if args.output_dir else None,
            leaf_only=args.leaf_only
        )
    else:
        # Multiple models: comparison
        plot_model_comparison(
            summaries,
            args.output_dir / f"latency_comparison.{args.format}" if args.output_dir else None
        )
        
        plot_operation_comparison(
            summaries,
            args.output_dir / f"operation_comparison.{args.format}" if args.output_dir else None
        )
        
        plot_forward_vs_backward(
            summaries,
            args.output_dir / f"forward_vs_backward.{args.format}" if args.output_dir else None
        )
        
        # Also generate individual breakdowns and pie charts
        for label, summary in summaries:
            safe_label = label.replace("/", "_").replace(" ", "_")
            plot_operation_breakdown(
                summary, label,
                args.output_dir / f"{safe_label}_breakdown.{args.format}" if args.output_dir else None,
                leaf_only=args.leaf_only
            )
            plot_pie_chart(
                summary, label,
                args.output_dir / f"{safe_label}_pie.{args.format}" if args.output_dir else None,
                leaf_only=args.leaf_only
            )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
