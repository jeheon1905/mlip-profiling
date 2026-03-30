#!/usr/bin/env python3
"""
Analyze CUDA kernels from PyTorch Profiler trace files.

This script provides deep kernel-level analysis to identify optimization targets.
It maps CUDA kernels to high-level operations (record_function tags) and provides
actionable insights for performance optimization.

Usage:
    # Analyze single trace
    python scripts/analyze_kernels.py results/.../model.trace.json

    # Compare multiple traces
    python scripts/analyze_kernels.py results/.../*.trace.json

    # Save plots
    python scripts/analyze_kernels.py -o output_dir results/.../*.trace.json
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Kernel Categories
# =============================================================================

KERNEL_CATEGORIES = {
    # Matrix operations
    r'sgemm|gemm|gemv|cublas|cutlass.*Kernel': 'GEMM',
    r'magma_': 'MAGMA',
    
    # cuEquivariance tensor products
    r'segmented_polynomial': 'cuEq TensorProduct',
    r'kernelcatcher.*tensor_product': 'cuEq TensorProduct',
    
    # Element-wise operations  
    r'elementwise_kernel|vectorized_elementwise': 'Elementwise',
    r'unrolled_elementwise': 'Elementwise',
    
    # Reduction operations  
    r'reduce_kernel|Reduce': 'Reduction',
    
    # Memory operations
    r'scatter.*gather|index.*kernel|gather_kernel': 'Scatter/Gather',
    r'CatArray|concat': 'Concat',
    r'copy|memcpy|memset': 'Memory',
    
    # Sorting
    r'RadixSort|sort': 'Sort',
}


def categorize_kernel(kernel_name: str) -> str:
    """Categorize a CUDA kernel name."""
    for pattern, category in KERNEL_CATEGORIES.items():
        if re.search(pattern, kernel_name, re.IGNORECASE):
            return category
    return 'Other'


def simplify_kernel_name(name: str, max_len: int = 40) -> str:
    """Simplify CUDA kernel name."""
    if '<' in name:
        name = name.split('<')[0]
    name = re.sub(r'^void\s+', '', name)
    name = re.sub(r'^at::native::', '', name)
    name = re.sub(r'^\(anonymous namespace\)::', '', name)
    return name[:max_len]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class KernelStats:
    total_us: float = 0
    count: int = 0
    kernels: dict = field(default_factory=dict)


@dataclass
class TraceAnalysis:
    name: str
    total_kernel_time_us: float
    ops_to_kernels: dict  # op_name -> KernelStats
    kernels_by_category: dict  # category -> {total_us, count}
    kernels_by_name: dict  # kernel_name -> {total_us, count}


# =============================================================================
# Analysis Functions
# =============================================================================

def load_trace(trace_path: Path) -> dict:
    """Load Chrome trace file."""
    with open(trace_path) as f:
        return json.load(f)


def analyze_trace(trace_path: Path) -> TraceAnalysis:
    """Analyze a trace file and extract kernel information."""
    data = load_trace(trace_path)
    events = data.get('traceEvents', [])
    
    # Collect user annotations (record_function tags)
    user_annotations = []
    for e in events:
        if e.get('cat') == 'user_annotation':
            user_annotations.append({
                'name': e.get('name', ''),
                'ts': e.get('ts', 0),
                'dur': e.get('dur', 0),
                'end': e.get('ts', 0) + e.get('dur', 0)
            })
    
    # Collect kernels
    kernels = []
    for e in events:
        if e.get('cat') == 'kernel':
            kernels.append({
                'name': e.get('name', ''),
                'ts': e.get('ts', 0),
                'dur': e.get('dur', 0)
            })
    
    # Map kernels to operations
    ops_to_kernels = defaultdict(lambda: KernelStats())
    kernels_by_category = defaultdict(lambda: {'total_us': 0, 'count': 0})
    kernels_by_name = defaultdict(lambda: {'total_us': 0, 'count': 0})
    
    for kernel in kernels:
        k_start = kernel['ts']
        k_end = k_start + kernel['dur']
        k_name_raw = kernel['name']
        k_name = simplify_kernel_name(k_name_raw)
        k_dur = kernel['dur']
        
        # Find containing annotation (most specific)
        containing = [a for a in user_annotations if a['ts'] <= k_start and a['end'] >= k_end]
        if containing:
            smallest = min(containing, key=lambda x: x['dur'])
            op_name = smallest['name']
            
            ops_to_kernels[op_name].total_us += k_dur
            ops_to_kernels[op_name].count += 1
            if k_name not in ops_to_kernels[op_name].kernels:
                ops_to_kernels[op_name].kernels[k_name] = 0
            ops_to_kernels[op_name].kernels[k_name] += k_dur
        
        # Categorize
        category = categorize_kernel(k_name_raw)
        kernels_by_category[category]['total_us'] += k_dur
        kernels_by_category[category]['count'] += 1
        
        # By name
        kernels_by_name[k_name]['total_us'] += k_dur
        kernels_by_name[k_name]['count'] += 1
    
    total_kernel_time = sum(k['total_us'] for k in kernels_by_name.values())
    
    # Get label from path
    parts = trace_path.parts
    try:
        name = f"{parts[-4]}/{parts[-3]}/{parts[-2]}"
    except IndexError:
        name = trace_path.stem
    
    return TraceAnalysis(
        name=name,
        total_kernel_time_us=total_kernel_time,
        ops_to_kernels=dict(ops_to_kernels),
        kernels_by_category=dict(kernels_by_category),
        kernels_by_name=dict(kernels_by_name),
    )


# =============================================================================
# Output Functions
# =============================================================================

def print_ops_to_kernels(analysis: TraceAnalysis):
    """Print operation to kernel mapping."""
    print(f"\n{'='*80}")
    print(f"{analysis.name}: Operations → CUDA Kernels")
    print(f"Total kernel time: {analysis.total_kernel_time_us/1000:.2f}ms")
    print(f"{'='*80}")
    
    sorted_ops = sorted(
        analysis.ops_to_kernels.items(),
        key=lambda x: x[1].total_us,
        reverse=True
    )
    
    for op_name, stats in sorted_ops[:15]:
        pct = stats.total_us / analysis.total_kernel_time_us * 100 if analysis.total_kernel_time_us > 0 else 0
        print(f"\n{op_name}: {stats.total_us/1000:.2f}ms ({pct:.1f}%)")
        
        sorted_kernels = sorted(stats.kernels.items(), key=lambda x: x[1], reverse=True)
        for kernel, dur in sorted_kernels[:3]:
            k_pct = dur / stats.total_us * 100 if stats.total_us > 0 else 0
            print(f"    └─ {kernel}: {dur/1000:.2f}ms ({k_pct:.0f}%)")


def print_kernel_categories(analysis: TraceAnalysis):
    """Print kernel category breakdown."""
    print(f"\n{'-'*80}")
    print(f"Kernel Categories:")
    print(f"{'-'*80}")
    
    sorted_cats = sorted(
        analysis.kernels_by_category.items(),
        key=lambda x: x[1]['total_us'],
        reverse=True
    )
    
    cumulative = 0
    for cat, stats in sorted_cats:
        pct = stats['total_us'] / analysis.total_kernel_time_us * 100 if analysis.total_kernel_time_us > 0 else 0
        cumulative += pct
        print(f"  {cat:20s}: {stats['total_us']/1000:8.2f}ms ({pct:5.1f}%) - {stats['count']:5d} calls - {cumulative:5.1f}% cum")


def print_optimization_hints(analysis: TraceAnalysis):
    """Print optimization recommendations."""
    print(f"\n{'-'*80}")
    print(f"Optimization Hints:")
    print(f"{'-'*80}")
    
    cats = analysis.kernels_by_category
    total = analysis.total_kernel_time_us
    
    # Element-wise
    elem_pct = cats.get('Elementwise', {}).get('total_us', 0) / total * 100 if total > 0 else 0
    if elem_pct > 25:
        print(f"  • Elementwise ops: {elem_pct:.1f}% → Kernel fusion (torch.compile) recommended")
    
    # GEMM
    gemm_pct = cats.get('GEMM', {}).get('total_us', 0) / total * 100 if total > 0 else 0
    if gemm_pct > 20:
        print(f"  • GEMM: {gemm_pct:.1f}% → Consider Tensor Cores (FP16/BF16)")
    
    # cuEq
    cueq_pct = cats.get('cuEq TensorProduct', {}).get('total_us', 0) / total * 100 if total > 0 else 0
    if cueq_pct > 5:
        print(f"  • cuEquivariance: {cueq_pct:.1f}% (already optimized)")
    elif 'convolution' in str(analysis.ops_to_kernels.keys()).lower():
        print(f"  • No cuEquivariance detected → Consider --backend cueq")
    
    # Reduction
    reduce_pct = cats.get('Reduction', {}).get('total_us', 0) / total * 100 if total > 0 else 0
    if reduce_pct > 10:
        print(f"  • Reduction: {reduce_pct:.1f}% → Segment reduction or hierarchical reduction")
    
    # Scatter/Gather
    scatter_pct = cats.get('Scatter/Gather', {}).get('total_us', 0) / total * 100 if total > 0 else 0
    if scatter_pct > 10:
        print(f"  • Scatter/Gather: {scatter_pct:.1f}% → Memory-bound; optimize data layout")


def plot_ops_breakdown(analysis: TraceAnalysis, output_path: Path = None):
    """Plot operations to kernel breakdown."""
    sorted_ops = sorted(
        analysis.ops_to_kernels.items(),
        key=lambda x: x[1].total_us,
        reverse=True
    )[:12]
    
    if not sorted_ops:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    op_names = [op for op, _ in sorted_ops]
    
    # Collect all kernel categories for stacking
    all_categories = set()
    for _, stats in sorted_ops:
        for k_name in stats.kernels.keys():
            all_categories.add(categorize_kernel(k_name))
    all_categories = sorted(all_categories)
    
    # Build data matrix
    data_matrix = []
    for cat in all_categories:
        row = []
        for op_name, stats in sorted_ops:
            cat_time = sum(
                dur for k_name, dur in stats.kernels.items()
                if categorize_kernel(k_name) == cat
            )
            row.append(cat_time / 1000)  # to ms
        data_matrix.append(row)
    
    # Stacked bar
    x = np.arange(len(op_names))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_categories)))
    
    bottom = np.zeros(len(op_names))
    for i, (cat, row_data) in enumerate(zip(all_categories, data_matrix)):
        ax.barh(x, row_data, left=bottom, label=cat, color=colors[i])
        bottom += np.array(row_data)
    
    ax.set_yticks(x)
    ax.set_yticklabels(op_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'{analysis.name}\nOperations → Kernel Categories')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_category_comparison(analyses: list[TraceAnalysis], output_path: Path = None):
    """Compare kernel categories across models."""
    # Collect all categories
    all_categories = set()
    for a in analyses:
        all_categories.update(a.kernels_by_category.keys())
    
    # Sort by total across all
    cat_totals = defaultdict(float)
    for a in analyses:
        for cat, stats in a.kernels_by_category.items():
            cat_totals[cat] += stats['total_us']
    sorted_categories = sorted(cat_totals.keys(), key=lambda x: cat_totals[x], reverse=True)[:8]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(sorted_categories))
    width = 0.8 / len(analyses)
    colors = plt.cm.Set1(np.linspace(0, 1, len(analyses)))
    
    for i, a in enumerate(analyses):
        values = [a.kernels_by_category.get(cat, {}).get('total_us', 0) / 1000 
                  for cat in sorted_categories]
        offset = (i - len(analyses)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=a.name, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_categories, rotation=45, ha='right')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Kernel Category Comparison')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA kernels from PyTorch Profiler traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single trace
    python scripts/analyze_kernels.py results/.../model.trace.json

    # Compare multiple models
    python scripts/analyze_kernels.py results/*/*/*/*.trace.json

    # Save plots
    python scripts/analyze_kernels.py -o plots results/*/*/*/*.trace.json
        """
    )
    parser.add_argument("trace_files", nargs="+", type=Path,
                       help="Path(s) to trace.json file(s)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                       help="Output directory for plots")
    parser.add_argument("--no-text", action="store_true",
                       help="Skip text output, only generate plots")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip plots, only show text output")
    
    args = parser.parse_args()
    
    # Find trace files
    trace_files = []
    for pattern in args.trace_files:
        if "*" in str(pattern):
            trace_files.extend(Path(".").glob(str(pattern)))
        elif pattern.exists():
            trace_files.append(pattern)
    
    if not trace_files:
        print("No trace files found")
        return
    
    # Create output directory
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze traces
    analyses = []
    for trace_path in trace_files:
        print(f"Analyzing: {trace_path.name}")
        try:
            analysis = analyze_trace(trace_path)
            analyses.append(analysis)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not analyses:
        print("No valid traces analyzed")
        return
    
    # Text output
    if not args.no_text:
        for analysis in analyses:
            print_ops_to_kernels(analysis)
            print_kernel_categories(analysis)
            print_optimization_hints(analysis)
    
    # Plots
    if not args.no_plot:
        for analysis in analyses:
            safe_name = analysis.name.replace("/", "_").replace(" ", "_")
            plot_ops_breakdown(
                analysis,
                args.output_dir / f"{safe_name}_ops_kernels.png" if args.output_dir else None
            )
        
        if len(analyses) > 1:
            plot_category_comparison(
                analyses,
                args.output_dir / "kernel_category_comparison.png" if args.output_dir else None
            )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
