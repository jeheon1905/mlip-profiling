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
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


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


def plot_operation_breakdown(summary: dict, title: str, output_path: Path = None):
    """Plot operation-level time breakdown for a single model."""
    # Get first result (assuming single structure)
    result_key = list(summary["results"].keys())[0]
    result = summary["results"][result_key]
    
    operations = result.get("operations", {})
    if not operations:
        print(f"No operation data found in {title}")
        return
    
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
    ax.set_title(f'{title}\nOperation Breakdown (Top 15 by GPU time)')
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


def plot_pie_chart(summary: dict, title: str, output_path: Path = None):
    """Plot pie chart of operation breakdown."""
    result_key = list(summary["results"].keys())[0]
    result = summary["results"][result_key]
    
    operations = result.get("operations", {})
    if not operations:
        return
    
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
    
    ax.set_title(f'{title}\nGPU Time Distribution')
    
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
    
    args = parser.parse_args()
    
    # Expand glob patterns and load summaries
    summaries = []
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
                print(f"Loaded: {label}")
    
    if not summaries:
        print("No summary files found")
        return
    
    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if len(summaries) == 1:
        # Single model: detailed breakdown
        label, summary = summaries[0]
        safe_label = label.replace("/", "_").replace(" ", "_")
        
        plot_operation_breakdown(
            summary, label,
            args.output_dir / f"{safe_label}_breakdown.{args.format}" if args.output_dir else None
        )
        
        plot_pie_chart(
            summary, label,
            args.output_dir / f"{safe_label}_pie.{args.format}" if args.output_dir else None
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
        
        # Also generate individual breakdowns
        for label, summary in summaries:
            safe_label = label.replace("/", "_").replace(" ", "_")
            plot_operation_breakdown(
                summary, label,
                args.output_dir / f"{safe_label}_breakdown.{args.format}" if args.output_dir else None
            )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
