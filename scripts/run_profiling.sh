#!/bin/bash
#SBATCH --job-name=mlip-profiling
#SBATCH --partition=GPU_A100_2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/profiling_%j.out
#SBATCH --error=logs/profiling_%j.err

# =============================================================================
# MLIP Profiling Job Script
# =============================================================================
# Usage:
#   sbatch scripts/run_profiling.sh
#
# Environment variables (optional overrides):
#   MACE_MODEL_PATH  - Path to MACE model file
#   STRUCTURE_FILES  - Space-separated list of structure files
#
# Output structure:
#   results/
#   └── YYYY-MM-DD_HHMMSS_<gpu_type>/
#       ├── esen/
#       │   └── e3nn/
#       │       └── <model_id>/
#       │           ├── summary.json
#       │           ├── *.trace.json
#       │           └── run.log
#       ├── mace/
#       │   ├── e3nn/
#       │   │   └── <model_id>/
#       │   └── cueq/
#       │       └── <model_id>/
#       └── sevenn/
#           ├── e3nn/
#           │   └── <model_id>/
#           └── cueq/
#               └── <model_id>/
# =============================================================================

# =============================================================================
# Configuration (auto-detect paths, allow environment variable overrides)
# =============================================================================

# Detect project directory
# - In SLURM: use SLURM_SUBMIT_DIR (directory where sbatch was run)
# - Otherwise: use script location
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
fi

# Change to project directory immediately
cd "${PROJECT_DIR}" || { echo "Failed to cd to ${PROJECT_DIR}"; exit 1; }

set -e  # Exit on error (after cd to project dir)

# =============================================================================
# Structure files configuration
# =============================================================================
# Default: test multiple sizes for scaling analysis
# Override with STRUCTURE_FILES environment variable for custom selection
#
# Available structures:
#   Cu FCC:  32, 108, 256, 500, 864, 1372, 2048, 2916, 4000 atoms
#   Water:   30, 150, 300, 750, 1500, 3000, 6000 atoms
# =============================================================================

if [[ -n "${STRUCTURE_FILES}" ]]; then
    # Use custom structure files from environment variable
    STRUCTURE_ARRAY=(${STRUCTURE_FILES})
else
    # Default: representative sizes for scaling analysis
    STRUCTURE_ARRAY=(
        "${PROJECT_DIR}/structures/Cu_fcc_3x3x3_108atoms.xyz"
        "${PROJECT_DIR}/structures/Cu_fcc_5x5x5_500atoms.xyz"
        "${PROJECT_DIR}/structures/Cu_fcc_7x7x7_1372atoms.xyz"
        "${PROJECT_DIR}/structures/Cu_fcc_9x9x9_2916atoms.xyz"
    )
fi

# Date+time and GPU type for results directory
DATE=$(date +%Y-%m-%d_%H%M%S)
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' || echo "unknown_gpu")
RESULT_BASE="${PROJECT_DIR}/results/${DATE}_${GPU_TYPE}"

# Initialize conda if not already done (for non-interactive shells like SLURM batch jobs)
if [[ "$(type -t conda)" != "function" ]]; then
    CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"
    if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        echo "Error: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
        exit 1
    fi
fi

# =============================================================================
# Models and backends to test (use environment variables for custom paths)
# =============================================================================

declare -A ESEN_MODELS=(
    ["esen-sm"]="esen-sm-conserving-all-omol"
)

declare -A MACE_MODELS=(
    ["mace-mp-medium"]="${MACE_MODEL_PATH:-${HOME}/Models/mace/mace-mp-0-medium.model}"
)
MACE_BACKENDS=("e3nn" "cueq")

SEVENN_MODELS=("7net-0")
SEVENN_BACKENDS=("e3nn" "cueq")

# =============================================================================
# Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_esen() {
    local model_name=$1
    local model_id=$2
    local output_dir="${RESULT_BASE}/esen/e3nn/${model_id}"
    
    log "Running eSEN: ${model_id} (${#STRUCTURE_ARRAY[@]} structures)"
    mkdir -p "${output_dir}"
    
    conda activate mlip-profiling-esen
    python "${PROJECT_DIR}/profile_mlip.py" \
        --model-type esen \
        --model-name "${model_name}" \
        --structure-files "${STRUCTURE_ARRAY[@]}" \
        --device cuda \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/run.log"
}

run_mace() {
    local model_path=$1
    local model_id=$2
    local backend=$3
    local output_dir="${RESULT_BASE}/mace/${backend}/${model_id}"
    
    log "Running MACE: ${model_id} (${backend}, ${#STRUCTURE_ARRAY[@]} structures)"
    mkdir -p "${output_dir}"
    
    conda activate mlip-profiling-mace
    python "${PROJECT_DIR}/profile_mlip.py" \
        --model-type mace \
        --model-path "${model_path}" \
        --backend "${backend}" \
        --structure-files "${STRUCTURE_ARRAY[@]}" \
        --device cuda \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/run.log"
}

run_sevenn() {
    local model_name=$1
    local backend=$2
    local output_dir="${RESULT_BASE}/sevenn/${backend}/${model_name}"
    
    log "Running SevenNet: ${model_name} (${backend}, ${#STRUCTURE_ARRAY[@]} structures)"
    mkdir -p "${output_dir}"
    
    conda activate mlip-profiling-sevenn
    python "${PROJECT_DIR}/profile_mlip.py" \
        --model-type sevenn \
        --model-name "${model_name}" \
        --backend "${backend}" \
        --structure-files "${STRUCTURE_ARRAY[@]}" \
        --device cuda \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/run.log"
}

generate_summary() {
    log "Generating summary report..."
    
    python3 << EOF
import json
import os
from pathlib import Path

result_dir = Path("${RESULT_BASE}")
summary = {"gpu": "${GPU_TYPE}", "date": "${DATE}", "results": {}}

for model_dir in result_dir.iterdir():
    if not model_dir.is_dir():
        continue
    model_name = model_dir.name
    summary["results"][model_name] = {}
    
    for backend_dir in model_dir.iterdir():
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name
        summary["results"][model_name][backend] = {}
        
        for run_dir in backend_dir.iterdir():
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                key = list(data["results"].keys())[0]
                r = data["results"][key]
                summary["results"][model_name][backend][run_dir.name] = {
                    "latency_ms": r["timeit_mean_ms"],
                    "latency_std_ms": r.get("timeit_std_ms", 0),
                    "qps": r["qps"],
                    "ns_per_day": r["ns_per_day"],
                }

# Save summary
with open(result_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print table
print("\n" + "=" * 70)
print(f"Profiling Results: {summary['gpu']} ({summary['date']})")
print("=" * 70)
print(f"{'Model':<15} {'Backend':<10} {'Latency (ms)':<18} {'QPS':<10} {'ns/day':<10}")
print("-" * 70)

for model, backends in summary["results"].items():
    for backend, runs in backends.items():
        for run_name, metrics in runs.items():
            latency = f"{metrics['latency_ms']:.2f} ± {metrics['latency_std_ms']:.2f}"
            print(f"{model:<15} {backend:<10} {latency:<18} {metrics['qps']:<10.1f} {metrics['ns_per_day']:<10.2f}")

print("=" * 70)
EOF
}

# =============================================================================
# Main
# =============================================================================

# Already in PROJECT_DIR from script initialization
mkdir -p logs "${RESULT_BASE}"

log "Starting MLIP profiling on ${GPU_TYPE}"
log "Results will be saved to: ${RESULT_BASE}"
log "Testing ${#STRUCTURE_ARRAY[@]} structures:"
for struct in "${STRUCTURE_ARRAY[@]}"; do
    log "  - $(basename "${struct}")"
done

# Run eSEN models
for model_id in "${!ESEN_MODELS[@]}"; do
    run_esen "${ESEN_MODELS[$model_id]}" "${model_id}" || log "Failed: eSEN ${model_id}"
done

# Run MACE models
for model_id in "${!MACE_MODELS[@]}"; do
    for backend in "${MACE_BACKENDS[@]}"; do
        run_mace "${MACE_MODELS[$model_id]}" "${model_id}" "${backend}" || log "Failed: MACE ${model_id} ${backend}"
    done
done

# Run SevenNet models
for model_name in "${SEVENN_MODELS[@]}"; do
    for backend in "${SEVENN_BACKENDS[@]}"; do
        run_sevenn "${model_name}" "${backend}" || log "Failed: SevenNet ${model_name} ${backend}"
    done
done

# Generate summary
generate_summary

log "Profiling complete!"
log "Results saved to: ${RESULT_BASE}"
