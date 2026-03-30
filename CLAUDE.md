# CLAUDE.md

This file provides context for Claude (AI assistant) when working with this codebase.

## User Preferences

- **Ask permission before**: `git commit`, job submission (`sbatch`, `srun` for batch jobs)
- **OK without asking**: file edits, running commands, reading files, searches

## Project Overview

**mlip-profiling** is a performance profiling toolkit for Machine Learning Interatomic Potential (MLIP) models. It provides detailed timing analysis using PyTorch Profiler with Chrome Trace export for visualization in Perfetto.

## Directory Structure

```
mlip-profiling/
├── profile_mlip.py        # Unified profiling script (ModelAdapter pattern)
├── profile_utils.py       # Shared profiling utilities
├── structure_builders.py  # Atomic structure generation
├── README.md
├── CLAUDE.md
├── scripts/
│   ├── run_profiling.sh       # SLURM batch profiling (all models)
│   └── generate_plots.py      # Plot generation (breakdown, pie, kernel, comparison)
├── structures/            # Pre-generated atomic structures (.xyz)
├── results/               # Profiling output (git-ignored)
└── packages/              # MLIP model source codes (modified for profiling)
    ├── fairchem-core/     # eSEN models (fairchem v2.15.0)
    ├── mace/              # MACE models
    └── sevenn/            # SevenNet models
```

## Supported Models

| Model | Adapter Class | Model Loading |
|-------|---------------|---------------|
| eSEN (fairchem) | `ESENAdapter` | `pretrained_mlip.get_predict_unit(model_name)` |
| MACE | `MACEAdapter` | `torch.load(model_path)` |
| SevenNet | `SevenNetAdapter` | `util.load_checkpoint(model_name_or_path)` |

## Key Components

### profile_mlip.py
- **ModelAdapter**: Abstract base class for model adapters
- **ESENAdapter**: fairchem/eSEN model adapter
- **MACEAdapter**: MACE model adapter (direct inference, no ASE calculator)
- **SevenNetAdapter**: SevenNet model adapter with backend options
- **run_profiling()**: Main profiling loop with PyTorch Profiler
- **get_system_info()**: Collects CPU/GPU/SLURM/PyTorch info for reproducibility
- **OOM handling**: `torch.cuda.OutOfMemoryError` → break (skip larger), general `Exception` → continue
- **Incremental save**: `summary_callback` saves `summary.json` after each structure

### profile_utils.py
- `synchronize(device)`: Multi-GPU barrier or CUDA synchronize
- `get_qps(inference_fn, ...)`: fairchem-style QPS measurement using timeit.repeat()
- `extract_operation_times_from_trace()`: Parse Chrome trace JSON
- `trace_handler()`: Save Chrome trace files

### structure_builders.py
- `get_fcc_crystal_by_num_cells()`: Generate FCC supercells
- `get_water_box()`: Generate water boxes with packmol
- `load_structures_from_files()`: Load structures from xyz/cif files

### scripts/generate_plots.py
- Generates operation breakdown, pie chart, kernel breakdown, and model comparison plots
- Uses **effective_time** metric: CPU time for CPU-bound ops, GPU time for GPU-bound ops
- `CPU_OPERATIONS` allowlist per model_type for explicit classification
- Merges small operations (<3%) into "Other" in pie charts

### scripts/run_profiling.sh
- SLURM batch script to profile all model configurations
- Result directory format: `results/{YYYY-MM-DD_HHMMSS}_{GPU_TYPE}/`
- Default structures: 108, 500, 1372, 2916 atoms (Cu FCC)

## Usage Patterns

```bash
# eSEN model
python profile_mlip.py --model-type esen --model-name esen-sm-conserving-all-omol \
    --structure-files structures/*.xyz --device cuda

# MACE model  
python profile_mlip.py --model-type mace --model-path model.pt \
    --structure-files structures/*.xyz --device cuda

# MACE with cuEquivariance backend
python profile_mlip.py --model-type mace --model-path model.pt --backend cueq \
    --structure-files structures/*.xyz --device cuda

# SevenNet model (pretrained)
python profile_mlip.py --model-type sevenn --model-name 7net-0 \
    --structure-files structures/*.xyz --device cuda

# SevenNet with cuEquivariance backend
python profile_mlip.py --model-type sevenn --model-name 7net-0 --backend cueq \
    --structure-files structures/*.xyz --device cuda

# SevenNet multi-fidelity model
python profile_mlip.py --model-type sevenn --model-name 7net-mf-ompa --modal mpa \
    --structure-files structures/*.xyz --device cuda
```

## Environment Setup

```bash
# eSEN environment
conda create -n mlip-profiling-esen python=3.12 -y
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
SETUPTOOLS_SCM_PRETEND_VERSION=2.15.0 pip install -e ./packages/fairchem-core

# MACE environment
conda create -n mlip-profiling-mace python=3.10 -y
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install ./packages/mace
# Optional accelerators:
# IMPORTANT: Both packages required for cuEquivariance GPU acceleration
# pip install cuequivariance-torch cuequivariance-ops-torch-cu12  # for --backend cueq (CUDA 12.x)
# pip install openequivariance      # for --backend oeq

# SevenNet environment
conda create -n mlip-profiling-sevenn python=3.10 -y
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install ./packages/sevenn
# Optional accelerators:
# IMPORTANT: Both packages required for cuEquivariance GPU acceleration
# pip install cuequivariance-torch cuequivariance-ops-torch-cu12  # for --backend cueq (CUDA 12.x)
# pip install openequivariance      # for --backend oeq
# FlashTP (build from source):
#   git clone https://github.com/SNU-ARC/flashTP.git && cd flashTP
#   pip install -r requirements.txt
#   CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
```

## Profiling Methodology

Following fairchem's `uma_speed_benchmark.py` methodology:
- **Profiler schedule**: wait=5, warmup=5, active=5 (default)
- **QPS measurement**: `timeit.repeat(number=10, repeat=5)`
- **ns/day metric**: `qps * 24 * 3600 / 1e6` (assumes 1 fs timestep)
- **Synchronization**: `torch.distributed.barrier()` for multi-GPU, `torch.cuda.synchronize()` for single-GPU

## Graph Generation

| Model | Graph Generation | Location |
|-------|-----------------|----------|
| eSEN | GPU (nvalchemiops) | Model forward() internal |
| MACE | CPU (matscipy) | Data preparation |
| SevenNet | CPU (ase/numpy) | Data preparation |

All models include graph generation in each inference for fair comparison.

## Accelerator Backends (MACE/SevenNet)

| Backend | Flag | Description | Required Package | MACE | SevenNet |
|---------|------|-------------|------------------|------|----------|
| e3nn | `--backend e3nn` | Standard e3nn (default) | Built-in | ✓ | ✓ |
| cuEquivariance | `--backend cueq` | NVIDIA GPU-accelerated tensor products | `cuequivariance-torch` + `cuequivariance-ops-torch-cu12` | ✓ | ✓ |
| FlashTP | `--backend flash` | Fast tensor product implementation | Build from source | ✗ | ✓ |
| OpenEquivariance | `--backend oeq` | Open-source equivariant acceleration | `openequivariance` | ✓ | ✓ |

> **Note**: Installing only `cuequivariance-torch` without `cuequivariance-ops-torch-cu*` causes fallback to naive Python implementation (8x slower).

## Adding New Models

1. Create a new adapter class inheriting from `ModelAdapter`
2. Implement required methods:
   - `load(**kwargs)`
   - `run_inference(atoms: Atoms)`
   - `tracked_operations` (property)
   - `model_info` (property)
3. Register in `get_adapter()` function
4. Add CLI options in `main()` if needed

## Code Conventions

- Use `record_function()` for profiling tags
- Prefix model-specific operations: `MACE::`, `eSEN::`, `SevenNet::`, etc.
- Follow fairchem naming for shared operations: `forward`, `generate_graph`

## Output Files

- `{name}.trace.json`: Chrome trace for Perfetto
- `summary.json`: Full results with model info, system info, and timing
- `timing_table.csv`: CSV with all metrics
- `timing_table.md`: Markdown summary table

`summary.json` now includes a `system_info` field with CPU model, GPU model/memory, SLURM allocation, and PyTorch/CUDA versions.
