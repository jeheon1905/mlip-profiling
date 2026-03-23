# CLAUDE.md

This file provides context for Claude (AI assistant) when working with this codebase.

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
└── packages/              # MLIP model source codes (modified for profiling)
    ├── fairchem-core/     # eSEN models (fairchem v2.15.0)
    └── mace/              # MACE models
```

## Supported Models

| Model | Adapter Class | Model Loading |
|-------|---------------|---------------|
| eSEN (fairchem) | `ESENAdapter` | `pretrained_mlip.get_predict_unit(model_name)` |
| MACE | `MACEAdapter` | `torch.load(model_path)` |

## Key Components

### profile_mlip.py
- **ModelAdapter**: Abstract base class for model adapters
- **ESENAdapter**: fairchem/eSEN model adapter
- **MACEAdapter**: MACE model adapter (direct inference, no ASE calculator)
- **run_profiling()**: Main profiling loop with PyTorch Profiler

### profile_utils.py
- `synchronize(device)`: Multi-GPU barrier or CUDA synchronize
- `get_qps(inference_fn, ...)`: fairchem-style QPS measurement using timeit.repeat()
- `extract_operation_times_from_trace()`: Parse Chrome trace JSON
- `trace_handler()`: Save Chrome trace files

### structure_builders.py
- `get_fcc_crystal_by_num_cells()`: Generate FCC supercells
- `get_water_box()`: Generate water boxes with packmol
- `load_structures_from_files()`: Load structures from xyz/cif files

## Usage Patterns

```bash
# eSEN model
python profile_mlip.py --model-type esen --model-name esen-sm-conserving-all-omol \
    --structure-files structures/*.xyz --device cuda

# MACE model  
python profile_mlip.py --model-type mace --model-path model.pt \
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

Both include graph generation in each inference for fair comparison.

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
- Prefix model-specific operations: `MACE::`, `eSEN::`, etc.
- Follow fairchem naming for shared operations: `forward`, `generate_graph`

## Output Files

- `{name}.trace.json`: Chrome trace for Perfetto
- `summary.json`: Full results with model info and timing
- `timing_table.csv`: CSV with all metrics
- `timing_table.md`: Markdown summary table
