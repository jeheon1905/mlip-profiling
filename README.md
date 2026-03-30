# MLIP Profiling

Performance profiling tools for Machine Learning Interatomic Potential (MLIP) models.

## Supported Models

- eSEN
- MACE
- SevenNet
- NequIP (not yet)
- Allegro (not yet)

## Model Architecture Overview

### Comparison Summary

| | eSEN | MACE | SevenNet |
|---|---|---|---|
| **Graph Generation** | GPU (nvalchemiops) | CPU (matscipy) | CPU (ASE/numpy) |
| **Message Passing** | 4 layers (SO2Conv) | 2 layers (Interaction) | 5 layers (Convolution) |
| **Force Calculation** | autograd.grad | autograd.grad | autograd.grad |
| **Backends** | e3nn only | e3nn/cueq/oeq | e3nn/cueq/flash/oeq |

### eSEN Inference Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ eSEN::data_preparation                                              │
│  • AtomicData.from_ase() → atomicdata_list_to_batch() → .to(device) │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ eSEN::model_forward (predictor.predict)                             │
│  ├─ eSEN::data_to_device                                            │
│  ├─ forward (backbone)                                              │
│  │   ├─ generate_graph         ← GPU (nvalchemiops)                 │
│  │   ├─ obtain wigner / rotmat                                      │
│  │   ├─ atom/edge embedding                                         │
│  │   ├─ message passing 0-3    ← SO2Conv, edgewise, atomwise        │
│  │   ├─ balance_channels                                            │
│  │   └─ final_norm                                                  │
│  ├─ eSEN::compute_forces       ← autograd.grad (dominant cost)      │
│  └─ eSEN::process_outputs                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### MACE Inference Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ generate_graph                  ← CPU (matscipy)                    │
│  • config_from_atoms() → AtomicData.from_config() → DataLoader      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ forward (model)                                                     │
│  ├─ MACE::prepare_graph        (edge vectors, distances)            │
│  ├─ MACE::atomic_energies      (isolated atom energies)             │
│  ├─ MACE::embeddings           (node features)                      │
│  ├─ MACE::interaction_0 → MACE::product_0                           │
│  │   └─ ProductBasis, SymmetricContraction                          │
│  ├─ MACE::interaction_1 → MACE::product_1                           │
│  ├─ MACE::readouts             (energy prediction)                  │
│  └─ MACE::compute_forces       ← autograd.grad                      │
└─────────────────────────────────────────────────────────────────────┘
```

### SevenNet Inference Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ generate_graph                  ← CPU (ASE/numpy)                   │
│  • AtomGraphData.from_ase() → .to(device)                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ forward (model)                                                     │
│  ├─ SevenNet::edge_embedding           (radial basis)               │
│  ├─ SevenNet::onehot_to_feature_x      (atom features)              │
│  ├─ Layer 0-4 (repeated structure)                                  │
│  │   ├─ SevenNet::N_self_connection_intro                           │
│  │   ├─ SevenNet::N_self_interaction_1                              │
│  │   ├─ SevenNet::N_convolution        ← tensor product             │
│  │   ├─ SevenNet::N_self_interaction_2                              │
│  │   ├─ SevenNet::N_self_connection_outro                           │
│  │   └─ SevenNet::N_equivariant_gate                                │
│  ├─ SevenNet::reduce_output            (energy sum)                 │
│  └─ [Force: autograd.grad]             ← backward                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mlip-profiling/
├── README.md
├── profile_mlip.py              # Unified profiling script
├── profile_utils.py             # Shared profiling utilities
├── structure_builders.py        # Structure generation utilities
├── scripts/
│   ├── run_profiling.sh         # SLURM batch profiling script
│   └── generate_plots.py        # Plot generation (breakdown, pie, kernel, comparison)
├── structures/                  # Pre-generated atomic structures (.xyz)
├── results/                     # Profiling output (git-ignored)
└── packages/                    # Source codes of each MLIP model
    ├── fairchem-core/           # eSEN (modified for profiling)
    ├── mace/                    # MACE (modified for profiling)
    └── sevenn/                  # SevenNet
```

Each model's source code is stored under `./packages/` and minimally modified to enable detailed profiling.  
Modifications are marked with `[PROFILING]` comments.

## Source Code Versions

| Model | Commit | Original Repository |
|-------|--------|---------------------|
| MACE | [`667eee4`](https://github.com/ACEsuit/mace/commit/667eee4e58d23a38ff5a75122109ec2025809649) | https://github.com/ACEsuit/mace |
| fairchem-core (eSEN) | [`8f74b9e`](https://github.com/facebookresearch/fairchem/commit/8f74b9ed7c44e3b8036b693b8cb201c85f7d3eda) | https://github.com/facebookresearch/fairchem |
| SevenNet | [`95c811f`](https://github.com/MDIL-SNU/SevenNet/commit/95c811f56e64ec0a72f315e4b50ddca5adfa0667) | https://github.com/MDIL-SNU/SevenNet |

> **Note**: Source code has been minimally modified for profiling. All modifications are marked with `[PROFILING]` comments.


## Output Files

Profiling generates the following files per model configuration:

| File | Description |
|------|-------------|
| `summary.json` | Full results with model info, system info, and timing data |
| `{structure}.trace.json` | Chrome trace for Perfetto visualization |
| `timing_table.csv` | CSV with all metrics |
| `timing_table.md` | Markdown summary table |

The `summary.json` includes `system_info` recording the CPU, GPU, SLURM, and PyTorch/CUDA versions for reproducibility.

## Visualization & Analysis

### Plot Generation

Generate plots from profiling results:

```bash
# Generate all plots for a results directory
python scripts/generate_plots.py results/2026-03-30_201144_NVIDIA_A100-PCIE-40GB
```

Generated plots per model configuration:
- **Operation breakdown** (`_breakdown_leaf.png`) — Stacked bar chart of per-operation effective time
- **Pie chart** (`_pie.png`) — Proportional time distribution (small ops <3% merged to "Other")
- **Kernel breakdown** (`_kernels.png`) — GPU kernel categories (Gemm, Elementwise, etc.)
- **Model comparison** (`_comparison_latency.png`, `_comparison_speedup.png`) — Cross-model latency scaling and backend speedup

### Perfetto Trace Viewer

1. Run profiling script to generate Chrome trace file (`.json`)
2. Open https://ui.perfetto.dev
3. Click "Open trace file" and upload the `.json` file
4. Analyze the timeline visualization

## Batch Profiling

Run all model configurations in one SLURM job:

```bash
sbatch scripts/run_profiling.sh
```

This profiles all supported models (eSEN, MACE, SevenNet) with their respective backends across multiple structure sizes. Results are saved to `results/{date}_{gpu_type}/`.

## Robustness

- **CUDA OOM handling**: If a structure causes out-of-memory, profiling breaks and skips larger structures. Partial results are preserved via incremental `summary.json` saves.
- **Error recovery**: Non-OOM errors skip the current structure and continue with the next one.

---

## Structure Generation

Generate atomic structures for profiling using `structure_builders.py`.

### Benchmark Systems

| System | Type | PBC | Purpose |
|--------|------|-----|---------|
| Cu FCC | Periodic bulk | True | Bulk/solid benchmark |
| Water box | Molecular | True | Molecular/liquid benchmark |

### Generate Structures

```bash
# Cu FCC supercells (periodic, for bulk benchmarks)
python structure_builders.py \
    --fcc-by-cells \
    --fcc-cell-counts 2 3 4 5 6 7 8 9 10 \
    --fcc-cell-element Cu \
    --output-dir structures/

# Water boxes (periodic, for molecular benchmarks)
python structure_builders.py \
    --water-box \
    --water-molecules 10 50 100 250 500 1000 2000 \
    --output-dir structures/
```

### Atom Counts Reference

| FCC cells | Atoms (4×n³) |
|-----------|--------------|
| 2×2×2 | 32 |
| 3×3×3 | 108 |
| 4×4×4 | 256 |
| 5×5×5 | 500 |
| 6×6×6 | 864 |
| 7×7×7 | 1372 |
| 8×8×8 | 2048 |
| 9×9×9 | 2916 |
| 10×10×10 | 4000 |

| Water molecules | Atoms (3×n) |
|-----------------|-------------|
| 10 | 30 |
| 50 | 150 |
| 100 | 300 |
| 250 | 750 |
| 500 | 1500 |
| 1000 | 3000 |
| 2000 | 6000 |

Generated structures are saved as `.xyz` files and can be used with any profiling script.

---

## eSEN

### Environment Setup

```bash
conda create -n mlip-profiling-esen python=3.12 -y
conda activate mlip-profiling-esen

# Install PyTorch (CUDA 12.6)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install fairchem-core
SETUPTOOLS_SCM_PRETEND_VERSION=2.15.0 pip install -e ./packages/fairchem-core
```

### Run Profiling

```bash
# Profile Cu FCC
python profile_mlip.py \
    --model-type esen \
    --model-name esen-sm-conserving-all-omol \
    --device cuda \
    --structure-files structures/Cu_fcc*.xyz \
    --output-dir profile_traces_esen

# Profile water boxes
python profile_mlip.py \
    --model-type esen \
    --model-name esen-sm-conserving-all-omol \
    --device cuda \
    --structure-files structures/water_*.xyz \
    --output-dir profile_traces_esen

# With InferenceSettings options
python profile_mlip.py \
    --model-type esen \
    --model-name uma-s-1 \
    --tf32 \
    --compile \
    --device cuda \
    --structure-files structures/*.xyz
```

---

## MACE

### Source Code Modifications

MACE source code (`./packages/mace/`) has been modified to support detailed profiling.
All modifications are marked with `[PROFILING]` comments in the source code.

**New file: `mace/modules/profiling.py`**
- `set_profiling_enabled(bool)` - Enable/disable profiling globally
- `is_profiling_enabled()` - Check profiling status
- `record_function_if_enabled(name)` - Conditional record_function wrapper
- `ProfilerContext` - Context manager for standalone profiling

**Modified: `mace/modules/models.py`** (MACE.forward method)
```python
# Line 293: Graph preparation
with record_function_if_enabled("MACE::prepare_graph"):
    ...

# Line 316: Atomic energies computation
with record_function_if_enabled("MACE::atomic_energies"):
    ...

# Line 328: Node/edge embeddings
with record_function_if_enabled("MACE::embeddings"):
    ...

# Line 379, 393: Interaction and product blocks (per layer)
with record_function_if_enabled(f"MACE::interaction_{i}"):
    ...
with record_function_if_enabled(f"MACE::product_{i}"):
    ...

# Line 400: Readout layers
with record_function_if_enabled("MACE::readouts"):
    ...

# Line 416: Final output computation
with record_function_if_enabled("MACE::get_outputs"):
    ...
```

**Modified: `mace/modules/blocks.py`** (Block-level profiling)
```python
# Line 507: ProductBasisBlock
with record_function_if_enabled("MACE::ProductBasis"):
    ...

# Line 526: SymmetricContraction
with record_function_if_enabled("MACE::SymmetricContraction"):
    ...

# Line 807-823: InteractionBlock internals
with record_function_if_enabled("MACE::Interaction::forward"):
    with record_function_if_enabled("MACE::Interaction::skip_tp"):
        ...
    with record_function_if_enabled("MACE::Interaction::linear_up"):
        ...
    with record_function_if_enabled("MACE::Interaction::conv_weights"):
        ...
    with record_function_if_enabled("MACE::Interaction::message_passing"):
        ...
```

### Environment Setup

```bash
conda create -n mlip-profiling-mace python=3.10 -y
conda activate mlip-profiling-mace

# Install PyTorch (CUDA 12.6)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install MACE
pip install ./packages/mace
```

### Accelerator Backends (Optional)

MACE supports tensor product accelerators for improved performance:

```bash
# cuEquivariance (NVIDIA GPU acceleration) - Recommended for NVIDIA GPUs
# IMPORTANT: Both packages are required for actual GPU acceleration
pip install cuequivariance-torch cuequivariance-ops-torch-cu12  # For CUDA 12.x
# pip install cuequivariance-torch cuequivariance-ops-torch-cu13  # For CUDA 13.x

# OpenEquivariance (open-source acceleration) - experimental
pip install openequivariance
```

> **Warning**: Installing only `cuequivariance-torch` without `cuequivariance-ops-torch-cu*` will cause fallback to naive Python implementation, which is **8x slower** than e3nn.

| Backend | Flag | Description |
|---------|------|-------------|
| e3nn | `--backend e3nn` | Standard e3nn (default) |
| cuEquivariance | `--backend cueq` | NVIDIA GPU-accelerated tensor products |
| OpenEquivariance | `--backend oeq` | Open-source equivariant acceleration |

### Run Profiling

```bash
# Basic profiling
python profile_mlip.py \
    --model-type mace \
    --model-path /path/to/mace_model.model \
    --structure-files structures/*.xyz \
    --device cuda \
    --output-dir profile_traces_mace

# With cuEquivariance backend
python profile_mlip.py \
    --model-type mace \
    --model-path /path/to/mace_model.model \
    --backend cueq \
    --structure-files structures/*.xyz \
    --device cuda
```

Results are saved as Chrome trace format (`.json`) in the output directory.  
Open with https://ui.perfetto.dev or `chrome://tracing`.

### Analysis using Perfetto

...

---

## SevenNet

SevenNet is a scalable E(3)-equivariant GNN-based MLIP from KAIST (MDIL-SNU).

### Source Code Modifications

SevenNet source code (`./packages/sevenn/`) has been modified to support detailed profiling:

- **New file**: `sevenn/nn/profiling.py` - Profiling utilities
- **Modified**: `sevenn/nn/sequential.py` - Added profiling hooks to `AtomGraphSequential.forward()`

Traced operations (per layer):
- `SevenNet::edge_embedding` - Radial basis and spherical harmonics
- `SevenNet::onehot_idx_to_onehot` - One-hot encoding
- `SevenNet::onehot_to_feature_x` - Initial node features
- `SevenNet::{i}_self_connection_intro` - Self-connection intro
- `SevenNet::{i}_self_interaction_1` - Linear transformation 1
- `SevenNet::{i}_convolution` - **Main tensor product convolution**
- `SevenNet::{i}_self_interaction_2` - Linear transformation 2
- `SevenNet::{i}_self_connection_outro` - Self-connection outro
- `SevenNet::{i}_equivariant_gate` - Equivariant gate activation
- `SevenNet::reduce_input_to_hidden` - Reduce to hidden
- `SevenNet::reduce_hidden_to_energy` - Reduce to energy
- `SevenNet::rescale_atomic_energy` - Shift/scale
- `SevenNet::force_output` - Force computation (autograd)

### Environment Setup

```bash
conda create -n mlip-profiling-sevenn python=3.10 -y
conda activate mlip-profiling-sevenn

# Install PyTorch (CUDA 12.6)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install SevenNet
pip install ./packages/sevenn
```

### Accelerator Backends (Optional)

SevenNet supports multiple tensor product accelerators. Install any of the following as needed:

```bash
# cuEquivariance (NVIDIA GPU acceleration) - Recommended for NVIDIA GPUs
# IMPORTANT: Both packages are required for actual GPU acceleration
pip install cuequivariance-torch cuequivariance-ops-torch-cu12  # For CUDA 12.x
# pip install cuequivariance-torch cuequivariance-ops-torch-cu13  # For CUDA 13.x

# OpenEquivariance (open-source acceleration)
pip install openequivariance

# FlashTP (build from source)
git clone https://github.com/SNU-ARC/flashTP.git && cd flashTP
pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
```

> **Warning**: Installing only `cuequivariance-torch` without `cuequivariance-ops-torch-cu*` will cause fallback to naive Python implementation, which is **8x slower** than e3nn.

> **Note**: All backends can be installed simultaneously without conflicts. SevenNet checks availability at runtime. However, only **one backend can be active** at a time during profiling.

| Backend | Flag | Description |
|---------|------|-------------|
| e3nn | `--backend e3nn` | Standard e3nn (default) |
| cuEquivariance | `--backend cueq` | NVIDIA GPU-accelerated tensor products |
| FlashTP | `--backend flash` | Fast tensor product implementation |
| OpenEquivariance | `--backend oeq` | Open-source equivariant acceleration |

### Pretrained Models

Available pretrained models:
- `7net-0` - Base model
- `7net-omni` - Multi-modal universal potential
- `7net-mf-ompa` - Multi-fidelity model (requires `--modal` option)
- `7net-omat` - OMat24-trained model
- `7net-l3i5` - Higher angular momentum model

### Run Profiling

```bash
# Basic profiling with 7net-0
python profile_mlip.py \
    --model-type sevenn \
    --model-name 7net-0 \
    --structure-files structures/*.xyz \
    --device cuda \
    --output-dir profile_traces_sevenn

# With cuEquivariance backend
python profile_mlip.py \
    --model-type sevenn \
    --model-name 7net-0 \
    --backend cueq \
    --structure-files structures/*.xyz \
    --device cuda

# Multi-fidelity model (modal required)
python profile_mlip.py \
    --model-type sevenn \
    --model-name 7net-mf-ompa \
    --modal mpa \
    --structure-files structures/*.xyz \
    --device cuda

# Using a local checkpoint file
python profile_mlip.py \
    --model-type sevenn \
    --model-path /path/to/checkpoint.pth \
    --structure-files structures/*.xyz \
    --device cuda
```


