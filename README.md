# MLIP Profiling

Performance profiling tools for Machine Learning Interatomic Potential (MLIP) models.

## Compared Models

- NequIP
- MACE
- Allegro
- SevenNet
- eSEN

## Project Structure

```
mlip-profiling/
├── README.md
├── packages/                    # Source codes of each MLIP model
│   ├── fairchem-core/          # eSEN (modified for profiling)
│   └── mace/                   # MACE (modified for profiling)
├── profile_esen.py             # eSEN profiling script
└── profile_mace.py             # MACE profiling script
```

Each model's source code is stored under `./packages/` and minimally modified to enable detailed profiling.  
Modifications are marked with `[PROFILING]` comments.

## Source Code Versions

| Model | Commit | Original Repository |
|-------|--------|---------------------|
| MACE | [`667eee4`](https://github.com/ACEsuit/mace/commit/667eee4e58d23a38ff5a75122109ec2025809649) | https://github.com/ACEsuit/mace |
| fairchem-core (eSEN) | [`8f74b9e`](https://github.com/facebookresearch/fairchem/commit/8f74b9ed7c44e3b8036b693b8cb201c85f7d3eda) | https://github.com/facebookresearch/fairchem |

## Analysis with Perfetto

1. Run profiling script to generate Chrome trace file (`.json`)
2. Open https://ui.perfetto.dev
3. Click "Open trace file" and upload the `.json` file
4. Analyze the timeline visualization

---

## eSEN

### Environment Setup

```bash
conda create -n mlip-profiling-esen python=3.12 -y
conda activate mlip-profiling-esen

# Install PyTorch (CUDA 12.6)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install fairchem-core
pip install -e ./packages/fairchem-core
```

### Run Profiling

```bash
python profile_with_trace.py \
    --device cuda \
    --cluster-element Al \
    --cluster-sizes 64 128 256 512 1024 2048 4096 8192 \
    --output-dir profile_traces_clusters
```

---

## MACE

### Source Code Modifications

MACE source code (`./packages/mace/`) has been modified to support detailed profiling:

- **New file**: `mace/modules/profiling.py` - Profiling utilities
- **Modified**: `mace/modules/models.py` - Added profiling hooks to `MACE.forward()`
- **Modified**: `mace/modules/blocks.py` - Added profiling hooks to `InteractionBlock` and `ProductBasisBlock`

Traced operations:
- `MACE::prepare_graph`
- `MACE::atomic_energies`
- `MACE::embeddings`
- `MACE::interaction_0`, `MACE::interaction_1`, ...
- `MACE::product_0`, `MACE::product_1`, ...
- `MACE::readouts`
- `MACE::get_outputs`
- `MACE::Interaction::skip_tp`, `linear_up`, `conv_weights`, `message_passing`
- `MACE::SymmetricContraction`

### Environment Setup

```bash
conda create -n mlip-profiling-mace python=3.10 -y
conda activate mlip-profiling-mace

# Install PyTorch (CUDA 12.6)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Install MACE
pip install ./packages/mace
```

### Run Profiling

```bash
python profile_mace.py \
    --model-path /path/to/mace_model.model \
    --structure-path /path/to/structures.xyz \
    --device cuda \
    --output-dir profile_traces_mace \
    --warmup-steps 5 \
    --profile-steps 10
```

Results are saved as Chrome trace format (`.json`) in `profile_traces_mace/` directory.  
Open with https://ui.perfetto.dev or `chrome://tracing`.

### Analysis using Perfetto

...


