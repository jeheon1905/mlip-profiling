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
├── structure_builders.py        # Structure generation utilities
├── profile_esen.py              # eSEN profiling script
├── profile_mace.py              # MACE profiling script
└── packages/                    # Source codes of each MLIP model
    ├── fairchem-core/           # eSEN (modified for profiling)
    └── mace/                    # MACE (modified for profiling)
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
# Profile Cu FCC only
python profile_esen.py \
    --device cuda \
    --structure-files structures/Cu_fcc*.xyz \
    --output-dir profile_traces_esen

# Profile water boxes only
python profile_esen.py \
    --device cuda \
    --structure-files structures/water_*.xyz \
    --output-dir profile_traces_esen
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
    --structure-files structures/*.xyz \
    --device cuda \
    --output-dir profile_traces_mace
```

Results are saved as Chrome trace format (`.json`) in `profile_traces_mace/` directory.  
Open with https://ui.perfetto.dev or `chrome://tracing`.

### Analysis using Perfetto

...


