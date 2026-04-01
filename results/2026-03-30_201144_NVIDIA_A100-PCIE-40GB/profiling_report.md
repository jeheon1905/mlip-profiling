# MLIP Profiling Report

Performance profiling of Machine Learning Interatomic Potential (MLIP) models using PyTorch Profiler.

**Environment**

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A100-PCIE-40GB |
| CPU | AMD EPYC 7252 8-Core (1 core allocated via SLURM) |
| PyTorch | 2.8.0+cu126 |
| CUDA | 12.6 |

**Tested Models**

| Model | Variant | Backend(s) |
|-------|---------|------------|
| eSEN | esen-sm-conserving-all-omol | e3nn |
| MACE | mace-mp-small | e3nn, cuEquivariance |
| SevenNet | 7net-0 | e3nn, cuEquivariance |

**Benchmark System**: Cu FCC supercells — 108, 500, 1,372, 2,916 atoms

---

## Table of Contents

1. [Profiling Methodology](#1-profiling-methodology)
2. [Tested MLIP Models](#2-tested-mlip-models)
3. [Operation Breakdown Details](#3-operation-breakdown-details)
4. [Profiling Results & Analysis](#4-profiling-results--analysis)

---

## 1. Profiling Methodology

### 1.1 PyTorch Profiler Overview

PyTorch Profiler traces both CPU and GPU activity during model execution. It captures:

- **CPU operators**: Python-level function calls, ATen operators, autograd operations
- **CUDA kernels**: GPU kernel launches, memory operations, synchronization
- **Correlation IDs**: Links between CPU-side launches and GPU-side kernel execution

The profiler outputs a **Chrome Trace** (JSON), which can be visualized in [Perfetto](https://ui.perfetto.dev) as a timeline.

### 1.2 Profiler Schedule

We follow fairchem's `uma_speed_benchmark.py` methodology:

```
|←  wait (5)  →|←  warmup (5)  →|←  active (5)  →|
                                  ← traces recorded →
```

- **Wait** (5 steps): Execute without profiling overhead — stabilize GPU state
- **Warmup** (5 steps): Profiler active but results discarded — allow JIT warmup
- **Active** (5 steps): Full profiling with trace recording — collect measurements

### 1.3 Instrumentation with `record_function`

Each model's source code is instrumented with `torch.profiler.record_function()` tags to label logical operations:

```python
with record_function("MACE::compute_forces"):
    forces = torch.autograd.grad(energy, positions, ...)
```

These tags create **user annotations** in the trace, which serve as the primary unit of analysis. Tags follow a `{Model}::{operation}` naming convention (e.g., `MACE::interaction_0`, `SevenNet::3_convolution`).

### 1.4 Latency Measurement

Two independent latency measurements are taken:

| Method | Purpose | Detail |
|--------|---------|--------|
| **Profiler latencies** | Per-step timing during traced execution | Manual `time.perf_counter()` around each step |
| **timeit latencies** | Clean measurement without profiling overhead | `timeit.repeat(number=10, repeat=5)` after profiler completes |

The **timeit** measurement is reported as the primary latency, as it is free from profiler instrumentation overhead. QPS (queries per second) and ns/day are derived from timeit.

### 1.5 Effective Time Metric

The profiler reports both `cpu_time` and `gpu_time` per operation. To produce a single meaningful metric, we use **effective time**:

- **CPU-bound operations** → use `cpu_time` (e.g., `generate_graph` on CPU, `data_preparation`)
- **GPU-bound operations** → use `gpu_time` (e.g., `compute_forces`, `convolution`)

Classification is done via an explicit **CPU_OPERATIONS allowlist** per model type, rather than heuristics:

| Model | CPU Operations |
|-------|----------------|
| eSEN | `data_preparation`, `data_to_device` |
| MACE | `generate_graph` |
| SevenNet | `generate_graph` |

**Important**: The sum of leaf effective times (leaf effective total) is **not** a wall-clock measurement. It represents the aggregate device-occupancy time across CPU and GPU. This value may differ from the timeit wall-clock latency for two reasons:

1. **Profiler overhead** — the profiler's instrumentation inflates individual operation times. This effect is largest for small structures where operation durations are short relative to overhead (e.g., eSEN 108 atoms: leaf effective total is 1.35× timeit, but at 1,372 atoms it converges to 1.01×).
2. **CPU-GPU overlap** — CPU and GPU operations can execute concurrently (e.g., CPU prepares the next kernel launch while GPU is computing). Simple summation double-counts this overlap, or conversely, untracked idle/synchronization gaps may cause the sum to undercount wall-clock time.

At large system sizes (≥1,372 atoms), the leaf effective total closely approximates timeit wall-clock latency for all three models (within ~1–9%).

Percentages reported in operation breakdowns represent **relative proportions within the traced total**, not fractions of wall-clock time. These percentages are meaningful for comparing the relative cost of different operations within a single inference.

---

## 2. Tested MLIP Models

### 2.1 Model Overview

All three models are E(3)-equivariant graph neural network potentials that predict energy and forces from atomic structures. Their inference pipelines share a common pattern:

```
Graph Construction → Node/Edge Embedding → Message Passing Layers → Energy Readout → Force (autograd)
```

However, they differ significantly in architecture, graph construction strategy, and available backends.

| | eSEN | MACE | SevenNet |
|---|---|---|---|
| **Architecture** | SO(2)-equivariant convolution | Higher-order message passing with symmetric contraction | E(3)-equivariant with tensor product convolution |
| **Parameters** | 6.3M | 3.8M | 0.8M |
| **Cutoff Radius** | 6.0 Å | 6.0 Å | 5.0 Å |
| **Graph Generation** | GPU (nvalchemiops) | CPU (matscipy) | CPU (ASE/numpy) |
| **Message Passing Layers** | 4 (SO2Conv) | 2 (Interaction + SymmetricContraction) | 5 (Convolution + Gate) |
| **Force Computation** | autograd.grad | autograd.grad | autograd.grad |
| **Backends** | e3nn only | e3nn / cuEquivariance / OpenEquivariance | e3nn / cuEquivariance / FlashTP / OpenEquivariance |

### 2.2 eSEN (fairchem)

eSEN (Equivariant Scalable Energy Network) from Meta's fairchem uses SO(2)-equivariant convolutions (eSCN-style). Key characteristics:

- **Graph construction on GPU**: Uses `nvalchemiops` for neighbor list construction, avoiding CPU-GPU data transfer
- **4 message passing layers**: Each layer contains SO2Conv + edgewise + atomwise sub-operations
- **Wigner/rotation matrices**: Pre-computed rotational features (`obtain wigner / rotmat original`)
- **Dominant cost**: Force computation via `autograd.grad` accounts for ~53% of traced time at 500 atoms. This single call traverses the entire computation graph in reverse, so its cost is the sum of backward gradients for all forward operations (SO2Conv, edgewise, atomwise, etc.). The profiler records only generic autograd ops (`MmBackward0`, `MulBackward0`, etc.) inside this call, making it impossible to attribute backward cost to individual forward operations.

### 2.3 MACE

MACE (Multi Atomic Cluster Expansion) uses higher-order equivariant message passing with a unique SymmetricContraction that captures many-body interactions:

- **Graph construction on CPU**: `matscipy`-based neighbor list, executed on CPU before transfer to GPU
- **2 interaction layers**: Each layer includes linear_up → conv_weights → message_passing → skip_tp, followed by ProductBasis → SymmetricContraction
- **SymmetricContraction**: A distinctive operation absent in other models — compresses high-order tensor products to invariant features
- **Force computation**: `compute_forces` (`autograd.grad`) is ~43% of traced time. As with all models, the backward pass is a single monolithic traversal of the full computation graph — the profiler cannot separate, e.g., gradients originating from `message_passing` vs. `SymmetricContraction`. Graph generation on CPU is also a significant portion (~31–36%).

### 2.4 SevenNet

SevenNet (Scalable E(3)-Equivariant Network) from SNU uses standard tensor product convolutions with self-interaction layers:

- **Graph construction on CPU**: ASE/numpy-based neighbor list
- **5 convolution layers**: Each includes self_connection_intro → self_interaction_1 → convolution → self_interaction_2 → self_connection_outro → equivariant_gate
- **Force computation dominance**: `force_output` (`autograd.grad`) accounts for ~63% of traced time at 500 atoms. Like the other models, this backward pass encompasses gradients from all 5 convolution layers, gates, and self-interactions combined, without per-operation attribution.
- **Best cuEquivariance scaling**: Shows the largest speedup from cuEquivariance (up to ~4x at 2,916 atoms)

---

## 3. Operation Breakdown Details

### 3.1 Leaf vs. Wrapper Operations

PyTorch Profiler records operations in a nested call tree. We distinguish:

- **Wrapper operations**: Contain child operations (e.g., `forward` wraps all model operations, `message passing 0` wraps `SO2Conv` + `edgewise` + `atomwise`)
- **Leaf operations**: Do not contain tracked children — represent the actual compute

**Leaf-only analysis** avoids double-counting. For example, in eSEN:

```
eSEN::model_forward          ← wrapper (excluded)
  ├─ forward                 ← wrapper (excluded)
  │   ├─ generate_graph      ← LEAF ✓
  │   ├─ SO2Conv             ← LEAF ✓ (aggregated across layers)
  │   └─ ...
  ├─ eSEN::compute_forces    ← LEAF ✓
  └─ eSEN::process_outputs   ← LEAF ✓
```

The **breakdown plots** and **pie charts** use only leaf operations to show where time is actually spent.

### 3.2 Kernel-Level Classification

Beyond user-annotated operations, we analyze the underlying **CUDA kernels** dispatched by each operation. The profiler's `gpu_user_annotation` field maps each kernel to its parent `record_function` tag.

Kernels are categorized by type:

| Category | Examples | Typical Source |
|----------|----------|---------------|
| **Gemm** | `ampere_sgemm`, `cutlass::Kernel` | Linear layers, matrix multiplications |
| **Elementwise** | `vectorized_elementwise_kernel` | Activations, additions, scaling |
| **Reduction** | `reduce_kernel`, `welford_reduce` | Sum, mean, normalization |
| **Scatter/Gather** | `scatter_kernel`, `index_select` | Message aggregation, graph operations |
| **Copy/Transpose** | `copy_kernel`, `transpose_kernel` | Data layout transformations |
| **Memcpy/Memset** | `Memcpy HtoD`, `Memset` | CPU↔GPU data transfer |
| **Idle/Overhead** | (gap between kernels) | Launch latency, synchronization |

The **kernel breakdown plots** show the proportion of GPU time in each category, revealing hardware utilization patterns.

---

## 4. Profiling Results & Analysis

### 4.1 Overall Latency Comparison

**Latency (ms) — timeit measurement**

| Model | Backend | 108 atoms | 500 atoms | 1,372 atoms | 2,916 atoms |
|-------|---------|----------:|----------:|------------:|------------:|
| eSEN | e3nn | 58.65 | 184.96 | 476.69 | OOM |
| MACE | e3nn | 32.64 | 47.45 | 123.33 | 259.93 |
| MACE | cueq | 36.54 | 39.46 | 71.92 | 134.33 |
| SevenNet | e3nn | 53.86 | 69.82 | 139.07 | 289.12 |
| SevenNet | cueq | 38.81 | 46.80 | 52.24 | 72.69 |

![Latency vs. atom count for all model configurations](plots/comparison_latency.png)

**Key observations**:
- MACE (e3nn) is the fastest at small sizes (108 atoms), but scaling is moderate
- SevenNet (cueq) shows the best large-system scaling — only 72.7 ms at 2,916 atoms
- eSEN has the steepest latency scaling and runs out of memory at 2,916 atoms on A100-40GB

### 4.2 Operation Breakdown

#### eSEN (e3nn) — 500 atoms

| Operation | Effective Time (ms) | % of total |
|-----------|--------------------:|--:|
| compute_forces | 101.66 | 53.3% |
| SO2Conv | 62.12 | 32.6% |
| generate_graph | 10.81 | 5.7% |
| obtain rotmat wigner original | 6.06 | 3.2% |
| Others (edge embedding, edgewise, data prep, ...) | ~10.1 | ~5.3% |

![eSEN e3nn 500 atoms pie chart](plots/esen_e3nn_esen-sm_500atoms_pie.png)
![eSEN e3nn 500 atoms kernel breakdown](plots/esen_e3nn_esen-sm_500atoms_kernels.png)

At 500 atoms, force computation (`autograd.grad`) and SO2Conv together account for 86% of traced time. The kernel breakdown reveals that `compute_forces` is dominated by **Gemm** (matrix multiplication) and **Elementwise** kernels from the backward pass, while SO2Conv's GPU time is split between **Scatter/Gather** operations (message aggregation) and **Gemm** (linear layers within convolutions). Graph generation on GPU remains nearly constant (~11 ms) regardless of system size.

#### MACE (e3nn) — 500 atoms

| Operation | Effective Time (ms) | % of total |
|-----------|--------------------:|--:|
| compute_forces | 20.52 | 42.0% |
| generate_graph | 15.02 (CPU) | 30.8% |
| message_passing | 5.30 | 10.9% |
| SymmetricContraction | 3.77 | 7.7% |
| embeddings | 1.78 | 3.7% |
| Others (conv_weights, skip_tp, readouts, ...) | ~2.43 | ~5.0% |

![MACE e3nn 500 atoms pie chart](plots/mace_e3nn_mace-mp-small_500atoms_pie.png)
![MACE e3nn 500 atoms kernel breakdown](plots/mace_e3nn_mace-mp-small_500atoms_kernels.png)

MACE's profile at 500 atoms is split between GPU compute (force + equivariant operations) and CPU graph generation. The kernel breakdown shows that `compute_forces` is primarily **Gemm** kernels, while `message_passing` uses **Scatter/Gather** for neighborhood aggregation and **Gemm** for linear projections. `SymmetricContraction` is heavily **Elementwise** — reflecting the many-body contraction operations. `generate_graph` (matscipy, CPU) shows minimal GPU kernel activity in the kernel plot, as its compute runs entirely on CPU.

#### SevenNet (e3nn) — 500 atoms

| Operation | Effective Time (ms) | % of total |
|-----------|--------------------:|--:|
| force_output | 53.04 | 63.1% |
| convolution (layers 0–4) | 13.72 total | 16.3% |
| generate_graph | 8.54 (CPU) | 10.2% |
| equivariant_gate (layers 0–4) | 3.33 total | 4.0% |
| self_interaction_1 (layers 0–4) | 1.74 total | 2.1% |
| Others (self_connection, edge_embedding, ...) | ~3.66 | ~4.4% |

![SevenNet e3nn 500 atoms pie chart](plots/sevenn_e3nn_7net-0_500atoms_pie.png)
![SevenNet e3nn 500 atoms kernel breakdown](plots/sevenn_e3nn_7net-0_500atoms_kernels.png)

SevenNet's force computation dominates at 63% of traced time. The kernel breakdown shows `force_output` is almost entirely **Gemm** and **Elementwise** kernels from the autograd backward pass. The 5 `convolution` layers are split between **Scatter/Gather** (tensor product neighborhood aggregation) and **Gemm** (linear transformations). Graph generation (10.2%, CPU) is growing as a fraction compared to 108 atoms (2.8%), which is expected since CPU graph construction scales linearly with system size while GPU operations scale more favorably.

### 4.3 Graph Generation: Implementation Differences

The three models use fundamentally different strategies for graph (neighbor list) construction:

| | eSEN | MACE | SevenNet |
|---|---|---|---|
| **Library** | nvalchemiops | matscipy | ASE (numpy) |
| **Device** | GPU | CPU | CPU |
| **Scaling** | Nearly constant (~11 ms) | Superlinear (14→80 ms) | Linear (2→35 ms) |
| **Fraction of total** | ~14–2% (decreasing) | ~36–31% (significant) | ~3–12% (growing) |

**Detailed scaling (cpu_time / gpu_time in ms)**:

| Atoms | eSEN | MACE | SevenNet |
|------:|-----:|-----:|---------:|
| 108 | 10.9 / 10.8 | 13.6 / 1.9 | 2.1 / 0.3 |
| 500 | 10.9 / 10.8 | 15.0 / 2.1 | 8.5 / 0.3 |
| 1,372 | 11.2 / 11.2 | 37.4 / 3.0 | 14.8 / 0.5 |
| 2,916 | OOM | 80.0 / 4.5 | 34.5 / 0.7 |

**Analysis**:

- **eSEN** performs graph generation on GPU using nvalchemiops. The time is nearly constant (~11 ms) regardless of system size, as the GPU parallelism absorbs the workload. eSEN hits OOM at 2,916 atoms on A100-40GB.

- **MACE** uses matscipy's `neighbour_list()` on CPU. This is the most expensive graph generation, scaling superlinearly from 13.6 ms (108 atoms) to 80.0 ms (2,916 atoms). Because this runs on CPU, it appears in `cpu_time` while the GPU sits idle — making it a clear bottleneck at all sizes. At 108 atoms, graph generation accounts for 36% of traced time.

- **SevenNet** uses ASE's neighbor list on CPU. It is the lightest implementation at small sizes (2.1 ms at 108 atoms) but scales linearly to 34.5 ms at 2,916 atoms. The fraction of traced time grows from ~3% to ~12% as system size increases.

**Implication**: For MACE, moving graph generation to GPU (e.g., using nvalchemiops or similar) could yield significant speedups, especially for larger systems where it still consumes ~31% of traced time.

### 4.4 cuEquivariance (CuEq) Acceleration

cuEquivariance replaces the standard e3nn tensor product operations with NVIDIA-optimized CUDA kernels. Since eSEN does not support cuEquivariance, this comparison is between MACE and SevenNet.

**Speedup (e3nn time / cueq time)**:

| Model | 108 atoms | 500 atoms | 1,372 atoms | 2,916 atoms |
|-------|----------:|----------:|------------:|------------:|
| MACE | 0.89x | 1.20x | 1.71x | 1.94x |
| SevenNet | 1.39x | 1.49x | 2.66x | 3.98x |

![CuEq speedup vs. atom count](plots/comparison_speedup.png)

**Key findings**:

1. **SevenNet benefits far more from cuEquivariance than MACE**. At 2,916 atoms, SevenNet (cueq) achieves a 3.98x speedup over e3nn, while MACE (cueq) only achieves 1.94x.

2. **MACE shows a regression at 108 atoms** (0.89x). The cueq kernel launch overhead exceeds the compute savings for small systems. cuEquivariance only becomes beneficial for MACE above ~300 atoms.

3. **Speedup grows with system size** for both models. This is expected because larger systems have more tensor product operations to parallelize, amortizing the kernel launch overhead.

4. **Why the difference?** cuEquivariance accelerates both forward tensor products and their backward gradients (inside `compute_forces`/`force_output`). SevenNet's 5 tensor product convolution layers make its backward pass heavily dependent on these operations — `force_output` speedup reaches 6.5x at 2,916 atoms. MACE benefits too (`compute_forces` speedup: 5.2x at 2,916 atoms), but its bottleneck includes `generate_graph` (CPU, unaffected by cueq), which limits the overall end-to-end speedup per Amdahl's law.

**Latency at 2,916 atoms** — the practical large-system comparison:

| Model | e3nn | cueq | Speedup |
|-------|-----:|-----:|--------:|
| MACE | 259.9 ms | 134.3 ms | 1.94x |
| SevenNet | 289.1 ms | 72.7 ms | 3.98x |

SevenNet (cueq) is 1.85x faster than MACE (cueq) at 2,916 atoms, despite being slower with e3nn. This illustrates how backend choice can invert model performance rankings.

---

## Appendix

### A. Profiling Configuration

```
Profiler schedule: wait=5, warmup=5, active=5
timeit settings: number=10, repeat=5
Structure: Cu FCC (periodic, cutoff=varies by model)
Device: CUDA (single GPU)
Synchronization: torch.cuda.synchronize() after each step
```

### B. Generated Plot Index

Per model configuration (5 configs × 3–4 sizes):
- `{model}_breakdown.png` — Stacked bar chart of leaf operations (effective time)
- `{model}_pie.png` — Proportional distribution (operations <3% merged to "Other")
- `{model}_kernels.png` — GPU kernel category breakdown

Cross-model:
- `comparison_latency.png` — Latency vs. atom count (all configurations)
- `comparison_speedup.png` — CuEq speedup vs. atom count (MACE and SevenNet)

### C. Reproduction

```bash
# Run profiling (SLURM)
sbatch scripts/run_profiling.sh

# Generate plots
python scripts/generate_plots.py results/{result_dir}
```
