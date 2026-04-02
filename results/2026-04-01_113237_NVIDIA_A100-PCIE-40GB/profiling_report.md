# MLIP Profiling Report

Performance profiling of Machine Learning Interatomic Potential (MLIP) models using PyTorch Profiler.
Source code and profiling scripts are available at [github.com/jeheon1905/mlip-profiling](https://github.com/jeheon1905/mlip-profiling).

**Environment**

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A100-PCIE-40GB |
| CPU | AMD EPYC 7252 8-Core (1 core allocated via SLURM) |
| PyTorch | 2.8.0+cu126 |
| CUDA | 12.6 |

**Tested Models**

| Model | Variant | Parameters | Backend(s) |
|-------|---------|-----------|------------|
| eSEN | esen-sm-conserving-all-omol | 6.3M | [e3nn](https://e3nn.org) |
| MACE | mace-mp-0 medium | 4.7M | e3nn, [cuEquivariance](https://github.com/NVIDIA/cuEquivariance) |
| SevenNet | 7net-0 | 0.8M | e3nn, cuEquivariance |

**Benchmark System**: Cu FCC supercells — 108, 500, 1,372, 2,916 atoms

---

## Table of Contents

1. [Profiling Methodology](#profiling-methodology)
2. [Tested MLIP Models](#tested-mlip-models)
3. [Operation Breakdown Details](#operation-breakdown-details)
4. [Profiling Results & Analysis](#profiling-results-analysis)

---

## 1. Profiling Methodology

### 1.1 PyTorch Profiler Overview

[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) traces both CPU and GPU activity during model execution. It captures:

- **CPU operators**: Python-level function calls, ATen operators, autograd operations
- **CUDA kernels**: GPU kernel launches, memory operations, synchronization
- **Correlation IDs**: Links between CPU-side launches and GPU-side kernel execution

The profiler outputs a **Chrome Trace** (JSON), which can be visualized in [Perfetto](https://ui.perfetto.dev) as a timeline.

### 1.2 Profiler Schedule

We follow fairchem's `uma_speed_benchmark.py` methodology:

```
|←  wait (5)  →|←  warmup (5)  →|←    active (5)   →|
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

1. **Profiler overhead** — the profiler's instrumentation inflates individual operation times. This effect is largest for small structures where operation durations are short relative to overhead (e.g., eSEN 108 atoms: leaf effective total is 1.42× timeit, but at 1,372 atoms it converges to 1.01×).
2. **CPU-GPU overlap** — CPU and GPU operations can execute concurrently (e.g., CPU prepares the next kernel launch while GPU is computing). Simple summation double-counts this overlap, or conversely, untracked idle/synchronization gaps may cause the sum to undercount wall-clock time.

At large system sizes (≥1,372 atoms), the leaf effective total closely approximates timeit wall-clock latency for all three models (within ~0–2%).

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
| **Variants** | esen-sm-conserving-all-omol | mace-mp-0 medium | 7net-0 |
| **Parameters** | 6.3M | 4.7M | 0.8M |
| **Cutoff Radius** | 6.0 Å | 6.0 Å | 5.0 Å |
| **Graph Generation** | GPU (nvalchemiops) | CPU (matscipy) | CPU (ASE/numpy) |
| **Message Passing Layers** | 4 (SO2Conv) | 2 (Interaction + SymmetricContraction) | 5 (Convolution + Gate) |
| **Force Computation** | autograd.grad | autograd.grad | autograd.grad |
| **Backends** | e3nn only | e3nn / cuEquivariance / OpenEquivariance | e3nn / cuEquivariance / FlashTP / OpenEquivariance |

### 2.2 eSEN (fairchem)

eSEN (Equivariant Scalable Energy Network) from Meta's [fairchem](https://github.com/FAIR-Chem/fairchem) uses SO(2)-equivariant convolutions (eSCN-style). Key characteristics:

- **Graph construction on GPU**: Uses `nvalchemiops` for neighbor list construction, avoiding CPU-GPU data transfer
- **4 message passing layers**: Each layer contains SO2Conv + edgewise + atomwise sub-operations
- **Wigner/rotation matrices**: Pre-computed rotational features (`obtain wigner / rotmat original`)
- **Dominant cost**: Force computation via `autograd.grad` accounts for ~53% of traced time at 500 atoms. This single call traverses the entire computation graph in reverse, so its cost is the sum of backward gradients for all forward operations (SO2Conv, edgewise, atomwise, etc.). The profiler records only generic autograd ops (`MmBackward0`, `MulBackward0`, etc.) inside this call, making it impossible to attribute backward cost to individual forward operations.

### 2.3 MACE

[MACE](https://github.com/ACEsuit/mace) (Multi Atomic Cluster Expansion) uses higher-order equivariant message passing with a unique SymmetricContraction that captures many-body interactions:

- **Graph construction on CPU**: [matscipy](https://github.com/libAtoms/matscipy)-based neighbor list, executed on CPU before transfer to GPU
- **2 interaction layers**: Each layer includes linear_up → conv_weights → message_passing → skip_tp, followed by ProductBasis → SymmetricContraction
- **SymmetricContraction**: A distinctive operation absent in other models — compresses high-order tensor products to invariant features
- **L_max=1**: The medium variant uses L_max=1 (vs. L_max=0 for small), enabling non-trivial equivariant tensor products that cuEquivariance can accelerate
- **Force computation**: `compute_forces` (`autograd.grad`) is ~56% of traced time at 500 atoms. As with all models, the backward pass is a single monolithic traversal of the full computation graph — the profiler cannot separate, e.g., gradients originating from `message_passing` vs. `SymmetricContraction`. Graph generation on CPU is also significant (~14%).

### 2.4 SevenNet

[SevenNet](https://github.com/MDIL-SNU/SevenNet) (Scalable E(3)-Equivariant Network) from SNU uses standard tensor product convolutions with self-interaction layers:

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
  └─ eSEN::compute_forces    ← LEAF ✓
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
| eSEN | e3nn | 58.26 | 184.79 | 476.50 | OOM |
| MACE | e3nn | 40.18 | 106.47 | 286.57 | OOM |
| MACE | cueq | 45.42 | 48.67 | 76.01 | 145.00 |
| SevenNet | e3nn | 53.69 | 67.69 | 139.17 | 288.51 |
| SevenNet | cueq | 38.30 | 46.33 | 51.98 | 72.14 |

**Throughput (ns/day)**

| Model | Backend | 108 atoms | 500 atoms | 1,372 atoms | 2,916 atoms |
|-------|---------|----------:|----------:|------------:|------------:|
| eSEN | e3nn | 1.48 | 0.47 | 0.18 | OOM |
| MACE | e3nn | 2.15 | 0.81 | 0.30 | OOM |
| MACE | cueq | 1.90 | 1.78 | 1.14 | 0.60 |
| SevenNet | e3nn | 1.61 | 1.28 | 0.62 | 0.30 |
| SevenNet | cueq | 2.26 | 1.86 | 1.66 | 1.20 |

![Latency vs. atom count for all model configurations](plots/comparison_latency.png)

**Key observations**:
- MACE (e3nn) is the fastest at small sizes (40.2 ms at 108 atoms), but its latency scales steeply and hits OOM at 2,916 atoms (the medium variant with L_max=1 is larger than the small variant)
- SevenNet (cueq) delivers the best large-system performance — only 72.1 ms at 2,916 atoms
- eSEN has the most aggressive latency scaling and the highest absolute latency at all sizes, hitting OOM at 2,916 atoms on A100-40GB
- Both MACE (e3nn) and eSEN run out of memory at 2,916 atoms; cuEquivariance allows MACE to handle this size (145.0 ms)

### 4.2 Operation Breakdown

#### eSEN (e3nn) — 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|--:|
| compute_forces | 101.52 | 53.4% |
| SO2Conv | 62.08 | 32.6% |
| generate_graph | 10.73 | 5.6% |
| obtain rotmat wigner original | 6.06 | 3.2% |
| Others (edge embedding, edgewise, data prep, ...) | ~9.8 | ~5.2% |

![eSEN e3nn 500 atoms pie chart](plots/esen_e3nn_esen-sm_500atoms_pie.png)

![eSEN e3nn 500 atoms kernel breakdown](plots/esen_e3nn_esen-sm_500atoms_kernels.png)

At 500 atoms, force computation (`autograd.grad`) and SO2Conv together account for 86% of leaf traced time. The kernel breakdown reveals that `compute_forces` is dominated by **Gemm** (matrix multiplication) and **Elementwise** kernels from the backward pass, while SO2Conv's GPU time is split between **Scatter/Gather** operations (message aggregation) and **Gemm** (linear layers within convolutions). Graph generation on GPU remains nearly constant (~11 ms) regardless of system size.

**Scaling**: As system size grows from 108 to 1,372 atoms, message passing (SO2Conv) time grows from 14.3 ms to 171.8 ms (12× for 12.7× atoms), indicating near-linear scaling. Force computation (backward pass) scales similarly from 41.2 ms to 272.3 ms. Graph generation on GPU remains essentially flat at ~11 ms across all sizes.

#### MACE (e3nn) — 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|--:|
| compute_forces | 60.03 | 55.8% |
| message_passing | 17.18 | 16.0% |
| generate_graph | 14.94 (CPU) | 13.9% |
| SymmetricContraction | 10.83 | 10.1% |
| Others (embeddings, conv_weights, skip_tp, ...) | ~4.6 | ~4.3% |

![MACE e3nn 500 atoms pie chart](plots/mace_e3nn_mace-mp-medium_500atoms_pie.png)

![MACE e3nn 500 atoms kernel breakdown](plots/mace_e3nn_mace-mp-medium_500atoms_kernels.png)

MACE's profile at 500 atoms is split between GPU compute (force + equivariant operations) and CPU graph generation. The kernel breakdown shows that `compute_forces` is primarily **Gemm** kernels, while `message_passing` uses **Scatter/Gather** for neighborhood aggregation and **Gemm** for linear projections. `SymmetricContraction` is heavily **Elementwise** — reflecting the many-body contraction operations. `generate_graph` (matscipy, CPU) shows minimal GPU kernel activity in the kernel plot, as its compute runs entirely on CPU.

**Scaling**: At 108 atoms, graph generation accounts for 41% of leaf total (24.1 ms CPU), though this value appears inflated by one-time initialization overhead (the cueq run measured 14.0 ms for the same operation). `compute_forces` is 38% (22.3 ms). By 1,372 atoms, `compute_forces` grows to 57% (162.6 ms) and `message_passing` to 16% (47.2 ms), while `generate_graph` grows to 38.7 ms CPU (13% of leaf total). The model OOMs at 2,916 atoms with e3nn.

#### SevenNet (e3nn) — 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|--:|
| force_output | 53.32 | 63.3% |
| convolution (layers 0–4) | 13.78 total | 16.3% |
| generate_graph | 8.49 (CPU) | 10.1% |
| equivariant_gate (layers 0–4) | 3.33 total | 4.0% |
| Others (self_interaction, edge_embedding, ...) | ~5.4 | ~6.4% |

![SevenNet e3nn 500 atoms pie chart](plots/sevenn_e3nn_7net-0_500atoms_pie.png)

![SevenNet e3nn 500 atoms kernel breakdown](plots/sevenn_e3nn_7net-0_500atoms_kernels.png)

SevenNet's force computation dominates at 63% of leaf traced time. The kernel breakdown shows `force_output` is almost entirely **Gemm** and **Elementwise** kernels from the autograd backward pass. The 5 `convolution` layers are split between **Scatter/Gather** (tensor product neighborhood aggregation) and **Gemm** (linear transformations). Graph generation (10.1%, CPU) grows as a fraction compared to 108 atoms (3%), as CPU graph construction scales linearly with system size while GPU operations scale more favorably.

**Scaling**: `force_output` remains a stable ~63–69% of leaf traced time across all sizes (52.5 ms at 108 → 191.3 ms at 2,916). Convolution layers grow from 11.9 ms to 59.8 ms (5× for 27× atoms), showing sub-linear GPU scaling. Graph generation scales linearly on CPU from 2.2 ms (108) to 35.0 ms (2,916), growing from 3% to 12% of leaf total.

### 4.3 Graph Generation: Implementation Differences

The three models use fundamentally different strategies for graph (neighbor list) construction.

> **Note**: Graph generation ms values for MACE and SevenNet in this section use measurements from the cueq run.

| | eSEN | MACE | SevenNet |
|---|---|---|---|
| **Library** | nvalchemiops | matscipy | ASE (numpy) |
| **Device** | GPU | CPU | CPU |
| **Scaling** | Nearly constant (~11 ms) | Linear (14→16→38→80 ms) | Linear (2–9→14→34 ms) |
| **Fraction of leaf total** | ~13–2% (decreasing) | ~24–13% (significant) | ~3–12% (growing) |

**Detailed scaling (cpu_time / gpu_time in ms)**:

| Atoms | eSEN | MACE | SevenNet |
|------:|-----:|-----:|---------:|
| 108 | 10.9 / 10.9 | 14.0 / 2.0 | 2.1 / 0.3 |
| 500 | 10.8 / 10.7 | 15.6 / 2.2 | 8.4 / 0.3 |
| 1,372 | 10.9 / 10.9 | 38.0 / 3.1 | 14.2 / 0.4 |
| 2,916 | OOM | 80.4 / 4.5 | 33.7 / 0.7 |

**Analysis**:

- **eSEN** performs graph generation on GPU using nvalchemiops. The time is nearly constant (~11 ms) regardless of system size, as the GPU parallelism absorbs the workload. eSEN hits OOM at 2,916 atoms on A100-40GB.

- **MACE** uses matscipy's `neighbour_list()` on CPU. The scaling is approximately linear: 14 ms (108 atoms), 16 ms (500), 38 ms (1,372), and 80 ms (2,916 atoms). Because this runs on CPU with only 1 SLURM core, the GPU sits idle during graph generation — making it a significant bottleneck (~24% of leaf traced time at 108 atoms).

- **SevenNet** uses ASE's neighbor list on CPU. It is the lightest implementation at small sizes (2.1 ms at 108 atoms) but scales linearly to 33.7 ms at 2,916 atoms. The fraction of traced time grows from ~3% to ~12% as system size increases.

**Implication**: For MACE, moving graph generation to GPU (e.g., using nvalchemiops or similar) could yield significant speedups, especially since CPU graph generation accounts for up to ~24% of leaf traced time at small sizes and grows to 80 ms at 2,916 atoms.

### 4.4 cuEquivariance (CuEq) Acceleration

cuEquivariance replaces the standard e3nn tensor product operations with NVIDIA-optimized CUDA kernels. Since eSEN does not support cuEquivariance, this comparison is between MACE and SevenNet.

**Speedup (e3nn time / cueq time)**:

| Model | 108 atoms | 500 atoms | 1,372 atoms | 2,916 atoms |
|-------|----------:|----------:|------------:|------------:|
| MACE | 0.88x | 2.19x | 3.77x | N/A (e3nn OOM) |
| SevenNet | 1.40x | 1.46x | 2.68x | 4.00x |

![CuEq speedup vs. atom count](plots/comparison_speedup.png)

**Key findings**:

1. **Both models show increasing speedup with system size**. This is expected because larger systems have more tensor product operations to parallelize, amortizing the kernel launch overhead.

2. **MACE shows a regression at 108 atoms** (0.88x). The cueq kernel launch overhead exceeds the compute savings for small systems. cuEquivariance becomes beneficial for MACE between 108 and 500 atoms, where it reaches 2.19x.

3. **MACE achieves steep speedup growth**: From 0.88x at 108 atoms to 3.77x at 1,372 atoms, MACE benefits dramatically as size increases. The medium variant (L_max=1) has non-trivial equivariant tensor products that cuEquivariance accelerates effectively. At 2,916 atoms, e3nn OOMs while cueq succeeds (145.0 ms), demonstrating that cuEquivariance also reduces memory consumption.

4. **Why the difference in scaling?** cuEquivariance accelerates both forward tensor products and their backward gradients (inside `compute_forces`/`force_output`). At 1,372 atoms, MACE's `compute_forces` speedup is 7.94x and `message_passing` reaches 16.80x. SevenNet's `force_output` speedup is 3.22x and convolution layers see 4.8–5.0x. However, MACE's `generate_graph` (CPU, unaffected by cueq) limits overall end-to-end speedup per Amdahl's law.

**CuEq operation breakdown at 500 atoms**:

#### MACE (cueq) — 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|--:|
| compute_forces | 20.05 | 41.7% |
| generate_graph | 15.63 (CPU) | 32.5% |
| SymmetricContraction | 3.47 | 7.2% |
| message_passing | 1.62 | 3.4% |
| Others (embeddings, conv_weights, skip_tp, ...) | ~7.3 | ~15.2% |

![MACE cueq 500 atoms pie chart](plots/mace_cueq_mace-mp-medium_500atoms_pie.png)

![MACE cueq 500 atoms kernel breakdown](plots/mace_cueq_mace-mp-medium_500atoms_kernels.png)

Compared to e3nn, the most striking change is `generate_graph` rising from 13.9% to **32.5%** of leaf total — not because graph generation slowed down, but because GPU operations became much faster under cueq. `compute_forces` shrank from 60.0 ms to 20.1 ms (3.0x), and `message_passing` from 17.2 ms to 1.6 ms (10.6x). The kernel breakdown shows the emergence of **cuEq TP** kernels (`segmented_polynomial`) replacing much of the Gemm/Elementwise work in the tensor product operations. CPU-bound `generate_graph` is now the single largest bottleneck by fraction.

#### SevenNet (cueq) — 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|--:|
| force_output | 28.36 | 52.5% |
| generate_graph | 8.35 (CPU) | 15.5% |
| convolution (layers 0–4) | 8.01 total | 14.8% |
| equivariant_gate (layers 0–4) | 2.97 total | 5.5% |
| Others (self_interaction, edge_embedding, ...) | ~6.3 | ~11.7% |

![SevenNet cueq 500 atoms pie chart](plots/sevenn_cueq_7net-0_500atoms_pie.png)

![SevenNet cueq 500 atoms kernel breakdown](plots/sevenn_cueq_7net-0_500atoms_kernels.png)

SevenNet shows a similar pattern: `force_output` drops from 53.3 ms to 28.4 ms (1.88x), while `generate_graph` rises from 10.1% to **15.5%** of leaf total. Convolution layers shrink from 13.8 ms to 8.0 ms (1.7x at 500 atoms — the larger speedups appear at bigger system sizes). The kernel breakdown reveals **cuEq TP** kernels within the convolution and gate operations, partially replacing the Scatter/Gather and Gemm patterns seen in e3nn.

**Per-operation speedup at 1,372 atoms**:

| Operation | MACE e3nn (ms) | MACE cueq (ms) | Speedup |
|-----------|-----:|-----:|--------:|
| compute_forces | 162.61 | 20.48 | 7.94x |
| message_passing | 47.25 | 2.81 | 16.80x |
| SymmetricContraction | 29.44 | 3.74 | 7.87x |
| generate_graph (CPU) | 38.69 | 38.03 | 1.02x |

| Operation | SevenNet e3nn (ms) | SevenNet cueq (ms) | Speedup |
|-----------|-----:|-----:|--------:|
| force_output | 93.08 | 28.91 | 3.22x |
| convolution (layers 1–3, L>0) | 8.5 avg | 1.76 avg | 4.84x |
| generate_graph (CPU) | 14.95 | 14.24 | 1.05x |

At 2,916 atoms, SevenNet's `force_output` speedup grows to 6.61x (191.3 → 28.9 ms) and convolution layers reach 7.5x, resulting in an overall 4.00x end-to-end speedup.

**Latency at largest shared size (1,372 atoms)**:

| Model | e3nn | cueq | Speedup |
|-------|-----:|-----:|--------:|
| MACE | 286.57 ms | 76.01 ms | 3.77x |
| SevenNet | 139.17 ms | 51.98 ms | 2.68x |

SevenNet (cueq) is 1.46x faster than MACE (cueq) at 1,372 atoms. At 2,916 atoms where only cueq results are available, SevenNet (cueq) at 72.1 ms is 2.01x faster than MACE (cueq) at 145.0 ms.

**GPU Pipeline Starvation in cueq kernel breakdown**: The cueq kernel plots show a larger Idle/Overhead (gray) component compared to e3nn — not only in proportion but also in **absolute time**. Trace-level gap analysis of the backward pass at 500 atoms:

| Metric | MACE e3nn | MACE cueq | SevenNet e3nn | SevenNet cueq |
|--------|----------:|----------:|--------------:|--------------:|
| Kernel total time | 59.3 ms | 4.9 ms | 35.4 ms | 3.6 ms |
| **Inter-kernel gap total** | **0.7 ms** | **15.0 ms** | **17.3 ms** | **24.1 ms** |
| Avg kernel duration | 104 μs | 13 μs | 28 μs | 6 μs |
| Avg inter-kernel gap | 1.2 μs | 41 μs | 14 μs | 39 μs |

The root cause is **GPU pipeline starvation**. With e3nn, each kernel runs 100–240 μs, long enough to hide the CPU's autograd dispatch overhead behind GPU execution. With cueq, kernels shrink to 6–36 μs, but the CPU still walks the same Python autograd graph (~40 μs per dispatch) — so the GPU idles waiting for the next launch. The gap distribution confirms this: e3nn gaps are 1–10 μs (hardware launch latency), while cueq gaps shift to 10–100 μs (CPU autograd overhead).

At 1,372 atoms, the gap fraction improves for both models — MACE from 75% to 35%, SevenNet from 87% to 78% — as larger systems increase per-kernel compute, partially re-hiding CPU overhead behind GPU execution.

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

All scripts referenced below are in the [mlip-profiling](https://github.com/jeheon1905/mlip-profiling) repository. Clone the repo and follow the environment setup in `README.md` before running.

```bash
# Run profiling (SLURM)
sbatch scripts/run_profiling.sh

# Generate plots (all structure sizes)
python scripts/generate_plots.py results/{result_dir}
```

### D. Document Export

```bash
# PDF (from this directory)
pandoc profiling_report.md -o profiling_report.pdf \
  --pdf-engine=xelatex \
  -V mainfont="FreeSerif" \
  -V sansfont="FreeSans" \
  -V monofont="DejaVu Sans Mono" \
  -V geometry:"margin=2cm" \
  -V fontsize=10pt \
  --highlight-style=tango \
  -H ../../pandoc/latex-header.tex

# HTML (from this directory)
pandoc profiling_report.md -o profiling_report.html \
  --standalone \
  --self-contained \
  --css=../../pandoc/style.css
```
