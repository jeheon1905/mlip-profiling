---
title: "MLIP Profiling: eSEN vs MACE vs SevenNet on A100"
author: Jeheon Woo — NVIDIA A100-PCIE-40GB
notes: |
  Profiling three E(3)-equivariant MLIP models on a single A100-PCIE-40GB.
  Covers per-model inference bottlenecks, operation breakdown analysis,
  cuEquivariance acceleration, and the GPU pipeline starvation mechanism.
  Source code: github.com/jeheon1905/mlip-profiling
---

## Table of Contents

- **Setup & Methodology**: hardware, model overview, profiler schedule, effective time metric, kernel classification
- **Latency Results**: ms and throughput tables, latency and speedup scaling charts
- **Operation Breakdown**: per-model analysis at 500 atoms (eSEN, MACE e3nn, SevenNet e3nn)
- **cuEquivariance Acceleration**: speedup table, per-model cueq analysis, per-op speedup
- **Graph Generation & GPU Starvation**: implementation comparison, kernel timeline, gap analysis

::: notes
Five sections. Setup → Latency → per-model breakdown → cuEq acceleration → pipeline starvation.
All latency values are timeit measurements (number=10, repeat=5), free from profiler instrumentation overhead.
Operation breakdown percentages are relative to the leaf-effective total, not wall-clock time.
:::

## cuEquivariance delivers up to 4× speedup and prevents OOM at 2,916 atoms

- **SevenNet cueq**: best large-system throughput — 72.1 ms at 2,916 atoms (1.20 ns/day)
- **MACE cueq**: 3.77× faster at 1,372 atoms; enables 2,916-atom run (145 ms) where e3nn OOMs
- **eSEN**: highest latency at all sizes (58–477 ms), no cueq support, OOMs at 2,916 atoms
- MACE e3nn fastest at small sizes (40.2 ms at 108 atoms) but scales steeply; OOMs at 2,916 atoms
- Backend choice (e3nn vs cueq) matters as much as model architecture choice

::: notes
Headline finding: cuEquivariance is not optional for production-scale MLIP inference on A100-40GB.
SevenNet cueq delivers 2.01× better throughput than MACE cueq at 2,916 atoms.
MACE regresses at 108 atoms (0.88×) — cueq kernel launch overhead exceeds compute savings at small sizes.
cuEquivariance also reduces MACE's memory footprint, enabling 2,916-atom inference that OOMs with e3nn.
:::

## Benchmark environment: A100-PCIE-40GB with PyTorch 2.8

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA A100-PCIE-40GB |
| **CPU** | AMD EPYC 7252 8-Core (1 core allocated via SLURM) |
| **PyTorch** | 2.8.0+cu126 |
| **CUDA** | 12.6 |
| **Benchmark system** | Cu FCC supercells — 108, 500, 1,372, 2,916 atoms |

::: notes
Single A100 with only 1 SLURM CPU core — critical context for CPU-bound operations like graph generation.
The 40 GB VRAM limit explains why eSEN and MACE e3nn run out of memory at 2,916 atoms.
Cu FCC supercells are periodic crystals scaled by repeating a unit cell.
:::

## All three models are E(3)-equivariant GNN potentials with different architectures

| | **eSEN** | **MACE** | **SevenNet** |
|---|---|---|---|
| **Variant** | esen-sm-conserving-all-omol | mace-mp-0 medium | 7net-0 |
| **Parameters** | 6.3 M | 4.7 M | 0.8 M |
| **Cutoff radius** | 6.0 Å | 6.0 Å | 5.0 Å |
| **Graph generation** | GPU (nvalchemiops) | CPU (matscipy) | CPU (ASE/numpy) |
| **Message passing** | 4 SO2Conv layers | 2 Interaction + SymContr | 5 Conv + Gate layers |
| **Backends** | e3nn only | e3nn / cuEquivariance | e3nn / cuEquivariance |

::: notes
Common pipeline: Graph Construction → Node/Edge Embedding → Message Passing → Energy Readout → Force (autograd).
eSEN uses SO(2)-equivariant convolutions. MACE adds SymmetricContraction for many-body interactions (absent in others).
Key architectural split: eSEN builds the neighbor graph on GPU; MACE and SevenNet build it on CPU.
Force computation for all models: a single autograd.grad call traversing the full computation graph.
:::

## Profiling methodology: schedule, measurement, and instrumentation

| Phase | Steps | Profiler state | Purpose |
|-------|------:|---------------|---------|
| **Wait** | 1–5 | Off | Execute without overhead — stabilize GPU state |
| **Warmup** | 6–10 | On (discard) | Allow JIT compilation and CUDA kernel caching |
| **Active** | 11–15 | On (record) | Collect Chrome trace and per-step timing |
| **timeit (post)** | — | Off | `timeit.repeat(number=10, repeat=5)` — clean latency |

::: notes
Two independent measurements: PyTorch Profiler traces for operation-level breakdown, timeit for primary wall-clock latency.
timeit runs after the profiler completes — no instrumentation overhead. QPS and ns/day derive from timeit.
Operations instrumented with torch.profiler.record_function() using Model::operation naming convention.
Profiler overhead is significant at small sizes: eSEN 108 atoms leaf-effective-total is 1.42× timeit, converges to 1.01× at 1,372 atoms.
:::

## Effective time metric and CPU operations allowlist per model

| Model | CPU-bound (use cpu_time) | GPU-bound (use gpu_time) |
|-------|--------------------------|--------------------------|
| **eSEN** | `data_preparation`, `data_to_device` | `SO2Conv`, `compute_forces`, all others |
| **MACE** | `generate_graph` | `message_passing`, `SymmetricContraction`, `compute_forces`, all others |
| **SevenNet** | `generate_graph` | `convolution`, `equivariant_gate`, `force_output`, all others |

::: notes
Effective time = cpu_time for CPU-bound ops, gpu_time for GPU-bound ops.
Classification via explicit allowlist per model — no heuristics.
The leaf-effective total is NOT wall-clock time: profiler overhead and CPU-GPU overlap cause divergence.
At ≥1,372 atoms all three models converge to within 0–2% of timeit latency.
Percentages in breakdowns = relative proportions within the traced total, not fractions of wall-clock time.
:::

## CUDA kernel categories used in breakdown analysis

| Category | Examples | Typical source |
|----------|----------|---------------|
| **Gemm** | `ampere_sgemm`, `cutlass::Kernel` | Linear layers, matrix multiplications |
| **Elementwise** | `vectorized_elementwise_kernel` | Activations, additions, scaling |
| **Reduction** | `reduce_kernel`, `welford_reduce` | Sum, mean, normalization |
| **Scatter/Gather** | `scatter_kernel`, `index_select` | Message aggregation, graph operations |
| **Copy/Transpose** | `copy_kernel`, `transpose_kernel` | Data layout transformations |
| **Memcpy/Memset** | `Memcpy HtoD`, `Memset` | CPU↔GPU data transfer |
| **Idle/Overhead** | (gap between kernels) | Launch latency, synchronization overhead |

::: notes
Classification is based on the profiler's gpu_user_annotation field linking each kernel to its parent record_function tag.
Gemm dominates forward and backward passes (linear layers, tensor products).
Scatter/Gather is characteristic of message-passing aggregation (neighborhood summation).
Idle/Overhead grows dramatically with cuEquivariance — this is the GPU pipeline starvation signal.
:::

## Latency (ms) and throughput (ns/day) both confirm cueq's large-system advantage

| Config | 108 atoms | 500 atoms | 1,372 atoms | 2,916 atoms |
|--------|----------:|----------:|------------:|------------:|
| **eSEN e3nn** | 58.3 ms | 184.8 ms | 476.5 ms | OOM |
| **MACE e3nn** | 40.2 ms | 106.5 ms | 286.6 ms | OOM |
| **MACE cueq** | 45.4 ms | 48.7 ms | 76.0 ms | 145.0 ms |
| **SevenNet e3nn** | 53.7 ms | 67.7 ms | 139.2 ms | 288.5 ms |
| **SevenNet cueq** | 38.3 ms | 46.3 ms | 52.0 ms | 72.1 ms |

| Config | 108 atoms | 500 atoms | 1,372 atoms | 2,916 atoms |
|--------|----------:|----------:|------------:|------------:|
| **eSEN e3nn** | 1.48 ns/day | 0.47 ns/day | 0.18 ns/day | OOM |
| **MACE e3nn** | 2.15 ns/day | 0.81 ns/day | 0.30 ns/day | OOM |
| **MACE cueq** | 1.90 ns/day | 1.78 ns/day | 1.14 ns/day | 0.60 ns/day |
| **SevenNet e3nn** | 1.61 ns/day | 1.28 ns/day | 0.62 ns/day | 0.30 ns/day |
| **SevenNet cueq** | 2.26 ns/day | 1.86 ns/day | 1.66 ns/day | 1.20 ns/day |

::: notes
Top table: timeit latency (number=10, repeat=5). Bottom table: ns/day = QPS × 24 × 3600 / 1e6, assumes 1 fs timestep.
MACE e3nn fastest at 108 atoms (40.2 ms) but 7.1× slower by 1,372 atoms — steep O(N^k) scaling.
SevenNet cueq shows near-flat scaling: only 1.9× slower from 108 to 2,916 atoms (27× more atoms).
SevenNet cueq at 2,916 atoms: 1.20 ns/day = 4.0× better than SevenNet e3nn, 2.0× better than MACE cueq.
:::

## e3nn models scale steeply; cueq lines stay near-flat

![Latency vs. atom count for all model configurations](plots/comparison_latency.png)
![CuEq speedup vs. atom count](plots/comparison_speedup.png)

- **Left**: e3nn lines slope sharply; cueq lines near-flat; both e3nn-only configs OOM at 2,916 atoms
- **Right**: MACE regresses at 108 atoms (0.88×), crosses positive between 108–500 atoms
- SevenNet speedup: 1.40× → 4.00×; MACE speedup: 0.88× → 3.77× (at 1,372 before e3nn OOM)
- More atoms = more tensor products to parallelize = larger cuEq benefit

::: notes
Left: y-axis in ms, x-axis is atom count. Widening gap between e3nn and cueq shows growing benefit.
Right: speedup accelerates with size as more atoms amortize kernel launch overhead.
MACE 0.88× at 108 atoms is the only sub-1.0 point — visible in the right chart.
:::

## eSEN e3nn operation breakdown at 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|----------------:|
| **compute_forces** (autograd backward) | 101.52 | 53.4% |
| **SO2Conv** (×4 layers, aggregated) | 62.08 | 32.6% |
| **generate_graph** (GPU, nvalchemiops) | 10.73 | 5.6% |
| **obtain rotmat / wigner original** | 6.06 | 3.2% |
| **Others** (edge embedding, edgewise, data prep, …) | ~9.8 | ~5.2% |

::: notes
Leaf-only analysis avoids double-counting: wrapper ops like eSEN::model_forward are excluded.
compute_forces is a monolithic autograd.grad call traversing all 4 SO2Conv + edgewise + atomwise layers.
Profiler records only generic autograd ops (MmBackward0, MulBackward0) inside compute_forces — no per-layer attribution possible.
GPU graph generation stays flat (~11 ms) regardless of system size — GPU parallelism absorbs the full workload.
:::

## eSEN e3nn: backward pass and SO2Conv account for 86% of traced time

![eSEN e3nn 500 atoms pie chart](plots/esen_e3nn_esen-sm_500atoms_pie.png)
![eSEN e3nn 500 atoms kernel breakdown](plots/esen_e3nn_esen-sm_500atoms_kernels.png)

- `compute_forces` (backward): **53.4%** — Gemm + Elementwise kernels from the autograd backward pass
- `SO2Conv` (×4): **32.6%** — Scatter/Gather for message aggregation + Gemm for linear layers
- GPU graph generation: ~11 ms constant at all sizes; no CPU↔GPU idle time
- SO2Conv scales 14.3→171.8 ms (12×) for 12.7× atoms — near-linear; no cueq acceleration path

::: notes
eSEN's architectural advantage: GPU graph generation stays constant, eliminating the CPU bottleneck seen in MACE/SevenNet.
But without cueq support, the dominant backward pass (53%) cannot be accelerated by NVIDIA's fused kernels.
eSEN OOMs at 2,916 atoms on A100-40GB — the 6.3M-parameter model with 6 Å cutoff requires >40 GB VRAM.
At 1,372 atoms: SO2Conv grows to 171.8 ms (32%), compute_forces to 272.3 ms (51%).
:::

## MACE e3nn operation breakdown at 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|----------------:|
| **compute_forces** (autograd backward) | 60.03 | 55.8% |
| **message_passing** (×2 layers) | 17.18 | 16.0% |
| **generate_graph** (CPU, matscipy) | 14.94 | 13.9% |
| **SymmetricContraction** (×2 layers) | 10.83 | 10.1% |
| **Others** (embeddings, conv_weights, skip_tp, …) | ~4.6 | ~4.3% |

::: notes
generate_graph runs entirely on CPU (matscipy) while the GPU sits idle — 14% wasted at 500 atoms.
At 108 atoms graph gen is ~41% of leaf total due to one-time matscipy initialization overhead.
SymmetricContraction is unique to MACE — compresses high-order tensor products to invariant features.
L_max=1 (medium variant) enables non-trivial equivariant tensor products that cuEquivariance can accelerate.
:::

## MACE e3nn: CPU graph generation is 14% of traced time at 500 atoms

![MACE e3nn 500 atoms pie chart](plots/mace_e3nn_mace-mp-medium_500atoms_pie.png)
![MACE e3nn 500 atoms kernel breakdown](plots/mace_e3nn_mace-mp-medium_500atoms_kernels.png)

- `compute_forces`: **55.8%** (60 ms) — Gemm-dominant backward over 2 interaction layers
- `message_passing`: **16.0%** (17 ms) — Scatter/Gather + Gemm across both layers
- `generate_graph` (CPU): **13.9%** (15 ms) — matscipy on 1 CPU core; GPU completely idle
- `SymmetricContraction`: **10.1%** (11 ms) — heavily Elementwise (many-body contraction)

::: notes
matscipy CPU graph gen grows approximately linearly: 14→16→38→80 ms (108→500→1,372→2,916 atoms).
e3nn OOMs at 2,916 atoms; at 1,372 atoms compute_forces grows to 162.6 ms (57%) and generate_graph to 38.7 ms (13%).
The Elementwise-heavy SymmetricContraction kernel pattern distinguishes MACE from eSEN/SevenNet in the kernel breakdown.
:::

## SevenNet e3nn operation breakdown at 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|----------------:|
| **force_output** (autograd backward) | 53.32 | 63.3% |
| **convolution** (layers 0–4, total) | 13.78 | 16.3% |
| **generate_graph** (CPU, ASE/numpy) | 8.49 | 10.1% |
| **equivariant_gate** (layers 0–4, total) | 3.33 | 4.0% |
| **Others** (self_interaction, edge_embedding, …) | ~5.4 | ~6.4% |

::: notes
Highest backward-pass dominance of the three e3nn models (63%). All 5 conv layers + gates traversed in one autograd call.
convolution layers show sub-linear GPU scaling: 11.9 ms at 108 atoms → 59.8 ms at 2,916 atoms (5× for 27× atom increase).
ASE graph gen is lightest at small sizes (2.1 ms at 108 atoms) but grows linearly to 33.7 ms at 2,916 atoms.
force_output is remarkably stable at 63–69% of leaf traced time across all atom sizes.
:::

## SevenNet e3nn: force_output dominates at 63%; graph gen grows to 12% at large size

![SevenNet e3nn 500 atoms pie chart](plots/sevenn_e3nn_7net-0_500atoms_pie.png)
![SevenNet e3nn 500 atoms kernel breakdown](plots/sevenn_e3nn_7net-0_500atoms_kernels.png)

- `force_output` (backward): **63.3%** (53 ms) — Gemm + Elementwise across all 5 conv + gate layers
- `convolution` (×5): **16.3%** (14 ms total) — Scatter/Gather for TP aggregation + Gemm
- `generate_graph` (CPU): grows from 3% (108 atoms, 2.2 ms) to **12%** (2,916 atoms, 33.7 ms)
- `equivariant_gate` (×5): **4.0%** (3.3 ms total) — SiLU-style activations after each convolution

::: notes
SevenNet's 5-layer architecture (vs 2 in MACE, 4 in eSEN) means more individual conv ops but smaller per-op cost.
The Scatter/Gather pattern in the kernel breakdown identifies neighborhood aggregation via tensor products.
ASE graph gen (pure numpy, CPU) is the lightest at small sizes but becomes 12% of traced time by 2,916 atoms.
:::

## cuEquivariance speedup grows with system size; MACE regresses at 108 atoms

| Model | 108 atoms | 500 atoms | 1,372 atoms | 2,916 atoms |
|-------|----------:|----------:|------------:|------------:|
| **MACE** | 0.88× | 2.19× | 3.77× | N/A (e3nn OOM) |
| **SevenNet** | 1.40× | 1.46× | 2.68× | 4.00× |

::: notes
Speedup = e3nn latency / cueq latency. Values >1 mean cueq is faster.
MACE regression at 108 atoms: cueq kernel launch overhead exceeds compute savings for the small workload.
SevenNet starts positive even at 108 atoms (1.40×) — lighter architecture has lower overhead relative to compute.
Both confirm the core pattern: more atoms = more tensor products = larger amortized speedup from cuEquivariance.
:::

## MACE cueq operation breakdown at 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|----------------:|
| **compute_forces** (autograd backward) | 20.05 | 41.7% |
| **generate_graph** (CPU, matscipy) | 15.63 | 32.5% |
| **SymmetricContraction** (×2 layers) | 3.47 | 7.2% |
| **message_passing** (×2 layers) | 1.62 | 3.4% |
| **Others** (embeddings, conv_weights, skip_tp, …) | ~7.3 | ~15.2% |

::: notes
Most striking change vs e3nn: generate_graph rises from 13.9% to 32.5% — not because it got slower, but because GPU ops got much faster.
compute_forces: 60.0 ms → 20.1 ms (3.0×); message_passing: 17.2 ms → 1.6 ms (10.6×).
cuEq TP kernels (segmented_polynomial) appear in the kernel breakdown, replacing much of the Gemm/Elementwise tensor product work.
Amdahl's law: generate_graph at 32.5% of traced time caps overall speedup at ~3× regardless of GPU acceleration.
:::

## MACE cueq: generate_graph rises from 14% to 33% as GPU ops accelerate

![MACE e3nn 500 atoms pie chart](plots/mace_e3nn_mace-mp-medium_500atoms_pie.png)
![MACE cueq 500 atoms pie chart](plots/mace_cueq_mace-mp-medium_500atoms_pie.png)

- `compute_forces`: 60 ms → **20 ms** (3.0×); `message_passing`: 17 ms → **1.6 ms** (10.6×)
- `generate_graph` (CPU): 15 ms → 16 ms — from **14%** to **33%** of traced time
- `SymmetricContraction`: 10.8 ms → **3.5 ms** (3.1×)
- cuEq TP kernels (`segmented_polynomial`) appear in the kernel breakdown

::: notes
Left pie = e3nn, right pie = cueq at 500 atoms. The percentage redistribution is the key visual insight.
generate_graph didn't get slower — GPU ops got much faster, so CPU graph gen's share grew.
message_passing reaches 16.8× at 1,372 atoms — the L_max=1 tensor products are especially well-suited for cuEquivariance.
At 1,372 atoms: MACE e3nn 286.6 ms → MACE cueq 76.0 ms = 3.77× overall speedup.
:::

## SevenNet cueq operation breakdown at 500 atoms

| Operation | Effective Time (ms) | % of leaf total |
|-----------|--------------------:|----------------:|
| **force_output** (autograd backward) | 28.36 | 52.5% |
| **generate_graph** (CPU, ASE/numpy) | 8.35 | 15.5% |
| **convolution** (layers 0–4, total) | 8.01 | 14.8% |
| **equivariant_gate** (layers 0–4, total) | 2.97 | 5.5% |
| **Others** (self_interaction, edge_embedding, …) | ~6.3 | ~11.7% |

::: notes
force_output drops from 53.3 ms to 28.4 ms (1.88×) at 500 atoms — larger speedups appear at bigger sizes.
generate_graph rises from 10.1% to 15.5% — same Amdahl pattern as MACE but less extreme.
Convolution layers shrink from 13.8 ms to 8.0 ms (1.72× at 500 atoms; grows to 7.5× at 2,916 atoms).
cuEq TP kernels visible in convolution + gate operations, replacing Scatter/Gather and Gemm patterns.
:::

## SevenNet cueq: force_output accelerates from 1.9× at 500 to 6.6× at 2,916 atoms

![SevenNet e3nn 500 atoms pie chart](plots/sevenn_e3nn_7net-0_500atoms_pie.png)
![SevenNet cueq 500 atoms pie chart](plots/sevenn_cueq_7net-0_500atoms_pie.png)

- `force_output`: 53 ms → **28 ms** (1.88×) at 500 atoms; reaches **6.61×** at 2,916 atoms
- `convolution` (×5): 14 ms → **8 ms** (1.72×) at 500; grows to **7.5×** at 2,916 atoms
- `generate_graph` (CPU): 8.5 → 8.4 ms — rises from **10%** to **16%** of traced time
- SevenNet cueq at 72.1 ms (2,916 atoms) is 2.01× faster than MACE cueq at 145.0 ms

::: notes
Left = e3nn, right = cueq at 500 atoms. SevenNet's shift is less dramatic than MACE's but still significant.
The speedup scaling story: 1.7× at 500 atoms becomes 7.5× at 2,916 atoms — very strong size dependence.
The smallest model (0.8M params) achieves the best large-system throughput with cueq — parameter efficiency wins at scale.
:::

## MACE per-operation speedup at 1,372 atoms: message_passing reaches 16.8×

| Operation | MACE e3nn (ms) | MACE cueq (ms) | Speedup |
|-----------|---------------:|---------------:|--------:|
| **compute_forces** | 162.61 | 20.48 | **7.94×** |
| **message_passing** | 47.25 | 2.81 | **16.80×** |
| **SymmetricContraction** | 29.44 | 3.74 | **7.87×** |
| **generate_graph (CPU)** | 38.69 | 38.03 | **1.02×** |

::: notes
At 1,372 atoms, GPU operations reach 8–17× speedup while CPU graph gen stays at 1.02×.
message_passing reaching 16.8× is extraordinary — MACE's L_max=1 equivariant tensor products map perfectly to cuEquivariance's fused kernels.
The 38.7 ms CPU graph gen floor limits overall end-to-end speedup: 286.6 ms → 76.0 ms = 3.77×.
The gap between GPU speedups (8–17×) and CPU graph gen (1×) is the Amdahl bottleneck.
:::

## SevenNet per-operation speedup at 1,372 atoms: speedups grow 2× vs 500 atoms

| Operation | SevenNet e3nn (ms) | SevenNet cueq (ms) | Speedup |
|-----------|-------------------:|-------------------:|--------:|
| **force_output** | 93.08 | 28.91 | **3.22×** |
| **convolution (L>0, avg)** | 8.50 | 1.76 | **4.84×** |
| **generate_graph (CPU)** | 14.95 | 14.24 | **1.05×** |

::: notes
At 1,372 atoms: SevenNet e3nn 139.2 ms → SevenNet cueq 52.0 ms = 2.68× overall.
Speedups roughly double from 500 to 1,372 atoms, and double again from 1,372 to 2,916 atoms.
At 2,916 atoms: force_output reaches 6.61× (191.3→28.9 ms) and convolution 7.5×, driving the 4.00× overall.
SevenNet cueq outperforms MACE cueq at 1,372 atoms: 52.0 ms vs 76.0 ms (1.46× faster).
:::

## Graph generation: implementation strategy comparison and detailed scaling

| | **eSEN** | **MACE** | **SevenNet** |
|---|---|---|---|
| **Library** | nvalchemiops | matscipy | ASE (numpy) |
| **Device** | GPU | CPU | CPU |
| **Scaling** | ~constant (~11 ms) | linear (14→80 ms) | linear (2→34 ms) |
| **% of leaf total** | ~13→2% (decreasing) | ~24→13% (significant) | ~3→12% (growing) |

| Atoms | eSEN (ms) | MACE cpu / gpu (ms) | SevenNet cpu / gpu (ms) |
|------:|----------:|--------------------:|------------------------:|
| **108** | 10.9 | 14.0 / 2.0 | 2.1 / 0.3 |
| **500** | 10.8 | 15.6 / 2.2 | 8.4 / 0.3 |
| **1,372** | 10.9 | 38.0 / 3.1 | 14.2 / 0.4 |
| **2,916** | OOM | 80.4 / 4.5 | 33.7 / 0.7 |

::: notes
eSEN (nvalchemiops, GPU): constant ~11 ms — GPU parallelism absorbs the full workload. Fraction decreases as GPU compute grows.
MACE (matscipy, CPU): linear — 14→80 ms over 27× atom increase. cpu_time is the actual computation; gpu_time is just data transfer.
SevenNet (ASE/numpy, CPU): lightest at 108 atoms (2.1 ms) but grows linearly to 33.7 ms at 2,916 atoms.
With 1 SLURM core, GPU idles completely during CPU graph construction.
Moving MACE/SevenNet to GPU-based graph gen could cut 14–80 ms per inference step.
Note: MACE and SevenNet values use measurements from the cueq run to avoid initialization overhead inflation.
:::

## cueq kernels outrun autograd dispatch, leaving the GPU idle between launches

![MACE e3nn 500 atoms kernel breakdown](plots/mace_e3nn_mace-mp-medium_500atoms_kernels.png)
![MACE cueq 500 atoms kernel breakdown](plots/mace_cueq_mace-mp-medium_500atoms_kernels.png)

- **e3nn**: 100–240 μs per kernel — long enough to hide CPU dispatch latency (~1–10 μs gaps)
- **cueq**: 6–36 μs per kernel — shorter than Python autograd dispatch overhead (~40 μs gaps)
- Gray "Idle/Overhead" band grows in **absolute time** with cueq (not just proportionally)
- At 1,372 atoms gap fraction improves: MACE 75%→35%, SevenNet 87%→78%

::: notes
Left = MACE e3nn kernel breakdown at 500 atoms; right = MACE cueq — both are backward pass (compute_forces).
Root cause: same Python autograd graph traversal (~40 μs per dispatch) runs for both e3nn and cueq.
With e3nn, each 100–240 μs kernel overlaps the next CPU dispatch. With cueq, each 6–36 μs kernel finishes first.
torch.compile or fused CUDA backward kernels would eliminate starvation by removing the Python dispatch loop.
:::

## GPU inter-kernel gaps expand 20× with cueq; 75–87% of backward time is wasted

| Metric | MACE e3nn | MACE cueq | SevenNet e3nn | SevenNet cueq |
|--------|----------:|----------:|--------------:|--------------:|
| **Kernel total time** | 59.3 ms | 4.9 ms | 35.4 ms | 3.6 ms |
| **Inter-kernel gap total** | 0.7 ms | 15.0 ms | 17.3 ms | 24.1 ms |
| **Avg kernel duration** | 104 μs | 13 μs | 28 μs | 6 μs |
| **Avg inter-kernel gap** | 1.2 μs | 41 μs | 14 μs | 39 μs |

::: notes
All measurements: backward pass (compute_forces / force_output) at 500 atoms — trace-level gap analysis.
MACE cueq: gap total is 15 ms out of ~20 ms backward — 75% of backward time is CPU starvation.
SevenNet cueq: 24 ms gap out of ~28 ms backward — 86% of backward time wasted on Python dispatch overhead.
Gap distribution shift: e3nn gaps cluster at 1–10 μs (hardware launch latency); cueq gaps shift to 10–100 μs (Python autograd).
At 1,372 atoms the gap fraction improves as kernels get more compute, partially re-hiding CPU overhead.
:::

## Three optimization paths emerge from these profiles

- **Move graph generation to GPU** — MACE/SevenNet leave GPU idle during CPU neighbor-list construction; GPU-based gen (e.g. nvalchemiops) could cut 14–80 ms/step; largest impact on MACE where graph gen is up to 33% of traced time with cueq
- **Fuse the backward pass** — cueq kernels (6–36 μs) expose ~40 μs autograd dispatch gaps; torch.compile or custom fused CUDA backward would recover 15–24 ms/step at 500 atoms and raise the cueq speedup ceiling
- **Profile at 4,000+ atoms on 80 GB GPU** — eSEN and MACE e3nn OOM at 2,916 atoms on A100-40GB; verify cuEq scaling continues; benchmark multi-GPU configs; test whether eSEN GPU graph gen also OOMs at larger sizes

::: notes
Each path targets a distinct bottleneck: Amdahl (CPU graph gen), pipeline starvation (backward), scale validation.
Graph-to-GPU is the most actionable: prior art exists (nvalchemiops), no model architecture changes needed.
Backward fusion requires CUDA engineering but would raise the cueq ceiling from ~4× toward ~6×+ at production sizes.
Larger-GPU profiling is needed to confirm whether SevenNet cueq's near-flat scaling holds at 5,000+ atoms.
:::
