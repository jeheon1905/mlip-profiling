---
title: "MLIP Profiling: eSEN vs MACE vs SevenNet on A100"
author: Profiling Team — NVIDIA A100-PCIE-40GB
notes: |
  Profiling results for three MLIP models on A100. Focus: inference bottlenecks,
  cuEquivariance acceleration, and GPU pipeline behavior.
---

## cuEquivariance delivers up to 4× end-to-end speedup and prevents OOM

- **SevenNet (cueq)**: 72.1 ms at 2,916 atoms — best large-system throughput (1.20 ns/day)
- **MACE (cueq)**: 3.77× faster at 1,372 atoms; enables 2,916 atoms (145 ms) where e3nn OOMs
- **eSEN**: highest latency at all sizes (58–477 ms), no cueq support, OOMs at 2,916 atoms
- MACE e3nn is fastest at small sizes (40.2 ms at 108 atoms) but scales steeply
- **Key insight**: backend choice (e3nn vs cueq) matters as much as model architecture
- Both MACE (e3nn) and eSEN hit OOM at 2,916 atoms on A100-40GB

::: notes
The headline finding. cuEquivariance is not optional for production-scale MLIP.
SevenNet cueq delivers 2.01x better throughput than MACE cueq at 2,916 atoms.
MACE regresses at 108 atoms (0.88x) — cueq overhead exceeds savings for small systems.
:::

## Setup: three E(3)-equivariant GNN potentials on A100-40GB

- **eSEN** — 6.3M params, SO(2)-equiv conv, 4 SO2Conv layers, GPU graph gen (nvalchemiops), 6.0 Å cutoff, e3nn only
- **MACE** — 4.7M params, higher-order MP + SymmetricContraction, 2 interaction layers, CPU graph gen (matscipy), 6.0 Å, e3nn / cueq
- **SevenNet** — 0.8M params, tensor product conv + gate, 5 conv layers, CPU graph gen (ASE/numpy), 5.0 Å, e3nn / cueq
- **Benchmark**: Cu FCC supercells — 108, 500, 1,372, 2,916 atoms on NVIDIA A100-PCIE-40GB
- **Profiler**: PyTorch Profiler (wait=5, warmup=5, active=5) + timeit (number=10, repeat=5)
- **Metric**: leaf effective time — cpu_time for CPU ops, gpu_time for GPU ops; avoids double-counting

::: notes
All three are E(3)-equivariant GNN potentials predicting energy and forces.
Common pipeline: Graph Construction → Embedding → Message Passing → Readout → Forces (autograd).
Timeit is the primary latency; profiler traces provide operation-level breakdown.
:::

## Latency scaling and cuEquivariance speedup

![Latency vs atom count](plots/comparison_latency.png)
![CuEq speedup vs atom count](plots/comparison_speedup.png)

::: notes
Left: MACE e3nn is fastest at 108 atoms but OOMs at 2,916. SevenNet cueq is the flattest curve.
Right: speedup grows with system size. SevenNet 1.4x→4.0x; MACE 0.88x→3.77x.
The cueq crossover for MACE is between 108 and 500 atoms.
:::

## eSEN: backward pass and SO2Conv account for 86% of traced time

![eSEN 500 atoms pie](plots/esen_e3nn_esen-sm_500atoms_pie.png)
![eSEN 500 atoms kernels](plots/esen_e3nn_esen-sm_500atoms_kernels.png)

::: notes
compute_forces (autograd backward): 53.4% — Gemm + Elementwise kernels in reverse pass.
SO2Conv (4 layers aggregated): 32.6% — Scatter/Gather for message aggregation + Gemm.
generate_graph (GPU, nvalchemiops): 5.6% — stays flat at ~11ms regardless of system size.
Scaling: SO2Conv grows 14.3→171.8ms (12x for 12.7x atoms) — near-linear.
:::

## eSEN operation analysis at 500 atoms

- `compute_forces` (autograd backward): **53.4%** — Gemm + Elementwise kernels in reverse pass
- `SO2Conv` (4 layers aggregated): **32.6%** — Scatter/Gather for message aggregation + Gemm
- `generate_graph` (GPU, nvalchemiops): **5.6%** — stays flat at ~11 ms regardless of system size
- `obtain rotmat/wigner`: **3.2%** — pre-computed rotational features
- Scaling: SO2Conv grows 14.3→171.8 ms (12× for 12.7× atoms); graph gen stays ~11 ms
- No cueq support → all tensor products use e3nn; backward pass is monolithic (autograd.grad)

::: notes
eSEN's architectural advantage: GPU graph generation stays constant.
But without cueq support, the backward pass can't be accelerated.
The autograd backward traverses the entire computation graph in one call.
:::

## MACE e3nn vs cueq: CPU graph generation becomes the bottleneck

![MACE e3nn 500 atoms pie](plots/mace_e3nn_mace-mp-medium_500atoms_pie.png)
![MACE cueq 500 atoms pie](plots/mace_cueq_mace-mp-medium_500atoms_pie.png)

::: notes
Same model, same input. cueq cut compute_forces 60→20ms and message_passing 17→1.6ms.
But generate_graph stayed at 16ms (CPU). It went from 14% to 33%. Amdahl's law in action.
:::

## MACE per-operation speedup with cuEquivariance (500 atoms)

- `compute_forces`: 60.0 → 20.1 ms (**3.0×**) — from 55.8% to 41.7% of leaf total
- `message_passing`: 17.2 → 1.6 ms (**10.6×**) — from 16.0% to 3.4%
- `SymmetricContraction`: 10.8 → 3.5 ms (**3.1×**) — from 10.1% to 7.2%
- `generate_graph` (CPU): 14.9 → 15.6 ms (**unchanged**) — from 13.9% to **32.5%**
- At 1,372 atoms: compute_forces **7.94×**, message_passing **16.80×**, generate_graph ~1.02×
- **Amdahl's law**: unaccelerated CPU graph gen is now the single largest bottleneck

::: notes
The key numerical detail. message_passing reaches an extraordinary 16.8x speedup at 1,372 atoms
but generate_graph (matscipy, CPU) grows to 38ms and limits overall end-to-end speedup.
:::

## SevenNet e3nn vs cueq: force_output speedup grows with system size

![SevenNet e3nn 500 atoms pie](plots/sevenn_e3nn_7net-0_500atoms_pie.png)
![SevenNet cueq 500 atoms pie](plots/sevenn_cueq_7net-0_500atoms_pie.png)

::: notes
force_output drops 53→28ms (1.9x) at 500 atoms. generate_graph rises from 10% to 16%.
At 2,916 atoms, force_output speedup reaches 6.6x, convolution 7.5x, overall 4.0x.
:::

## SevenNet per-operation speedup with cuEquivariance (500 atoms)

- `force_output`: 53.3 → 28.4 ms (**1.88×**) — from 63.3% to 52.5% of leaf total
- `convolution` (×5): 13.8 → 8.0 ms (**1.72×**) — from 16.3% to 14.8%
- `generate_graph` (CPU): 8.5 → 8.4 ms (**unchanged**) — from 10.1% to **15.5%**
- `equivariant_gate` (×5): 3.3 → 3.0 ms (1.12×) — from 4.0% to 5.5%
- At 2,916 atoms: force_output **6.61×** (191→29 ms), convolution **7.5×**, overall **4.00×**
- Graph gen scales linearly on CPU: 2.1→33.7 ms (3%→12% of leaf total)

::: notes
SevenNet shows the clearest scaling story. At 500 atoms speedups are modest (1.7–1.9x)
but at 2,916 atoms they reach 6.6x for backward and 7.5x for convolutions.
The smallest model (0.8M params) delivers the best large-system throughput with cueq.
:::

## Kernel-level view: cueq replaces Gemm/Scatter with fused TP kernels

![MACE e3nn kernels 500 atoms](plots/mace_e3nn_mace-mp-medium_500atoms_kernels.png)
![MACE cueq kernels 500 atoms](plots/mace_cueq_mace-mp-medium_500atoms_kernels.png)

::: notes
e3nn: Gemm (ampere_sgemm, cutlass) + Scatter/Gather + Elementwise dominate GPU time.
cueq: segmented_polynomial kernels (cuEq TP) replace much of Gemm/Scatter.
Note the large Idle/Overhead (grey) in cueq — GPU pipeline starvation.
:::

## GPU pipeline starvation: cueq kernels outrun Python autograd dispatch

- **e3nn kernels**: 100–240 μs each → GPU execution hides CPU autograd dispatch overhead
- **cueq kernels**: 6–36 μs each → GPU idles ~40 μs between launches waiting for CPU
- MACE backward gap: 0.7 ms (e3nn) → **15.0 ms** (cueq); avg inter-kernel gap 1.2 → 41 μs
- SevenNet backward gap: 17.3 ms → **24.1 ms**; avg kernel 28 μs → 6 μs
- **Root cause**: Python autograd dispatch ~40 μs/op can't keep up with <36 μs kernels
- At 1,372 atoms gap fraction improves (MACE 75%→35%) as per-kernel work grows

::: notes
The kernel breakdown reveals WHY cueq has diminishing returns at small sizes.
E3nn kernels are long enough to overlap with CPU dispatch. Cueq kernels are too fast.
A compiled or fused backward pass (torch.compile) would eliminate this bottleneck.
:::

## Graph generation strategies: GPU constant vs CPU linear

- **eSEN** (nvalchemiops, GPU): constant ~11 ms at all sizes — GPU absorbs workload; no CPU↔GPU idle
- **MACE** (matscipy, CPU): linear 14→80 ms; GPU idle during construction; **24–33%** of traced time
- **SevenNet** (ASE/numpy, CPU): linear 2→34 ms; lightest at small sizes; 3%→12% of leaf total
- **Tradeoff**: GPU graph gen avoids idle but consumes memory (eSEN OOMs first at 2,916 atoms)
- **Implication**: moving MACE/SevenNet to GPU graph gen could cut 14–80 ms per inference step

::: notes
For MACE at 2,916 atoms, 80ms of CPU graph construction is the single largest bottleneck.
eSEN's approach avoids CPU-GPU transfer bottleneck but uses more GPU memory.
:::

## Three optimisation paths emerge from these profiles

- **1. Move graph generation to GPU** — MACE/SevenNet leave GPU idle during CPU graph construction; nvalchemiops could cut 14–80 ms/step; largest impact on MACE (24–33% of traced time)
- **2. Fuse the backward pass** — cueq kernels (6–36 μs) expose 40 μs autograd dispatch overhead; torch.compile or fused CUDA kernels would close the 15+ ms gap; ~20% improvement at 500 atoms
- **3. Profile at 4,000+ atoms on 80GB GPUs** — eSEN + MACE (e3nn) OOM at 2,916 on A100-40GB; verify optimisations compound at production scale; benchmark multi-GPU with eSEN distributed ops

::: notes
Graph-to-GPU targets the Amdahl bottleneck (biggest impact on MACE).
Backward fusion targets the starvation bottleneck (biggest impact on cueq at moderate sizes).
Larger profiling validates scaling at production sizes. These are actionable next steps.
:::
