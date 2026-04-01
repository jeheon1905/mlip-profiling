# Analysis Guide

How to interpret MLIP profiling data and write analytical prose for each section.

---

## Model Description Patterns

### Per-model template

For each model, cover these points in order:

1. **Architecture type** (one sentence): E(3)-equivariant, SO(2)-equivariant, etc.
2. **Graph construction**: GPU (nvalchemiops) vs CPU (matscipy/ASE). State the library name.
3. **Message passing layers**: Count and name (e.g., "4 SO2Conv layers", "2 Interaction + SymmetricContraction", "5 Convolution + Gate")
4. **Force computation**: Always `autograd.grad`. Note that backward pass is monolithic.
5. **Distinctive feature**: What makes this model unique (e.g., eSEN's SO(2) rotational symmetry, MACE's SymmetricContraction for many-body interactions, SevenNet's tensor product convolutions)
6. **Backend support**: List available backends

### Key facts per model

Extract architecture details from `summary.json → model` and CLAUDE.md. For each model, look up:
- `model.type` → architecture family
- `model.cutoff` → cutoff radius
- `model.num_params` → parameter count (if available)
- `model.backend` → supported backends
- Operation names in `results.*.operations` → layer structure

**eSEN (fairchem)**:
- SO(2)-equivariant convolutions (eSCN-style)
- GPU graph construction via nvalchemiops → no CPU-GPU data transfer overhead
- Message passing layers: each contains SO2Conv + edgewise + atomwise (count from operation names)
- Pre-computed Wigner/rotation matrices

**MACE**:
- Higher-order equivariant message passing
- CPU graph construction via matscipy → significant CPU bottleneck
- Interaction layers: linear_up → conv_weights → message_passing → skip_tp, then ProductBasis → SymmetricContraction (count from operation names like `MACE::interaction_0`, `_1`, ...)
- SymmetricContraction is unique to MACE — compresses high-order tensor products to invariant features

**SevenNet**:
- Standard E(3)-equivariant with tensor product convolutions
- CPU graph construction via ASE/numpy
- Convolution layers: self_connection_intro → self_interaction_1 → convolution → self_interaction_2 → self_connection_outro → equivariant_gate (count from operation names like `SevenNet::0_convolution`, `1_convolution`, ...)

> **Note**: Avoid citing specific percentages or speedup numbers in model descriptions. Those belong in Section 4 where the data tables provide context.

---

## Operation Breakdown Analysis

### How to interpret the operation table

1. **Identify the dominant operation**: Usually the backward pass (`compute_forces` / `force_output`). State its percentage.
2. **Explain what the backward pass contains**: It's a single `autograd.grad` call that traverses the entire forward computation graph in reverse. The profiler sees only generic autograd ops (MmBackward0, MulBackward0, etc.), making it impossible to attribute backward cost to individual forward operations.
3. **Identify the 2nd-largest operation**: Usually message passing / convolution or graph generation  
4. **Note CPU vs GPU distinction**: Graph generation is CPU for MACE/SevenNet, GPU for eSEN

### Kernel breakdown interpretation

Map kernel categories to computational meaning:

| Kernel Category | What it means |
|----------------|---------------|
| **Gemm** | Linear layers, matrix multiplications, tensor contractions |
| **Elementwise** | Activations, additions, scaling, element-wise ops |
| **Reduction** | Sum, mean, normalization (BatchNorm, LayerNorm) |
| **Scatter/Gather** | Message aggregation in GNNs, neighborhood pooling |
| **Copy/Transpose** | Data layout transformations, tensor reshaping |
| **Memcpy/Memset** | CPU↔GPU data transfer, memory initialization |
| **cuEq TP** | cuEquivariance tensor product kernels (segmented_polynomial) |
| **Idle/Overhead** | GPU idle time between kernels — launch latency, CPU overhead |

### Scaling analysis patterns

When describing how operations scale across atom counts:

- **Compute the actual multiplier**: If 108→1372 is 12.7× atoms and time goes from 14 ms → 172 ms (12.3×), say "near-linear scaling (12.3× for 12.7× atoms)"
- **Note sub-linear GPU scaling**: GPU operations often scale sub-linearly due to better parallelism utilization at larger sizes
- **Note linear CPU scaling**: CPU graph generation typically scales linearly
- **Note what stays constant**: eSEN's GPU graph generation is ~11 ms regardless of system size
- **Note growing fractions**: As GPU ops get faster (with cueq), CPU graph generation becomes a larger fraction

### Template paragraph for operation breakdown

```
At {N} atoms, {backward_pass_op} and {second_op} together account for {combined_pct}% 
of leaf traced time. The kernel breakdown reveals that {backward_pass_op} is dominated 
by {kernel_types} from the backward pass, while {second_op}'s GPU time is split between 
{kernel_types}. {Graph_gen_observation}.

**Scaling**: As system size grows from {min_atoms} to {max_atoms} atoms, {key_op} time 
grows from {min_time} ms to {max_time} ms ({multiplier}× for {atom_multiplier}× atoms), 
indicating {linear_or_sublinear} scaling. {second_scaling_observation}.
```

---

## Graph Generation Analysis

### Key comparisons

1. **GPU vs CPU**: eSEN's GPU graph gen is constant (~11 ms) vs MACE/SevenNet's linear CPU growth
2. **CPU implementation difference**: matscipy (MACE) vs ASE/numpy (SevenNet) — both are CPU but differ in overhead
3. **Fraction trend**: For CPU implementations, fraction of leaf total grows with cueq because GPU ops get faster but CPU graph gen doesn't
4. **Implication**: Moving graph generation to GPU could yield significant speedups for MACE (up to ~24% of leaf time at small sizes)

### Caveat about initialization overhead

The first invocation of graph generation may include one-time initialization costs. If multiple backend runs show different graph_generation times for the same structure, note this discrepancy and use the lower value as the steady-state estimate.

---

## CuEq Acceleration Analysis

### Speedup interpretation

1. **Increasing speedup with size**: Expected — larger systems have more tensor product operations to parallelize, amortizing kernel launch overhead
2. **Regression at small sizes**: cueq kernel launch overhead exceeds compute savings. Note the crossover point.
3. **Memory benefit**: If e3nn OOMs but cueq succeeds, this is a significant finding — cueq reduces memory consumption
4. **Amdahl's law**: CPU graph generation is unaffected by cueq, limiting overall end-to-end speedup

### Per-operation speedup interpretation

- **Forward tensor products**: `message_passing`, `convolution`, `SymmetricContraction` typically show the largest speedups. State the actual values from the per-op speedup table.
- **Backward pass**: `compute_forces` / `force_output` shows aggregate speedup combining all backward gradients. This is typically lower than individual forward tensor product speedups.
- **CPU operations**: Should be ~1.0× (no GPU acceleration applies). Verify from the table.
- **Insight**: The per-operation speedups reveal that cueq accelerates equivariant tensor products most, but the backward pass speedup is modulated by non-tensor-product gradients

### GPU Pipeline Starvation explanation

This is a key analytical insight. Structure the explanation as:

1. **State the data**: cueq kernel breakdown shows larger Idle/Overhead, both in proportion and absolute time. Use the starvation table from the auto-generated report.
2. **Present the mechanism**: Compare avg kernel duration (e3nn vs cueq) against avg inter-kernel gap. When kernels are shorter than gaps, GPU starves.
3. **Support with gap distribution**: e3nn gaps should cluster in the 1–10 μs range (hardware launch latency), while cueq gaps shift higher (CPU autograd overhead). Read actual values from the table.
4. **Note scaling improvement**: At larger atom counts, per-kernel compute increases, partially re-hiding CPU overhead. Compare gap fractions across atom counts.

### Template paragraph for pipeline starvation

```
The cueq kernel plots show a larger Idle/Overhead (gray) component compared to e3nn — 
not only in proportion but also in **absolute time**. [Preserve starvation table from 
auto-generated report]. The root cause is **GPU pipeline starvation**. With e3nn, each 
kernel runs [avg_kernel_e3nn from table] μs, long enough to hide the CPU's autograd 
dispatch overhead. With cueq, kernels shrink to [avg_kernel_cueq from table] μs, but 
the CPU still walks the same Python autograd graph ([avg_gap_cueq from table] μs per 
dispatch) — so the GPU idles waiting for the next launch.

At [larger_atoms] atoms, the gap fraction improves ([cite before/after values from 
table]) as larger systems increase per-kernel compute, partially re-hiding CPU overhead 
behind GPU execution.
```

> **Important**: All numbers in brackets must come from the auto-generated starvation table or summary.json. Never estimate or round beyond what the data shows.

---

## Cross-Model Comparison

When comparing models, organize by:

1. **Latency ranking**: Which model is fastest at each system size? Does the ranking change?
2. **Scaling efficiency**: Which model's latency grows most/least aggressively?
3. **Memory efficiency**: Which models survive at the largest sizes?
4. **cueq benefit**: Which model benefits more from cueq and why?
5. **Architecture-driven differences**: Relate performance to architectural choices (SO(2) vs tensor product, GPU vs CPU graph gen, number of layers)
