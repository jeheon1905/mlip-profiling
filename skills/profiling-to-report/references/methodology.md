# Methodology Reference Text

Use the following as baseline text for Sections 1.1–1.5. Adapt values if the profiler settings in `summary.json` differ from defaults.

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
|←  wait ({wait})  →|←  warmup ({warmup})  →|←  active ({active})  →|
                                              ← traces recorded →
```

- **Wait** ({wait} steps): Execute without profiling overhead — stabilize GPU state
- **Warmup** ({warmup} steps): Profiler active but results discarded — allow JIT warmup
- **Active** ({active} steps): Full profiling with trace recording — collect measurements

> **Adapt**: Replace `{wait}`, `{warmup}`, `{active}` with values from `summary.json → profiler_schedule`. Defaults are 5/5/5.

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
| **timeit latencies** | Clean measurement without profiling overhead | `timeit.repeat(number={number}, repeat={repeat})` after profiler completes |

The **timeit** measurement is reported as the primary latency, as it is free from profiler instrumentation overhead. QPS (queries per second) and ns/day are derived from timeit.

> **Adapt**: Replace `{number}` and `{repeat}` with values from `summary.json → timeit_settings`. Defaults are 10/5.

### 1.5 Effective Time Metric

The profiler reports both `cpu_time` and `gpu_time` per operation. To produce a single meaningful metric, we use **effective time**:

- **CPU-bound operations** → use `cpu_time` (e.g., `generate_graph` on CPU, `data_preparation`)
- **GPU-bound operations** → use `gpu_time` (e.g., `compute_forces`, `convolution`)

Classification is done via an explicit **CPU_OPERATIONS allowlist** per model type in `generate_report.py`, rather than heuristics. Check the `CPU_OPERATIONS` dict in the script for the current mapping. At time of writing:

| Model | CPU Operations |
|-------|----------------|
| eSEN | `data_preparation`, `data_to_device` |
| MACE | `generate_graph` |
| SevenNet | `generate_graph` |

> **Adapt**: If new models are added, verify the CPU_OPERATIONS dict in `generate_report.py` and update this table accordingly.

**Important**: The sum of leaf effective times (leaf effective total) is **not** a wall-clock measurement. It represents the aggregate device-occupancy time across CPU and GPU. This value may differ from the timeit wall-clock latency for two reasons:

1. **Profiler overhead** — the profiler's instrumentation inflates individual operation times. This effect is largest for small structures where operation durations are short relative to overhead.
2. **CPU-GPU overlap** — CPU and GPU operations can execute concurrently. Simple summation double-counts this overlap, or conversely, untracked idle/synchronization gaps may cause the sum to undercount wall-clock time.

At large system sizes (≥1,372 atoms), the leaf effective total closely approximates timeit wall-clock latency for all three models (within ~0–2%).

Percentages reported in operation breakdowns represent **relative proportions within the traced total**, not fractions of wall-clock time.
