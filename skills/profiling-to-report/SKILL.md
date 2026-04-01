---
name: profiling-to-report
description: transform auto-generated profiling data report into a complete analytical profiling report. the input is an auto-generated markdown report (tables, plots, pipeline starvation data) produced by generate_report.py. the agent adds methodology sections, model descriptions, interpretive analysis, scaling observations, cross-model comparisons, and editorial prose to produce a publication-ready profiling report.
---

# Profiling To Report

Transform an auto-generated data report (tables and plot references) into a complete, interpretive profiling report. The input report is produced by [scripts/generate_report.py](scripts/generate_report.py) and contains numerical tables, plot references, and basic pipeline starvation analysis. This skill adds the explanatory prose, methodology background, model descriptions, and analytical observations that make the report useful to readers.

> **Scope**: This skill is specific to the **mlip-profiling** project. It assumes the profiling infrastructure in `profile_mlip.py` (ModelAdapter pattern, `record_function` instrumentation) and the model types registered in `get_adapter()`. Currently supported models: eSEN, MACE, SevenNet. When new models are added, update [references/analysis_guide.md](references/analysis_guide.md) with their architecture details.

## Inputs

- **Auto-generated report**: `{results_dir}/profiling_report_generated.md` (output of `generate_report.py`)
- **summary.json files**: `{results_dir}/{type}/{backend}/{model}/summary.json` (for additional context)
- **Existing report** (optional): `{results_dir}/profiling_report.md` (if a previous version exists, use it as a style/structure reference)
- **CLAUDE.md**: Project-level context (model details, environment setup, conventions)

## Output

- **`{results_dir}/profiling_report.md`**: Complete analytical profiling report

> If `profiling_report.md` already exists, it will be **overwritten**. The existing version is used only as a style reference (step 1), not as a base for incremental edits.

## Default workflow

1. **Read inputs**: Read the auto-generated report, all summary.json files, and CLAUDE.md for project context. If a previous `profiling_report.md` exists, read it to understand the expected writing style and level of detail — but do not copy from it. The new report is built fresh from the auto-generated data.
2. **Build report skeleton**: Create the full report structure per [references/report_structure.md](references/report_structure.md), starting from the auto-generated tables.
3. **Insert methodology sections** (1.1–1.5): Use [references/methodology.md](references/methodology.md) as baseline text. Adapt if profiler settings differ from defaults.
4. **Write model descriptions** (2.1–2.4): Describe each model's architecture, graph strategy, and distinctive features using data from summary.json and model_info. Follow patterns in [references/analysis_guide.md](references/analysis_guide.md).
5. **Add operation breakdown details** (Section 3): Explain leaf vs. wrapper distinction, kernel categories. This section is largely static.
6. **Interpret each data section** (Section 4): For every auto-generated table and plot reference, add 1–3 paragraphs of analysis:
   - What the data shows (headline finding)
   - Why it happens (mechanism)
   - How it scales (trend across atom counts)
   - Use [references/analysis_guide.md](references/analysis_guide.md) for interpretation patterns.
7. **Add Appendix D** (Document Export): Include pandoc commands for PDF and HTML export. Use [references/report_structure.md](references/report_structure.md) for the template.
8. **Write draft file**: Save as `{results_dir}/profiling_report.md`.
9. **Final review**: Re-read the complete report end-to-end. Validate against [references/quality_checklist.md](references/quality_checklist.md). Fix any issues found before delivering.

## Interpretation rules

- **Data-first**: Every claim must be traceable to a number in the auto-generated report or summary.json. Never invent numbers.
- **Explain mechanisms**: Don't just state "X is faster" — explain why (e.g., tensor product fusion, GPU parallelism, Amdahl's law).
- **Scaling language**: Use precise scaling terms — "near-linear" (10× atoms → 10× time), "sub-linear" (10× atoms → 5× time), "super-linear" (10× atoms → 15× time). Always cite the actual multipliers.
- **Percentage context**: When citing percentages, clarify the denominator (e.g., "53% of leaf traced time" not just "53%").
- **OOM as data**: Treat OOM as an informative result, not a missing value. Note which models/backends survive at what sizes.
- **Avoid superlatives**: Don't say "extremely fast" or "very efficient". Use specific numbers instead.
- **CPU vs GPU distinction**: Always clarify when an operation runs on CPU vs GPU, especially for graph generation.

## Writing style

- **Concise technical prose**: 1–3 sentences per observation. No filler.
- **Present tense** for describing behavior ("SO2Conv dominates at 500 atoms").
- **Bold** key findings in the first sentence of each analytical paragraph.
- **Tables over prose** when comparing ≥3 values. Prose for ≤2.
- **Plot references**: Every plot should have at least one sentence of interpretation above or below it.
- **Section flow**: Each section should read independently but build toward the comparative conclusions.

## Formatting conventions

### TOC anchor links

TOC entries must link to heading anchors **without** section numbers. Markdown renderers (GitHub, pandoc) generate anchors by lowercasing and replacing spaces/special characters — they strip leading numbers.

- ✅ `[Profiling Methodology](#profiling-methodology)`
- ❌ `[Profiling Methodology](#1-profiling-methodology)`

For `## 4. Profiling Results & Analysis`, the anchor is `#profiling-results-analysis` (ampersand is stripped).

### External hyperlinks

Add external links for key tools and projects on first mention:

- **PyTorch Profiler**: `[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)`
- **Perfetto**: `[Perfetto](https://ui.perfetto.dev)`
- **Model repos**: Link each model name to its GitHub repo on first mention in Section 2 (e.g., `[MACE](https://github.com/ACEsuit/mace)`, `[SevenNet](https://github.com/MDIL-SNU/SevenNet)`, `[fairchem](https://github.com/FAIR-Chem/fairchem)`)
- **Libraries**: Link matscipy, e3nn, cuEquivariance on first mention
- Do not repeat links after the first occurrence

## Section-specific guidance

### Methodology (Sections 1.1–1.5)
Mostly static text from [references/methodology.md](references/methodology.md). Adapt only if:
- Profiler schedule (wait/warmup/active) differs from 5/5/5
- timeit settings differ from number=10, repeat=5
- Device is not single CUDA GPU

### Model Descriptions (Section 2)
For each model, cover:
1. Architecture type (one sentence)
2. Graph construction strategy (GPU vs CPU, which library)
3. Number of message passing / interaction layers
4. Force computation method (always autograd.grad)
5. Available backends
6. Any distinctive feature (e.g., MACE's SymmetricContraction, eSEN's SO(2) convolutions)

Use the table in CLAUDE.md as a factual reference.

### Operation Breakdown (Section 4.2)
For each model config at the representative atom count:
- State the top 2–3 operations and their combined percentage
- Explain what the backward pass contains (all forward ops' gradients, monolithic, not attributable)
- Describe kernel composition from the kernel breakdown plot
- Add a "Scaling" paragraph showing how the top operations change across atom counts

### Graph Generation (Section 4.3)
- Compare GPU (eSEN) vs CPU (MACE, SevenNet) approaches
- Note the scaling behavior (constant for GPU, linear for CPU)
- Discuss the implication: CPU graph generation becomes a larger fraction as GPU ops get faster (especially with cueq)

### CuEq Acceleration (Section 4.4)
- Explain the speedup trend (increasing with atom count)
- Explain any regressions at small sizes (kernel launch overhead > compute savings)
- Show per-operation speedup to identify where cueq helps most
- Discuss memory benefit (e3nn OOM → cueq succeeds)
- Explain GPU pipeline starvation with the data from the auto-generated table
- Discuss Amdahl's law limitation from CPU graph generation

## Required deliverables

- `{results_dir}/profiling_report.md` — complete report with all sections numbered 1–4 plus Appendix
- Every auto-generated table and plot reference must be preserved
- All added prose must be factually grounded in the data
