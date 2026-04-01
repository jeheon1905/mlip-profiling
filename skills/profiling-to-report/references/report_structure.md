# Report Structure

Expected final report structure with section numbering. All sections marked `[AUTO]` come from `generate_report.py` output. Sections marked `[ADD]` must be written by the AI agent. Sections marked `[ENHANCE]` use auto-generated tables but need interpretive prose added.

---

## Full outline

```
# MLIP Profiling Report                                           [AUTO] header + env table

## Table of Contents                                              [ADD]

---

## 1. Profiling Methodology                                       [ADD] from methodology.md
### 1.1 PyTorch Profiler Overview                                 [ADD]
### 1.2 Profiler Schedule                                         [ADD]
### 1.3 Instrumentation with record_function                      [ADD]
### 1.4 Latency Measurement                                       [ADD]
### 1.5 Effective Time Metric                                     [ADD]

---

## 2. Tested MLIP Models                                          [ADD] from analysis_guide.md
### 2.1 Model Overview                                            [ADD] architecture comparison table
### 2.N {Model Name}                                              [ADD] one subsection per model (2-3 paragraphs each)

---

## 3. Operation Breakdown Details                                 [ADD]
### 3.1 Leaf vs. Wrapper Operations                               [ADD] with example tree
### 3.2 Kernel-Level Classification                               [ADD] with category table

---

## 4. Profiling Results & Analysis
### 4.1 Overall Latency Comparison                                [ENHANCE]
  - Latency table                                                 [AUTO]
  - Throughput table                                              [AUTO]
  - Latency plot reference                                        [AUTO]
  - Key observations (3-5 bullets)                                [ADD]

### 4.2 Operation Breakdown                                       [ENHANCE]
  Per model config:
  - Operation table                                               [AUTO]
  - Pie chart + kernel breakdown plot refs                        [AUTO]
  - Interpretation paragraph (kernel composition, what backward pass contains)  [ADD]
  - Scaling paragraph (how ops change across atom counts)         [ADD]

### 4.3 Graph Generation: Implementation Differences              [ENHANCE]
  - Graph gen timing table                                        [AUTO]
  - Comparison prose (GPU vs CPU, scaling, implications)          [ADD]
  - Note on cueq graph gen values for MACE/SevenNet              [ADD]

### 4.4 cuEquivariance (CuEq) Acceleration                       [ENHANCE]
  - End-to-end speedup table                                     [AUTO]
  - Speedup plot reference                                        [AUTO]
  - Key findings (3-5 points with explanations)                   [ADD]
  - Per-model cueq operation breakdown tables                    [AUTO]
  - Per-model cueq pie + kernel plot refs                        [AUTO]
  - Cueq interpretation paragraphs                               [ADD]
  - Per-operation speedup tables                                  [AUTO]
  - Per-op speedup interpretation                                [ADD]
  - GPU pipeline starvation table                                 [AUTO]
  - Pipeline starvation explanation                               [ADD/ENHANCE] (auto has basic text)
  - Largest shared size comparison table                           [AUTO]
  - Comparison summary                                            [ADD]

---

## Appendix
### A. Profiling Configuration                                    [AUTO]
### B. Generated Plot Index                                       [AUTO]
### C. Reproduction                                               [AUTO]
### D. Document Export                                            [ADD] pandoc commands for PDF/HTML
```

---

## Section sizing guidelines

| Section | Target length | Content type |
|---------|--------------|--------------|
| 1. Methodology | ~120 lines | Mostly static text, adapt numbers |
| 2. Models | ~80 lines | Model descriptions + comparison table |
| 3. Operation Details | ~50 lines | Static explanation + example tree |
| 4.1 Latency | ~30 lines | Tables (auto) + 3-5 bullet observations |
| 4.2 Operations | ~40 lines per config | Table (auto) + 2 paragraphs each |
| 4.3 Graph Gen | ~40 lines | Table (auto) + comparison prose |
| 4.4 CuEq | ~80 lines | Tables (auto) + analysis paragraphs |
| Appendix | ~40 lines | Auto-generated + Document Export |
| **Total** | ~400-450 lines | |

---

## How to merge auto-generated and added content

1. **Start from auto-generated report**: Copy it as the base
2. **Insert Sections 1-3 before the data sections**: These are entirely new sections
3. **Add Table of Contents after the title**
4. **Wrap each auto-generated section with a proper heading**: Add `## 4. Profiling Results & Analysis` before the first data section
5. **Add interpretive paragraphs after each auto-generated table/plot**: Place analysis immediately after the relevant plots, before the next subsection
6. **Preserve all auto-generated tables exactly**: Do not modify numbers, column alignment, or plot references
7. **Add section dividers (`---`)** between major sections (after 1, 2, 3)

## TOC anchor format

TOC links must use heading text **without** section numbers. Markdown renderers strip leading numbers when generating anchors.

```markdown
## Table of Contents

1. [Profiling Methodology](#profiling-methodology)
2. [Tested MLIP Models](#tested-mlip-models)
3. [Operation Breakdown Details](#operation-breakdown-details)
4. [Profiling Results & Analysis](#profiling-results-analysis)
```

Note: `&` in "Results & Analysis" is stripped → `#profiling-results-analysis`.

## Appendix D template

```markdown
### D. Document Export

\`\`\`bash
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
\`\`\`
```

Adjust paths if the results directory depth changes. The `pandoc/` directory at the project root contains shared templates.
