# Inference guide

Infer the deck shape from the report instead of asking the user to label the presentation.

## Typical signals

### Decision memo / recommendation
1. Recommendation up front
2. Problem or opportunity
3. Evidence
4. Options or tradeoffs
5. Recommended plan
6. Next steps

### Research or technical result
1. Objective
2. Method or setup
3. Key findings
4. Interpretation
5. Implication
6. Next steps

### Status or progress report
1. Overall status
2. What changed
3. Evidence or metrics
4. Risks and blockers
5. Upcoming milestones

### Proposal or plan
1. Why change now
2. Current state
3. Proposed approach
4. Benefits and tradeoffs
5. Rollout plan
6. Ask or approval needed

## Slide count guidance

| Report type | Target range | Notes |
|---|---|---|
| Short report (≤ 3 main sections) | 6 – 8 | |
| Typical report (3 – 5 main sections) | 8 – 14 | |
| Long or data-heavy (5+ sections) | 14 – 20 | Common for benchmark and profiling reports |
| Multi-model / multi-phase benchmark | 1 – 2 slides per model per analysis phase | Do not merge subjects to hit a count target |

### Benchmark / profiling report shape

For reports comparing multiple ML models or system configurations:

1. Main takeaway / headline finding (surface early)
2. Hardware and environment (table slide)
3. Model or subject comparison (table — one column per subject)
4. Measurement methodology (table — schedule, metrics, definitions)
5. Metric definition or classification scheme (table)
6. Primary results table (all configurations, all sizes)
7. Throughput or secondary metric table
8. Scaling charts (image+text — one takeaway per visual)
9. Per-subject baseline analysis (one image+text slide per subject)
10. Acceleration or variant analysis (one image+text slide per subject per backend)
11. Per-operation speedup tables (one slide per subject)
12. Cross-cutting comparison (graph gen, kernel analysis)
13. Root cause / mechanism analysis (image+text)
14. Inter-kernel or trace-level data (table)
15. Recommendations / next steps (bullets)

These are starting estimates, not hard ceilings. Let the source content drive the count.

## Why HTML lifts pandoc-era count pressure

The PPTX skill split many "image + text" or "table + bullet" combinations across multiple slides because pandoc could not render them on one canvas. In HTML, those combinations live on a single slide. Expect the HTML deck to be **slightly shorter** than the PPTX equivalent for the same source, because logically related content can collapse into one richer slide.

## Prioritization rule

If content is too long, cut prose and background detail — not entire report sections. Omitting a whole section to hit a slide count target is always wrong. Every top-level (`##`) section of the source report must map to at least one slide (Appendix sections may be omitted).
