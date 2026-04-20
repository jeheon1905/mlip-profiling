# Inference guide

Infer the deck shape from the report instead of asking the user to label the presentation.

## Typical signals

### Decision memo / recommendation
Use a sequence like:
1. Recommendation up front
2. Problem or opportunity
3. Evidence
4. Options or tradeoffs
5. Recommended plan
6. Next steps

### Research or technical result
Use a sequence like:
1. Objective
2. Method or setup
3. Key findings
4. Interpretation
5. Implication
6. Next steps

### Status or progress report
Use a sequence like:
1. Overall status
2. What changed
3. Evidence or metrics
4. Risks and blockers
5. Upcoming milestones

### Proposal or plan
Use a sequence like:
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

For reports that compare multiple ML models or system configurations, use this expanded sequence:

1. Main takeaway / headline finding (surface it early, before setup)
2. Hardware and environment (table)
3. Model or subject comparison (table — one column per subject)
4. Measurement methodology (table — schedule, metrics, definitions)
5. Metric definition or classification scheme (table)
6. Primary results table (table-only — all configurations, all sizes)
7. Throughput or secondary metric table (table-only)
8. Scaling charts (image+text — one takeaway per visual)
9. Per-subject baseline analysis (one image+text slide per subject)
10. Acceleration or variant analysis (one image+text slide per subject per backend)
11. Per-operation speedup tables (one table slide per subject)
12. Cross-cutting comparison (graph gen, kernel analysis, etc.) (table or image+text)
13. Root cause / mechanism analysis (image+text)
14. Inter-kernel or trace-level data (table)
15. Recommendations / next steps (bullets)

These are starting estimates, not hard ceilings. Let the source content drive the count: if the report has 6 result subsections, produce 6 result slides.

## Prioritization rule

If content is too long, cut prose and background detail — not entire report sections. Omitting a whole section to hit a slide count target is always wrong. Every top-level (`##`) section of the source report must map to at least one slide (Appendix sections may be omitted).
