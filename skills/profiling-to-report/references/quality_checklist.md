# Quality Checklist

Review the completed report against this checklist before delivering.

---

## Structural checks

- [ ] **Table of Contents** present and matches all section headings
- [ ] **TOC anchors correct**: Links use heading text without section numbers (e.g., `#profiling-methodology` not `#1-profiling-methodology`)
- [ ] **Sections 1-3** (Methodology, Models, Operation Details) appear before data sections
- [ ] **Section 4** contains all auto-generated tables and plot references
- [ ] **Appendix** (A, B, C, D) present at the end — D includes pandoc export commands
- [ ] **Section dividers** (`---`) separate major sections (after 1, 2, 3)
- [ ] **External hyperlinks**: Key tools and model repos linked on first mention (PyTorch Profiler, Perfetto, model GitHub repos, matscipy, e3nn, cuEquivariance)

## Data integrity

- [ ] **No invented numbers**: Every number in prose matches a value from auto-generated tables or summary.json
- [ ] **All auto-generated tables preserved**: Tables have not been modified (same numbers, same columns)
- [ ] **All plot references preserved**: Every `![...](...png)` from the auto-generated report is in the final report
- [ ] **OOM marked correctly**: Models that OOM at certain sizes show "OOM" not blank or N/A
- [ ] **Profiler settings match**: Wait/warmup/active and timeit number/repeat match summary.json values

## Analytical quality

- [ ] **Every plot has interpretation**: At least one sentence of analysis accompanies each plot reference
- [ ] **Backward pass explained**: For each model's operation breakdown, explain that `compute_forces`/`force_output` is a monolithic autograd traversal
- [ ] **Scaling quantified**: Scaling paragraphs include actual multipliers (e.g., "12× for 12.7× atoms")
- [ ] **CPU vs GPU clarified**: Graph generation explicitly states whether it runs on CPU or GPU
- [ ] **CuEq mechanism explained**: Not just "faster" but why (tensor product fusion, kernel launch overhead, Amdahl's law)
- [ ] **Pipeline starvation explained**: Root cause (GPU idle due to CPU autograd dispatch) with data support
- [ ] **Percentages contextualized**: Denominators stated (e.g., "of leaf traced time")

## Writing quality

- [ ] **No superlatives without numbers**: Replace "very fast" with actual ms or speedup values
- [ ] **Consistent terminology**: Use the same operation names throughout (e.g., always `compute_forces` not "backward pass" and "force computation" interchangeably)
- [ ] **Present tense** for behavior descriptions
- [ ] **Bold** key findings in first sentence of analytical paragraphs
- [ ] **Concise**: No filler sentences. Most paragraphs are 2-4 sentences.
- [ ] **No repetition**: Information stated once in the correct section, not repeated

## Cross-reference checks

- [ ] **Model names match**: Model names in prose match the table headers
- [ ] **Atom counts consistent**: Same atom counts used in tables and scaling discussion
- [ ] **Backend labels consistent**: "e3nn" and "cueq" (lowercase) in data, proper case in prose only when starting a sentence
- [ ] **Plot filenames valid**: Plot references use the naming convention from `generate_report.py` (currently: `{model_type}_{backend}_{dir_model_name}_{N}atoms_{type}.png`). Verify against actual files in `{results_dir}/plots/`.
