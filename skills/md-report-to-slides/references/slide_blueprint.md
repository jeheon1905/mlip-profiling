# Slide blueprint

Use this as the canonical structure for `slides.md`.

## Required skeleton

```md
---
title: Presentation Title
author: Author Name
notes: |
  Opening remarks for the title slide.
  Keep this natural and brief.
---

# Optional section divider

## The main takeaway appears in the slide title
- Short bullet
- Short bullet
- Short bullet

::: notes
What to say on this slide.
Add nuance, implication, or transition.
:::
```

## Source rules

- Use level 1 headings only for optional section dividers.
- Use level 2 headings for normal slides.
- Write sentence-style titles.
- Keep the source human-editable.
- Put title-slide speaker notes in YAML under `notes`.
- Put content-slide notes in a `::: notes` block immediately after the slide content.
- Prefer no more than 5 bullets on a slide.

## Table of contents slide

Add as the second slide (after the title) for decks with 6 or more content slides:

```md
## Table of Contents

- **Section 1**: brief description of what is covered
- **Section 2**: brief description
- **Section 3**: brief description

::: notes
Brief orientation: how many sections, which is most important.
:::
```

Keep items to 5 or fewer. Use bold to label the section name, then a colon and short description.

## Table slide

Use markdown tables for naturally tabular data. Keep tables short to avoid pandoc overflow:

```md
## Model Architecture Comparison

| | Model A | Model B | Model C |
|---|---|---|---|
| **Parameter 1** | value | value | value |
| **Parameter 2** | value | value | value |
| **Parameter 3** | value | value | value |

::: notes
Explain the most important differences between columns.
:::
```

Rules for tables:
- Max 6 rows (data rows), max 4 columns per table. Wider or taller tables risk pandoc splitting the slide.
- **Multiple tables per slide are supported**: list two markdown tables in sequence; `postprocess_slides.py` merges the pandoc-generated continuation slide back and stacks both tables vertically. Use this for thematically related data (e.g., latency + throughput, library comparison + scaling).
- **CRITICAL**: Never put bullets and a table on the same slide. Pandoc always moves the table to a separate continuation slide with no title. Table slides must contain only tables (one or more) — no bullets.
- On image+text slides, use structured bullet lists in the lower zone — never a table (not supported in combined layout).
- Use `---:` for right-aligned numeric columns.
- Bold the first-column labels in comparison tables.
- Avoid empty cells — use `--` or `N/A` as a placeholder.

### Multi-table slide

```md
## Latency (ms) and throughput (ns/day) both confirm cueq advantage

| Config | 108 atoms | 500 atoms | 1,372 atoms |
|--------|----------:|----------:|------------:|
| **MACE e3nn** | 40.2 | 106.5 | 286.6 |
| **MACE cueq** | 45.4 | 48.7 | 76.0 |

| Config | 108 atoms | 500 atoms | 1,372 atoms |
|--------|----------:|----------:|------------:|
| **MACE e3nn** | 2.15 | 0.81 | 0.30 |
| **MACE cueq** | 1.90 | 1.78 | 1.14 |

::: notes
Both tables support the same conclusion — include both because they measure different things.
postprocess_slides.py detects the second table as a pandoc continuation orphan and stacks them.
:::
```

## Preferred slide types

### Insight slide

```md
## The main finding is visible in the title
- Evidence 1
- Evidence 2
- Implication

::: notes
Explain why the finding matters now.
:::
```

### Comparison slide

```md
## Option B is preferred because it improves speed with manageable risk

:::: {.columns}
::: {.column}
### Option A
- Slower
- Higher setup burden
:::

::: {.column}
### Option B
- Faster
- Lower setup burden
:::
::::

::: notes
State the decision logic explicitly.
:::
```

### Process slide

```md
## The rollout works best as a three-step process
1. Prepare inputs
2. Pilot on a limited scope
3. Scale after review

::: notes
Explain the dependency between steps.
:::
```

### Timeline slide

```md
## The work can be phased across three milestones
- Phase 1: baseline and setup
- Phase 2: pilot and validation
- Phase 3: rollout and monitoring

::: notes
Highlight timing assumptions and gates.
:::
```

### Image / plot slide

Use `![alt text](path/to/image.png)` to embed images. Pandoc converts these into full-slide images in the pptx. Place one image per slide for maximum impact.

```md
## Throughput scales linearly with system size

![Throughput comparison across models](plots/model_comparison_throughput.png)

::: notes
Walk the audience through the x-axis (atom count) and highlight the crossover point.
:::
```

For side-by-side images, use a column layout:

```md
## Backend acceleration varies by model architecture

:::: {.columns}
::: {.column}
![MACE breakdown](plots/mace_e3nn_breakdown.png)
:::

::: {.column}
![SevenNet breakdown](plots/sevenn_e3nn_breakdown.png)
:::
::::

::: notes
Compare the dominance of tensor products in both models.
:::
```

**Image path rules:**

- Use relative paths from where `slides.md` lives.
- Prefer `.png` for plots and `.svg` where supported.
- Pandoc embeds images into the pptx, so the file must exist at render time.
- Keep image filenames descriptive — they become alt text fallback.

### Recommendation slide

```md
## The recommended action is to launch a controlled pilot this quarter
- Start with the highest-value scope
- Track 3 success metrics
- Review after the first cycle

::: notes
End with the ask, owner, and timing.
:::
```

## What to avoid

- Titles like `overview`, `analysis`, `results`, or `conclusion` with no takeaway
- More than 5 dense bullets
- Paragraph blocks copied from the report
- Notes that merely repeat slide bullets
- Fake figures, fake charts, or invented evidence
- Broken image paths (always verify images exist before rendering)
- **Markdown tables** — pandoc splits slides when tables + bullets overflow; convert to structured bullet lists instead
