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
