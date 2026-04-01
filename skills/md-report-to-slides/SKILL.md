---
name: md-report-to-slides
description: convert one completed markdown report into a clean presentation workflow that produces slides.md, an editable pptx, and presenter notes. use when the user provides a finished .md report and wants the AI agent to infer the presentation mode, rebuild the report into a speaking-first slide narrative, apply generic presentation design rules, validate the slide source, and render through pandoc by default or quarto when requested or helpful.
---

# Md Report To Slides

Convert a single completed markdown report into a speaking-first slide deck. Always write `slides.md` first, validate it, then render `slides.pptx`. Include presenter notes for the title slide and every content slide.

> **Scope**: This is a **general-purpose** skill — it works with any completed markdown report, not just this project's profiling reports.

## Script paths

All scripts live under this skill's `scripts/` directory. When invoking them, use the full path from the project root:

```bash
SKILL_DIR=skills/md-report-to-slides
python $SKILL_DIR/scripts/validate_slides.py slides.md
bash $SKILL_DIR/scripts/render_pandoc.sh slides.md slides.pptx
```

The examples below use short names for readability, but always resolve to `skills/md-report-to-slides/scripts/`.

## Default workflow

1. Read the markdown report and infer the deck goal from the report itself.
2. Use [references/inference_guide.md](references/inference_guide.md) to choose a deck shape and slide count.
3. Rebuild the report into a compact storyline: situation, evidence, implication, recommendation, next step.
4. Write `slides.md` using [references/slide_blueprint.md](references/slide_blueprint.md). Include plot images from the report as `![alt](path)` references — see the **Images and plots** section below.
5. Run `python scripts/validate_slides.py slides.md` and fix any structural issues before rendering.
6. Generate a styled template: `python scripts/create_template.py template.pptx`. Place the template next to `slides.md` — the render script auto-detects it.
7. Render `slides.pptx` with `bash scripts/render_pandoc.sh slides.md slides.pptx` (template is auto-detected) or pass it explicitly: `bash scripts/render_pandoc.sh slides.md slides.pptx template.pptx`.
8. Post-process for multi-image slides: `python scripts/postprocess_slides.py slides.pptx slides.md slides.pptx`. This merges orphan slides (pandoc puts each image on a separate slide), lays out multiple images in a grid with correct aspect ratios, and transfers notes from orphan slides.
9. (Optional) If the user explicitly asks for Quarto, create `slides.qmd` with `python scripts/md_to_qmd.py slides.md slides.qmd`, then run `bash scripts/render_quarto.sh slides.qmd`. If Quarto is not installed, fall back to Pandoc and inform the user.
10. Return `slides.md`, `slides.pptx`, and a short note explaining the inferred presentation mode plus any major assumptions.

## Interpretation rules

- Treat the source markdown as a completed report, not raw notes.
- Rebuild the content for speaking, not for reading.
- Prefer one message per slide.
- Use sentence-style titles that already communicate the takeaway.
- Surface the "so what" early.
- Remove repetition, background detail, and appendix-style prose unless needed for the narrative.
- Prefer structured slides over bullet walls: comparison, process, timeline, drivers, options, recommendation, next steps.
- Keep only the most decision-relevant numbers.
- If a visual is clearly needed but source data is insufficient, insert a short placeholder such as `[visual: 3-step process]` rather than inventing numbers.
- Do not ask the user to classify the presentation type unless the report is genuinely too ambiguous to infer.

## Images and plots

When the source report references or relies on images/plots, include them in slides for visual evidence:

- Use standard markdown image syntax: `![alt text](relative/path/to/image.png)`
- Use **relative paths** from where `slides.md` lives. The render script sets `--resource-path` automatically.
- Prefer one image per slide with a takeaway title.
- For side-by-side comparison, use column layouts (see [slide_blueprint.md](references/slide_blueprint.md)).
- Verify all referenced image files exist before rendering — missing images produce empty slides.
- Pandoc embeds images directly into the pptx, so recipients don't need the source files.

## Generic design rules

Apply a neutral style by default:

- Keep the title slide minimal.
- Use strong whitespace.
- Keep most titles on one line.
- Keep most slides to 3 to 5 bullets.
- Keep bullets short and parallel.
- Prefer 2-column layouts for contrast and process/timeline layouts for sequence.
- Avoid decorative labels, badge collections, dense dashboards, and long tables.
- Put nuance and transitions into notes, not onto the slide canvas.
- **Avoid markdown tables** — pandoc splits slides with tables when content overflows. Convert tabular data to structured bullet lists instead.

## Validation and repair

Before rendering, validate `slides.md`.

Common repair actions:
- Replace generic titles such as `overview` or `results` with takeaway titles.
- Split slides that exceed 5 bullets or mix multiple messages.
- Add missing title-slide notes in YAML and missing content-slide notes blocks.
- Remove malformed fenced divs or unsupported markdown constructs.

Use [references/quality_checklist.md](references/quality_checklist.md) for final review.

## Rendering choices

### Template generation

Generate a styled reference template before rendering. This controls fonts, colors, and layout in the output pptx:

```bash
python scripts/create_template.py template.pptx
```

Place `template.pptx` next to `slides.md`. The render script auto-detects `template.pptx` or `reference.pptx` in the same directory. Alternatively, pass it as the third argument.

Edit the `THEME` dict in `create_template.py` to customize colors, fonts, and sizes.

### Pandoc path

Use Pandoc as the default. Run:

```bash
bash scripts/render_pandoc.sh slides.md slides.pptx
```

The script automatically:
- Detects `template.pptx` or `reference.pptx` next to the input file
- Sets `--resource-path` to the input directory for image resolution

To pass a template explicitly:

```bash
bash scripts/render_pandoc.sh slides.md slides.pptx template.pptx
```

### Quarto path

Use Quarto only when requested or when its source syntax is materially clearer for the needed slide structure.

```bash
python scripts/md_to_qmd.py slides.md slides.qmd
bash scripts/render_quarto.sh slides.qmd
```

If Quarto is unavailable, fall back to Pandoc and say so.

## Required deliverables

Always return:

- `slides.md`
- `slides.pptx`
- presenter notes embedded in the source and preserved in the rendered deck

## Quick prompt pattern

If the user gives only a report file, proceed without asking for deck type. Internally apply this framing:

> Infer the most likely presentation mode from the report, convert it into an 8 to 12 slide speaking-first deck, keep one message per slide, preserve only decision-relevant evidence, and include concise presenter notes for every slide.
