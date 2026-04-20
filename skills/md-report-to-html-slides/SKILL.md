---
name: md-report-to-html-slides
description: convert one completed markdown report into a single self-contained reveal.js HTML presentation. use when the user provides a finished .md report and wants the AI agent to infer the presentation mode, rebuild the report into a speaking-first slide narrative, write slides.html directly (no markdown intermediate), validate it, and ship a single portable HTML file (reveal.js, CSS, and images all inlined) with presenter notes that opens offline in any browser.
---

# Md Report To HTML Slides

Convert a single completed markdown report into a speaking-first **HTML** slide deck (reveal.js). The output is a **single self-contained `.html` file** — reveal.js, CSS, and images are all inlined. The user can download it, email it, or drop it on any web server and it just works. The AI writes `slides.html` directly — there is no markdown intermediate. Include presenter notes for the title slide and every content slide.

> **Scope**: This is a **general-purpose** skill — it works with any completed markdown report.
>
> **Why HTML, not pptx?** Pandoc → pptx imposes restrictions (no bullets+table, no image+table, fragile multi-image layout) and requires post-processing hacks. Authoring HTML directly removes those constraints and gives the AI full control over layout via CSS grid.
>
> **Why single-file?** A presentation with sibling `reveal/` and `plots/` directories breaks the moment someone downloads or moves the `.html` alone. Inlining everything makes the deck truly portable.

## Script paths

All scripts live under this skill's `scripts/` directory. From the project root:

```bash
SKILL_DIR=skills/md-report-to-html-slides
python $SKILL_DIR/scripts/create_html_shell.py path/to/slides.html [--theme white|simple] [--title "Deck title"]
python $SKILL_DIR/scripts/validate_html_slides.py path/to/slides.html [--max 20]
python $SKILL_DIR/scripts/pack_html_slides.py path/to/slides.html --in-place
```

- `create_html_shell.py` writes a self-contained shell (reveal.js + CSS inlined; ~1 MB). No sidecar `reveal/` directory is created.
- `pack_html_slides.py` base64-embeds all `<img src="…">` paths so the final file has zero external dependencies.

## Default workflow

1. Read the markdown report. Infer the deck goal from the report itself.
2. Use [references/inference_guide.md](references/inference_guide.md) to choose a deck shape and target slide count.
3. Build a **section map**: for every `##` heading in the source, note which slide title will cover it (or why it is deliberately omitted — only Appendix-level material may be omitted).
4. Run `create_html_shell.py path/to/slides.html --title "Deck title"` to produce an empty self-contained reveal.js shell.
5. Open `slides.html` and write `<section>` elements directly inside `<div class="slides">`, using the patterns in [references/slide_blueprint.md](references/slide_blueprint.md). Include `<aside class="notes">` on every slide.
6. Reference plots with relative `<img src="...">` paths from where `slides.html` lives. The pack step will inline them later.
7. Run `python scripts/validate_html_slides.py slides.html` and fix any **FAIL** items. Review WARN items.
8. Apply the [references/quality_checklist.md](references/quality_checklist.md) review.
9. Run `python scripts/pack_html_slides.py slides.html --in-place` to embed all images as base64 data URLs. The file now has no external dependencies.
10. Re-validate to confirm "image(s) embedded as data: URLs" appears in the report.
11. Sanity-check by opening `slides.html` in a browser (arrow keys to navigate, `S` for speaker view).
12. Return `slides.html` (single portable file) and a short note on the inferred presentation mode plus any major assumptions.

## Interpretation rules

- Treat the source markdown as a completed report, not raw notes.
- Rebuild the content for speaking, not for reading.
- Prefer one message per slide.
- Surface the report's own key findings early (typically the report's own "Key observations" or "Key findings" lists).
- Remove repetition, background detail, and appendix-style prose unless needed for the narrative.
- Prefer structured slides over bullet walls: comparison, table, image+text.
- Keep only the most decision-relevant numbers.
- If a visual is clearly needed but source data is insufficient, insert a short placeholder such as `<div class="callout warn">[visual: 3-step process needed]</div>` rather than inventing numbers.
- Do not ask the user to classify the presentation type unless the report is genuinely ambiguous.

## Faithfulness principle (strict)

**The slides are a condensation of the source report, not a new document.** Everything on every slide — titles, bullets, tables, callouts, speaker notes — must be directly traceable to the source report.

Do **not**:

- **Invent recommendations, next steps, or action items** the report does not contain. If the report has no "Recommendations" or "Next steps" section, do not fabricate one. A "Summary of findings" slide drawn from the report's own "Key observations" or "Key findings" is the correct closing slide; a "Recommendation" slide is not.
- **Add deployment advice, installation tips, or optimization suggestions** ("use X for production", "keep an e3nn fallback", "next optimization target: Y") unless the report states them.
- **Editorialize titles** with words like *decisively*, *flattens*, *pulls ahead*, *the answer*, *production-ready*. Prefer factual section-style titles ("Latency vs atom count", "MACE (e3nn) — 500 atoms breakdown") or direct paraphrases of the report's headings ("Graph generation: implementation differences").
- **Extrapolate beyond the data**. If the report does not measure MACE at 2,916 atoms e3nn, do not speculate about its "extrapolated speedup".
- **Write speaker notes that promote the narrative** ("lead with the answer", "most important comparison", "end with the ask"). Notes should restate or paraphrase the report's own analysis — typically pointing to the specific section of the report the slide is drawn from.

Do:

- Prefer factual section-style titles and direct paraphrases of the report's language.
- Use the report's own summary lists (Key observations, Key findings, Implications) as the source for summary/headline slides.
- In speaker notes, cite the source section when useful ("From Section 4.2 of the report").
- If a fact would strengthen the narrative but is not in the report, **drop it** rather than invent it.

The test: for every slide, you should be able to point to the exact report section(s) the content came from. If you can't, the content does not belong on the slide.

## Images and plots

- **Scan all `![](path)` references in the source report** and include each relevant figure as `<img src="...">` in `slides.html`. Do not silently omit figures.
- Use **relative paths** from where `slides.html` lives. Validator will fail if a path does not resolve.
- Prefer one image per slide (or paired comparison via `.cols`) with a takeaway title.
- Use `<figcaption>` inside `<figure class="fig">` for short captions.
- Verify all referenced image files exist before validating.
- The deck must work offline — no external CDN images.

## Generic design rules

Apply a neutral style by default:

- Keep the title slide minimal (use the `title-slide` class).
- Use strong whitespace.
- Keep most titles on one line.
- Keep bullet lists ≤ 6 items, parallel and short.
- Prefer 2-column layouts (`.cols`, `.cols-1-2`, `.cols-2-1`) for contrast and image+text slides.
- Use compact tables for naturally tabular data — HTML tables have no row/column limit.
- **HTML lifts the PPTX restrictions**: bullets + table, image + table, multiple stacked tables, custom column ratios are all valid on a single slide.
- Put nuance and transitions into `<aside class="notes">`, not onto the slide canvas.
- **Slide density**: every content slide should feel full. Image+text slides need 3–5 substantive interpretive bullets in addition to the image(s).

## Validation and repair

Before delivering, validate `slides.html`. Common repair actions:

- Replace generic titles such as `overview` or `results` with takeaway titles.
- Split slides that mix multiple messages.
- Add missing `<aside class="notes">` blocks.
- Fix broken image paths reported by the validator.
- Remove empty `<section>` shells.

Use [references/quality_checklist.md](references/quality_checklist.md) for the final review.

## Required deliverables

Always return:

- `slides.html` — a **single self-contained file** with reveal.js, CSS, and images all inlined
- presenter notes embedded in `<aside class="notes">` on every slide

No sidecar `reveal/` directory and no external image references — verified by the validator.

## Quick prompt pattern

If the user gives only a report file, proceed without asking for deck type. Internally apply this framing:

> Infer the most likely presentation mode from the report, convert it into an 8 to 16 slide speaking-first HTML deck, keep one message per slide, preserve only decision-relevant evidence, and include concise presenter notes for every slide.

## Source section mapping

Before writing a single slide, build an explicit section map. For every `##` heading in the source report, decide: which slide title will cover it, or why it is deliberately omitted (only Appendix-level material may be omitted). Write this map as a scratch list, then drive the slide outline from it.

**Default mapping rules for technical reports:**

| Source section type | Slide pattern | Notes |
|---|---|---|
| Setup / hardware environment | Table slide | One row per component; GPU, CPU, framework versions |
| Methodology / measurement approach | Table slide | Schedule, metrics, important caveats |
| Metric definition or classification scheme | Table slide | e.g., effective-time classification |
| Subject overview / comparison across subjects | Table slide | One column per subject, rows = key attributes |
| Per-subject analysis (per-model, per-experiment) | Image + text slide (`.cols-1-2`) | Pie/breakdown plot + 3–5 bullets; **one slide per subject per backend/phase** |
| Quantitative results table | Table slide | Reproduce numeric tables verbatim; never merge or drop rows |
| Chart / figure | Image slide or image+text | One figure per slide; side-by-side via `.cols` for direct comparisons |
| Root cause / mechanism analysis | Image + text slide | Trace plots + interpretive bullets |
| Speedup or performance gain summary | Table slide | Rows = operations or configs, cols = baseline/variant/speedup |
| Recommendations / next steps | Bullet + callout slide | Max 5 bullets; lead with action verb; wrap headline ask in `.callout.good` |

**Per-subject per-phase rule**: If the report analyzes N subjects (models, systems) each with M phases (backends, configurations), produce at minimum N × M analysis slides. Never merge subjects to reduce count.

**Figure coverage rule**: Scan every `![alt](path)` reference in the source report. Every plot illustrating a distinct result must appear in `slides.html`. Group plots side-by-side only when comparing the same subject across exactly two conditions.

**Quantitative table rule**: Every multi-row numeric table in the report containing decision-relevant data (latencies, speedups, throughputs) must be reproduced as a table slide. Do not paraphrase numeric tables into bullets — the numbers are the evidence.

**Omission rules**: The following may be omitted without justification:
- Appendix sections (reproduction steps, configuration dumps, document export instructions)
- Code listings that illustrate instrumentation rather than results
- Prose that merely restates a table or figure already included as a slide

Everything else requires an explicit reason to drop.
