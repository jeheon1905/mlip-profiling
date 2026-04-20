# Quality checklist

Review before delivering.

## Structure
- For 6+ content slides: is a Table of Contents slide present as the second slide?
- Does the deck include setup/environment context (hardware, software, methodology)?
- For reports comparing multiple subjects: is a comparison table slide included?
- For technical reports with numeric results: are data tables included alongside charts?
- For benchmark/profiling reports specifically:
  - Is the hardware and environment reproduced as a table slide?
  - Is the model/subject comparison reproduced as a table slide?
  - Is the profiling methodology (schedule, measurement method) a table slide?
  - Is the metric definition (e.g., effective time classification) a table slide?
  - Are **all** per-configuration latency/throughput numbers in a table slide (not just charts)?
  - Is there a per-operation speedup table for each model or comparison pair?
  - Is there a separate analysis slide per model per backend (not merged to reduce count)?

## Narrative
- Does the first content slide surface the main takeaway early?
- Does every slide have exactly one message?
- Does the sequence feel like a spoken presentation rather than a written report?

## Slide canvas
- Are most titles one line and sentence-style?
- Are bullets short, parallel, and limited to 3 to 5?
- Are tables compact (max 6 rows × 4 columns) with brief cell content?
- Are heavy tables split into smaller tables or converted to structured bullets?
- Are visual placeholders used only when necessary and clearly marked?
- Do image+text slides have a full 3–5 bullets in addition to the images? (image-only and empty-bullet slides look sparse)
- For image+text slides: are the bullets quantitative and interpretive, not just repeating the image alt text?

## Notes
- Does the title slide include YAML notes?
- Does every content slide include a `::: notes` block?
- Do notes add context instead of repeating bullets?

## Images and plots
- Are relevant plots from the source report included as `![alt](path)` references?
- Do all referenced image files exist at the specified paths?
- Is each image slide focused on one visual with a takeaway title?
- Are side-by-side images using column layouts rather than stacked vertically?

## Template and design
- Was a styled template generated with `python scripts/create_template.py template.pptx`?
- Was the template passed to pandoc via `--reference-doc` (third argument to render script)?
- Does the rendered pptx have consistent fonts and colors (not plain pandoc defaults)?

## Post-processing
- Was `python scripts/postprocess_slides.py slides.pptx slides.md slides.pptx` run after rendering?
- Are multi-image slides showing all images in a grid (not split across separate slides)?
- Are image aspect ratios preserved (no stretching)?
- Were orphan slide notes transferred to the parent slide?

## Output
- Does `python scripts/validate_slides.py slides.md` pass?
- Does `slides.pptx` render successfully with the template?
- If Quarto was requested, was `slides.qmd` created only as an auxiliary file while `slides.md` remained canonical?
