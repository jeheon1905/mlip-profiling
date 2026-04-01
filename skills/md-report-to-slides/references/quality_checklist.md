# Quality checklist

Review before delivering.

## Narrative
- Does the first content slide surface the main takeaway early?
- Does every slide have exactly one message?
- Does the sequence feel like a spoken presentation rather than a written report?

## Slide canvas
- Are most titles one line and sentence-style?
- Are bullets short, parallel, and limited to 3 to 5?
- Are heavy tables or long paragraphs removed?
- Are visual placeholders used only when necessary and clearly marked?

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
