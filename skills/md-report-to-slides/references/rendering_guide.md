# Rendering guide

## Template generation

Before rendering, generate a styled reference template for consistent fonts, colors, and sizes:

```bash
python scripts/create_template.py template.pptx
```

Place `template.pptx` in the same directory as `slides.md`. The render script auto-detects it. To customise the design, edit the `THEME` dict in `create_template.py`.

## Default path

Prefer Pandoc because the required canonical source is `slides.md` and the required output is an editable `slides.pptx`.

```bash
bash scripts/render_pandoc.sh slides.md slides.pptx
```

The render script:
- Auto-detects `template.pptx` or `reference.pptx` next to the input file
- Sets `--resource-path` to the input directory so relative image paths resolve correctly
- Accepts an explicit template as third argument: `bash scripts/render_pandoc.sh slides.md slides.pptx template.pptx`

## Post-processing for multi-image slides

Pandoc (v2.x) puts each `![alt](path)` on a separate slide, even when multiple images appear under one heading. After rendering, run the postprocess script to merge them back:

```bash
python scripts/postprocess_slides.py slides.pptx slides.md slides.pptx
```

The script:
- Parses `slides.md` to know which images belong to which slide title
- Detects orphan slides (image-only slides pandoc split off) and removes them
- Lays out images in a grid on the parent slide with correct aspect ratios (using PIL)
- Transfers notes from orphan slides to the parent
- Handles smart-quote title matching (pandoc converts `'` to `\u2019`)

## Images

Pandoc embeds `![alt](path)` images directly into the pptx. Ensure:
- Image paths are relative to the `slides.md` location
- All referenced files exist before rendering
- Use `.png` for plots (universally supported)

## Optional Quarto path

Use Quarto when requested or when the slide source is easier to maintain in `.qmd` form.

```bash
python scripts/md_to_qmd.py slides.md slides.qmd
bash scripts/render_quarto.sh slides.qmd
```

## Failure handling

If rendering fails:

1. Re-run `python scripts/validate_slides.py slides.md`.
2. Inspect heading levels, fenced divs, and notes blocks.
3. Simplify unusual markdown constructs.
4. Retry Pandoc.
5. Use Quarto only if it is installed and clearly beneficial.

## Notes preservation

- Title-slide notes belong in YAML as `notes: |`.
- Content-slide notes belong in `::: notes` blocks.
- Keep notes concise and speakable.
