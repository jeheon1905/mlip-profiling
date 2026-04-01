# Pandoc Configuration

Pandoc templates and styles for converting Markdown to PDF/HTML.

## Files

| File | Purpose | Used with |
|------|---------|-----------|
| `latex-header.tex` | LaTeX customization (image sizing, table wrapping, code blocks) | `--pdf-engine=xelatex` |
| `style.css` | CSS styling for HTML/weasyprint output | `--css` |

## Usage

### 1. PDF Generation (xelatex)

Best for: academic-quality output, good typography, proper page breaks.

```bash
cd results/<result_dir>
pandoc profiling_report.md -o profiling_report.pdf \
  --pdf-engine=xelatex \
  -V mainfont="FreeSerif" \
  -V sansfont="FreeSans" \
  -V monofont="DejaVu Sans Mono" \
  -V geometry:"margin=2cm" \
  -V fontsize=10pt \
  --highlight-style=tango \
  -H ../../pandoc/latex-header.tex
```

**Requirements**: `xelatex` (from TeX Live or similar)

### 2. HTML Generation

Best for: web viewing, easy sharing, browser print-to-PDF.

```bash
cd results/<result_dir>
pandoc profiling_report.md -o profiling_report.html \
  --standalone \
  --self-contained \
  --css=../../pandoc/style.css
```

Then open in browser or print to PDF.

**Optional - weasyprint PDF**:
```bash
pandoc profiling_report.md -o profiling_report.pdf \
  --pdf-engine=weasyprint \
  --css=../../pandoc/style.css
```

## Customization

### Image Size (LaTeX)

Edit `latex-header.tex`:
```tex
\def\maxwidth{0.6\linewidth}   % 60% of page width
\def\maxheight{0.5\textheight} % 50% of page height
```

### Font Size (CSS)

Edit `style.css`:
```css
body { font-size: 10pt; }
code, pre { font-size: 9pt; }
table { font-size: 9pt; }
```

## Available Fonts

Check installed fonts with:
```bash
fc-list : family | grep -iE "liberation|freesans|dejavu|noto" | sort -u
```

Common Unicode-compatible fonts:
- **Serif**: FreeSerif, Liberation Serif, Noto Serif
- **Sans**: FreeSans, Liberation Sans, DejaVu Sans, Noto Sans  
- **Mono**: DejaVu Sans Mono, Liberation Mono, Noto Sans Mono
