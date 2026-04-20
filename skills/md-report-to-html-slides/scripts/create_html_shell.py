#!/usr/bin/env python3
"""Create a self-contained reveal.js slide shell (single HTML file, no sidecar dirs).

Usage:
    python create_html_shell.py <output.html> [--theme white|simple] [--title "Deck title"]

Behavior:
    Inlines reveal.js, reset.css, reveal.css, theme CSS, and the notes + highlight
    plugins directly into <style>/<script> tags. The output HTML has no external
    dependencies (no reveal/ directory, no CDN), so a single file can be downloaded
    and opened in any browser offline.

    Fonts referenced by themes are dropped (system sans-serif fallback) — embedding
    them as data: URLs would add ~1 MB of base64 for marginal visual benefit.

After the shell is created, the AI agent writes <section> elements directly into
the empty `<div class="slides">` container. To embed images for portability, run
`pack_html_slides.py` after the deck is finished.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
ASSETS_REVEAL = SKILL_DIR / "assets" / "reveal"


CUSTOM_CSS = """
/* ---- Custom slide patterns (kept minimal; AI may extend per slide) ---- */
.reveal .slides section {
  text-align: left;
  font-size: 22px;
  line-height: 1.45;
}
.reveal h1, .reveal h2, .reveal h3 {
  text-transform: none;
  letter-spacing: normal;
  margin-bottom: 0.4em;
}
.reveal h2 { font-size: 1.6em; }
.reveal h3 { font-size: 1.15em; color: #444; }
.reveal ul, .reveal ol { margin-left: 1.1em; }
.reveal li { margin-bottom: 0.25em; }
.reveal small, .reveal .caption { color: #666; font-size: 0.7em; }

/* Two-column layout */
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2em; align-items: start; }
.cols-1-2 { display: grid; grid-template-columns: 1fr 2fr; gap: 1.2em; align-items: start; }
.cols-2-1 { display: grid; grid-template-columns: 2fr 1fr; gap: 1.2em; align-items: start; }

/* Image+text grid (image left, bullets right by default) */
.fig { text-align: center; }
.fig img { max-width: 100%; max-height: 65vh; object-fit: contain; }
.fig figcaption { color: #666; font-size: 0.7em; margin-top: 0.3em; }

/* Compact tables */
.reveal table { font-size: 0.7em; border-collapse: collapse; width: 100%; }
.reveal th, .reveal td { padding: 0.3em 0.5em; border-bottom: 1px solid #ddd; }
.reveal th { background: #f4f4f4; text-align: left; }
.reveal table.tight td, .reveal table.tight th { padding: 0.2em 0.4em; }

/* Stacked tables on one slide */
.stacked-tables { display: flex; flex-direction: column; gap: 0.8em; }

/* Title slide */
.title-slide { text-align: center; }
.title-slide h1 { font-size: 2.2em; margin-bottom: 0.2em; }
.title-slide .subtitle { font-size: 1.1em; color: #555; }
.title-slide .meta { margin-top: 1.2em; color: #777; font-size: 0.85em; }

/* Callouts */
.callout { background: #f7f7f7; border-left: 4px solid #888; padding: 0.6em 0.9em; }
.callout.good { border-color: #2a7; }
.callout.warn { border-color: #c83; }
.callout.bad  { border-color: #c33; }

/* Code blocks: keep fonts readable on slides */
.reveal pre code { font-size: 0.65em; line-height: 1.3; max-height: 65vh; }
"""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_font_imports(css: str) -> str:
    """Drop @import url(.../fonts/...) and @font-face blocks so we don't need sidecar font files."""
    # Remove @import lines that reference fonts
    css = re.sub(r"@import\s+url\([^)]*fonts/[^)]*\)\s*;", "", css)
    # Remove @font-face blocks (greedy match the full block)
    css = re.sub(r"@font-face\s*\{[^}]*\}", "", css, flags=re.DOTALL)
    return css


def _gather_inlined_css(theme: str) -> str:
    """Concatenate reset + reveal + theme + highlight CSS, with font imports stripped."""
    parts = [
        ("/* reset.css */", _read(ASSETS_REVEAL / "dist" / "reset.css")),
        ("/* reveal.css */", _read(ASSETS_REVEAL / "dist" / "reveal.css")),
        (f"/* theme/{theme}.css */",
         _strip_font_imports(_read(ASSETS_REVEAL / "dist" / "theme" / f"{theme}.css"))),
        ("/* highlight/monokai.css */",
         _read(ASSETS_REVEAL / "plugin" / "highlight" / "monokai.css")),
        ("/* custom slide patterns */", CUSTOM_CSS),
    ]
    return "\n\n".join(f"{label}\n{body}" for label, body in parts)


def _gather_inlined_js() -> str:
    """Concatenate reveal.js core + notes + highlight plugin scripts.

    Only UMD bundles (reveal.js, notes.js, highlight.js) are inlined.  plugin.js
    is the ES-module source of the highlight plugin and would raise a SyntaxError
    inside a plain <script> tag — highlight.js is the browser-ready bundle that
    already exposes `RevealHighlight` globally.
    """
    parts = [
        ("/* reveal.js */", _read(ASSETS_REVEAL / "dist" / "reveal.js")),
        ("/* notes plugin */", _read(ASSETS_REVEAL / "plugin" / "notes" / "notes.js")),
        ("/* highlight plugin */", _read(ASSETS_REVEAL / "plugin" / "highlight" / "highlight.js")),
    ]
    return "\n\n".join(f"// {label}\n{body}" for label, body in parts)


REVEAL_INIT = """
Reveal.initialize({
  hash: true,
  slideNumber: 'c/t',
  transition: 'fade',
  controls: true,
  progress: true,
  center: false,
  width: 1280,
  height: 720,
  margin: 0.04,
  plugins: [ RevealNotes, RevealHighlight ]
});
"""


def render_html(theme: str, title: str) -> str:
    css = _gather_inlined_css(theme)
    js = _gather_inlined_js()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
/* System font fallback (theme fonts intentionally omitted to keep file portable) */
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
                    Arial, sans-serif; }}
{css}
</style>
</head>
<body>
<div class="reveal">
  <div class="slides">

    <!-- AGENT: write <section> elements here.
         Each content slide should contain an <aside class="notes">…</aside> block.
         See skills/md-report-to-html-slides/references/slide_blueprint.md for patterns. -->

  </div>
</div>

<script>
{js}
</script>
<script>
{REVEAL_INIT}
</script>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path, help="Path to the slides .html file to create")
    p.add_argument("--theme", default="white", choices=["white", "simple"],
                   help="reveal.js theme (default: white)")
    p.add_argument("--title", default="Slides", help="HTML <title> value")
    p.add_argument("--force", action="store_true", help="Overwrite existing output file")
    args = p.parse_args()

    output: Path = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        print(f"{output} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    html = render_html(args.theme, args.title)
    output.write_text(html, encoding="utf-8")
    size_kb = len(html.encode("utf-8")) / 1024
    print(f"Wrote single-file shell: {output}  ({size_kb:.0f} KB)")
    print(f"  No sidecar 'reveal/' directory needed — the file is self-contained.")
    print(f"  After writing your <section> elements, run pack_html_slides.py to embed images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
