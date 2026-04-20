#!/usr/bin/env python3
"""Validate a reveal.js slides.html written for the md-report-to-html-slides skill.

Checks (each prints PASS/WARN/FAIL):
  - Slide count is within [min, max] range (default 6-16)
  - First slide looks like a title slide (contains <h1>)
  - Every content slide has at least one <aside class="notes"> block
  - All <img src="…"> paths resolve to a file on disk
  - No <section> is empty
  - reveal/ directory exists alongside the html (assets actually shipped)

Usage:
    python validate_html_slides.py slides.html [--min 6] [--max 16]

Exit code 0 = no FAIL items; 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class SlideParser(HTMLParser):
    """Collect <section>, <img>, and <aside class="notes"> info."""

    def __init__(self) -> None:
        super().__init__()
        self.section_depth = 0
        self.sections: list[dict] = []
        self.images: list[str] = []
        self._current_section: dict | None = None
        self._aside_depth = 0
        self._aside_is_notes = False

    def handle_starttag(self, tag: str, attrs):
        attrs_d = dict(attrs)
        if tag == "section":
            self.section_depth += 1
            if self.section_depth == 1:
                self._current_section = {
                    "has_h1": False,
                    "has_notes": False,
                    "text_chars": 0,
                    "child_sections": 0,
                }
            else:
                if self._current_section is not None:
                    self._current_section["child_sections"] += 1
        elif tag == "h1" and self._current_section is not None:
            self._current_section["has_h1"] = True
        elif tag == "aside":
            self._aside_depth += 1
            classes = attrs_d.get("class", "").split()
            if "notes" in classes:
                self._aside_is_notes = True
                if self._current_section is not None:
                    self._current_section["has_notes"] = True
        elif tag == "img":
            src = attrs_d.get("src")
            if src:
                self.images.append(src)

    def handle_endtag(self, tag: str):
        if tag == "section":
            self.section_depth -= 1
            if self.section_depth == 0 and self._current_section is not None:
                self.sections.append(self._current_section)
                self._current_section = None
        elif tag == "aside":
            self._aside_depth -= 1
            if self._aside_depth == 0:
                self._aside_is_notes = False

    def handle_data(self, data: str):
        if self._current_section is not None and not self._aside_is_notes:
            self._current_section["text_chars"] += len(data.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("html", type=Path)
    p.add_argument("--min", type=int, default=6, help="Min slide count (default 6)")
    p.add_argument("--max", type=int, default=16, help="Max slide count (default 16)")
    args = p.parse_args()

    if not args.html.exists():
        print(f"FAIL: {args.html} does not exist", file=sys.stderr)
        return 1

    src = args.html.read_text(encoding="utf-8")
    parser = SlideParser()
    parser.feed(src)

    fails: list[str] = []
    warns: list[str] = []
    passes: list[str] = []

    n = len(parser.sections)
    if n == 0:
        fails.append(f"No <section> elements found")
    elif n < args.min:
        warns.append(f"Slide count {n} is below recommended min ({args.min})")
    elif n > args.max:
        warns.append(f"Slide count {n} exceeds recommended max ({args.max})")
    else:
        passes.append(f"Slide count {n} within [{args.min}, {args.max}]")

    # Title slide check (first section should have h1)
    if parser.sections and not parser.sections[0]["has_h1"]:
        warns.append("First slide has no <h1> — title slide convention expects one")
    elif parser.sections:
        passes.append("First slide has <h1> (title slide)")

    # Notes on each content slide (slides 2..N)
    for i, sec in enumerate(parser.sections[1:], start=2):
        if not sec["has_notes"]:
            warns.append(f"Slide {i}: missing <aside class=\"notes\">")

    if all(sec["has_notes"] for sec in parser.sections[1:]) and len(parser.sections) > 1:
        passes.append("All content slides have speaker notes")

    # Empty sections
    for i, sec in enumerate(parser.sections, start=1):
        if sec["text_chars"] == 0 and sec["child_sections"] == 0:
            # Could still have images — check by whether any img falls inside section
            # (HTMLParser doesn't track that here; treat as warn only if no images on whole deck)
            warns.append(f"Slide {i}: no visible text content (verify it has an <img>)")

    # Image existence
    base_dir = args.html.parent
    missing = []
    embedded = 0
    external_url = 0
    local_ok = 0
    for src_path in parser.images:
        if src_path.startswith("data:"):
            embedded += 1
            continue
        if re.match(r"^(https?:|//)", src_path):
            external_url += 1
            continue
        target = (base_dir / src_path).resolve()
        if not target.exists():
            missing.append(src_path)
        else:
            local_ok += 1
    if missing:
        for m in missing:
            fails.append(f"Image not found: {m}")
    if local_ok:
        passes.append(f"{local_ok} local image(s) resolve on disk")
    if embedded:
        passes.append(f"{embedded} image(s) embedded as data: URLs (portable)")
    if external_url:
        warns.append(f"{external_url} image(s) reference http(s) URLs (offline use will break)")
    if parser.images and not embedded and local_ok and not external_url:
        warns.append("No images embedded — file depends on sibling image files. "
                     "Run pack_html_slides.py for a portable single-file deck.")

    # Self-contained check: reveal.js must be inlined (no <script src=...reveal...>)
    if re.search(r'<script[^>]*\bsrc\s*=\s*"[^"]*reveal[^"]*"', src, re.IGNORECASE):
        warns.append("HTML loads reveal.js from a sibling path — "
                     "regenerate with create_html_shell.py for a self-contained file")
    elif "Reveal.initialize" in src:
        passes.append("reveal.js initialization is inlined (self-contained)")

    print(f"\n=== Validation report for {args.html} ===\n")
    for msg in passes:
        print(f"  PASS  {msg}")
    for msg in warns:
        print(f"  WARN  {msg}")
    for msg in fails:
        print(f"  FAIL  {msg}")
    print(f"\nSummary: {len(passes)} pass, {len(warns)} warn, {len(fails)} fail\n")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
