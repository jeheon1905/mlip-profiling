#!/usr/bin/env python3
"""Embed all local <img src="…"> images of a slides.html as base64 data: URLs.

Usage:
    python pack_html_slides.py slides.html [-o slides_packed.html]

After packing, the output HTML has no external file dependencies (images, reveal.js,
CSS all inline) and can be downloaded as a single file and opened in any browser.

Skips images whose src starts with http://, https://, //, or data:.
Reports total size delta and number of images embedded.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path


IMG_PATTERN = re.compile(r'<img\b([^>]*?)\bsrc\s*=\s*"([^"]+)"([^>]*)>', re.IGNORECASE)
TAG_OPEN = re.compile(r"<(script|style)\b", re.IGNORECASE)


def _split_outside_script_style(text: str):
    """Yield (segment, is_processable) tuples.

    Linearly walks the document once. When a <script> or <style> opening tag is
    hit, the paired closing tag is located and the entire block is emitted as a
    single non-processable segment. This avoids the bug where a nested pattern
    match (e.g., <style> inside a JS string literal within <script>) caused
    overlapping matches and content duplication.
    """
    cursor = 0
    while True:
        m = TAG_OPEN.search(text, cursor)
        if not m:
            if cursor < len(text):
                yield text[cursor:], True
            return
        if m.start() > cursor:
            yield text[cursor:m.start()], True
        tag = m.group(1).lower()
        close_re = re.compile(rf"</{tag}\s*>", re.IGNORECASE)
        close_m = close_re.search(text, m.end())
        if close_m:
            yield text[m.start():close_m.end()], False
            cursor = close_m.end()
        else:
            yield text[m.start():], False
            return


def is_external(src: str) -> bool:
    return src.startswith(("http://", "https://", "//", "data:"))


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{guess_mime(path)};base64,{b64}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("html", type=Path, help="Input HTML file (read; not modified by default)")
    p.add_argument("-o", "--output", type=Path,
                   help="Output path (default: <input>_packed.html). Use the same path as input "
                        "with --in-place to overwrite.")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite the input file instead of writing a _packed copy")
    args = p.parse_args()

    if not args.html.exists():
        print(f"FAIL: {args.html} does not exist", file=sys.stderr)
        return 1

    base_dir = args.html.parent
    src_text = args.html.read_text(encoding="utf-8")
    original_size = len(src_text.encode("utf-8"))

    embedded = 0
    skipped_external = 0
    missing: list[str] = []

    def replace(m: re.Match) -> str:
        nonlocal embedded, skipped_external
        before, src, after = m.group(1), m.group(2), m.group(3)
        if is_external(src):
            skipped_external += 1
            return m.group(0)
        target = (base_dir / src).resolve()
        if not target.exists():
            missing.append(src)
            return m.group(0)
        try:
            data_url = encode_image(target)
        except OSError as e:
            print(f"  WARN  Could not read {target}: {e}", file=sys.stderr)
            return m.group(0)
        embedded += 1
        return f'<img{before}src="{data_url}"{after}>'

    out_parts: list[str] = []
    for segment, processable in _split_outside_script_style(src_text):
        out_parts.append(IMG_PATTERN.sub(replace, segment) if processable else segment)
    out_text = "".join(out_parts)

    if missing:
        for m in missing:
            print(f"  FAIL  Image not found: {m}", file=sys.stderr)
        return 1

    if args.in_place:
        out_path = args.html
    elif args.output:
        out_path = args.output
    else:
        out_path = args.html.with_name(args.html.stem + "_packed" + args.html.suffix)

    out_path.write_text(out_text, encoding="utf-8")
    new_size = len(out_text.encode("utf-8"))

    print(f"Embedded {embedded} image(s); {skipped_external} external skipped.")
    print(f"  Input:  {original_size / 1024:.0f} KB  ({args.html})")
    print(f"  Output: {new_size / 1024:.0f} KB  ({out_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
