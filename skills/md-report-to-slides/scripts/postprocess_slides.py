#!/usr/bin/env python3
"""Post-process pandoc pptx to fix multi-image slides.

Pandoc 2.x splits multiple images under one heading into separate slides,
and only embeds the first image — subsequent ones become text-only orphans.

This script:
  1. Parses slides.md to learn which slide titles map to which images
  2. Removes orphan slides (pandoc-generated splits with no real content)
  3. Lays out all intended images on the correct parent slide

Usage:
    python scripts/postprocess_slides.py slides.pptx slides.md [--resource-path DIR]

If --resource-path is not given, images are resolved relative to slides.md's directory.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

try:
    from pptx import Presentation
    from pptx.util import Inches, Emu
except ImportError:
    print("python-pptx is required: pip install python-pptx", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Parse slides.md to extract slide → images mapping
# ---------------------------------------------------------------------------

def parse_slide_images(md_path: Path, resource_path: Path) -> list[dict]:
    """Parse slides.md and return list of {title, images, has_text} per slide."""
    text = md_path.read_text(encoding="utf-8")

    # Strip YAML frontmatter
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5:]

    # Collect all image alt texts for split detection
    all_alts: set[str] = set()

    # Split on ## headings
    slides = []
    matches = list(re.finditer(r"(?m)^##\s+(.+)$", text))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Extract image references
        images = []
        for img_match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", block):
            alt = img_match.group(1)
            src = img_match.group(2)
            all_alts.add(normalize(alt))
            full_path = resource_path / src
            if full_path.exists():
                images.append({"alt": alt, "path": full_path})

        # Check if block has bullet text (not just images and notes)
        # Strip images, notes blocks, and blank lines — see if bullets remain
        stripped = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", block)
        stripped = re.sub(r":::.*?:::", "", stripped, flags=re.DOTALL)
        has_text = bool(re.search(r"(?m)^- .+", stripped))

        slides.append({"title": title, "images": images, "has_text": has_text})

    return slides, all_alts


# ---------------------------------------------------------------------------
# PPTX helpers
# ---------------------------------------------------------------------------

def get_slide_title(slide) -> str:
    """Extract title text from a slide."""
    for shape in slide.shapes:
        try:
            if shape.placeholder_format is not None and shape.placeholder_format.idx == 0:
                return shape.text_frame.text.strip()
        except (ValueError, AttributeError):
            pass
    # Fallback: first text shape
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            return shape.text_frame.text.strip()
    return ""


def get_slide_images(slide) -> list:
    """Return image shapes on a slide."""
    return [s for s in slide.shapes if hasattr(s, "image")]


def has_real_content(slide) -> bool:
    """Check if slide has meaningful content beyond just a caption."""
    images = get_slide_images(slide)
    text_shapes = [s for s in slide.shapes if s.has_text_frame]
    # Title placeholder (idx=0) with real title counts
    for s in text_shapes:
        try:
            if s.placeholder_format is not None and s.placeholder_format.idx == 0:
                if len(s.text_frame.text.strip().split()) > 4:
                    return True
        except (ValueError, AttributeError):
            pass
    # Has bullets or paragraphs in body
    for s in text_shapes:
        try:
            if s.placeholder_format is not None and s.placeholder_format.idx == 1:
                text = s.text_frame.text.strip()
                if text and "\n" in text:
                    return True
        except (ValueError, AttributeError):
            pass
    return len(images) > 0


def normalize(s: str) -> str:
    """Normalize title for fuzzy matching. Replaces smart quotes and collapses whitespace."""
    s = s.strip().lower()
    # Normalize smart quotes to ASCII
    s = s.replace("\u2018", "'").replace("\u2019", "'")  # ' '
    s = s.replace("\u201c", '"').replace("\u201d", '"')  # " "
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # – —
    s = re.sub(r"\s+", " ", s)
    return s[:80]


def get_body_placeholder(slide):
    """Return the body placeholder (idx=1) if it exists."""
    for shape in slide.shapes:
        try:
            if shape.placeholder_format is not None and shape.placeholder_format.idx == 1:
                return shape
        except (ValueError, AttributeError):
            pass
    return None


def slide_has_text_content(slide) -> bool:
    """Check if slide has real text bullets in the body placeholder."""
    body = get_body_placeholder(slide)
    if body is None:
        return False
    text = body.text_frame.text.strip()
    # Has at least 2 non-empty paragraphs (real bullets, not just image alt text)
    lines = [l for l in text.split("\n") if l.strip()]
    return len(lines) >= 2


def layout_images_with_text(slide, image_paths: list[Path], slide_w, slide_h):
    """Layout images on the LEFT side and resize body text to the RIGHT side.

    Used when a slide has both images and bullet text content.
    Images get ~45% width (left), text gets ~52% width (right).
    """
    from pptx.util import Inches, Emu

    n = len(image_paths)
    if n == 0:
        return

    top = Inches(1.3)
    bottom_margin = Inches(0.2)
    gap = Inches(0.15)
    side_margin = Inches(0.4)

    # Image area: left 45%
    img_area_w = int(slide_w * 0.45)
    img_left = side_margin
    # Text area: right 52%
    text_left = int(side_margin + img_area_w + gap)
    text_w = int(slide_w - text_left - side_margin)

    avail_h = slide_h - top - bottom_margin

    # Resize body placeholder to right side
    body = get_body_placeholder(slide)
    if body is not None:
        body.left = int(text_left)
        body.top = int(top)
        body.width = int(text_w)
        body.height = int(avail_h)

    # Layout images stacked vertically on the left
    img_inner_w = int(img_area_w - side_margin)
    if n == 1:
        cell_h = int(avail_h)
    else:
        cell_h = int((avail_h - gap * (n - 1)) / n)

    for i, img_path in enumerate(image_paths):
        cell_y = int(top + i * (cell_h + gap))

        pic = slide.shapes.add_picture(str(img_path), int(img_left), cell_y)
        native_w = pic.width
        native_h = pic.height

        if native_w > 0 and native_h > 0:
            scale = min(img_inner_w / native_w, cell_h / native_h)
            new_w = int(native_w * scale)
            new_h = int(native_h * scale)
        else:
            new_w, new_h = img_inner_w, cell_h

        pic.width = new_w
        pic.height = new_h
        # Center horizontally in image area
        pic.left = int(img_left) + (img_inner_w - new_w) // 2
        pic.top = cell_y + (cell_h - new_h) // 2


def layout_images_on_slide(slide, image_paths: list[Path], slide_w, slide_h):
    """Add images to a slide in a responsive grid, preserving aspect ratio."""
    n = len(image_paths)
    if n == 0:
        return

    # Content area below header bar
    top = Inches(1.3)
    bottom_margin = Inches(0.2)
    left = Inches(0.4)
    right_margin = Inches(0.4)
    gap = Inches(0.2)

    avail_w = slide_w - left - right_margin
    avail_h = slide_h - top - bottom_margin

    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = 2, 1
    elif n == 3:
        cols, rows = 3, 1
    elif n <= 4:
        cols, rows = 2, 2
    else:
        cols = 3
        rows = (n + cols - 1) // cols

    cell_w = int((avail_w - gap * (cols - 1)) / cols)
    cell_h = int((avail_h - gap * (rows - 1)) / rows)

    for i, img_path in enumerate(image_paths):
        col = i % cols
        row = i // cols
        cell_x = int(left + col * (cell_w + gap))
        cell_y = int(top + row * (cell_h + gap))

        # Add image without specifying dimensions to get native size
        pic = slide.shapes.add_picture(str(img_path), cell_x, cell_y)

        # Read native size (the actual image dimensions in EMU)
        native_w = pic.width
        native_h = pic.height

        # Fit within cell while preserving aspect ratio
        if native_w > 0 and native_h > 0:
            scale = min(cell_w / native_w, cell_h / native_h)
            new_w = int(native_w * scale)
            new_h = int(native_h * scale)
        else:
            new_w, new_h = cell_w, cell_h

        pic.width = new_w
        pic.height = new_h

        # Center in cell
        pic.left = cell_x + (cell_w - new_w) // 2
        pic.top = cell_y + (cell_h - new_h) // 2


# ---------------------------------------------------------------------------
# Main postprocess logic
# ---------------------------------------------------------------------------

def postprocess(pptx_path: Path, md_path: Path, resource_path: Path):
    slide_specs, all_image_alts = parse_slide_images(md_path, resource_path)
    prs = Presentation(str(pptx_path))
    slide_w = prs.slide_width
    slide_h = prs.slide_height
    slides = list(prs.slides)

    from pptx.oxml.ns import qn
    from copy import deepcopy

    if not slide_specs:
        print("No slide specs found in markdown.")
        return

    # Build lookup: normalized title prefix → spec
    spec_lookup = {}
    for spec in slide_specs:
        key = normalize(spec["title"])
        spec_lookup[key] = spec

    # Identify parent slides (match to spec by title) and orphans
    orphan_indices = set()
    parent_map = {}  # slide_index → spec
    last_parent_idx = None  # track for orphan → parent note transfer
    orphan_to_parent = {}  # orphan_idx → parent_idx

    for i, slide in enumerate(slides):
        title = normalize(get_slide_title(slide))
        # Try matching to a spec
        matched = False
        if title:  # skip empty titles
            best_score = 0
            best_spec = None
            for key, spec in spec_lookup.items():
                # Score by longest common prefix
                min_len = min(len(title), len(key))
                common = 0
                for c1, c2 in zip(title, key):
                    if c1 == c2:
                        common += 1
                    else:
                        break
                if common >= 20 and common > best_score:
                    best_score = common
                    best_spec = spec
            if best_spec:
                parent_map[i] = best_spec
                matched = True
                last_parent_idx = i

        if not matched:
            # Never remove the first slide (title slide) or slides with notes
            if i == 0:
                last_parent_idx = 0
                continue
            # Check if this slide's title is an image alt text (pandoc split)
            # Pandoc may concatenate alt text + bullet text, so use startswith
            is_image_split = False
            if title:
                for alt in all_image_alts:
                    if title.startswith(alt):
                        is_image_split = True
                        break
            if is_image_split:
                # This is a pandoc-generated split — always an orphan
                orphan_indices.add(i)
                if last_parent_idx is not None:
                    orphan_to_parent[i] = last_parent_idx
                continue
            if has_real_content(slide):
                last_parent_idx = i
                continue
            # This is an orphan
            orphan_indices.add(i)
            if last_parent_idx is not None:
                orphan_to_parent[i] = last_parent_idx

    # Transfer notes AND text content from orphan slides to their parent
    for orphan_idx, parent_idx in orphan_to_parent.items():
        orphan = slides[orphan_idx]
        parent = slides[parent_idx]
        # Transfer notes
        if orphan.has_notes_slide:
            orphan_notes = orphan.notes_slide.notes_text_frame.text.strip()
            if orphan_notes and not parent.has_notes_slide:
                parent_notes_slide = parent.notes_slide  # creates if needed
                parent_notes_slide.notes_text_frame.text = orphan_notes
        # Transfer text content from orphan body to parent body
        orphan_body = get_body_placeholder(orphan)
        parent_body = get_body_placeholder(parent)
        if orphan_body and parent_body:
            orphan_text = orphan_body.text_frame.text.strip()
            parent_text = parent_body.text_frame.text.strip()
            if orphan_text and not parent_text:
                # Copy paragraph-by-paragraph to preserve formatting
                src_txBody = orphan_body._element.find(qn("p:txBody"))
                dst_txBody = parent_body._element.find(qn("p:txBody"))
                if src_txBody is not None and dst_txBody is not None:
                    # Remove existing empty paragraphs from parent body
                    for p in list(dst_txBody.findall(qn("a:p"))):
                        dst_txBody.remove(p)
                    # Copy all paragraphs from orphan
                    for p in src_txBody.findall(qn("a:p")):
                        dst_txBody.append(deepcopy(p))

    # Remove existing images from parent slides (will re-add from source)
    for idx, spec in parent_map.items():
        if not spec["images"]:
            continue
        slide = slides[idx]
        spec_alts = {normalize(img["alt"]) for img in spec["images"]}

        # Remove existing image shapes
        for img_shape in get_slide_images(slide):
            sp = img_shape._element
            sp.getparent().remove(sp)

        # Remove non-placeholder text shapes containing image alt text
        for sh in list(slide.shapes):
            try:
                if sh.placeholder_format is not None:
                    continue  # keep placeholders
            except (ValueError, AttributeError):
                pass
            if sh.has_text_frame:
                txt = normalize(sh.text_frame.text)
                if txt in spec_alts:
                    sh._element.getparent().remove(sh._element)

        # Transfer body content from orphan slides to parent
        # Pandoc puts the title on the parent but the body (bullets) on the orphan
        for orphan_idx, parent_idx in orphan_to_parent.items():
            if parent_idx != idx:
                continue
            orphan = slides[orphan_idx]
            orphan_body = get_body_placeholder(orphan)
            if orphan_body is None:
                continue

            # Clean alt text lines from orphan body
            orphan_txBody = orphan_body._element.find(qn("p:txBody"))
            if orphan_txBody is not None:
                for p in list(orphan_txBody.findall(qn("a:p"))):
                    p_text = normalize("".join(
                        r.text or "" for r in p.findall(f".//{qn('a:t')}")))
                    if p_text in spec_alts:
                        orphan_txBody.remove(p)

            remaining = orphan_body.text_frame.text.strip()
            if not remaining:
                continue

            # Check if parent already has a body placeholder
            parent_body = get_body_placeholder(slide)
            if parent_body is None:
                # Clone the orphan's body placeholder onto the parent slide
                slide._element.find(qn("p:cSld")).find(
                    qn("p:spTree")).append(deepcopy(orphan_body._element))
            else:
                # Merge: append orphan paragraphs to parent body
                dst_txBody = parent_body._element.find(qn("p:txBody"))
                src_txBody = orphan_body._element.find(qn("p:txBody"))
                if dst_txBody is not None and src_txBody is not None:
                    # Remove empty paragraphs from parent
                    for p in list(dst_txBody.findall(qn("a:p"))):
                        p_text = "".join(
                            r.text or "" for r in p.findall(f".//{qn('a:t')}")).strip()
                        if not p_text:
                            dst_txBody.remove(p)
                    for p in src_txBody.findall(qn("a:p")):
                        dst_txBody.append(deepcopy(p))

        # Decide layout based on whether slide now has text content
        image_paths = [img["path"] for img in spec["images"]]
        if slide_has_text_content(slide):
            layout_images_with_text(slide, image_paths, slide_w, slide_h)
        else:
            layout_images_on_slide(slide, image_paths, slide_w, slide_h)

    # Remove orphan slides
    if orphan_indices:
        sldIdLst = prs.slides._sldIdLst
        sldIds = list(sldIdLst)
        for idx in sorted(orphan_indices, reverse=True):
            if idx < len(sldIds):
                sldIdLst.remove(sldIds[idx])

    prs.save(str(pptx_path))
    total = len(prs.slides)
    img_slides = sum(1 for spec in parent_map.values() if spec["images"])
    print(f"Post-processed: {len(orphan_indices)} orphan(s) removed, "
          f"{img_slides} slide(s) got images. Result: {total} slides. "
          f"Saved {pptx_path}")


def main() -> int:
    args = sys.argv[1:]
    resource_path = None

    # Parse --resource-path
    filtered = []
    i = 0
    while i < len(args):
        if args[i] == "--resource-path" and i + 1 < len(args):
            resource_path = Path(args[i + 1])
            i += 2
        else:
            filtered.append(args[i])
            i += 1

    if len(filtered) < 2:
        print("usage: python scripts/postprocess_slides.py slides.pptx slides.md "
              "[--resource-path DIR]", file=sys.stderr)
        return 2

    pptx_path = Path(filtered[0])
    md_path = Path(filtered[1])

    if not pptx_path.exists():
        print(f"pptx not found: {pptx_path}", file=sys.stderr)
        return 1
    if not md_path.exists():
        print(f"markdown not found: {md_path}", file=sys.stderr)
        return 1

    if resource_path is None:
        resource_path = md_path.parent

    postprocess(pptx_path, md_path, resource_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
