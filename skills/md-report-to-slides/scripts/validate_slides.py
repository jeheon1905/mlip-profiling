#!/usr/bin/env python3
from __future__ import annotations
import re
import sys
from pathlib import Path

GENERIC_TITLES = {"overview", "analysis", "results", "conclusion", "summary", "background"}


def split_frontmatter(text: str):
    if not text.startswith('---\n'):
        return None, text
    end = text.find('\n---\n', 4)
    if end == -1:
        return None, text
    return text[4:end], text[end+5:]


def count_bullets(block: str) -> int:
    """Count top-level bullets, excluding those inside column fenced divs and nested bullets."""
    in_column = 0
    count = 0
    for line in block.splitlines():
        stripped = line.strip()
        # Track column fenced divs (:::: {.columns} / ::: {.column})
        if re.match(r'^:{3,4}\s*\{\.column', stripped):
            in_column += 1
        elif in_column > 0 and re.match(r'^:{3,4}\s*$', stripped):
            in_column -= 1
        elif in_column == 0 and re.match(r'([-*+]\s+|\d+[.)]\s+)', line):
            # Top-level only: line starts at column 0 (no leading whitespace)
            count += 1
    return count


def validate(text: str) -> list[str]:
    errors: list[str] = []
    frontmatter, body = split_frontmatter(text)
    if frontmatter is None:
        errors.append('missing YAML frontmatter')
    elif 'notes:' not in frontmatter:
        errors.append('missing title-slide notes in YAML frontmatter')

    slides = re.split(r'(?m)^##\s+', body)
    if len(slides) <= 1:
        errors.append('no level-2 slide headings found')
        return errors

    matches = list(re.finditer(r'(?m)^##\s+(.+)$', body))
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[start:end]
        bullet_count = count_bullets(block)
        if title.lower() in GENERIC_TITLES:
            errors.append(f'generic slide title: {title}')
        if bullet_count > 8:
            errors.append(f'too many bullets on slide: {title} ({bullet_count})')
        if '::: notes' not in block:
            errors.append(f'missing notes block: {title}')

    open_notes = body.count('::: notes')
    # Count closing fences that are NOT opening fenced divs (columns, etc.)
    # A bare `:::` on its own line (no text after) is a closing fence.
    # `:::: ...` or `::: {.something}` or `::: notes` are opening fences.
    close_fences = len(re.findall(r'(?m)^:{3,4}\s*$', body))
    if close_fences < open_notes:
        errors.append('one or more notes blocks may be unclosed')
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print('usage: python scripts/validate_slides.py <slides.md>', file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f'input not found: {path}', file=sys.stderr)
        return 1
    text = path.read_text(encoding='utf-8')
    errors = validate(text)
    if errors:
        print('validation failed:')
        for err in errors:
            print(f'- {err}')
        return 1
    print('validation passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
