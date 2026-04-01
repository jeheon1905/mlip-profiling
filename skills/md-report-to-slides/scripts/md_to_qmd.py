#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path


def split_frontmatter(text: str):
    if not text.startswith('---\n'):
        return None, text
    end = text.find('\n---\n', 4)
    if end == -1:
        return None, text
    return text[4:end], text[end+5:]


def ensure_format(frontmatter: str | None) -> str:
    if frontmatter is None:
        return 'title: Presentation\nformat: pptx\n'
    lines = frontmatter.splitlines()
    if any(line.strip().startswith('format:') for line in lines):
        return frontmatter + ('\n' if not frontmatter.endswith('\n') else '')
    return frontmatter + ('\n' if not frontmatter.endswith('\n') else '') + 'format: pptx\n'


def main() -> int:
    if len(sys.argv) != 3:
        print('usage: python scripts/md_to_qmd.py <slides.md> <slides.qmd>', file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if not src.exists():
        print(f'input not found: {src}', file=sys.stderr)
        return 1
    text = src.read_text(encoding='utf-8')
    frontmatter, body = split_frontmatter(text)
    qmd = f"---\n{ensure_format(frontmatter)}---\n{body.lstrip()}"
    dst.write_text(qmd, encoding='utf-8')
    print(f'wrote {dst}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
