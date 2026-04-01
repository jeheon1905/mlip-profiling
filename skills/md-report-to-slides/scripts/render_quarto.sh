#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--check" ]]; then
  if command -v quarto >/dev/null; then
    echo "quarto available: $(quarto --version | head -n 1)"
    exit 0
  fi
  echo "quarto is not installed" >&2
  exit 1
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/render_quarto.sh <input.qmd> [output.pptx]" >&2
  exit 2
fi

infile="$1"
outfile="${2:-}"
[[ -f "$infile" ]] || { echo "input not found: $infile" >&2; exit 1; }
command -v quarto >/dev/null || { echo "quarto is not installed" >&2; exit 1; }

quarto render "$infile" --to pptx

if [[ -n "$outfile" ]]; then
  base="${infile%.qmd}.pptx"
  [[ -f "$base" ]] || { echo "render failed: output not created" >&2; exit 1; }
  mv -f "$base" "$outfile"
  echo "rendered $outfile"
else
  echo "rendered ${infile%.qmd}.pptx"
fi
