#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--check" ]]; then
  command -v pandoc >/dev/null || { echo "pandoc is not installed" >&2; exit 1; }
  echo "pandoc available: $(pandoc --version | head -n 1)"
  exit 0
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: bash scripts/render_pandoc.sh <input.md> <output.pptx> [reference.pptx]" >&2
  exit 2
fi

infile="$1"
outfile="$2"
reference="${3:-}"

[[ -f "$infile" ]] || { echo "input not found: $infile" >&2; exit 1; }
command -v pandoc >/dev/null || { echo "pandoc is not installed" >&2; exit 1; }

# Auto-detect template: look next to input file if not explicitly provided
if [[ -z "$reference" ]]; then
  indir="$(dirname "$infile")"
  for candidate in "$indir/template.pptx" "$indir/reference.pptx"; do
    if [[ -f "$candidate" ]]; then
      reference="$candidate"
      echo "auto-detected template: $reference"
      break
    fi
  done
fi

# Set resource-path to input file's directory so relative image paths resolve
resource_dir="$(cd "$(dirname "$infile")" && pwd)"

cmd=(pandoc "$infile" -o "$outfile" --resource-path="$resource_dir")
if [[ -n "$reference" ]]; then
  [[ -f "$reference" ]] || { echo "reference deck not found: $reference" >&2; exit 1; }
  cmd+=(--reference-doc="$reference")
fi

"${cmd[@]}"
[[ -f "$outfile" ]] || { echo "render failed: output not created" >&2; exit 1; }
echo "rendered $outfile"
