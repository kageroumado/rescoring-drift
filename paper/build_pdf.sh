#!/usr/bin/env bash
# Build paper/rescoring-drift.pdf from paper/paper.md: figures from the committed results, then pandoc + typst.
set -euo pipefail
export SOURCE_DATE_EPOCH=1792281600  # fixed UTC date: reproducible PDF, no local time zone in its metadata
cd "$(dirname "$0")"
python3 -I make_figures.py
pandoc paper.md -o rescoring-drift.pdf --pdf-engine=typst \
  -V papersize=a4 -V fontsize=10pt -V margin.x=2.2cm -V margin.y=2.2cm
echo "built $(pwd)/rescoring-drift.pdf"
