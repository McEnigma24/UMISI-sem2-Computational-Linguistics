#!/usr/bin/env bash
# Generuje fragmenty LaTeX w kontenerze i buduje PDF na hoście.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
./shell.sh uv run python src/finetune_report.py
cd report
pdflatex -interaction=nonstopmode raport.tex >/dev/null
pdflatex -interaction=nonstopmode raport.tex
echo "Gotowe: ${ROOT}/report/raport.pdf"