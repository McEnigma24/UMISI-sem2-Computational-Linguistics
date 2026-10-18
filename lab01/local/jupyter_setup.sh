#!/bin/bash
# Jupyter + kernel „lab01 (uv)” dla Cursor / VS Code (Remote SSH).
# Lekka instalacja — uruchom z katalogu lab01 (najlepiej krótki job CPU, nie długi trening na loginie).

set -euo pipefail
source "$(dirname "$0")/_common_" 2>/dev/null || true
[[ -f local/env.sh ]] && source local/env.sh

cd "$(dirname "$0")/.."

echo "==> uv sync (grupa dev: jupyterlab, ipykernel)"
uv sync --group dev

echo "==> rejestracja kernela Jupyter"
uv run python -m ipykernel install --user --name lab01 --display-name "lab01 (uv)"

echo ""
echo "Gotowe. W Cursorze: otwórz notebooks/*.ipynb → wybierz kernel „lab01 (uv)”."
echo "Środowisko: $(uv run python -c 'import sys; print(sys.executable)')"
