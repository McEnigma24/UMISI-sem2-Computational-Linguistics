#!/bin/bash
# Kernel na loginie (lekka eksploracja). Na węźle: ./grid/run_* j … + start_jupyter.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source local/env.sh

KERNEL_NAME="${PLGRID_KERNEL_NAME:-plgrid-template}"
KERNEL_LABEL="${PLGRID_KERNEL_LABEL:-plgrid-template (uv)}"

uv sync --group dev
uv run python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_LABEL"

echo "Gotowe: kernel „${KERNEL_LABEL}” lub .venv w Cursorze."
