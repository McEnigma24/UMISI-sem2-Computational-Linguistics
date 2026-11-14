#!/bin/bash
set -euo pipefail

KIND="${1:-cpu}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
[[ -f local/env.sh ]] && source local/env.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

echo "=== PLGrid — tryb Jupyter (${KIND}) ==="
echo "Host:     $(hostname)"
echo "Czas:     limit SLURM (GRID_JUPYTER_TIME, domyślnie 2 h)"
echo "Katalog:  $(pwd)"
if [[ "$KIND" == "gpu" ]]; then
  module load cuda
  echo "GPU:"
  nvidia-smi -L || true
fi
echo ""
echo "Następny krok:  ./local/start_jupyter.sh"
echo "URL do Cursor:  cat grid/.jupyter_connect"
echo "Wyjdź z węzła:  exit"
echo ""

exec bash -l
