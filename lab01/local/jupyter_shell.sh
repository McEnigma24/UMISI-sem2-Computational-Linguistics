#!/bin/bash
# Powłoka na węźle obliczeniowym (srun --pty). Wywoływana z grid/run_*_jupyter.
set -euo pipefail

KIND="${1:-cpu}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=/dev/null
[[ -f local/env.sh ]] && source local/env.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

echo "=== Lab01 — tryb Jupyter (${KIND}) ==="
echo "Host:     $(hostname)"
echo "Czas:     limit SLURM (domyślnie 2 h — zmienna GRID_JUPYTER_TIME przy starcie)"
echo "Katalog:  $(pwd)"
if [[ "$KIND" == "gpu" ]]; then
  module load cuda
  echo "GPU:"
  nvidia-smi -L || true
fi
echo ""
echo "Następny krok:  ./local/start_jupyter.sh"
echo "URL do Cursor:  cat grid/.jupyter_connect   (po starcie Jupyter)"
echo "Wyjdź z węzła:  exit"
echo ""

exec bash -l
