#!/bin/bash
# Użycie:
#   ./start                    — nowy run
#   ./start latest             — wznowienie z najnowszego runu tej architektury (nowy katalog)
#   ./start saved_models/...   — wznowienie z podanego katalogu runu
#
# Zmienna LANGUAGE_RNN_RESUME (latest lub ścieżka) — to samo co pierwszy argument.
# Porównanie metryk (multi-line PNG): python src/metrics/plot_metrics.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -t 1 ]]; then
  clear
fi

export LANGUAGE_RNN_RESUME="${1:-}"
unset LANGUAGE_RNN_CHECKPOINT

if [[ -x build/venv/bin/python ]]; then
  exec build/venv/bin/python src/main.py
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run python src/main.py
fi
exec python src/main.py