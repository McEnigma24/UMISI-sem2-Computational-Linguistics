#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
[[ -f local/env.sh ]] && source local/env.sh

PORT="${JUPYTER_PORT:-8888}"
NODE="$(hostname -s)"
TOKEN="${JUPYTER_TOKEN:-$(openssl rand -hex 24)}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  BIND_IP="${JUPYTER_IP:-0.0.0.0}"
else
  BIND_IP="${JUPYTER_IP:-127.0.0.1}"
fi

if ! uv run python -c "import jupyterlab" 2>/dev/null; then
  echo "Instaluję Jupyter (grupa dev)…"
  uv sync --group dev
fi

CONNECT_FILE="$ROOT/grid/.jupyter_connect"

if [[ "$BIND_IP" == "0.0.0.0" ]]; then
  CURSOR_URL="http://${NODE}:${PORT}/lab?token=${TOKEN}"
else
  CURSOR_URL="http://127.0.0.1:${PORT}/lab?token=${TOKEN}"
fi

printf '%s\n' "$CURSOR_URL" >"$CONNECT_FILE"

cat <<EOF

=== Jupyter Lab ===
Węzeł:  ${NODE}   Port: ${PORT}   bind: ${BIND_IP}

>>> URL (Existing Jupyter Server) — jedna linia:
${CURSOR_URL}

(zapisano: grid/.jupyter_connect)

EOF

if [[ "$BIND_IP" == "127.0.0.1" ]]; then
  echo "Tunel na loginie: ./local/jupyter_tunnel.sh ${NODE} ${PORT}"
else
  echo "Cursor (login): użyj hostname ${NODE}, nie 127.0.0.1"
fi

exec uv run jupyter lab \
  --no-browser \
  --ip="$BIND_IP" \
  --port="$PORT" \
  --ServerApp.token="$TOKEN" \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True
