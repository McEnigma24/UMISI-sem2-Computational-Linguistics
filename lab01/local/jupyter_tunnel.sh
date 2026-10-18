#!/bin/bash
# Uruchom na LOGINIE (nie na węźle ag*). Jupyter na węźle musi mieć JUPYTER_IP=127.0.0.1
set -euo pipefail

NODE="${1:?użycie: ./local/jupyter_tunnel.sh <hostname_węzła> [port]}"
PORT="${2:-8888}"

echo "Tunel: login 127.0.0.1:${PORT} → ${NODE}:127.0.0.1:${PORT}"
echo "W Cursorze: http://127.0.0.1:${PORT}/lab?token=…  (token z grid/.jupyter_connect na węźle lub z logu Jupyter)"
echo "Ctrl+C kończy tunel."
exec ssh -N -L "${PORT}:127.0.0.1:${PORT}" "$NODE"
