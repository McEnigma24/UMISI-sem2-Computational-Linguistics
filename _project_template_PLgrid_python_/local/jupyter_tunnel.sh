#!/bin/bash
set -euo pipefail

NODE="${1:?użycie: ./local/jupyter_tunnel.sh <hostname_węzła> [port]}"
PORT="${2:-8888}"

echo "Tunel: login 127.0.0.1:${PORT} → ${NODE}:127.0.0.1:${PORT}"
exec ssh -N -L "${PORT}:127.0.0.1:${PORT}" "$NODE"
