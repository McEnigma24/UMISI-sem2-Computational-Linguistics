#!/usr/bin/env bash
# Trening w tle → nohup.out; PID w nohup.pid; na końcu tail -f logu.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PID_FILE="${ROOT}/nohup.pid"
LOG_FILE="${ROOT}/nohup.out"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Już działa PID ${old_pid} — najpierw ./cancel.sh" >&2
    exit 1
  fi
fi

# Własna sesja — inaczej Ctrl+C przy tail -f trafia w tę samą grupę procesów co docker.
setsid nohup ./shell.sh ./start.sh >"${LOG_FILE}" 2>&1 </dev/null &
echo "$!" >"${PID_FILE}"
disown -h "$!" 2>/dev/null || true
echo "Start w tle (PID $(cat "${PID_FILE}"))."

exec "${ROOT}/background_process_tail.sh"
