#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT}/nohup.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "Brak ${PID_FILE} — nic nie było uruchomione przez background_process.sh" >&2
  exit 1
fi

pid="$(cat "${PID_FILE}")"
if ! [[ "${pid}" =~ ^[0-9]+$ ]]; then
  echo "Niepoprawny PID w ${PID_FILE}: ${pid!r}" >&2
  exit 1
fi

if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Proces ${pid} nie działa — usuwam ${PID_FILE}"
  rm -f "${PID_FILE}"
  exit 0
fi

kill -9 "${pid}"
rm -f "${PID_FILE}"
echo "Zabito PID ${pid}"