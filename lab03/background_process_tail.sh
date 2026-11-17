#!/usr/bin/env bash
# Status (ACTIVE/DEAD) → sleep → clear → tail -f nohup.out
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT}/nohup.pid"
LOG_FILE="${ROOT}/nohup.out"

active=0
pid=""

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    active=1
  else
    rm -f "${PID_FILE}"
    pid=""
  fi
fi

if [[ "${active}" -eq 1 ]]; then
  printf '\n\n\nACTIVE (PID %s)\n\n\n' "${pid}"
else
  printf '\n\n\nDEAD\n\n\n'
fi

sleep 2
clear

touch "${LOG_FILE}"
exec tail -f "${LOG_FILE}"