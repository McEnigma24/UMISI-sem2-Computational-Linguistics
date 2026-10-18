#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*

# ./grid/run_cpu.sh [NTASKS] [CPUS_PER_TASK]
#   NTASKS         — liczba procesów (MPI), domyślnie 8
#   CPUS_PER_TASK  — rdzenie na proces (wątki OMP w 1 procesie), domyślnie 1
# Przykład: ./grid/run_cpu.sh 2 1  → 2 procesy MPI × 1 rdzeń
#           ./grid/run_cpu.sh 1 8  → 1 proces × 8 rdzeni (SentencePiece, numpy, …)

NTASKS="${1:-8}"
CPUS_PER_TASK="${2:-1}"

sbatch --parsable --ntasks="$NTASKS" --cpus-per-task="$CPUS_PER_TASK" \
  --export=ALL,OMP_NUM_THREADS="$CPUS_PER_TASK" \
  grid/job_cpu | tee grid/.latest_job_started
