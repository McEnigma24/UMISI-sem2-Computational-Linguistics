#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*

# ./grid/run_gpu.sh [GPUS] [CPUS_PER_TASK]
#   GPUS           — liczba kart, domyślnie 1
#   CPUS_PER_TASK  — rdzenie CPU na proces, domyślnie 4

GPUS="${1:-1}"
CPUS_PER_TASK="${2:-4}"

sbatch --parsable --ntasks=1 --gres=gpu:"$GPUS" --cpus-per-task="$CPUS_PER_TASK" \
  --export=ALL,GPUS="$GPUS",OMP_NUM_THREADS="$CPUS_PER_TASK" \
  grid/job_gpu | tee grid/.latest_job_started
