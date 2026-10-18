#!/bin/bash
source "_common_" 2>/dev/null

clear
rm -f output/*

proc_number="${1:-4}"
uv run mpiexec -n "$proc_number" python -m lab01_tokenization | tee output/out
