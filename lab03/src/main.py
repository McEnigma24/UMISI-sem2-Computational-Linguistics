"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess
import traceback

from _utils_.log_print import log_print, reset_log_timer, tensorboard_run_dir
from datetime import datetime

from finetune_presets import FINETUNE_QUEUE
import finetune


def main(session_started_at: datetime | None = None) -> str:
    reset_log_timer()
    when = session_started_at or datetime.now()
    log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
    run_dir = tensorboard_run_dir(log_root, when)

    try:
        import torch

        log_print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        log_print(f"TensorBoard logi: {run_dir}  (w shellu: tb)")
    except ImportError:
        log_print("Brak torch — uruchom na hoście: ./scripts/drun sync")

    subprocess.run(["clear"], check=False)
    return run_dir


if __name__ == "__main__":
    run_dir = main()
    n_ok = 0
    n_skip = 0
    n_fail = 0
    for i, preset in enumerate(FINETUNE_QUEUE, start=1):
        tag = preset.get("run_tag", "?")
        done = finetune.completed_run_dir(preset)
        if done is not None:
            log_print(f"[{i}/{len(FINETUNE_QUEUE)}] POMIJAM (gotowe): {tag} → {done.name}")
            n_skip += 1
            continue
        log_print(f"[{i}/{len(FINETUNE_QUEUE)}] START: {tag}")
        try:
            finetune.main(tensorboard_dir=run_dir, **preset)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            log_print(f"BŁĄD run {tag}: {type(e).__name__}: {e}")
            log_print(traceback.format_exc())

    log_print(f"Kolejka: ok={n_ok}, pominięte={n_skip}, błędy={n_fail}")

    try:
        import finetune_report

        finetune_report.main()
    except Exception as e:
        log_print(f"finetune_report nie powiódł się: {type(e).__name__}: {e}")
        log_print(traceback.format_exc())