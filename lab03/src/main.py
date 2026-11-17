"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timedelta

import infer
import train
from _utils_.log_print import log_print, reset_log_timer, tensorboard_run_dir

from training_presets import DUAL3060_ALL


def _session_started_at() -> datetime:
    """Jeden timestamp na oba procesy DDP — drugi proces to re-exec main.py z kopią env."""
    ts = os.environ.setdefault("LANGUAGE_RNN_SESSION_TS", str(int(time.time())))
    return datetime.fromtimestamp(int(ts))

def main(session_started_at: datetime) -> str:
    reset_log_timer()
    log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
    run_dir = tensorboard_run_dir(log_root, session_started_at)

    try:
        import torch

        log_print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        log_print(f"TensorBoard logi: {run_dir}  (w shellu: tb)")
    except ImportError:
        log_print("Brak torch — uruchom na hoście: ./scripts/drun sync")

    subprocess.run(["clear"], check=False)
    return run_dir



if __name__ == "__main__":
    session_started_at = _session_started_at()
    run_dir = main(session_started_at)
    
    # Oba procesy DDP idą przez tę pętlę w tej samej kolejności; +job_index s = osobny katalog na job.
    for job_index, (architecture, preset) in enumerate(DUAL3060_ALL):
        train.main(
            tensorboard_dir=run_dir,
            architecture=architecture,
            run_started_at=session_started_at + timedelta(seconds=job_index),
            **preset,
        )


# # COMPARE THEM ALL #
# length = 25
# infer.compare_models(
#     models=[(arch, None) for arch, _ in DUAL3060_ALL],
#     prompts=[
#         ("AGH to ", length),
#     ],
#     temperature=0.85,
#     run_tag="dual3060-110M",
# )
