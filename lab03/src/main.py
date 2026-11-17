"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import subprocess

import infer
import train
from _utils_.log_print import log_print, reset_log_timer, tensorboard_run_dir
from datetime import datetime

from training_presets import DUAL3060_ALL

from finetune_presets import FINETUNE_ALL
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


    # for architecture, preset in DUAL3060_ALL:
    #     train.main(tensorboard_dir=run_dir, architecture=architecture, **preset)

    for preset in FINETUNE_ALL:
        finetune.main(tensorboard_dir=run_dir, **preset)












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