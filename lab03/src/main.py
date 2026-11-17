"""Domyślny punkt wejścia w kontenerze: run_main"""

from __future__ import annotations

import os
import shutil
import subprocess

import infer
import train
from _utils_.log_print import log_print, reset_log_timer, tensorboard_run_dir

from training_presets import DUAL3060_ALL

_TRAIN_JOB_ENV = "LANGUAGE_RNN_TRAIN_JOB"
_TENSORBOARD_JOB_ENV = "LANGUAGE_RNN_TENSORBOARD_DIR"


def main() -> str:
    reset_log_timer()
    log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
    run_dir = tensorboard_run_dir(log_root)

    try:
        import torch

        log_print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        log_print(f"TensorBoard logi: {run_dir}  (w shellu: tb)")
    except ImportError:
        log_print("Brak torch — uruchom na hoście: ./scripts/drun sync")

    subprocess.run(["clear"], check=False)
    return run_dir


def _train_job(run_dir: str, job_index: int) -> None:
    architecture, preset = DUAL3060_ALL[job_index]
    jobstate = os.path.join(os.path.dirname(__file__), ".train_jobstate")
    os.makedirs(jobstate, exist_ok=True)
    os.environ[_TRAIN_JOB_ENV] = str(job_index)
    os.environ[_TENSORBOARD_JOB_ENV] = run_dir
    os.environ["LANGUAGE_RNN_JOBSTATE_DIR"] = jobstate
    try:
        train.main(tensorboard_dir=run_dir, architecture=architecture, **preset)
    finally:
        os.environ.pop(_TRAIN_JOB_ENV, None)
        os.environ.pop(_TENSORBOARD_JOB_ENV, None)
        os.environ.pop("LANGUAGE_RNN_JOBSTATE_DIR", None)


if __name__ == "__main__":
    # Lightning DDP uruchamia ten skrypt ponownie z LOCAL_RANK=1 przy pierwszym fit();
    # oba procesy muszą przejść całą pętlę w tej samej kolejności (każdy Trainer.fit to wspólny krok).
    if os.environ.get("LOCAL_RANK", "0") == "0":
        shutil.rmtree(os.path.join(os.path.dirname(__file__), ".train_jobstate"), ignore_errors=True)
    run_dir = main()
    for job_index in range(len(DUAL3060_ALL)):
        _train_job(run_dir, job_index)


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