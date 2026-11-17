"""Czas treningu / synchronizacja CUDA (wspólne dla train i lit)."""

from __future__ import annotations

import time

import torch

from _utils_.log_print import log_print


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours} h {minutes} min"


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def log_eta_from_first_batch(*, phase: str, batch_seconds: float, n_batches: int) -> None:
    if n_batches <= 1 or batch_seconds <= 0:
        return
    est_total = batch_seconds * n_batches
    log_print(
        f"  szac. {phase}: ~{fmt_duration(est_total)} "
        f"(1. batch {batch_seconds:.2f}s × {n_batches:,} batchy)"
    )