from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def param_groups_no_decay(model: nn.Module, weight_decay: float) -> list[dict]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    total = max(total_steps, 1)
    warmup = min(warmup_steps, max(1, total // 20))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        if total <= warmup:
            return 1.0
        progress = (step - warmup) / float(max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)