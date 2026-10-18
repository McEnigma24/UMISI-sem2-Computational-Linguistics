"""Train language model on encoded bins."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from lab01_tokenization.config import (
    LOGS,
    ModelConfig,
    TrainConfig,
    paths_for_tokenizer,
)
from lab01_tokenization.model import GPT
from lab01_tokenization.utils import read_json, save_hardware_info, set_seed, timer, write_json


def load_token_memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_ix = len(data) - block_size - 1
    if max_ix <= 0:
        raise ValueError("dataset too small for block_size")
    ix = torch.randint(0, max_ix, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


def cosine_lr(step: int, warmup: int, total_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_tokenizer_lm(
    tokenizer_name: str,
    smoke: bool = False,
    max_train_mb: float | None = None,
) -> dict:
    from lab01_tokenization.encode import encode_all

    paths = paths_for_tokenizer(tokenizer_name)
    if not paths["bin_train"].exists():
        encode_all(tokenizer_name, max_train_mb=max_train_mb)

    train_cfg = TrainConfig()
    model_cfg = ModelConfig()
    if smoke:
        train_cfg = TrainConfig(
            batch_size=4,
            warmup_steps=10,
            max_epochs=1,
            eval_interval=20,
            log_interval=10,
        )
        model_cfg = ModelConfig(n_layer=2, n_head=4, n_embd=128, n_ff=512, block_size=64)

    set_seed(train_cfg.seed)
    save_hardware_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_token_memmap(paths["bin_train"])
    valid_data = load_token_memmap(paths["bin_valid"])

    tokens_per_epoch = len(train_data)
    steps_per_epoch = max(1, tokens_per_epoch // (train_cfg.batch_size * model_cfg.block_size))
    total_steps = steps_per_epoch * train_cfg.max_epochs
    if smoke:
        total_steps = min(total_steps, 40)

    model = GPT(model_cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )
    scaler = GradScaler(enabled=train_cfg.use_amp and device.type == "cuda")

    paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"train_{tokenizer_name}.jsonl"
    losses: list[float] = []
    best_val = float("inf")
    t_start = timer()
    step = 0
    pbar = tqdm(total=total_steps, desc=f"train {tokenizer_name}")

    while step < total_steps:
        lr = cosine_lr(step, train_cfg.warmup_steps, total_steps, train_cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, train_cfg.batch_size, model_cfg.block_size, device)
        with autocast(enabled=train_cfg.use_amp and device.type == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        loss_f = float(loss.item())
        losses.append(loss_f)
        if step % train_cfg.log_interval == 0:
            rec = {"step": step, "loss": loss_f, "lr": lr}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        if step % train_cfg.eval_interval == 0 and step > 0:
            val_loss = _eval_loss(model, valid_data, model_cfg, train_cfg, device)
            if val_loss < best_val:
                best_val = val_loss
                ckpt = paths["ckpt_dir"] / "best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_cfg": model_cfg.__dict__,
                        "tokenizer": tokenizer_name,
                        "step": step,
                        "val_loss": val_loss,
                    },
                    ckpt,
                )
        step += 1
        pbar.update(1)

    pbar.close()
    wall = timer() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    metrics = {
        "tokenizer": tokenizer_name,
        "train_wall_seconds": wall,
        "total_steps": total_steps,
        "tokens_per_second": (total_steps * train_cfg.batch_size * model_cfg.block_size) / wall,
        "peak_gpu_mem_mb": peak_mem,
        "final_train_loss": losses[-1] if losses else None,
        "best_val_loss": best_val,
        "num_train_tokens": int(len(train_data)),
        "smoke": smoke,
        "model_cfg": model_cfg.__dict__,
    }
    write_json(paths["metrics"], metrics)
    torch.save(
        {"model": model.state_dict(), "model_cfg": model_cfg.__dict__, "tokenizer": tokenizer_name},
        paths["ckpt_dir"] / "last.pt",
    )
    print(f"train {tokenizer_name} done in {wall:.1f}s, best_val={best_val:.4f}")
    return metrics


def _eval_loss(
    model: GPT,
    data: np.memmap,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(max_batches):
            try:
                x, y = get_batch(data, train_cfg.batch_size, model_cfg.block_size, device)
            except ValueError:
                break
            with autocast(enabled=train_cfg.use_amp and device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("inf")
