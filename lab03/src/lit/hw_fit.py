"""Przedbieg batcha, wybór DDP/FSDP, skalowanie LR."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from _utils_.log_print import log_print
from lit.batch_fit_cache import (
    build_fingerprint,
    cache_path,
    get_probe_result,
    store_probe_result,
)
from models.transformer_miniLLaMA import transformer_miniLLaMA


@dataclass
class FitPlan:
    strategy: str
    per_gpu_batch: int
    global_batch: int
    reason: str
    fingerprint_short: str
    batch_source: str  # cache | probe | fixed


def scaled_learning_rate(
    *,
    baseline_batch_size: int,
    baseline_learning_rate: float,
    global_batch: int,
    lr_scaling: str,
    learning_rate_max: float,
) -> float:
    if baseline_batch_size < 1:
        raise ValueError("baseline_batch_size >= 1")
    ratio = global_batch / float(baseline_batch_size)
    if lr_scaling == "linear":
        lr = baseline_learning_rate * ratio
    elif lr_scaling == "sqrt":
        lr = baseline_learning_rate * math.sqrt(ratio)
    else:
        raise ValueError(f"lr_scaling: {lr_scaling!r}")
    return min(lr, learning_rate_max)


def _bytes_fit_on_one_gpu(model: nn.Module) -> int:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return 16 * trainable + 2 * frozen


def _model_code_paths(architecture: str) -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "models"
    if architecture == "transformer_miniLLaMA":
        return [root / "transformer_miniLLaMA.py", root / "_rope.py"]
    return [root / f"{architecture}.py"]


def _try_training_step(
    model: nn.Module,
    batch: tuple,
    *,
    precision: str,
    task: str,
) -> None:
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    opt.zero_grad(set_to_none=True)
    use_amp = "16" in precision or "bf16" in precision
    dtype = torch.bfloat16 if "bf16" in precision else torch.float16

    def _loss() -> torch.Tensor:
        if task == "cls":
            input_ids, attention_mask, labels = batch
            out = model(input_ids, attention_mask=attention_mask, labels=labels)
            if isinstance(out, dict):
                return out["loss"]
            return out
        x, y = batch
        if hasattr(model, "rnn"):
            logits, _ = model(x, None)
            return model.cost_function(logits, y.reshape(-1))
        out = model(x)
        return out if out.dim() == 0 else out.mean()

    first = batch[0]
    if use_amp and first.is_cuda:
        with torch.amp.autocast("cuda", dtype=dtype):
            loss = _loss()
    else:
        loss = _loss()
    loss.backward()
    opt.step()
    del loss, opt
    if first.is_cuda:
        torch.cuda.synchronize()


def probe_ddp_batch(
    model: nn.Module,
    *,
    task: str,
    seq_len: int,
    vocab_size: int,
    precision: str,
    batch_probe_max: int,
    device: torch.device,
    num_labels: int = 3,
) -> tuple[int, float, float]:
    dev_model = model.to(device)
    vocab = vocab_size
    lo, hi = 1, batch_probe_max
    best = 0
    t0 = time.perf_counter()
    peak = 0.0

    def attempt(b: int) -> bool:
        nonlocal peak
        torch.cuda.empty_cache()
        gc.collect()
        try:
            if task == "cls":
                x = torch.randint(0, vocab, (b, seq_len), device=device)
                mask = torch.ones(b, seq_len, device=device, dtype=torch.long)
                y = torch.randint(0, num_labels, (b,), device=device)
                _try_training_step(dev_model, (x, mask, y), precision=precision, task=task)
            else:
                x = torch.randint(0, vocab, (b, seq_len), device=device)
                y = torch.randint(0, vocab, (b, seq_len), device=device)
                _try_training_step(dev_model, (x, y), precision=precision, task=task)
            if device.type == "cuda":
                peak = max(peak, float(torch.cuda.max_memory_allocated(device)))
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    b = batch_probe_max
    while b >= 1 and not attempt(b):
        b //= 2
    if b < 1:
        model.cpu()
        return 0, peak, time.perf_counter() - t0
    lo, hi, best = 1, batch_probe_max, b
    while lo <= hi:
        mid = (lo + hi) // 2
        if attempt(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    model.cpu()
    torch.cuda.empty_cache()
    return best, peak, time.perf_counter() - t0


def choose_batch_and_strategy(
    model: nn.Module,
    *,
    architecture: str,
    task: str,
    seq_len: int,
    vocab_size: int,
    precision: str,
    strategy: str,
    ddp_min_per_gpu_batch: int,
    batch_size: int | None,
    batch_probe_max: int,
    batch_probe_safety: float,
    batch_probe_refresh: bool,
    lightning_devices: int,
    module_tree: str,
    code_paths: list[Path] | None = None,
    num_labels: int = 3,
) -> FitPlan:
    if batch_size is not None and batch_size > 0:
        return FitPlan(
            strategy=strategy if strategy != "auto" else "ddp",
            per_gpu_batch=batch_size,
            global_batch=batch_size * lightning_devices,
            reason="batch_size stały z presetu",
            fingerprint_short="-",
            batch_source="fixed",
        )

    fp = build_fingerprint(
        model,
        task=task,
        seq_len=seq_len,
        precision=precision,
        strategy=strategy,
        devices=lightning_devices,
        code_paths=code_paths or _model_code_paths(architecture),
    )
    cpath = cache_path(architecture, fp)

    probe_source = "probe"

    def run_probe(mode: str) -> int:
        nonlocal probe_source
        cached, src = get_probe_result(cpath, mode, probe_max=batch_probe_max, refresh=batch_probe_refresh)
        if cached is not None:
            probe_source = "cache"
            log_print(f"batch fit: cache hit {fp.short} ({mode}={cached}/GPU)")
            return cached
        probe_source = "probe"
        log_print(f"batch fit: cache miss {fp.short} — przedbieg {mode}")
        dev = torch.device("cuda:0")
        raw, peak, secs = probe_ddp_batch(
            model,
            task=task,
            seq_len=seq_len,
            vocab_size=vocab_size,
            precision=precision,
            batch_probe_max=batch_probe_max,
            device=dev,
            num_labels=num_labels,
        )
        store_probe_result(
            cpath,
            fingerprint=fp,
            mode=mode,
            max_per_gpu_batch=raw,
            probe_max=batch_probe_max,
            peak_mem_bytes=peak,
            seconds=secs,
            module_tree=module_tree,
        )
        return raw

    bytes_one = _bytes_fit_on_one_gpu(model)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    force_fsdp = bytes_one > int(gpu_mem * 0.85)

    chosen = strategy
    reason = ""
    raw_batch = 0

    if strategy == "fsdp" or (strategy == "auto" and force_fsdp):
        chosen = "fsdp"
        raw_batch = run_probe("fsdp") if lightning_devices > 1 else run_probe("ddp")
        reason = "model nie mieści się w pełnej kopii na karcie" if force_fsdp else "wymuszony fsdp"
    elif strategy == "ddp":
        chosen = "ddp"
        raw_batch = run_probe("ddp")
        reason = "wymuszony ddp"
    else:
        raw_ddp = run_probe("ddp")
        if raw_ddp >= ddp_min_per_gpu_batch:
            chosen, raw_batch, reason = "ddp", raw_ddp, f"przedbieg DDP ≥ {ddp_min_per_gpu_batch}"
        else:
            chosen = "fsdp"
            raw_batch = run_probe("fsdp")
            reason = f"DDP batch={raw_ddp} < {ddp_min_per_gpu_batch} → FSDP"

    per_gpu = max(1, int(raw_batch * batch_probe_safety))
    if per_gpu < 1:
        raise RuntimeError(
            f"Żaden batch się nie zmieścił (arch={architecture}, raw={raw_batch}). "
            "Zmniejsz seq_len albo model."
        )
    return FitPlan(
        strategy=chosen,
        per_gpu_batch=per_gpu,
        global_batch=per_gpu * lightning_devices,
        reason=reason,
        fingerprint_short=fp.short,
        batch_source=probe_source,
    )