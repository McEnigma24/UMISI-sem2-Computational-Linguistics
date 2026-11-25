"""Presety fine-tuningu: siatka metod × backbone (Tweet Eval, 5 epok)."""

from __future__ import annotations

import math
from typing import Any

_CLS_BASE: dict[str, Any] = dict(
    dataset="tweet_eval_sentiment",
    max_train_examples=0,
    max_val_examples=0,
    max_test_examples=0,
    epochs=5,
    time_budget_min=120.0,
    lightning_devices=2,
    max_len=128,
    batch_size=32,
    lr_scaling="sqrt",
    precision="bf16-mixed",
    warmup_ratio=0.06,
    max_grad_norm=1.0,
    weight_decay=0.01,
    pooling="last",
    head_hidden=256,
    head_dropout=0.1,
    inference_bench_examples=1000,
    val_checks_per_epoch=1,
    log_every_n_steps=50,
    unfreeze_last_n=None,
    lora_r=None,
    lora_alpha=None,
    lora_dropout=None,
    baseline_batch_size=64,
    learning_rate_max=1e-3,
)

UNFREEZE_FRACTIONS = (0.1, 0.2, 0.3, 0.5)

_PRETRAIN_ROOT = "saved_models/pretrain/transformer_miniLLaMA"
PRETRAINED_LM: dict[str, dict[str, Any]] = {
    "dual3060-5M": {
        "pt_path": (
            f"{_PRETRAIN_ROOT}/8205087482___dual3060-5M___2026-11-17_11-48/"
            "transformer_miniLLaMA_best_eval.pt"
        ),
        "num_layers": 2,
        "baseline_learning_rate": 2e-4,
        "learning_rate_max": 8e-4,
    },
    "dual3060-10M": {
        "pt_path": (
            f"{_PRETRAIN_ROOT}/8205085591___dual3060-10M___2026-11-17_12-20/"
            "transformer_miniLLaMA_best_eval.pt"
        ),
        "num_layers": 3,
        "baseline_learning_rate": 2e-4,
        "learning_rate_max": 8e-4,
    },
    "dual3060-25M": {
        "pt_path": (
            f"{_PRETRAIN_ROOT}/8205083231___dual3060-25M___2026-11-17_12-59/"
            "transformer_miniLLaMA_best_eval.pt"
        ),
        "num_layers": 5,
        "baseline_learning_rate": 1e-4,
        "learning_rate_max": 5e-4,
    },
    "dual3060-50M": {
        "pt_path": (
            f"{_PRETRAIN_ROOT}/8205078901___dual3060-50M___2026-11-17_14-11/"
            "transformer_miniLLaMA_best_eval.pt"
        ),
        "num_layers": 8,
        "baseline_learning_rate": 7e-5,
        "learning_rate_max": 3e-4,
    },
    "dual3060-110M": {
        "pt_path": (
            f"{_PRETRAIN_ROOT}/8205073474___dual3060-110M___2026-11-17_15-42/"
            "transformer_miniLLaMA_best_eval.pt"
        ),
        "num_layers": 12,
        "baseline_learning_rate": 5e-5,
        "learning_rate_max": 2e-4,
    },
}

_GPT2_ID = "hf:openai-community/gpt2"
_GPT2_LAYERS = 12

_LORA_PT = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, baseline_learning_rate=5e-4, learning_rate_max=5e-4)
_LORA_GPT2 = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, baseline_learning_rate=2e-4, learning_rate_max=5e-4)
_HEAD_LR = dict(baseline_learning_rate=1e-3, learning_rate_max=1e-3)

_SCRATCH_DIMS = dict(
    embed_dim=144,
    hidden_size=144,
    num_layers=2,
    num_heads=4,
    ff_size=352,
    dropout=0.1,
    baseline_learning_rate=5e-4,
    learning_rate_max=1e-3,
)


def _unique_unfreeze_ks(num_layers: int) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for frac in UNFREEZE_FRACTIONS:
        k = max(1, math.ceil(frac * num_layers))
        out.setdefault(k, []).append(frac)
    return out


def _preset(**kwargs: Any) -> dict[str, Any]:
    p = dict(_CLS_BASE)
    p.update(kwargs)
    return p


def _pt_presets(lm_tag: str, *, finetune_mode: str, run_suffix: str, extra: dict | None = None) -> dict:
    meta = PRETRAINED_LM[lm_tag]
    short = lm_tag.replace("dual3060-", "pt")
    base = _preset(
        source=f"pt:{meta['pt_path']}",
        run_tag=f"ft-{short}-{run_suffix}",
        finetune_mode=finetune_mode,
        baseline_learning_rate=meta["baseline_learning_rate"],
        learning_rate_max=meta["learning_rate_max"],
    )
    if extra:
        base.update(extra)
    return base


def _gpt2_preset(*, finetune_mode: str, run_suffix: str, extra: dict | None = None) -> dict:
    base = _preset(
        source=_GPT2_ID,
        run_tag=f"ft-gpt2-{run_suffix}",
        finetune_mode=finetune_mode,
        baseline_learning_rate=2e-5,
        learning_rate_max=1e-4,
    )
    if extra:
        base.update(extra)
    return base


def _build_grid() -> list[dict]:
    grid: list[dict] = []

    grid.append(
        _preset(
            source="scratch",
            run_tag="ft-scratch-5M-full",
            finetune_mode="full",
            **_SCRATCH_DIMS,
        )
    )
    grid.append(_gpt2_preset(finetune_mode="full", run_suffix="full"))

    for lm_tag in PRETRAINED_LM:
        grid.append(_pt_presets(lm_tag, finetune_mode="full", run_suffix="full"))

    grid.append(_gpt2_preset(finetune_mode="lora", run_suffix="lora", extra=_LORA_GPT2))
    for lm_tag in PRETRAINED_LM:
        grid.append(_pt_presets(lm_tag, finetune_mode="lora", run_suffix="lora", extra=_LORA_PT))

    grid.append(_gpt2_preset(finetune_mode="head_only", run_suffix="head", extra=_HEAD_LR))
    for lm_tag in PRETRAINED_LM:
        grid.append(_pt_presets(lm_tag, finetune_mode="head_only", run_suffix="head", extra=_HEAD_LR))

    for lm_tag, meta in PRETRAINED_LM.items():
        short = lm_tag.replace("dual3060-", "pt")
        L = int(meta["num_layers"])
        for k, fracs in sorted(_unique_unfreeze_ks(L).items()):
            grid.append(
                _pt_presets(
                    lm_tag,
                    finetune_mode="unfreeze_last",
                    run_suffix=f"top{k}of{L}",
                    extra={
                        "unfreeze_last_n": k,
                        "unfreeze_fractions": fracs,
                        "backbone_num_layers": L,
                    },
                )
            )

    for k, fracs in sorted(_unique_unfreeze_ks(_GPT2_LAYERS).items()):
        grid.append(
            _gpt2_preset(
                finetune_mode="unfreeze_last",
                run_suffix=f"top{k}of{_GPT2_LAYERS}",
                extra={
                    "unfreeze_last_n": k,
                    "unfreeze_fractions": fracs,
                    "backbone_num_layers": _GPT2_LAYERS,
                },
            )
        )

    return grid


FINETUNE_ALL: list[dict] = _build_grid()

_ABLATION_TRAIN = (2000, 10_000)
_FINETUNE_DATA_ABLATION: list[dict] = []
for n_train in _ABLATION_TRAIN:
    _FINETUNE_DATA_ABLATION.append(
        _preset(
            source="scratch",
            run_tag=f"ft-scratch-5M-full-train{n_train}",
            finetune_mode="full",
            max_train_examples=n_train,
            **_SCRATCH_DIMS,
        )
    )
    _FINETUNE_DATA_ABLATION.append(
        _pt_presets(
            "dual3060-110M",
            finetune_mode="full",
            run_suffix=f"full-train{n_train}",
            extra={"max_train_examples": n_train},
        )
    )
    _FINETUNE_DATA_ABLATION.append(
        _gpt2_preset(
            finetune_mode="full",
            run_suffix=f"full-train{n_train}",
            extra={"max_train_examples": n_train},
        )
    )
    _FINETUNE_DATA_ABLATION.append(
        _gpt2_preset(
            finetune_mode="lora",
            run_suffix=f"lora-train{n_train}",
            extra={**_LORA_GPT2, "max_train_examples": n_train},
        )
    )

FINETUNE_QUEUE: list[dict] = FINETUNE_ALL + _FINETUNE_DATA_ABLATION