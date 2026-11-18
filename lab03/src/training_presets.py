"""Presety hparamów: PLGrid (big) + lokalne GPU 2x RTX 3060 12 GB."""

from __future__ import annotations

_DUAL3060_BASE = dict(

    # FULL #
        max_corpus_chars=0,
        epochs=3,
        time_budget_min=90,

    # SMOKE #
        # max_corpus_chars=100_000,
        # epochs=3,
        # time_budget_min=5,


    checkpoint_every=1,
    eval_every=1,
    checkpoint_prune_post_training_mod=1,
    checkpoint_prune_post_training_tail=1,

    seq_len=512,
    window_stride=512,
    lightning_devices=2,
    batch_size=12,
    baseline_batch_size=24,
    baseline_learning_rate=6e-4,
    lr_scaling="sqrt",
    learning_rate_max=1.5e-3,
    val_checks_per_epoch=4,
    log_every_n_steps=50,
)

_TRANSFORMER_COMMON = dict(
    dropout=0.1,
    precision="bf16-mixed",
)

DUAL3060_LLAMA_110M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 768,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "ff_size": 2048,
}

DUAL3060_LLAMA_50M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 576,
    "hidden_size": 576,
    "num_layers": 8,
    "num_heads": 9,
    "ff_size": 1536,
}

DUAL3060_LLAMA_25M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 432,
    "hidden_size": 432,
    "num_layers": 5,
    "num_heads": 6,
    "ff_size": 1152,
}

DUAL3060_LLAMA_10M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 240,
    "hidden_size": 240,
    "num_layers": 3,
    "num_heads": 6,
    "ff_size": 768,
}

DUAL3060_LLAMA_5M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 144,
    "hidden_size": 144,
    "num_layers": 2,
    "num_heads": 4,
    "ff_size": 352,
}

DUAL3060_ALL: list[tuple[str, dict]] = [
    ("transformer_miniLLaMA", {"run_tag": "dual3060-5M", **DUAL3060_LLAMA_5M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-10M", **DUAL3060_LLAMA_10M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-25M", **DUAL3060_LLAMA_25M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-50M", **DUAL3060_LLAMA_50M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-110M", **DUAL3060_LLAMA_110M}),
]