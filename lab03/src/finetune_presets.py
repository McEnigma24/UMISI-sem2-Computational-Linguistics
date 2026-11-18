"""Presety fine-tuningu klasyfikacji Tweet Eval sentiment."""

from __future__ import annotations

_CLS_BASE: dict = dict(

    dataset="tweet_eval_sentiment",

    # FULL #
        max_train_examples=0,   # 0 = pełny train HF; >0 = limit (smoke)
        max_val_examples=0,     # 0 = pełna walidacja
        max_test_examples=0,    # 0 = pełny test; >0 skraca ewaluację i część inferencji

        epochs=100,
        time_budget_min=45.0,


    # SMOKE #
        # max_train_examples=800,
        # max_val_examples=400,
        # max_test_examples=500,

        # epochs=3,
        # time_budget_min=5.0,


    lightning_devices=2,
    max_len=128,
    batch_size=32,
    lr_scaling="sqrt",
    precision="bf16-mixed",
    early_stopping_patience=2,
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
)

FT_SCRATCH_5M: dict = {
    **_CLS_BASE,
    "source": "scratch",
    "run_tag": "cls-scratch-5M",
    "finetune_mode": "full",
    "embed_dim": 144,
    "hidden_size": 144,
    "num_layers": 2,
    "num_heads": 4,
    "ff_size": 352,
    "dropout": 0.1,
    "baseline_batch_size": 64,
    "baseline_learning_rate": 5e-4,
    "learning_rate_max": 1e-3,
}

FT_PT_5M: dict = {
    **_CLS_BASE,
    "source": "pt-latest:dual3060-5M",
    "run_tag": "cls-pt-5M",
    "finetune_mode": "full",
    "baseline_batch_size": 64,
    "baseline_learning_rate": 2e-4,
    "learning_rate_max": 8e-4,
}

FT_PT_10M: dict = {
    **_CLS_BASE,
    "source": "pt-latest:dual3060-10M",
    "run_tag": "cls-pt-10M",
    "finetune_mode": "full",
    "baseline_batch_size": 64,
    "baseline_learning_rate": 2e-4,
    "learning_rate_max": 8e-4,
}

FT_PT_25M: dict = {
    **_CLS_BASE,
    "source": "pt-latest:dual3060-25M",
    "run_tag": "cls-pt-25M",
    "finetune_mode": "full",
    "baseline_batch_size": 64,
    "baseline_learning_rate": 1e-4,
    "learning_rate_max": 5e-4,
}

FT_PT_50M: dict = {
    **_CLS_BASE,
    "source": "pt-latest:dual3060-50M",
    "run_tag": "cls-pt-50M",
    "finetune_mode": "full",
    "baseline_batch_size": 64,
    "baseline_learning_rate": 7e-5,
    "learning_rate_max": 3e-4,
}

FT_PT_110M: dict = {
    **_CLS_BASE,
    "source": "pt-latest:dual3060-110M",
    "run_tag": "cls-pt-110M",
    "finetune_mode": "full",
    "baseline_batch_size": 64,
    "baseline_learning_rate": 5e-5,
    "learning_rate_max": 2e-4,
}

FT_HF_GPT2_LORA: dict = {
    **_CLS_BASE,
    "source": "hf:openai-community/gpt2",
    "run_tag": "cls-gpt2-lora",
    "finetune_mode": "lora",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "baseline_batch_size": 64,
    "baseline_learning_rate": 2e-4,
    "learning_rate_max": 5e-4,
}

FT_HF_GPT2_FULL: dict = {
    **_CLS_BASE,
    "source": "hf:openai-community/gpt2",
    "run_tag": "cls-gpt2-full",
    "finetune_mode": "full",
    "baseline_batch_size": 64,
    "baseline_learning_rate": 2e-5,
    "learning_rate_max": 1e-4,
}

FINETUNE_ALL: list[dict] = [

    FT_SCRATCH_5M,
    FT_PT_5M,
    FT_PT_10M,
    FT_PT_25M,
    FT_PT_50M,
    FT_PT_110M,
    FT_HF_GPT2_LORA,
    FT_HF_GPT2_FULL,
]
