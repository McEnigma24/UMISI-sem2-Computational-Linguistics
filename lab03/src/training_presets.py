"""Presety hparamów: PLGrid (big) + lokalne GPU 2x RTX 3060 12 GB."""

from __future__ import annotations

_DUAL3060_BASE = dict(
    lightning_devices=2,

    checkpoint_every=1,
    eval_every=1,
    checkpoint_prune_post_training_mod=1,
    checkpoint_prune_post_training_tail=1,
    max_corpus_chars=0,  # 0 = pełny split train WikiText; >0 = limit znaków (testy)

    seq_len=512,
    window_stride=512,

    epochs=3,               # max epochs - or interrupted by time_budget
    time_budget_min=90,     # max training time
)

_TRANSFORMER_COMMON = dict(
    dropout=0.1,
    batch_size=12,              #   x2 bo tyle na 1 kartę idzie
    precision="bf16-mixed",
    learning_rate=6e-4,
)



# ~110M param. (vocab=32k, tied emb.) — transformer_miniLLaMA
DUAL3060_LLAMA_110M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 768,
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "ff_size": 2048,
}

# ~50M param. (vocab=32k, tied emb.) — transformer_miniLLaMA
DUAL3060_LLAMA_50M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 576,
    "hidden_size": 576,
    "num_layers": 8,
    "num_heads": 9,
    "ff_size": 1536,
}

# ~25M param. (vocab=32k, tied emb.)
DUAL3060_LLAMA_25M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 432,
    "hidden_size": 432,
    "num_layers": 5,
    "num_heads": 6,
    "ff_size": 1152,
}

# ~10M param. (vocab=32k, tied emb.)
DUAL3060_LLAMA_10M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 240,
    "hidden_size": 240,
    "num_layers": 3,
    "num_heads": 6,
    "ff_size": 768,
}

# ~5M param. (vocab=32k, tied emb.)
DUAL3060_LLAMA_5M: dict = {
    **_DUAL3060_BASE,
    **_TRANSFORMER_COMMON,
    "embed_dim": 144,
    "hidden_size": 144,
    "num_layers": 2,
    "num_heads": 4,
    "ff_size": 352,
}


# DUAL3060_GPT2: dict = {
#     **_DUAL3060_BASE,
#     **_TRANSFORMER_COMMON,
#     "embed_dim": 768,
#     "hidden_size": 768,
#     "num_layers": 12,
#     "num_heads": 12,
#     "ff_size": 3072,
# }

# DUAL3060_NANO: dict = {
#     **_DUAL3060_BASE,
#     **_TRANSFORMER_COMMON,
#     "embed_dim": 1024,
#     "hidden_size": 1024,
#     "num_layers": 6,
#     "num_heads": 16,
#     "ff_size": 4096,
# }


DUAL3060_ALL: list[tuple[str, dict]] = [

    # ("transformer_GPT2small", DUAL3060_GPT2),
    # ("transformer_nanoGPT", DUAL3060_NANO),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-5M", **DUAL3060_LLAMA_5M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-10M", **DUAL3060_LLAMA_10M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-25M", **DUAL3060_LLAMA_25M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-50M", **DUAL3060_LLAMA_50M}),
    ("transformer_miniLLaMA", {"run_tag": "dual3060-110M", **DUAL3060_LLAMA_110M}),
]
