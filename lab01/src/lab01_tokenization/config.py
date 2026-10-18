"""Paths and hyperparameters for lab01."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_BINS = ROOT / "data" / "bins"
DATA_TOKENIZERS = ROOT / "data" / "tokenizers"
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"
LOGS = RESULTS / "logs"

VOCAB_SIZE = 50257
TOKENIZER_NAMES = ("gpt2", "whitespace", "spm")

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
MIN_TEST_BYTES = 1_000_000

WIKI_MARKUP = (
    ("@-@", "-"),
    ("@,@", ","),
    ("@.@", "."),
)


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    n_ff: int = 2048
    block_size: int = 256
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE

    @property
    def n_params_approx(self) -> int:
        # tied embeddings + transformer blocks (rough)
        v = self.vocab_size
        d = self.n_embd
        block = 12 * d * d + 13 * d * self.n_ff
        return v * d + self.n_layer * block + d


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_epochs: int = 1
    eval_interval: int = 500
    log_interval: int = 50
    seed: int = 42
    use_amp: bool = True


@dataclass(frozen=True)
class EvalConfig:
    block_size: int = 256
    stride: int = 128
    batch_size: int = 8


def paths_for_tokenizer(name: str) -> dict[str, Path]:
    return {
        "bin_train": DATA_BINS / name / "train.bin",
        "bin_valid": DATA_BINS / name / "valid.bin",
        "bin_test": DATA_BINS / name / "test.bin",
        "meta": DATA_BINS / name / "meta.json",
        "ckpt_dir": CHECKPOINTS / name,
        "metrics": RESULTS / f"metrics_{name}.json",
    }
