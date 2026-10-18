"""Tokenizer implementations (name avoids clash with HF `tokenizers` package)."""

from lab01_tokenization.tokenizers_impl.base import BaseTokenizer
from lab01_tokenization.tokenizers_impl.gpt2 import GPT2Wrapper
from lab01_tokenization.tokenizers_impl.spm import SPMTokenizer
from lab01_tokenization.tokenizers_impl.whitespace import WhitespaceTokenizer

__all__ = [
    "BaseTokenizer",
    "GPT2Wrapper",
    "WhitespaceTokenizer",
    "SPMTokenizer",
    "load_tokenizer",
    "train_all_tokenizers",
]


def load_tokenizer(name: str) -> BaseTokenizer:
    if name == "gpt2":
        return GPT2Wrapper.load()
    if name == "whitespace":
        return WhitespaceTokenizer.load()
    if name == "spm":
        return SPMTokenizer.load()
    raise ValueError(f"unknown tokenizer: {name}")


def train_all_tokenizers(force: bool = False) -> dict[str, float]:
    from lab01_tokenization.tokenizers_impl import gpt2, spm, whitespace

    times: dict[str, float] = {}
    times["gpt2"] = gpt2.ensure_pretrained(force=force)
    times["whitespace"] = whitespace.train_and_save(force=force)
    times["spm"] = spm.train_and_save(force=force)
    return times
