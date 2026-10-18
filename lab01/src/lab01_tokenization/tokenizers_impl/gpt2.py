"""Pre-trained GPT-2 BPE tokenizer."""

from __future__ import annotations

import time

from transformers import GPT2TokenizerFast

from lab01_tokenization.config import DATA_TOKENIZERS, VOCAB_SIZE
from lab01_tokenization.tokenizers_impl.base import BaseTokenizer
from lab01_tokenization.utils import write_json


class GPT2Wrapper(BaseTokenizer):
    name = "gpt2"

    def __init__(self, tok: GPT2TokenizerFast) -> None:
        self._tok = tok
        if tok.eos_token_id is None:
            raise ValueError("GPT-2 tokenizer missing eos_token_id")

    @classmethod
    def load(cls) -> GPT2Wrapper:
        path = DATA_TOKENIZERS / "gpt2"
        if path.exists():
            return cls(GPT2TokenizerFast.from_pretrained(str(path)))
        return cls(GPT2TokenizerFast.from_pretrained("gpt2"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def tokenize(self, text: str) -> list[str]:
        return self._tok.tokenize(text)

    @property
    def eos_id(self) -> int:
        return int(self._tok.eos_token_id)

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def is_direct(self, word: str) -> bool:
        return len(self.encode(word)) == 1


def ensure_pretrained(force: bool = False) -> float:
    DATA_TOKENIZERS.mkdir(parents=True, exist_ok=True)
    out = DATA_TOKENIZERS / "gpt2"
    if out.exists() and not force:
        return 0.0
    t0 = time.perf_counter()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    out = DATA_TOKENIZERS / "gpt2"
    tok.save_pretrained(str(out))
    elapsed = time.perf_counter() - t0
    write_json(
        DATA_TOKENIZERS / "gpt2_meta.json",
        {
            "vocab_size": len(tok),
            "target_vocab_size": VOCAB_SIZE,
            "train_wall_seconds": elapsed,
        },
    )
    return elapsed
