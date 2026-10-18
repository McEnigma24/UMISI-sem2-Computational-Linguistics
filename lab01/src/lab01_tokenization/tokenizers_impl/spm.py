"""SentencePiece subword tokenizer."""

from __future__ import annotations

import time
from pathlib import Path

import sentencepiece as spm

from lab01_tokenization.config import DATA_RAW, DATA_TOKENIZERS, VOCAB_SIZE
from lab01_tokenization.tokenizers_impl.base import BaseTokenizer
from lab01_tokenization.utils import write_json

MODEL_PREFIX = DATA_TOKENIZERS / "spm"
MODEL_PATH = MODEL_PREFIX.with_suffix(".model")


class SPMTokenizer(BaseTokenizer):
    name = "spm"

    def __init__(self, sp: spm.SentencePieceProcessor) -> None:
        self._sp = sp

    @classmethod
    def load(cls) -> SPMTokenizer:
        if not MODEL_PATH.exists():
            train_and_save()
        sp = spm.SentencePieceProcessor()
        sp.load(str(MODEL_PATH))
        return cls(sp)

    def encode(self, text: str) -> list[int]:
        return list(self._sp.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)

    def tokenize(self, text: str) -> list[str]:
        return self._sp.encode(text, out_type=str)

    @property
    def eos_id(self) -> int:
        return self._sp.eos_id()

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    def is_direct(self, word: str) -> bool:
        return len(self.encode(word)) == 1


def train_and_save(algorithm: str = "unigram", force: bool = False) -> float:
    DATA_TOKENIZERS.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force:
        return 0.0
    corpus = DATA_RAW / "train.txt"
    if not corpus.exists():
        from lab01_tokenization.data import prepare_data

        prepare_data()

    t0 = time.perf_counter()
    # Subsample sentences to keep RAM bounded on login nodes; still trains on full distribution.
    sentence_sample = 150_000

    algo_flag = "unigram" if algorithm == "unigram" else "bpe"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type=algo_flag,
        character_coverage=0.9999,
        input_sentence_size=sentence_sample,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        max_sentence_length=8192,
        num_threads=4,
        bos_id=-1,
        eos_id=0,
        pad_id=1,
        unk_id=2,
    )
    elapsed = time.perf_counter() - t0
    write_json(
        DATA_TOKENIZERS / "spm_meta.json",
        {
            "vocab_size": VOCAB_SIZE,
            "algorithm": algo_flag,
            "model": str(MODEL_PATH),
            "train_wall_seconds": elapsed,
            "input_sentence_size": sentence_sample,
        },
    )
    return elapsed
