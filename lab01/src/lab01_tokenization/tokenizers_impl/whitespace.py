"""Whitespace + punctuation tokenizer with fixed dictionary."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from lab01_tokenization.config import DATA_RAW, DATA_TOKENIZERS, VOCAB_SIZE
from lab01_tokenization.tokenizers_impl.base import BaseTokenizer
from lab01_tokenization.utils import split_pieces, write_json

PAD = "<PAD>"
EOS = "<EOS>"
UNK = "<UNK>"


class WhitespaceTokenizer(BaseTokenizer):
    name = "whitespace"

    def __init__(
        self,
        word_to_id: dict[str, int],
        id_to_word: dict[int, str],
        unk_id: int,
        eos_id: int,
        pad_id: int,
    ) -> None:
        self._word_to_id = word_to_id
        self._id_to_word = id_to_word
        self._unk_id = unk_id
        self._eos_id = eos_id
        self._pad_id = pad_id

    @classmethod
    def load(cls) -> WhitespaceTokenizer:
        path = DATA_TOKENIZERS / "whitespace_vocab.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        word_to_id = data["word_to_id"]
        id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
        return cls(
            word_to_id=word_to_id,
            id_to_word=id_to_word,
            unk_id=data["unk_id"],
            eos_id=data["eos_id"],
            pad_id=data["pad_id"],
        )

    def encode(self, text: str) -> list[int]:
        return [self._word_to_id.get(p, self._unk_id) for p in split_pieces(text)]

    def decode(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            if i in (self._eos_id, self._pad_id):
                continue
            w = self._id_to_word.get(i, UNK)
            if w == UNK:
                parts.append(UNK)
            elif len(w) == 1 and not w.isalnum():
                parts.append(w)
            else:
                if parts and parts[-1] not in ("", UNK) and not parts[-1][-1].isspace():
                    if parts[-1][-1].isalnum() and w[0].isalnum():
                        parts.append(" ")
                parts.append(w)
        return "".join(parts).replace(f" {UNK}", UNK)

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for p in split_pieces(text):
            out.append(p if p in self._word_to_id else UNK)
        return out

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_word)

    def is_direct(self, word: str) -> bool:
        return word in self._word_to_id

    def is_unk_id(self, token_id: int) -> bool:
        return token_id == self._unk_id


def train_and_save(force: bool = False) -> float:
    DATA_TOKENIZERS.mkdir(parents=True, exist_ok=True)
    out_path = DATA_TOKENIZERS / "whitespace_vocab.json"
    if out_path.exists() and not force:
        return 0.0
    t0 = time.perf_counter()
    corpus = DATA_RAW / "train.txt"
    if not corpus.exists():
        from lab01_tokenization.data import prepare_data

        prepare_data()
    counter: Counter[str] = Counter()
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                counter.update(split_pieces(line))

    specials = [PAD, EOS, UNK]
    top_n = VOCAB_SIZE - len(specials)
    most_common = [w for w, _ in counter.most_common(top_n)]

    vocab: list[str] = specials + most_common
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}

    payload = {
        "unk_id": word_to_id[UNK],
        "eos_id": word_to_id[EOS],
        "pad_id": word_to_id[PAD],
        "word_to_id": word_to_id,
        "id_to_word": {str(k): v for k, v in id_to_word.items()},
        "vocab_size": len(vocab),
        "train_corpus": str(DATA_RAW / "train.txt"),
    }
    elapsed = time.perf_counter() - t0
    payload["train_wall_seconds"] = elapsed
    write_json(DATA_TOKENIZERS / "whitespace_vocab.json", payload)
    return elapsed
