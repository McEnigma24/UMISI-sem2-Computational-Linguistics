"""Encode text splits to uint16 token binaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lab01_tokenization.config import TOKENIZER_NAMES, paths_for_tokenizer
from lab01_tokenization.data import iter_split_lines
from lab01_tokenization.tokenizers_impl import load_tokenizer
from lab01_tokenization.utils import write_json

_FLUSH = 500_000


def encode_split(
    tokenizer_name: str,
    split: str,
    max_train_mb: float | None = None,
) -> tuple[Path, int]:
    tok = load_tokenizer(tokenizer_name)
    paths = paths_for_tokenizer(tokenizer_name)
    out_path = {
        "train": paths["bin_train"],
        "validation": paths["bin_valid"],
        "valid": paths["bin_valid"],
        "test": paths["bin_test"],
    }[split]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids: list[int] = []
    total = 0
    max_id = 0
    budget = int(max_train_mb * 1_000_000) if split == "train" and max_train_mb else None
    used = 0

    with out_path.open("wb") as fout:
        for line in iter_split_lines(split):
            if budget is not None:
                b = len(line.encode("utf-8")) + 1
                if used + b > budget:
                    break
                used += b
            ids.extend(tok.encode_line(line))
            if len(ids) >= _FLUSH:
                max_id = max(max_id, max(ids))
                if max_id >= 65536:
                    raise ValueError(f"token id {max_id} exceeds uint16 for {tokenizer_name}")
                np.array(ids, dtype=np.uint16).tofile(fout)
                total += len(ids)
                ids.clear()

        if ids:
            max_id = max(max_id, max(ids))
            if max_id >= 65536:
                raise ValueError(f"token id {max_id} exceeds uint16 for {tokenizer_name}")
            np.array(ids, dtype=np.uint16).tofile(fout)
            total += len(ids)

    return out_path, total


def encode_all(tokenizer_name: str | None = None, max_train_mb: float | None = None) -> None:
    names = [tokenizer_name] if tokenizer_name else list(TOKENIZER_NAMES)
    for name in names:
        meta = {"tokenizer": name, "splits": {}}
        for split in ("train", "validation", "test"):
            path, n = encode_split(name, split, max_train_mb=max_train_mb if split == "train" else None)
            meta["splits"][split] = {"path": str(path), "num_tokens": n}
        tok = load_tokenizer(name)
        meta["vocab_size"] = tok.vocab_size
        meta["eos_id"] = tok.eos_id
        write_json(paths_for_tokenizer(name)["meta"], meta)
        print(f"encoded {name}: {meta['splits']}")
