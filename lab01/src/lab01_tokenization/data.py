"""Download WikiText-103 raw and write cleaned splits."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from lab01_tokenization.config import (
    DATA_RAW,
    MIN_TEST_BYTES,
    WIKI_MARKUP,
    WIKITEXT_CONFIG,
    WIKITEXT_DATASET,
)
from lab01_tokenization.utils import read_text, write_json


def clean_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    for old, new in WIKI_MARKUP:
        s = s.replace(old, new)
    return s


def _write_split(ds_split, path: Path) -> tuple[int, int]:
    n_lines = 0
    nbytes = 0
    with path.open("w", encoding="utf-8") as f:
        for row in ds_split:
            text = row["text"] if isinstance(row, dict) else row
            line = clean_line(text)
            if not line:
                continue
            f.write(line + "\n")
            n_lines += 1
            nbytes += len(line.encode("utf-8")) + 1
    return n_lines, nbytes


def prepare_data(force: bool = False) -> dict[str, Path]:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": DATA_RAW / "train.txt",
        "validation": DATA_RAW / "valid.txt",
        "test": DATA_RAW / "test.txt",
    }
    if not force and all(p.exists() for p in paths.values()):
        _check_test_size(paths["test"])
        return paths

    ds = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG)
    stats: dict[str, dict[str, int]] = {}
    for split, path in paths.items():
        n_lines, nbytes = _write_split(ds[split], path)
        stats[split] = {"lines": n_lines, "bytes": nbytes}
        print(f"wrote {path} ({nbytes} bytes, {n_lines} lines)")

    meta = {
        "dataset": WIKITEXT_DATASET,
        "config": WIKITEXT_CONFIG,
        "splits": {k: str(v) for k, v in paths.items()},
        "stats": stats,
    }
    write_json(DATA_RAW / "meta.json", meta)
    _check_test_size(paths["test"])
    return paths


def _check_test_size(test_path: Path) -> None:
    size = test_path.stat().st_size
    if size < MIN_TEST_BYTES:
        raise RuntimeError(
            f"test split {test_path} is {size} bytes; need >= {MIN_TEST_BYTES}"
        )


def iter_split_lines(split: str):
    name = {"valid": "valid", "validation": "valid"}.get(split, split)
    path = DATA_RAW / f"{name}.txt"
    if not path.exists():
        prepare_data()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_split_lines(split: str) -> list[str]:
    name = {"valid": "valid", "validation": "valid"}.get(split, split)
    path = DATA_RAW / f"{name}.txt"
    if not path.exists():
        prepare_data()
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_split_text(split: str) -> str:
    name = {"valid": "valid", "validation": "valid"}.get(split, split)
    return read_text(DATA_RAW / f"{name}.txt")
