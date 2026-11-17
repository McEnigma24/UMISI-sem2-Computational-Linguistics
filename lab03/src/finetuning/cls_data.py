"""Tweet Eval sentiment — czyszczenie, tokenizacja, dynamic padding."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from _utils_.log_print import log_print

CLS_DATASET_NAME = "cardiffnlp/tweet_eval"
CLS_CONFIG = "sentiment"
NUM_LABELS = 3
LABEL_NAMES = ("negative", "neutral", "positive")


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t


@dataclass
class TokenizeSpec:
    kind: str  # hf | bpe
    encode: Callable[[str], list[int]]
    vocab_size: int
    pad_token_id: int
    max_len: int
    tokenizer_repr: str


class TweetSentimentDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        spec: TokenizeSpec,
    ) -> None:
        self.rows = rows
        self.spec = spec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        ids = self.spec.encode(row["text"])
        if len(ids) > self.spec.max_len:
            ids = ids[: self.spec.max_len]
        return {"input_ids": ids, "label": int(row["label"])}


def cls_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("pusty batch")
    pad_id = batch[0].get("pad_token_id")
    if pad_id is None:
        pad_id = 0
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(item["label"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def effective_max_examples(max_examples: int) -> int | None:
    """Jawnie w presecie: 0 = pełny split HF; >0 = limit przykładów."""
    if max_examples < 1:
        return None
    return max_examples


def _rows_from_split(split, *, max_examples: int) -> list[dict]:
    out: list[dict] = []
    for i, ex in enumerate(split):
        limit = effective_max_examples(max_examples)
        if limit is not None and i >= limit:
            break
        text = _clean_text(ex.get("text"))
        if not text:
            continue
        out.append({"text": text, "label": int(ex["label"])})
    return out


def _class_distribution(rows: list[dict]) -> dict[str, int]:
    c = Counter(int(r["label"]) for r in rows)
    return {LABEL_NAMES[k] if k < len(LABEL_NAMES) else str(k): int(v) for k, v in sorted(c.items())}


def _token_length_stats(rows: list[dict], encode: Callable[[str], list[int]]) -> dict[str, float]:
    lengths = [len(encode(r["text"])) for r in rows[: min(5000, len(rows))]]
    if not lengths:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    lengths.sort()
    mean = sum(lengths) / len(lengths)
    p95_idx = max(0, int(0.95 * len(lengths)) - 1)
    return {"mean": float(mean), "p95": float(lengths[p95_idx]), "max": float(lengths[-1])}


def _make_tokenize_spec(
    *,
    tokenizer_kind: str,
    max_len: int,
    hf_tokenizer=None,
    bpe_tokenizer=None,
) -> TokenizeSpec:
    if tokenizer_kind == "hf":
        if hf_tokenizer is None:
            raise ValueError("hf_tokenizer wymagany dla tokenizer_kind=hf")
        tok = hf_tokenizer
        if tok.pad_token_id is None:
            if tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
        pad_id = int(tok.pad_token_id)
        vocab = int(tok.vocab_size)

        def encode(text: str) -> list[int]:
            return tok.encode(text, add_special_tokens=True, truncation=False)

        return TokenizeSpec(
            kind="hf",
            encode=encode,
            vocab_size=vocab,
            pad_token_id=pad_id,
            max_len=max_len,
            tokenizer_repr=str(getattr(tok, "name_or_path", "hf")),
        )

    if bpe_tokenizer is None:
        from text_and_tokenizer import load_tokenizer

        bpe_tokenizer = load_tokenizer()
    pad_id = int(bpe_tokenizer.pad_token_id or 0)
    vocab = int(bpe_tokenizer.vocab_size)

    def encode_bpe(text: str) -> list[int]:
        return bpe_tokenizer.encode(text, add_special_tokens=True)

    return TokenizeSpec(
        kind="bpe",
        encode=encode_bpe,
        vocab_size=vocab,
        pad_token_id=pad_id,
        max_len=max_len,
        tokenizer_repr="text_and_tokenizer BPE",
    )


def prepare_cls_data(
    *,
    max_len: int,
    max_train_examples: int,
    max_val_examples: int,
    max_test_examples: int,
    tokenizer_kind: str,
    hf_tokenizer=None,
    bpe_tokenizer=None,
) -> tuple[TweetSentimentDataset, TweetSentimentDataset, TweetSentimentDataset, TokenizeSpec, dict]:
    log_print(f"HF dataset: {CLS_DATASET_NAME} config={CLS_CONFIG}")
    if max_train_examples < 1:
        log_print("Tweet Eval train: pełny split HF (max_train_examples=0)")
    else:
        log_print(f"Tweet Eval train: limit {max_train_examples:,} przykładów (max_train_examples)")
    if max_val_examples > 0:
        log_print(f"Tweet Eval val: limit {max_val_examples:,} (max_val_examples)")
    if max_test_examples > 0:
        log_print(f"Tweet Eval test: limit {max_test_examples:,} (max_test_examples)")
    ds = load_dataset(CLS_DATASET_NAME, CLS_CONFIG)
    train_rows = _rows_from_split(ds["train"], max_examples=max_train_examples)
    val_rows = _rows_from_split(ds["validation"], max_examples=max_val_examples)
    test_rows = _rows_from_split(ds["test"], max_examples=max_test_examples)

    spec = _make_tokenize_spec(
        tokenizer_kind=tokenizer_kind,
        max_len=max_len,
        hf_tokenizer=hf_tokenizer,
        bpe_tokenizer=bpe_tokenizer,
    )

    for split_name, rows in ("train", train_rows), ("validation", val_rows), ("test", test_rows):
        log_print(f"  {split_name}: {len(rows):,} przykładów, rozkład {_class_distribution(rows)}")

    sample_n = min(3, len(train_rows))
    for i in range(sample_n):
        ex = train_rows[i]
        log_print(f"  przykład[{i}]: label={LABEL_NAMES[ex['label']]} text={ex['text'][:120]!r}")

    train_ds = TweetSentimentDataset(train_rows, spec)
    val_ds = TweetSentimentDataset(val_rows, spec)
    test_ds = TweetSentimentDataset(test_rows, spec)

    truncated = 0
    for r in train_rows[: min(5000, len(train_rows))]:
        if len(spec.encode(r["text"])) > max_len:
            truncated += 1
    trunc_pct = 100.0 * truncated / max(1, min(5000, len(train_rows)))
    len_stats = _token_length_stats(train_rows, spec.encode)

    summary = dataset_summary(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        spec=spec,
        trunc_pct=trunc_pct,
        len_stats=len_stats,
        max_train_examples=max_train_examples,
        max_val_examples=max_val_examples,
        max_test_examples=max_test_examples,
    )
    return train_ds, val_ds, test_ds, spec, summary


def dataset_summary(
    *,
    train_ds: TweetSentimentDataset,
    val_ds: TweetSentimentDataset,
    test_ds: TweetSentimentDataset,
    spec: TokenizeSpec,
    trunc_pct: float,
    len_stats: dict[str, float],
    max_train_examples: int,
    max_val_examples: int,
    max_test_examples: int,
) -> dict[str, Any]:
    def _portion(limit: int, size: int) -> str:
        if limit < 1:
            return f"pełny split HF ({size:,} przykładów)"
        return f"limit {limit:,} (załadowano {size:,})"

    return {
        "hf_dataset": CLS_DATASET_NAME,
        "hf_config": CLS_CONFIG,
        "num_labels": NUM_LABELS,
        "label_names": list(LABEL_NAMES),
        "max_train_examples": max_train_examples,
        "max_val_examples": max_val_examples,
        "max_test_examples": max_test_examples,
        "train_portion": _portion(max_train_examples, len(train_ds)),
        "val_portion": _portion(max_val_examples, len(val_ds)),
        "test_portion": _portion(max_test_examples, len(test_ds)),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "train_class_dist": _class_distribution(train_ds.rows),
        "val_class_dist": _class_distribution(val_ds.rows),
        "max_len": spec.max_len,
        "tokenizer_kind": spec.kind,
        "tokenizer": spec.tokenizer_repr,
        "vocab_size": spec.vocab_size,
        "pad_token_id": spec.pad_token_id,
        "token_len_mean": len_stats["mean"],
        "token_len_p95": len_stats["p95"],
        "token_len_max": len_stats["max"],
        "truncated_pct_train_sample": trunc_pct,
    }


def collate_from_spec(spec: TokenizeSpec):
    def _fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        enriched = []
        for item in batch:
            d = dict(item)
            d["pad_token_id"] = spec.pad_token_id
            enriched.append(d)
        return cls_collate_fn(enriched)

    return _fn