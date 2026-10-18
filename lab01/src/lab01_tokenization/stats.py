"""Tokenization statistics on the test split."""

from __future__ import annotations

import time

from lab01_tokenization.config import LOGS, RESULTS, TOKENIZER_NAMES
from lab01_tokenization.data import load_split_text
from lab01_tokenization.tokenizers_impl import load_tokenizer
from lab01_tokenization.tokenizers_impl.whitespace import WhitespaceTokenizer
from lab01_tokenization.utils import count_words, iter_words, read_json, timer, write_json


def _throughput(tok, text: str) -> dict[str, float]:
    nbytes = len(text.encode("utf-8"))
    nwords = count_words(text)
    t0 = timer()
    ntok = 0
    for line in text.splitlines():
        line = line.strip()
        if line:
            ntok += len(tok.encode_line(line))
    elapsed = timer() - t0
    return {
        "encode_seconds": elapsed,
        "tokens_per_second": ntok / elapsed if elapsed > 0 else 0.0,
        "mb_per_second": (nbytes / 1_000_000) / elapsed if elapsed > 0 else 0.0,
        "words_per_second": nwords / elapsed if elapsed > 0 else 0.0,
        "num_tokens": ntok,
        "num_bytes": nbytes,
        "num_words": nwords,
    }


def compute_stats(tokenizer_name: str | None = None) -> dict:
    text = load_split_text("test")
    names = [tokenizer_name] if tokenizer_name else list(TOKENIZER_NAMES)
    all_stats: dict[str, dict] = {}

    for name in names:
        tok = load_tokenizer(name)
        total_tokens = 0
        total_words = 0
        direct_occ = 0
        direct_types: set[str] = set()
        all_types: set[str] = set()
        oov_occ = 0
        oov_types: set[str] = set()
        for word in iter_words(text):
            total_words += 1
            all_types.add(word)
            total_tokens += len(tok.encode(word))

            if tok.is_direct(word):
                direct_occ += 1
                direct_types.add(word)
            elif isinstance(tok, WhitespaceTokenizer):
                oov_occ += 1
                oov_types.add(word)

        throughput = _throughput(tok, text)
        stats = {
            "tokenizer": name,
            "vocab_size": tok.vocab_size,
            "tokens_per_word": total_tokens / total_words if total_words else 0.0,
            "direct_word_occurrence_pct": 100.0 * direct_occ / total_words if total_words else 0.0,
            "direct_word_type_pct": 100.0 * len(direct_types) / len(all_types) if all_types else 0.0,
            "words_in_dictionary_occurrences": direct_occ,
            "unique_word_types": len(all_types),
            "unique_direct_types": len(direct_types),
            "throughput": throughput,
        }
        if isinstance(tok, WhitespaceTokenizer):
            stats["oov_occurrences"] = oov_occ
            stats["oov_occurrence_pct"] = 100.0 * oov_occ / total_words if total_words else 0.0
            stats["oov_unique_types"] = len(oov_types)
            stats["oov_type_pct"] = 100.0 * len(oov_types) / len(all_types) if all_types else 0.0

        out = RESULTS / f"tok_stats_{name}.json"
        write_json(out, stats)
        all_stats[name] = stats
        print(f"tok-stats {name}: tokens/word={stats['tokens_per_word']:.3f}")

    write_json(RESULTS / "tok_stats_all.json", all_stats)
    return all_stats


def save_tokenizer_train_times(times: dict[str, float]) -> None:
    """Persist wall times; never overwrite known values with 0 (skipped rebuild)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / "tokenizer_train_times.json"
    merged = load_tokenizer_train_times()
    for key, seconds in times.items():
        if seconds > 0:
            merged[key] = seconds
    write_json(path, merged)


def load_tokenizer_train_times() -> dict[str, float]:
    path = LOGS / "tokenizer_train_times.json"
    if path.exists():
        return read_json(path)
    return {}


def tokenizer_train_times_for_report() -> dict[str, float]:
    """Merge log file with train_wall_seconds stored in tokenizer artifacts."""
    from lab01_tokenization.config import DATA_TOKENIZERS, TOKENIZER_NAMES

    times = {k: float(v) for k, v in load_tokenizer_train_times().items() if float(v) > 0}
    meta_paths = {
        "gpt2": DATA_TOKENIZERS / "gpt2_meta.json",
        "whitespace": DATA_TOKENIZERS / "whitespace_vocab.json",
        "spm": DATA_TOKENIZERS / "spm_meta.json",
    }
    for name in TOKENIZER_NAMES:
        if times.get(name, 0) > 0:
            continue
        meta = read_json(meta_paths[name]) if meta_paths[name].exists() else {}
        sec = meta.get("train_wall_seconds")
        if sec is not None and float(sec) > 0:
            times[name] = float(sec)
    return times
