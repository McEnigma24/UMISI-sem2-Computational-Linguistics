"""Aggregate metrics into tables and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from lab01_tokenization.config import LOGS, RESULTS, ROOT, TOKENIZER_NAMES, ModelConfig, TrainConfig
from lab01_tokenization.stats import tokenizer_train_times_for_report
from lab01_tokenization.utils import read_json


def _load_optional(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def build_report_tables() -> str:
    lines = [
        "# Lab01 Tokenization Benchmark — generated summary",
        "",
        "## Model and training defaults",
        "",
        f"- Architecture: GPT decoder-only (pre-LN), defaults {ModelConfig().__dict__}",
        f"- Optimizer: AdamW, {TrainConfig().__dict__}",
        "",
        "## Perplexity (test split, word/char level)",
        "",
        "| Tokenizer | Word PPL | Char PPL | BPC | Word PPL (UNK-corrected) |",
        "|-----------|----------|----------|-----|---------------------------|",
    ]
    for name in TOKENIZER_NAMES:
        ev = _load_optional(RESULTS / f"eval_{name}.json")
        if not ev:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {ev.get('word_perplexity', 0):.2f} | {ev.get('char_perplexity', 0):.2f} | "
            f"{ev.get('bits_per_char', 0):.4f} | {ev.get('word_perplexity_unk_corrected', 0):.2f} |"
        )

    lines.extend(["", "## Tokenization statistics (test)", "", "| Tokenizer | Tokens/word | Direct words % | OOV % (whitespace) | Encode tok/s |", "|-----------|-------------|----------------|--------------------|--------------|"])
    for name in TOKENIZER_NAMES:
        st = _load_optional(RESULTS / f"tok_stats_{name}.json")
        if not st:
            lines.append(f"| {name} | — | — | — | — |")
            continue
        oov = st.get("oov_occurrence_pct", "n/a")
        if oov != "n/a":
            oov = f"{oov:.2f}"
        tpw = st.get("throughput", {})
        lines.append(
            f"| {name} | {st.get('tokens_per_word', 0):.3f} | {st.get('direct_word_occurrence_pct', 0):.1f} | "
            f"{oov} | {tpw.get('tokens_per_second', 0):.0f} |"
        )

    lines.extend(["", "## Training efficiency", "", "| Tokenizer | Wall time (s) | Train tok/s | Peak GPU MB |", "|-----------|---------------|-------------|-------------|"])
    for name in TOKENIZER_NAMES:
        m = _load_optional(RESULTS / f"metrics_{name}.json")
        if not m:
            lines.append(f"| {name} | — | — | — |")
            continue
        lines.append(
            f"| {name} | {m.get('train_wall_seconds', 0):.1f} | {m.get('tokens_per_second', 0):.0f} | "
            f"{m.get('peak_gpu_mem_mb', 0):.0f} |"
        )

    tok_times = tokenizer_train_times_for_report()
    if tok_times:
        lines.extend(["", "## Tokenizer training time (seconds)", ""])
        for k in TOKENIZER_NAMES:
            v = tok_times.get(k)
            if v is not None and v > 0:
                lines.append(f"- {k}: {v:.2f}s")
            else:
                lines.append(f"- {k}: — (cached; run `uv run lab01 train-tokenizers --force` to measure)")
    else:
        lines.extend(
            [
                "",
                "## Tokenizer training time (seconds)",
                "",
                "Not recorded yet. Run `uv run lab01 train-tokenizers --force` on CPU (SentencePiece ~2 min).",
            ]
        )

    hw = _load_optional(RESULTS / "hardware.json")
    if hw:
        lines.extend(["", "## Hardware snapshot", "", f"- Host: {hw.get('hostname')}", f"- Torch: {hw.get('torch')}", f"- CUDA device: {hw.get('cuda_device', 'cpu')}"])

    qual = RESULTS / "qualitative.md"
    if qual.exists():
        lines.extend(["", "## Qualitative", "", f"See [{qual.relative_to(ROOT)}]({qual.relative_to(ROOT)})."])

    return "\n".join(lines) + "\n"


def plot_loss_curves() -> None:
    fig, axes = plt.subplots(1, len(TOKENIZER_NAMES), figsize=(12, 3), squeeze=False)
    for ax, name in zip(axes[0], TOKENIZER_NAMES):
        log_path = LOGS / f"train_{name}.jsonl"
        if not log_path.exists():
            ax.set_title(f"{name} (no log)")
            continue
        steps, losses = [], []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            steps.append(rec["step"])
            losses.append(rec["loss"])
        ax.plot(steps, losses)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
    fig.tight_layout()
    out = RESULTS / "train_loss.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_tokens_per_word() -> None:
    names = []
    vals = []
    for name in TOKENIZER_NAMES:
        st = _load_optional(RESULTS / f"tok_stats_{name}.json")
        if st:
            names.append(name)
            vals.append(st.get("tokens_per_word", 0))
    if not names:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, vals)
    ax.set_ylabel("tokens / word")
    ax.set_title("Test split tokenization granularity")
    fig.tight_layout()
    out = RESULTS / "tokens_per_word.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def run_report() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = build_report_tables()
    (RESULTS / "summary_tables.md").write_text(summary, encoding="utf-8")
    plot_loss_curves()
    plot_tokens_per_word()
    report_path = ROOT / "REPORT.md"
    if report_path.exists():
        body = report_path.read_text(encoding="utf-8")
        marker = "<!-- AUTO-GENERATED -->"
        if marker in body:
            head, _ = body.split(marker, 1)
            report_path.write_text(head + marker + "\n\n" + summary, encoding="utf-8")
        else:
            report_path.write_text(body + "\n\n" + marker + "\n\n" + summary, encoding="utf-8")
    else:
        report_path.write_text(summary, encoding="utf-8")
    print(f"updated {report_path}")
