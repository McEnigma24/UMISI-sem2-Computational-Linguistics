"""Wykresy pojedynczego runu treningu (saved_models/*/metrics.csv)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _utils_.log_print import log_print
from metrics.run_metrics import (
    META_FILENAME,
    METRICS_EVAL_FILENAME,
    METRICS_FILENAME,
    METRICS_STEPS_FILENAME,
    TRAINING_REPORT_FILENAME,
    load_run_metrics,
)


def _load_epoch_frame(save_dir: Path) -> pd.DataFrame:
    rows = load_run_metrics(save_dir)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    for col in (
        "train_loss",
        "train_perplexity",
        "eval_loss",
        "eval_perplexity",
        "train_seconds",
        "eval_seconds",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "eval_loss" in df.columns:
        df["eval_loss"] = pd.to_numeric(df["eval_loss"], errors="coerce")
    if "eval_perplexity" in df.columns:
        df["eval_perplexity"] = pd.to_numeric(df["eval_perplexity"], errors="coerce")
    return df.sort_values("epoch")


def _load_run_meta(save_dir: Path) -> dict:
    meta_path = save_dir / META_FILENAME
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_train_total_seconds(save_dir: Path) -> float | None:
    raw = _load_run_meta(save_dir).get("train_total_seconds")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _timing_rows_for_wall_compare(df: pd.DataFrame, start_epoch: int) -> pd.DataFrame:
    """Wall clock dotyczy bieżącej sesji — sumuj tylko epoki po wznowieniu (epoch > start_epoch)."""
    if start_epoch <= 0:
        return df
    return df[df["epoch"] > start_epoch]


def plot_training_run(save_dir: Path, *, dpi: int = 120) -> list[Path]:
    """Loss, perplexity (train/eval), czasy epoki + podsumowanie całego treningu."""
    save_dir = Path(save_dir)
    if not (save_dir / METRICS_FILENAME).is_file():
        log_print(f"Brak {METRICS_FILENAME} w {save_dir} — pomijam wykresy treningu")
        return []

    df = _load_epoch_frame(save_dir)
    if df.empty:
        return []

    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sns.set_theme(style="whitegrid")
    arch = ""
    meta = _load_run_meta(save_dir)
    if meta:
        arch = str(meta.get("architecture") or "")
    if not arch and save_dir.parent.name not in ("saved_models", "model_evaluations"):
        arch = save_dir.parent.name
    if not arch:
        arch = save_dir.name.split("-", 1)[0]

    eval_df = df[df["eval_loss"].notna() & (df["eval_loss"] != "")]

    # Loss #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="train_loss")
    if not eval_df.empty:
        ax.plot(eval_df["epoch"], eval_df["eval_loss"], marker="s", label="eval_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Loss — {arch}")
    ax.legend()
    fig.tight_layout()
    out = plots_dir / "loss.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    # Perplexity #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["train_perplexity"], marker="o", label="train_perplexity")
    if not eval_df.empty:
        ax.plot(eval_df["epoch"], eval_df["eval_perplexity"], marker="s", label="eval_perplexity")
    ax.set_xlabel("epoch")
    ax.set_ylabel("perplexity")
    ax.set_title(f"Perplexity — {arch}")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    out = plots_dir / "perplexity.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    # Czas epoki #
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    epochs = df["epoch"].tolist()
    x = list(range(len(epochs)))
    ax.bar([i - width / 2 for i in x], df["train_seconds"], width=width, label="train_seconds")
    ax.bar([i + width / 2 for i in x], df["eval_seconds"], width=width, label="eval_seconds")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(e):g}" for e in epochs])
    ax.set_xlabel("epoch")
    ax.set_ylabel("sekundy")
    ax.set_title(f"Czas epoki (train + eval) — {arch}")
    ax.legend()
    fig.tight_layout()
    out = plots_dir / "epoch_timing.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    # Cały trening #
    start_epoch = int(meta.get("start_epoch") or 0)
    timing_df = _timing_rows_for_wall_compare(df, start_epoch)
    total_wall = _load_train_total_seconds(save_dir)
    sum_train = float(timing_df["train_seconds"].sum()) if "train_seconds" in timing_df else float("nan")
    sum_eval = float(timing_df["eval_seconds"].sum()) if "eval_seconds" in timing_df else float("nan")
    sum_logged = sum_train + sum_eval

    fig, ax = plt.subplots(figsize=(7, 4))
    labels: list[str] = []
    values: list[float] = []
    if total_wall is not None:
        labels.append("cały run (wall)")
        values.append(total_wall)
    if sum_logged == sum_logged and len(timing_df) > 0:
        if start_epoch > 0:
            labels.append(f"suma train+eval (ep. {start_epoch + 1}–{int(df['epoch'].max())})")
        else:
            labels.append("suma train+eval (log)")
        values.append(sum_logged)
    if labels:
        ax.barh(labels, values, color=["#4c72b0", "#55a868"][: len(labels)])
        ax.set_xlabel("sekundy")
        ax.set_title(f"Czas treningu — {arch}")
        for i, v in enumerate(values):
            ax.text(v, i, f"  {v:.1f}s", va="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "brak danych czasu", ha="center", va="center")
    fig.tight_layout()
    out = plots_dir / "training_total_timing.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    saved.append(out)

    steps_path = save_dir / METRICS_STEPS_FILENAME
    if steps_path.is_file():
        sdf = pd.read_csv(steps_path)
        if not sdf.empty and "train_loss" in sdf.columns:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(sdf["step"], sdf["train_loss"], alpha=0.7, label="train_loss (step)")
            eval_path = save_dir / METRICS_EVAL_FILENAME
            if eval_path.is_file():
                edf = pd.read_csv(eval_path)
                if not edf.empty:
                    ax.scatter(edf["step"], edf["eval_loss"], s=12, c="orange", label="eval (mid-epoch)")
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_title(f"Loss per step — {arch}")
            ax.legend()
            fig.tight_layout()
            out = plots_dir / "loss_steps.png"
            fig.savefig(out, dpi=dpi)
            plt.close(fig)
            saved.append(out)
            if "lr" in sdf.columns:
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(sdf["step"], sdf["lr"], color="green")
                ax.set_xlabel("step")
                ax.set_ylabel("lr")
                ax.set_title(f"Learning rate — {arch}")
                fig.tight_layout()
                out = plots_dir / "lr.png"
                fig.savefig(out, dpi=dpi)
                plt.close(fig)
                saved.append(out)

    test_m = meta.get("test_metrics") or {}
    if test_m and not eval_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        last_eval = float(eval_df["eval_perplexity"].iloc[-1])
        test_ppl = float(test_m.get("test_perplexity", float("nan")))
        ax.bar(
            ["val (last epoch)", "test"],
            [last_eval, test_ppl],
            color=["#4c72b0", "#dd8452"],
        )
        ax.set_ylabel("perplexity")
        ax.set_title(f"Val vs test — {arch}")
        fig.tight_layout()
        out = plots_dir / "summary.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        saved.append(out)

    for p in saved:
        log_print(f"Wykres treningu → {p}")
    return saved


def write_training_report(save_dir: Path) -> Path:
    save_dir = Path(save_dir)
    meta = _load_run_meta(save_dir)
    rows = load_run_metrics(save_dir)
    lines = [
        f"run_id: {meta.get('run_id', save_dir.name)}",
        f"architecture: {meta.get('architecture')}",
        f"train_total_seconds: {meta.get('train_total_seconds')}",
        f"epochs: {meta.get('epochs')}",
    ]
    fit = meta.get("fit") or {}
    if fit:
        lines.extend(
            [
                "--- training fit ---",
                f"strategy: {fit.get('strategy')}",
                f"per_gpu_batch: {fit.get('per_gpu_batch')}",
                f"global_batch: {fit.get('global_batch')}",
                f"learning_rate: {fit.get('learning_rate')}",
                f"batch_source: {fit.get('batch_source')}",
            ]
        )
    if rows:
        best = min(
            (r for r in rows if r.get("eval_loss") not in ("", None)),
            key=lambda r: float(r["eval_loss"]),
            default=None,
        )
        if best:
            lines.append(
                f"best_val: epoch={best.get('epoch')} loss={best.get('eval_loss')} "
                f"ppl={best.get('eval_perplexity')}"
            )
    test_m = meta.get("test_metrics") or {}
    if test_m:
        lines.extend(
            [
                "--- test ---",
                f"test_loss: {test_m.get('test_loss')}",
                f"test_perplexity: {test_m.get('test_perplexity')}",
            ]
        )
    out = save_dir / TRAINING_REPORT_FILENAME
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_print(f"Raport treningu → {out}")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Wykresy treningu z saved_models/<run>/metrics.csv")
    parser.add_argument("save_dir", type=Path, help="Katalog runu treningu")
    args = parser.parse_args()
    plot_training_run(args.save_dir)


if __name__ == "__main__":
    main()
