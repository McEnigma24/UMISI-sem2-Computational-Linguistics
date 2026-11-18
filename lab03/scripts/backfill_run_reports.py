#!/usr/bin/env python3
"""Odtwarza metrics.csv z eventów TensorBoard dla istniejących runów LM."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from metrics.plot_training import plot_training_run
from metrics.run_metrics import METRICS_COLUMNS, METRICS_FILENAME, META_FILENAME


def _parse_ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def _find_tb_version(runs_root: Path, started_at: str | None, finished_at: str | None) -> Path | None:
    t0 = _parse_ts(started_at)
    t1 = _parse_ts(finished_at) or (t0 + 86400 * 7 if t0 else None)
    if t0 is None:
        return None
    best: tuple[float, Path] | None = None
    for version in sorted(runs_root.glob("version_*")):
        events = sorted(version.glob("events.out.tfevents.*"))
        if not events:
            continue
        mtime = events[-1].stat().st_mtime
        if mtime < t0 - 120:
            continue
        if t1 is not None and mtime > t1 + 3600:
            continue
        score = abs(mtime - t0)
        if best is None or score < best[0]:
            best = (score, version)
    return best[1] if best else None


def _scalar_series(ea: EventAccumulator, tag: str) -> dict[int, float]:
    if tag not in ea.Tags().get("scalars", []):
        return {}
    return {int(e.step): float(e.value) for e in ea.Scalars(tag)}


def backfill_run(save_dir: Path, tb_version: Path) -> None:
    events = sorted(tb_version.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"Brak eventów w {tb_version}")
    ea = EventAccumulator(str(tb_version))
    ea.Reload()
    train_loss = _scalar_series(ea, "train/loss")
    train_ppl = _scalar_series(ea, "train/perplexity")
    eval_loss = _scalar_series(ea, "eval/loss")
    eval_ppl = _scalar_series(ea, "eval/perplexity")
    steps = sorted(set(train_loss) | set(eval_loss))
    if not steps:
        raise RuntimeError(f"Brak skalarów train/eval w {tb_version}")

    rows = []
    for i, step in enumerate(steps, start=1):
        rows.append(
            {
                "epoch": i,
                "train_loss": train_loss.get(step, ""),
                "train_perplexity": train_ppl.get(step, ""),
                "eval_loss": eval_loss.get(step, ""),
                "eval_perplexity": eval_ppl.get(step, ""),
                "train_seconds": "",
                "eval_seconds": "",
                "step": step,
                "tokens_seen": "",
            }
        )
    out = save_dir / METRICS_FILENAME
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    plot_training_run(save_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "saved_models" / "pretrain" / "transformer_miniLLaMA",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Pojedynczy katalog runu")
    args = parser.parse_args()

    run_dirs: list[Path]
    if args.run_dir:
        run_dirs = [args.run_dir]
    else:
        run_dirs = [p for p in sorted(args.models_root.iterdir()) if p.is_dir()]

    tb_candidates = sorted(args.runs_root.glob("*/*/version_*")) + sorted(args.runs_root.glob("*/version_*"))

    for save_dir in run_dirs:
        meta_path = save_dir / META_FILENAME
        if not meta_path.is_file():
            print(f"pomijam {save_dir.name}: brak run_meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tb_dir = meta.get("tensorboard_dir")
        version: Path | None = None
        if tb_dir:
            p = Path(tb_dir)
            if p.name.startswith("version_"):
                version = p
            elif (p / "version_0").is_dir():
                version = p / "version_0"
        if version is None:
            runs_parent = args.runs_root
            for cand in tb_candidates:
                if _find_tb_version(runs_parent, meta.get("started_at"), meta.get("finished_at")) == cand:
                    version = cand
                    break
        if version is None:
            version = _find_tb_version(args.runs_root, meta.get("started_at"), meta.get("finished_at"))
        if version is None:
            print(f"pomijam {save_dir.name}: brak dopasowanego TensorBoard")
            continue
        try:
            backfill_run(save_dir, version)
            print(f"OK {save_dir.name} ← {version}")
        except Exception as exc:
            print(f"BŁĄD {save_dir.name}: {exc}")


if __name__ == "__main__":
    main()