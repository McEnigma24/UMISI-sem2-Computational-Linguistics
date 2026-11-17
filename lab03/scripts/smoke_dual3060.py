"""Krótki smoke DUAL3060 (Lightning DDP). Uruchamiaj jako plik, nie przez stdin — spawn wymaga __file__."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import train
from training_presets import DUAL3060_ALL


def main() -> None:
    budget = float(os.environ.get("LANGUAGE_RNN_TIME_BUDGET_MIN", "1.5"))
    started = datetime.fromtimestamp(
        int(os.environ.setdefault("LANGUAGE_RNN_SESSION_TS", str(int(time.time()))))
    )
    for job_index, (architecture, preset) in enumerate(DUAL3060_ALL):
        p = dict(preset)
        p["max_corpus_chars"] = 500_000
        p["time_budget_min"] = budget
        p["batch_size"] = max(2, p.get("batch_size", 8) // 4)
        print(f"\n=== smoke {architecture} ===", flush=True)
        train.main(
            tensorboard_dir=None,
            architecture=architecture,
            run_started_at=started + timedelta(seconds=job_index),
            **p,
        )


if __name__ == "__main__":
    main()
