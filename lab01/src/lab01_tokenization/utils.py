"""Shared helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lab01_tokenization.config import RESULTS

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_words(text: str) -> int:
    return len(text.split())


def count_chars(text: str) -> int:
    return len(text)


def iter_words(text: str):
    for line in text.splitlines():
        if line.strip():
            yield from line.split()


def split_pieces(text: str) -> list[str]:
    return WORD_RE.findall(text)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def timer() -> float:
    return time.perf_counter()


def save_hardware_info() -> Path:
    out = RESULTS / "hardware.json"
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_arch_list"] = torch.cuda.get_arch_list()
    try:
        info["lscpu"] = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=30, check=False
        ).stdout[:8000]
    except OSError:
        pass
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except OSError:
        pass
    slurm = {k: os.environ.get(k) for k in os.environ if k.startswith("SLURM_")}
    if slurm:
        info["slurm"] = slurm
    write_json(out, info)
    return out


def build_char_unigram(train_text: str) -> dict[str, float]:
    counts = Counter(train_text)
    total = sum(counts.values())
    return {ch: c / total for ch, c in counts.items()}


def unk_char_nll(word: str, char_probs: dict[str, float], eps: float = 1e-12) -> float:
    nll = 0.0
    for ch in word:
        p = char_probs.get(ch, eps)
        nll -= np.log(p)
    return nll
