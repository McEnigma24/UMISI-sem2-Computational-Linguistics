"""Odczyt RAM/swap/GPU i log do nohup + opcjonalnie mem_log.csv w katalogu runu."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from _utils_.log_print import log_print

_MEM_CSV = ("timestamp", "tag", "mem_avail_mib", "mem_used_mib", "swap_free_mib", "rss_mib", "gpu_mib")


def _read_kv_kb(path: Path, key: str) -> int | None:
    if not path.is_file():
        return None
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return None


def _gpu_mem_mib() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        parts: list[str] = []
        for i in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(i)
            used_mib = (total_b - free_b) // (1024 * 1024)
            total_mib = total_b // (1024 * 1024)
            parts.append(f"cuda:{i} {used_mib}/{total_mib} MiB")
        return ", ".join(parts) if parts else None
    except Exception:
        return None


def _sample() -> dict[str, int | str | None]:
    mem_total = _read_kv_kb(Path("/proc/meminfo"), "MemTotal")
    mem_avail = _read_kv_kb(Path("/proc/meminfo"), "MemAvailable")
    swap_free = _read_kv_kb(Path("/proc/meminfo"), "SwapFree")
    rss_kb = _read_kv_kb(Path("/proc/self/status"), "VmRSS")
    mem_used: int | None = None
    if mem_total is not None and mem_avail is not None:
        mem_used = max(0, mem_total - mem_avail)
    return {
        "mem_avail_mib": mem_avail // 1024 if mem_avail is not None else None,
        "mem_used_mib": mem_used // 1024 if mem_used is not None else None,
        "swap_free_mib": swap_free // 1024 if swap_free is not None else None,
        "rss_mib": rss_kb // 1024 if rss_kb is not None else None,
        "gpu": _gpu_mem_mib(),
    }


def _append_mem_csv(run_dir: Path, tag: str, sample: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "mem_log.csv"
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    row = {
        "timestamp": ts,
        "tag": tag,
        "mem_avail_mib": sample.get("mem_avail_mib") or "",
        "mem_used_mib": sample.get("mem_used_mib") or "",
        "swap_free_mib": sample.get("swap_free_mib") or "",
        "rss_mib": sample.get("rss_mib") or "",
        "gpu_mib": sample.get("gpu") or "",
    }
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MEM_CSV)
        if write_header:
            w.writeheader()
        w.writerow(row)


def log_memory(tag: str, *, run_dir: str | Path | None = None) -> None:
    """Jedna linia w logu treningu (+ opcjonalnie mem_log.csv w katalogu runu)."""
    s = _sample()
    chunks = [f"Pamięć [{tag}]:"]
    if s["mem_avail_mib"] is not None:
        chunks.append(f"MemAvail {s['mem_avail_mib']} MiB")
    if s["mem_used_mib"] is not None:
        chunks.append(f"MemUsed {s['mem_used_mib']} MiB")
    if s["swap_free_mib"] is not None:
        chunks.append(f"SwapFree {s['swap_free_mib']} MiB")
    if s["rss_mib"] is not None:
        chunks.append(f"RSS {s['rss_mib']} MiB")
    if s["gpu"]:
        chunks.append(str(s["gpu"]))
    log_print("  ".join(chunks))
    if run_dir is not None:
        _append_mem_csv(Path(run_dir), tag, s)