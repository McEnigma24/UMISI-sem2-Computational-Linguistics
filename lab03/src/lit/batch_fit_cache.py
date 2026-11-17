"""Cache przedbiegu batcha: saved_models/batch_fits/<arch>/<hash>.json."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from _utils_.log_print import log_print
from _utils_.run_paths import models_root


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module),
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return tree


def code_hash_for_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8")
        tree = _strip_docstrings(ast.parse(src))
        h.update(ast.dump(tree, annotate_fields=True).encode())
        h.update(str(p).encode())
    return h.hexdigest()


def structure_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, mod in model.named_modules():
        h.update(name.encode())
        h.update(type(mod).__name__.encode())
        for pname, p in mod.named_parameters(recurse=False):
            h.update(pname.encode())
            h.update(str(tuple(p.shape)).encode())
            h.update(str(p.dtype).encode())
            h.update(str(p.requires_grad).encode())
        for bname, b in mod.named_buffers(recurse=False):
            h.update(bname.encode())
            h.update(str(tuple(b.shape)).encode())
    return h.hexdigest()


def gpu_descriptor() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu": "cpu", "gpu_mem": 0, "torch": torch.__version__, "cuda": None}
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    return {
        "gpu": props.name,
        "gpu_mem": int(props.total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


@dataclass
class Fingerprint:
    full_hash: str
    components: dict[str, Any]

    @property
    def short(self) -> str:
        return self.full_hash[:12]


def build_fingerprint(
    model: nn.Module,
    *,
    task: str,
    seq_len: int,
    precision: str,
    strategy: str,
    devices: int,
    code_paths: list[Path],
) -> Fingerprint:
    comp = {
        "task": task,
        "seq_len": seq_len,
        "precision": precision,
        "strategy_probe": strategy,
        "devices": devices,
        "params": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "code_hash": code_hash_for_paths(code_paths),
        "structure_hash": structure_hash(model),
        **gpu_descriptor(),
    }
    payload = json.dumps(comp, sort_keys=True).encode()
    full = _sha256(payload)
    return Fingerprint(full_hash=full, components=comp)


def cache_path(architecture: str, fp: Fingerprint) -> Path:
    root = models_root() / "batch_fits" / architecture
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{fp.short}.json"


def load_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, *, fingerprint: Fingerprint, results: dict, module_tree: str) -> None:
    payload = {
        "fingerprint": fingerprint.full_hash,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "components": fingerprint.components,
        "results": results,
        "module_tree": module_tree,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def get_probe_result(
    path: Path,
    mode: str,
    *,
    probe_max: int,
    refresh: bool,
) -> tuple[int | None, str]:
    """Zwraca (max_per_gpu_batch, source) — source cache|miss."""
    if refresh:
        return None, "miss"
    data = load_cache(path)
    if data is None:
        return None, "miss"
    block = (data.get("results") or {}).get(mode)
    if not block:
        return None, "miss"
    saved_max = int(block.get("max_per_gpu_batch", 0))
    saved_probe = int(block.get("probe_max", 0))
    if saved_probe < probe_max:
        return None, "miss"
    return min(saved_max, probe_max), "cache"


def store_probe_result(
    path: Path,
    *,
    fingerprint: Fingerprint,
    mode: str,
    max_per_gpu_batch: int,
    probe_max: int,
    peak_mem_bytes: float,
    seconds: float,
    module_tree: str,
) -> None:
    data = load_cache(path) or {
        "fingerprint": fingerprint.full_hash,
        "components": fingerprint.components,
        "results": {},
        "module_tree": module_tree,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    data["fingerprint"] = fingerprint.full_hash
    data["components"] = fingerprint.components
    data["module_tree"] = module_tree
    data.setdefault("results", {})[mode] = {
        "max_per_gpu_batch": max_per_gpu_batch,
        "probe_max": probe_max,
        "peak_mem_bytes": peak_mem_bytes,
        "seconds": seconds,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    log_print(f"batch fit: zapis cache {path.name}")