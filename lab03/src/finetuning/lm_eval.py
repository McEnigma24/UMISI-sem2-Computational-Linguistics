"""Perplexity LM na tekstach klasyfikacji (NLL następnego tokenu, bity/znak)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finetuning.cls_data import TweetSentimentDataset, collate_from_spec


@torch.no_grad()
def _shifted_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[float, int, int]:
    """logits [B,T,V], labels [B,T] next-token targets, mask [B,T] na label positions."""
    b, t, v = logits.shape
    flat_logits = logits.reshape(-1, v)
    flat_labels = labels.reshape(-1)
    flat_mask = mask.reshape(-1).bool()
    if not flat_mask.any():
        return float("nan"), 0, 0
    sel_logits = flat_logits[flat_mask]
    sel_labels = flat_labels[flat_mask]
    loss = F.cross_entropy(sel_logits, sel_labels, reduction="sum")
    n_tok = int(sel_labels.numel())
    n_char = 0
    return float(loss.item()), n_tok, n_char


def _minillama_lm_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """MiniLlamSequenceClassifier lub PEFT wrapper — forward LM na backbone."""
    core = model
    if hasattr(core, "base_model"):
        core = core.base_model
    if hasattr(core, "model"):
        core = core.model
    backbone = core.backbone if hasattr(core, "backbone") else core
    b, t = input_ids.shape
    pos = torch.arange(0, t, device=input_ids.device)
    h = backbone.drop(backbone.tok_emb(input_ids))
    for block in backbone.blocks:
        h, _ = block(h, None, pos)
    h = backbone.norm_f(h)
    return backbone.lm_head(h)


def _resolve_gpt2_core(model: nn.Module) -> nn.Module:
    core = model.core if hasattr(model, "core") else model
    for _ in range(10):
        if hasattr(core, "transformer"):
            return core
        if hasattr(core, "model"):
            core = core.model
            continue
        if hasattr(core, "base_model"):
            core = core.base_model
            continue
        if hasattr(core, "get_base_model"):
            core = core.get_base_model()
            continue
        break
    raise ValueError("Nie rozpoznano GPT-2 do lm_eval")


def _gpt2_lm_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    core = _resolve_gpt2_core(model)
    if hasattr(core, "transformer"):
        tr = core.transformer
        outputs = tr(input_ids)
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs.last_hidden_state
        if hasattr(core, "lm_head") and core.lm_head is not None:
            return core.lm_head(h)
        return h @ tr.wte.weight.t()
    raise ValueError("Nie rozpoznano GPT-2 do lm_eval")


def _char_count_batch(batch: dict, dataset: TweetSentimentDataset, indices: list[int]) -> int:
    n = 0
    for i in indices:
        n += len(dataset.rows[i]["text"])
    return n


@torch.no_grad()
def language_model_perplexity_on_dataset(
    model: nn.Module,
    dataset: TweetSentimentDataset,
    collate,
    device: torch.device,
    *,
    batch_size: int = 16,
    max_batches: int | None = None,
    source_kind: str,
) -> dict[str, float]:
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    total_nll = 0.0
    total_tokens = 0
    total_chars = 0
    idx_base = 0
    for batch_i, batch in enumerate(loader):
        if max_batches is not None and batch_i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        b, t = input_ids.shape
        if source_kind in ("pt", "scratch"):
            logits = _minillama_lm_logits(model, input_ids)
        elif source_kind == "hf":
            logits = _gpt2_lm_logits(model, input_ids)
        else:
            raise ValueError(source_kind)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        mask = torch.ones_like(labels, dtype=torch.bool)
        mask[:, -1] = False
        if attn is not None:
            mask &= attn.bool()
            shift_attn = attn.clone()
            shift_attn[:, -1] = 0
            mask &= shift_attn.bool()
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid = (labels != -100) & mask
        n_tok = int(valid.sum().item())
        if n_tok < 1:
            continue
        total_nll += float(nll.item())
        total_tokens += n_tok
        batch_indices = list(range(idx_base, min(idx_base + b, len(dataset))))
        total_chars += _char_count_batch(batch, dataset, batch_indices)
        idx_base += b

    if total_tokens < 1:
        return {
            "lm_loss_mean": float("nan"),
            "lm_perplexity_token": float("nan"),
            "lm_bits_per_char": float("nan"),
            "lm_n_tokens": 0,
            "lm_n_chars": 0,
        }
    mean_nll = total_nll / total_tokens
    ppl = math.exp(min(mean_nll, 20.0))
    bits_per_char = (total_nll / math.log(2)) / max(1, total_chars)
    return {
        "lm_loss_mean": mean_nll,
        "lm_perplexity_token": ppl,
        "lm_bits_per_char": bits_per_char,
        "lm_n_tokens": total_tokens,
        "lm_n_chars": total_chars,
    }