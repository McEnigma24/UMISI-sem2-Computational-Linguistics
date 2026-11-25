"""Perplexity LM na tekstach klasyfikacji (NLL następnego tokenu, bity/znak)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finetuning.cls_data import TweetSentimentDataset


def _minillama_lm_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """MiniLlamSequenceClassifier lub PEFT wrapper — forward LM na backbone."""
    core = model
    if hasattr(core, "base_model"):
        core = core.base_model
    if hasattr(core, "model"):
        core = core.model
    backbone = core.backbone if hasattr(core, "backbone") else core
    return backbone.lm_head(backbone.forward_hidden(input_ids))


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
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
        if attn is not None:
            # Cel na pozycji i to token i+1 — liczymy tylko, gdy i+1 jest prawdziwym tokenem (nie padding).
            target_valid = torch.zeros_like(attn, dtype=torch.bool)
            target_valid[:, :-1] = attn[:, 1:].bool()
            labels[~target_valid] = -100
        n_tok = int((labels != -100).sum().item())
        if n_tok < 1:
            continue
        nll = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
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
    ppl = math.exp(min(mean_nll, 700.0))
    bits_per_char = (total_nll / math.log(2)) / max(1, total_chars)
    return {
        "lm_loss_mean": mean_nll,
        "lm_perplexity_token": ppl,
        "lm_bits_per_char": bits_per_char,
        "lm_n_tokens": total_tokens,
        "lm_n_chars": total_chars,
    }