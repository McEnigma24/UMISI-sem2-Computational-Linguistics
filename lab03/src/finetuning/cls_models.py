"""Klasyfikator: scratch / checkpoint LM / Hugging Face."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from _utils_.log_print import log_print
from _utils_.run_paths import models_root, run_tag_from_dir_name
from finetuning.cls_data import NUM_LABELS
from models.transformer_miniLLaMA import transformer_miniLLaMA
from train import load_checkpoint_payload

CLS_ARCHITECTURE = "cls_tweet_eval_sentiment"

MINILLAMA_LORA_TARGETS = ("qkv", "proj", "w1", "w2", "w3")


class ClassifierHead(nn.Module):
    def __init__(self, hidden: int, num_labels: int, *, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    pooling: str,
) -> torch.Tensor:
    if pooling == "mean" and attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
    if attention_mask is not None:
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        idx = lengths.long().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)
    return hidden[:, -1, :]


class MiniLlamSequenceClassifier(nn.Module):
    """Backbone miniLLaMA + pooling + MLP."""

    def __init__(
        self,
        backbone: transformer_miniLLaMA,
        *,
        num_labels: int,
        pooling: str,
        head_hidden: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.cost_function = nn.CrossEntropyLoss()
        self.head = ClassifierHead(
            backbone.n_embd,
            num_labels,
            head_hidden=head_hidden,
            dropout=head_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.backbone.forward_hidden(input_ids, attention_mask=attention_mask)
        pooled = _pool_hidden(hidden, attention_mask, pooling=self.pooling)
        logits = self.head(pooled)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = self.cost_function(logits, labels)
        return out


class HfSequenceClassifier(nn.Module):
    """Owinięcie AutoModelForSequenceClassification dla spójnego API."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core
        self.cost_function = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.core(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        result: dict[str, torch.Tensor] = {"logits": out.logits}
        if labels is not None and out.loss is not None:
            result["loss"] = out.loss
        return result


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, sum(p.numel() * p.element_size() for p in model.parameters())


def _default_num_heads(hidden_size: int, num_heads: int | None) -> int:
    if num_heads is not None:
        return num_heads
    return max(1, hidden_size // 64)


def build_scratch_mini_llama(
    *,
    vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int | None,
    ff_size: int | None,
    dropout: float,
    seq_len: int,
    pooling: str,
    head_hidden: int,
    head_dropout: float,
) -> MiniLlamSequenceClassifier:
    n_heads = _default_num_heads(hidden_size, num_heads)
    if ff_size is None:
        ff_size = hidden_size * 4
    backbone = transformer_miniLLaMA(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=num_layers,
        num_heads=n_heads,
        ff_size=ff_size,
        seq_len=seq_len,
    )
    return MiniLlamSequenceClassifier(
        backbone,
        num_labels=NUM_LABELS,
        pooling=pooling,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    )


def _load_rnn_state_from_lm_checkpoint(payload: dict) -> dict[str, torch.Tensor]:
    state = payload.get("model_state_dict") or {}
    rnn_only: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if key.startswith("rnn."):
            rnn_only[key[len("rnn.") :]] = val
    if not rnn_only:
        for key, val in state.items():
            if not key.startswith("cost_function"):
                rnn_only[key] = val
    return rnn_only


def build_mini_llama_from_pt(
    checkpoint_path: Path,
    *,
    seq_len: int,
    pooling: str,
    head_hidden: int,
    head_dropout: float,
) -> tuple[MiniLlamSequenceClassifier, dict[str, Any]]:
    payload = load_checkpoint_payload(checkpoint_path)
    hp = dict(payload.get("hparams") or {})
    architecture = hp.get("architecture") or hp.get("model_name")
    if architecture != "transformer_miniLLaMA":
        raise ValueError(f"Checkpoint {checkpoint_path} nie jest transformer_miniLLaMA ({architecture!r})")
    vocab_size = int(hp.get("vocab_size", 0))
    if vocab_size < 1:
        raise ValueError("Brak vocab_size w hparams checkpointu")
    embed_dim = int(hp.get("embed_dim", hp.get("hidden_size", 256)))
    hidden_size = int(hp.get("hidden_size", 256))
    num_layers = int(hp.get("num_layers", 3))
    num_heads = hp.get("num_heads")
    ff_size = hp.get("ff_size")
    dropout = float(hp.get("dropout", 0.1))

    model = build_scratch_mini_llama(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=int(num_heads) if num_heads is not None else None,
        ff_size=int(ff_size) if ff_size is not None else None,
        dropout=dropout,
        seq_len=seq_len,
        pooling=pooling,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    )
    rnn_state = _load_rnn_state_from_lm_checkpoint(payload)
    missing, unexpected = model.backbone.load_state_dict(rnn_state, strict=False)
    if missing:
        log_print(f"Wczytanie PT: brakujące klucze backbone ({len(missing)}), m.in. {missing[:3]}")
    if unexpected:
        log_print(f"Wczytanie PT: nieoczekiwane klucze ({len(unexpected)})")
    meta = {
        "source_checkpoint": str(checkpoint_path),
        "lm_hparams": hp,
        "eval_loss": payload.get("eval_loss"),
        "eval_perplexity": payload.get("eval_perplexity"),
    }
    return model, meta


def find_lm_best_eval_by_run_tag(run_tag: str) -> Path:
    arch_root = models_root() / "transformer_miniLLaMA"
    if not arch_root.is_dir():
        raise FileNotFoundError(f"Brak {arch_root}")
    candidates: list[tuple[str, Path]] = []
    for child in arch_root.iterdir():
        if not child.is_dir():
            continue
        tag = run_tag_from_dir_name(child.name)
        if tag != run_tag:
            continue
        p = child / "transformer_miniLLaMA_best_eval.pt"
        if p.is_file():
            candidates.append((child.name, p))
    if not candidates:
        raise FileNotFoundError(f"Brak best_eval dla run_tag={run_tag!r} w {arch_root}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_source(source: str) -> dict[str, Any]:
    source = source.strip()
    if source == "scratch":
        return {"kind": "scratch"}
    if source.startswith("pt:"):
        path = Path(source[3:].strip())
        if not path.is_file():
            path = Path(source[3:].strip()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Brak pliku checkpointu: {source}")
        return {"kind": "pt", "path": path}
    m = re.match(r"^pt-latest:(.+)$", source)
    if m:
        run_tag = m.group(1).strip()
        path = find_lm_best_eval_by_run_tag(run_tag)
        return {"kind": "pt", "path": path, "run_tag": run_tag}
    if source.startswith("hf:"):
        model_id = source[3:].strip()
        if not model_id:
            raise ValueError("hf: wymaga model_id")
        return {"kind": "hf", "model_id": model_id}
    raise ValueError(f"Nieznane źródło: {source!r}")


def build_hf_classifier(model_id: str) -> tuple[HfSequenceClassifier, Any]:
    log_print(f"HF model: {model_id}")
    config = AutoConfig.from_pretrained(model_id, num_labels=NUM_LABELS)
    core = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)
    if getattr(core.config, "pad_token_id", None) is None:
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is not None:
            core.config.pad_token_id = tok.pad_token_id
    return HfSequenceClassifier(core), core


def apply_finetune_mode(
    model: nn.Module,
    *,
    finetune_mode: str,
    unfreeze_last_n: int | None,
    lora_r: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
    source_kind: str,
) -> tuple[nn.Module, dict[str, Any]]:
    info: dict[str, Any] = {"finetune_mode": finetune_mode}
    if finetune_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return model, info

    if finetune_mode == "unfreeze_last":
        if unfreeze_last_n is None or unfreeze_last_n < 1:
            raise ValueError("unfreeze_last wymaga unfreeze_last_n >= 1")
        for p in model.parameters():
            p.requires_grad = False
        if isinstance(model, MiniLlamSequenceClassifier):
            n_blocks = len(model.backbone.blocks)
            start = max(0, n_blocks - unfreeze_last_n)
            for p in model.backbone.tok_emb.parameters():
                p.requires_grad = False
            for i, block in enumerate(model.backbone.blocks):
                req = i >= start
                for p in block.parameters():
                    p.requires_grad = req
            for p in model.backbone.norm_f.parameters():
                p.requires_grad = True
        elif isinstance(model, HfSequenceClassifier):
            core = model.core
            layers = None
            if hasattr(core, "transformer") and hasattr(core.transformer, "h"):
                layers = core.transformer.h
            elif hasattr(core, "gpt_neox") and hasattr(core.gpt_neox, "layers"):
                layers = core.gpt_neox.layers
            elif hasattr(core, "roberta") and hasattr(core.roberta, "encoder"):
                layers = core.roberta.encoder.layer
            if layers is None:
                raise ValueError("Nie rozpoznano warstw HF do unfreeze_last")
            n_layers = len(layers)
            start = max(0, n_layers - unfreeze_last_n)
            for i, layer in enumerate(layers):
                req = i >= start
                for p in layer.parameters():
                    p.requires_grad = req
            if hasattr(core, "score"):
                for p in core.score.parameters():
                    p.requires_grad = True
        if isinstance(model, MiniLlamSequenceClassifier):
            for p in model.head.parameters():
                p.requires_grad = True
        info["unfreeze_last_n"] = unfreeze_last_n
        return model, info

    if finetune_mode == "lora":
        if source_kind == "scratch":
            raise ValueError("LoRA nie ma sensu dla scratch — użyj full")
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise ImportError("Zainstaluj peft (uv sync)") from e
        if lora_r is None or lora_alpha is None or lora_dropout is None:
            raise ValueError("lora wymaga lora_r, lora_alpha, lora_dropout")
        if isinstance(model, MiniLlamSequenceClassifier):
            target = list(MINILLAMA_LORA_TARGETS)
            modules_to_save = ["head"]
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target,
                modules_to_save=modules_to_save,
                task_type=TaskType.SEQ_CLS,
            )
            wrapped = get_peft_model(model, peft_config)
            info["peft"] = True
            return wrapped, info
        if isinstance(model, HfSequenceClassifier):
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type=TaskType.SEQ_CLS,
            )
            wrapped = get_peft_model(model.core, peft_config)
            model.core = wrapped
            info["peft"] = True
            return model, info

    raise ValueError(f"Nieznany finetune_mode: {finetune_mode!r}")


def _resolved_for_meta(resolved: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": resolved["kind"]}
    if resolved["kind"] == "pt":
        out["path"] = str(resolved["path"])
        if "run_tag" in resolved:
            out["run_tag"] = resolved["run_tag"]
    elif resolved["kind"] == "hf":
        out["model_id"] = resolved["model_id"]
    return out


def build_classifier(
    resolved: dict[str, Any],
    *,
    tokenizer_kind: str,
    vocab_size: int,
    max_len: int,
    scratch_kwargs: dict[str, Any] | None,
    pooling: str,
    head_hidden: int,
    head_dropout: float,
) -> tuple[nn.Module, dict[str, Any]]:
    meta: dict[str, Any] = {"source": _resolved_for_meta(resolved)}
    kind = resolved["kind"]
    if kind == "scratch":
        if scratch_kwargs is None:
            raise ValueError("scratch_kwargs wymagane dla source=scratch")
        model = build_scratch_mini_llama(
            vocab_size=vocab_size,
            embed_dim=int(scratch_kwargs["embed_dim"]),
            hidden_size=int(scratch_kwargs["hidden_size"]),
            num_layers=int(scratch_kwargs["num_layers"]),
            num_heads=scratch_kwargs.get("num_heads"),
            ff_size=scratch_kwargs.get("ff_size"),
            dropout=float(scratch_kwargs["dropout"]),
            seq_len=max_len,
            pooling=pooling,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )
        meta["architecture"] = "transformer_miniLLaMA+head"
        return model, meta
    if kind == "pt":
        model, pt_meta = build_mini_llama_from_pt(
            Path(resolved["path"]),
            seq_len=max_len,
            pooling=pooling,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )
        meta.update(pt_meta)
        meta["architecture"] = "transformer_miniLLaMA+head"
        return model, meta
    if kind == "hf":
        model, _core = build_hf_classifier(resolved["model_id"])
        meta["architecture"] = resolved["model_id"]
        meta["hf_model_id"] = resolved["model_id"]
        return model, meta
    raise ValueError(kind)