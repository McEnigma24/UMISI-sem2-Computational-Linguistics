"""Przeliczenie perplexity LM (przed/po fine-tune) dla ukończonych runów z FINETUNE_QUEUE.

Wczytuje wagi *_best_eval.pt, nie trenuje. Nadpisuje pola lm_before_* / lm_after_*
w test_metrics.json i run_meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from _utils_.log_print import log_print
from finetune import _release_gpu_memory, _resolve_eval_weights_path, completed_run_dir
from finetune_presets import FINETUNE_QUEUE
from finetuning.cls_data import collate_from_spec, prepare_cls_data
from finetuning.cls_models import CLS_ARCHITECTURE, apply_finetune_mode, build_classifier, resolve_source
from finetuning.hf_cache import offline_first
from finetuning.lm_eval import language_model_perplexity_on_dataset
from metrics.run_metrics import META_FILENAME, TEST_METRICS_FILENAME
from text_and_tokenizer import load_tokenizer

_LM_BATCH = 16


def _test_data(resolved: dict[str, Any], max_len: int, cache: dict[str, Any]):
    kind = "hf" if resolved["kind"] == "hf" else "bpe"
    if kind in cache:
        return cache[kind]
    hf_tok = bpe_tok = None
    if kind == "hf":
        mid = resolved["model_id"]
        hf_tok = offline_first(mid, AutoTokenizer.from_pretrained, mid)
        if hf_tok.pad_token_id is None and hf_tok.eos_token_id is not None:
            hf_tok.pad_token = hf_tok.eos_token
    else:
        bpe_tok = load_tokenizer()
    _, _, test_ds, spec, _ = prepare_cls_data(
        max_len=max_len,
        max_train_examples=0,
        max_val_examples=0,
        max_test_examples=0,
        tokenizer_kind=kind,
        hf_tokenizer=hf_tok,
        bpe_tokenizer=bpe_tok,
    )
    cache[kind] = (kind, test_ds, spec)
    return cache[kind]


def _build_model(preset: dict[str, Any], resolved: dict[str, Any], tok_kind: str, vocab_size: int):
    scratch_kwargs = None
    if resolved["kind"] == "scratch":
        scratch_kwargs = {
            k: preset.get(k)
            for k in ("embed_dim", "hidden_size", "num_layers", "num_heads", "ff_size", "dropout")
        }
    model, _ = build_classifier(
        resolved,
        tokenizer_kind=tok_kind,
        vocab_size=vocab_size,
        max_len=preset["max_len"],
        scratch_kwargs=scratch_kwargs,
        pooling=preset["pooling"],
        head_hidden=preset["head_hidden"],
        head_dropout=preset["head_dropout"],
    )
    model, _ = apply_finetune_mode(
        model,
        finetune_mode=preset["finetune_mode"],
        unfreeze_last_n=preset.get("unfreeze_last_n"),
        lora_r=preset.get("lora_r"),
        lora_alpha=preset.get("lora_alpha"),
        lora_dropout=preset.get("lora_dropout"),
        source_kind=resolved["kind"],
    )
    return model


def _write_lm_metrics(run_dir: Path, before: dict[str, float], after: dict[str, float]) -> None:
    test_path = run_dir / TEST_METRICS_FILENAME
    test_m = json.loads(test_path.read_text(encoding="utf-8"))
    for k, v in before.items():
        test_m[f"lm_before_{k}"] = v
    for k, v in after.items():
        test_m[f"lm_after_{k}"] = v
    test_path.write_text(json.dumps(test_m, indent=2) + "\n", encoding="utf-8")
    meta_path = run_dir / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["test_metrics"] = test_m
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_cache: dict[str, Any] = {}
    before_cache: dict[str, dict[str, float]] = {}
    for i, preset in enumerate(FINETUNE_QUEUE, start=1):
        tag = preset["run_tag"]
        run_dir = completed_run_dir(preset)
        if run_dir is None:
            log_print(f"[{i}/{len(FINETUNE_QUEUE)}] brak ukończonego runu: {tag}")
            continue
        resolved = resolve_source(preset["source"])
        tok_kind, test_ds, spec = _test_data(resolved, preset["max_len"], data_cache)
        collate = collate_from_spec(spec)
        torch.manual_seed(0)
        model = _build_model(preset, resolved, tok_kind, spec.vocab_size)

        src_key = str(preset["source"])
        if src_key not in before_cache:
            before_cache[src_key] = language_model_perplexity_on_dataset(
                model, test_ds, collate, device, batch_size=_LM_BATCH, source_kind=resolved["kind"]
            )
        before = before_cache[src_key]

        weights = _resolve_eval_weights_path(run_dir, CLS_ARCHITECTURE)
        payload = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
        model.cpu()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
        after = language_model_perplexity_on_dataset(
            model, test_ds, collate, device, batch_size=_LM_BATCH, source_kind=resolved["kind"]
        )
        _write_lm_metrics(run_dir, before, after)
        log_print(
            f"[{i}/{len(FINETUNE_QUEUE)}] {tag}: LM PPL przed={before['lm_perplexity_token']:.1f} "
            f"po={after['lm_perplexity_token']:.1f} (bpc {before['lm_bits_per_char']:.3f} → "
            f"{after['lm_bits_per_char']:.3f})"
        )
        model.cpu()
        del model
        _release_gpu_memory()


if __name__ == "__main__":
    main()
