"""Fine-tuning klasyfikacji Tweet Eval (sentiment)."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from _utils_.log_print import log_print, tensorboard_run_dir
from _utils_.mem_info import log_memory
from _utils_.run_paths import PRETRAIN_PHASE, arch_root, models_root, resolve_save_dir, run_tag_from_dir_name
from _utils_.timing import sync_device
from finetuning.cls_data import LABEL_NAMES, NUM_LABELS, collate_from_spec, prepare_cls_data
from finetuning.hf_cache import offline_first
from finetuning.cls_models import (
    CLS_ARCHITECTURE,
    apply_finetune_mode,
    build_classifier,
    count_parameters,
    resolve_source,
)
from finetune_presets import FINETUNE_ALL, FINETUNE_QUEUE
from finetuning.lm_eval import language_model_perplexity_on_dataset
from metrics.run_metrics import TEST_METRICS_FILENAME
from lit.cls_callbacks import ClsRunLogCallback
from lit.cls_module import LitClassifier
from lit.callbacks import TimeBudgetCallback
from lit.lm_module import default_lightning_precision
from lit.optim import scaled_learning_rate
from metrics.run_metrics import METRICS_FILENAME, META_FILENAME, RunMetricsLogger
from text_and_tokenizer import load_tokenizer
from train import load_checkpoint_payload


def _release_gpu_memory() -> None:
    """WSL2: pamięć zarezerwowana na GPU liczy się do commitu hosta, także gdy proces jej nie używa."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_eval_weights_path(save_dir: Path, architecture: str) -> Path:
    best_path = save_dir / f"{architecture}_best_eval.pt"
    if best_path.is_file():
        return best_path
    legacy_final = save_dir / f"{architecture}_final.pt"
    if legacy_final.is_file():
        return legacy_final
    ckpt_dir = save_dir / "checkpoints"
    if ckpt_dir.is_dir():
        epoch_files = sorted(ckpt_dir.glob("epoch_*.pt"))
        if epoch_files:
            return epoch_files[-1]
    raise FileNotFoundError(
        f"Brak wag do testu w {save_dir} ({architecture}_best_eval.pt lub checkpoints/epoch_*.pt)"
    )


def _tokenizer_kind(resolved: dict[str, Any]) -> str:
    return "hf" if resolved["kind"] == "hf" else "bpe"


def _scratch_kwargs_from_preset(preset: dict[str, Any]) -> dict[str, Any] | None:
    if not str(preset.get("source", "")).startswith("scratch"):
        return None
    return {
        "embed_dim": preset["embed_dim"],
        "hidden_size": preset["hidden_size"],
        "num_layers": preset["num_layers"],
        "num_heads": preset.get("num_heads"),
        "ff_size": preset.get("ff_size"),
        "dropout": preset["dropout"],
    }


def _validate_preset(preset: dict[str, Any]) -> None:
    mode = preset["finetune_mode"]
    source = preset["source"]
    if mode == "lora":
        for key in ("lora_r", "lora_alpha", "lora_dropout"):
            if preset.get(key) is None:
                raise ValueError(f"finetune_mode=lora wymaga {key}")
    if mode == "unfreeze_last" and preset.get("unfreeze_last_n") is None:
        raise ValueError("unfreeze_last wymaga unfreeze_last_n")
    if mode == "head_only" and source == "scratch":
        raise ValueError("head_only nie ma sensu dla scratch — użyj full")
    if source == "scratch" and mode != "full":
        raise ValueError("scratch wspiera wyłącznie finetune_mode=full")
    if source == "scratch":
        for key in ("embed_dim", "hidden_size", "num_layers", "dropout"):
            if key not in preset:
                raise ValueError(f"scratch wymaga {key}")


def _best_val_f1_from_metrics(save_dir: Path) -> tuple[int, float]:
    metrics_path = save_dir / METRICS_FILENAME
    if not metrics_path.is_file():
        return 0, 0.0
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if not rows:
        return 0, 0.0
    best = max(rows, key=lambda r: float(r["eval_f1_macro"]))
    return int(float(best["epoch"])), float(best["eval_f1_macro"])


def _source_from_hparams(hp_source: Any) -> str | None:
    if isinstance(hp_source, str):
        return hp_source.strip()
    if isinstance(hp_source, dict):
        kind = hp_source.get("kind")
        if kind == "scratch":
            return "scratch"
        if kind == "hf":
            mid = hp_source.get("model_id")
            return f"hf:{mid}" if mid else None
        if kind == "pt":
            path = hp_source.get("path")
            return f"pt:{path}" if path else None
    return None


def _source_preset_match(hp_source: Any, preset_source: str) -> bool:
    meta_src = _source_from_hparams(hp_source)
    preset_src = str(preset_source).strip()
    if meta_src is None:
        return False
    if meta_src == preset_src:
        return True
    if meta_src.startswith("pt:") and preset_src.startswith("pt:"):
        try:
            return Path(meta_src[3:]).resolve() == Path(preset_src[3:]).resolve()
        except OSError:
            return meta_src == preset_src
    return False


def _preset_matches_run_meta(meta: dict[str, Any], preset: dict[str, Any]) -> bool:
    if int(meta.get("epochs", -1)) != int(preset["epochs"]):
        return False
    hp = meta.get("hparams") or {}
    for key in ("max_train_examples", "max_val_examples", "max_test_examples"):
        if int(hp.get(key, -1)) != int(preset[key]):
            return False
    if hp.get("finetune_mode") != preset["finetune_mode"]:
        return False
    if not _source_preset_match(hp.get("source"), preset["source"]):
        return False
    if preset.get("unfreeze_last_n") is not None:
        if int(hp.get("unfreeze_last_n", -1)) != int(preset["unfreeze_last_n"]):
            return False
    return True


def completed_run_dir(preset: dict[str, Any]) -> Path | None:
    """Najnowszy ukończony run (test_metrics.json + zgodność z presetem)."""
    root = arch_root(CLS_ARCHITECTURE)
    if not root.is_dir():
        return None
    tag = preset["run_tag"]
    matches: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if run_tag_from_dir_name(child.name) != tag:
            continue
        if not (child / TEST_METRICS_FILENAME).is_file():
            continue
        meta_path = child / META_FILENAME
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _preset_matches_run_meta(meta, preset):
            matches.append(child)
    if not matches:
        return None
    matches.sort(key=lambda p: p.name)
    return matches[0]


def _training_stats_from_disk(save_dir: Path, architecture: str) -> tuple[float, int]:
    metrics_path = save_dir / "metrics.csv"
    completed_epochs = 0
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8", newline="") as f:
            completed_epochs = sum(1 for _ in csv.DictReader(f))
    best_f1 = 0.0
    best_path = save_dir / f"{architecture}_best_eval.pt"
    if best_path.is_file():
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        best_f1 = float(payload.get("best_f1_macro", 0.0))
        del payload
    return best_f1, completed_epochs


def _estimate_total_steps(n_train: int, per_gpu_batch: int, devices: int, epochs: int) -> int:
    global_bs = max(1, per_gpu_batch * devices)
    steps_per_epoch = max(1, math.ceil(n_train / global_bs))
    return steps_per_epoch * max(1, epochs)


@torch.no_grad()
def evaluate_on_device(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    model.to(device)
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = out["loss"]
        preds = out["logits"].argmax(dim=-1)
        bs = batch["labels"].size(0)
        total_loss += float(loss) * bs
        n += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())
    avg_loss = total_loss / max(1, n)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_LABELS)))
    report = classification_report(
        all_labels,
        all_preds,
        target_names=list(LABEL_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": avg_loss,
        "accuracy": float(report["accuracy"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_examples": n,
    }


def _inference_timing(
    model: nn.Module,
    collate,
    dataset,
    device: torch.device,
    *,
    bench_n: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)
    sync_device(device)
    t0 = time.perf_counter()
    evaluate_on_device(model, loader, device)
    sync_device(device)
    full_seconds = time.perf_counter() - t0
    n_test = len(dataset)

    model.eval()
    subset = list(range(min(bench_n, len(dataset))))
    sync_device(device)
    t1 = time.perf_counter()
    for idx in subset:
        batch = collate([dataset[idx]])
        batch = {k: v.to(device) for k, v in batch.items()}
        model(batch["input_ids"], attention_mask=batch.get("attention_mask"))
    sync_device(device)
    single_seconds = time.perf_counter() - t1
    return {
        "inference_test_full_seconds": full_seconds,
        "inference_test_examples_per_s": n_test / max(full_seconds, 1e-6),
        "inference_bench_n": len(subset),
        "inference_bench_seconds": single_seconds,
        "inference_ms_per_example": 1000.0 * single_seconds / max(1, len(subset)),
    }


def _write_cls_summary(save_dir: Path, sections: dict[str, list[str]]) -> None:
    lines: list[str] = []
    for title, body in sections.items():
        lines.append(f"--- {title} ---")
        lines.extend(body)
        lines.append("")
    (save_dir / "cls_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _plot_cls_training(save_dir: Path) -> list[Path]:
    path = save_dir / METRICS_FILENAME
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col, title in zip(
        axes.flat,
        ("train_loss", "eval_loss", "eval_acc", "eval_f1_macro"),
        ("train loss", "val loss", "val acc", "val F1 macro"),
    ):
        if col in df.columns:
            ax.plot(df["epoch"], pd.to_numeric(df[col], errors="coerce"), marker="o")
        ax.set_title(title)
        ax.set_xlabel("epoch")
    fig.tight_layout()
    p = plots_dir / "training_curves.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)
    meta_path = save_dir / "test_metrics.json"
    if meta_path.is_file():
        tm = json.loads(meta_path.read_text(encoding="utf-8"))
        cm = tm.get("confusion_matrix")
        if cm:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax2)
            ax2.set_title("confusion matrix (test)")
            p2 = plots_dir / "confusion_matrix.png"
            fig2.savefig(p2, dpi=120)
            plt.close(fig2)
            out.append(p2)
    return out


def list_sources() -> None:
    pretrain = models_root() / PRETRAIN_PHASE
    log_print("Dostępne checkpointy LM (*_best_eval.pt):")
    if pretrain.is_dir():
        for pt in sorted(pretrain.glob("**/*_best_eval.pt")):
            tag = run_tag_from_dir_name(pt.parent.name)
            try:
                payload = load_checkpoint_payload(pt, mmap=True)
                hp = payload.get("hparams") or {}
                ppl = payload.get("eval_perplexity")
                params = hp.get("hidden_size")
                log_print(f"  pt:{pt}  run_tag={tag}  hidden={params}  val_ppl={ppl}")
            except Exception as e:
                log_print(f"  pt:{pt}  (błąd odczytu: {e})")
    log_print("Źródła hf: z presetów FINETUNE_ALL:")
    for preset in FINETUNE_ALL:
        src = preset.get("source", "")
        if str(src).startswith("hf:"):
            log_print(f"  {src}  preset_tag={preset.get('run_tag')}")


def main(
    tensorboard_dir: str | None = None,
    *,
    source: str,
    run_tag: str,
    finetune_mode: str,
    dataset: str,
    max_len: int,
    max_train_examples: int,
    max_val_examples: int,
    max_test_examples: int,
    lightning_devices: int,
    batch_size: int,
    lr_scaling: str,
    precision: str,
    epochs: int,
    time_budget_min: float,
    warmup_ratio: float,
    max_grad_norm: float,
    weight_decay: float,
    pooling: str,
    head_hidden: int,
    head_dropout: float,
    inference_bench_examples: int,
    baseline_batch_size: int,
    baseline_learning_rate: float,
    learning_rate_max: float,
    unfreeze_last_n: int | None,
    lora_r: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
    embed_dim: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    ff_size: int | None = None,
    dropout: float | None = None,
    val_checks_per_epoch: int = 1,
    log_every_n_steps: int = 50,
    unfreeze_fractions: list[float] | None = None,
    backbone_num_layers: int | None = None,
) -> None:
    _ = (dataset, val_checks_per_epoch, log_every_n_steps)
    preset = {
        "source": source,
        "run_tag": run_tag,
        "finetune_mode": finetune_mode,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "unfreeze_last_n": unfreeze_last_n,
    }
    _validate_preset(preset)

    if tensorboard_dir is None:
        log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
        tensorboard_dir = tensorboard_run_dir(log_root, datetime.now())

    resolved = resolve_source(source)
    tok_kind = _tokenizer_kind(resolved)
    hf_tok = None
    bpe_tok = None
    hf_model_id = None
    if tok_kind == "hf":
        hf_model_id = resolved.get("model_id")
        hf_tok = offline_first(hf_model_id, AutoTokenizer.from_pretrained, hf_model_id)
        if hf_tok.pad_token_id is None and hf_tok.eos_token_id is not None:
            hf_tok.pad_token = hf_tok.eos_token
    else:
        bpe_tok = load_tokenizer()

    train_ds, val_ds, test_ds, tok_spec, ds_summary = prepare_cls_data(
        max_len=max_len,
        max_train_examples=max_train_examples,
        max_val_examples=max_val_examples,
        max_test_examples=max_test_examples,
        tokenizer_kind=tok_kind,
        hf_tokenizer=hf_tok,
        bpe_tokenizer=bpe_tok,
    )

    scratch_kwargs = None
    if resolved["kind"] == "scratch":
        scratch_kwargs = {
            "embed_dim": embed_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_size": ff_size,
            "dropout": dropout,
        }

    model, model_meta = build_classifier(
        resolved,
        tokenizer_kind=tok_kind,
        vocab_size=tok_spec.vocab_size,
        max_len=max_len,
        scratch_kwargs=scratch_kwargs,
        pooling=pooling,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    )
    model, mode_info = apply_finetune_mode(
        model,
        finetune_mode=finetune_mode,
        unfreeze_last_n=unfreeze_last_n,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        source_kind=resolved["kind"],
    )

    total_p, train_p, bytes_p = count_parameters(model)
    log_print(
        f"Model: params={total_p:,} trainable={train_p:,} "
        f"({bytes_p / 2**20:.2f} MiB) mode={finetune_mode}"
    )

    save_dir = resolve_save_dir(CLS_ARCHITECTURE, run_tag=run_tag)
    log_memory(f"start runu {run_tag}", run_dir=save_dir)
    log_memory("model załadowany", run_dir=save_dir)

    hparams = {
        "task": "classification",
        "architecture": CLS_ARCHITECTURE,
        "run_tag": run_tag,
        "finetune_mode": finetune_mode,
        "pooling": pooling,
        "head_hidden": head_hidden,
        "head_dropout": head_dropout,
        "max_len": max_len,
        "max_train_examples": max_train_examples,
        "max_val_examples": max_val_examples,
        "max_test_examples": max_test_examples,
        "unfreeze_last_n": unfreeze_last_n,
        "unfreeze_fractions": unfreeze_fractions,
        "backbone_num_layers": backbone_num_layers,
        **mode_info,
        **model_meta,
        "source": source,
        "source_resolved": model_meta.get("source"),
    }
    metrics_logger = RunMetricsLogger(save_dir)
    metrics_logger.write_meta(
        architecture=CLS_ARCHITECTURE,
        run_id=save_dir.name,
        tensorboard_dir=tensorboard_dir,
        hparams=hparams,
    )
    model_report = {
        "architecture": model_meta.get("architecture", CLS_ARCHITECTURE),
        "params": total_p,
        "trainable": train_p,
        "model_bytes": bytes_p,
        "source": source,
        "finetune_mode": finetune_mode,
    }
    metrics_logger.write_model_report(model_report, repr(model), dataset=ds_summary)

    per_gpu = int(batch_size)
    devices = lightning_devices if torch.cuda.is_available() else 1
    global_batch = per_gpu * devices
    learning_rate = scaled_learning_rate(
        baseline_batch_size=baseline_batch_size,
        baseline_learning_rate=baseline_learning_rate,
        global_batch=global_batch,
        lr_scaling=lr_scaling,
        learning_rate_max=learning_rate_max,
    )
    log_print(
        f"Fit: batch={per_gpu}/GPU (global {global_batch}), LR={learning_rate:.2e} "
        f"(baseline batch {baseline_batch_size}, lr {baseline_learning_rate})"
    )
    metrics_logger.write_fit(
        {
            "per_gpu_batch": per_gpu,
            "global_batch": global_batch,
            "learning_rate": learning_rate,
            "baseline_batch_size": baseline_batch_size,
            "baseline_learning_rate": baseline_learning_rate,
            "lr_scaling": lr_scaling,
        }
    )

    total_steps = _estimate_total_steps(len(train_ds), per_gpu, devices, epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    collate = collate_from_spec(tok_spec)

    from datetime import timedelta

    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger

    eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lm_src = resolved["kind"]
    log_print("LM perplexity (baza, przed treningiem) na teście…")
    log_memory("przed lm eval (baza)", run_dir=save_dir)
    lm_before = language_model_perplexity_on_dataset(
        model,
        test_ds,
        collate,
        eval_device,
        batch_size=min(16, per_gpu),
        source_kind=lm_src,
    )
    model.cpu()
    _release_gpu_memory()
    log_memory("po lm eval (baza)", run_dir=save_dir)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    use_gpu = torch.cuda.is_available()
    if devices > 1 and use_gpu:
        from lightning.pytorch.strategies import DDPStrategy

        find_unused = finetune_mode in ("lora", "head_only", "unfreeze_last")
        strat = DDPStrategy(start_method="spawn", find_unused_parameters=find_unused)
    else:
        strat = "auto"
    prec = precision or default_lightning_precision("transformer_miniLLaMA")
    log_print(f"Lightning fit: devices={devices}, strategy={strat}, batch/GPU={per_gpu}")
    log_memory("przed fit (Lightning)", run_dir=save_dir)

    lit = LitClassifier(
        model,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=weight_decay,
    )
    pin = use_gpu
    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu,
        shuffle=True,
        collate_fn=collate,
        pin_memory=pin,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_gpu,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin,
        num_workers=0,
    )
    pl_logger = TensorBoardLogger(save_dir=tensorboard_dir, name="") if tensorboard_dir else False
    run_cb = ClsRunLogCallback(
        metrics_save_dir=str(save_dir),
        max_epochs_cap=epochs,
        completed_epochs_at_resume=0,
        resume_best_f1=0.0,
        architecture=CLS_ARCHITECTURE,
    )
    callbacks = [
        TimeBudgetCallback(float(time_budget_min) * 60.0),
        run_cb,
    ]
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        strategy=strat,
        precision=prec,
        max_epochs=epochs,
        max_time=timedelta(minutes=time_budget_min * 1.1),
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=pl_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=max_grad_norm,
        callbacks=callbacks,
    )
    t_run0 = time.perf_counter()
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_segment = time.perf_counter() - t_run0
    if not (save_dir / "metrics.csv").is_file():
        raise RuntimeError(f"Brak metrics.csv po fine-tune w {save_dir}")
    best_f1_macro, completed_epochs = _training_stats_from_disk(save_dir, CLS_ARCHITECTURE)
    best_val_epoch, _ = _best_val_f1_from_metrics(save_dir)

    model = lit.model
    del train_loader, val_loader, trainer, lit
    _release_gpu_memory()
    log_memory("koniec fit (przed testem)", run_dir=save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weights_path = _resolve_eval_weights_path(save_dir, CLS_ARCHITECTURE)
    payload = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    log_print(f"Test na wagach z {weights_path.name}")
    test_loader = DataLoader(test_ds, batch_size=per_gpu, shuffle=False, collate_fn=collate)
    test_metrics = evaluate_on_device(model, test_loader, device)
    timing = _inference_timing(
        model,
        collate,
        test_ds,
        device,
        bench_n=inference_bench_examples,
    )
    test_metrics.update(timing)
    test_metrics["cls_perplexity"] = float(math.exp(min(test_metrics["loss"], 20.0)))
    test_metrics["best_val_f1_macro"] = best_f1_macro
    test_metrics["best_val_f1_epoch"] = best_val_epoch
    for k, v in lm_before.items():
        test_metrics[f"lm_before_{k}"] = v
    log_print("LM perplexity (po fine-tune, best_eval) na teście…")
    log_memory("przed lm eval (po ft)", run_dir=save_dir)
    lm_after = language_model_perplexity_on_dataset(
        model,
        test_ds,
        collate,
        device,
        batch_size=min(16, per_gpu),
        source_kind=lm_src,
    )
    for k, v in lm_after.items():
        test_metrics[f"lm_after_{k}"] = v
    model.cpu()
    del test_loader
    _release_gpu_memory()
    log_memory("koniec runu (po teście)", run_dir=save_dir)
    metrics_logger.write_test_metrics(test_metrics)
    metrics_logger.write_timing(
        train_total_seconds=train_segment,
        epochs=epochs,
        start_epoch=0,
        train_segment_seconds=train_segment,
    )

    _write_cls_summary(
        save_dir,
        {
            "model": [
                f"source: {source}",
                f"finetune_mode: {finetune_mode}",
                f"params: {total_p:,}",
                f"trainable: {train_p:,}",
                f"tokenizer: {tok_spec.tokenizer_repr}",
            ],
            "dataset": [json.dumps(ds_summary, ensure_ascii=False)],
            "training fit": [
                f"per_gpu_batch: {per_gpu}",
                f"global_batch: {global_batch}",
                f"learning_rate: {learning_rate}",
            ],
            "training": [
                f"train_total_seconds: {train_segment:.1f}",
                f"completed_epochs: {completed_epochs}",
                f"best_f1_macro: {best_f1_macro:.4f}",
            ],
            "test": [
                f"accuracy: {test_metrics['accuracy']:.4f}",
                f"f1_macro: {test_metrics['f1_macro']:.4f}",
                f"f1_weighted: {test_metrics['f1_weighted']:.4f}",
                f"inference_ms_per_example: {timing['inference_ms_per_example']:.3f}",
            ],
        },
    )
    _plot_cls_training(save_dir)
    log_print(
        f"Test: acc={test_metrics['accuracy']:.4f}, f1_macro={test_metrics['f1_macro']:.4f}, "
        f"infer={timing['inference_ms_per_example']:.2f} ms/ex"
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning klasyfikacji Tweet Eval")
    parser.add_argument("--list-sources", action="store_true", help="Wypisz dostępne źródła modelu")
    args = parser.parse_args()
    if args.list_sources:
        list_sources()
        return
    parser.print_help()


if __name__ == "__main__":
    _cli()