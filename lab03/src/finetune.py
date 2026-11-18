"""Fine-tuning klasyfikacji Tweet Eval (sentiment)."""

from __future__ import annotations

import argparse
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
from finetune_presets import FINETUNE_ALL
from lit.cls_callbacks import ClsRunLogCallback, cls_early_stopping
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
    if source == "scratch" and mode != "full":
        raise ValueError("scratch wspiera wyłącznie finetune_mode=full")
    if source == "scratch":
        for key in ("embed_dim", "hidden_size", "num_layers", "dropout"):
            if key not in preset:
                raise ValueError(f"scratch wymaga {key}")


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


def compare_runs(run_dirs: list[Path] | None = None, *, run_tags: list[str] | None = None) -> Path:
    cls_runs_root = arch_root(CLS_ARCHITECTURE)
    candidates: list[Path] = []
    if run_dirs:
        candidates = [Path(p) for p in run_dirs]
    elif run_tags:
        for child in cls_runs_root.iterdir() if cls_runs_root.is_dir() else []:
            if run_tag_from_dir_name(child.name) in run_tags:
                candidates.append(child)
    else:
        if cls_runs_root.is_dir():
            for child in cls_runs_root.iterdir():
                if (child / METRICS_FILENAME).is_file():
                    candidates.append(child)
    if not candidates:
        raise FileNotFoundError("Brak runów do porównania")

    rows = []
    for save_dir in candidates:
        meta = {}
        if (save_dir / META_FILENAME).is_file():
            meta = json.loads((save_dir / META_FILENAME).read_text(encoding="utf-8"))
        test_m = meta.get("test_metrics") or {}
        model_m = meta.get("model") or {}
        rows.append(
            {
                "run_id": save_dir.name,
                "run_tag": run_tag_from_dir_name(save_dir.name),
                "source": meta.get("hparams", {}).get("source"),
                "finetune_mode": meta.get("hparams", {}).get("finetune_mode"),
                "params_total": model_m.get("params"),
                "params_trainable": model_m.get("trainable"),
                "test_accuracy": test_m.get("accuracy"),
                "test_f1_macro": test_m.get("f1_macro"),
                "test_f1_weighted": test_m.get("f1_weighted"),
                "train_total_seconds": meta.get("train_total_seconds"),
                "inference_ms_per_example": test_m.get("inference_ms_per_example"),
            }
        )
    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parents[1] / "model_evaluations" / f"cls_compare_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cls_compare.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for run_id, grp in df.groupby("run_id"):
        save_dir = next(p for p in candidates if p.name == run_id)
        hist = pd.read_csv(save_dir / METRICS_FILENAME)
        ax.plot(
            pd.to_numeric(hist["epoch"], errors="coerce"),
            pd.to_numeric(hist["eval_f1_macro"], errors="coerce"),
            marker="o",
            label=run_id,
        )
    ax.set_title("val F1 macro — porównanie runów")
    ax.legend(fontsize=7)
    ax.set_xlabel("epoch")
    fig.savefig(out_dir / "cls_val_curves.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    x = range(len(df))
    ax2.bar(x, pd.to_numeric(df["test_f1_macro"], errors="coerce"))
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(df["run_tag"], rotation=30, ha="right")
    ax2.set_title("test F1 macro")
    fig2.savefig(out_dir / "cls_test_bars.png", dpi=120)
    plt.close(fig2)
    log_print(f"compare_runs → {out_dir}")
    return out_dir


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
    early_stopping_patience: int,
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

    hparams = {
        "task": "classification",
        "architecture": CLS_ARCHITECTURE,
        "source": source,
        "finetune_mode": finetune_mode,
        "pooling": pooling,
        "head_hidden": head_hidden,
        "head_dropout": head_dropout,
        "max_len": max_len,
        **mode_info,
        **model_meta,
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

    from datetime import timedelta

    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger

    model.cpu()
    _release_gpu_memory()
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    use_gpu = torch.cuda.is_available()
    strat = "ddp_spawn" if devices > 1 and use_gpu else "auto"
    prec = precision or default_lightning_precision("transformer_miniLLaMA")
    log_print(f"Lightning fit: devices={devices}, strategy={strat}, batch/GPU={per_gpu}")

    lit = LitClassifier(
        model,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=weight_decay,
    )
    collate = collate_from_spec(tok_spec)
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
        cls_early_stopping(patience=early_stopping_patience),
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
    result = {
        "best_f1_macro": run_cb.best_f1_macro,
        "early_stopped": run_cb.early_stopped,
        "completed_epochs": run_cb.completed_epochs,
    }
    if not (save_dir / "metrics.csv").is_file():
        raise RuntimeError(f"Brak metrics.csv po fine-tune w {save_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = lit.model
    model.to(device)
    best_path = save_dir / f"{CLS_ARCHITECTURE}_best_eval.pt"
    final_path = save_dir / f"{CLS_ARCHITECTURE}_final.pt"
    weights_path = best_path if best_path.is_file() else final_path
    if not weights_path.is_file():
        raise FileNotFoundError(f"Brak {best_path.name} i {final_path.name} w {save_dir}")
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
    model.cpu()
    del test_loader
    _release_gpu_memory()
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
                f"best_f1_macro: {result.get('best_f1_macro')}",
                f"early_stopped: {result.get('early_stopped')}",
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
    parser.add_argument("--compare", action="store_true", help="Porównaj runy cls")
    args = parser.parse_args()
    if args.list_sources:
        list_sources()
        return
    if args.compare:
        compare_runs()
        return
    parser.print_help()


if __name__ == "__main__":
    _cli()