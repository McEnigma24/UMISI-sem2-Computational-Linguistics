"""Uruchamianie workerów GPU bez ponownego exec main.py."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.utils.data import DataLoader, DistributedSampler

from _utils_.log_print import log_print
from lit.callbacks import ResumeLitCallback, RunLogCallback, TimeBudgetCallback
from lit.lm_module import LitLanguageModel, default_lightning_precision


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _isolate_gpu_for_rank(rank: int) -> None:
    """Jeden proces = jedna widoczna karta (inaczej NCCL: duplicate GPU)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        ids = [p.strip() for p in env.split(",") if p.strip()]
        if len(ids) > 1 and rank < len(ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = ids[rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)


def _set_dist_env(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["NODE_RANK"] = "0"
    # Po izolacji GPU w procesie jest tylko cuda:0
    os.environ["LOCAL_RANK"] = "0"


def _setup_worker(rank: int, world_size: int, port: int) -> None:
    _isolate_gpu_for_rank(rank)
    _set_dist_env(rank, world_size, port)
    _init_process_group(world_size)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")


def _init_process_group(world_size: int) -> None:
    """Przed DistributedSampler — Lightning nie inicjuje grupy przed utworzeniem loadera."""
    if world_size <= 1:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")


@dataclass
class LmFitSpec:
    save_dir: str
    architecture: str
    model_kwargs: dict[str, Any]
    seq_len: int
    window_stride: int
    max_corpus_chars: int
    batch_size: int
    learning_rate: float
    epochs: int
    start_epoch: int
    devices: int
    strategy: str  # ddp | fsdp
    precision: str | None
    checkpoint_every: int
    eval_every: int
    time_budget_min: float
    max_grad_norm: float
    tensorboard_dir: str | None
    resume_payload: dict | None
    val_checks_per_epoch: int
    log_every_n_steps: int
    completed_epochs_at_resume: int
    resume_best_eval: float


FIT_RESULT_NAME = "_fit_result.pt"


def _fsdp_strategy(architecture: str):
    from models.transformer_miniLLaMA import Block as LlamaBlock

    auto_wrap = {LlamaBlock}
    return FSDPStrategy(auto_wrap_policy=auto_wrap, sharding_strategy="FULL_SHARD", use_orig_params=True)


def lm_fit_worker(rank: int, spec: LmFitSpec, port: int) -> None:
    _setup_worker(rank, spec.devices, port)

    from pathlib import Path

    from text_and_tokenizer import prepare_language_data
    from train import LanguageModelNet, resolve_device

    device = resolve_device(torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(spec.save_dir)
    train_ds, val_ds, _test_ds, _tok, _vocab, _summary = prepare_language_data(
        spec.seq_len,
        window_stride=spec.window_stride,
        max_corpus_chars=spec.max_corpus_chars,
    )
    net = LanguageModelNet(
        architecture=spec.architecture,
        device=device,
        save_dir=save_dir,
        seq_len=spec.seq_len,
        window_stride=spec.window_stride,
        max_corpus_chars=spec.max_corpus_chars,
        **{k: v for k, v in spec.model_kwargs.items() if k not in ("seq_len", "window_stride", "max_corpus_chars")},
    )
    net.save_dir = save_dir
    lit = LitLanguageModel(net, learning_rate=spec.learning_rate, max_grad_norm=spec.max_grad_norm)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if spec.devices > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if spec.devices > 1 else None
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=pin,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=spec.batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=pin,
        num_workers=0,
    )

    pl_logger = TensorBoardLogger(save_dir=spec.tensorboard_dir, name="") if spec.tensorboard_dir else False
    prec = spec.precision or default_lightning_precision(spec.architecture)
    budget_sec = float(spec.time_budget_min) * 60.0
    fit_epochs = max(0, spec.epochs - spec.start_epoch)
    val_interval = 1.0 / max(1, spec.val_checks_per_epoch)

    run_cb = RunLogCallback(
        metrics_save_dir=spec.save_dir,
        checkpoint_every=spec.checkpoint_every,
        eval_every=spec.eval_every,
        max_epochs_cap=spec.epochs,
        completed_epochs_at_resume=spec.completed_epochs_at_resume,
        resume_best_eval=spec.resume_best_eval,
        log_every_n_steps=spec.log_every_n_steps,
        architecture=spec.architecture,
    )
    load_opt = spec.strategy == "ddp"
    callbacks = [
        TimeBudgetCallback(budget_sec),
        run_cb,
        ResumeLitCallback(spec.resume_payload, load_optimizer=load_opt),
    ]

    cluster = LightningEnvironment()
    strategy: Any = "auto"
    if spec.devices > 1 and torch.cuda.is_available():
        if spec.strategy == "ddp":
            strategy = DDPStrategy(cluster_environment=cluster)
        else:
            strategy = _fsdp_strategy(spec.architecture)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        precision=prec,
        max_epochs=fit_epochs,
        max_time=timedelta(minutes=spec.time_budget_min * 1.1),
        check_val_every_n_epoch=max(1, spec.eval_every),
        val_check_interval=val_interval,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=pl_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=spec.max_grad_norm,
        callbacks=callbacks,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if rank == 0:
        opt = run_cb.last_optimizer
        opt_state = opt.state_dict() if opt is not None else {}
        state = dict(net.state_dict())
        if hasattr(trainer.strategy, "lightning_module_state_dict"):
            try:
                state = dict(trainer.strategy.lightning_module_state_dict())
            except Exception:
                pass
        out = {
            "model_state_dict": state,
            "optimizer_state_dict": opt_state,
            "best_eval_loss": run_cb.best_eval_loss,
        }
        torch.save(out, save_dir / FIT_RESULT_NAME)


def _spawn_entry(rank: int, spec: LmFitSpec, port: int) -> None:
    lm_fit_worker(rank, spec, port)


@dataclass
class ClsFitSpec:
    save_dir: str
    architecture: str
    init_state_path: str
    source: str
    finetune_mode: str
    unfreeze_last_n: int | None
    lora_r: int | None
    lora_alpha: int | None
    lora_dropout: float | None
    scratch_kwargs: dict[str, Any] | None
    pooling: str
    head_hidden: int
    head_dropout: float
    max_len: int
    max_train_examples: int
    max_val_examples: int
    max_test_examples: int
    tokenizer_kind: str
    hf_model_id: str | None
    batch_size: int
    learning_rate: float
    epochs: int
    start_epoch: int
    devices: int
    strategy: str
    precision: str | None
    time_budget_min: float
    max_grad_norm: float
    tensorboard_dir: str | None
    weight_decay: float
    warmup_steps: int
    total_steps: int
    early_stopping_patience: int
    completed_epochs_at_resume: int
    resume_best_f1: float


def _cls_fsdp_strategy(model):
    core = model
    if hasattr(core, "core"):
        core = core.core
    policy = None
    if hasattr(core, "_no_split_modules") and core._no_split_modules:
        mods = set(core._no_split_modules)
        policy = {getattr(core, m.split(".")[-1], type(core)) for m in mods}
    if not policy:
        from models.transformer_miniLLaMA import Block as LlamaBlock

        policy = {LlamaBlock}
    return FSDPStrategy(auto_wrap_policy=policy, sharding_strategy="FULL_SHARD", use_orig_params=True)


def cls_fit_worker(rank: int, spec: ClsFitSpec, port: int) -> None:
    _setup_worker(rank, spec.devices, port)

    from finetuning.cls_data import collate_from_spec, prepare_cls_data
    from finetuning.cls_models import apply_finetune_mode, build_classifier, resolve_source
    from lit.cls_callbacks import ClsRunLogCallback, cls_early_stopping
    from lit.cls_module import LitClassifier
    from lit.lm_module import default_lightning_precision
    from text_and_tokenizer import load_tokenizer
    from transformers import AutoTokenizer

    save_dir = Path(spec.save_dir)
    resolved = resolve_source(spec.source)
    hf_tok = None
    bpe_tok = None
    if spec.tokenizer_kind == "hf":
        model_id = spec.hf_model_id or (resolved.get("model_id") if resolved["kind"] == "hf" else None)
        if model_id:
            hf_tok = AutoTokenizer.from_pretrained(model_id)
            if hf_tok.pad_token_id is None and hf_tok.eos_token_id is not None:
                hf_tok.pad_token = hf_tok.eos_token
    else:
        bpe_tok = load_tokenizer()

    train_ds, val_ds, _test, tok_spec, _summary = prepare_cls_data(
        max_len=spec.max_len,
        max_train_examples=spec.max_train_examples,
        max_val_examples=spec.max_val_examples,
        max_test_examples=spec.max_test_examples,
        tokenizer_kind=spec.tokenizer_kind,
        hf_tokenizer=hf_tok,
        bpe_tokenizer=bpe_tok,
    )
    model, _meta = build_classifier(
        resolved,
        tokenizer_kind=spec.tokenizer_kind,
        vocab_size=tok_spec.vocab_size,
        max_len=spec.max_len,
        scratch_kwargs=spec.scratch_kwargs,
        pooling=spec.pooling,
        head_hidden=spec.head_hidden,
        head_dropout=spec.head_dropout,
    )
    model, _mode_info = apply_finetune_mode(
        model,
        finetune_mode=spec.finetune_mode,
        unfreeze_last_n=spec.unfreeze_last_n,
        lora_r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        source_kind=resolved["kind"],
    )
    init_path = Path(spec.init_state_path)
    if init_path.is_file():
        state = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)

    lit = LitClassifier(
        model,
        learning_rate=spec.learning_rate,
        max_grad_norm=spec.max_grad_norm,
        warmup_steps=spec.warmup_steps,
        total_steps=spec.total_steps,
        weight_decay=spec.weight_decay,
    )
    collate = collate_from_spec(tok_spec)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if spec.devices > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if spec.devices > 1 else None
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=spec.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
        pin_memory=pin,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=spec.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate,
        pin_memory=pin,
        num_workers=0,
    )

    pl_logger = TensorBoardLogger(save_dir=spec.tensorboard_dir, name="") if spec.tensorboard_dir else False
    prec = spec.precision or default_lightning_precision("transformer_miniLLaMA")
    fit_epochs = max(0, spec.epochs - spec.start_epoch)
    run_cb = ClsRunLogCallback(
        metrics_save_dir=spec.save_dir,
        max_epochs_cap=spec.epochs,
        completed_epochs_at_resume=spec.completed_epochs_at_resume,
        resume_best_f1=spec.resume_best_f1,
        architecture=spec.architecture,
    )
    callbacks = [
        TimeBudgetCallback(float(spec.time_budget_min) * 60.0),
        run_cb,
        cls_early_stopping(patience=spec.early_stopping_patience),
    ]
    cluster = LightningEnvironment()
    strategy: Any = "auto"
    if spec.devices > 1 and torch.cuda.is_available():
        if spec.strategy == "ddp":
            strategy = DDPStrategy(cluster_environment=cluster)
        else:
            strategy = _cls_fsdp_strategy(lit.model)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        precision=prec,
        max_epochs=fit_epochs,
        max_time=timedelta(minutes=spec.time_budget_min * 1.1),
        check_val_every_n_epoch=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=pl_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=spec.max_grad_norm,
        callbacks=callbacks,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if rank == 0:
        opt = run_cb.last_optimizer
        opt_state = opt.state_dict() if opt is not None else {}
        state = dict(lit.model.state_dict())
        if hasattr(trainer.strategy, "lightning_module_state_dict"):
            try:
                state = dict(trainer.strategy.lightning_module_state_dict())
            except Exception:
                pass
        out = {
            "model_state_dict": state,
            "optimizer_state_dict": opt_state,
            "best_f1_macro": run_cb.best_f1_macro,
            "early_stopped": run_cb.early_stopped,
        }
        torch.save(out, save_dir / FIT_RESULT_NAME)


def _cls_spawn_entry(rank: int, spec: ClsFitSpec, port: int) -> None:
    cls_fit_worker(rank, spec, port)


def run_cls_fit(spec: ClsFitSpec, *, devices: int) -> dict | None:
    if devices < 1:
        raise ValueError("devices >= 1")
    port = _free_port()
    if devices == 1:
        cls_fit_worker(0, spec, port)
    else:
        mp.spawn(_cls_spawn_entry, args=(spec, port), nprocs=devices, join=True)
    result_path = Path(spec.save_dir) / FIT_RESULT_NAME
    if not result_path.is_file():
        log_print("Brak _fit_result.pt po cls fit")
        return None
    payload = torch.load(result_path, map_location="cpu", weights_only=False)
    result_path.unlink(missing_ok=True)
    return payload


def run_lm_fit(spec: LmFitSpec, *, devices: int) -> dict | None:
    """Uruchamia trening na workerach; zwraca wynik z rank 0 (w pliku)."""
    if devices < 1:
        raise ValueError("devices >= 1")
    port = _free_port()
    if devices == 1:
        lm_fit_worker(0, spec, port)
    else:
        mp.spawn(_spawn_entry, args=(spec, port), nprocs=devices, join=True)
    result_path = Path(spec.save_dir) / FIT_RESULT_NAME
    if not result_path.is_file():
        log_print("Brak _fit_result.pt po fit — worker mógł przerwać trening")
        return None
    payload = torch.load(result_path, map_location="cpu", weights_only=False)
    result_path.unlink(missing_ok=True)
    return payload