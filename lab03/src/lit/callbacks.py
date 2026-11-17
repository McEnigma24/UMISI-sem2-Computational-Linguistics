"""Callbacki Lightning — LM (checkpointy, CSV, budżet czasu)."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

from _utils_.log_print import log_print
from _utils_.timing import fmt_duration, log_eta_from_first_batch, sync_device
from metrics.run_metrics import METRICS_EVAL_FILENAME, METRICS_STEPS_FILENAME, RunMetricsLogger


def _loss_from_batch_outputs(outputs) -> torch.Tensor | None:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
        if isinstance(loss, torch.Tensor):
            return loss
    if hasattr(outputs, "loss") and isinstance(outputs.loss, torch.Tensor):
        return outputs.loss
    return None


def _epoch_train_loss(trainer: pl.Trainer, fallback_sum: float, fallback_n: int) -> float:
    cm = trainer.callback_metrics
    for key in ("train/loss_step_epoch", "train/loss_step"):
        v = cm.get(key)
        if v is not None:
            return float(v)
    return fallback_sum / max(1, fallback_n)


def _perplexity_for_log(loss: float) -> tuple[float, str]:
    capped = min(loss, 20.0)
    ppl = math.exp(capped)
    if loss > 20.0:
        return ppl, f"{ppl:.2f} (cap; loss={loss:.2f})"
    return ppl, f"{ppl:.2f}"


def _strategy_uses_fsdp(trainer: pl.Trainer) -> bool:
    name = type(trainer.strategy).__name__.lower()
    return "fsdp" in name


def _full_state_dict(trainer: pl.Trainer, pl_module) -> dict[str, torch.Tensor]:
    strategy = trainer.strategy
    if hasattr(strategy, "lightning_module_state_dict"):
        return dict(strategy.lightning_module_state_dict())
    net = pl_module.net if hasattr(pl_module, "net") else pl_module
    return dict(net.state_dict())


def _optimizer_state_dict(trainer: pl.Trainer, optimizer: torch.optim.Optimizer) -> dict:
    if _strategy_uses_fsdp(trainer):
        return {}
    try:
        return optimizer.state_dict()
    except Exception:
        return {}


class TimeBudgetCallback(Callback):
    def __init__(self, budget_seconds: float, warmup_probe_steps: int = 50) -> None:
        super().__init__()
        self.budget_seconds = budget_seconds
        self.warmup_probe_steps = warmup_probe_steps
        self._t0: float | None = None
        self._planned_logged = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module) -> None:
        self._t0 = time.perf_counter()

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        step = trainer.global_step
        if (
            trainer.is_global_zero
            and not self._planned_logged
            and step >= self.warmup_probe_steps
            and self._t0 is not None
            and hasattr(pl_module, "set_total_steps")
        ):
            elapsed = time.perf_counter() - self._t0
            sps = step / max(elapsed, 1e-6)
            planned = max(step, int(self.budget_seconds * sps))
            pl_module.set_total_steps(planned)
            n_batch = max(1, trainer.num_training_batches)
            log_print(
                f"plan treningu: ~{planned:,} kroków ≈ {planned / n_batch:.1f} epok "
                f"w budżecie {fmt_duration(self.budget_seconds)} ({sps:.2f} krok/s)"
            )
            self._planned_logged = True
        if self._t0 is not None and (time.perf_counter() - self._t0) >= self.budget_seconds:
            if trainer.is_global_zero:
                log_print(f"Budżet czasu wyczerpany ({fmt_duration(self.budget_seconds)}) — stop treningu")
            trainer.should_stop = True


def _log_tb(trainer: pl.Trainer, metrics: dict[str, float], step: int) -> None:
    logger = trainer.logger
    if logger is None:
        return
    if hasattr(logger, "log_metrics"):
        logger.log_metrics(metrics, step=step)
    elif hasattr(logger, "experiment"):
        for k, v in metrics.items():
            logger.experiment.add_scalar(k, v, step)


class RunLogCallback(Callback):
    def __init__(
        self,
        *,
        metrics_save_dir: str,
        checkpoint_every: int,
        eval_every: int,
        max_epochs_cap: int,
        completed_epochs_at_resume: int,
        resume_best_eval: float = float("inf"),
        log_every_n_steps: int = 50,
        architecture: str,
    ) -> None:
        super().__init__()
        self.metrics_save_dir = Path(metrics_save_dir)
        self._metrics_logger: RunMetricsLogger | None = None
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.max_epochs_cap = max_epochs_cap
        self.completed_epochs_at_resume = completed_epochs_at_resume
        self.best_eval_loss = resume_best_eval
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.architecture = architecture
        self._epoch_train_sum = 0.0
        self._epoch_train_n = 0
        self._t_train0 = 0.0
        self._t_eval0 = 0.0
        self._global_tokens = 0
        self._eta_logged = False
        self._last_eval_loss: float | None = None
        self._last_eval_ppl: float | None = None
        self._last_eval_seconds = 0.0
        self._val_this_epoch = False
        self.last_optimizer: torch.optim.Optimizer | None = None
        self._steps_path = self.metrics_save_dir / METRICS_STEPS_FILENAME
        self._eval_path = self.metrics_save_dir / METRICS_EVAL_FILENAME

    def _logical_epoch(self, trainer: pl.Trainer) -> int:
        return self.completed_epochs_at_resume + trainer.current_epoch + 1

    def on_fit_start(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.is_global_zero and self._metrics_logger is None:
            self.metrics_save_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_logger = RunMetricsLogger(self.metrics_save_dir)
            if hasattr(pl_module, "net"):
                pl_module.net.metrics_logger = self._metrics_logger

    def _append_step_row(self, row: dict[str, Any]) -> None:
        write_header = not self._steps_path.is_file()
        fields = list(row.keys())
        with self._steps_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        loss = _loss_from_batch_outputs(outputs)
        if loss is not None:
            self._epoch_train_sum += loss.detach().float().item()
            self._epoch_train_n += 1
        tok = getattr(pl_module, "_tokens_per_step", 0) * trainer.world_size
        self._global_tokens += tok
        if trainer.is_global_zero and batch_idx == 0 and not self._eta_logged:
            t0 = time.perf_counter()
            sync_device(pl_module.device)
            log_eta_from_first_batch(
                phase="trening epoki",
                batch_seconds=time.perf_counter() - t0,
                n_batches=len(trainer.train_dataloader),
            )
            self._eta_logged = True
        step = trainer.global_step
        if trainer.is_global_zero and step > 0 and step % self.log_every_n_steps == 0:
            lr = 0.0
            opt = pl_module.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            if opt is not None and opt.param_groups:
                lr = float(opt.param_groups[0]["lr"])
            loss_v = float(loss.detach()) if loss is not None else float("nan")
            self._append_step_row(
                {
                    "step": step,
                    "epoch": self._logical_epoch(trainer),
                    "train_loss": loss_v,
                    "lr": lr,
                    "tokens_seen": self._global_tokens,
                }
            )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        self._val_this_epoch = False
        self._t_train0 = time.perf_counter()
        self._epoch_train_sum = 0.0
        self._epoch_train_n = 0
        self._eta_logged = False
        loader = trainer.train_dataloader
        n = len(loader) if loader is not None else 0
        bs = getattr(loader, "batch_size", "?")
        ep = self._logical_epoch(trainer)
        log_print(f"Epoka {ep}/{self.max_epochs_cap}: {n:,} batchy (batch_size={bs})")

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if not trainer.is_global_zero:
            return
        loss_t = outputs if isinstance(outputs, torch.Tensor) else _loss_from_batch_outputs(outputs)
        if loss_t is None:
            return
        loss = float(loss_t.detach())
        ep = self._logical_epoch(trainer)
        write_header = not self._eval_path.is_file()
        fields = ["epoch", "batch_idx", "step", "eval_loss", "eval_perplexity"]
        ppl = math.exp(min(loss, 20.0))
        row = {
            "epoch": ep,
            "batch_idx": batch_idx,
            "step": trainer.global_step,
            "eval_loss": loss,
            "eval_perplexity": ppl,
        }
        with self._eval_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self._t_eval0 = time.perf_counter()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        self._val_this_epoch = True
        eval_loss_t = trainer.callback_metrics.get("val_loss")
        if eval_loss_t is None:
            return
        eval_loss = float(eval_loss_t)
        eval_ppl, eval_ppl_s = _perplexity_for_log(eval_loss)
        self._last_eval_loss = eval_loss
        self._last_eval_ppl = eval_ppl
        self._last_eval_seconds = time.perf_counter() - self._t_eval0
        if trainer.is_global_zero:
            ep = self._logical_epoch(trainer)
            log_print(f"Val: loss={eval_loss:.4f}, eval_perplexity={eval_ppl_s}")
            _log_tb(
                trainer,
                {"eval/loss": eval_loss, "eval/perplexity": eval_ppl},
                step=trainer.global_step,
            )

        net = pl_module.net
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            opt = pl_module.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            state = _full_state_dict(trainer, pl_module)
            opt_state = _optimizer_state_dict(trainer, opt) if opt is not None else {}
            if trainer.is_global_zero:
                net.save_model_bundle_from_state(
                    state,
                    opt_state,
                    self._logical_epoch(trainer),
                    eval_loss,
                    eval_ppl,
                    self.best_eval_loss,
                    stem="best_eval",
                    export_onnx=False,
                )
                log_print(f"★ Nowy best eval: loss={eval_loss:.4f}, ppl={eval_ppl:.2f}")

        fsdp = _strategy_uses_fsdp(trainer)
        ep = self._logical_epoch(trainer)
        do_sample = not fsdp and (ep % self.eval_every == 0 or ep == 1)
        if do_sample and trainer.is_global_zero:
            sample, _ = net.generate(net, seed="W", length=200, temperature=0.8)
            log_print(f"Generacja (200 tok.): {sample[:200]!r}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        train_seconds = time.perf_counter() - self._t_train0
        train_loss = _epoch_train_loss(trainer, self._epoch_train_sum, self._epoch_train_n)
        train_ppl, train_ppl_s = _perplexity_for_log(train_loss)
        ep = self._logical_epoch(trainer)
        if trainer.is_global_zero:
            log_print(
                f"Epoka {ep}: loss={train_loss:.4f} train_perplexity={train_ppl_s} "
                f"({train_seconds:.1f}s train)"
            )
            _log_tb(
                trainer,
                {"train/loss": train_loss, "train/perplexity": train_ppl},
                step=trainer.global_step,
            )

        if self._val_this_epoch and self._last_eval_loss is not None:
            eval_loss = self._last_eval_loss
            eval_ppl = float(self._last_eval_ppl or math.exp(min(eval_loss, 20.0)))
            eval_seconds = self._last_eval_seconds
        else:
            eval_loss = float("nan")
            eval_ppl = float("nan")
            eval_seconds = 0.0

        if trainer.is_global_zero and self._metrics_logger is not None:
            self._metrics_logger.log_epoch(
                float(ep),
                train_loss,
                train_ppl,
                eval_loss,
                eval_ppl,
                train_seconds,
                eval_seconds,
                step=trainer.global_step,
                tokens_seen=self._global_tokens,
            )

        opt = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        self.last_optimizer = opt
        if ep % self.checkpoint_every == 0:
            ckpt_eval = float(self._last_eval_loss or self.best_eval_loss)
            ckpt_ppl = float(self._last_eval_ppl or math.exp(min(ckpt_eval, 20.0)))
            state = _full_state_dict(trainer, pl_module)
            opt_state = _optimizer_state_dict(trainer, opt) if opt is not None else {}
            if trainer.is_global_zero:
                pl_module.net.save_model_bundle_from_state(
                    state,
                    opt_state,
                    ep,
                    ckpt_eval if ckpt_eval == ckpt_eval else 0.0,
                    ckpt_ppl,
                    self.best_eval_loss,
                    stem=f"epoch_{ep:04d}",
                    export_onnx=False,
                    checkpoints_subdir=True,
                )


class ResumeLitCallback(Callback):
    def __init__(
        self,
        payload: dict | None,
        *,
        load_optimizer: bool = True,
    ) -> None:
        super().__init__()
        self.payload = payload
        self.load_optimizer = load_optimizer

    def on_fit_start(self, trainer: pl.Trainer, pl_module) -> None:
        if self.payload is None:
            return
        state = self.payload["model_state_dict"]
        if hasattr(pl_module, "net"):
            pl_module.net.load_state_dict(state)
        else:
            pl_module.load_state_dict(state, strict=False)
        if not self.load_optimizer or _strategy_uses_fsdp(trainer):
            if trainer.is_global_zero and _strategy_uses_fsdp(trainer):
                log_print("Resume: stan optymalizatora pominięty (FSDP)")
            elif trainer.is_global_zero:
                log_print(f"Lightning: wczytano wagi (epoka {self.payload.get('epoch')})")
            return
        opt = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        try:
            opt.load_state_dict(self.payload["optimizer_state_dict"])
        except Exception:
            pass
        if trainer.is_global_zero:
            log_print(f"Lightning: wczytano resume (epoka {self.payload.get('epoch')})")


def trainer_is_global_zero() -> bool:
    import os

    return os.environ.get("LOCAL_RANK", "0") == "0"