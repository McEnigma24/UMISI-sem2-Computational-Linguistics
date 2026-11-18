"""Callbacki Lightning dla klasyfikacji (CSV, best F1, early stopping)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping

from _utils_.log_print import log_print
from _utils_.timing import log_eta_from_first_batch, sync_device
from lit.callbacks import _loss_from_batch_outputs


CLS_METRICS_COLUMNS = (
    "epoch",
    "train_loss",
    "train_acc",
    "eval_loss",
    "eval_acc",
    "eval_f1_macro",
    "eval_f1_weighted",
    "train_seconds",
    "eval_seconds",
)


def _unwrap_classifier_state(pl_module) -> dict[str, torch.Tensor]:
    state = dict(pl_module.model.state_dict())
    if any(k.startswith("model.") for k in state):
        state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    return state


class ClsRunLogCallback(Callback):
    def __init__(
        self,
        *,
        metrics_save_dir: str,
        max_epochs_cap: int,
        completed_epochs_at_resume: int,
        resume_best_f1: float,
        architecture: str,
    ) -> None:
        super().__init__()
        self.metrics_save_dir = Path(metrics_save_dir)
        self.max_epochs_cap = max_epochs_cap
        self.completed_epochs_at_resume = completed_epochs_at_resume
        self.best_f1_macro = resume_best_f1
        self.architecture = architecture
        self._t_train0 = 0.0
        self._t_eval0 = 0.0
        self._epoch_train_sum = 0.0
        self._epoch_train_n = 0
        self._last_eval: dict[str, float] = {}
        self.last_optimizer: torch.optim.Optimizer | None = None
        self.early_stopped = False
        self.completed_epochs = 0

    def _logical_epoch(self, trainer: pl.Trainer) -> int:
        return self.completed_epochs_at_resume + trainer.current_epoch + 1

    def on_fit_start(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self.metrics_save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        self._t_train0 = time.perf_counter()
        self._epoch_train_sum = 0.0
        self._epoch_train_n = 0
        loader = trainer.train_dataloader
        n = len(loader) if loader is not None else 0
        bs = getattr(loader, "batch_size", "?")
        ep = self._logical_epoch(trainer)
        log_print(f"Epoka {ep}/{self.max_epochs_cap}: {n:,} batchy (batch_size={bs})")

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx: int
    ) -> None:
        loss = _loss_from_batch_outputs(outputs)
        if loss is not None:
            self._epoch_train_sum += loss.detach().float().item()
            self._epoch_train_n += 1
        if trainer.is_global_zero and batch_idx == 0:
            t0 = time.perf_counter()
            sync_device(pl_module.device)
            log_eta_from_first_batch(
                phase="trening epoki",
                batch_seconds=time.perf_counter() - t0,
                n_batches=len(trainer.train_dataloader),
            )

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self._t_eval0 = time.perf_counter()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        cm = trainer.callback_metrics
        eval_loss = float(cm.get("val/loss", float("nan")))
        eval_acc = float(cm.get("val/acc", float("nan")))
        eval_f1m = float(cm.get("val/f1_macro", float("nan")))
        eval_f1w = float(cm.get("val/f1_weighted", float("nan")))
        self._last_eval = {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_f1_macro": eval_f1m,
            "eval_f1_weighted": eval_f1w,
        }
        if trainer.is_global_zero:
            ep = self._logical_epoch(trainer)
            log_print(
                f"Val: loss={eval_loss:.4f}, acc={eval_acc:.4f}, "
                f"f1_macro={eval_f1m:.4f}, f1_weighted={eval_f1w:.4f}"
            )

        if eval_f1m > self.best_f1_macro and trainer.is_global_zero:
            self.best_f1_macro = eval_f1m
            state = _unwrap_classifier_state(pl_module)
            payload = {
                "epoch": self._logical_epoch(trainer),
                "model_state_dict": state,
                "best_f1_macro": self.best_f1_macro,
                "eval_acc": eval_acc,
                "eval_f1_weighted": eval_f1w,
                "eval_loss": eval_loss,
            }
            path = self.metrics_save_dir / f"{self.architecture}_best_eval.pt"
            torch.save(payload, path)
            del payload, state
            log_print(f"★ Nowy best val f1_macro={eval_f1m:.4f}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        ep = self._logical_epoch(trainer)
        opt = pl_module.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        self.completed_epochs = ep
        train_seconds = time.perf_counter() - self._t_train0
        eval_seconds = time.perf_counter() - self._t_eval0 if self._t_eval0 else 0.0
        cm = trainer.callback_metrics
        train_loss = float(cm.get("train/loss_step_epoch", cm.get("train/loss", float("nan"))))
        train_acc = float(cm.get("train/acc", float("nan")))
        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": self._last_eval.get("eval_loss", ""),
            "eval_acc": self._last_eval.get("eval_acc", ""),
            "eval_f1_macro": self._last_eval.get("eval_f1_macro", ""),
            "eval_f1_weighted": self._last_eval.get("eval_f1_weighted", ""),
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        }
        metrics_path = self.metrics_save_dir / "metrics.csv"
        write_header = not metrics_path.is_file()
        with metrics_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CLS_METRICS_COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow(row)
        log_print(
            f"Epoka {ep}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} "
            f"({train_seconds:.1f}s train, {eval_seconds:.1f}s val)"
        )
        self.last_optimizer = opt

    def on_train_end(self, trainer: pl.Trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        state = _unwrap_classifier_state(pl_module)
        payload = {
            "epoch": self.completed_epochs,
            "model_state_dict": state,
            "best_f1_macro": self.best_f1_macro,
        }
        final_path = self.metrics_save_dir / f"{self.architecture}_final.pt"
        torch.save(payload, final_path)
        del payload, state

    def on_fit_end(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.is_global_zero and trainer.should_stop:
            self.early_stopped = True


def cls_early_stopping(*, patience: int) -> EarlyStopping:
    return EarlyStopping(
        monitor="val/f1_macro",
        mode="max",
        patience=patience,
        check_on_train_epoch_end=False,
    )