"""Trening language """

from __future__ import annotations

import copy
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model_report import build_model_report
from models.transformer_GPT2small import transformer_GPT2small
from models.transformer_miniLLaMA import transformer_miniLLaMA
from models.transformer_nanoGPT import transformer_nanoGPT
from text_and_tokenizer import load_tokenizer, prepare_language_data, tokenizer_dir
from metrics.plot_training import plot_training_run, write_training_report
from metrics.run_metrics import METRICS_FILENAME, RunMetricsLogger
from lit.hw_fit import choose_batch_and_strategy, scaled_learning_rate
from _utils_.log_print import log_print, tensorboard_run_dir
from _utils_.timing import fmt_duration as _fmt_duration
from _utils_.timing import log_eta_from_first_batch as _log_eta_from_first_batch
from _utils_.timing import sync_device as _sync_device
from _utils_.run_paths import (
    find_latest_run_dir,
    latest_checkpoint,
    resume_run_relocate,
    project_root,
    resolve_resume_from_env_or_arg,
    resolve_save_dir,
    resolve_source_run_dir,
)


def checkpoint_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_checkpoint_payload(path: Path | str) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _epoch_from_checkpoint_path(path: Path) -> int | None:
    if not path.name.startswith("epoch_") or path.suffix != ".pt":
        return None
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _max_epoch_in_checkpoints(checkpoints_dir: Path) -> int:
    mx = 0
    if not checkpoints_dir.is_dir():
        return 0
    for path in checkpoints_dir.glob("epoch_*.pt"):
        ep = _epoch_from_checkpoint_path(path)
        if ep is not None:
            mx = max(mx, ep)
    return mx


def _checkpoint_mod_epochs(final_epoch: int, mod: int) -> set[int]:
    """Reguła cykliczna: ep. 1 + każda wielokrotność mod (nie kasuje poza sobą)."""
    if final_epoch < 1 or mod < 1:
        return set()
    keep = {1}
    if mod > 1:
        keep.update(range(mod, final_epoch + 1, mod))
    else:
        keep.update(range(1, final_epoch + 1))
    return keep


def _checkpoint_tail_epochs(final_epoch: int, tail_keep: int) -> set[int]:
    """Reguła ogona: ostatnie tail_keep epok (nie wyłącznie — łączy się z mod)."""
    if final_epoch < 1 or tail_keep < 1:
        return set()
    tail_start = max(1, final_epoch - tail_keep + 1)
    return set(range(tail_start, final_epoch + 1))


def _checkpoint_may_delete(ep: int, mod_epochs: set[int], tail_epochs: set[int]) -> bool:
    """Usuń tylko gdy obie reguły na to pozwalają (nie chroni mod i nie chroni tail)."""
    return ep not in mod_epochs and ep not in tail_epochs


def compact_epoch_checkpoints(
    save_dir: Path | str,
    final_epoch: int,
    *,
    checkpoint_prune_post_training_mod: int,
    checkpoint_prune_post_training_tail: int,
) -> None:
    """Jeden przebieg: dla każdego pliku decyzja AND (usuń wyłącznie poza unią reguł)."""
    ckpt_dir = Path(save_dir) / "checkpoints"
    if final_epoch < 1 or not ckpt_dir.is_dir():
        return
    mod = checkpoint_prune_post_training_mod
    tail = checkpoint_prune_post_training_tail
    if mod < 1 or tail < 1:
        raise ValueError("checkpoint_prune_post_training_mod i _tail muszą być >= 1")
    mod_epochs = _checkpoint_mod_epochs(final_epoch, mod)
    tail_epochs = _checkpoint_tail_epochs(final_epoch, tail)
    removed = 0
    kept = 0
    for path in sorted(ckpt_dir.glob("epoch_*.pt")):
        ep = _epoch_from_checkpoint_path(path)
        if ep is None:
            continue
        if not _checkpoint_may_delete(ep, mod_epochs, tail_epochs):
            kept += 1
            continue
        path.unlink(missing_ok=True)
        removed += 1
    if removed:
        log_print(
            f"Checkpointy po treningu: zostało {kept} plików "
            f"(unia: ep.1 + co {mod} + tail {tail} do ep.{final_epoch}) — usunięto {removed}"
        )


def best_eval_loss_from_save_dir(save_dir: Path | str | None, architecture: str) -> float | None:
    if save_dir is None:
        return None
    path = Path(save_dir) / f"{architecture}_best_eval.pt"
    if not path.is_file():
        return None
    payload = load_checkpoint_payload(path)
    loss = float(payload.get("eval_loss", float("inf")))
    return loss if math.isfinite(loss) else None


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRANSFORMER_ARCHITECTURES = frozenset(
    {"transformer_nanoGPT", "transformer_miniLLaMA", "transformer_GPT2small"}
)


def _default_num_layers(architecture: str) -> int:
    match architecture:
        case "rnnBaseLSTM" | "rnnAWDLSTM":
            return 4
        case "rnnGoldenLSTM":
            return 1
        case "transformer_nanoGPT":
            return 6
        case "transformer_miniLLaMA" | "transformer_GPT2small":
            return 12
        case _:
            return 3


def _default_num_heads(hidden_size: int, num_heads: int | None) -> int:
    if num_heads is not None:
        return num_heads
    return max(1, hidden_size // 64)


class LanguageModelNet(nn.Module):
    def __init__(
        self,
        architecture: str,
        embed_dim: int = 128,
        hidden_size: int = 256,
        dropout: float = 0.3,
        seq_len: int = 100,
        device: torch.device | str | None = None,
        save_dir: Path | str | None = None,
        *,
        num_layers: int | None = None,
        num_heads: int | None = None,
        ff_size: int | None = None,
        window_stride: int = 1,
        max_corpus_chars: int = 0,
        prepare_data: bool = True,
    ):
        super().__init__()

        self.architecture = architecture
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers if num_layers is not None else _default_num_layers(architecture)
        self.num_heads = _default_num_heads(hidden_size, num_heads)
        self.ff_size = ff_size
        self.window_stride = window_stride
        self.max_corpus_chars = max_corpus_chars
        self.model_name = architecture
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.metrics_logger: RunMetricsLogger | None = None
        self.dataset_summary: dict | None = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Datasets (trening) albo sam tokenizer (inferencja) #
        if prepare_data:
            self.dataset_prep()
        else:
            self.tokenizer_prep()

        # Compute ENV #
        self.device = resolve_device(device)

        # Network #
        self.init_network()

        self.to(self.device)

    def init_network(self) -> None:
        self.cost_function = nn.CrossEntropyLoss()
        kwargs = {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "num_layers": self.num_layers,
        }
        match self.architecture:
            case "rnnLSTM":
                self.rnn = rnnLSTM(**kwargs)
            case "rnnGRU":
                self.rnn = rnnGRU(**kwargs)
            case "rnnRNN":
                self.rnn = rnnRNN(**kwargs)
            case "rnnBaseLSTM":
                self.rnn = rnnBaseLSTM(**kwargs)
            case "rnnAWDLSTM":
                self.rnn = rnnAWDLSTM(**kwargs)
            case "rnnGoldenLSTM":
                self.rnn = rnnGoldenLSTM(**kwargs)
            case "transformer_nanoGPT":
                self.rnn = transformer_nanoGPT(
                    **kwargs,
                    num_heads=self.num_heads,
                    ff_size=self.ff_size,
                    seq_len=self.seq_len,
                )
            case "transformer_miniLLaMA":
                self.rnn = transformer_miniLLaMA(
                    **kwargs,
                    num_heads=self.num_heads,
                    ff_size=self.ff_size,
                    seq_len=self.seq_len,
                )
            case "transformer_GPT2small":
                self.rnn = transformer_GPT2small(
                    **kwargs,
                    num_heads=self.num_heads,
                    ff_size=self.ff_size,
                    seq_len=self.seq_len,
                )
            case _:
                raise ValueError(f"Nieznana architektura: {self.architecture!r}")

    def forward(self, x: torch.Tensor, hidden=None, *, pos_start: int = 0):
        if self.architecture in TRANSFORMER_ARCHITECTURES:
            return self.rnn(x, hidden, pos_start=pos_start)
        return self.rnn(x, hidden)

    def tokenizer_prep(self) -> None:
        """Wczytuje gotowy BPE — bez korpusu i datasetów (inferencja / load_checkpoint)."""
        self.tokenizer = load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size

    def dataset_prep(self) -> None:
        (
            self.train_ds,
            self.val_ds,
            self.test_ds,
            self.tokenizer,
            self.vocab_size,
            self.dataset_summary,
        ) = prepare_language_data(
            self.seq_len,
            window_stride=self.window_stride,
            max_corpus_chars=self.max_corpus_chars,
        )

    def _data_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        pin_memory = self.device.type == "cuda"
        num_workers = 2 if pin_memory else 0
        kwargs: dict = {"batch_size": batch_size, "shuffle": shuffle, "pin_memory": pin_memory}
        if num_workers > 0:
            kwargs["num_workers"] = num_workers
            kwargs["persistent_workers"] = True
        return DataLoader(dataset, **kwargs)

    def _clamp_eval_batch_size(self, batch_size: int) -> int:
        """Ogranicza batch walidacji, gdy szczyt pamięci (gł. logits [B,S,V]) przekracza VRAM."""
        bs = max(1, int(batch_size))
        env_cap = os.environ.get("LANGUAGE_RNN_EVAL_BATCH_CAP", "").strip()
        if env_cap:
            try:
                bs = min(bs, max(1, int(env_cap)))
            except ValueError:
                pass

        if self.architecture not in TRANSFORMER_ARCHITECTURES or self.device.type != "cuda":
            return bs

        # fp32 logits + margines na CE / fragmentację
        bytes_per_batch = max(1, 8 * self.seq_len * self.vocab_size)
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return bs

        if free <= 0:
            return bs

        budget = int(free * 0.45)
        max_by_mem = max(1, budget // bytes_per_batch)
        if max_by_mem < bs:
            log_print(
                f"Eval batch: {bs} → {max_by_mem} "
                f"(seq={self.seq_len}, vocab={self.vocab_size:,}, wolne VRAM ~{free / 2**30:.2f} GiB)"
            )
            bs = max_by_mem
        return bs

    def _evall_once(
        self,
        model: LanguageModelNet,
        batch_size: int,
        writer: SummaryWriter | None,
        epoch: int | None,
    ) -> tuple[float, float]:
        model.to(self.device)
        val_loader = self._data_loader(self.val_ds, batch_size, shuffle=False)
        log_print(f"Walidacja: {len(val_loader):,} batchy (batch_size={batch_size})")

        model.eval()
        running_loss = torch.zeros((), device=self.device)
        total_targets = 0
        n_batches = len(val_loader)
        batch_idx = 0
        with torch.no_grad():
            for x, y in val_loader:
                t_batch0 = time.perf_counter()
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                batch_n = x.size(0)
                hidden = self._batch_hidden(batch_n)
                logits, _ = model(x, hidden)
                loss = self.cost_function(logits, y.reshape(-1))
                n_targets = y.numel()
                running_loss += loss * n_targets
                total_targets += n_targets
                _sync_device(self.device)
                batch_idx += 1
                if batch_idx == 1:
                    _log_eta_from_first_batch(
                        phase="walidacja",
                        batch_seconds=time.perf_counter() - t_batch0,
                        n_batches=n_batches,
                    )

        if total_targets:
            avg_loss = (running_loss / total_targets).item()
            perplexity = math.exp(min(avg_loss, 20.0))
        else:
            avg_loss = 0.0
            perplexity = 1.0

        log_print(f"Val: loss={avg_loss:.4f}, eval_perplexity={perplexity:.2f}")
        if writer is not None and epoch is not None:
            writer.add_scalar("eval/loss", avg_loss, epoch)
            writer.add_scalar("eval/perplexity", perplexity, epoch)
        return avg_loss, perplexity

    def _log_custom_scalars_layout(self, writer: SummaryWriter) -> None:
        writer.add_custom_scalars(
            {
                "train_and_eval": {
                    "loss": ["Multiline", ["train/loss", "eval/loss"]],
                    "perplexity": ["Multiline", ["train/perplexity", "eval/perplexity"]],
                }
            }
        )

    def _training_hparams(self) -> dict:
        hp = {
            "architecture": self.architecture,
            "model_name": self.model_name,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ff_size": self.ff_size,
            "window_stride": self.window_stride,
            "max_corpus_chars": self.max_corpus_chars,
            "vocab_size": self.vocab_size,
            "tokenizer_dir": str(tokenizer_dir()),
        }
        if self.architecture in TRANSFORMER_ARCHITECTURES:
            hp["ff_size"] = getattr(self.rnn, "ff_size", self.ff_size)
        return hp

    def _model_report(self) -> dict:
        desc_fn = getattr(type(self.rnn), "description", None)
        description = desc_fn() if callable(desc_fn) else ""
        return build_model_report(
            self,
            architecture=self.architecture,
            description=description,
            configured_dropout=self.dropout,
        )

    def _ensure_save_dir(self) -> Path:
        if self.save_dir is None:
            self.save_dir = resolve_save_dir(self.architecture)
            log_print(f"Zapis modeli: {self.save_dir}")
        return self.save_dir

    def _checkpoint_payload(
        self,
        model: LanguageModelNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "best_eval_loss": best_eval_loss,
            "hparams": self._training_hparams(),
        }

    def save_model_bundle(
        self,
        model: LanguageModelNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
        stem: str,
        *,
        export_onnx: bool = True,
    ) -> Path:
        model.eval()
        root = self._ensure_save_dir()
        pt_path = root / f"{self.model_name}_{stem}.pt"
        torch.save(
            self._checkpoint_payload(model, optimizer, epoch, eval_loss, eval_perplexity, best_eval_loss),
            pt_path,
        )
        do_onnx = export_onnx and self.architecture == "rnnLSTM"
        if do_onnx:
            self.export_to_onnx(model, root / f"{self.model_name}_{stem}.onnx")
        elif export_onnx:
            log_print(f"ONNX pominięty dla {self.architecture}")
        log_print(f"Zapis {stem}: {pt_path.name}" + (f", {self.model_name}_{stem}.onnx" if do_onnx else ""))
        return pt_path

    def save_checkpoint(
        self,
        model: LanguageModelNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> Path:
        root = self._ensure_save_dir()
        path = root / "checkpoints" / f"epoch_{epoch:04d}.pt"
        torch.save(
            self._checkpoint_payload(model, optimizer, epoch, eval_loss, eval_perplexity, best_eval_loss),
            path,
        )
        log_print(f"Checkpoint → {path}")
        return path

    def _payload_from_state(
        self,
        model_state: dict,
        optimizer_state: dict,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "best_eval_loss": best_eval_loss,
            "hparams": self._training_hparams(),
        }

    def save_model_bundle_from_state(
        self,
        model_state: dict,
        optimizer_state: dict,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
        stem: str,
        *,
        export_onnx: bool = True,
        checkpoints_subdir: bool = False,
    ) -> Path:
        payload = self._payload_from_state(
            model_state, optimizer_state, epoch, eval_loss, eval_perplexity, best_eval_loss
        )
        root = self._ensure_save_dir()
        if checkpoints_subdir:
            path = root / "checkpoints" / f"{stem}.pt"
        else:
            path = root / f"{self.model_name}_{stem}.pt"
        torch.save(payload, path)
        if not checkpoints_subdir:
            log_print(f"Zapis {stem}: {path.name}")
        else:
            log_print(f"Checkpoint → {path}")
        return path

    def save_final(
        self,
        model: LanguageModelNet,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        eval_loss: float,
        eval_perplexity: float,
        best_eval_loss: float,
    ) -> None:
        self.save_model_bundle(
            model,
            optimizer,
            epoch,
            eval_loss,
            eval_perplexity,
            best_eval_loss,
            stem="final",
            export_onnx=True,
        )

    def export_to_onnx(self, model: LanguageModelNet, onnx_path: Path) -> None:
        """Jeden krok LSTM: token (B,1), h0, c0 → logity (B,V), h1, c1 (do inferencji)."""

        class RnnStep(nn.Module):
            def __init__(self, core: rnnLSTM) -> None:
                super().__init__()
                self.core = core

            def forward(
                self,
                token_ids: torch.Tensor,
                h0: torch.Tensor,
                c0: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                logits, (h1, c1) = self.core(token_ids, (h0, c0))
                return logits, h1, c1

        core = copy.deepcopy(model.rnn).cpu().eval()
        wrapper = RnnStep(core)
        batch = 1
        dummy_x = torch.zeros(batch, 1, dtype=torch.long)
        h0, c0 = core.init_hidden(batch)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            (dummy_x, h0, c0),
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input", "h0", "c0"],
            output_names=["logits", "h1", "c1"],
            dynamic_axes={
                "input": {0: "batch"},
                "h0": {1: "batch"},
                "c0": {1: "batch"},
                "logits": {0: "batch"},
                "h1": {1: "batch"},
                "c1": {1: "batch"},
            },
        )

    def _batch_hidden(self, batch_n: int):
        if self.architecture in TRANSFORMER_ARCHITECTURES:
            return None
        match self.architecture:
            case "rnnLSTM" | "rnnBaseLSTM" | "rnnAWDLSTM":
                return self.rnn.init_hidden(batch_n)
            case "rnnGoldenLSTM":
                return self.rnn.init_hidden(batch_n)
            case "rnnGRU" | "rnnRNN":
                return self.rnn.init_hidden(batch_n)

    def _should_use_lightning(self) -> bool:
        raw = os.environ.get("LANGUAGE_RNN_LIGHTNING", "auto").strip().lower()
        if raw in ("0", "false", "no"):
            return False
        if raw in ("1", "true", "yes"):
            return True
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2

    def _trainn_lightning(
        self,
        model: LanguageModelNet,
        *,
        epochs: int,
        batch_size: int,
        writer: SummaryWriter | None,
        max_grad_norm: float,
        resume_payload: dict | None,
        checkpoint_every: int,
        eval_every: int,
        time_budget_min: float,
        precision: str | None,
        learning_rate: float,
        lightning_devices: int,
        tensorboard_dir: str | None,
        strategy: str,
        val_checks_per_epoch: int,
        log_every_n_steps: int,
    ) -> tuple[torch.optim.Optimizer, float]:
        from lit.dist_launch import LmFitSpec, run_lm_fit
        from lit.lm_module import default_lightning_precision

        os.environ.setdefault("NCCL_P2P_DISABLE", "1")

        start_epoch = 0
        best_eval_loss = float("inf")
        if resume_payload is not None:
            start_epoch = int(resume_payload["epoch"])
            best_eval_loss = float(
                resume_payload.get("best_eval_loss", resume_payload.get("eval_loss", float("inf")))
            )
            log_print(
                f"Wczytano checkpoint (po {start_epoch} epokach), "
                f"val_ppl={resume_payload.get('eval_perplexity', float('nan')):.2f}, "
                f"best_eval_loss={best_eval_loss:.4f}"
            )

        if start_epoch >= epochs:
            log_print(
                f"Checkpoint ma epokę {start_epoch} ≥ cel {epochs} — pomijam trening Lightning."
            )
            opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            if resume_payload is not None:
                model.load_state_dict(resume_payload["model_state_dict"])
                opt.load_state_dict(resume_payload["optimizer_state_dict"])
            return opt, best_eval_loss

        if model.metrics_logger is None:
            raise RuntimeError("metrics_logger wymagany przed treningiem Lightning")
        metrics_dir = str(model.metrics_logger.save_dir)
        prec = precision or default_lightning_precision(model.architecture)
        log_print(
            f"Lightning fit: devices={lightning_devices}, strategy={strategy}, "
            f"batch/GPU={batch_size}, precision={prec}, budżet={time_budget_min:.0f} min"
        )
        model_kwargs = {
            "embed_dim": model.embed_dim,
            "hidden_size": model.hidden_size,
            "dropout": model.dropout,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "ff_size": model.ff_size,
        }
        spec = LmFitSpec(
            save_dir=metrics_dir,
            architecture=model.architecture,
            model_kwargs=model_kwargs,
            seq_len=model.seq_len,
            window_stride=model.window_stride,
            max_corpus_chars=model.max_corpus_chars,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            start_epoch=start_epoch,
            devices=lightning_devices if torch.cuda.is_available() else 1,
            strategy=strategy,
            precision=prec,
            checkpoint_every=checkpoint_every,
            eval_every=eval_every,
            time_budget_min=float(time_budget_min),
            max_grad_norm=max_grad_norm,
            tensorboard_dir=tensorboard_dir,
            resume_payload=resume_payload,
            val_checks_per_epoch=val_checks_per_epoch,
            log_every_n_steps=log_every_n_steps,
            completed_epochs_at_resume=start_epoch,
            resume_best_eval=best_eval_loss,
        )
        result = run_lm_fit(spec, devices=spec.devices)
        if result is None:
            raise RuntimeError("Trening Lightning zakończony bez wyniku (_fit_result.pt)")
        model.load_state_dict(result["model_state_dict"])
        model.to(self.device)
        if model.tokenizer is None:
            model.tokenizer_prep()
        best_eval_loss = float(result.get("best_eval_loss", best_eval_loss))
        opt_state = result.get("optimizer_state_dict") or {}
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if opt_state:
            try:
                opt.load_state_dict(opt_state)
            except Exception:
                pass
        metrics_path = Path(metrics_dir) / "metrics.csv"
        if not metrics_path.is_file():
            raise RuntimeError(f"Brak {metrics_path.name} po treningu — logowanie metryk nie działało")
        if not math.isfinite(best_eval_loss) or best_eval_loss == float("inf"):
            from_disk = best_eval_loss_from_save_dir(model.save_dir, model.architecture)
            if from_disk is not None:
                best_eval_loss = from_disk
        return opt, best_eval_loss

    def trainn(
        self,
        model: LanguageModelNet,
        epochs: int = 30,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        max_grad_norm: float = 1.0,
        resume_payload: dict | None = None,
        checkpoint_every: int = 5,
        eval_every: int = 5,
        *,
        time_budget_min: float | None = None,
        precision: str | None = None,
        learning_rate: float = 3e-4,
        lightning_devices: int = 2,
        tensorboard_dir: str | None = None,
        strategy: str = "ddp",
        val_checks_per_epoch: int = 4,
        log_every_n_steps: int = 50,
    ) -> tuple[torch.optim.Optimizer, float]:
        if time_budget_min is not None and self._should_use_lightning():
            return self._trainn_lightning(
                model,
                epochs=epochs,
                batch_size=batch_size,
                writer=writer,
                max_grad_norm=max_grad_norm,
                resume_payload=resume_payload,
                checkpoint_every=checkpoint_every,
                eval_every=eval_every,
                time_budget_min=time_budget_min,
                precision=precision,
                learning_rate=learning_rate,
                lightning_devices=lightning_devices,
                tensorboard_dir=tensorboard_dir,
                strategy=strategy,
                val_checks_per_epoch=val_checks_per_epoch,
                log_every_n_steps=log_every_n_steps,
            )

        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        start_epoch = 0
        best_eval_loss = float("inf")
        if resume_payload is not None:
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            start_epoch = int(resume_payload["epoch"])
            best_eval_loss = float(resume_payload.get("best_eval_loss", resume_payload.get("eval_loss", float("inf"))))
            log_print(
                f"Wczytano wagi + optimizer (po {start_epoch} epokach), "
                f"val_ppl={resume_payload.get('eval_perplexity', float('nan')):.2f}, "
                f"best_eval_loss={best_eval_loss:.4f}"
            )

        if start_epoch >= epochs:
            log_print(
                f"Checkpoint ma epokę {start_epoch} ≥ cel {epochs} — pomijam trening. "
                f"Ustaw więcej epok (np. epochs=60 w train.main)."
            )
            return optimizer, best_eval_loss

        train_loader = self._data_loader(self.train_ds, batch_size, shuffle=True)

        for epoch in range(start_epoch, epochs):
            log_print(
                f"Epoka {epoch + 1}/{epochs}: {len(train_loader):,} batchy (batch_size={batch_size})"
            )
            t_train0 = time.perf_counter()
            model.train()
            running_loss = torch.zeros((), device=self.device)
            total_targets = 0
            n_batches = len(train_loader)
            batch_idx = 0

            for x, y in train_loader:
                t_batch0 = time.perf_counter()
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                batch_n = x.size(0)

                hidden = self._batch_hidden(batch_n)
                optimizer.zero_grad()
                logits, _hidden = model(x, hidden)
                loss = self.cost_function(logits, y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                n_targets = y.numel()
                running_loss += loss.detach() * n_targets
                total_targets += n_targets
                _sync_device(self.device)
                batch_idx += 1
                if batch_idx == 1:
                    _log_eta_from_first_batch(
                        phase="trening epoki",
                        batch_seconds=time.perf_counter() - t_batch0,
                        n_batches=n_batches,
                    )

            avg_loss = (running_loss / total_targets).item()
            train_seconds = time.perf_counter() - t_train0
            perplexity = math.exp(min(avg_loss, 20.0))
            log_print(
                f"Epoka {epoch + 1}: loss={avg_loss:.4f} train_perplexity={perplexity:.2f} "
                f"({train_seconds:.1f}s train)"
            )
            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/perplexity", perplexity, step)

            t_eval0 = time.perf_counter()
            eval_loss, eval_ppl = self.evall(model, batch_size=batch_size, writer=writer, epoch=epoch + 1)
            eval_seconds = time.perf_counter() - t_eval0

            if self.metrics_logger is not None:
                self.metrics_logger.log_epoch(
                    epoch + 1,
                    avg_loss,
                    perplexity,
                    eval_loss,
                    eval_ppl,
                    train_seconds,
                    eval_seconds,
                )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_model_bundle(
                    model,
                    optimizer,
                    epoch + 1,
                    eval_loss,
                    eval_ppl,
                    best_eval_loss,
                    stem="best_eval",
                    export_onnx=True,
                )
                log_print(f"★ Nowy best eval: loss={eval_loss:.4f}, ppl={eval_ppl:.2f}")

            if epoch % checkpoint_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(model, optimizer, epoch + 1, eval_loss, eval_ppl, best_eval_loss)

            if epoch % eval_every == 0 or epoch == epochs - 1:
                sample, _stats = self.generate(model, seed="W", length=200, temperature=0.8)
                log_print(f"Generacja (200 tok.): {sample[:200]!r}")

        return optimizer, best_eval_loss

    def evall(
        self,
        model: LanguageModelNet,
        batch_size: int = 128,
        writer: SummaryWriter | None = None,
        epoch: int | None = None,
    ) -> tuple[float, float]:
        bs = self._clamp_eval_batch_size(batch_size)
        while True:
            try:
                return self._evall_once(model, bs, writer, epoch)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or bs <= 1:
                    raise
                torch.cuda.empty_cache()
                new_bs = max(1, bs // 2)
                log_print(
                    f"CUDA OOM przy walidacji (batch_size={bs}) — ponawiam z batch_size={new_bs}"
                )
                bs = new_bs

    def eval_test(
        self,
        model: LanguageModelNet,
        batch_size: int = 128,
    ) -> tuple[float, float]:
        if self.test_ds is None:
            log_print("Brak test_ds — pomijam ewaluację test")
            return float("nan"), float("nan")
        val_backup = self.val_ds
        self.val_ds = self.test_ds
        try:
            loss, ppl = self.evall(model, batch_size=batch_size, writer=None, epoch=None)
            log_print(f"Test: loss={loss:.4f}, perplexity={ppl:.2f}")
            return loss, ppl
        finally:
            self.val_ds = val_backup

    def _perplexity_from_logits(
        self,
        logits_list: list[torch.Tensor],
        targets: list[int],
        temperature: float = 1.0,
    ) -> float:
        if not targets or len(logits_list) < len(targets):
            return float("nan")
        running = 0.0
        for logit, target in zip(logits_list[: len(targets)], targets):
            scaled = logit.squeeze(0) / max(temperature, 1e-6)
            loss = self.cost_function(
                scaled.unsqueeze(0),
                torch.tensor([target], device=scaled.device, dtype=torch.long),
            )
            running += loss.item()
        avg = running / len(targets)
        return math.exp(min(avg, 20.0))

    def generate(
        self,
        model: LanguageModelNet,
        seed: str = "W",
        length: int = 400,
        temperature: float = 1.0,
        *,
        measure: bool = True,
    ) -> tuple[str, dict]:
        model.eval()
        model.to(self.device)
        tok = self.tokenizer
        seed_ids = tok.encode(seed or "", add_special_tokens=False)
        if not seed_ids:
            seed_ids = [tok.unk_token_id or 0]

        token_ids: list[int] = []
        prefill_logits: list[torch.Tensor] = []
        gen_logits: list[torch.Tensor] = []
        prefill_seconds = 0.0
        generate_seconds = 0.0

        with torch.no_grad():
            if self.architecture in TRANSFORMER_ARCHITECTURES:
                pos_start = 0
                ids_tensor = torch.tensor([seed_ids], device=self.device, dtype=torch.long)
                if measure:
                    _sync_device(self.device)
                    t_prefill = time.perf_counter()
                window = ids_tensor[:, -self.seq_len :]
                pos_start = max(0, len(seed_ids) - window.size(1))
                logits_flat, cache = model(window, None, pos_start=pos_start)
                logits = logits_flat.view(window.size(0), window.size(1), -1)
                prefill_logits.extend(logits[:, i, :] for i in range(logits.size(1)))
                token_ids.extend(seed_ids)
                if measure:
                    _sync_device(self.device)
                    prefill_seconds = time.perf_counter() - t_prefill
                    _sync_device(self.device)
                    t_gen = time.perf_counter()
                logits = logits[:, -1, :]
                for _ in range(length):
                    gen_logits.append(logits)
                    scaled = logits.squeeze(0) / max(temperature, 1e-6)
                    last_idx = int(torch.multinomial(torch.softmax(scaled, dim=-1), 1).item())
                    token_ids.append(last_idx)
                    total_len = len(token_ids)
                    if total_len >= self.seq_len:
                        restart = max(self.seq_len // 2, 1)
                        chunk = torch.tensor([token_ids[-restart:]], device=self.device, dtype=torch.long)
                        pos_start = total_len - restart
                        logits_flat, cache = model(chunk, None, pos_start=pos_start)
                        logits = logits_flat.view(1, chunk.size(1), -1)[:, -1, :]
                    else:
                        step = torch.tensor([[last_idx]], device=self.device, dtype=torch.long)
                        pos_start = total_len - 1
                        logits_flat, cache = model(step, cache, pos_start=pos_start)
                        logits = logits_flat.view(1, 1, -1)[:, -1, :]
                if measure:
                    _sync_device(self.device)
                    generate_seconds = time.perf_counter() - t_gen
            else:
                match self.architecture:
                    case "rnnGRU" | "rnnRNN" | "rnnLSTM" | "rnnBaseLSTM" | "rnnAWDLSTM" | "rnnGoldenLSTM":
                        hidden = model.rnn.init_hidden(1)
                        if measure:
                            _sync_device(self.device)
                            t_prefill = time.perf_counter()
                        # Prefill #
                        for idx in seed_ids:
                            x = torch.tensor([[idx]], device=self.device, dtype=torch.long)
                            logits, hidden = model(x, hidden)
                            prefill_logits.append(logits)
                            token_ids.append(idx)
                        if measure:
                            _sync_device(self.device)
                            prefill_seconds = time.perf_counter() - t_prefill
                            _sync_device(self.device)
                            t_gen = time.perf_counter()
                        # Generate #
                        for _ in range(length):
                            gen_logits.append(logits)
                            scaled = logits.squeeze(0) / max(temperature, 1e-6)
                            probs = torch.softmax(scaled, dim=-1)
                            last_idx = int(torch.multinomial(probs, num_samples=1).item())
                            token_ids.append(last_idx)
                            x = torch.tensor([[last_idx]], device=self.device, dtype=torch.long)
                            logits, hidden = model(x, hidden)
                        if measure:
                            _sync_device(self.device)
                            generate_seconds = time.perf_counter() - t_gen
                    case _:
                        raise ValueError(f"Generacja nieobsługiwana: {self.architecture!r}")

        prompt_targets = seed_ids[1:]
        prompt_logits = prefill_logits[:-1] if len(prefill_logits) > 1 else []
        generated_targets = token_ids[len(seed_ids) :]

        stats = {
            "prompt_tokens": len(seed_ids),
            "generated_tokens": length,
            "prefill_seconds": prefill_seconds,
            "generate_seconds": generate_seconds,
            "inference_seconds": prefill_seconds + generate_seconds,
            "tokens_per_second": (
                length / generate_seconds if measure and generate_seconds > 0 else float("nan")
            ),
            "prompt_perplexity": self._perplexity_from_logits(prompt_logits, prompt_targets, 1.0),
            "generate_perplexity": self._perplexity_from_logits(gen_logits, generated_targets, 1.0),
            "sampling_perplexity": self._perplexity_from_logits(
                gen_logits, generated_targets, temperature
            ),
            "temperature": temperature,
        }
        text = tok.decode(token_ids, skip_special_tokens=True)
        return text, stats


def _net_from_hparams(
    hp: dict,
    device: torch.device,
    save_dir: Path,
    *,
    prepare_data: bool = True,
) -> LanguageModelNet:
    architecture = hp.get("architecture") or hp.get("model_name")
    if not architecture:
        raise ValueError("checkpoint hparams bez architecture / model_name")
    net = LanguageModelNet(
        architecture=architecture,
        embed_dim=hp.get("embed_dim", 128),
        hidden_size=hp.get("hidden_size", 256),
        dropout=hp.get("dropout", 0.3),
        seq_len=hp.get("seq_len", 100),
        device=device,
        save_dir=save_dir,
        num_layers=hp.get("num_layers"),
        num_heads=hp.get("num_heads"),
        ff_size=hp.get("ff_size"),
        window_stride=hp.get("window_stride", 1),
        max_corpus_chars=int(hp.get("max_corpus_chars", 0)),
        prepare_data=prepare_data,
    )
    net.model_name = hp.get("model_name", architecture)
    net.save_dir = save_dir
    return net


def load_checkpoint(
    path: Path | str,
    device: torch.device | str | None = None,
    *,
    save_dir: Path | str | None = None,
) -> tuple[dict, LanguageModelNet]:
    """Wczytuje checkpoint .pt — tokenizer z text_and_tokenizer/data/tokenizer."""
    path = Path(path)
    payload = load_checkpoint_payload(path)
    hp = payload["hparams"]
    dev = resolve_device(device)
    resolved_save = Path(save_dir) if save_dir is not None else checkpoint_run_dir(path)
    net = _net_from_hparams(hp, dev, resolved_save, prepare_data=False)
    if net.vocab_size != hp.get("vocab_size", net.vocab_size):
        raise RuntimeError(
            f"vocab checkpoint ({hp.get('vocab_size')}) ≠ tokenizer ({net.vocab_size}) — "
            f"retrening tokenizera w {tokenizer_dir()}"
        )
    net.load_state_dict(payload["model_state_dict"])
    net.to(dev)
    return payload, net


def build_net_for_run(
    device: torch.device,
    save_dir: Path,
    checkpoint_path: Path | None,
    architecture: str,
    **model_kwargs,
) -> tuple[LanguageModelNet, dict | None]:
    if checkpoint_path is None:
        net = LanguageModelNet(architecture=architecture, device=device, save_dir=save_dir, **model_kwargs)
        net.save_dir = save_dir
        return net, None

    payload = load_checkpoint_payload(checkpoint_path)
    log_print(f"Wznawiam z checkpointu: {checkpoint_path} (epoka {payload['epoch']})")
    net = _net_from_hparams(payload["hparams"], device, save_dir)
    return net, payload


def main(
    tensorboard_dir: str | None = None,
    *,
    architecture: str,
    epochs: int,
    checkpoint_every: int,
    eval_every: int,
    run_tag: str | None = None,
    resume: str | None = None,
    embed_dim: int,
    hidden_size: int,
    num_layers: int | None,
    num_heads: int | None,
    ff_size: int | None,
    dropout: float,
    seq_len: int,
    window_stride: int,
    batch_size: int | None,
    time_budget_min: float | None,
    precision: str | None,
    lightning_devices: int,
    checkpoint_prune_post_training_mod: int,
    checkpoint_prune_post_training_tail: int,
    max_corpus_chars: int,
    strategy: str,
    ddp_min_per_gpu_batch: int,
    batch_probe_max: int,
    batch_probe_safety: float,
    batch_probe_refresh: bool,
    baseline_batch_size: int,
    baseline_learning_rate: float,
    lr_scaling: str,
    learning_rate_max: float,
    val_checks_per_epoch: int,
    log_every_n_steps: int,
) -> None:
    if tensorboard_dir is None:
        log_root = os.environ.get("TENSORBOARD_LOGDIR", "/workspace/runs")
        tensorboard_dir = tensorboard_run_dir(log_root, datetime.now())

    device = resolve_device()
    log_print(f"Architektura: {architecture}")
    log_print(f"Urządzenie treningu: {device}")
    log_print(f"TensorBoard run: {tensorboard_dir}")
    if max_corpus_chars < 1:
        log_print("Korpus train: pełny split HF (max_corpus_chars=0)")
    else:
        log_print(f"Korpus train: limit {max_corpus_chars:,} znaków (max_corpus_chars)")

    resume_spec = resolve_resume_from_env_or_arg(resume)
    if resume_spec == "latest":
        try:
            find_latest_run_dir(architecture)
        except FileNotFoundError:
            log_print(f"Brak wcześniejszego runu dla {architecture} — start od zera")
            resume_spec = None
    save_env = os.environ.get("LANGUAGE_RNN_SAVE_DIR", "").strip()
    resumed_from: dict[str, str] | None = None
    ckpt_path: Path | None = None
    train_prior_seconds = 0.0

    if resume_spec:
        source_run = resolve_source_run_dir(architecture, resume_spec)
        log_print(f"Wznawiam z runu: {source_run}")
        if save_env:
            save_dir = Path(save_env).resolve()
            if source_run != save_dir:
                raise ValueError(
                    "LANGUAGE_RNN_SAVE_DIR musi wskazywać ten sam katalog co resume "
                    f"({source_run} ≠ {save_dir})"
                )
            save_dir, train_prior_seconds, ckpt_path = resume_run_relocate(
                source_run, architecture, run_tag=run_tag, relocate=False
            )
        else:
            save_dir, train_prior_seconds, ckpt_path = resume_run_relocate(
                source_run, architecture, run_tag=run_tag
            )
        resumed_from = {
            "run_dir": str(source_run),
            "save_dir": str(save_dir),
            "checkpoint": str(ckpt_path),
        }
    elif save_env:
        save_dir = Path(save_env)
    else:
        save_dir = resolve_save_dir(architecture, run_tag=run_tag)

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    log_print(f"Katalog zapisu: {save_dir}")
    if ckpt_path:
        log_print(f"Checkpoint wejściowy: {ckpt_path}")

    model_kwargs = {
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ff_size": ff_size,
        "window_stride": window_stride,
        "max_corpus_chars": max_corpus_chars,
    }
    net, resume_payload = build_net_for_run(device, save_dir, ckpt_path, architecture, **model_kwargs)

    metrics_logger = RunMetricsLogger(save_dir)
    metrics_logger.write_meta(
        architecture=architecture,
        run_id=save_dir.name,
        tensorboard_dir=tensorboard_dir,
        hparams=net._training_hparams(),
        resumed_from=resumed_from,
    )
    report = net._model_report()
    metrics_logger.write_model_report(report, repr(net.rnn), dataset=net.dataset_summary)
    log_print(
        f"Model: {report['params']:,} param, {report['model_bytes']:,} B, "
        f"tying={report['weight_tying']}, dtype={report['dtype']}"
    )
    net.metrics_logger = metrics_logger

    per_gpu = int(batch_size) if batch_size else 8
    learning_rate = baseline_learning_rate
    strategy_use = strategy if strategy != "auto" else "ddp"
    meta_fit = metrics_logger.load_meta().get("fit")
    fit_plan = None
    if resume_payload and meta_fit:
        per_gpu = int(meta_fit["per_gpu_batch"])
        learning_rate = float(meta_fit["learning_rate"])
        strategy_use = str(meta_fit["strategy"])
        log_print(f"Resume: batch={per_gpu}/GPU, LR={learning_rate}, strategy={strategy_use}")
    else:
        fit_plan = choose_batch_and_strategy(
            net,
            architecture=architecture,
            task="lm",
            seq_len=seq_len,
            vocab_size=net.vocab_size,
            precision=precision or "bf16-mixed",
            strategy=strategy,
            ddp_min_per_gpu_batch=ddp_min_per_gpu_batch,
            batch_size=batch_size,
            batch_probe_max=batch_probe_max,
            batch_probe_safety=batch_probe_safety,
            batch_probe_refresh=batch_probe_refresh,
            lightning_devices=lightning_devices,
            module_tree=repr(net.rnn),
        )
        learning_rate = scaled_learning_rate(
            baseline_batch_size=baseline_batch_size,
            baseline_learning_rate=baseline_learning_rate,
            global_batch=fit_plan.global_batch,
            lr_scaling=lr_scaling,
            learning_rate_max=learning_rate_max,
        )
        strategy_use = fit_plan.strategy
        per_gpu = fit_plan.per_gpu_batch
        log_print(
            f"Fit: strategy={strategy_use} ({fit_plan.reason}), "
            f"batch={per_gpu}/GPU (global {fit_plan.global_batch}), LR={learning_rate:.2e}, "
            f"źródło batch={fit_plan.batch_source}"
        )
        metrics_logger.write_fit(
            {
                "strategy": strategy_use,
                "reason": fit_plan.reason,
                "per_gpu_batch": per_gpu,
                "global_batch": fit_plan.global_batch,
                "learning_rate": learning_rate,
                "batch_source": fit_plan.batch_source,
                "fingerprint": fit_plan.fingerprint_short,
            }
        )
        metrics_logger.append_fit_summary(fit_plan, learning_rate)

    writer = SummaryWriter(tensorboard_dir)
    if writer is not None:
        net._log_custom_scalars_layout(writer)
    ckpt_every = int(os.environ.get("LANGUAGE_RNN_CHECKPOINT_EVERY", str(checkpoint_every)))
    sample_every = int(os.environ.get("LANGUAGE_RNN_EVAL_EVERY", str(eval_every)))
    start_epoch = int(resume_payload["epoch"]) if resume_payload else 0
    try:
        t_run0 = time.perf_counter()
        env_budget = os.environ.get("LANGUAGE_RNN_TIME_BUDGET_MIN", "").strip()
        budget_min = time_budget_min
        if budget_min is None and env_budget:
            budget_min = float(env_budget)

        optimizer, best_eval_loss = net.trainn(
            net,
            epochs=epochs,
            batch_size=per_gpu,
            writer=writer,
            resume_payload=resume_payload,
            checkpoint_every=ckpt_every,
            eval_every=sample_every,
            time_budget_min=budget_min,
            precision=precision,
            learning_rate=learning_rate,
            lightning_devices=lightning_devices,
            tensorboard_dir=tensorboard_dir,
            strategy=strategy_use,
            val_checks_per_epoch=val_checks_per_epoch,
            log_every_n_steps=log_every_n_steps,
        )
        final_ep = _max_epoch_in_checkpoints(save_dir / "checkpoints")
        if final_ep > 0:
            compact_epoch_checkpoints(
                save_dir,
                final_ep,
                checkpoint_prune_post_training_mod=checkpoint_prune_post_training_mod,
                checkpoint_prune_post_training_tail=checkpoint_prune_post_training_tail,
            )
        best_path = save_dir / f"{architecture}_best_eval.pt"
        if best_path.is_file():
            payload, _ = load_checkpoint(best_path, device=device, save_dir=save_dir)
            net.load_state_dict(payload["model_state_dict"])
        eval_loss, eval_ppl = net.evall(net, batch_size=per_gpu, writer=writer, epoch=epochs)
        test_loss, test_ppl = net.eval_test(net, batch_size=per_gpu)
        metrics_logger.write_test_metrics(
            {"test_loss": test_loss, "test_perplexity": test_ppl, "split": "wikitext_test"}
        )
        net.save_final(net, optimizer, epochs, eval_loss, eval_ppl, best_eval_loss)
        train_segment_seconds = time.perf_counter() - t_run0
        train_total_seconds = train_prior_seconds + train_segment_seconds
        metrics_logger.write_timing(
            train_total_seconds=train_total_seconds,
            epochs=epochs,
            start_epoch=start_epoch,
            train_prior_seconds=train_prior_seconds,
            train_segment_seconds=train_segment_seconds,
        )
        if not (save_dir / METRICS_FILENAME).is_file():
            raise RuntimeError(f"Brak {METRICS_FILENAME} w {save_dir}")
        plot_training_run(save_dir)
        write_training_report(save_dir)
        if train_prior_seconds > 0:
            log_print(
                f"Czas treningu: {train_total_seconds:.1f}s "
                f"(wcześniej {train_prior_seconds:.1f}s + ta sesja {train_segment_seconds:.1f}s)"
            )
        else:
            log_print(f"Czas treningu: {train_total_seconds:.1f}s")
        log_print(
            f"Best eval loss w runie: {best_eval_loss:.4f} (pliki {architecture}_best_eval.*)"
        )
        final, _stats = net.generate(net, seed="W polskim", length=500, temperature=0.85)
        log_print("--- Ostateczna generacja (wagi final, nie best) ---")
        log_print(final)
    finally:
        writer.close()
