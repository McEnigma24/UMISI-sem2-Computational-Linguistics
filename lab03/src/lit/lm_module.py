"""LightningModule dla LanguageModelNet."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import torch

from lit.optim import param_groups_no_decay, warmup_cosine_scheduler

if TYPE_CHECKING:
    from train import LanguageModelNet


class LitLanguageModel(pl.LightningModule):
    def __init__(
        self,
        net: LanguageModelNet,
        *,
        learning_rate: float,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 500,
        total_steps: int,
        weight_decay: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.cost_function = net.cost_function
        self._tokens_per_step = 0
        self.save_hyperparameters(ignore=["net"])

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate") and getattr(self.net, "tokenizer", None) is None:
            self.net.tokenizer_prep()

    def forward(self, x: torch.Tensor, hidden=None, *, pos_start: int = 0):
        return self.net(x, hidden, pos_start=pos_start)

    def _batch_hidden(self, batch_n: int):
        return self.net._batch_hidden(batch_n)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        hidden = self._batch_hidden(x.size(0))
        logits, _ = self.net(x, hidden)
        loss = self.cost_function(logits, y.reshape(-1))
        self._tokens_per_step = y.numel()
        lr = self.trainer.optimizers[0].param_groups[0]["lr"] if self.trainer.optimizers else 0.0
        self.log("train/loss_step", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/lr", lr, on_step=True, on_epoch=False, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        hidden = self._batch_hidden(x.size(0))
        logits, _ = self.net(x, hidden)
        loss = self.cost_function(logits, y.reshape(-1))
        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            param_groups_no_decay(self.net, weight_decay=self.weight_decay),
            lr=self.learning_rate,
        )
        scheduler = warmup_cosine_scheduler(
            optimizer, warmup_steps=self.warmup_steps, total_steps=self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

TRANSFORMER_ARCHITECTURES = frozenset(
    {"transformer_nanoGPT", "transformer_miniLLaMA", "transformer_GPT2small"}
)


def default_lightning_precision(architecture: str) -> str:
    if architecture in TRANSFORMER_ARCHITECTURES:
        return "bf16-mixed"
    return "16-mixed"