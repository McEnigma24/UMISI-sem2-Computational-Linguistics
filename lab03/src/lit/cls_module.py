"""LightningModule dla klasyfikacji Tweet Eval."""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from finetuning.cls_data import NUM_LABELS
from lit.optim import param_groups_no_decay, warmup_cosine_scheduler


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float,
        max_grad_norm: float,
        warmup_steps: int,
        total_steps: int,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.train_acc = MulticlassAccuracy(num_classes=NUM_LABELS)
        self.val_acc = MulticlassAccuracy(num_classes=NUM_LABELS)
        self.val_f1_macro = MulticlassF1Score(num_classes=NUM_LABELS, average="macro")
        self.val_f1_weighted = MulticlassF1Score(num_classes=NUM_LABELS, average="weighted")
        self.save_hyperparameters(ignore=["model"])

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def _step(self, batch, stage: str) -> torch.Tensor:
        out = self.model(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = out["loss"]
        preds = out["logits"].argmax(dim=-1)
        if stage == "train":
            self.train_acc.update(preds, batch["labels"])
            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, sync_dist=True)
        else:
            self.val_acc.update(preds, batch["labels"])
            self.val_f1_macro.update(preds, batch["labels"])
            self.val_f1_weighted.update(preds, batch["labels"])
            self.log("val/loss", loss, on_epoch=True, sync_dist=True, prog_bar=False)
            self.log("val/acc", self.val_acc, on_epoch=True, sync_dist=True)
            self.log("val/f1_macro", self.val_f1_macro, on_epoch=True, sync_dist=True)
            self.log("val/f1_weighted", self.val_f1_weighted, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()
        self.val_f1_macro.reset()
        self.val_f1_weighted.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            param_groups_no_decay(self.model, weight_decay=self.weight_decay),
            lr=self.learning_rate,
        )
        scheduler = warmup_cosine_scheduler(
            optimizer, warmup_steps=self.warmup_steps, total_steps=self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def set_total_steps(self, total_steps: int) -> None:
        self.total_steps = total_steps