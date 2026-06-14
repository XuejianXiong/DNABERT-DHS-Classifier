import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import AutoModel

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score
)


class DNAClassifier(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        num_classes: int
    ):
        super().__init__()

        self.config = config
        self.num_classes = num_classes

        # =====================================================
        # CONFIG
        # =====================================================
        model_cfg = config["model"]
        train_cfg = config["training"]

        model_name = model_cfg["name"]
        tuning = model_cfg["tuning"]
        hidden_factor = model_cfg.get("hidden_factor", 1)
        dropout = model_cfg.get("dropout", 0.2)
        pooling = model_cfg.get("pooling", "cls")

        self.lr = train_cfg["lr"]
        self.optimizer_name = train_cfg["optimizer"]
        metric_average = train_cfg.get("metric_average", "macro")

        self.pooling = pooling

        # =====================================================
        # BACKBONE
        # =====================================================
        self.transformer = AutoModel.from_pretrained(model_name)

        hidden_size = self.transformer.config.hidden_size
        hidden_dim = int(hidden_size * hidden_factor)

        # =====================================================
        # CLASSIFIER HEAD
        # =====================================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # =====================================================
        # LOSS
        # =====================================================
        self.loss_fn = nn.CrossEntropyLoss()

        # =====================================================
        # METRICS
        # =====================================================
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average=metric_average
        )

        self.val_auroc = MulticlassAUROC(
            num_classes=num_classes,
            average=metric_average
        )

        self.val_pr_auc = MulticlassAveragePrecision(
            num_classes=num_classes,
            average=metric_average
        )

        self.test_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average=metric_average
        )

        self.test_auroc = MulticlassAUROC(
            num_classes=num_classes,
            average=metric_average
        )

        self.test_pr_auc = MulticlassAveragePrecision(
            num_classes=num_classes,
            average=metric_average
        )

        self.test_f1_macro = MulticlassF1Score(
            num_classes=num_classes,
            average=metric_average
        )

        # =====================================================
        # TUNING
        # =====================================================
        self.set_tuning_layers(tuning)

        # =====================================================
        # BEST VALIDATION METRICS
        # =====================================================
        self.best_val_metrics = {
            "val_loss": float("inf"),
            "val_accu": 0.0,
            "val_roc_auc": 0.0,
            "val_pr_auc": 0.0,
        }

    # =====================================================
    # TUNING STRATEGY
    # =====================================================
    def set_tuning_layers(self, tuning: int):

        for param in self.transformer.parameters():
            param.requires_grad = False

        if tuning == 0:
            return

        if tuning == 1:
            for param in self.transformer.parameters():
                param.requires_grad = True
            return

        if tuning < 0:

            if not hasattr(self.transformer, "encoder"):
                raise ValueError(
                    "Transformer model does not expose encoder layers."
                )

            n_layers = len(self.transformer.encoder.layer)
            k = abs(tuning)

            if k > n_layers:
                raise ValueError(
                    f"Requested {k} layers but model only has {n_layers}"
                )

            for layer in self.transformer.encoder.layer[n_layers - k:]:
                for param in layer.parameters():
                    param.requires_grad = True

            return

        raise ValueError(
            "tuning must be 0, 1, or negative integer"
        )

    # =====================================================
    # POOLING
    # =====================================================
    def pool_embeddings(self, hidden, attention_mask):

        if self.pooling == "cls":
            return hidden[:, 0, :]

        elif self.pooling == "mean":

            mask = attention_mask.unsqueeze(-1)
            masked_hidden = hidden * mask

            return (
                masked_hidden.sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )

        else:
            raise ValueError(
                f"Unknown pooling method: {self.pooling}"
            )

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, input_ids, attention_mask):

        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        hidden = output.last_hidden_state

        embedding = self.pool_embeddings(hidden, attention_mask)

        logits = self.classifier(embedding)

        return logits

    # =====================================================
    # TRAINING
    # =====================================================
    def training_step(self, batch, batch_idx):

        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # =====================================================
    # VALIDATION
    # =====================================================
    def on_validation_start(self):

        self.val_accuracy.reset()
        self.val_auroc.reset()
        self.val_pr_auc.reset()

    def validation_step(self, batch, batch_idx):

        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.val_accuracy.update(preds, batch["labels"])
        self.val_auroc.update(probs, batch["labels"])
        self.val_pr_auc.update(probs, batch["labels"])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accu", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_roc_auc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pr_auc", self.val_pr_auc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):

        metrics = self.trainer.callback_metrics
        val_loss = metrics.get("val_loss")

        if val_loss is None:
            return

        if float(val_loss) < self.best_val_metrics["val_loss"]:

            self.best_val_metrics = {
                "val_loss": float(val_loss),
                "val_accu": float(metrics["val_accu"]),
                "val_roc_auc": float(metrics["val_roc_auc"]),
                "val_pr_auc": float(metrics["val_pr_auc"]),
            }

    # =====================================================
    # TEST
    # =====================================================
    def on_test_start(self):

        self.test_probs = []
        self.test_labels = []
        self.test_ids = []

        self.test_accuracy.reset()
        self.test_auroc.reset()
        self.test_pr_auc.reset()

    def test_step(self, batch, batch_idx):

        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.test_accuracy.update(preds, batch["labels"])
        self.test_auroc.update(probs, batch["labels"])
        self.test_pr_auc.update(probs, batch["labels"])
        self.test_f1_macro.update(preds, batch["labels"])

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_accu", self.test_accuracy, on_epoch=True, prog_bar=True)
        self.log("test_roc_auc", self.test_auroc, on_epoch=True, prog_bar=True)
        self.log("test_pr_auc", self.test_pr_auc, on_epoch=True, prog_bar=True)
        self.log("test_f1_macro", self.test_f1_macro, on_epoch=True, prog_bar=True)

        self.test_probs.append(probs.cpu())
        self.test_labels.append(batch["labels"].cpu())
        self.test_ids.extend(batch["id"])

    def on_test_end(self):

        self.test_probs = torch.cat(self.test_probs)
        self.test_labels = torch.cat(self.test_labels)

    # =====================================================
    # OPTIMIZER
    # =====================================================
    def configure_optimizers(self):

        if self.optimizer_name.lower() == "adamw":

            return torch.optim.AdamW(self.parameters(), lr=self.lr)

        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")