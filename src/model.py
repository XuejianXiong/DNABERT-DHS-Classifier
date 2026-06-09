import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics.classification import (
          BinaryAccuracy, 
          BinaryAUROC, 
          BinaryAveragePrecision
)


# Lightning Model (Transformer classifier)
class DNAClassifier(pl.LightningModule):
    def __init__(self, model_name: str, tuning: int = 1, lr: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()

        self.transformer = AutoModel.from_pretrained(model_name)
        if tuning < 1:
            self.set_tuning_layers(tuning)

        hidden_size = self.transformer.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = lr

        self.val_accuracy = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_pr_auc = BinaryAveragePrecision()

        self.test_accuracy = BinaryAccuracy()
        self.test_auroc = BinaryAUROC()
        self.test_pr_auc = BinaryAveragePrecision()

        self.best_val_metrics = {
            "val_loss": float("inf"),
            "val_accu": 0.0,
            "val_roc_auc": 0.0,
            "val_pr_auc": 0.0,
            }

    def set_tuning_layers(self, tuning: int = 0):
        for param in self.transformer.parameters():
            param.requires_grad = False 

        if tuning == 1:
            for param in self.transformer.parameters():
                param.requires_grad = True
            return    
        elif tuning == 0:
            return
        elif tuning < 0:
            for layer in self.transformer.encoder.layer[tuning:]:
                for param in layer.parameters():
                    param.requires_grad = True
            return    
        
        raise ValueError("tuning must be 1, 0, or negative integer e.g. -2, -4.")
            
    
    def forward(self, input_ids, attention_mask):
        output = self.transformer(
            input_ids = input_ids, 
            attention_mask = attention_mask
            )
        
        cls_emb = output.last_hidden_state[:,0,:]
        logits = self.classifier(cls_emb)

        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])

        # Record the value of loss and label it as train_loss.
        # self.log() is a PyTorch Lightning method used to record metrics 
        # during training, validation, or testing.
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss  # This line is needed
    
    def on_validation_start(self):
        self.val_accuracy.reset()
        self.val_auroc.reset()
        self.val_pr_auc.reset()

    def validation_step(self, batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])

        # argmax() outputs the index of the highest score  
        # dim=1 means: look across columns (class dimension)
        preds = torch.argmax(logits, dim=1)
        #accuracy = (preds == batch['labels']).float().mean()
        self.val_accuracy.update(preds, batch['labels'])

        probs = torch.softmax(logits, dim=1)[:,1]
        self.val_auroc.update(probs, batch['labels'])
        self.val_pr_auc.update(probs, batch['labels'])

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

    def on_test_epoch_start(self):
        self.test_probs = []
        self.test_labels = []

        self.test_accuracy.reset()
        self.test_auroc.reset()
        self.test_pr_auc.reset()

    def test_step(self,  batch, batch_idx):
        logits = self(batch['input_ids'], batch['attention_mask'])
        loss = self.loss_fn(logits, batch['labels'])

        preds = torch.argmax(logits, dim=1)
        #accuracy = (preds == batch['labels']).float().mean()
        self.test_accuracy.update(preds, batch['labels'])

        probs = torch.softmax(logits, dim=1)[:,1]
        self.test_auroc.update(probs, batch['labels'])
        self.test_pr_auc.update(probs, batch['labels'])

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accu", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_roc_auc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_pr_auc", self.test_pr_auc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_probs.append(probs.cpu())
        self.test_labels.append(batch['labels'].cpu())
        
    def on_test_epoch_end(self):
        self.test_probs = torch.cat(self.test_probs)
        self.test_labels = torch.cat(self.test_labels)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr
        )
        return optimizer

