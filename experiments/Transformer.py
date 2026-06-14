import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import time
from datetime import datetime

from sklearn.metrics import (
          roc_curve,
          precision_recall_curve, 
          confusion_matrix,
          ConfusionMatrixDisplay,
          classification_report,
          auc,
          average_precision_score
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.classification import (
          BinaryAccuracy, 
          BinaryAUROC, 
          BinaryAveragePrecision
)


from src.utils import count_trainable_params, count_total_params
from src.datasets import DNADataModule
from src.model import DNAClassifier
from src.plots import (
    plot_roc,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_multiclass_roc, 
    plot_multiclass_pr, 
    plot_multiclass_confusion_matrix
)


# -----------------------------
# CONFIG LOADER
# -----------------------------
def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main(config_path: str):

    config = load_config(config_path)

    generator = torch.Generator().manual_seed(config["training"]["seed"])
 
    start_time = time.time()

    # =========================
    # MODEL CONFIG
    # =========================
    model_name = config["model"]["name"]
    tuning_value = config["model"]["tuning"]

    # =========================
    # DATA CONFIG
    # =========================
    data_cfg = config["data"]
    batch_size = data_cfg["batch_size"]
    data_type = data_cfg["type"]
    input_file = data_cfg["input_file"]
    split_mode = data_cfg["split_mode"]

    train_chroms = data_cfg.get("train_chroms")
    val_chroms = data_cfg.get("val_chroms")
    test_chroms = data_cfg.get("test_chroms")

    # =========================
    # OUTPUT CONFIG
    # =========================
    output_path = config["output"]["dir"]
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # =========================
    # DATA MODULE
    # =========================
    data_module = DNADataModule(config)

    data_module.setup()
    num_classes = data_module.num_classes
    class_names = data_module.class_names

    print("\nClasses:", class_names)

    # =========================
    # MODEL
    # =========================
    model = DNAClassifier(config=config, num_classes=num_classes)

    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)

    # =========================
    # CALLBACKS
    # =========================
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_path}/checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        verbose=True
    )

    # =========================
    # TRAINER
    # =========================
    trainer_cfg = config["training"]

    trainer = Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator="auto",
        precision=trainer_cfg.get("precision", "16-mixed"),
        callbacks=[checkpoint_callback, early_stop_callback],
        check_val_every_n_epoch=trainer_cfg["check_val_every_n_epoch"],
    )

    # =========================
    # TRAIN
    # =========================
    train_start = time.time()
    trainer.fit(model, data_module)
    train_end = time.time()

    best_val_metrics = model.best_val_metrics.copy()

    # =========================
    # LOAD BEST MODEL
    # =========================
    best_model_path = checkpoint_callback.best_model_path
    #best_model = DNAClassifier.load_from_checkpoint(best_model_path)
    best_model = DNAClassifier.load_from_checkpoint(
        best_model_path,
        config=config,
        num_classes=num_classes
    )

    # save model + tokenizer
    best_model.transformer.save_pretrained(f"{output_path}/model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{output_path}/tokenizer")

    # =========================
    # TEST
    # =========================
    metrics = trainer.test(best_model, datamodule=data_module)[0]
    print(metrics)

    # =========================
    # INFERENCE RESULTS
    # =========================
    ids = np.array(best_model.test_ids)
    probs = best_model.test_probs.detach().cpu().numpy()
    labels = best_model.test_labels.detach().cpu().numpy()
    preds = probs.argmax(axis=1)

    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # =========================
    # SAVE PREDICTIONS
    # =========================
    output_df = pd.DataFrame({
        "id": ids,
        "true_label": [class_names[i] for i in labels],
        "predicted_label": [class_names[i] for i in preds],
        "confidence": probs.max(axis=1)
    })

    for i, name in enumerate(class_names):
        output_df[f"prob_{name}"] = probs[:, i]

    output_df.to_csv(f"{output_path}/test_predictions.csv", index=False)

    # =========================
    # PLOTS
    # =========================
    if num_classes == 2:
        plot_roc(probs[:, 1], labels, f"{output_path}/roc.png")
        plot_pr_curve(probs[:, 1], labels, f"{output_path}/pr.png")
        plot_confusion_matrix(preds, labels, f"{output_path}/cm.png")

    else:
        plot_multiclass_roc(probs, labels, class_names, f"{output_path}/roc.png")
        plot_multiclass_pr(probs, labels, class_names, f"{output_path}/pr.png")
        plot_multiclass_confusion_matrix(preds, labels, class_names, f"{output_path}/cm.png")

    # =========================
    # SUMMARY
    # =========================
    tuning_name = (
        "full" if tuning_value == 1 else
        "freeze" if tuning_value == 0 else
        f"last_{abs(tuning_value)}"
    )

    end_time = time.time()

    result = {
        "tuning": tuning_name,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "epochs_trained": trainer.current_epoch + 1,

        "train_time_min": (train_end - train_start) / 60,
        "run_time_min": (end_time - start_time) / 60,

        **best_val_metrics,

        "test_acc": metrics["test_accu"],
        "test_loss": metrics["test_loss"],
        "test_auc": metrics["test_roc_auc"],
        "test_pr_auc": metrics["test_pr_auc"],
        "test_f1_macro": metrics.get("test_f1_macro"),
    }

    result_df = pd.DataFrame([result])
    result_file = f"{output_path}/experiment_summary.csv"

    if Path(result_file).exists():
        result_df.to_csv(result_file, mode="a", header=False, index=False)
    else:
        result_df.to_csv(result_file, index=False)

    print("\nDONE ✔")
    print(f"Results saved to {output_path}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)