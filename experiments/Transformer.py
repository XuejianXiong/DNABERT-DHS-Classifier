import json
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from transformers import AutoTokenizer

from src.datasets import DNADataModule
from src.model import DNAClassifier
from src.utils import count_trainable_params, count_total_params
from src.plots import (
    plot_roc,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_multiclass_pr,
    plot_multiclass_confusion_matrix
)

# -----------------------------
# CONFIG
# -----------------------------
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# MAIN
# -----------------------------
def main(config_path: str):

    config = load_config(config_path)

    pl.seed_everything(config["training"]["seed"])

    start_time = time.time()

    # -------------------------
    # CONFIG SHORTCUTS
    # -------------------------
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    out_cfg = config["output"]

    Path(out_cfg["dir"]).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # DATA MODULE
    # -------------------------
    data_module = DNADataModule(config)
    data_module.setup()

    num_classes = data_module.num_classes
    class_names = data_module.class_names

    print("\nClasses:", class_names)

    # -------------------------
    # MODEL
    # -------------------------
    model = DNAClassifier(config=config, num_classes=num_classes)

    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)

    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")

    # -------------------------
    # CALLBACKS
    # -------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{out_cfg['dir']}/checkpoints",
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

    # -------------------------
    # MLflow LIGHTNING LOGGER (ONLY TRACKING SYSTEM)
    # -------------------------
    mlf_logger = MLFlowLogger(
        experiment_name=f"DNA-BERT-{data_cfg['type']}",
        tracking_uri="file:./mlruns"
    )

    # log hyperparameters once
    mlf_logger.log_hyperparams({
        "model_name": model_cfg["name"],
        "tuning": model_cfg["tuning"],
        "batch_size": data_cfg["batch_size"],
        "max_len": data_cfg["max_len"],
        "kmer_size": data_cfg["kmer_size"],
        "lr": train_cfg["lr"],
        "max_epochs": train_cfg["max_epochs"],
        "precision": train_cfg["precision"],
        "num_classes": num_classes,
        "trainable_params": trainable_params,
        "total_params": total_params
    })

    # -------------------------
    # TRAINER
    # -------------------------
    trainer = Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        precision=train_cfg.get("precision", "16-mixed"),
        callbacks=[checkpoint_callback, early_stop_callback],
        check_val_every_n_epoch=train_cfg["check_val_every_n_epoch"],
        logger=mlf_logger,
    )

    # -------------------------
    # TRAIN
    # -------------------------
    train_start = time.time()
    trainer.fit(model, data_module)
    train_end = time.time()

    # -------------------------
    # LOAD BEST MODEL
    # -------------------------
    best_model = DNAClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        config=config,
        num_classes=num_classes
    )

    # save model + tokenizer
    best_model.transformer.save_pretrained(f"{out_cfg['dir']}/model")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    tokenizer.save_pretrained(f"{out_cfg['dir']}/tokenizer")

    # -------------------------
    # TEST
    # -------------------------
    metrics = trainer.test(best_model, datamodule=data_module)[0]
    print(metrics)

    # -------------------------
    # INFERENCE
    # -------------------------
    ids = np.array(best_model.test_ids)
    probs = best_model.test_probs.detach().cpu().numpy()
    labels = best_model.test_labels.detach().cpu().numpy()
    preds = probs.argmax(axis=1)

    # -------------------------
    # SAVE PREDICTIONS
    # -------------------------
    output_df = pd.DataFrame({
        "id": ids,
        "true_label": [class_names[i] for i in labels],
        "predicted_label": [class_names[i] for i in preds],
        "confidence": probs.max(axis=1)
    })

    for i, name in enumerate(class_names):
        output_df[f"prob_{name}"] = probs[:, i]

    pred_file = f"{out_cfg['dir']}/test_predictions.csv"
    output_df.to_csv(pred_file, index=False)

    # -------------------------
    # PLOTS
    # -------------------------
    if num_classes == 2:
        plot_roc(probs[:, 1], labels, f"{out_cfg['dir']}/test_roc_curve.png")
        plot_pr_curve(probs[:, 1], labels, f"{out_cfg['dir']}/test_pr_curve.png")
        plot_confusion_matrix(preds, labels, f"{out_cfg['dir']}/test_cm_table.png")
    else:
        plot_multiclass_roc(probs, labels, class_names, f"{out_cfg['dir']}/test_roc_curve.png")
        plot_multiclass_pr(probs, labels, class_names, f"{out_cfg['dir']}/test_pr_curve.png")
        plot_multiclass_confusion_matrix(preds, labels, class_names, f"{out_cfg['dir']}/test_cm_table.png")

    # =========================
    # EXPERIMENT SUMMARY CSV
    # =========================
    end_time = time.time()

    result = {
        "model": model_cfg["name"],
        "tuning": model_cfg["tuning"],
        "dataset": data_cfg["type"],

        "trainable_params": trainable_params,
        "total_params": total_params,

        "epochs": train_cfg["max_epochs"],

        "train_time_min": (train_end - train_start) / 60,
        "total_time_min": (time.time() - start_time) / 60,

        "test_acc": float(metrics["test_accu"]),
        "test_loss": float(metrics["test_loss"]),
        "test_auc": float(metrics["test_roc_auc"]),
        "test_pr_auc": float(metrics["test_pr_auc"]),
        "test_f1_macro": float(metrics.get("test_f1_macro", np.nan)),
    }

    summary_path = f"{out_cfg['dir']}/experiment_summary.csv"
    df = pd.DataFrame([result])

    if Path(summary_path).exists():
        df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_path, index=False)


    # -------------------------
    # LOG ARTIFACTS (LIGHTNING MLflow RUN)
    # -------------------------
    run_id = mlf_logger.run_id
    client = mlf_logger.experiment

    client.log_artifact(run_id, pred_file)
    client.log_artifact(run_id, f"{out_cfg['dir']}/test_roc_curve.png")
    client.log_artifact(run_id, f"{out_cfg['dir']}/test_pr_curve.png")
    client.log_artifact(run_id, f"{out_cfg['dir']}/test_cm_table.png")
    client.log_artifact(run_id, config_path)
    client.log_artifact(run_id, summary_path)    


    # -------------------------
    # DONE
    # -------------------------
    print("\nDONE ✔")
    print(f"Total time: {(end_time - start_time)/60:.2f} min")
    print(f"MLflow run saved under experiment: DNA-BERT-{data_cfg['type']}")


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)