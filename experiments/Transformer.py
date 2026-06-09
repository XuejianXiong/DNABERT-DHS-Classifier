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
from src.plots import plot_confusion_matrix, plot_pr_curve, plot_roc


generator = torch.Generator().manual_seed(42)

Path("outputs").mkdir(parents=True, exist_ok=True)
#Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)



def main():
    
    start_time = time.time()

    model_name = "zhihan1996/DNA_bert_6"
    #model_name = "Peltarion/dnabert-minilm-small"

    tuning_value = -2          # 0, -2, -4, -6, 1
    input_data = "medium"      # easy, medium, realistic, noisy, grammar
    output_path = f"outputs/{input_data}"

    data_module = DNADataModule(
        csv_file=f"data/{input_data}.csv",
        tokenizer_name=model_name,
        batch_size=64
    )
    
    model = DNAClassifier(model_name=model_name, tuning=tuning_value)

    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)      


    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_path}/checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,   # stop if no improvement for 2 epochs
        verbose=True
    )

    # In PyTorch Lightning, the Trainer is the central engine 
    # that runs your entire training workflow.
    # It is a high-level orchestration layer 
    # that controls training, validation, logging, checkpointing, and hardware execution.
    trainer = Trainer(
        max_epochs=10,
        #accelerator="gpu" if torch.cuda.is_available() else "cpu",
        accelerator="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stop_callback],
        check_val_every_n_epoch=2,
    )

    train_start = time.time()
    trainer.fit(model, data_module)
    train_end = time.time()

    best_val_metrics = model.best_val_metrics.copy()

    best_model_path = checkpoint_callback.best_model_path
    best_model = DNAClassifier.load_from_checkpoint(best_model_path)

    best_model.transformer.save_pretrained(f"{output_path}/model")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{output_path}/tokenizer")

    metrics = trainer.test(best_model, datamodule=data_module)[0]
    print(metrics)
    '''
    pd.DataFrame(metrics).to_csv(
        "outputs/test_metrics.csv",
        index=False
    )
    '''

    # detach() Remove gradient tracking
    # cpu() Move to CPU memory because numpy() only works on CPU memory
    probs = best_model.test_probs.detach().cpu().numpy()
    labels = best_model.test_labels.detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)

    print(
          classification_report(
              labels,
              preds,
              target_names=['non-enhancer','enhancer']
          )
    )

    print("Min prob:", probs.min())
    print("Max prob:", probs.max())
    print("Mean prob:", probs.mean())

    output_df = pd.DataFrame({
        "true_label": np.where(
            labels == 1,
            "enhancer",
            "non-enhancer"
        ),
        "predicted_label": np.where(
            preds == 1,
            "enhancer",
            "non-enhancer"
        ),
        "probability_enhancer": probs
    })
    output_df.to_csv(f"{output_path}/test_predictions.csv", index=False)

    plot_roc(probs, labels, f"{output_path}/test_roc_curve.png")
    plot_pr_curve(probs, labels, f"{output_path}/test_pr_curve.png")
    plot_confusion_matrix(preds, labels, f"{output_path}/test_confusion_matrix.png")

    if tuning_value == 1:
        tuning_name = "full"
    elif tuning_value == 0:
        tuning_name = "freeze"
    else:
        tuning_name = f"last_{abs(tuning_value)}"

    end_time = time.time()
    print("=" * 60)
    print(f"Tuning Mode: {tuning_name}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Total Params: {total_params:,}")
    print(f"Model Training Time: {(train_end - train_start)/60:.2f} minutes")
    print(f"Total Running Time: {(end_time - start_time)/60:.2f} minutes")
    print("=" * 60)
    
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
    }

    results_file = f"{output_path}/experiment_summary.csv"
    result_df = pd.DataFrame([result])
    if Path(results_file).exists():
        result_df.to_csv(results_file, mode="a", header=False, index=False, float_format="%.2f")
    else:
        result_df.to_csv(results_file, index=False, float_format="%.2f")


if __name__ == "__main__":
    main()