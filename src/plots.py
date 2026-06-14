import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
          roc_curve,
          precision_recall_curve, 
          confusion_matrix,
          ConfusionMatrixDisplay,
          classification_report,
          auc,
          average_precision_score
)

# =====================================
# Plot ROC for binary classification
# =====================================
def plot_roc(probs, labels, figfile):
    fpr, tpr, threshold = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
    
    plt.savefig(figfile)
    plt.close()
    
# =====================================
# Plot Precision-Recall curve for binary classification
# =====================================
def plot_pr_curve(probs, labels, figfile):    
    precision, recall, threshold = precision_recall_curve(labels, probs)
    avg_p = average_precision_score(labels, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Avg_P={avg_p:.3f})")

    plt.savefig(figfile)
    plt.close()
    
# =====================================
# Plot Confusion Matrix for binary classification
# =====================================
def plot_confusion_matrix(preds, labels, figfile):  
    cm = confusion_matrix(labels, preds)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, aux = plt.subplots()
    disp.plot(ax=aux)

    plt.savefig(figfile)
    plt.close()


# =====================================
# Plot ROC for multiclass classification
# =====================================
def plot_multiclass_roc(probs, labels, class_names, outfile):

    n_classes = len(class_names)

    y_bin = label_binarize(
        labels,
        classes=np.arange(n_classes)
    )

    plt.figure(figsize=(8,6))

    for i in range(n_classes):

        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            probs[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[i]} (AUC={roc_auc:.3f})"
        )

    plt.plot([0,1],[0,1],"k--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC")

    plt.legend()
    plt.tight_layout()

    plt.savefig(outfile)
    plt.close()


# =====================================
# Plot Precison Recall curve for multiclass classification
# =====================================
def plot_multiclass_pr(probs, labels, class_names, outfile):

    n_classes = len(class_names)

    y_bin = label_binarize(
        labels,
        classes=np.arange(n_classes)
    )

    plt.figure(figsize=(8,6))

    for i in range(n_classes):

        precision, recall, _ = precision_recall_curve(
            y_bin[:, i],
            probs[:, i]
        )

        plt.plot(
            recall,
            precision,
            label=class_names[i]
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall")

    plt.legend()
    plt.tight_layout()

    plt.savefig(outfile)
    plt.close()

# =====================================
# Plot Confusion Matrix for multiclass classification
# =====================================
def plot_multiclass_confusion_matrix(
    preds,
    labels,
    class_names,
    outfile
):

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()