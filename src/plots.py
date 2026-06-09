import matplotlib.pyplot as plt

from sklearn.metrics import (
          roc_curve,
          precision_recall_curve, 
          confusion_matrix,
          ConfusionMatrixDisplay,
          classification_report,
          auc,
          average_precision_score
)


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
    
def plot_confusion_matrix(preds, labels, figfile):  
    cm = confusion_matrix(labels, preds)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, aux = plt.subplots()
    disp.plot(ax=aux)

    plt.savefig(figfile)
    plt.close()

