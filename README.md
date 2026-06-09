# Genomic Sequence Classification with DNABERT

## Abstract

We present a deep learning framework for genomic sequence classification using transformer-based DNA language models (DNABERT). The goal is to classify regulatory DNA sequences, specifically enhancer vs non-enhancer regions, using k-mer tokenized genomic input.

We systematically evaluate different fine-tuning strategies, including frozen backbone, partial fine-tuning, and full fine-tuning of transformer layers. Experimental results demonstrate that partial fine-tuning of the last transformer layers achieves the best performance, outperforming both frozen and full fine-tuning approaches in terms of accuracy and ROC-AUC.

These findings suggest that pretrained DNA language models capture general genomic structure, while task-specific adaptation is most effective in higher transformer layers.

---

## 1. Introduction

Regulatory element prediction is a fundamental problem in computational genomics. Enhancers play a critical role in gene regulation, yet their identification from raw DNA sequences remains challenging.

Recent advances in transformer-based language models, such as DNABERT, enable learning contextual representations of genomic sequences using k-mer tokenization. However, optimal fine-tuning strategies for downstream genomic tasks remain an open question.

This study investigates how different levels of model fine-tuning affect classification performance on enhancer prediction.

---

## 2. Methods

### 2.1 Dataset

We use a binary classification dataset:

- Enhancer sequences
- Non-enhancer sequences

Sequences are encoded using **k-mer tokenization (k=6)**.

The dataset includes:
- Synthetic enhancer classification data (easy.csv)
- Balanced class distribution across train/validation/test splits

---

### 2.2 Model Architecture

We use a pretrained DNABERT model:

- Base model: `zhihan1996/DNA_bert_6`
- Transformer encoder (BERT architecture)
- Hidden representation extracted from `[CLS]` token
- Classification head:
  - Linear → ReLU → Dropout → Linear

---

### 2.3 Training Strategies

We evaluate three fine-tuning strategies:

- **Frozen backbone**: only classifier head is trained
- **Partial fine-tuning**: last N transformer layers are trainable
- **Full fine-tuning**: entire transformer model is trainable

Loss function: Cross-Entropy Loss  
Optimizer: AdamW  
Framework: PyTorch Lightning  
Precision: Mixed precision (16-bit AMP)

---

### 2.4 Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- ROC-AUC
- Precision-Recall AUC
- Confusion matrix

---

## 3. Results

### 3.1 Performance Comparison

| Tuning Strategy | Trainable Params | Test Accuracy | ROC-AUC | PR-AUC | Training Time |
|----------------|------------------|--------------|---------|--------|---------------|
| Frozen | ~0.6M | 0.52 | 0.53 | 0.52 | ~1.7 min |
| Last 2 layers | ~14.8M | **0.88** | **0.95** | **0.95** | ~3.3 min |
| Last 4 layers | ~28.9M | 0.83 | 0.91 | 0.90 | ~4.7 min |
| Full fine-tuning | ~89.8M | 0.50 | 0.50 | 0.50 | ~81.6 min |

---

### 3.2 Key Finding

Partial fine-tuning of the last transformer layers achieves optimal performance, suggesting that DNABERT’s pretrained representations capture general genomic structure, while task-specific adaptation is localized to higher-level layers.

---

## 4. Discussion

Our results highlight the importance of selecting an appropriate fine-tuning strategy when applying large pretrained language models to genomic sequences.

### Key observations:

- **Frozen models underfit**, indicating that classifier-only training is insufficient for capturing task-specific genomic signals.
- **Full fine-tuning leads to performance collapse**, likely due to overfitting and limited dataset size.
- **Partial fine-tuning provides the best trade-off**, preserving pretrained representations while allowing task-specific adaptation.

These findings are consistent with transfer learning behavior observed in natural language processing models, where lower layers capture general structure and higher layers adapt to task-specific patterns.

---

## 5. Conclusion

We present a modular framework for genomic sequence classification using DNABERT. Our experiments demonstrate that partial fine-tuning yields optimal performance for enhancer classification tasks.

Future work will extend this framework to:

- Promoter prediction
- Multi-class regulatory element classification
- Integration of ENCODE and FANTOM5 datasets
- Chromosome-level train/test splitting

---

## 6. Installation

```
git clone <repo>

cd Genomic-Sequence-Classification

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python3 -m experiments.Transformer
```

---

