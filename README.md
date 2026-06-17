# Genomic Foundation Models for Cell-Type Specific DHS Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![DNABERT](https://img.shields.io/badge/DNABERT-Genome%20Transformer%20(6--mer)-8A2BE2)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Lightning](https://img.shields.io/badge/PyTorch--Lightning-792EE5)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![License](https://img.shields.io/badge/License-MIT-green)


## Overview

This project explores the use of genomic foundation models to classify cell-type-specific DNase I hypersensitive sites (DHSs) from DNA sequence alone.

DNase I hypersensitive sites are markers of regulatory DNA and represent regions of open chromatin associated with promoters, enhancers, and other regulatory elements. Because many human diseases arise from dysregulation of gene regulatory programs, understanding sequence determinants of DHS activity is an important step toward interpreting non-coding variation and regulatory mechanisms.

This project was developed as part of a computational biology and AI technical assessment focused on leveraging pretrained genomic language models for regulatory sequence prediction.

---

## Objectives

The primary goals were:

1. Construct a curated DHS sequence dataset from the Meuleman DHS Index.
2. Extract cell-type-specific DHS regions from multiple human cell types.
3. Fine-tune a pretrained DNABERT model on DHS sequences.
4. Evaluate whether sequence information alone can distinguish cell-type-specific regulatory elements.
5. Establish a reproducible workflow suitable for future disease-related regulatory sequence studies.

---

## Dataset

### DHS Index

The DHS Index contains approximately 3.6 million regulatory elements derived from hundreds of human tissues and cell types. 
Data were obtained from [meuleman.org](https://www.meuleman.org/research/dhsindex/) . 

---

### Cell Types

The following cell types were selected:

| Label   | Cell Type                    |
| ------- | ---------------------------- |
| K562    | Chronic myelogenous leukemia |
| HepG2   | Liver carcinoma              |
| GM12878 | B lymphoblastoid             |
| hESCT0  | Human embryonic stem cells   |

Only DHS peaks uniquely associated with a single selected cell type were retained.

---

## Data Processing

The workflow follows the preprocessing strategy implemented in the [DNA-Diffusion Project](https://github.com/pinellolab/DNA-Diffusion?utm_source=chatgpt.com) :

Specifically:

1. Download DHS metadata and peak matrices.
2. Download the hg38 reference genome.
3. Extract DHS coordinates.
4. Retrieve genomic sequences centered on DHS summits.
5. Filter peaks to retain cell-type-exclusive DHS regions.
6. Balance the dataset across cell types.

The resulting dataset contains:

* DNA sequence
* DHS identifier
* Chromosome
* Cell-type label

---

## Model

### Foundation Model

This project uses: **DNABERT-6**,  a transformer language model pretrained on genomic sequences using 6-mer tokenization.

---

### Fine-Tuning Strategy

Input DNA sequences are converted into overlapping 6-mers before tokenization.

Example:
```
DNA:  ATCGTACG
6-mers:   ATCGTA TCGTAC CGTACG
```

The pretrained DNABERT encoder is fine-tuned using a classification head consisting of:

* Linear layer
* ReLU activation
* Dropout
* Output layer

Only the final transformer layers were updated during training:

```
tuning = -2
```

This significantly reduces computational requirements while preserving most pretrained representations.

---

## Train / Validation / Test Split

To avoid information leakage between highly similar genomic regions, chromosome-wise splitting was used.

Example:
```
Training:   chr1–chr16
Validation:   chr17–chr19
Testing:    chr20–chr22, chrX
```

This evaluation strategy is more realistic than random splitting because the model must generalize to unseen genomic loci.

---

## Repository Structure

```
.
├── data/
│   ├── hg38.fa
│   └── filtered_dataset.txt
│
├── experiments/
│   ├── Prepare_DHS_data.py
│   └── Transformer.py
│
├── src/
│   ├── datasets.py
│   ├── model.py
│   ├── plots.py
│   └── utils.py
│
├── outputs/dhs/
│   ├── checkpoints
│   ├── model
│   ├── tokenizer
│   ├── experiment_summary.csv
│   ├── test_cm_table.png
│   ├── test_roc_curve.png
│   ├── test_pr_curve.png
│   └── test_predictions.csv
│
├── configs/
│   └── config_dhs.json
│
├── requirements.txt
└── README.md
```

---

## Installation

Create a Python environment:
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

---

## Running the Pipeline

### Train DNABERT

```
python3 -m experiments.Transformer --config configs/config_dhs.json
```

### Evaluate Model

The evaluation pipeline automatically generates:

* Classification report
* Confusion matrix
* ROC curves
* Precision-recall curves
* Prediction table


---

## Experiment Tracking with MLflow

This project integrates MLflow through PyTorch Lightning's MLFlowLogger for reproducible experiment tracking.

After training completes, 

```mlflow ui```

Open

```http://127.0.0.1:5000```

to visualize experiment results. The dashboard allows comparison of multiple experiments, hyperparameter configurations, and model versions.

---

## Results

Example performance obtained using chromosome-wise evaluation:

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~70%  |
| ROC-AUC  | ~0.89 |
| PR-AUC   | ~0.77 |

The model successfully captures sequence-level regulatory signatures associated with individual cell types despite being trained solely on DNA sequence.

The model achieves strong performance on DHS classification.

### ROC Curve
![ROC Curve](outputs/dhs/test_roc_curve.png)

### Precision-Recall Curve
![PR Curve](outputs/dhs/test_pr_curve.png)

### Confusion Matrix
![Confusion Matrix](outputs/dhs/test_cm_table.png)

---

## Limitations

Several limitations remain:

* Only four cell types were analyzed.
* DHS activity is influenced by chromatin state and transcription factor occupancy not included in the model.
* Limited fine-tuning was performed on consumer hardware.
* No external epigenomic signals were incorporated.

---

## Future Work

Potential extensions include:

* Larger foundation models (Nucleotide Transformer, HyenaDNA, GENA-LM).
* Multi-task prediction across hundreds of cell types.
* Variant effect prediction.
* Integration with GWAS and disease-associated variants.
* Attention-based interpretation of regulatory motifs.
* Retrieval-augmented genomic foundation models.

---

## References

Meuleman W, et al. (2020). Index and biological spectrum of human DNase I hypersensitive sites.

Ji Y, et al. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome.

Pinello Lab DNA-Diffusion Project:  https://github.com/pinellolab/DNA-Diffusion

---

## License

MIT License – feel free to use, adapt, and share.