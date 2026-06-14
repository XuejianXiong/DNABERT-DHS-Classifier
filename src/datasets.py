import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer
import pytorch_lightning as pl


# =========================
# Utility
# =========================
def seq_to_kmers(seq: str, k: int = 6) -> str:
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


# =========================
# Data Loader Function
# =========================
def get_data_for_model(config: dict):
    data_type = config["data"]["type"]
    file_path = config["data"]["input_file"]

    if data_type == "synthetic":
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["sequence", "label"])

        label_map = {
            "non-enhancer": 0,
            "enhancer": 1,
        }

        df["label"] = df["label"].map(label_map)
        df["id"] = [f"seq_{i}" for i in range(len(df))]

    elif data_type == "dhs":
        df = pd.read_csv(file_path, sep="\t")

        df = df[["dhs_id", "sequence", "chr", "TAG"]].copy()
        df.rename(columns={"dhs_id": "id"}, inplace=True)

        df = df.dropna(subset=["sequence", "TAG"])

        label_map = {
            "K562_ENCLB843GMH": 0,
            "hESCT0_ENCLB449ZZZ": 1,
            "HepG2_ENCLB029COU": 2,
            "GM12878_ENCLB441ZZZ": 3,
        }

        df["label"] = df["TAG"].map(label_map)

    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # reverse map for interpretation
    label_map_trans = {v: k for k, v in label_map.items()}

    return df, label_map_trans


# =========================
# Dataset
# =========================
class DNASequenceDataset(Dataset):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.df, self.label_map_trans = get_data_for_model(config)

        self.ids = self.df["id"].tolist()
        self.sequences = self.df["sequence"].tolist()
        self.labels = self.df["label"].values

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

        self.k = config["data"]["kmer_size"]
        self.max_len = config["data"]["max_len"]
        self.precompute = config["data"]["precompute_kmers"]

        # optional speed-up
        if self.precompute:
            self.kmers = [seq_to_kmers(seq, self.k) for seq in self.sequences]
        else:
            self.kmers = None

    def __len__(self):
        return len(self.labels)

    def encode(self, kmers: str):
        encoding = self.tokenizer(
            kmers,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx):

        seq = (
            self.kmers[idx]
            if self.kmers is not None
            else seq_to_kmers(self.sequences[idx], self.k)
        )

        enc = self.encode(seq)

        return {
            "id": self.ids[idx],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# =========================
# DataModule
# =========================
class DNADataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

    def setup(self, stage=None):

        dataset = DNASequenceDataset(self.config)

        self.dataset = dataset
        self.df = dataset.df
        labels = dataset.labels

        self.num_classes = len(dataset.label_map_trans)
        self.class_names = [dataset.label_map_trans[i] for i in range(self.num_classes)]

        indices = np.arange(len(dataset))

        split_mode = self.config["data"]["split_mode"]

        # =========================
        # RANDOM SPLIT
        # =========================
        if split_mode == "random":

            train_ind, test_ind = train_test_split(
                indices,
                test_size=0.2,
                stratify=labels,
                random_state=self.config["training"]["seed"]
            )

            train_labels = labels[train_ind]

            train_ind, val_ind = train_test_split(
                train_ind,
                test_size=0.2,
                stratify=train_labels,
                random_state=self.config["training"]["seed"]
            )

        # =========================
        # CHROMOSOME SPLIT
        # =========================
        elif split_mode == "chromosome":

            chrom = self.df["chr"].values

            train_ind = np.where(np.isin(chrom, self.config["data"]["train_chroms"]))[0]
            val_ind = np.where(np.isin(chrom, self.config["data"]["val_chroms"]))[0]
            test_ind = np.where(np.isin(chrom, self.config["data"]["test_chroms"]))[0]

        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        self.train_data = Subset(dataset, train_ind)
        self.val_data = Subset(dataset, val_ind)
        self.test_data = Subset(dataset, test_ind)

        print("\nSplit Summary:")
        print("Train:", pd.Series(labels[train_ind]).value_counts().to_dict())
        print("Val:", pd.Series(labels[val_ind]).value_counts().to_dict())
        print("Test:", pd.Series(labels[test_ind]).value_counts().to_dict())

    # =========================
    # DATALOADERS
    # =========================
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            persistent_workers=True
        )