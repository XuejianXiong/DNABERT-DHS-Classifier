
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import AutoTokenizer
import pytorch_lightning as pl


def seq_to_kmers(seq: str, k: int = 6) -> str:
    return " ".join(
        seq[i:i+k]
        for i in range(len(seq) - k + 1)
    )


# =========================
# Classes
# =========================
# Dataset class
class DNASequenceDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer_name: str, max_len: int = 256):
        super().__init__()

        self.df = pd.read_csv(csv_file)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        self.labels = self.df["label"].map(
            {"non-enhancer": 0, "enhancer": 1}
        ).values

        self.encodings = []
        for seq in self.df["sequence"].tolist():
            kmers = seq_to_kmers(seq)

            encoding = self.tokenizer(
                kmers,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

            self.encodings.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            "input_ids": self.encodings[index]["input_ids"],
            "attention_mask": self.encodings[index]["attention_mask"],
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }
        return item
    

# DataModule (Lightning standard)
class DNADataModule(pl.LightningDataModule):
    def __init__(self, csv_file: str, tokenizer_name: str, batch_size: int = 32):
        super().__init__()

        self.csv_file = csv_file
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size

    '''
    def setup(self, stage = None):
        dataset = DNASequenceDataset(self.csv_file, self.tokenizer_name)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_data, test_data = random_split(
            dataset, 
            [train_size, test_size],
            generator=generator
            )

        val_size = int(0.2 * train_size)
        train_size = train_size - val_size

        self.train_data, self.val_data = random_split(
            train_data,
            [train_size, val_size],
            generator=generator
            )
        self.test_data = test_data
    '''
    def setup(self, stage = None):
        dataset = DNASequenceDataset(self.csv_file, self.tokenizer_name)

        labels = dataset.labels
        indices = np.arange(len(dataset))

        train_ind, test_ind = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        train_labels = labels[train_ind]

        train_ind, val_ind = train_test_split(
            train_ind,
            test_size=0.2,
            stratify=train_labels,
            random_state=42
        )

        self.train_data = Subset(dataset, train_ind)
        self.val_data = Subset(dataset, val_ind)
        self.test_data = Subset(dataset, test_ind)

        print("Train:")
        print(pd.Series(labels[train_ind]).value_counts())
        print("Validation:")
        print(pd.Series(labels[val_ind]).value_counts())
        print("Test:")
        print(pd.Series(labels[test_ind]).value_counts())


    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            persistent_workers=True
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            persistent_workers=True
            )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4,
            persistent_workers=True
            )
    
