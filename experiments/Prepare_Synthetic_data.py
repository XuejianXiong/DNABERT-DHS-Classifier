import os
import random
import pandas as pd

random.seed(42)

# Setup parameters for different types of enhancer and non-enhancer sequencen data
DATASETS = {
    "easy": {
        "enhancer_motifs": [
            "AAAAAA",
            "AAAACC"
        ],
        "non_enhancer_motifs": [
            "CCCCCC",
            "CCCCAA"
        ],
        "noise_rate": 0.0,
        "min_length": 200,
        "max_length": 200,
    },

    "medium": {
        "enhancer_motifs": [
            "AAAAAA",
            "GGGGGG",
            "TTTTTT"
        ],
        "non_enhancer_motifs": [
            "CCCCCC",
            "ACACAC",
            "TGTGTG"
        ],
        "noise_rate": 0.0,
        "min_length": 200,
        "max_length": 200,
    },

    "realistic": {        
        "enhancer_motifs": [
            "CACGTG",
            "CATGTG",
            "CACGCG",
            "CACGTT"
        ],
        
        "non_enhancer_motifs": [
            "CACGTA",
            "CATGTA",
            "CACGCA",
            "CACGCT"
        ],

        "noise_rate": 0.0,
        "min_length": 150,
        "max_length": 300,
    },

    "noisy": {
        "enhancer_motifs": [
            "CACGTG",
            "TGACTCA",
            "GGAAAG",
            "CGCGCG"
        ],
        "non_enhancer_motifs": [
            "CACGTA",
            "TGATTCA",
            "GGATAG",
            "CGTGCG"
        ],
        "noise_rate": 0.10,
        "min_length": 150,
        "max_length": 300,
    },

    "grammar": {
        "enhancer_motifs": [
            ("CACGTG", "TGACTCA")
        ],
        "non_enhancer_motifs": [
            ("CACGTG", "TGACTCA")
        ],
        "noise_rate": 0.10,
        "min_length": 300,
        "max_length": 500,
    }
}


'''
def random_dna(length: int) -> str:
    """
    Generate a random DNA sequence.
    """
    return "".join(
        random.choices("ACGT", k=length)
    )
'''
def random_dna(length):
    """
    Generate a random DNA sequence.
    """
    return "".join(random.choices(
        population="ACGT",
        weights=[0.3, 0.2, 0.2, 0.3],
        k=length
    ))

def insert_motif(
    sequence: str,
    motif: str
) -> str:
    """
    Insert a motif at a random position.
    """

    pos = random.randint(
        0,
        len(sequence) - len(motif)
    )

    return (
        sequence[:pos]
        + motif
        + sequence[pos + len(motif):]
    )


def insert_motif_pair(
    sequence: str,
    motif1: str,
    motif2: str,
    close: bool
) -> str:
    """
    Insert two motifs.

    close=True:
        motif2 placed within 20 bp of motif1

    close=False:
        motif2 placed at least 100 bp away
    """

    length = len(sequence)

    if close:

        pos1 = random.randint(
            0,
            length - len(motif1) - len(motif2) - 25
        )

        pos2 = pos1 + random.randint(
            len(motif1),
            len(motif1) + 20
        )

    else:

        pos1 = random.randint(
            0,
            length // 3
        )

        pos2 = random.randint(
            pos1 + 100,
            length - len(motif2)
        )

    sequence = (
        sequence[:pos1]
        + motif1
        + sequence[pos1 + len(motif1):]
    )

    sequence = (
        sequence[:pos2]
        + motif2
        + sequence[pos2 + len(motif2):]
    )

    return sequence


def generate_sequence(
    seq_length: int,
    motifs,
    grammar: bool = False,
    positive: bool = True
) -> str:
    """
    Generate a single sequence.
    """

    seq = random_dna(seq_length)

    if grammar:

        motif1, motif2 = motifs[0]

        seq = insert_motif_pair(
            seq,
            motif1,
            motif2,
            close=positive
        )

    else:

        # Randomly pick an integer from 1,2,3 as the number of motifs
        #n_motifs = random.randint(1, 3)
        n_motifs = random.randint(4, 8)

        selected = random.sample(
            motifs,
            min(n_motifs, len(motifs))
        )

        for motif in selected:
            seq = insert_motif(
                seq,
                motif
            )

    return seq


def generate_dataset(
    n_samples: int,
    enhancer_motifs,
    non_enhancer_motifs,
    noise_rate: float,
    min_length: int,
    max_length: int,
    grammar: bool = False
) -> pd.DataFrame:
    """
    Generate a complete dataset.
    """

    if n_samples % 2 != 0:
        raise ValueError(
            "n_samples must be even"
        )

    sequences = []
    labels = []

    n_positive = n_samples // 2
    n_negative = n_samples // 2

    for _ in range(n_positive):

        seq_length = random.randint(
            min_length,
            max_length
        )

        seq = generate_sequence(
            seq_length,
            enhancer_motifs,
            grammar=grammar,
            positive=True
        )

        sequences.append(seq)
        labels.append("enhancer")

    for _ in range(n_negative):

        seq_length = random.randint(
            min_length,
            max_length
        )

        seq = generate_sequence(
            seq_length,
            non_enhancer_motifs,
            grammar=grammar,
            positive=False
        )

        sequences.append(seq)
        labels.append("non-enhancer")

    n_noisy = int(
        n_samples * noise_rate
    )

    noisy_indices = random.sample(
        range(n_samples),
        n_noisy
    )

    for idx in noisy_indices:

        labels[idx] = (
            "non-enhancer"
            if labels[idx] == "enhancer"
            else "enhancer"
        )

    df = pd.DataFrame({
        "sequence": sequences,
        "label": labels
    })

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return df


def main():

    os.makedirs("data", exist_ok=True)

    for dataset_name, config in DATASETS.items():

        grammar = (
            dataset_name == "grammar"
        )

        df = generate_dataset(
            n_samples=1000,
            grammar=grammar,
            **config
        )

        outfile = (
            f"data/{dataset_name}.csv"
        )

        df.to_csv(
            outfile,
            index=False
        )

        print(
            f"\nGenerated: {outfile}"
        )

        print(
            df["label"].value_counts()
        )

        print(df.head())


if __name__ == "__main__":
    main()