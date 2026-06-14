import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# =========================
# TF cooperative grammar model
# =========================

TF_PAIRS = {
    "HIF1A_AP1": {
        "tf1": ["CACGTG", "CACGCG", "CACATG"],        # TF1 motif family
        "tf2": ["TGACTCA", "TGACGTA", "TGAGTCA"],     # TF2 motif family
        "optimal_distance": 10,                       # biological preferred spacing
        "sigma": 8,                                   # variability in spacing
        "mutation_rate": 0.05
    }
}

# =========================
# DNA background model
# =========================

def random_dna(length: int) -> str:
    return "".join(random.choices("ACGT", k=length))


def mutate_motif(motif: str, mutation_rate: float = 0.05) -> str:
    """Introduce biological degeneracy in TF binding sites."""
    bases = list(motif)
    for i in range(len(bases)):
        if random.random() < mutation_rate:
            bases[i] = random.choice("ACGT")
    return "".join(bases)


def interaction_distance(mu: int, sigma: float) -> int:
    """Sample biologically realistic TF-TF spacing."""
    d = int(np.random.normal(mu, sigma))
    return max(3, min(d, 80))  # clamp realistic enhancer spacing


# =========================
# Sequence construction
# =========================

def insert_motif(seq, motif, pos):
    return seq[:pos] + motif + seq[pos + len(motif):]


def generate_enhancer(seq_len: int, tf_pair: dict) -> str:
    """
    Enhancer = cooperative TF binding:
    TF1 + TF2 with preferred spacing distribution
    """

    seq = random_dna(seq_len)

    tf1 = mutate_motif(random.choice(tf_pair["tf1"]), tf_pair["mutation_rate"])
    tf2 = mutate_motif(random.choice(tf_pair["tf2"]), tf_pair["mutation_rate"])

    distance = interaction_distance(
        tf_pair["optimal_distance"],
        tf_pair["sigma"]
    )

    pos1 = random.randint(50, seq_len - 100)
    pos2 = pos1 + len(tf1) + distance

    # boundary safety
    if pos2 + len(tf2) >= seq_len:
        pos2 = max(0, pos1 - distance)

    seq = insert_motif(seq, tf1, pos1)
    seq = insert_motif(seq, tf2, pos2)

    return seq


def generate_non_enhancer(seq_len: int, tf_pair: dict) -> str:
    """
    Negative class = no cooperative binding
    (biologically meaningful negatives, NOT label flips)
    """

    seq = random_dna(seq_len)

    tf1 = mutate_motif(random.choice(tf_pair["tf1"]), tf_pair["mutation_rate"])
    tf2 = mutate_motif(random.choice(tf_pair["tf2"]), tf_pair["mutation_rate"])

    mode = random.choice(["missing_tf", "wrong_spacing"])

    if mode == "missing_tf":
        # only one TF present
        if random.random() < 0.5:
            pos = random.randint(0, seq_len - len(tf1))
            seq = insert_motif(seq, tf1, pos)
        else:
            pos = random.randint(0, seq_len - len(tf2))
            seq = insert_motif(seq, tf2, pos)

    elif mode == "wrong_spacing":
        # both TFs present but non-cooperative spacing
        pos1 = random.randint(0, seq_len - 100)
        pos2 = random.randint(150, seq_len - len(tf2))

        seq = insert_motif(seq, tf1, pos1)
        seq = insert_motif(seq, tf2, pos2)

    return seq


# =========================
# dataset generator
# =========================

def generate_dataset(n_samples: int = 2000, seq_len: int = 300):
    tf_pair = TF_PAIRS["HIF1A_AP1"]

    sequences = []
    labels = []

    half = n_samples // 2

    # positive (enhancer)
    for _ in range(half):
        sequences.append(generate_enhancer(seq_len, tf_pair))
        labels.append("enhancer")

    # negative (non-enhancer)
    for _ in range(half):
        sequences.append(generate_non_enhancer(seq_len, tf_pair))
        labels.append("non-enhancer")

    df = pd.DataFrame({
        "sequence": sequences,
        "label": labels
    })

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================
# main
# =========================

def main():
    os.makedirs("data", exist_ok=True)

    df = generate_dataset(n_samples=2000, seq_len=300)

    outfile = "data/grammar_proper_tf.csv"
    df.to_csv(outfile, index=False)

    print(f"\nSaved: {outfile}")
    print(df["label"].value_counts())
    print(df.head())


if __name__ == "__main__":
    main()