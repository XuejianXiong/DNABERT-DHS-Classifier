import random
import subprocess
import pandas as pd
from pathlib import Path

random.seed(42)


# =========================
# CONFIG
# =========================

CONFIG = {
    "enhancer_bed": "data/GRCh38-ELS.bed",
    "promoter_bed": "data/GRCh38-PLS.bed",
    "genome_file": "data/hg38.genome",
    "fasta": "data/hg38.fa",

    "output_dir": "data",
    "tmp_dir": "data/tmp",
    "n_samples": 20000,

    "seq_length": 256,
    "negative_multiplier": 1.0,  # same size as positives

    "blacklist": None  # optional ENCODE blacklist BED
}


# =========================
# BEDTOOLS WRAPPERS
# =========================

def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def bed_to_fasta(bed, fasta, out_fa):
    run_cmd([
        "bedtools", "getfasta",
        "-fi", fasta,
        "-bed", bed,
        "-fo", out_fa
    ])


def shuffle_genome_background(genome_file, exclude_bed, out_bed, n_regions, size):
    """
    Generates random genomic regions excluding ENCODE features.
    """
    cmd = [
        "bedtools", "shuffle",
        "-i", exclude_bed,
        "-g", genome_file,
        "-excl", exclude_bed,
        "-chrom",
        "-noOverlapping"
    ]

    print("Generating background regions...")
    subprocess.run(cmd, check=True, stdout=open(out_bed, "w"))


# =========================
# FASTA PARSER
# =========================

def read_fasta(fa_path):
    sequences = []
    seq = ""

    with open(fa_path) as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.strip()

        if seq:
            sequences.append(seq)

    return sequences


def clean_and_trim(seqs, target_len=256):
    out = []
    for s in seqs:
        if len(s) >= target_len:
            start = random.randint(0, len(s) - target_len)
            out.append(s[start:start + target_len])
    return out


# =========================
# NEGATIVE SAMPLING (IMPORTANT)
# =========================

def create_template_bed(n, length, genome_file):
    """
    Create dummy intervals for bedtools shuffle.
    """
    df = pd.read_csv(genome_file, sep="\t", header=None, names=["chr", "size"])

    chroms = df["chr"].tolist()
    sizes = dict(zip(df["chr"], df["size"]))

    rows = []

    for _ in range(n):
        chrom = random.choice(chroms)
        max_start = max(1, sizes[chrom] - length)

        start = random.randint(0, max_start)
        end = start + length

        rows.append([chrom, start, end])

    bed = pd.DataFrame(rows)
    bed_file = CONFIG["tmp_dir"]+"/tmp_template.bed"
    bed.to_csv(bed_file, sep="\t", header=False, index=False)

    return bed_file


def generate_negative_sequences(
    genome_fa,
    genome_file,
    exclude_bed,
    n,
    length=256
):

    template_bed = create_template_bed(n, length, genome_file)
    shuffled_bed = CONFIG["tmp_dir"]+"/tmp_shuffled.bed"
    fasta_out = CONFIG["tmp_dir"]+"/tmp_neg.fa"

    # shuffle existing intervals
    subprocess.run([
        "bedtools", "shuffle",
        "-i", template_bed,
        "-g", genome_file,
        "-excl", exclude_bed,
        "-chrom",
        "-noOverlapping"
    ], check=True, stdout=open(shuffled_bed, "w"))

    # extract sequences
    subprocess.run([
        "bedtools", "getfasta",
        "-fi", genome_fa,
        "-bed", shuffled_bed,
        "-fo", fasta_out
    ], check=True)

    # parse fasta
    def read_fasta(fp):
        seqs, s = [], ""
        with open(fp) as f:
            for line in f:
                if line.startswith(">"):
                    if s:
                        seqs.append(s); s = ""
                else:
                    s += line.strip()
        if s:
            seqs.append(s)
        return seqs

    return read_fasta(fasta_out)


# =========================
# POSITIVE SET LOADER
# =========================

def load_positive_sequences(bed_file, fasta, length=256):
    tmp_fa = CONFIG["tmp_dir"]+"/tmp_pos.fa"

    bed_to_fasta(bed_file, fasta, tmp_fa)
    seqs = read_fasta(tmp_fa)

    return clean_and_trim(seqs, length)


# =========================
# MAIN PIPELINE
# =========================

def build_dataset():

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["tmp_dir"]).mkdir(parents=True, exist_ok=True)

    print("\nLoading ENCODE Enhancers (ELS)...")
    enhancers = load_positive_sequences(
        CONFIG["enhancer_bed"],
        CONFIG["fasta"],
        CONFIG["seq_length"]
    )

    print("Loading ENCODE Promoters (PLS)...")
    promoters = load_positive_sequences(
        CONFIG["promoter_bed"],
        CONFIG["fasta"],
        CONFIG["seq_length"]
    )

    print("Generating genomic negatives...")
    exclude_bed = CONFIG["tmp_dir"] + "/exclude_all.bed"

    # combine enhancer + promoter exclusion
    run_cmd([
        "bash", "-c",
        f"cat {CONFIG['enhancer_bed']} {CONFIG['promoter_bed']} > {exclude_bed}"
    ])

    negatives = generate_negative_sequences(
        CONFIG["fasta"],
        CONFIG["genome_file"],
        exclude_bed,
        n=int(len(enhancers) * CONFIG["negative_multiplier"]),
        length=CONFIG["seq_length"]
    )

    # =========================
    # BUILD DATASET
    # =========================

    df = pd.DataFrame({
        "sequence": enhancers + negatives,
        "label": (
            ["enhancer"] * len(enhancers) +
            ["non-enhancer"] * len(negatives)
        )
    })

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_file = f"{CONFIG['output_dir']}/encode_dataset.csv"
    df.to_csv(out_file, index=False)

    print("\nSaved:", out_file)
    print(df["label"].value_counts())


if __name__ == "__main__":
    build_dataset()