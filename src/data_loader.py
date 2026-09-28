import os
import numpy as np
from Bio import Entrez, SeqIO


# NCBI requires an email parameter for API requests
Entrez.email = "name@example.com"

def download_genome_data(accession: str="NC_000908.2", output_dir: str="data/raw") -> str:
    """Downloads complete GenBank flat file containing sequence and annotations."""
    os.makedirs(output_dir, exist_ok=True)
    gbk_path = os.path.join(output_dir, f"{accession}.gbk")

    if not os.path.exists(gbk_path):
        print(f"Fetching {accession} from NCBI Entrez...")
        with Entrez.efetch(db="nuccore", id=accession, rettype="gbwithparts", retmode="text") as handle:
            with open (gbk_path, "w") as f_out:
                f_out.write(handle.read())
        print(f"Saved at: {gbk_path}")
    else:
        print(f"Using cached file: {gbk_path}")

    return gbk_path


def parse_genome_and_labels_4state(gbk_path: str):
    """
    Parses GenBank file and builds:
      - dna_seq: Full nucleotide sequence as uppercase string
      - labels: np.ndarray of shape (len(dna_seq),):
        0: Intergenic
        1: Codon Position 1 (C1)
        2: Codon Position 2 (C2)
        3: Codon Position 3 (C3)
    """
    record = SeqIO.read(gbk_path, "genbank")
    dna_seq = str(record.seq).upper()
    seq_len = len(dna_seq)

    labels = np.zeros(seq_len, dtype=np.int32)
    cds_features = []

    for feature in record.features:
        if feature.type == "CDS":
            start = int(feature.location.start)
            end = int(feature.location.end)
            strand = feature.location.strand
            
            # ONLY label forward-strand genes for the forward model!
            if strand == 1:
                cds_features.append((start, end, strand))
                length = end - start
                for offset in range(length):
                    frame = (offset % 3) + 1
                    labels[start + offset] = frame
            else:
                # Store reverse features separately so we know where they are
                cds_features.append((start, end, strand))

    return dna_seq, labels, cds_features


def create_train_test_split(dna_seq: str, labels: np.ndarray, train_ratio: float = 0.8):
    """Performs a continuous split along the chromosome to preserve biological structure."""
    split_idx = int(len(dna_seq) * train_ratio)

    train_seq = dna_seq[:split_idx]
    train_labels = labels[:split_idx]

    test_seq = dna_seq[split_idx:]
    test_labels = labels[split_idx:]

    return train_seq, train_labels, test_seq, test_labels, split_idx


if __name__ == "__main__":
    gbk_file = download_genome_data()
    seq, lbl = parse_genome_and_labels_4state(gbk_file)

    print(f"Total genome length: {len(seq):,} bp")
    print(f"Coding base percentage: {np.mean(lbl) * 100:.2f}%")
