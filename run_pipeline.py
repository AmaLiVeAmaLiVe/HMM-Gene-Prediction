import numpy as np

from src.data_loader import download_genome_data, parse_genome_and_labels_4state, create_train_test_split
from src.train import train_supervised_4state_hmm
from src.hmm_model import HMMGenePredictor
from src.evaluate import (
    extract_predicted_genes,
    evaluate_gene_boundaries,
    evaluate_nucleotide_level
)


def main():
    print("[1/5] Downloading/Loading Genome Data...")
    gbk_path = download_genome_data("NC_000908.2", output_dir="data/raw")
    dna_seq, labels, cds_features = parse_genome_and_labels_4state(gbk_path)

    print(f"[2/5] Creating 80/20 Continuous Train/Test Split")
    train_seq, train_labels, test_seq, test_labels, split_pos = create_train_test_split(
        dna_seq, labels, train_ratio=0.8
    )
    print(f"Split index at base: {split_pos:,}")
    print(f"Train length: {len(train_seq):,} bp | Test length: {len(test_seq):,} bp")

    # Filter ground truths CDS lying inside test interval
    test_cds = [
        (start-split_pos, end-split_pos)
        for start, end, strand in cds_features
        # Forward strand evaluation
        if start >= split_pos and end <= len(dna_seq) and strand == 1
    ]

    print("[3/5] Estimating 4-state HMM Parameters via Supervised MLE...")
    log_initial, log_trans, log_emiss = train_supervised_4state_hmm(train_seq, train_labels)

    # Initialize model
    states = ["INTERGENIC", "C1", "C2", "C3"]
    model = HMMGenePredictor(
        states=states,
        log_initial=log_initial,
        log_trans=log_trans,
        log_emiss=log_emiss,
        alphabet="ACGT"
    )

    print("[4/5] Running Viterbi Decoding over Test Set...")
    predicted_path = model.viterbi(test_seq)
    
    # Binary masks for nucleotide-level evaluation (0 = Intergenic, 1 = Any Coding Frame)
    test_binary_true = np.where(test_labels > 0, 1, 0)
    test_binary_pred = np.array([0 if s == "INTERGENIC" else 1 for s in predicted_path], dtype=np.int32)
    
    print("[5/5] Computing Performance Reports...")
    
    # 1. Nucleotide Metrics
    nuc_metrics = evaluate_nucleotide_level(test_binary_true, test_binary_pred)
    
    # 2. Gene-Level Boundaries
    pred_genes = extract_predicted_genes(predicted_path, min_length_bp=90)
    gene_metrics = evaluate_gene_boundaries(test_cds, pred_genes, slack_bp=6)
    
    # Display Output
    print("\n" + "=" * 55)
    print("NUCLEOTIDE-LEVEL EVALUATION REPORT")
    print("=" * 55)
    for k, v in nuc_metrics.items():
        if k != "Confusion Matrix":
            print(f"{k:35}: {v:6.2f}%")
    print("\nConfusion Matrix:", nuc_metrics["Confusion Matrix"])
    
    print("\n" + "=" * 55)
    print("GENE-LEVEL EVALUATION REPORT")
    print("=" * 55)
    for k, v in gene_metrics.items():
        if isinstance(v, float):
            print(f"{k:35}: {v:6.2f}%")
        else:
            print(f"{k:35}: {v}")
            
    print("\nFirst 5 Predicted Genes (Offset Coordinates):")
    for idx, (g_start, g_end, g_len) in enumerate(pred_genes[:5], 1):
        print(f"  Gene {idx:02d}: {g_start:6d} to {g_end:6d}  ({g_len:4d} bp) -> Triplets: {g_len % 3 == 0}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()


