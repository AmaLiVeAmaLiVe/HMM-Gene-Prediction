import numpy as np

from src.data_loader import download_genome_data, parse_genome_and_labels_4state, create_train_test_split
from src.train import train_2nd_order_hmm
from src.hmm_model import HMMGenePredictor2ndOrder
from src.evaluate import (
    extract_predicted_genes, 
    evaluate_gene_boundaries, 
    evaluate_nucleotide_level
)


def main():
    accession = "NC_000913.3"  # E. coli K-12
    print(f"[1/5] Loading {accession} and Constructing 4-State Frame Labels...")
    gbk_path = download_genome_data(accession, output_dir="data/raw")
    dna_seq, labels, cds_features = parse_genome_and_labels_4state(gbk_path)
    
    print("[2/5] Creating 80/20 Train/Test Split...")
    train_seq, train_labels, test_seq, test_labels, split_pos = create_train_test_split(
        dna_seq, labels, train_ratio=0.8
    )
    print(f"Split index at base: {split_pos:,}")
    print(f"Train length: {len(train_seq):,} bp | Test length: {len(test_seq):,} bp")
    
    # Ground truth forward-strand CDS within the test partition
    test_cds = [
        (start - split_pos, end - split_pos)
        for start, end, strand in cds_features
        if start >= split_pos and end <= len(dna_seq) and strand == 1
    ]
    
    print("[3/5] Estimating 2nd-Order HMM Parameters (Codon-Context MLE)...")
    log_initial, log_trans, log_emiss, log_emiss_0th = train_2nd_order_hmm(train_seq, train_labels)
    
    states = ["INTERGENIC", "C1", "C2", "C3"]
    model = HMMGenePredictor2ndOrder(
        states=states,
        log_initial=log_initial,
        log_trans=log_trans,
        log_emiss=log_emiss,
        log_emiss_0th=log_emiss_0th,
        alphabet="ACGT"
    )
    
    print("[4/5] Running 2nd-Order Viterbi Decoding with Start/Stop Constraints...")
    predicted_path = model.viterbi(test_seq)
    
    # Binary masks for nucleotide evaluation
    test_binary_true = np.where(test_labels > 0, 1, 0)
    test_binary_pred = np.array([0 if s == "INTERGENIC" else 1 for s in predicted_path], dtype=np.int32)
    
    print("[5/5] Computing Performance Reports...")
    nuc_metrics = evaluate_nucleotide_level(test_binary_true, test_binary_pred)
    pred_genes = extract_predicted_genes(predicted_path, min_length_bp=90)
    gene_metrics = evaluate_gene_boundaries(test_cds, pred_genes, slack_bp=6)
    
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
        start_trip = test_seq[g_start : g_start + 3]
        stop_trip = test_seq[g_end - 3 : g_end]
        print(f"  Gene {idx:02d}: {g_start:6d} to {g_end:6d} ({g_len:4d} bp) | Start: {start_trip} | Stop: {stop_trip}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
