import argparse
import numpy as np

from src.data_loader import download_genome_data, parse_genome_and_labels_4state, create_train_test_split
from src.train import train_2nd_order_hmm
from src.hmm_model import HMMGenePredictor2ndOrder
from src.evaluate import (
    extract_predicted_genes, 
    evaluate_gene_boundaries, 
    evaluate_nucleotide_level,
    map_reverse_predictions,
    resolve_strand_overlaps
)
from src.utils import reverse_complement


def main():
    parser = argparse.ArgumentParser(description="Benchmark HMM Gene Predictor against NCBI Ground Truth")
    parser.add_argument("--accession", "-a", default="NC_000913.3", help="NCBI Accession (e.g. NC_000913.3, NC_000908.2)")
    parser.add_argument("-g", "--translation-table", type=int, choices=[11, 4], help="Genetic code table (11=Standard, 4=Mycoplasma)")
    args = parser.parse_args()

    accession = args.accession
    table_id = args.translation_table

    gbk_path = download_genome_data(accession, output_dir="data/raw")
    dna_seq, labels, cds_features, metadata_table = parse_genome_and_labels_4state(gbk_path)

    # Resolve active translation table
    if args.translation_table is not None:
        table_id = args.translation_table
        print(f"      -> Translation table explicitly set to: Table {table_id}")
    else:
        table_id = metadata_table
        print(f"      -> Translation table auto-detected from metadata: Table {table_id}")

    print(f"[1/5] Loading {accession} (Table {table_id}) and Parsing CDS Features...")
        
    print("[2/5] Creating 80/20 Train/Test Split...")
    train_seq, train_labels, test_seq, test_labels, split_pos = create_train_test_split(
        dna_seq, labels, train_ratio=0.8
    )
    L_test = len(test_seq)
    
    # Ground truth: ALL CDS (both forward +1 and reverse -1) within the test slice
    all_test_cds = [
        (start - split_pos, end - split_pos)
        for start, end, strand in cds_features
        if start >= split_pos and end <= len(dna_seq)
    ]
    fwd_count = sum(1 for s, e, st in cds_features if s >= split_pos and e <= len(dna_seq) and st == 1)
    rev_count = sum(1 for s, e, st in cds_features if s >= split_pos and e <= len(dna_seq) and st == -1)
    print(f"      Test Slice: {fwd_count} Forward Genes + {rev_count} Reverse Genes = {len(all_test_cds)} Total")

    print("[3/5] Estimating 2nd-Order HMM Parameters on Forward Training Split...")
    log_initial, log_trans, log_emiss, log_emiss_0th = train_2nd_order_hmm(train_seq, train_labels)
    
    states = ["INTERGENIC", "C1", "C2", "C3"]
    model = HMMGenePredictor2ndOrder(
        states=states,
        log_initial=log_initial,
        log_trans=log_trans,
        log_emiss=log_emiss,
        log_emiss_0th=log_emiss_0th,
        table_id=table_id,
        alphabet="ACGT"
    )
    
    print("[4/5] Running Dual-Strand Viterbi Decoding...")
    # --- Pass 1: Forward Strand ---
    print("      -> Decoding Forward Strand...")
    fwd_path = model.viterbi(test_seq)
    fwd_genes = extract_predicted_genes(fwd_path, min_length_bp=180)
    print(f"         Found {len(fwd_genes)} candidate forward genes.")
    
    # --- Pass 2: Reverse Strand ---
    print("      -> Decoding Reverse Complement Strand...")
    rev_test_seq = reverse_complement(test_seq)
    rev_path = model.viterbi(rev_test_seq)
    raw_rev_genes = extract_predicted_genes(rev_path, min_length_bp=180)
    rev_genes_mapped = map_reverse_predictions(raw_rev_genes, seq_len=L_test)
    print(f"         Found {len(rev_genes_mapped)} candidate reverse genes.")
    
    # --- Resolve Antisense Overlaps ---
    combined_genes = resolve_strand_overlaps(fwd_genes, rev_genes_mapped, max_allowed_overlap=15)
    print(f"      -> Final Resolved Gene Set: {len(combined_genes)} predicted genes across both strands.")

    # Build dual-strand binary ground truth & predicted arrays for nucleotide metrics
    true_dual_mask = np.zeros(L_test, dtype=np.int32)
    for g_start, g_end in all_test_cds:
        true_dual_mask[max(0, g_start):min(L_test, g_end)] = 1
        
    pred_dual_mask = np.zeros(L_test, dtype=np.int32)
    # Unpack 4 items (start, end, length, strand)
    for g_start, g_end, _, _ in combined_genes:
        pred_dual_mask[max(0, g_start):min(L_test, g_end)] = 1

    print("[5/5] Computing Performance Reports (Both Strands Combined)...")
    nuc_metrics = evaluate_nucleotide_level(true_dual_mask, pred_dual_mask)
    gene_metrics = evaluate_gene_boundaries(all_test_cds, combined_genes, slack_bp=6)
    
    print("\n" + "=" * 55)
    print(f"   NUCLEOTIDE-LEVEL EVALUATION (Table {table_id})")
    print("=" * 55)
    for k, v in nuc_metrics.items():
        if k != "Confusion Matrix":
            print(f"{k:35}: {v:6.2f}%")
    print("\nConfusion Matrix:\n", nuc_metrics["Confusion Matrix"])
    
    print("\n" + "=" * 55)
    print(f"     GENE-LEVEL EVALUATION (Table {table_id})")
    print("=" * 55)
    for k, v in gene_metrics.items():
        if isinstance(v, float):
            print(f"{k:35}: {v:6.2f}%")
        else:
            print(f"{k:35}: {v}")
            
    print("\nFirst 5 Predicted Genes on Forward Reference Coordinates:")
    # Unpack 4 items (start, end, length, strand)
    for idx, (g_start, g_end, g_len, st) in enumerate(combined_genes[:5], 1):
        st_char = "+" if st == 1 else "-"
        print(f"  Gene {idx:02d}: {g_start:6d} to {g_end:6d} ({g_len:4d} bp) [strand {st_char}]")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()