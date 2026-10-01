import argparse
import sys
import os
from Bio import SeqIO

from src.train import load_trained_model, dynamic_self_train
from src.hmm_model import HMMGenePredictor2ndOrder
from src.evaluate import (
    extract_predicted_genes,
    map_reverse_predictions,
    resolve_strand_overlaps
)
from src.utils import translate_dna, reverse_complement


def main():
    parser = argparse.ArgumentParser(
        description="Ab initio Bacterial Gene Predictor (Dual-Strand 2nd-Order HMM)"
    )
    parser.add_argument("--input", "-i", required=True, help="Input raw FASTA file")
    parser.add_argument("--output", "-o", default="predictions.gff3", help="Output GFF3 file path")
    parser.add_argument("--proteins", "-p", default="predicted_proteins.faa", help="Output protein FASTA path")
    parser.add_argument("--model", "-m", default=None, help="Path to pre-trained .npz model (if omitted, self-trains on input)")
    parser.add_argument("-g", "--translation-table", type=int, default=11, choices=[11, 4],
                        help="Genetic code translation table (default: 11, 4 for Mycoplasma)")
    parser.add_argument("--min-len", type=int, default=180, help="Minimum ORF length in bp (default: 180)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input FASTA '{args.input}' not found.")
        sys.exit(1)

    records = list(SeqIO.parse(args.input, "fasta"))
    if not records:
        print("[!] Error: No sequences found in input FASTA.")
        sys.exit(1)

    table_id = args.translation_table

    # --- Model Resolution ---
    if args.model is not None:
        model = load_trained_model(args.model, table_id=table_id)
    else:
        # Concatenate sequences for representative statistical estimation
        combined_seq = "".join(str(r.seq).upper() for r in records)
        total_len = len(combined_seq)
        gc_pct = (combined_seq.count('G') + combined_seq.count('C')) / total_len * 100.0 if total_len > 0 else 0.0

        print(f"[*] No pre-trained model specified.")
        print(f"    Assembly Size : {total_len:,} bp across {len(records)} contig(s)")
        print(f"    Assembly GC   : {gc_pct:.1f}%")
        print(f"    Genetic Code  : Table {table_id}")
        print(f"[*] Initiating ab initio self-training on unambiguous long ORFs (>= 500 bp)...")

        try:
            model, orf_count = dynamic_self_train(combined_seq, table_id=table_id, min_len_bp=500)
            print(f"[+] Self-training successful! Extracted {orf_count} bootstrap ORFs to estimate 2nd-order parameters.")
        except ValueError as err:
            default_fallback = "models/ecoli_model.npz"
            print(f"[!] Warning: {err}")
            print(f"[*] Falling back to default pre-trained model: {default_fallback}")
            model = load_trained_model(default_fallback, table_id=table_id)

    # --- Output Preparation ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.proteins) or ".", exist_ok=True)

    all_gff_lines = ["##gff-version 3\n"]
    all_protein_lines = []
    gene_counter = 1

    print(f"[*] Decoding contigs...")
    for record in records:
        contig_id = record.id
        contig_seq = str(record.seq).upper()
        L_contig = len(contig_seq)
        print(f"    -> Processing contig: {contig_id} ({L_contig:,} bp)...")

        # Pass 1: Forward Strand
        fwd_path = model.viterbi(contig_seq)
        fwd_genes = extract_predicted_genes(fwd_path, min_length_bp=args.min_len)

        # Pass 2: Reverse Complement Strand
        rev_seq = reverse_complement(contig_seq)
        rev_path = model.viterbi(rev_seq)
        raw_rev = extract_predicted_genes(rev_path, min_length_bp=args.min_len)
        rev_genes = map_reverse_predictions(raw_rev, seq_len=L_contig)

        # Antisense Overlap Resolution
        genes = resolve_strand_overlaps(fwd_genes, rev_genes, max_allowed_overlap=15)

        for start, end, length, strand in genes:
            strand_char = "+" if strand == 1 else "-"
            gene_name = f"gene_{gene_counter:04d}"

            gff_line = (
                f"{contig_id}\tHMMGeneFinder\tCDS\t{start + 1}\t{end}\t.\t{strand_char}\t0\t"
                f"ID={gene_name};gene_biotype=protein_coding\n"
            )
            all_gff_lines.append(gff_line)

            gene_dna = contig_seq[start:end] if strand == 1 else reverse_complement(contig_seq[start:end])
            protein_seq = translate_dna(gene_dna, table_id=table_id)
            header = f">{contig_id}_{gene_name} [location={start + 1}..{end}] [strand={strand_char}] [length={len(protein_seq)}aa]"
            all_protein_lines.append(f"{header}\n{protein_seq}\n")

            gene_counter += 1

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(all_gff_lines)

    with open(args.proteins, "w", encoding="utf-8") as f:
        f.writelines(all_protein_lines)

    print("\n" + "=" * 50)
    print("[+] Prediction Complete!")
    print(f"[+] Total Predicted Genes : {gene_counter - 1}")
    print(f"[+] GFF3 Annotations      : {args.output}")
    print(f"[+] Translated Proteins   : {args.proteins}")
    print("=" * 50)


if __name__ == "__main__":
    main()