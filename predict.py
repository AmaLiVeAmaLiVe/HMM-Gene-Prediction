#!/usr/bin/env python3
"""
predict.py — Production CLI for Bacterial Gene Prediction using Higher-Order HMM
Usage:
    python predict.py --input genome.fasta --output predictions.gff3 --proteins proteins.faa
"""

import argparse
import sys
import os
from Bio import SeqIO

from src.train import train_and_save_model, load_trained_model
from src.hmm_model import HMMGenePredictor2ndOrder
from src.evaluate import (
    extract_predicted_genes,
    map_reverse_predictions,
    resolve_strand_overlaps
)
from src.utils import translate_dna, reverse_complement


DEFAULT_MODEL_PATH = "models/ecoli_model.npz"


def predict_on_sequence(model: HMMGenePredictor2ndOrder, seq: str, min_len: int = 180):
    seq_len = len(seq)
    
    # Pass 1: Forward Strand
    fwd_path = model.viterbi(seq)
    fwd_genes = extract_predicted_genes(fwd_path, min_length_bp=min_len)
    
    # Pass 2: Reverse Complement Strand
    rev_seq = reverse_complement(seq)
    rev_path = model.viterbi(rev_seq)
    raw_rev = extract_predicted_genes(rev_path, min_length_bp=min_len)
    rev_genes = map_reverse_predictions(raw_rev, seq_len=seq_len)
    
    # Resolve Overlaps (returns (start, end, length, strand))
    resolved = resolve_strand_overlaps(fwd_genes, rev_genes, max_allowed_overlap=15)
    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Ab initio Bacterial Gene Predictor (Dual-Strand 2nd-Order HMM + Shine-Dalgarno)"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input raw FASTA file")
    parser.add_argument("--output", "-o", default="output/predictions.gff3", help="Output GFF3 file path (default: output/predictions.gff3)")
    parser.add_argument("--proteins", "-p", default="output/proteins.faa", help="Output protein FASTA path (default: output/proteins.faa)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH, help="Path to pre-trained .npz model file")
    parser.add_argument("--train", action="store_true", help="Force retrain the model before prediction")
    parser.add_argument("--ref-genome", default="NC_000913.3", help="NCBI Accession to train on if model is missing or --train is set")
    parser.add_argument("--min-len", type=int, default=180, help="Minimum ORF length in base pairs (default: 180)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: Input FASTA '{args.input}' not found.")
        sys.exit(1)

    # 1. Ensure models directory exists & handle model persistence
    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    if args.train or not os.path.exists(args.model):
        print(f"[*] Pre-trained model '{args.model}' not found. Training on reference ({args.ref_genome})...")
        train_and_save_model(reference_accession=args.ref_genome, output_path=args.model)

    model = load_trained_model(args.model)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.proteins) or ".", exist_ok=True)

    print(f"[*] Reading contigs from {args.input}...")
    total_genes = 0
    all_gff_lines = ["##gff-version 3\n"]
    all_protein_lines = []

    gene_counter = 1
    for record in SeqIO.parse(args.input, "fasta"):
        contig_id = record.id
        contig_seq = str(record.seq).upper()
        print(f"    -> Processing contig: {contig_id} ({len(contig_seq):,} bp)...")
        
        genes = predict_on_sequence(model, contig_seq, min_len=args.min_len)
        total_genes += len(genes)

        for start, end, length, strand in genes:
            strand_char = "+" if strand == 1 else "-"
            gene_name = f"gene_{gene_counter:04d}"
            
            # GFF3 line (1-based inclusive coordinates)
            gff_line = (
                f"{contig_id}\tHMMGeneFinder\tCDS\t{start + 1}\t{end}\t.\t{strand_char}\t0\t"
                f"ID={gene_name};gene_biotype=protein_coding\n"
            )
            all_gff_lines.append(gff_line)

            # Extract DNA and translate
            if strand == 1:
                gene_dna = contig_seq[start:end]
            else:
                gene_dna = reverse_complement(contig_seq[start:end])

            protein_seq = translate_dna(gene_dna)
            header = f">{contig_id}_{gene_name} [location={start + 1}..{end}] [strand={strand_char}] [length={len(protein_seq)}aa]"
            all_protein_lines.append(f"{header}\n{protein_seq}\n")

            gene_counter += 1

    # Write GFF3 file
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(all_gff_lines)

    # Write Protein FASTA file
    with open(args.proteins, "w", encoding="utf-8") as f:
        f.writelines(all_protein_lines)

    print("\n" + "=" * 50)
    print(f"[+] Prediction Complete!")
    print(f"[+] Total Predicted Genes : {total_genes}")
    print(f"[+] GFF3 Annotations      : {args.output}")
    print(f"[+] Translated Proteins   : {args.proteins}")
    print("=" * 50)


if __name__ == "__main__":
    main()
