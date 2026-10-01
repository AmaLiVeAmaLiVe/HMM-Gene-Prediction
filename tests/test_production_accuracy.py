import os
import sys
import subprocess
import numpy as np

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import download_genome_data, parse_genome_and_labels_4state, create_train_test_split
from src.evaluate import evaluate_gene_boundaries

def run_production_validation():
    accession = "NC_000913.3"
    gbk_path = download_genome_data(accession, output_dir="data/raw")
    dna_seq, labels, cds_features = parse_genome_and_labels_4state(gbk_path)
    
    # 80/20 split
    _, _, test_seq, _, split_pos = create_train_test_split(dna_seq, labels, train_ratio=0.8)
    L_test = len(test_seq)
    
    # Ground truth CDS in the test slice (adjusted to test-slice relative 0-based coords)
    true_test_cds = [
        (start - split_pos, end - split_pos)
        for start, end, strand in cds_features
        if start >= split_pos and end <= len(dna_seq)
    ]
    
    # 1. Prepare raw input FASTA
    os.makedirs("data/test", exist_ok=True)
    raw_fasta = "data/test/validation_input.fasta"
    with open(raw_fasta, "w") as f:
        f.write(f">test_contig\n{test_seq}\n")
        
    gff_out = "data/test/validation_out.gff3"
    faa_out = "data/test/validation_out.faa"
    
    # 2. Execute CLI as a real subprocess (verifying full CLI integration)
    cmd = [
        sys.executable, "predict.py",
        "--input", raw_fasta,
        "--output", gff_out,
        "--proteins", faa_out,
        "--model", "models/ecoli_model.npz",
        "--min-len", "180"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Production CLI failed: {result.stderr}"
    
    # 3. Parse predicted coordinates back from the generated GFF3
    gff_pred_genes = []
    with open(gff_out, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 5 and parts[2] == "CDS":
                # Convert 1-based inclusive GFF3 back to 0-based half-open [start-1, end)
                start_0 = int(parts[3]) - 1
                end_0 = int(parts[4])
                gff_pred_genes.append((start_0, end_0, end_0 - start_0))
                
    # 4. Evaluate production predictions against NCBI reference
    metrics = evaluate_gene_boundaries(true_test_cds, gff_pred_genes, slack_bp=6)
    
    print("\n" + "=" * 50)
    print("      PRODUCTION MODE ACCURACY VERIFICATION       ")
    print("=" * 50)
    print(f"Annotated Genes in Test Set       : {metrics['Annotated Genes in Test Set']}")
    print(f"Total Predicted Genes in GFF3     : {metrics['Total Predicted Genes']}")
    print(f"Exact Boundary Matches            : {metrics['Exact Boundary Matches (Start & Stop)']}")
    print(f"Gene Sensitivity (Overlap)        : {metrics['Gene Sensitivity (Overlap)']:.2f}%")
    print(f"Gene Precision (Overlap)          : {metrics['Gene Precision (Overlap)']:.2f}%")
    print("=" * 50)
    
    # Quality Assertions
    assert metrics["Gene Precision (Overlap)"] >= 95.0, "Production Precision fell below 95%!"
    assert metrics["Gene Sensitivity (Overlap)"] >= 80.0, "Production Sensitivity fell below 80%!"
    print("\n[PASSED] Production mode runs end-to-end and reproduces high benchmark accuracy.")

if __name__ == "__main__":
    run_production_validation()