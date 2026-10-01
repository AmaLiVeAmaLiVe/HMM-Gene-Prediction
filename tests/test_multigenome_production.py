import os
import sys
import subprocess
import pytest

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import download_genome_data, parse_genome_and_labels_4state, create_train_test_split
from src.evaluate import evaluate_gene_boundaries

# Test configuration: accession, expected min sensitivity, expected min precision
BENCHMARK_CONFIGS = [
    {
        "accession": "NC_000913.3",
        "organism": "Escherichia coli K-12",
        "min_sensitivity": 85.0,
        "min_precision": 95.0,
    },
    {
        "accession": "NC_000964.3",
        "organism": "Bacillus subtilis 168",
        "min_sensitivity": 85.0,
        "min_precision": 95.0,
    },
    {
        "accession": "NC_002516.2",
        "organism": "Pseudomonas aeruginosa PAO1",
        "min_sensitivity": 90.0,
        "min_precision": 95.0,
    },
    {
        "accession": "NC_000908.2",
        "organism": "Mycoplasma genitalium G37 (Code 4)",
        "min_sensitivity": 75.0,  # Lower threshold expected due to non-canonical Code 4 TGA stops
        "min_precision": 95.0,
    }
]

def parse_gff3_predictions(gff_path: str):
    """Parses 1-based inclusive GFF3 back into 0-based half-open (start, end, length) tuples."""
    genes = []
    with open(gff_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 5 and parts[2] == "CDS":
                start_0 = int(parts[3]) - 1
                end_0 = int(parts[4])
                genes.append((start_0, end_0, end_0 - start_0))
    return genes

def run_single_genome_production_test(config: dict):
    acc = config["accession"]
    org = config["organism"]
    print(f"\n[*] Evaluating Production Pipeline on {org} ({acc})...")

    # 1. Download and prepare test partition
    gbk_path = download_genome_data(acc, output_dir="data/raw")
    dna_seq, labels, cds_features = parse_genome_and_labels_4state(gbk_path)

    _, _, test_seq, _, split_pos = create_train_test_split(dna_seq, labels, train_ratio=0.8)

    true_test_cds = [
        (start - split_pos, end - split_pos)
        for start, end, strand in cds_features
        if start >= split_pos and end <= len(dna_seq)
    ]

    # 2. Write raw FASTA file
    test_dir = f"data/test/{acc}"
    os.makedirs(test_dir, exist_ok=True)
    raw_fasta = os.path.join(test_dir, f"{acc}_test.fasta")
    with open(raw_fasta, "w", encoding="utf-8") as f:
        f.write(f">{acc}_test_contig\n{test_seq}\n")

    gff_out = os.path.join(test_dir, f"{acc}_predicted.gff3")
    faa_out = os.path.join(test_dir, f"{acc}_proteins.faa")
    model_path = f"models/{acc}_model.npz"

    # 3. Execute predict.py CLI via subprocess
    cmd = [
        sys.executable, "predict.py",
        "--input", raw_fasta,
        "--output", gff_out,
        "--proteins", faa_out,
        "--model", model_path,
        "--ref-genome", acc,
        "--min-len", "180"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CLI Error Output:\n{result.stderr}")
        raise RuntimeError(f"predict.py failed for {acc} with exit code {result.returncode}")

    # 4. Parse output and calculate metrics
    gff_pred_genes = parse_gff3_predictions(gff_out)
    metrics = evaluate_gene_boundaries(true_test_cds, gff_pred_genes, slack_bp=6)

    # 5. Verify protein FASTA was populated
    assert os.path.exists(faa_out), f"Protein FASTA {faa_out} was not generated!"
    assert os.path.getsize(faa_out) > 0, f"Protein FASTA {faa_out} is empty!"

    # 6. Report
    print(f"    Annotated Test Genes : {metrics['Annotated Genes in Test Set']}")
    print(f"    Predicted Genes      : {metrics['Total Predicted Genes']}")
    print(f"    Exact Boundary Hits  : {metrics['Exact Boundary Matches (Start & Stop)']}")
    print(f"    Sensitivity (Recall) : {metrics['Gene Sensitivity (Overlap)']:.2f}%")
    print(f"    Precision            : {metrics['Gene Precision (Overlap)']:.2f}%")

    # Assertions
    assert metrics["Gene Precision (Overlap)"] >= config["min_precision"], (
        f"Precision {metrics['Gene Precision (Overlap)']:.2f}% below threshold {config['min_precision']}%"
    )
    assert metrics["Gene Sensitivity (Overlap)"] >= config["min_sensitivity"], (
        f"Sensitivity {metrics['Gene Sensitivity (Overlap)']:.2f}% below threshold {config['min_sensitivity']}%"
    )
    print(f"[+] {org} PASSED all criteria.")

def test_all_production_genomes():
    for config in BENCHMARK_CONFIGS:
        run_single_genome_production_test(config)

if __name__ == "__main__":
    test_all_production_genomes()