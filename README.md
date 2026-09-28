# HMM Gene Predictor: Ab Initio Prokaryotic Gene Finding

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Biopython](https://img.shields.io/badge/Biopython-1.80%2B-green.svg)](https://biopython.org/)
[![Tested with Pytest](https://img.shields.io/badge/tested_with-pytest-009688.svg)](https://docs.pytest.org/)

An end-to-end, production-ready bioinformatics pipeline for **_ab initio_ prokaryotic gene prediction** using **Higher-Order Hidden Markov Models (HMMs)** with periodic reading frame grammar, Shine-Dalgarno (RBS) motif scoring, and dual-strand overlap resolution.

---

## Table of Contents

- [Overview](#overview)
- [Biological & Algorithmic Architecture](#biological--algorithmic-architecture)
  - [1. 4-State Cyclic Codon Grammar](#1-4-state-cyclic-codon-grammar)
  - [2. 2nd-Order Markov Chain Emissions](#2-2nd-order-markov-chain-emissions)
  - [3. Shine-Dalgarno (RBS) Motif Scoring](#3-shine-dalgarno-rbs-motif-scoring)
  - [4. Start and Stop Codon Constraints](#4-start-and-stop-codon-constraints)
  - [5. Dual-Strand Decoding & Overlap Resolution](#5-dual-strand-decoding--overlap-resolution)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Production Mode (CLI Usage)](#production-mode-cli-usage)
  - [Quick Start](#quick-start)
  - [Using Pre-Trained GC-Matched Models](#using-pre-trained-gc-matched-models)
  - [On-the-Fly Training on Any NCBI Genome](#on-the-fly-training-on-any-ncbi-genome)
  - [CLI Reference Table](#cli-reference-table)
- [Multi-Genome Benchmark Dashboard](#multi-genome-benchmark-dashboard)
- [Testing Suite](#testing-suite)
- [Output Specifications](#output-specifications)
- [Mathematical Formulation](#mathematical-formulation)
- [License & References](#license--references)

---

## Overview

When a microbial genome is assembled from sequencing reads, researchers receive continuous, unannotated nucleotide contigs (`A`, `C`, `G`, `T`). Identifying protein-coding genes (*Coding Sequences* or CDS) without prior transcriptomic or homology data is the foundational challenge of genome annotation.

Standard 0th-order 2-state HMMs often fail in bacterial genomes due to the **Accuracy Paradox**: bacterial chromosomes are gene-dense (>85–95% coding), and single-nucleotide frequencies between coding and non-coding regions in AT- or GC-skewed genomes are nearly indistinguishable.

This pipeline implements a **Dual-Strand 2nd-Order Hidden Markov Model** that solves this challenge by integrating:
* Triplet codon reading frame awareness ($L \equiv 0 \pmod 3$).
* Trinucleotide contextual emission probabilities.
* Ribosome Binding Site (Shine-Dalgarno) affinity bonuses.
* Dual-strand (forward and reverse complement) prediction and strand overlap filtering.
* Out-of-the-box production export to standard **GFF3** annotations and translated **Protein FASTA (`.faa`)** sequences.

---

## Biological & Algorithmic Architecture

```
                      +-------------------------------------------------+
                      |                                                 |
                      v                                                 |
[ INTERGENIC (0) ] ------> [ C1 (State 1) ] -> [ C2 (State 2) ] -> [ C3 (State 3) ]
        ^        (ATG/GTG/TTG)                                          |
        |       + RBS Bonus                                             |
        +---------------------------------------------------------------+
                                  (TAA/TAG/TGA)
```

### 1. 4-State Cyclic Codon Grammar
Instead of a naive binary (Coding vs. Intergenic) model, the pipeline models coding regions as a deterministic 3-phase cycle:
* `INTERGENIC` (State 0): Non-coding spacer DNA. Can self-transition or transition to `C1`.
* `C1` (State 1): First nucleotide of a triplet codon. Must transition to `C2` ($P=1.0$).
* `C2` (State 2): Second nucleotide of a triplet codon. Must transition to `C3` ($P=1.0$).
* `C3` (State 3): Third nucleotide of a triplet codon. Transitions either back to `C1` (continuing codon chain) or exits to `INTERGENIC` upon encountering a valid stop codon.

This structural constraint mathematically guarantees that every predicted gene has a length that is an exact multiple of 3.

### 2. 2nd-Order Markov Chain Emissions
The emission probability of nucleotide $X_t$ is conditioned on the current state $S_t$ and the preceding two nucleotides:
$$P(X_t \mid X_{t-2}, X_{t-1}, S_t)$$
This captures non-random trinucleotide preferences (codon usage bias, hexameric base dependencies, and GC wobble at position 3) without requiring the prohibitive parameter space of 5th-order interpolated Markov models. Parameters are smoothed using Laplace (+1) pseudocounts across all $4 \times 4 \times 4$ combinations.

### 3. Shine-Dalgarno (RBS) Motif Scoring
Bacterial translation initiation requires the 16S rRNA of the 30S ribosomal subunit to bind the **Shine-Dalgarno (SD)** sequence upstream of the start codon. When evaluating candidate $0 \to \text{C1}$ transitions, the model scans the window between $-15\text{ bp}$ and $-4\text{ bp}$ upstream for core consensus motifs and adds empirical log-odds weights to the Viterbi path:

| Motif | Biological Strength | Log-Odds Bonus |
| :--- | :--- | :---: |
| `AGGAGG` | Full consensus / Strongest binding | `+3.5` |
| `GGAGG` | High affinity core | `+3.0` |
| `AGGAG` | High affinity | `+2.8` |
| `GAGG` | Moderate affinity | `+2.2` |
| `AGGA` | Moderate affinity | `+2.0` |
| `GGAG` | Canonical minimal | `+1.8` |

### 4. Start and Stop Codon Constraints
* **Initiation**: The model allows canonical bacterial start codons (`ATG`, `GTG`, `TTG`). Canonical `ATG` receives preferential weight over rare alternative starts.
* **Termination**: Transitions from `C3` to `INTERGENIC` are only permitted if the terminal triplet matches a valid universal stop codon (`TAA`, `TAG`, `TGA`).
* **In-Frame Stops**: Internal transitions from `C3` to `C1` strictly prohibit stop codons, preventing premature termination or non-sense ORF prediction.

### 5. Dual-Strand Decoding & Overlap Resolution
Bacterial genomes encode genes on both the forward ($+$) and reverse ($-$) strands. The pipeline executes:
1. **Forward Pass**: Viterbi decoding across the $5' \to 3'$ sense sequence.
2. **Reverse Pass**: Viterbi decoding across the reverse complement sequence, with coordinates dynamically re-mapped to positive-strand orientation:
   $$\text{start}_{\text{orig}} = L - \text{end}_{\text{rev}}, \quad \text{end}_{\text{orig}} = L - \text{start}_{\text{rev}}$$
3. **Overlap Resolution**: When forward and reverse candidate genes collide on opposite strands with an overlap exceeding $15\text{ bp}$, the model evaluates cumulative Viterbi log-likelihood densities and retains the higher-scoring biological candidate.

---

## Project Structure

```text
HMM Gene Prediction/
│
├── predict.py                   # Production CLI entry point for FASTA annotation
├── run_pipeline.py              # 5-step development pipeline script
├── requirements.txt             # Project dependencies (Biopython, NumPy, Pytest, Matplotlib)
├── pytest.ini                   # Pytest path and runner configuration
├── .gitignore                   # Git exclusion rules for large datasets and caches
│
├── src/                         # Core algorithmic package
│   ├── __init__.py              # Package marker
│   ├── data_loader.py           # NCBI Entrez fetcher, GenBank parser, continuous train/test splitter
│   ├── train.py                 # 2nd-order parameter estimation & .npz model serialization
│   ├── hmm_model.py             # Log-space 2nd-order Viterbi decoder with signal constraints
│   ├── evaluate.py              # Dual-strand mapper, overlap resolver, boundary & nucleotide metrics
│   └── utils.py                 # Genetic code translation table & reverse complement routines
│
├── models/                      # Pre-trained, compressed model parameter archives (~3 KB each)
│   ├── ecoli_model.npz          # E. coli K-12 reference model (default)
│   ├── NC_000908.2_model.npz    # Mycoplasma genitalium G37 (32% GC)
│   ├── NC_000964.3_model.npz    # Bacillus subtilis 168 (43.5% GC)
│   ├── NC_000913.3_model.npz    # Escherichia coli K-12 (50.8% GC)
│   └── NC_002516.2_model.npz    # Pseudomonas aeruginosa PAO1 (66.6% GC)
│
├── notebooks/                   # Interactive analytical dashboards
│   └── benchmark_dashboard.ipynb# Multi-genome comparative benchmark across bacterial phyla
│
├── tests/                       # Automated testing suite
│   ├── test_viterbi.py          # Synthetic GC-island validation of the Viterbi algorithm
│   ├── test_production_accuracy.py   # End-to-end CLI integration and accuracy check on E. coli
│   ├── test_multigenome_production.py# Multi-genome benchmark test across 4 reference species
│   └── test_output_formats.py   # Formal validation of GFF3 specification and protein FASTA compliance
│
├── test_unseen.py               # Out-of-the-box validation on novel genome (Salmonella enterica LT2)
└── create_test_fasta.py         # Utility script to generate simulated contig inputs
```

---

## Installation

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/AmaLiVeAmaLiVe/HMM-Gene-Prediction.git
cd HMM-Gene-Prediction

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Production Mode (CLI Usage)

The primary entry point for annotating newly sequenced bacterial contigs is [predict.py](/HMM%20Gene%20Prediction/predict.py).

### Quick Start
Run gene prediction on the included sample contig:

```powershell
python predict.py --input data/test/ecoli_test_contigs.fasta
```

The pipeline automatically creates the `output/` folder and generates:
* `output/predictions.gff3`: Complete GFF3 feature annotations.
* `output/proteins.faa`: Translated protein sequences for all predicted genes.

---

### Standard Production Run with Custom Paths

```powershell
python predict.py `
  --input path/to/my_contigs.fasta `
  --output results/annotated_genes.gff3 `
  --proteins results/translated_proteins.faa
```

---

### Using Pre-Trained GC-Matched Models
Because codon and base frequencies vary with genomic GC content, you can select one of the bundled pre-trained models matching your target organism:

```powershell
# For High-GC genomes (~66% GC, e.g., Pseudomonas aeruginosa):
python predict.py --input contigs.fasta --model models/NC_002516.2_model.npz

# For Moderate-GC genomes (~50% GC, e.g., Escherichia coli, Salmonella):
python predict.py --input contigs.fasta --model models/ecoli_model.npz

# For Low-GC genomes (~32% GC, e.g., Mycoplasma genitalium):
python predict.py --input contigs.fasta --model models/NC_000908.2_model.npz

# For Bacillus subtilis (~43.5% GC):
python predict.py --input contigs.fasta --model models/NC_000964.3_model.npz
```

---

### On-the-Fly Training on Any NCBI Genome
To annotate an organism from a novel genus, supply the NCBI RefSeq accession with the `--train` flag. The pipeline will automatically download the reference, train the 2nd-order parameters, save the `.npz` archive, and run prediction:

```powershell
python predict.py `
  --input contigs.fasta `
  --train `
  --ref-genome NC_003197.2 `
  --model models/salmonella_model.npz
```

---

### CLI Reference Table

| Argument | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | *(Required)* | Path to input raw FASTA file (multi-contig supported) |
| `--output` | `-o` | `output/predictions.gff3` | Destination path for GFF3 annotation file |
| `--proteins` | `-p` | `output/proteins.faa` | Destination path for translated protein FASTA |
| `--model` | `-m` | `models/ecoli_model.npz` | Path to pre-trained `.npz` parameter archive |
| `--train` | | `False` | Forces training a new model before running prediction |
| `--ref-genome`| | `NC_000913.3` | NCBI Accession to fetch & train when `--train` is active |
| `--min-len` | | `180` | Minimum predicted gene length in bp (default: 180 bp / 60 aa) |

---

## Multi-Genome Benchmark Dashboard

The repository includes a comprehensive Jupyter benchmark dashboard: [notebooks/benchmark_dashboard.ipynb](/HMM%20Gene%20Prediction/notebooks/benchmark_dashboard.ipynb).

The model was systematically evaluated across a diverse phylogenetic spectrum spanning extreme GC contents:

| Organism | RefSeq Accession | Phylum | Genomic GC % | Gene Sensitivity (Recall) | Gene Precision |
| :--- | :--- | :--- | :---: | :---: | :---: |
| ***Mycoplasma genitalium* G37** | `NC_000908.2` | *Mycoplasmatota* | 32.0% | **76.4%** | **95.8%** |
| ***Bacillus subtilis* 168** | `NC_000964.3` | *Bacillota* | 43.5% | **88.2%** | **96.4%** |
| ***Escherichia coli* K-12** | `NC_000913.3` | *Pseudomonadota* | 50.8% | **88.7%** | **96.9%** |
| ***Pseudomonas aeruginosa* PAO1** | `NC_002516.2` | *Pseudomonadota* | 66.6% | **92.4%** | **97.1%** |
| ***Salmonella enterica* LT2** *(Unseen)* | `NC_003197.2` | *Pseudomonadota* | 52.2% | **87.5%** | **97.8%** |

### Key Benchmark Takeaways:
1. **Precision Stability**: Gene-level precision consistently exceeds **95–97%** across all bacterial phyla, indicating an extremely low false-positive rate.
2. **GC Skew Robustness**: Even under extreme GC content (*P. aeruginosa* at 66.6% GC), the 2nd-order context emissions correctly isolate coding frames with >92% sensitivity.
3. **Out-of-the-Box Generalization**: When evaluated on *Salmonella enterica* without any organism-specific retraining, the default *E. coli* model achieved **97.8% Precision** and **87.5% Recall**.

---

## Testing Suite

The project includes unit, integration, and format compliance tests built with `pytest`.

Run all tests from the repository root:

```powershell
pytest -v
```

### Test Breakdown

* **[tests/test_viterbi.py](/HMM%20Gene%20Prediction/tests/test_viterbi.py)**: Validates log-space Viterbi path reconstruction on a controlled, synthetic GC-island sequence.
* **[tests/test_production_accuracy.py](/HMM%20Gene%20Prediction/tests/test_production_accuracy.py)**: Executes `predict.py` via subprocess on an unseen 20% chromosomal split of *E. coli* K-12 and verifies that precision exceeds 95% and sensitivity exceeds 80%.
* **[tests/test_multigenome_production.py](/HMM%20Gene%20Prediction/tests/test_multigenome_production.py)**: Automatically verifies production CLI accuracy across all 4 reference species (*M. genitalium*, *B. subtilis*, *E. coli*, *P. aeruginosa*).
* **[tests/test_output_formats.py](/HMM%20Gene%20Prediction/tests/test_output_formats.py)**: Ensures strict adherence to the **GFF3** 9-column specification (1-based positive coordinates, valid strand characters, `ID=` tags) and checks that protein FASTA outputs contain valid IUPAC amino acids with stripped terminal stop codons.

---

## Output Specifications

### 1. GFF3 Format (`output/predictions.gff3`)
Follows standard [GFF3 specification](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md):
```text
##gff-version 3
contig_01	HMMGeneFinder	CDS	181	1260	.	+	0	ID=gene_0001;gene_biotype=protein_coding
contig_01	HMMGeneFinder	CDS	1420	2115	.	-	0	ID=gene_0002;gene_biotype=protein_coding
```
* **Col 1 (SeqID)**: Contig / chromosome identifier from input FASTA.
* **Col 2 (Source)**: `HMMGeneFinder`
* **Col 3 (Type)**: `CDS` (Coding Sequence)
* **Col 4–5 (Coordinates)**: 1-based inclusive start and end positions.
* **Col 7 (Strand)**: `+` (Forward strand) or `-` (Reverse complement strand).
* **Col 9 (Attributes)**: Unique gene identifier (`ID=gene_XXXX`) and biotype.

### 2. Protein FASTA Format (`output/proteins.faa`)
Exports fully translated amino acid sequences:
```fasta
>contig_01_gene_0001 [location=181..1260] [strand=+] [length=359aa]
MKISTLTLPVGKLVVGLDESKGQGKKLTINPAAGAVKAYVEEVLEK...
>contig_01_gene_0002 [location=1420..2115] [strand=-] [length=231aa]
MKLNIIKEDFLSKLRDEDKLYVFVAGKKIFAEM...
```
* All proteins begin with valid translation start residues.
* Terminal stop codons (`*`) are automatically stripped for downstream tools (BLASTp, InterProScan).

---

## Mathematical Formulation

### Log-Space Viterbi DP Recursion
To prevent floating-point underflow over chromosomes exceeding millions of base pairs, all dynamic programming operations are computed in log-space:

$$V_t(j) = \max_{1 \le i \le K} \left[ V_{t-1}(i) + \log A_{i, j} + R(i, j, t) \right] + \log B_j(X_t \mid X_{t-2}, X_{t-1})$$

Where:
* $V_t(j)$ is the maximum log-likelihood of a state sequence ending in state $j$ at sequence position $t$.
* $A_{i, j}$ is the state transition probability from state $i$ to state $j$.
* $B_j(X_t \mid X_{t-2}, X_{t-1})$ is the 2nd-order emission probability of nucleotide $X_t$ given the preceding dimer $(X_{t-2}, X_{t-1})$ under state $j$.
* $R(i, j, t)$ is the biological signal bonus function:
  $$R(i, j, t) = \begin{cases} 
  \text{Score}_{\text{RBS}}(t-15 : t-4) + \log(\omega_{\text{start}}) & \text{if } i = \text{Intergenic} \text{ and } j = \text{C1} \\
  -\infty & \text{if } i = \text{C3}, j = \text{C1} \text{ and } X_{t-3:t} \in \text{StopCodons} \\
  -\infty & \text{if } i = \text{C3}, j = \text{Intergenic} \text{ and } X_{t-3:t} \notin \text{StopCodons} \\
  0 & \text{otherwise}
  \end{cases}$$

---

## License & References

### License
This project is open-source and released under the **MIT License**.

### Key Scientific References
1. **Borodovsky, M., & McIninch, J. (1993)**. *GENMARK: Parallel gene recognition for both DNA strands*. Computers & Chemistry, 17(2), 123-133.
2. **Delcher, A. L., Harmon, D., Kasif, S., White, O., & Salzberg, S. L. (1999)**. *Improved microbial gene identification with GLIMMER*. Nucleic Acids Research, 27(23), 4636-4641.
3. **Shine, J., & Dalgarno, L. (1974)**. *The 3'-terminal sequence of Escherichia coli 16S ribosomal RNA: complementarity to nonsense triplets and ribosome binding sites*. Proceedings of the National Academy of Sciences, 71(4), 1342-1346.
4. **Durbin, R., Eddy, S. R., Krogh, A., & Mitchison, G. (1998)**. *Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids*. Cambridge University Press.

