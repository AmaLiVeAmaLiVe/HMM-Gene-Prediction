import os
import numpy as np

from src.data_loader import download_genome_data, parse_genome_and_labels_4state
from src.hmm_model import HMMGenePredictor2ndOrder
from src.utils import get_table_config


# Constant representation of 0 probability
NEG_INF = -1e9


def extract_unambiguous_orfs(sequence: str, table_id: int = 11, min_len_bp: int = 500):
    """
    Scans forward and reverse strands of an unannotated sequence for long ORFs.
    In prokaryotes, ORFs >= 500 bp without internal stop codons are statistically
    almost certain (>99%) to represent authentic protein-coding genes.
    """
    config = get_table_config(table_id)
    stop_codons = config["stop_codons"]
    start_codons = config["start_codons"]

    seq = sequence.upper()
    L = len(seq)
    candidate_orfs = []

    # 1. Forward 3 frames
    for frame in range(3):
        in_orf = False
        orf_start = 0
        for i in range(frame, L - 2, 3):
            codon = seq[i:i + 3]
            if not in_orf and codon in start_codons:
                in_orf = True
                orf_start = i
            elif in_orf and codon in stop_codons:
                orf_end = i + 3
                if (orf_end - orf_start) >= min_len_bp:
                    candidate_orfs.append((orf_start, orf_end, 1))
                in_orf = False

    return candidate_orfs


def dynamic_self_train(sequence: str, table_id: int = 11, min_len_bp: int = 500):
    """
    Unsupervised ab initio parameter estimation directly from input sequence.
    Mines long unambiguous ORFs, builds empirical 4-state labels, and derives
    2nd-order emission and transition matrices on the fly.
    """
    seq = sequence.upper()
    L = len(seq)

    long_orfs = extract_unambiguous_orfs(seq, table_id=table_id, min_len_bp=min_len_bp)

    # Need sufficient statistical power to estimate 2nd-order frequencies (64 x 4 cells)
    if len(long_orfs) < 20:
        raise ValueError(
            f"Insufficient long ORFs found ({len(long_orfs)} found, need at least 20 of length >= {min_len_bp} bp). "
            f"Input sequence is too short for reliable self-training."
        )

    # Create 4-state label array from confident coding regions
    labels = np.zeros(L, dtype=np.int32)
    for start, end, strand in long_orfs:
        for pos in range(start, end):
            codon_pos = (pos - start) % 3
            labels[pos] = codon_pos + 1  # 1=C1, 2=C2, 3=C3

    # Derive 2nd-order parameters using existing trainer
    log_initial, log_trans, log_emiss, log_emiss_0th = train_2nd_order_hmm(seq, labels)

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

    return model, len(long_orfs)


def train_2nd_order_hmm(train_seq: str, train_labels: np.ndarray, alphabet: str = "ACGT"):
    """
    Trains a 4-state HMM with the 2nd-order Markov emission:
        States: 0: Intergenic, 1: C1, 2: C2, 3: C3
        Emissions: P(X_t | X_{t-2}, X_{t-1}, Curr_state)
    """
    n_states = 4
    n_chars = len(alphabet)
    char_map = {char: i for i, char in enumerate(alphabet)}

    # Map raw string to integer array for fast indexing
    obs = np.array([char_map.get(ch, 0) for ch in train_seq], dtype=np.int32)
    N = len(obs)

    # 1. Estimate 2nd-Order Emissions: shape(n_states, 4, 4, 4)
    # emiss_counts[state, prev2, prev1, curr]
    emiss_counts = np.zeros((n_states, n_chars, n_chars, n_chars), dtype=np.float64)

    for t in range (2, N):
        s = train_labels[t]
        p2 = obs[t-2]
        p1 = obs[t-1]
        c = obs[t]
        emiss_counts[s, p2, p1, c] += 1.0

    # Laplace smoothing (+1 across the 4 possible current bases)
    emiss_counts += 1.0
    emiss_sums = emiss_counts.sum(axis=-1, keepdims=True)
    log_emiss = np.log(emiss_counts/emiss_sums)

    # Fallback 0th-order emissions for the boundary bases t=0 and t=1
    base_counts = np.zeros((n_states, n_chars), dtype=np.float64) + 1.0
    for t in range(N):
        base_counts[train_labels[t], obs[t]] += 1.0
    log_emiss_0th = np.log(base_counts / base_counts.sum(axis=-1, keepdims=True))

    # 2. Structural Transition Matrix (4-state cyclic grammar)
    # Enforce zero-probability transitions via NEG_INF
    log_trans = np.full((n_states, n_states), NEG_INF, dtype=np.float64)
    trans_counts = np.zeros((n_states, n_states), dtype=np.float64)

    for t in range(N-1):
        trans_counts[train_labels[t], train_labels[t+1]] += 1.0 

    # State 0 (Intergenic) -> Stay in 0 or enter C1
    total_0 = trans_counts[0, 0] + trans_counts[0, 1] + 2.0
    log_trans[0, 0] = np.log((trans_counts[0, 0] + 1.0) / total_0)
    log_trans[0, 1] = np.log((trans_counts[0, 1] + 1.0) / total_0) - 2.0

    # State 1 (C1) -> Must go to C2
    log_trans[1, 2] = 0.0

    # State 2 (C2) -> Must go to C3
    log_trans[2, 3] = 0.0

    # State 3 (C3) -> Loop to C1 or exit to 0
    total_3 =  trans_counts[3, 1] + trans_counts[3, 0] + 2.0
    log_trans[3, 1] = np.log((trans_counts[3, 1] + 1.0) / total_3)
    log_trans[3, 0] = np.log((trans_counts[3, 0] + 1.0) / total_3)

    # 3. Initial Probabilities
    log_initial = np.full(n_states, NEG_INF, dtype=np.float64)
    log_initial[0] = np.log(0.9)         # Bias slightly towards starting intergenic 
    log_initial[1] = np.log(0.1)

    return log_initial, log_trans, log_emiss, log_emiss_0th


def train_and_save_model(
    reference_accession: str = "NC_000913.3",
    output_path: str = "models/ecoli_model.npz"
) -> str:
    """
    Downloads reference genome data, trains 2nd-order HMM parameters,
    and serializes them to a compressed .npz archive.
    """
    print(f"[*] Training model using reference genome: {reference_accession}...")
    gbk_path = download_genome_data(reference_accession, output_dir="data/raw")
    ref_dna, ref_labels, _ = parse_genome_and_labels_4state(gbk_path)

    log_initial, log_trans, log_emiss, log_emiss_0th = train_2nd_order_hmm(ref_dna, ref_labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        log_initial=log_initial,
        log_trans=log_trans,
        log_emiss=log_emiss,
        log_emiss_0th=log_emiss_0th
    )
    print(f"[+] Model weights successfully written to: {output_path}")

    return output_path


def load_trained_model(model_path: str = "models/ecoli_model.npz", table_id: int = 11) -> HMMGenePredictor2ndOrder:
    """
    Loads pre-trained HMM parameters from disk and initializes the predictor 
    with the selected translation table.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pre-trained model not found at '{model_path}'. "
            f"Run training first or pass --train."
        )

    print(f"[*] Loading pre-trained model from: {model_path}")
    data = np.load(model_path)

    states = ["INTERGENIC", "C1", "C2", "C3"]
    return HMMGenePredictor2ndOrder(
        states=states,
        log_initial=data["log_initial"],
        log_trans=data["log_trans"],
        log_emiss=data["log_emiss"],
        log_emiss_0th=data["log_emiss_0th"],
        table_id=table_id
    )
