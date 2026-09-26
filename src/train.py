import numpy as np


# Constant representation of 0 probability
NEG_INF = -1e9


def train_supervised_4state_hmm(train_seq: str, train_labels: np.ndarray, alphabet: str = "ACGT"):
    """
    Trains HMM with 4 states:
        0: Intergenic
        1: Codon Pos 1 (C1)
        2: Codon Pos 1 (C2)
        3: Codon Pos 1 (C3)
    """
    n_states = 4
    char_map = {char: i for i, char in enumerate(alphabet)}

    # 1. Emission frequencies: P(Nucleotide | State) 
    emiss_counts = np.zeros((n_states, len(alphabet)), dtype=np.float64)
    for char, state in zip(train_seq, train_labels):
        if char in char_map:
            emiss_counts[state, char_map[char]] += 1.0

    # Laplace smoothing (+1)
    emiss_counts += 1.0
    emiss_probs = emiss_counts / emiss_counts.sum(axis=1, keepdims=True)
    log_emiss = np.log(emiss_probs)

    # 2. Structural Transition Matrix
    # Enforce zero-probability transitions via NEG_INF
    log_trans = np.full((n_states, n_states), NEG_INF, dtype=np.float64)

    # Count transitions empirically for valid paths
    trans_counts = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(len(train_labels)-1):
        s_from = train_labels[i]
        s_to = train_labels[i+1]
        trans_counts[s_from, s_to] += 1.0

    # State 0: Can stay in 0 or enter gene at State 1 (C1)
    total_0 = trans_counts[0, 0] + trans_counts[0, 1] + 2.0
    log_trans[0, 0] = np.log((trans_counts[0, 0] + 1.0) / total_0)
    log_trans[0, 1] = np.log((trans_counts[0, 1] + 1.0) / total_0)

    # State 1 (C1): Must transition to State 2 (C2)
    log_trans[1, 2] = 0.0

    # State 2 (C2): Must transition to State 3 (C3)
    log_trans[2, 3] = 0.0

    # State 3 (C3): Loops back to State 1 (C1) or exits to State 0
    total_3 =  trans_counts[3, 1] + trans_counts[3, 0] + 2.0
    log_trans[3, 1] = np.log((trans_counts[3, 1] + 1.0) / total_3)
    log_trans[3, 0] = np.log((trans_counts[3, 0] + 1.0) / total_3)

    # 3. Initial Probabilities
    log_initial = np.full(n_states, NEG_INF, dtype=np.float64)
    log_initial[0] = np.log(0.9)         # Bias slightly towards starting intergenic 
    log_initial[1] = np.log(0.1)

    return log_initial, log_trans, log_emiss
