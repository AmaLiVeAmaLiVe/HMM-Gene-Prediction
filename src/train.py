import numpy as np


# Constant representation of 0 probability
NEG_INF = -1e9


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
