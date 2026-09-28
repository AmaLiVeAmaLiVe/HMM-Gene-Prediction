import numpy as np


NEG_INF = -1e9


class HMMGenePredictor2ndOrder:
    def __init__(self, states, log_initial, log_trans, log_emiss, log_emiss_0th, alphabet="ACGT"):
        self.states = states
        self.n_states = len(states)
        self.log_initial = log_initial
        self.log_trans = log_trans
        self.log_emiss = log_emiss          # Shape: (n_states, 4, 4, 4)
        self.log_emiss_0th = log_emiss_0th  # Shape: (n_states, 4)
        self.alphabet = alphabet
        self.char_map = {char: idx for idx, char in enumerate(alphabet)}
        
        # Valid biological signals in bacteria (forward strand)
        self.start_codons = {"ATG", "GTG", "TTG"}
        self.stop_codons = {"TAA", "TAG", "TGA"}

    def viterbi(self, sequence: str):
        T = len(sequence)
        K = self.n_states
        obs = np.array([self.char_map.get(ch, 0) for ch in sequence], dtype=np.int32)

        v_table = np.full((T, K), NEG_INF, dtype=np.float64)
        backpointer = np.zeros((T, K), dtype=np.int32)

        # Base case t = 0
        v_table[0, :] = self.log_initial + self.log_emiss_0th[:, obs[0]]

        # Dynamic programming forward pass
        for t in range(1, T):
            curr_base = obs[t]

            # 2nd-order emission lookup
            if t == 1:
                emiss_col = self.log_emiss_0th[:, curr_base]
            else:
                p2 = obs[t - 2]
                p1 = obs[t - 1]
                emiss_col = self.log_emiss[:, p2, p1, curr_base]

            for j in range(K):
                best_val = NEG_INF
                best_prev = 0

                for i in range(K):
                    trans_score = self.log_trans[i, j]
                    if trans_score <= NEG_INF / 2:
                        continue

                    # Rule 1: Entering a gene (0 -> C1)
                    # The upcoming codon starting at t must be a valid start codon
                    if i == 0 and j == 1:
                        if t + 2 < T:
                            start_trip = sequence[t : t + 3]
                            if start_trip not in self.start_codons:
                                continue
                        else:
                            continue

                    # Rule 2: Exiting a gene (C3 -> 0)
                    # At index t (first intergenic base), the codon that finished at t-1 was sequence[t-3:t]
                    if i == 3 and j == 0:
                        if t >= 3:
                            last_codon = sequence[t - 3 : t]
                            if last_codon not in self.stop_codons:
                                continue
                        else:
                            continue

                    # Rule 3: Continuing inside a gene (C3 -> C1)
                    # The codon that just completed cannot be a stop codon
                    if i == 3 and j == 1:
                        if t >= 3:
                            last_codon = sequence[t - 3 : t]
                            if last_codon in self.stop_codons:
                                continue

                    score = v_table[t - 1, i] + trans_score
                    if score > best_val:
                        best_val = score
                        best_prev = i

                v_table[t, j] = best_val + emiss_col[j]
                backpointer[t, j] = best_prev

        # Traceback
        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(v_table[-1, :])

        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return [self.states[idx] for idx in best_path]