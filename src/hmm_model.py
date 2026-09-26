import numpy as np


class HMMGenePredictor():
    def __init__(self, states, log_initial, log_trans, log_emiss, alphabet="ACGT"):
        self.states = states
        self.n_states = len(states)
        self.log_initial = log_initial              # Shpae: (K, )
        self.log_trans = log_trans                  # Shape: (K, K)
        self.log_emiss = log_emiss                  # Shape: (K, |Alphabet|)
        self.char_map = {char: idx for idx, char in enumerate(alphabet)}

    def viterbi(self, sequence: str):
        T = len(sequence)
        K = self.n_states

        # Map string sequence to integers
        obs = np.array([self.char_map.get(ch, 0) for ch in sequence], dtype=np.int32)

        # DP tables: v_table[t, k], backpointer[t, k]
        v_table = np.full((T, K), -np.inf)
        backpointer = np.zeros((T, K), dtype=np.int32)

        # Base case t = 0
        v_table[0, :] = self.log_initial + self.log_emiss[:, obs[0]]

        # DP forward pass
        for t in range(1, T):
            emission_col =  self.log_emiss[:, obs[t]]

            # Broadcasting: (K, 1) + (K, K) -> (K, K) where (i, j) is i -> j
            trans_scores = v_table[t-1, :][:, np.newaxis] + self.log_trans
            best_prev = np.argmax(trans_scores, axis=0)

            backpointer[t, :] = best_prev
            v_table[t, :] = trans_scores[best_prev, np.arange(K)] + emission_col

        # Traceback
        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(v_table[-1, :])

        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        return [self.states [idx] for idx in best_path]
    