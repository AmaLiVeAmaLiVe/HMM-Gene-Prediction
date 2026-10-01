import numpy as np

from src.utils import get_table_config


NEG_INF = -1e9


class HMMGenePredictor2ndOrder:
    def __init__(self, states, log_initial, log_trans, log_emiss, log_emiss_0th, table_id: int = 11, alphabet="ACGT"):
        self.states = states
        self.n_states = len(states)
        self.log_initial = log_initial
        self.log_trans = log_trans
        self.log_emiss = log_emiss          # Shape: (n_states, 4, 4, 4)
        self.log_emiss_0th = log_emiss_0th  # Shape: (n_states, 4)
        self.alphabet = alphabet
        self.char_map = {char: idx for idx, char in enumerate(alphabet)}


        # Configure boundary codons dynamically based on translation table
        self.table_id = table_id
        config = get_table_config(table_id)
        self.stop_codons = config["stop_codons"]
        self.start_codons = config["start_codons"]
        

    @staticmethod
    def calculate_rbs_bonus(upstream_window: str) -> float:
        """
        Scans the -15 to -4 bp region upstream of an ATG/GTG/TTG start codon
        for Shine-Dalgarno core consensus motifs.
        """
        # Core Shine-Dalgarno motifs and their empirical log-odds weights
        rbs_motifs = {
            "AGGAGG": 3.5,
            "GGAGG":  3.0,
            "AGGAG":  2.8,
            "GAGG":   2.2,
            "AGGA":   2.0,
            "GGAG":   1.8
        }
        
        bonus = 0.0
        for motif, weight in rbs_motifs.items():
            if motif in upstream_window:
                bonus = max(bonus, weight)
                
        return bonus

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
                p2 = obs[t-2]
                p1 = obs[t-1]
                emiss_col = self.log_emiss[:, p2, p1, curr_base]

            for j in range(K):
                best_val = NEG_INF
                best_prev = 0

                for i in range(K):
                    trans_score = self.log_trans[i, j]
                    if trans_score <= NEG_INF / 2:
                        continue

                    # Bonus/weight modifiers
                    bonus = 0.0

                    # Rule 1: Entering a gene (0 -> C1)
                    # The upcoming codon starting at t must be a valid start codon
                    if i == 0 and j == 1:
                        if t + 2 < T:
                            start_trip = sequence[t : t+3]
                            if start_trip not in self.start_codons:
                                continue

                            # Check upstream window (-15 to -4) for Shine-Dalgarno sequence
                            if t >= 15:
                                upstream = sequence[t-15 : t-4]
                                bonus += self.calculate_rbs_bonus(upstream)

                            # Favor canonical ATG over rarer GTG/TTG starts
                            start_weight = 1.0 if start_trip == "ATG" else 0.3
                            bonus += np.log(start_weight)
                        else:
                            continue

                    # Rule 2: Exiting a gene (C3 -> 0)
                    # At index t (first intergenic base), the codon that finished at t-1 was sequence[t-3:t]
                    if i == 3 and j == 0:
                        if t >= 3:
                            last_codon = sequence[t-3 : t]
                            if last_codon not in self.stop_codons:
                                continue
                        else:
                            continue

                    # Rule 3: Continuing inside a gene (C3 -> C1)
                    # The codon that just completed cannot be a stop codon
                    if i == 3 and j == 1:
                        if t >= 3:
                            last_codon = sequence[t-3 : t]
                            if last_codon in self.stop_codons:
                                continue

                    score = v_table[t-1, i] + trans_score + bonus
                    if score > best_val:
                        best_val = score
                        best_prev = i

                v_table[t, j] = best_val + emiss_col[j]
                backpointer[t, j] = best_prev

        # Traceback
        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(v_table[-1, :])

        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        return [self.states[idx] for idx in best_path]