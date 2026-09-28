import numpy as np
import pytest

from src.hmm_model import HMMGenePredictor2ndOrder


def test_viterbi_synthetic_gc_island():
    # 2 states: 0 = Low GC (Intergenic), 1 = High GC (Coding)
    states = ["INTERGENIC", "CODING"]

    # Prior: Equal probability 
    log_initial = np.log([0.5, 0.5])

    # Transitions: High chance to stay in the current state
    trans = np.array([
        [0.90, 0.10],     # Intergenic -> Intergenic, Coding
        [0.10, 0.90]      # Coding -> Intergenic, Coding
    ])
    log_trans = np.log(trans)

    # Emissions: Order is [A, C, G, T]
    # State 0 (Intergenic): AT-rich (80% AT, 20% GC)
    # State 1 (Coding): GC-rich (20% AT, 80% GC)
    emiss = np.array([
        [0.40, 0.10, 0.10, 0.40],
        [0.10, 0.40, 0.40, 0.10]
    ])
    log_emiss = np.log(emiss)

    model = HMMGenePredictor2ndOrder(
        states=states,
        log_initial=log_initial,
        log_trans=log_trans,
        log_emiss=log_emiss,
        alphabet="ACGT"
    )

    # Sequence: 15 bases AT-rich, 15 bases GC-rich, 15 bases AT-rich
    toy_dna = "ATTTATATAAATATA" + "GCGCGGCCGGCGCGC" + "ATATTTATATATATA"

    predicted_path = model.viterbi(toy_dna)

    # Verify the middle segment was identified as Coding
    middle_segment = predicted_path[15:30]
    assert middle_segment.count("CODING") >= 12, "Viterbi failed to detect the GC-rich island"
    