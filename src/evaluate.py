import numpy as np


def extract_predicted_genes(predicted_states: list, min_length_bp: int = 60):
    """
    Groups continuous sequences of coding states (C1, C2, C3) into gene intervals.
    Returns: List of tuples (start_idx, end_idx, length)
    """
    genes = []
    in_gene = False
    start_pos = 0

    for i, state in enumerate(predicted_states):
        # Coding states are 1, 2, 3 (or string names)
        is_coding = state in [1, 2, 3, "C1", "C2", "C3"]

        if is_coding and not in_gene:
            in_gene = True
            start_pos = i
        elif not is_coding and in_gene:
            in_gene = False
            length = i - start_pos

            if length >= min_length_bp:
                genes.append((start_pos, i, length))

    if in_gene:
        length = len(predicted_states) - start_pos
        if length >= min_length_bp:
            genes.append((start_pos, len(predicted_states), length))

    return genes


def evaluate_gene_boundaries(test_cds, pred_genes, slack_bp: int = 6):
    """
    Evaluates predicted gene boundaries against ground truth CDS annotations.
    """
    n_true = len(test_cds)
    n_pred = len(pred_genes)
    
    if n_true == 0 or n_pred == 0:
        return {
            "Annotated Genes in Test Set": n_true,
            "Total Predicted Genes": n_pred,
            "Exact Boundary Matches (Start & Stop)": 0,
            "Start Codon Matches": 0,
            "Stop Codon Matches": 0,
            "Overlapping / Detected Genes": 0,
            "Gene Sensitivity (Overlap)": 0.0,
            "Gene Precision (Overlap)": 0.0,
        }

    # 1. Sensitivity perspective: which true genes were detected?
    detected_true = 0
    exact_matches = 0
    start_matches = 0
    stop_matches = 0

    for t_start, t_end in test_cds:
        t_hit = False
        for p_start, p_end, _ in pred_genes:
            # Overlap check
            overlap = min(t_end, p_end) - max(t_start, p_start)
            if overlap > 0:
                t_hit = True
                
            # Boundary checks (allowing small slack or exact 0)
            if abs(t_start - p_start) <= slack_bp and abs(t_end - p_end) <= slack_bp:
                exact_matches += 1
                start_matches += 1
                stop_matches += 1
                break
            elif abs(t_start - p_start) <= slack_bp:
                start_matches += 1
            elif abs(t_end - p_end) <= slack_bp:
                stop_matches += 1
                
        if t_hit:
            detected_true += 1

    # 2. Precision perspective: which predicted genes hit a real gene?
    valid_predictions = 0
    for p_start, p_end, _ in pred_genes:
        for t_start, t_end in test_cds:
            overlap = min(t_end, p_end) - max(t_start, p_start)
            if overlap > 0:
                valid_predictions += 1
                break  # Count this prediction at most once

    sensitivity = (detected_true / n_true) * 100.0
    precision = (valid_predictions / n_pred) * 100.0

    return {
        "Annotated Genes in Test Set": n_true,
        "Total Predicted Genes": n_pred,
        "Exact Boundary Matches (Start & Stop)": exact_matches,
        "Start Codon Matches": start_matches,
        "Stop Codon Matches": stop_matches,
        "Overlapping / Detected Genes": detected_true,
        "Valid Predictions (Hits True CDS)": valid_predictions,
        "Gene Sensitivity (Overlap)": sensitivity,
        "Gene Precision (Overlap)": precision,
    }


def evaluate_nucleotide_level(true_binary: np.ndarray, pred_binary: np.ndarray):
    tp = np.sum((true_binary == 1) & (pred_binary == 1))
    fp = np.sum((true_binary == 0) & (pred_binary == 1))
    tn = np.sum((true_binary == 0) & (pred_binary == 0))
    fn = np.sum((true_binary == 1) & (pred_binary == 0))

    sens = tp / (tp+fn) if (tp+fn) > 0 else 0.0          # Recall
    spec = tn / (tn+fp) if (tn+fp) > 0 else 0.0
    prec   = tp / (tp+fp) if (tp+fp) > 0 else 0.0

    f1 = 2 * (prec*sens) / (prec+sens) if (prec+sens) > 0 else  0.0
    acc = (tp+tn) / len(true_binary)

    return {
            "Nucleotide Sensitivity (Recall)": sens * 100,
            "Nucleotide Specificity": spec * 100,
            "Nucleotide Precision": prec * 100,
            "Nucleotide F1-Score": f1 * 100,
            "Nucleotide Accuracy": acc * 100,
            "Confusion Matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
        }


def compute_orf_lod_score(dna_segment: str, log_emiss_c3, log_emiss_0th) -> float:
    """
    Calculates the cumulative Log-Odds score comparing coding vs. non-coding likelihood
    for an in-frame triplet sequence
    """
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = dna_segment.upper()
    L = len(seq)

    if L % 3 != 0 or L < 6:
        return -999.0

    lod_total = 0.0

    # Iterate codon by codod
    for idx in range(0, L, 3):
        c1, c2, c3 = seq[idx], seq[idx+1], seq[idx+2]
        if any (base not in char_map for base in (c1, c2, c3)):
            continue

        i1, i2, i3 = char_map[c1], char_map[c2], char_map[c3]

        # P(Codon Base 3 | Base 1, Base 2, State 3)
        log_p_coding = log_emiss_c3[i1, i2, i3]

        # P(Codon Base 3 | Background State 0)
        log_p_bg = log_emiss_0th[0, i3]

        lod_total += (log_p_coding - log_p_bg)

    return lod_total


def filter_genes_by_lod(pred_genes, test_seq: str, log_emiss, log_emiss_0th, lod_threshold: float = 0.9):
    """
    Filters out predicted ORFs whose coding log-odds score doens not exceed the threshold
    """
    # log_emiss[3] Corresponds to State C3 (Codon Position 3)
    log_emiss_c3 = log_emiss[3]

    filtered = []
    for start, end, length in pred_genes:
        orf_seq = test_seq[start:end]
        score = compute_orf_lod_score(orf_seq, log_emiss_c3, log_emiss_0th)
        if score > lod_threshold:
            filtered.append((start, end, length, score))

    return filtered


def reverse_complement(dna: str) -> str:
    """Returns the reverse complement of a DNA string"""
    complement = str.maketrans("ACGTacgt", "TGCAtgca")

    return dna.translate(complement)[::-1]


def map_reverse_predictions(rev_pred_genes, seq_len: int):
    """
    Maps genes coordinates called on the reverse complement back 
    to the forward reference coordinates
    """
    mapped = []
    for r_start, r_end, length in rev_pred_genes:
        f_start = seq_len - r_end
        f_end = seq_len -  r_start

        mapped.append((f_start, f_end, length, -1))    # Strand = -1 (Reversed Gene Strand)

    return mapped


def resolve_strand_overlaps(fwd_genes, rev_genes, max_allowed_overlap: int = 15):
    """
    Mergres forward and reverse predicted genes, pruning spruious antisense 
    shadow calls where an ORF on one strand deeply overlaps an ORF on the other
    """
    # Annotate forward genes with strand = +1
    all_genes = [(s, e, l, 1) for s, e, l in fwd_genes] + rev_genes

    # Sort by start coordindate
    all_genes.sort(key=lambda g: g[0])

    kept = []
    for g in all_genes:
        if not kept:
            kept.append(g)
            continue
        prev = kept[-1]

        # Calculate overlap between consecutive predicted genes
        overlap = min(prev[1], g[1]) - max(prev[0], g[0])

        # If they are on the opposite strands and deeply overlap
        if prev[3] != g[3] and overlap > max_allowed_overlap:
            # Keep the longer ORF (longer ORFs have higher statistical validity)
            if g[2] > prev[2]:
                kept[-1] = g
        else:
            kept.append(g)

    return [(s, e ,l) for s, e, l, _ in kept]