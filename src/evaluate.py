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


def evaluate_gene_boundaries(true_genes: list, pred_genes: list, slack_bp: int = 6):
    """
    Evaluates gene detection at the entity level:
        - Exact start matches (within slack_bp)
        - Exact stop matches (within slack_bp)
        - Both ends match (fully recovered gene)
        - Overlapping genes (overlap >= 50% reciprocal)
    """
    n_true = len(true_genes)
    n_pred = len(pred_genes)

    if n_true == 0 or n_pred == 0:
        return {"True Genes": n_true, "Predicted Genes": n_pred, 
                "Exact Genes": 0, "Partial Matches": 0}

    matched_exact = 0
    matched_starts = 0
    matched_stops = 0
    matched_overlap = 0

    for t_start, t_end in true_genes:
        has_overlap = False
        start_matched = False
        end_matched = False

        for p_start, p_end, _ in pred_genes:
            # Overlap check
            overlap = max(0, min(t_end, p_end) - max(t_start, p_start))
            if overlap > 0:
                has_overlap = True

            # Boundary checks with tolerance
            if abs(t_start - p_start) <= slack_bp:
                start_matched = True
            if abs(t_end - p_end) <= slack_bp:
                end_matched = True

        if has_overlap:
            matched_overlap += 1
        if start_matched:
            matched_starts += 1
        if end_matched:
            matched_stops += 1
        if start_matched and end_matched:
            matched_exact += 1

    return {
        "Annotated Genes in Test Set": n_true,
        "Total Predicted Genes": n_pred,
        "Exact Boundary Matches (Start & Stop)": matched_exact,
        "Start Codon Matches": matched_starts,
        "Stop Codon Matches": matched_stops,
        "Overlapping / Detected Genes": matched_overlap,
        "Gene Sensitivity (Overlap)": (matched_overlap / n_true) * 100 if n_true else 0.0,
        "Gene Precision (Overlap)": (matched_overlap / n_pred) * 100 if n_pred else 0.0
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


def print_evaluation_report(metrics: dict):
    print("\n" + "="*45)
    print("GENE PREDICTION EVALUATION REPORT")
    print("="*45)

    for key, val in metrics.items():
        if key == "Confusion Matrix":
            print("\nConfusion Matrix:")
            for cm_key, cm_val in val.items():
                print(f"{cm_key}: {cm_val:,}")
        else:
            print(f"{key:22}: {val*100:6.2f}%")

    print("="*45 + "\n")
