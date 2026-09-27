"""Evaluation metrics, thresholding, and submission formatting helpers."""

import logging
from typing import Dict, List, Set

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def f_beta(precision, recall, beta=config.BETA):
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def per_entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    """F_0.5 for a single Source 1 entity."""
    if not true_set and not pred_set:
        return 1.0
    if not true_set and pred_set:
        return 0.0
    if not pred_set and true_set:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return f_beta(precision, recall, config.BETA)


def macro_f05(predictions: Dict[str, Set[str]],
              ground_truth: Dict[str, Set[str]]) -> float:
    """
    Compute macro-averaged F_0.5 over all Source 1 entities in the ground
    truth. Predictions missing from `predictions` are treated as empty.
    """
    if not ground_truth:
        return 0.0
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(per_entity_f05(pred_set, true_set))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------
def find_best_threshold(candidates_scores: List[np.ndarray],
                        candidates_ids: List[List[str]],
                        ground_truth: Dict[str, Set[str]],
                        s1_ids: List[str],
                        thresholds=None) -> float:
    """
    Given per-source1 score arrays and their candidate ID lists, pick the
    threshold that maximises macro F_0.5 on the provided ground truth.
    """
    thresholds = thresholds or config.THRESHOLD_GRID
    best_thr, best_score = 0.5, -1.0

    for thr in thresholds:
        preds = {}
        for i, s1_id in enumerate(s1_ids):
            ids = candidates_ids[i]
            scores = candidates_scores[i]
            matched = {ids[j] for j in range(len(ids)) if scores[j] >= thr}
            preds[s1_id] = matched
        score = macro_f05(preds, ground_truth)
        if score > best_score:
            best_score = score
            best_thr = thr
    logger.info("Best threshold = %.3f (F_0.5 = %.4f)", best_thr, best_score)
    return best_thr


# ---------------------------------------------------------------------------
# Submission formatting
# ---------------------------------------------------------------------------
def format_id_list(ids):
    """Return a comma-separated string of unique ids preserving order."""
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return ",".join(out)


def build_matching_results(s1_records, candidates, candidates_scores,
                           threshold: float):
    """
    Return list of (source1_entity_id, matched_entity_ids_string).
    """
    rows = []
    for i, s1_rec in enumerate(s1_records):
        s1_id = s1_rec["entity_id"]
        ids = candidates[i]      # list of entity_id strings
        scores = candidates_scores[i]
        matched = [ids[j] for j in range(len(ids)) if scores[j] >= threshold]
        rows.append((s1_id, format_id_list(matched)))
    return rows


def build_candidate_pairs(s1_records, candidates):
    """Return list of (source1_entity_id, candidate_entity_ids_string)."""
    rows = []
    for i, s1_rec in enumerate(s1_records):
        s1_id = s1_rec["entity_id"]
        rows.append((s1_id, format_id_list(candidates[i])))
    return rows


def write_tsv(rows, path, header=("source1_entity_id", "matched_entity_ids")):
    """Write rows as a tab-separated file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for a, b in rows:
            f.write(f"{a}\t{b}\n")
    logger.info("Wrote %d rows to %s", len(rows), path)