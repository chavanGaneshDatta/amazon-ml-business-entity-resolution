"""
Diagnostic script for the Amazon ML Challenge 2026
Business Entity Resolution pipeline.

Measures:
1. Candidate recall
2. Effect of MAX_CANDIDATES_PER_S1
3. Candidate count distribution
4. Candidate recall by match type
5. Validation F0.5 across thresholds
6. Existing LightGBM model performance

Run from project root:

    python -m code.business_entity_resolution.diagnose
"""

import pickle
import random
from collections import Counter

import numpy as np

from . import config
from . import data_loader
from . import preprocessing
from . import feature_engineering
from . import model
from . import utils


# ============================================================
# Configuration
# ============================================================

SEED = config.RANDOM_SEED

# We generate candidates once with a very high cap.
# Because preprocessing.py sorts candidates before applying
# the cap, this lets us reproduce cap=15/50/100 exactly.
DIAGNOSTIC_CAP = 1000

CAPS_TO_TEST = [15, 50, 100]

THRESHOLDS = [
    i / 100.0
    for i in range(1, 100)
]


# ============================================================
# Helpers
# ============================================================

def split_s1(s1_records):
    """Reproduce the same S1-level split used by run_pipeline.py."""

    rng = random.Random(SEED)

    indices = list(range(len(s1_records)))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * config.VALIDATION_SPLIT))

    val_idx = set(indices[:n_val])

    return val_idx


def cap_candidates(candidates, cap):
    """Apply the same cap logic used by preprocessing.py."""

    return [
        cand[:cap]
        for cand in candidates
    ]


def get_true_matches(gt, s1_id):
    return gt.get(s1_id, set())


def candidate_recall(
    s1_records,
    candidates,
    gt,
    only_indices=None,
):
    """
    Candidate recall:

        true matches appearing in candidates
        ------------------------------------
              total true matches

    This is NOT the percentage of S1s having candidates.
    """

    if only_indices is None:
        indices = range(len(s1_records))
    else:
        indices = only_indices

    total_true = 0
    found_true = 0

    for i in indices:
        s1_id = s1_records[i]["entity_id"]

        true_ids = get_true_matches(gt, s1_id)
        candidate_ids = {
            # candidate indices are converted outside this function
            # when needed
        }

        total_true += len(true_ids)

    return total_true, found_true


def calculate_candidate_metrics(
    s1_records,
    s23_records,
    candidates,
    gt,
    indices,
):
    """Calculate true-match recall and match-type statistics."""

    s23_ids = [r["entity_id"] for r in s23_records]

    total_true = 0
    found_true = 0

    type_stats = {
        "singleton": {
            "s1": 0,
            "true_matches": 0,
            "found_matches": 0,
        },
        "one_match": {
            "s1": 0,
            "true_matches": 0,
            "found_matches": 0,
        },
        "many_match": {
            "s1": 0,
            "true_matches": 0,
            "found_matches": 0,
        },
    }

    candidate_counts = []

    for i in indices:

        s1_id = s1_records[i]["entity_id"]

        true_ids = gt.get(s1_id, set())

        candidate_ids = {
            s23_ids[j]
            for j in candidates[i]
        }

        found = true_ids & candidate_ids

        n_true = len(true_ids)

        total_true += n_true
        found_true += len(found)

        candidate_counts.append(len(candidates[i]))

        if n_true == 0:
            key = "singleton"
        elif n_true == 1:
            key = "one_match"
        else:
            key = "many_match"

        type_stats[key]["s1"] += 1
        type_stats[key]["true_matches"] += n_true
        type_stats[key]["found_matches"] += len(found)

    recall = (
        found_true / total_true
        if total_true > 0
        else 1.0
    )

    return {
        "total_true": total_true,
        "found_true": found_true,
        "recall": recall,
        "candidate_counts": candidate_counts,
        "type_stats": type_stats,
    }


def print_candidate_metrics(name, metrics):

    print()
    print("=" * 70)
    print(f"CANDIDATE RECALL: {name}")
    print("=" * 70)

    print(
        f"True matches:       {metrics['total_true']:,}"
    )

    print(
        f"Found in candidates:{metrics['found_true']:,}"
    )

    print(
        f"Candidate recall:   {metrics['recall'] * 100:.3f}%"
    )

    counts = metrics["candidate_counts"]

    if counts:

        arr = np.asarray(counts)

        print()
        print("Candidate count statistics:")
        print(f"  Mean:     {arr.mean():.2f}")
        print(f"  Median:   {np.median(arr):.2f}")
        print(f"  Min:      {arr.min()}")
        print(f"  Max:      {arr.max()}")
        print(f"  P90:      {np.percentile(arr, 90):.2f}")
        print(f"  P95:      {np.percentile(arr, 95):.2f}")
        print(f"  P99:      {np.percentile(arr, 99):.2f}")

    print()
    print("Recall by S1 match type:")

    for key, label in [
        ("singleton", "0 matches"),
        ("one_match", "1 match"),
        ("many_match", "2+ matches"),
    ]:

        s = metrics["type_stats"][key]

        if s["true_matches"] > 0:
            r = s["found_matches"] / s["true_matches"]
        else:
            r = 1.0

        print(
            f"  {label:<12} "
            f"S1={s['s1']:,}  "
            f"true={s['true_matches']:,}  "
            f"found={s['found_matches']:,}  "
            f"recall={r * 100:.3f}%"
        )


def calculate_macro_f05_for_threshold(
    s1_records,
    candidates,
    scores_per_s1,
    gt,
    threshold,
):
    """
    Calculate the actual macro F0.5 over ALL supplied S1s.

    Includes S1s with:
        0 candidates
        1 candidate
        2+ candidates
    """

    predictions = {}

    for i, s1_rec in enumerate(s1_records):

        s1_id = s1_rec["entity_id"]

        ids = candidates[i]
        scores = scores_per_s1[i]

        pred = {
            ids[j]
            for j in range(len(ids))
            if scores[j] >= threshold
        }

        predictions[s1_id] = pred

    return utils.macro_f05(predictions, gt)


def threshold_sweep(
    s1_records,
    candidates,
    scores_per_s1,
    gt,
):
    """Find best F0.5 threshold."""

    results = []

    for threshold in THRESHOLDS:

        score = calculate_macro_f05_for_threshold(
            s1_records,
            candidates,
            scores_per_s1,
            gt,
            threshold,
        )

        results.append((threshold, score))

    results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return results


def print_threshold_results(results):

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    print()
    print("Top 10 thresholds:")

    for threshold, score in results[:10]:

        print(
            f"  threshold={threshold:.2f} "
            f"F0.5={score:.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ENTITY RESOLUTION DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Loading training data...")

    s1_df, s2_df, s3_df = data_loader.load_train_sources()
    gt = data_loader.load_ground_truth()

    s1_records = preprocessing.prepare_records(
        s1_df,
        is_source1=True,
    )

    s2_records = preprocessing.prepare_records(s2_df)
    s3_records = preprocessing.prepare_records(s3_df)

    s23_records = s2_records + s3_records

    print(
        f"S1 records:       {len(s1_records):,}"
    )

    print(
        f"S2 records:       {len(s2_records):,}"
    )

    print(
        f"S3 records:       {len(s3_records):,}"
    )

    print(
        f"S2 + S3 records:  {len(s23_records):,}"
    )

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    val_idx = split_s1(s1_records)

    print()
    print(
        f"Validation S1:    {len(val_idx):,}"
    )

    # --------------------------------------------------------
    # Generate candidates with large cap
    # --------------------------------------------------------

    print()
    print(
        f"Generating candidates with diagnostic cap "
        f"{DIAGNOSTIC_CAP}..."
    )

    old_cap = preprocessing.MAX_CANDIDATES_PER_S1

    preprocessing.MAX_CANDIDATES_PER_S1 = DIAGNOSTIC_CAP

    try:

        all_candidates = preprocessing.generate_candidates(
            s1_records,
            s23_records,
        )

    finally:

        preprocessing.MAX_CANDIDATES_PER_S1 = old_cap

    # --------------------------------------------------------
    # Candidate recall experiments
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BLOCKING DIAGNOSTICS")
    print("=" * 70)

    val_indices = sorted(val_idx)

    for cap in CAPS_TO_TEST:

        capped = cap_candidates(
            all_candidates,
            cap,
        )

        metrics = calculate_candidate_metrics(
            s1_records,
            s23_records,
            capped,
            gt,
            val_indices,
        )

        print_candidate_metrics(
            f"CAP = {cap}",
            metrics,
        )

    # --------------------------------------------------------
    # Current cap=15
    # --------------------------------------------------------

    current_candidates = cap_candidates(
        all_candidates,
        15,
    )

    # --------------------------------------------------------
    # Load existing model
    # --------------------------------------------------------

    model_path = config.OUTPUT_DIR / "lgbm_model.txt"
    tfidf_path = config.OUTPUT_DIR / "tfidf.pkl"

    print()
    print("=" * 70)
    print("LOADING EXISTING MODEL")
    print("=" * 70)

    if not model_path.exists():

        print()
        print(
            f"ERROR: model not found:\n"
            f"{model_path}"
        )

        print()
        print(
            "Run the training pipeline first."
        )

        return

    if not tfidf_path.exists():

        print()
        print(
            f"ERROR: TF-IDF model not found:\n"
            f"{tfidf_path}"
        )

        return

    import lightgbm as lgb

    lgbm_model = lgb.Booster(
        model_file=str(model_path)
    )

    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)

    print("Loaded LightGBM model.")
    print("Loaded TF-IDF vectorizer.")

    # --------------------------------------------------------
    # Validation records
    # --------------------------------------------------------

    val_s1_records = [
        s1_records[i]
        for i in val_indices
    ]

    # Convert global candidate indices to local S23 indices.
    #
    # S23 is unchanged, so the indices remain valid.
    val_candidates = [
        current_candidates[i]
        for i in val_indices
    ]

    # --------------------------------------------------------
    # Build validation features
    # --------------------------------------------------------

    print()
    print("Building validation feature matrix...")

    X_val, groups_val, pair_meta = (
        feature_engineering.build_feature_matrix(
            val_s1_records,
            s23_records,
            val_candidates,
            tfidf,
        )
    )

    print(
        f"Validation pairs: {len(X_val):,}"
    )

    print(
        f"Validation groups: {len(groups_val):,}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print()
    print("Predicting validation scores...")

    scores = model.predict_scores(
        lgbm_model,
        X_val,
    )

    scores_per_s1 = model.aggregate_scores_by_source1(
        scores,
        groups_val,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # build_feature_matrix can contain groups of size 0.
    # aggregate_scores_by_source1 preserves those groups.
    # --------------------------------------------------------

    print(
        f"Score groups: {len(scores_per_s1):,}"
    )

    assert len(scores_per_s1) == len(val_s1_records)

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    val_gt = {
        r["entity_id"]: gt.get(
            r["entity_id"],
            set(),
        )
        for r in val_s1_records
    }

    threshold_results = threshold_sweep(
        val_s1_records,
        [
            [
                s23_records[j]["entity_id"]
                for j in cand
            ]
            for cand in val_candidates
        ],
        scores_per_s1,
        val_gt,
    )

    print_threshold_results(
        threshold_results
    )

    # --------------------------------------------------------
    # Current saved threshold
    # --------------------------------------------------------

    threshold_path = config.OUTPUT_DIR / "threshold.txt"

    if threshold_path.exists():

        with open(threshold_path) as f:
            saved_threshold = float(
                f.read().strip()
            )

        saved_score = calculate_macro_f05_for_threshold(
            val_s1_records,
            [
                [
                    s23_records[j]["entity_id"]
                    for j in cand
                ]
                for cand in val_candidates
            ],
            scores_per_s1,
            val_gt,
            saved_threshold,
        )

        print()
        print("=" * 70)
        print("SAVED THRESHOLD")
        print("=" * 70)

        print(
            f"Saved threshold: {saved_threshold:.3f}"
        )

        print(
            f"F0.5 at saved threshold: "
            f"{saved_score:.6f}"
        )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    best_threshold, best_score = threshold_results[0]

    print()
    print("=" * 70)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"Best validation threshold: {best_threshold:.2f}"
    )

    print(
        f"Best validation F0.5:       {best_score:.6f}"
    )

    print()
    print(
        "Compare the CAP=15 / CAP=50 / CAP=100 "
        "candidate recalls above."
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()