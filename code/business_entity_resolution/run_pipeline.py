"""End-to-end pipeline for the Amazon ML Challenge 2026 entity resolution task.

Usage:
    python -m code.business_entity_resolution.run_pipeline --mode full
    python -m code.business_entity_resolution.run_pipeline --mode train_only
"""

import argparse
import logging
import os
import pickle
import random
import sys

import numpy as np

# Allow direct execution via `python run_pipeline.py`
if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    from code.business_entity_resolution import (
        config, data_loader, preprocessing, feature_engineering, model, utils
    )
else:
    from . import (
        config, data_loader, preprocessing, feature_engineering, model, utils
    )

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def _scores_to_id_lists(candidates_idx, s23_records, scores_per_s1):
    id_lists = []
    for cand_idx, scores in zip(candidates_idx, scores_per_s1):
        ids = [s23_records[j]["entity_id"] for j in cand_idx]
        id_lists.append(ids)
    return id_lists


def filter_small_groups(groups, X, y):
    """Drop groups with < 2 members (lambdarank requires >= 2 per group)."""
    keep_idx = []
    new_groups = []
    pos = 0
    for n in groups:
        if n >= 2:
            keep_idx.extend(range(pos, pos + n))
            new_groups.append(n)
        pos += n
    if not keep_idx:
        return [], X[:0], y[:0]
    return new_groups, X[keep_idx], y[keep_idx]


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def run_training(seed=config.RANDOM_SEED):
    logger.info("=== Loading training data ===")
    s1_df, s2_df, s3_df = data_loader.load_train_sources()
    gt = data_loader.load_ground_truth()

    s1_records = preprocessing.prepare_records(s1_df, is_source1=True)
    s2_records = preprocessing.prepare_records(s2_df)
    s3_records = preprocessing.prepare_records(s3_df)
    s23_records = s2_records + s3_records

    # ---- Train / validation split at the Source-1 level ------------------
    rng = random.Random(seed)
    indices = list(range(len(s1_records)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * config.VALIDATION_SPLIT))
    val_idx = set(indices[:n_val])

    logger.info("Train Source1: %d, Val Source1: %d",
                len(s1_records) - n_val, n_val)

    # ---- TF-IDF: load existing (fixed vectorizer) or fit new ------------
    logger.info("Loading or fitting TF-IDF ...")
    tfidf_path = config.OUTPUT_DIR / "tfidf.pkl"
    if tfidf_path.exists():
        logger.info("  Using existing TF-IDF vectorizers from %s", tfidf_path)
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
    else:
        logger.info("  No existing vectorizers — fitting new ones on training corpus")
        tfidf = feature_engineering.fit_tfidf(s1_records, s23_records)
        with open(tfidf_path, "wb") as f:
            pickle.dump(tfidf, f)
        logger.info("  Saved new vectorizers to %s", tfidf_path)

    # ---- Candidate generation -------------------------------------------
    logger.info("Generating candidates on full training corpus")
    all_candidates = preprocessing.generate_candidates(s1_records, s23_records)

    # ---- Build feature matrix -------------------------------------------
    logger.info("Building feature matrix on full corpus")
    X_all, groups_all, pair_meta_all = feature_engineering.build_feature_matrix(
        s1_records, s23_records, all_candidates, tfidf
    )
    y_all = model.build_labels(groups_all, pair_meta_all,
                               s1_records, s23_records, gt)

    # ---- Split into train/val rows --------------------------------------
    train_mask = [i not in val_idx for i in range(len(s1_records))]
    train_groups, val_groups = [], []
    train_pair_idx, val_pair_idx = [], []
    pos = 0
    for s1_idx, n in enumerate(groups_all):
        if train_mask[s1_idx]:
            train_groups.append(n)
            train_pair_idx.extend(range(pos, pos + n))
        else:
            val_groups.append(n)
            val_pair_idx.extend(range(pos, pos + n))
        pos += n

    # ---- Filter singleton groups ---------------------------------------
    train_groups, X_train, y_train = filter_small_groups(
        train_groups,
        X_all[train_pair_idx] if train_pair_idx else X_all[:0],
        y_all[train_pair_idx] if train_pair_idx else y_all[:0],
    )
    val_groups, X_val, y_val = filter_small_groups(
        val_groups,
        X_all[val_pair_idx] if val_pair_idx else X_all[:0],
        y_all[val_pair_idx] if val_pair_idx else y_all[:0],
    )

    logger.info(
        "After filtering groups <2: Train %d groups / %d pairs (pos=%d), "
        "Val %d groups / %d pairs (pos=%d)",
        len(train_groups), len(X_train),
        int(y_train.sum()) if len(y_train) else 0,
        len(val_groups), len(X_val),
        int(y_val.sum()) if len(y_val) else 0,
    )

    if len(X_train) == 0:
        raise RuntimeError(
            "No training pairs after filtering. Blocking produced no "
            "groups with >= 2 candidates."
        )

    # ---- Train -----------------------------------------------------------
    logger.info("Training LightGBM ranker")
    lgbm_model = model.train_model(
        X_train, y_train, train_groups,
        X_val, y_val, val_groups,
    )

    # ---- Threshold tuning on validation ---------------------------------
    if len(X_val) > 0:
        val_s1_records = []
        val_candidates_idx = []
        for i in range(len(s1_records)):
            if i in val_idx and len(all_candidates[i]) >= 2:
                val_s1_records.append(s1_records[i])
                val_candidates_idx.append(all_candidates[i])

        val_scores = model.predict_scores(lgbm_model, X_val)
        val_scores_per_s1 = model.aggregate_scores_by_source1(val_scores, val_groups)
        val_ids_per_s1 = _scores_to_id_lists(
            val_candidates_idx, s23_records, val_scores_per_s1
        )
        val_s1_ids = [r["entity_id"] for r in val_s1_records]
        val_gt = {s1_id: gt.get(s1_id, set()) for s1_id in val_s1_ids}

        assert len(val_s1_ids) == len(val_scores_per_s1), (
            f"Val S1 count {len(val_s1_ids)} != scores count {len(val_scores_per_s1)}"
        )
        for i, (ids, scores) in enumerate(zip(val_ids_per_s1, val_scores_per_s1)):
            assert len(ids) == len(scores), (
                f"Group {i}: {len(ids)} ids vs {len(scores)} scores"
            )

        best_thr = utils.find_best_threshold(
            val_scores_per_s1, val_ids_per_s1, val_gt, val_s1_ids
        )
    else:
        logger.warning("No validation pairs; using default threshold 0.5")
        best_thr = 0.5

    # ---- Save model -----------------------------------------------------
    model_path = config.OUTPUT_DIR / "lgbm_model.txt"
    lgbm_model.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)

    with open(config.OUTPUT_DIR / "threshold.txt", "w") as f:
        f.write(str(best_thr))
    logger.info("Saved threshold (%.4f)", best_thr)

    return lgbm_model, best_thr, tfidf, s1_records, s23_records


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------
def run_prediction(lgbm_model, threshold, tfidf, s1_records, s23_records,
                   write_outputs=True):
    logger.info("=== Running prediction ===")
    logger.info("Generating test candidates")
    candidates_idx = preprocessing.generate_candidates(s1_records, s23_records)

    logger.info("Building test feature matrix")
    X_test, groups_test, _ = feature_engineering.build_feature_matrix(
        s1_records, s23_records, candidates_idx, tfidf
    )

    if X_test.shape[0] == 0:
        logger.warning("No test candidate pairs; all predictions will be singletons")

    scores = model.predict_scores(lgbm_model, X_test)
    scores_per_s1 = model.aggregate_scores_by_source1(scores, groups_test)

    candidates_ids = _scores_to_id_lists(
        candidates_idx, s23_records, scores_per_s1
    )

    matching_rows = utils.build_matching_results(
        s1_records, candidates_ids, scores_per_s1, threshold
    )
    candidate_rows = utils.build_candidate_pairs(s1_records, candidates_ids)

    if write_outputs:
        utils.write_tsv(
            matching_rows, config.MATCHING_RESULTS,
            header=("source1_entity_id", "matched_entity_ids"),
        )
        utils.write_tsv(
            candidate_rows, config.CANDIDATE_PAIRS,
            header=("source1_entity_id", "candidate_entity_ids"),
        )

    return matching_rows, candidate_rows


# ---------------------------------------------------------------------------
# Full train + predict on test (not used for --mode train_only)
# ---------------------------------------------------------------------------
def run_full(seed=config.RANDOM_SEED):
    _set_seeds(seed)

    try:
        lgbm_model, best_thr, _, _, _ = run_training(seed=seed)
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Best threshold: {best_thr}")
        print("=" * 60)
    except Exception:
        import traceback
        print("=" * 60)
        print("ERROR DURING TRAINING:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        raise

    logger.info("=== Loading test data ===")
    t1_df, t2_df, t3_df = data_loader.load_test_sources()
    t1_records = preprocessing.prepare_records(t1_df, is_source1=True)
    t2_records = preprocessing.prepare_records(t2_df)
    t3_records = preprocessing.prepare_records(t3_df)
    t23_records = t2_records + t3_records

    # Reuse the SAME vectorizer (fixed-vectorizer principle)
    tfidf_path = config.OUTPUT_DIR / "tfidf.pkl"
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    logger.info("Loaded TF-IDF vectorizers from %s", tfidf_path)

    matching_rows, candidate_rows = run_prediction(
        lgbm_model, best_thr, tfidf,
        t1_records, t23_records,
        write_outputs=True,
    )

    n_singletons = sum(1 for _, m in matching_rows if m == "")
    logger.info("Predicted singletons: %d / %d",
                n_singletons, len(matching_rows))
    logger.info("Done. Outputs written to %s", config.OUTPUT_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "train_only"], default="full")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.mode == "full":
        run_full(seed=args.seed)
    elif args.mode == "train_only":
        run_training(seed=args.seed)


if __name__ == "__main__":
    main()