"""LightGBM learning-to-rank model for entity resolution."""

import logging
import numpy as np
import lightgbm as lgb

from . import config

logger = logging.getLogger(__name__)


def build_labels(groups, pair_meta, s1_records, s23_records, ground_truth):
    """
    Build binary labels for training pairs.

    A pair is label 1 if s23_records[s23_idx]['entity_id'] is in
    ground_truth[s1_records[s1_idx]['entity_id']].
    """
    labels = np.zeros(sum(groups), dtype=np.int8)
    pos = 0
    n_pos = 0
    for s1_idx, n in enumerate(groups):
        s1_id = s1_records[s1_idx]["entity_id"]
        gt_set = ground_truth.get(s1_id, set())
        for _ in range(n):
            _, s23_idx = pair_meta[pos]
            s23_id = s23_records[s23_idx]["entity_id"]
            if s23_id in gt_set:
                labels[pos] = 1
                n_pos += 1
            pos += 1
    logger.info("Training labels: %d positives out of %d pairs", n_pos, len(labels))
    return labels


def train_model(X_train, y_train, groups_train,
                X_val=None, y_val=None, groups_val=None,
                params=None):
    """Train a LightGBM lambdarank model."""
    params = params or config.LGBM_PARAMS

    print(f">>> Starting LightGBM training")
    print(f">>> X_train shape: {X_train.shape}")
    print(f">>> y_train shape: {y_train.shape}, positives: {int(y_train.sum())}")
    print(f">>> groups_train: {len(groups_train)} groups, sizes sum: {sum(groups_train)}")
    print(f">>> params: {params}")

    train_set = lgb.Dataset(
        X_train, label=y_train, group=groups_train, free_raw_data=False
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=50)]

    if X_val is not None and len(X_val) > 0 and sum(groups_val) > 0:
        val_set = lgb.Dataset(
            X_val, label=y_val, group=groups_val,
            reference=train_set, free_raw_data=False
        )
        valid_sets.append(val_set)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(config.EARLY_STOPPING_ROUNDS))

    # If no positives exist at all we cannot train lambdarank.
    if y_train.sum() == 0:
        raise ValueError("No positive pairs in training data; cannot train ranker.")

    model = lgb.train(
        params,
        train_set,
        num_boost_round=params.get("n_estimators", 500),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def predict_scores(model, X):
    """Return raw ranking scores for each candidate pair."""
    if X.shape[0] == 0:
        return np.array([], dtype=np.float32)
    return model.predict(X, num_iteration=model.best_iteration or None)


def aggregate_scores_by_source1(scores, groups):
    """
    Given a flat array of scores and group sizes, return a list of
    per-source1 arrays of scores (same order as candidates).
    """
    out = []
    pos = 0
    for n in groups:
        out.append(scores[pos:pos + n])
        pos += n
    return out