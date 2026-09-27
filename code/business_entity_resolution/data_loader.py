"""Loads and validates the TSV files for the entity resolution task."""

import logging
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _read_tsv(path, name):
    """Read a tab-separated file with explicit separator."""
    logger.info("Loading %s from %s", name, path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    # Fill any residual NAs and strip whitespace
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    logger.info("  -> %d records", len(df))
    return df[REQUIRED_COLUMNS].reset_index(drop=True)


def load_train_sources():
    """Return (source1, source2, source3) DataFrames for training."""
    s1 = _read_tsv(config.TRAIN_SOURCE1, "train_source1")
    s2 = _read_tsv(config.TRAIN_SOURCE2, "train_source2")
    s3 = _read_tsv(config.TRAIN_SOURCE3, "train_source3")
    return s1, s2, s3


def load_test_sources():
    """Return (source1, source2, source3) DataFrames for testing."""
    s1 = _read_tsv(config.TEST_SOURCE1, "test_source1")
    s2 = _read_tsv(config.TEST_SOURCE2, "test_source2")
    s3 = _read_tsv(config.TEST_SOURCE3, "test_source3")
    return s1, s2, s3


def load_ground_truth():
    """Return a dict {source1_entity_id: set(matched_entity_ids)}."""
    logger.info("Loading ground truth from %s", config.TRAIN_GROUND_TRUTH)
    df = pd.read_csv(
        config.TRAIN_GROUND_TRUTH, sep="\t", dtype=str, keep_default_na=False
    )
    df.columns = [c.strip() for c in df.columns]

    if "source1_entity_id" not in df.columns or "matched_entity_ids" not in df.columns:
        raise ValueError(
            "Ground truth file must contain 'source1_entity_id' and 'matched_entity_ids'"
        )

    gt = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        raw = row["matched_entity_ids"].strip()
        if not raw:
            gt[s1_id] = set()
        else:
            gt[s1_id] = {x.strip() for x in raw.split(",") if x.strip()}
    logger.info("  -> %d ground truth entries", len(gt))
    return gt


def combine_sources(s2, s3):
    """Stack source2 and source3 records into one lookup DataFrame."""
    combined = pd.concat([s2, s3], ignore_index=True)
    return combined.reset_index(drop=True)