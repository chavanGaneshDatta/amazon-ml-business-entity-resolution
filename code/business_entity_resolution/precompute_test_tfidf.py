"""Precompute TF-IDF matrices for the test S2/S3 records.

Saves:
  output/test_s23_tfidf_names.npz   — sparse matrix for names
  output/test_s23_tfidf_addrs.npz   — sparse matrix for addresses
  output/test_s23_records.pkl       — prepared record dicts (for blocking)
  output/test_s23_entity_ids.pkl    — list of entity_ids in order

Then predict_test.py just loads these and skips the transform entirely.
"""

import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    from code.business_entity_resolution import config, preprocessing
else:
    from . import config, preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("precompute")


def main():
    t0 = time.time()
    output_dir = config.OUTPUT_DIR

    # ---- Load TF-IDF vectorizers (fitted on training) ---------------------
    logger.info("Loading TF-IDF vectorizers ...")
    with open(output_dir / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # ---- Load S2 --------------------------------------------------------
    logger.info("Loading test_source2 ...")
    s2 = pd.read_csv(config.TEST_SOURCE2, sep="\t", dtype=str, keep_default_na=False)
    logger.info("  -> %d records", len(s2))

    logger.info("Loading test_source3 ...")
    s3 = pd.read_csv(config.TEST_SOURCE3, sep="\t", dtype=str, keep_default_na=False)
    logger.info("  -> %d records", len(s3))

    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    logger.info("Combined: %d records", len(s23))

    # ---- Prepare records ---------------------------------------------------
    logger.info("Preparing records (normalisation) ...")
    t1 = time.time()
    s23_records = preprocessing.prepare_records(s23)
    logger.info("  Prepared in %.1f sec", time.time() - t1)

    # Save records for later blocking use
    logger.info("Saving records to disk ...")
    with open(output_dir / "test_s23_records.pkl", "wb") as f:
        pickle.dump(s23_records, f, protocol=4)
    logger.info("  Saved test_s23_records.pkl")
        # Precompute and save blocking keys + inverted index
    logger.info("Precomputing blocking keys for all S2/S3 records ...")
    t1 = time.time()
    s23_keys = [preprocessing.generate_blocking_keys(r) for r in s23_records]
    logger.info("  Keys computed in %.1f sec", time.time() - t1)

    # Build the inverted index once
    logger.info("Building inverted index ...")
    from collections import defaultdict
    index = defaultdict(list)
    for idx, keys in enumerate(s23_keys):
        for k in keys:
            index[k].append(idx)
    logger.info("  Index built with %d keys", len(index))

    logger.info("Saving blocking keys and index to disk ...")
    with open(output_dir / "test_s23_blocking.pkl", "wb") as f:
        pickle.dump({"keys": s23_keys, "index": dict(index)}, f, protocol=4)
    logger.info("  Saved test_s23_blocking.pkl")

    # ---- Transform names ---------------------------------------------------
    logger.info("Transforming names through TF-IDF (this takes a while) ...")
    t1 = time.time()
    names = [r["name"] for r in s23_records]
    name_matrix = tfidf.name_vec.transform(names)
    del names
    logger.info("  Name matrix: %s in %.1f sec", name_matrix.shape, time.time() - t1)

    # Save name matrix
    logger.info("Saving name matrix ...")
    save_npz(str(output_dir / "test_s23_tfidf_names.npz"), name_matrix)
    logger.info("  Saved test_s23_tfidf_names.npz")
    del name_matrix

    # ---- Transform addresses ----------------------------------------------
    logger.info("Transforming addresses through TF-IDF ...")
    t1 = time.time()
    addrs = [r["address"] for r in s23_records]
    addr_matrix = tfidf.addr_vec.transform(addrs)
    del addrs
    logger.info("  Addr matrix: %s in %.1f sec", addr_matrix.shape, time.time() - t1)

    logger.info("Saving addr matrix ...")
    save_npz(str(output_dir / "test_s23_tfidf_addrs.npz"), addr_matrix)
    logger.info("  Saved test_s23_tfidf_addrs.npz")
    del addr_matrix

    # Done
    logger.info("=" * 60)
    logger.info("PRECOMPUTE DONE in %.1f min", (time.time() - t0) / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()