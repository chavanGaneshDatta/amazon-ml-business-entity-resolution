"""Optimized char n-gram precompute using sklearn CountVectorizer.

Uses C-optimized sparse matrices (not Python sets) → ~15-25 min for 10M records.
"""

import logging
import os
import pickle
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    from code.business_entity_resolution import config
else:
    from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("precompute_ngrams")


def main():
    t0 = time.time()
    output_dir = config.OUTPUT_DIR

    logger.info("Loading precomputed records ...")
    with open(output_dir / "test_s23_records.pkl", "rb") as f:
        s23_records = pickle.load(f)
    logger.info("  Loaded %d records", len(s23_records))

    names = [r["name"] for r in s23_records]
    addrs = [r["address"] for r in s23_records]

    # Name char 2-grams (binary)
    logger.info("Computing name char 2-grams ...")
    t1 = time.time()
    name_c2_vec = CountVectorizer(analyzer="char", ngram_range=(2, 2),
                                  binary=True, dtype=np.float32, min_df=1)
    name_c2_matrix = name_c2_vec.fit_transform(names)
    logger.info("  Shape: %s, %.1f sec", name_c2_matrix.shape, time.time() - t1)

    logger.info("Computing name char 3-grams ...")
    t1 = time.time()
    name_c3_vec = CountVectorizer(analyzer="char", ngram_range=(3, 3),
                                  binary=True, dtype=np.float32, min_df=1)
    name_c3_matrix = name_c3_vec.fit_transform(names)
    logger.info("  Shape: %s, %.1f sec", name_c3_matrix.shape, time.time() - t1)

    logger.info("Computing address char 2-grams ...")
    t1 = time.time()
    addr_c2_vec = CountVectorizer(analyzer="char", ngram_range=(2, 2),
                                  binary=True, dtype=np.float32, min_df=1)
    addr_c2_matrix = addr_c2_vec.fit_transform(addrs)
    logger.info("  Shape: %s, %.1f sec", addr_c2_matrix.shape, time.time() - t1)

    logger.info("Computing address char 3-grams ...")
    t1 = time.time()
    addr_c3_vec = CountVectorizer(analyzer="char", ngram_range=(3, 3),
                                  binary=True, dtype=np.float32, min_df=1)
    addr_c3_matrix = addr_c3_vec.fit_transform(addrs)
    logger.info("  Shape: %s, %.1f sec", addr_c3_matrix.shape, time.time() - t1)

    logger.info("Saving matrices ...")
    t1 = time.time()
    save_npz(str(output_dir / "test_s23_ngrams_name_c2.npz"), name_c2_matrix)
    save_npz(str(output_dir / "test_s23_ngrams_name_c3.npz"), name_c3_matrix)
    save_npz(str(output_dir / "test_s23_ngrams_addr_c2.npz"), addr_c2_matrix)
    save_npz(str(output_dir / "test_s23_ngrams_addr_c3.npz"), addr_c3_matrix)
    with open(output_dir / "test_s23_ngrams_vecs.pkl", "wb") as f:
        pickle.dump({
            "name_c2": name_c2_vec,
            "name_c3": name_c3_vec,
            "addr_c2": addr_c2_vec,
            "addr_c3": addr_c3_vec,
        }, f)
    logger.info("  Saved in %.1f sec", time.time() - t1)

    logger.info("=" * 60)
    logger.info("NGRAMS PRECOMPUTE DONE in %.1f min", (time.time() - t0) / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()