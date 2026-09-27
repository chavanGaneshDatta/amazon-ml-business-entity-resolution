"""Precompute and save blocking keys + inverted index for test S2/S3.

Loads the existing test_s23_records.pkl (no need to recompute records).
Saves test_s23_blocking.pkl with the keys and inverted index.
"""

import logging
import os
import pickle
import sys
import time
from collections import defaultdict

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
logger = logging.getLogger("precompute_blocking")


def main():
    t0 = time.time()
    output_dir = config.OUTPUT_DIR

    logger.info("Loading precomputed records ...")
    with open(output_dir / "test_s23_records.pkl", "rb") as f:
        s23_records = pickle.load(f)
    logger.info("  Loaded %d records", len(s23_records))

    logger.info("Precomputing blocking keys ...")
    t1 = time.time()
    s23_keys = [preprocessing.generate_blocking_keys(r) for r in s23_records]
    logger.info("  Keys computed in %.1f sec", time.time() - t1)

    logger.info("Building inverted index ...")
    t1 = time.time()
    index = defaultdict(list)
    for idx, keys in enumerate(s23_keys):
        for k in keys:
            index[k].append(idx)
    logger.info("  Index built in %.1f sec", time.time() - t1)
    logger.info("  Unique keys: %d", len(index))

    logger.info("Saving blocking data to disk ...")
    t1 = time.time()
    with open(output_dir / "test_s23_blocking.pkl", "wb") as f:
        pickle.dump({"keys": s23_keys, "index": dict(index)}, f, protocol=4)
    logger.info("  Saved in %.1f sec", time.time() - t1)

    logger.info("=" * 60)
    logger.info("BLOCKING PRECOMPUTE DONE in %.1f min", (time.time() - t0) / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()