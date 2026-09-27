"""Lightweight precompute: phonetic keys + first/last tokens only.

Skips char n-grams (too slow on 10M records). Those features will be
filled with neutral 0.5 in the predictor.
"""

import logging
import os
import pickle
import sys
import time

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
logger = logging.getLogger("precompute_features")


def main():
    t0 = time.time()
    output_dir = config.OUTPUT_DIR

    logger.info("Loading precomputed records ...")
    with open(output_dir / "test_s23_records.pkl", "rb") as f:
        s23_records = pickle.load(f)
    logger.info("  Loaded %d records", len(s23_records))

    n = len(s23_records)

    logger.info("Computing phonetic keys ...")
    t1 = time.time()
    name_phonetic = [preprocessing.phonetic_key(r["name"]) for r in s23_records]
    logger.info("  Phonetic done in %.1f sec", time.time() - t1)

    logger.info("Computing soundex keys ...")
    t1 = time.time()
    name_soundex = [preprocessing.soundex_key(r["name"]) for r in s23_records]
    logger.info("  Soundex done in %.1f sec", time.time() - t1)

    logger.info("Computing first/last tokens ...")
    name_first_tok = [r["name_tokens"][0] if r["name_tokens"] else "" for r in s23_records]
    name_last_tok = [r["name_tokens"][-1] if r["name_tokens"] else "" for r in s23_records]

    logger.info("Saving features ...")
    t1 = time.time()
    features = {
        "name_phonetic": name_phonetic,
        "name_soundex": name_soundex,
        "name_first_tok": name_first_tok,
        "name_last_tok": name_last_tok,
    }
    with open(output_dir / "test_s23_features.pkl", "wb") as f:
        pickle.dump(features, f, protocol=4)
    logger.info("  Saved in %.1f sec", time.time() - t1)

    logger.info("=" * 60)
    logger.info("FEATURES PRECOMPUTE DONE in %.1f min", (time.time() - t0) / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()