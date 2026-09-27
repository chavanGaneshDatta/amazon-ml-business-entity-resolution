"""Fast predictor — no sparse matrix slicing, no S23-side Python sets.

Computes all 22 features per pair using raw string/token operations on
the precomputed records. Avoids the memory-paging bottleneck of loading
10M-row sparse matrices and slicing them per chunk.
"""

import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    )
    from code.business_entity_resolution import (
        config, preprocessing, model, utils
    )
else:
    from . import config, preprocessing, model, utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("predict")

S1_CHUNK_SIZE = 25_000


def _ngrams(text, k):
    """Return a set of character k-grams (spaces removed)."""
    text = text.replace(" ", "")
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _jac(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _lev_sim(a, b):
    """Levenshtein-based similarity (normalized)."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a, b) / 100.0
    except ImportError:
        import Levenshtein
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b))


def _jw_sim(a, b):
    """Jaro-Winkler similarity."""
    try:
        from rapidfuzz import fuzz
        return fuzz.WRatio(a, b) / 100.0
    except ImportError:
        import jellyfish
        if not a or not b:
            return 0.0
        return jellyfish.jaro_winkler_similarity(a, b)


def main():
    t0 = time.time()
    output_dir = config.OUTPUT_DIR

    # ---- Load model -----------------------------------------------------
    logger.info("Loading trained model ...")
    import lightgbm as lgb
    lgbm_model = lgb.Booster(model_file=str(output_dir / "lgbm_model.txt"))
    with open(output_dir / "threshold.txt") as f:
        threshold = float(f.read().strip())
    logger.info("Threshold = %.4f", threshold)

    # ---- Load precomputed records ---------------------------------------
    logger.info("Loading precomputed S2/S3 records ...")
    with open(output_dir / "test_s23_records.pkl", "rb") as f:
        s23_records = pickle.load(f)
    logger.info("  %d records", len(s23_records))

    # ---- Load blocking data ---------------------------------------------
    logger.info("Loading precomputed blocking keys ...")
    with open(output_dir / "test_s23_blocking.pkl", "rb") as f:
        blocking_data = pickle.load(f)
    s23_keys = blocking_data["keys"]
    s23_index = blocking_data["index"]
    logger.info("  %d unique keys", len(s23_index))

    # ---- Load phonetic/soundex/first-last tokens ------------------------
    logger.info("Loading precomputed features ...")
    with open(output_dir / "test_s23_features.pkl", "rb") as f:
        s23_feats = pickle.load(f)
    logger.info("  Phonetic: %d, Soundex: %d, First: %d, Last: %d",
                len(s23_feats["name_phonetic"]),
                len(s23_feats["name_soundex"]),
                len(s23_feats["name_first_tok"]),
                len(s23_feats["name_last_tok"]))

    # ---- Open outputs ---------------------------------------------------
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    f_match = open(matching_path, "w", encoding="utf-8", newline="\n")
    f_cand = open(candidate_path, "w", encoding="utf-8", newline="\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # ---- Iterate S1 chunks ----------------------------------------------
    logger.info("Starting chunked prediction ...")
    reader = pd.read_csv(
        config.TEST_SOURCE1, sep="\t", dtype=str, keep_default_na=False,
        chunksize=S1_CHUNK_SIZE,
    )

    total = 0
    total_matched = 0
    total_singletons = 0
    chunk_num = 0

    for chunk_df in reader:
        chunk_num += 1
        chunk_t0 = time.time()
        logger.info("=" * 60)
        logger.info("Chunk %d: %d S1 records", chunk_num, len(chunk_df))
        logger.info("=" * 60)

        chunk_df = chunk_df[["entity_id", "business_name", "business_address", "country"]]
        s1_records = preprocessing.prepare_records(chunk_df, is_source1=True)

        logger.info("  Generating candidates ...")
        candidates_idx = preprocessing.generate_candidates_with_index(
            s1_records, s23_records, s23_keys, s23_index
        )
        n_pairs = sum(len(c) for c in candidates_idx)
        logger.info("  %d pairs (avg %.1f)", n_pairs, n_pairs / max(len(s1_records), 1))

        if n_pairs == 0:
            for s1_rec in s1_records:
                s1_id = s1_rec["entity_id"]
                f_match.write(f"{s1_id}\t\n")
                f_cand.write(f"{s1_id}\t\n")
            f_match.flush()
            f_cand.flush()
            total += len(s1_records)
            total_singletons += len(s1_records)
            continue

        # Precompute S1-side data
        s1_names = [r["name"] for r in s1_records]
        s1_addrs = [r["address"] for r in s1_records]
        s1_nlen = [len(r["name"]) for r in s1_records]
        s1_alen = [len(r["address"]) for r in s1_records]
        s1_ntoks = [set(r["name_tokens"]) for r in s1_records]
        s1_atoks = [set(r["address_tokens"]) for r in s1_records]
        s1_ndig = [set(r["address_digits"]) for r in s1_records]
        s1_nphon = [preprocessing.phonetic_key(r["name"]) for r in s1_records]
        s1_nsdx = [preprocessing.soundex_key(r["name"]) for r in s1_records]
        s1_nftok = [r["name_tokens"][0] if r["name_tokens"] else "" for r in s1_records]
        s1_nltok = [r["name_tokens"][-1] if r["name_tokens"] else "" for r in s1_records]
        s1_countries = [r["country"] for r in s1_records]
        s1_ids = [r["entity_id"] for r in s1_records]

        # Precompute char n-grams for S1 once
        s1_nchar2 = [_ngrams(n, 2) for n in s1_names]
        s1_nchar3 = [_ngrams(n, 3) for n in s1_names]
        s1_achar2 = [_ngrams(a, 2) for a in s1_addrs]
        s1_achar3 = [_ngrams(a, 3) for a in s1_addrs]

        # Build pair arrays
        s1_idx_arr = np.empty(n_pairs, dtype=np.int64)
        s23_idx_arr = np.empty(n_pairs, dtype=np.int64)
        groups = np.empty(len(s1_records), dtype=np.int32)
        k = 0
        for i, cand_list in enumerate(candidates_idx):
            groups[i] = len(cand_list)
            for j in cand_list:
                s1_idx_arr[k] = i
                s23_idx_arr[k] = j
                k += 1

        # Compute all 22 features per pair
        n = n_pairs
        name_jac_tok = np.empty(n, dtype=np.float32)
        name_jac_c2 = np.empty(n, dtype=np.float32)
        name_jac_c3 = np.empty(n, dtype=np.float32)
        name_lev = np.empty(n, dtype=np.float32)
        name_jw = np.empty(n, dtype=np.float32)
        name_tfidf = np.empty(n, dtype=np.float32)
        name_phon = np.empty(n, dtype=np.float32)
        name_sdx = np.empty(n, dtype=np.float32)
        name_ftok = np.empty(n, dtype=np.float32)
        name_ltok = np.empty(n, dtype=np.float32)
        name_len_r = np.empty(n, dtype=np.float32)
        addr_jac_tok = np.empty(n, dtype=np.float32)
        addr_jac_c2 = np.empty(n, dtype=np.float32)
        addr_jac_c3 = np.empty(n, dtype=np.float32)
        addr_lev = np.empty(n, dtype=np.float32)
        addr_tfidf = np.empty(n, dtype=np.float32)
        addr_len_r = np.empty(n, dtype=np.float32)
        addr_dig = np.empty(n, dtype=np.float32)
        country_eq = np.empty(n, dtype=np.float32)
        same_prefix = np.empty(n, dtype=np.float32)

        logger.info("  Computing %d × 22 features ...", n)

        for k in range(n):
            i = s1_idx_arr[k]
            j = s23_idx_arr[k]

            s23_name = s23_records[j]["name"]
            s23_addr = s23_records[j]["address"]
            s23_ntoks = s23_records[j]["name_tokens"]
            s23_atoks = s23_records[j]["address_tokens"]
            s23_ndig = s23_records[j]["address_digits"]

            # Name features
            name_jac_tok[k] = _jac(s1_ntoks[i], set(s23_ntoks))
            name_jac_c2[k] = _jac(s1_nchar2[i], _ngrams(s23_name, 2))
            name_jac_c3[k] = _jac(s1_nchar3[i], _ngrams(s23_name, 3))
            name_lev[k] = _lev_sim(s1_names[i], s23_name)
            name_jw[k] = _jw_sim(s1_names[i], s23_name)
            name_tfidf[k] = _jac(s1_ntoks[i], set(s23_ntoks))
            name_phon[k] = 1.0 if (s1_nphon[i] and
                                    s1_nphon[i] == s23_feats["name_phonetic"][j]) else 0.0
            name_sdx[k] = 1.0 if (s1_nsdx[i] and
                                   s1_nsdx[i] == s23_feats["name_soundex"][j]) else 0.0
            name_ftok[k] = 1.0 if (s1_nftok[i] and
                                    s1_nftok[i] == s23_feats["name_first_tok"][j]) else 0.0
            name_ltok[k] = 1.0 if (s1_nltok[i] and
                                    s1_nltok[i] == s23_feats["name_last_tok"][j]) else 0.0
            la, lb = s1_nlen[i], len(s23_name)
            name_len_r[k] = min(la, lb) / max(la, lb) if max(la, lb) else 0.0

            # Address features
            addr_jac_tok[k] = _jac(s1_atoks[i], set(s23_atoks))
            addr_jac_c2[k] = _jac(s1_achar2[i], _ngrams(s23_addr, 2))
            addr_jac_c3[k] = _jac(s1_achar3[i], _ngrams(s23_addr, 3))
            addr_lev[k] = _lev_sim(s1_addrs[i], s23_addr)
            addr_tfidf[k] = _jac(s1_atoks[i], set(s23_atoks))
            la, lb = s1_alen[i], len(s23_addr)
            addr_len_r[k] = min(la, lb) / max(la, lb) if max(la, lb) else 0.0

            # Numeric / categorical
            addr_dig[k] = _jac(s1_ndig[i], set(s23_ndig))
            country_eq[k] = 1.0 if (s1_countries[i] and
                                     s1_countries[i] == s23_records[j]["country"]) else 0.0
            same_prefix[k] = 1.0 if (s1_ids[i][:3] ==
                                      s23_records[j]["entity_id"][:3]) else 0.0

        # Combined features (matching training pipeline)
        name_combo = 0.7 * name_tfidf + 0.3 * name_jw
        addr_combo = 0.6 * addr_tfidf + 0.4 * addr_jac_tok

        X = np.stack([
            name_jac_tok, name_jac_c2, name_jac_c3, name_lev, name_jw, name_tfidf,
            name_phon, name_sdx, name_ftok, name_ltok, name_len_r,
            addr_jac_tok, addr_jac_c2, addr_jac_c3, addr_lev, addr_tfidf, addr_len_r,
            addr_dig, country_eq, name_combo, addr_combo, same_prefix,
        ], axis=1).astype(np.float32)

        logger.info("  Features built in %.1f sec", time.time() - chunk_t0)

        scores = model.predict_scores(lgbm_model, X)

        chunk_matches = 0
        chunk_singletons = 0
        pos = 0
        for i, s1_rec in enumerate(s1_records):
            s1_id = s1_rec["entity_id"]
            n_i = int(groups[i])
            if n_i == 0:
                f_match.write(f"{s1_id}\t\n")
                f_cand.write(f"{s1_id}\t\n")
                chunk_singletons += 1
                continue
            cand_indices = candidates_idx[i]
            cand_ids = [s23_records[j]["entity_id"] for j in cand_indices]
            sc = scores[pos:pos + n_i]
            pos += n_i
            matched = [cand_ids[j] for j in range(n_i) if sc[j] >= threshold]
            f_match.write(f"{s1_id}\t{utils.format_id_list(matched)}\n")
            f_cand.write(f"{s1_id}\t{utils.format_id_list(cand_ids)}\n")
            if matched:
                chunk_matches += 1
            else:
                chunk_singletons += 1

        f_match.flush()
        f_cand.flush()
        total += len(s1_records)
        total_matched += chunk_matches
        total_singletons += chunk_singletons

        logger.info(
            "  Chunk %d done in %.1f sec: %d matched, %d singletons | cum %d",
            chunk_num, time.time() - chunk_t0,
            chunk_matches, chunk_singletons, total,
        )

    f_match.close()
    f_cand.close()
    logger.info("=" * 60)
    logger.info("DONE in %.1f min. Total: %d | Matched: %d | Singletons: %d",
                (time.time() - t0) / 60, total, total_matched, total_singletons)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()