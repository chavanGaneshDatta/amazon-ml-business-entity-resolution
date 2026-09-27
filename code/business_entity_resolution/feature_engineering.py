"""Fast feature extraction for candidate pairs.

All 23 features preserved, but computed in batches:
- TF-IDF cosine: single sparse matrix operation
- Levenshtein / Jaro: rapidfuzz batch
- Jaccard: precomputed n-gram sets
- Phonetic: precomputed keys
"""

import logging
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import preprocessing

logger = logging.getLogger(__name__)

try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False

try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed — falling back to python-Levenshtein")

try:
    import Levenshtein
    _HAS_LEV = True
except ImportError:
    _HAS_LEV = False


FEATURE_NAMES = [
    "name_jaccard_tokens",
    "name_jaccard_char2",
    "name_jaccard_char3",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_tfidf_cosine",
    "name_phonetic_eq",
    "name_soundex_eq",
    "name_first_token_eq",
    "name_last_token_eq",
    "name_length_ratio",
    "addr_jaccard_tokens",
    "addr_jaccard_char2",
    "addr_jaccard_char3",
    "addr_levenshtein",
    "addr_tfidf_cosine",
    "addr_length_ratio",
    "addr_digit_overlap",
    "country_match",
    "name_weighted_combo",
    "addr_weighted_combo",
    "same_source_prefix",
]


# ---------------------------------------------------------------------------
# Fast batched helpers
# ---------------------------------------------------------------------------
def _char_ngrams(text, n=2):
    text = text.replace(" ", "")
    if len(text) < n:
        return set([text]) if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_set(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _lev_py(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def levenshtein_sim_fast(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b) / 100.0
    if _HAS_LEV:
        dist = Levenshtein.distance(a, b)
    else:
        dist = _lev_py(a, b)
    return 1.0 - dist / max(len(a), len(b))


def jaro_winkler_sim_fast(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return fuzz.WRatio(a, b) / 100.0
    if _HAS_JELLYFISH:
        try:
            return jellyfish.jaro_winkler_similarity(a, b)
        except Exception:
            pass
    return levenshtein_sim_fast(a, b)


# ---------------------------------------------------------------------------
# TF-IDF (batched)
# ---------------------------------------------------------------------------
class TfidfSimilarity:
    """
    Fast batch cosine similarity. Precomputes norms once.
    """

    def __init__(self):
        self.name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                        min_df=2, max_features=50000)
        self.addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                        min_df=2, max_features=50000)
        self.name_matrix = None
        self.addr_matrix = None
        self.name_norms = None
        self.addr_norms = None
        self._fitted = False

    def _compute_norms(self):
        self.name_norms = np.sqrt(np.asarray(self.name_matrix.multiply(self.name_matrix).sum(axis=1))).flatten()
        self.addr_norms = np.sqrt(np.asarray(self.addr_matrix.multiply(self.addr_matrix).sum(axis=1))).flatten()
        # avoid division by zero
        self.name_norms[self.name_norms == 0] = 1e-9
        self.addr_norms[self.addr_norms == 0] = 1e-9

    def fit_transform(self, names, addresses):
        self.name_matrix = self.name_vec.fit_transform(names)
        self.addr_matrix = self.addr_vec.fit_transform(addresses)
        self._compute_norms()
        self._fitted = True

    def batch_cosine(self, s1_idx_array, s23_idx_array, field="name"):
        """
        Compute cosine similarity for a batch of pairs using sparse ops.
        """
        if not self._fitted:
            return np.zeros(len(s1_idx_array), dtype=np.float32)

        if field == "name":
            matrix = self.name_matrix
            norms = self.name_norms
        else:
            matrix = self.addr_matrix
            norms = self.addr_norms

        v1 = matrix[s1_idx_array]
        v2 = matrix[s23_idx_array]
        # Element-wise multiply, sum along columns
        dots = np.asarray(v1.multiply(v2).sum(axis=1)).flatten()
        denom = norms[s1_idx_array] * norms[s23_idx_array]
        return (dots / denom).astype(np.float32)


# ---------------------------------------------------------------------------
# Precomputed per-record features
# ---------------------------------------------------------------------------
class RecordCache:
    """Precompute all per-record features once."""

    def __init__(self, records):
        self.records = records
        n = len(records)

        self.name_tokens = [set(r["name_tokens"]) for r in records]
        self.addr_tokens = [set(r["address_tokens"]) for r in records]
        self.name_char2 = [_char_ngrams(r["name"], 2) for r in records]
        self.name_char3 = [_char_ngrams(r["name"], 3) for r in records]
        self.addr_char2 = [_char_ngrams(r["address"], 2) for r in records]
        self.addr_char3 = [_char_ngrams(r["address"], 3) for r in records]
        self.name_phonetic = [preprocessing.phonetic_key(r["name"]) for r in records]
        self.name_soundex = [preprocessing.soundex_key(r["name"]) for r in records]
        self.name_first_tok = [(r["name_tokens"][0] if r["name_tokens"] else "") for r in records]
        self.name_last_tok = [(r["name_tokens"][-1] if r["name_tokens"] else "") for r in records]
        self.name_len = [len(r["name"]) for r in records]
        self.addr_len = [len(r["address"]) for r in records]
        self.country = [r["country"] for r in records]
        self.addr_digits = [set(r["address_digits"]) for r in records]
        self.entity_id = [r["entity_id"] for r in records]


# ---------------------------------------------------------------------------
# Build the full feature matrix for a chunk
# ---------------------------------------------------------------------------
def build_feature_matrix_from_meta(s1_cache, s23_cache, pair_meta, tfidf):
    """
    Vectorized feature construction for a batch of pairs.

    pair_meta: list of (s1_idx, s23_idx) — indices into the two RecordCaches.
    """
    n = len(pair_meta)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    s1_idx = np.array([p[0] for p in pair_meta], dtype=np.int64)
    s23_idx = np.array([p[1] for p in pair_meta], dtype=np.int64)

    # --- TF-IDF cosines (batched) ----------------------------------------
    name_tfidf = tfidf.batch_cosine(s1_idx, s23_idx, "name")
    addr_tfidf = tfidf.batch_cosine(s1_idx, s23_idx, "address")

    # --- Jaccard (loops over cached sets, still fast) --------------------
    name_jac_tok = np.empty(n, dtype=np.float32)
    addr_jac_tok = np.empty(n, dtype=np.float32)
    name_jac_c2 = np.empty(n, dtype=np.float32)
    name_jac_c3 = np.empty(n, dtype=np.float32)
    addr_jac_c2 = np.empty(n, dtype=np.float32)
    addr_jac_c3 = np.empty(n, dtype=np.float32)
    name_lev = np.empty(n, dtype=np.float32)
    name_jw = np.empty(n, dtype=np.float32)
    addr_lev = np.empty(n, dtype=np.float32)
    name_phon = np.empty(n, dtype=np.float32)
    name_sdx = np.empty(n, dtype=np.float32)
    name_ftok = np.empty(n, dtype=np.float32)
    name_ltok = np.empty(n, dtype=np.float32)
    name_len_r = np.empty(n, dtype=np.float32)
    addr_len_r = np.empty(n, dtype=np.float32)
    addr_dig = np.empty(n, dtype=np.float32)
    country_eq = np.empty(n, dtype=np.float32)
    same_prefix = np.empty(n, dtype=np.float32)

    s1_records = s1_cache.records
    s23_records = s23_cache.records

    for k in range(n):
        i, j = s1_idx[k], s23_idx[k]

        name_a = s1_records[i]["name"]
        name_b = s23_records[j]["name"]
        addr_a = s1_records[i]["address"]
        addr_b = s23_records[j]["address"]

        name_jac_tok[k] = jaccard_set(s1_cache.name_tokens[i], s23_cache.name_tokens[j])
        addr_jac_tok[k] = jaccard_set(s1_cache.addr_tokens[i], s23_cache.addr_tokens[j])
        name_jac_c2[k] = jaccard_set(s1_cache.name_char2[i], s23_cache.name_char2[j])
        name_jac_c3[k] = jaccard_set(s1_cache.name_char3[i], s23_cache.name_char3[j])
        addr_jac_c2[k] = jaccard_set(s1_cache.addr_char2[i], s23_cache.addr_char2[j])
        addr_jac_c3[k] = jaccard_set(s1_cache.addr_char3[i], s23_cache.addr_char3[j])

        name_lev[k] = levenshtein_sim_fast(name_a, name_b)
        name_jw[k] = jaro_winkler_sim_fast(name_a, name_b)
        addr_lev[k] = levenshtein_sim_fast(addr_a, addr_b)

        name_phon[k] = 1.0 if (s1_cache.name_phonetic[i]
                               and s1_cache.name_phonetic[i] == s23_cache.name_phonetic[j]) else 0.0
        name_sdx[k] = 1.0 if (s1_cache.name_soundex[i]
                              and s1_cache.name_soundex[i] == s23_cache.name_soundex[j]) else 0.0
        name_ftok[k] = 1.0 if (s1_cache.name_first_tok[i]
                               and s1_cache.name_first_tok[i] == s23_cache.name_first_tok[j]) else 0.0
        name_ltok[k] = 1.0 if (s1_cache.name_last_tok[i]
                               and s1_cache.name_last_tok[i] == s23_cache.name_last_tok[j]) else 0.0

        la, lb = s1_cache.name_len[i], s23_cache.name_len[j]
        name_len_r[k] = min(la, lb) / max(la, lb) if max(la, lb) > 0 else 0.0
        la, lb = s1_cache.addr_len[i], s23_cache.addr_len[j]
        addr_len_r[k] = min(la, lb) / max(la, lb) if max(la, lb) > 0 else 0.0

        addr_dig[k] = jaccard_set(s1_cache.addr_digits[i], s23_cache.addr_digits[j])
        country_eq[k] = 1.0 if (s1_cache.country[i]
                                and s1_cache.country[i] == s23_cache.country[j]) else 0.0
        same_prefix[k] = 1.0 if (s1_cache.entity_id[i][:3] == s23_cache.entity_id[j][:3]) else 0.0

    # Combined features
    name_combo = 0.7 * name_tfidf + 0.3 * name_jw
    addr_combo = 0.6 * addr_tfidf + 0.4 * addr_jac_tok

    # Stack in FEATURE_NAMES order
    X = np.stack([
        name_jac_tok, name_jac_c2, name_jac_c3, name_lev, name_jw, name_tfidf,
        name_phon, name_sdx, name_ftok, name_ltok, name_len_r,
        addr_jac_tok, addr_jac_c2, addr_jac_c3, addr_lev, addr_tfidf, addr_len_r,
        addr_dig, country_eq,
        name_combo, addr_combo, same_prefix,
    ], axis=1).astype(np.float32)

    return X


def build_feature_matrix(s1_records, s23_records, candidates, tfidf):
    """Legacy interface — returns pair_meta with LOCAL indices."""
    logger.info("Building feature matrix (batched) ...")

    # Compute TF-IDF matrices for both sides (transform only, don't refit)
    names = [r["name"] for r in s1_records] + [r["name"] for r in s23_records]
    addrs = [r["address"] for r in s1_records] + [r["address"] for r in s23_records]
    tfidf.name_matrix = tfidf.name_vec.transform(names)
    tfidf.addr_matrix = tfidf.addr_vec.transform(addrs)
    tfidf._compute_norms()
    tfidf._fitted = True

    # Build pair_meta with LOCAL indices (S1: 0..n1, S23: 0..n23)
    pair_meta_local = []
    pair_meta_combined = []
    groups = []
    offset = len(s1_records)

    for s1_idx, cand_list in enumerate(candidates):
        groups.append(len(cand_list))
        for s23_idx in cand_list:
            pair_meta_local.append((s1_idx, s23_idx))
            pair_meta_combined.append((s1_idx, offset + s23_idx))

    # Build combined cache once (contains S1 + S23)
    combined_cache = RecordCache(s1_records + s23_records)
    X = build_feature_matrix_from_meta(
        combined_cache, combined_cache, pair_meta_combined, tfidf,
    )

    # IMPORTANT: return LOCAL pair_meta so callers can index s23_records directly
    return X, groups, pair_meta_local

def fit_tfidf(s1_records, s23_records, max_samples=100000):
    """Fit TF-IDF on a subsample for speed."""
    logger.info("Fitting TF-IDF on subsample (max %d per side)", max_samples)
    rng = random.Random(42)
    s1_sample = s1_records if len(s1_records) <= max_samples else rng.sample(s1_records, max_samples)
    s23_sample = s23_records if len(s23_records) <= max_samples else rng.sample(s23_records, max_samples)
    names = [r["name"] for r in s1_sample] + [r["name"] for r in s23_sample]
    addrs = [r["address"] for r in s1_sample] + [r["address"] for r in s23_sample]
    tfidf = TfidfSimilarity()
    tfidf.fit_transform(names, addrs)
    return tfidf