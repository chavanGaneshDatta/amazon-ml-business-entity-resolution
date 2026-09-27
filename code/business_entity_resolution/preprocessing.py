"""Text normalisation, blocking-key generation, and candidate generation.

Optimized version:
- Requires 2+ shared blocking keys (co-occurrence) for a candidate.
- Precomputes phonetic keys per record.
- Uses efficient inverted index.
"""

import logging
import re
import unicodedata
from collections import defaultdict

from . import config

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")

MAX_CANDIDATES_PER_S1 = 15   # hard cap per Source-1 record
MIN_SHARED_KEYS = 2          # require 2+ shared keys


def _strip_accents(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalise_text(text):
    if not text:
        return ""
    text = _strip_accents(str(text)).lower()
    for k, v in config.WORD_REPLACEMENTS.items():
        if v:
            text = text.replace(k, f" {v} ")
        else:
            text = text.replace(k, " ")
    text = _PUNCT_RE.sub(" ", text)
    tokens = []
    for tok in text.split():
        tok = config.ABBREVIATIONS.get(tok, tok)
        if tok in config.STOPWORDS:
            continue
        if tok:
            tokens.append(tok)
    text = " ".join(tokens)
    return _MULTISPACE_RE.sub(" ", text).strip()


def normalise_name(name):
    return normalise_text(name)


def normalise_address(address):
    return normalise_text(address)


def normalise_country(country):
    return _strip_accents(str(country)).lower().strip()


def extract_tokens(text):
    return text.split() if text else []


def extract_digits(text):
    return "".join(_DIGIT_RE.findall(text or ""))


def _safe_prefix(text, n):
    compact = text.replace(" ", "")
    if len(compact) < n:
        return ""
    return compact[:n]


try:
    import jellyfish
    _HAS_JELLYFISH = True
except ImportError:
    _HAS_JELLYFISH = False
    logger.warning("jellyfish not installed; phonetic keys disabled")


def phonetic_key(text):
    """Metaphone of the first two tokens (concatenated)."""
    if not _HAS_JELLYFISH or not text:
        return ""
    toks = text.split()
    if not toks:
        return ""
    try:
        key = jellyfish.metaphone(toks[0])
        if len(toks) > 1:
            key += "_" + jellyfish.metaphone(toks[1])
        return key
    except Exception:
        return ""


def soundex_key(text):
    """Soundex of the first token."""
    if not _HAS_JELLYFISH or not text:
        return ""
    toks = text.split()
    if not toks:
        return ""
    try:
        return jellyfish.soundex(toks[0])
    except Exception:
        return ""


def generate_blocking_keys(record):
    """
    Three tight blocking keys:
      1. Name prefix (5 chars)
      2. Metaphone of first 2 tokens
      3. Postal code (first 6 digits)
    """
    keys = set()
    name = record.get("name", "")
    address = record.get("address", "")

    p = _safe_prefix(name, 5)
    if len(p) >= 4:
        keys.add(f"NP::{p}")

    pk = phonetic_key(name)
    if pk and len(pk) >= 3:
        keys.add(f"PH::{pk}")

    digits = extract_digits(address)
    if len(digits) >= 5:
        keys.add(f"PC::{digits[:6]}")

    return keys


def precompute_keys(records):
    """Precompute blocking keys for all records once."""
    return [generate_blocking_keys(r) for r in records]


def build_inverted_index(records, precomputed_keys=None):
    if precomputed_keys is None:
        precomputed_keys = precompute_keys(records)
    index = defaultdict(list)
    for idx, keys in enumerate(precomputed_keys):
        for key in keys:
            index[key].append(idx)
    return index


def generate_candidates(source1_records, source23_records):
    """
    Two-key co-occurrence candidate generation.

    A source2/3 record becomes a candidate for a source1 record only if
    they share >= MIN_SHARED_KEYS blocking keys.
    """
    logger.info("Precomputing blocking keys for S2/S3 (%d records)",
                len(source23_records))
    s23_keys = precompute_keys(source23_records)

    logger.info("Building inverted index")
    index = defaultdict(list)
    for idx, keys in enumerate(s23_keys):
        for key in keys:
            index[key].append(idx)

    logger.info("Precomputing blocking keys for S1 (%d records)",
                len(source1_records))
    s1_keys = precompute_keys(source1_records)

    logger.info("Matching candidates (2-key co-occurrence)")
    candidates = []
    for i, keys in enumerate(s1_keys):
        if not keys:
            candidates.append([])
            continue

        # Count how many keys each S2/S3 record shares
        hit_count = defaultdict(int)
        for key in keys:
            for idx in index.get(key, []):
                hit_count[idx] += 1

        keep = [idx for idx, cnt in hit_count.items() if cnt >= MIN_SHARED_KEYS]
        keep.sort()

        if len(keep) > MAX_CANDIDATES_PER_S1:
            keep = keep[:MAX_CANDIDATES_PER_S1]

        candidates.append(keep)

        if (i + 1) % 10000 == 0:
            logger.info("  Candidate gen progress: %d / %d", i + 1, len(source1_records))

    non_empty = sum(1 for c in candidates if c)
    total_pairs = sum(len(c) for c in candidates)
    avg = total_pairs / len(source1_records) if source1_records else 0
    logger.info("Candidate generation: %d/%d S1 have candidates, %d pairs, avg %.1f",
                non_empty, len(source1_records), total_pairs, avg)
    return candidates

def generate_candidates_with_index(source1_records, s23_records,
                                    s23_keys, s23_index):
    """
    Fast candidate generation using a precomputed inverted index.
    Avoids recomputing keys for 10M S2/S3 records on each call.
    """
    logger.info("Precomputing blocking keys for S1 (%d records)",
                len(source1_records))
    s1_keys = [generate_blocking_keys(r) for r in source1_records]

    logger.info("Matching candidates (2-key co-occurrence)")
    candidates = []
    for i, keys in enumerate(s1_keys):
        if not keys:
            candidates.append([])
            continue

        hit_count = defaultdict(int)
        for key in keys:
            for idx in s23_index.get(key, []):
                hit_count[idx] += 1

        keep = [idx for idx, cnt in hit_count.items() if cnt >= MIN_SHARED_KEYS]
        keep.sort()

        if len(keep) > MAX_CANDIDATES_PER_S1:
            keep = keep[:MAX_CANDIDATES_PER_S1]

        candidates.append(keep)

    non_empty = sum(1 for c in candidates if c)
    total_pairs = sum(len(c) for c in candidates)
    avg = total_pairs / len(source1_records) if source1_records else 0
    logger.info("Candidate generation: %d/%d S1 have candidates, %d pairs, avg %.1f",
                non_empty, len(source1_records), total_pairs, avg)
    return candidates

def prepare_records(df, is_source1=False):
    """Fastest version — builds records via to_dict then augments."""
    names_norm = df["business_name"].fillna("").astype(str).map(normalise_name).tolist()
    addrs_norm = df["business_address"].fillna("").astype(str).map(normalise_address).tolist()
    countries_norm = df["country"].fillna("").astype(str).map(normalise_country).tolist()
    eids = df["entity_id"].tolist()

    records = []
    for i in range(len(eids)):
        name = names_norm[i]
        addr = addrs_norm[i]
        records.append({
            "entity_id": eids[i],
            "name": name,
            "address": addr,
            "country": countries_norm[i],
            "name_tokens": name.split() if name else [],
            "address_tokens": addr.split() if addr else [],
            "name_digits": extract_digits(name),
            "address_digits": extract_digits(addr),
        })
    return records