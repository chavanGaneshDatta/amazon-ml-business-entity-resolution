"""Central configuration for the entity resolution pipeline."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"

# TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
# TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
# TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
# TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TRAIN_SOURCE1 = TRAIN_DIR / "large_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "large_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "large_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "large_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"

# ---------------------------------------------------------------------------
# Abbreviation dictionary (name + address normalisation)
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    # Legal suffixes - business names
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "pvt": "private",
    "pve": "private",
    "p": "private",
    "intl": "international",
    "int": "international",
    "bros": "brothers",
    "assoc": "associates",
    "ent": "enterprises",
    "entp": "enterprises",
    "mfg": "manufacturing",
    "tech": "technology",
    "svc": "services",
    "svcs": "services",
    "govt": "government",
    "dept": "department",
    "univ": "university",
    "natl": "national",
    "fed": "federal",
    # Address abbreviations
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "sq": "square",
    "ct": "court",
    "pl": "place",
    "ter": "terrace",
    "pkwy": "parkway",
    "cir": "circle",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "opp": "opposite",
    "nr": "near",
    "nxt": "next",
    "po": "post office",
    "pin": "postal index number",
    "zip": "zone improvement plan",
}

# Common word replacements
WORD_REPLACEMENTS = {
    "&": "and",
    "+": "and",
    "dba": "",
    "t/a": "",
    "o/b": "",
}

STOPWORDS = {
    "the", "a", "an", "of", "at", "in", "on", "for", "to",
    "and", "or", "with", "by",
}

# ---------------------------------------------------------------------------
# Blocking parameters
# ---------------------------------------------------------------------------
NAME_PREFIX_LEN = 4          # chars used for name prefix key
ADDRESS_PREFIX_LEN = 4       # chars used for address prefix key
TOKEN_KEY_LEN = 3            # first N tokens used for token key
MIN_KEY_LEN = 3              # ignore keys shorter than this

# ---------------------------------------------------------------------------
# Model hyperparameters (LightGBM lambdarank)
# ---------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1],
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "min_child_samples": 10,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "max_depth": 5,
    "max_bin": 63,
    "verbose": 1,
    "n_jobs": 1,
    "random_state": 42,
    "min_data_in_leaf": 10,
    "min_sum_hessian_in_leaf": 1e-3,
}
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# Evaluation / thresholding
# ---------------------------------------------------------------------------
BETA = 0.5
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Threshold search range for F_0.5 optimisation
THRESHOLD_GRID = [round(x, 3) for x in
                  [i / 100 for i in range(5, 96, 5)]]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)