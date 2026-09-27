"""Build a large, stratified training subset for the entity resolution model.

Samples:
- 60,000 S1 records with exactly 1 positive match
- 40,000 S1 records with 2+ positive matches
- 50,000 S1 records that are singletons (no matches)
- All matched S2/S3 records referenced by the positives
- 200,000 random distractor S2 records
- 200,000 random distractor S3 records

Output files (dataset/train/):
- large_source1.tsv
- large_source2.tsv
- large_source3.tsv
- large_ground_truth.tsv
"""

import random
import sys
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SINGLE = 50_000        # S1 records with no matches
N_ONE_MATCH = 20_000     # S1 records with exactly 1 match
N_MANY_MATCH = 80_000    # S1 records with 2+ matches
N_RANDOM_S2 = 200_000    # Extra S2 distractors
N_RANDOM_S3 = 200_000    # Extra S3 distractors
RANDOM_SEED = 42

TRAIN_DIR = "dataset/train"


def main():
    t0 = time.time()
    random.seed(RANDOM_SEED)

    # ---- Load ground truth -------------------------------------------------
    print("[1/7] Loading ground truth ...", flush=True)
    gt = pd.read_csv(
        f"{TRAIN_DIR}/train_ground_truth.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    print(f"      Total ground truth rows: {len(gt):,}", flush=True)

    non_empty = gt[gt["matched_entity_ids"] != ""].copy()
    empty = gt[gt["matched_entity_ids"] == ""].copy()
    print(f"      With matches: {len(non_empty):,}", flush=True)
    print(f"      Singletons:   {len(empty):,}", flush=True)

    # ---- Classify positives by number of matches ---------------------------
    non_empty["n_matches"] = non_empty["matched_entity_ids"].str.split(",").str.len()
    one_match = non_empty[non_empty["n_matches"] == 1]
    many_matches = non_empty[non_empty["n_matches"] >= 2]
    print(f"      Positives with 1 match:  {len(one_match):,}", flush=True)
    print(f"      Positives with 2+ match: {len(many_matches):,}", flush=True)

    # ---- Stratified samples ------------------------------------------------
    n_single = min(N_SINGLE, len(empty))
    n_one = min(N_ONE_MATCH, len(one_match))
    n_many = min(N_MANY_MATCH, len(many_matches))

    single_sample = empty.sample(n=n_single, random_state=RANDOM_SEED)
    one_sample = one_match.sample(n=n_one, random_state=RANDOM_SEED)
    many_sample = many_matches.sample(n=n_many, random_state=RANDOM_SEED)

    pos_sample = pd.concat([one_sample, many_sample], ignore_index=True)
    print(f"      Selected singletons: {len(single_sample):,}", flush=True)
    print(f"      Selected positives:  {len(pos_sample):,}", flush=True)

    s1_ids_to_keep = (
        set(pos_sample["source1_entity_id"]) |
        set(single_sample["source1_entity_id"])
    )
    print(f"      Total S1 IDs to keep: {len(s1_ids_to_keep):,}", flush=True)

    # ---- Collect matched S2/S3 IDs ----------------------------------------
    print("[2/7] Collecting matched S2/S3 IDs ...", flush=True)
    matched_s23_ids = set()
    for ids_str in pos_sample["matched_entity_ids"]:
        for x in ids_str.split(","):
            x = x.strip()
            if x:
                matched_s23_ids.add(x)
    print(f"      Unique matched S2/S3 IDs: {len(matched_s23_ids):,}", flush=True)

    matched_s2_ids = {x for x in matched_s23_ids if x.startswith("S2-")}
    matched_s3_ids = {x for x in matched_s23_ids if x.startswith("S3-")}
    print(f"      Of which S2: {len(matched_s2_ids):,}", flush=True)
    print(f"      Of which S3: {len(matched_s3_ids):,}", flush=True)

    # ---- Filter train_source1 ---------------------------------------------
    print("[3/7] Filtering train_source1 ...", flush=True)
    s1 = pd.read_csv(
        f"{TRAIN_DIR}/train_source1.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    s1_subset = s1[s1["entity_id"].isin(s1_ids_to_keep)].copy()
    print(f"      S1 subset: {len(s1_subset):,}", flush=True)
    s1_subset.to_csv(f"{TRAIN_DIR}/large_source1.tsv", sep="\t", index=False)

    # ---- Filter train_source2 ---------------------------------------------
    print("[4/7] Filtering train_source2 ...", flush=True)
    s2 = pd.read_csv(
        f"{TRAIN_DIR}/train_source2.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    s2_matched = s2[s2["entity_id"].isin(matched_s2_ids)]
    n_random_s2 = min(N_RANDOM_S2, len(s2))
    s2_extra = s2.sample(n=n_random_s2, random_state=RANDOM_SEED)
    s2_subset = pd.concat([s2_matched, s2_extra], ignore_index=True)
    s2_subset = s2_subset.drop_duplicates(subset=["entity_id"])
    print(f"      S2 subset: {len(s2_subset):,}", flush=True)
    s2_subset.to_csv(f"{TRAIN_DIR}/large_source2.tsv", sep="\t", index=False)

    # ---- Filter train_source3 ---------------------------------------------
    print("[5/7] Filtering train_source3 ...", flush=True)
    s3 = pd.read_csv(
        f"{TRAIN_DIR}/train_source3.tsv",
        sep="\t", dtype=str, keep_default_na=False,
    )
    s3_matched = s3[s3["entity_id"].isin(matched_s3_ids)]
    n_random_s3 = min(N_RANDOM_S3, len(s3))
    s3_extra = s3.sample(n=n_random_s3, random_state=RANDOM_SEED)
    s3_subset = pd.concat([s3_matched, s3_extra], ignore_index=True)
    s3_subset = s3_subset.drop_duplicates(subset=["entity_id"])
    print(f"      S3 subset: {len(s3_subset):,}", flush=True)
    s3_subset.to_csv(f"{TRAIN_DIR}/large_source3.tsv", sep="\t", index=False)

    # ---- Ground truth subset -----------------------------------------------
    print("[6/7] Building ground truth subset ...", flush=True)
    gt_subset = gt[gt["source1_entity_id"].isin(s1_ids_to_keep)].copy()
    print(f"      GT subset: {len(gt_subset):,}", flush=True)
    gt_subset.to_csv(f"{TRAIN_DIR}/large_ground_truth.tsv", sep="\t", index=False)

    # ---- Summary -----------------------------------------------------------
    print("[7/7] Done.", flush=True)
    print()
    print("=" * 60)
    print("LARGE STRATIFIED SUBSET READY")
    print("=" * 60)
    print(f"S1 with 1 match:        {len(one_sample):>10,}")
    print(f"S1 with 2+ matches:     {len(many_sample):>10,}")
    print(f"S1 singletons:          {len(single_sample):>10,}")
    print(f"Total S1:               {len(s1_subset):>10,}")
    print()
    print(f"S2 records:             {len(s2_subset):>10,}")
    print(f"S3 records:             {len(s3_subset):>10,}")
    print(f"Total S2/S3:            {len(s2_subset) + len(s3_subset):>10,}")
    print()
    print(f"Ground truth rows:      {len(gt_subset):>10,}")
    print("=" * 60)
    print(f"Elapsed: {time.time() - t0:.1f} sec")
    print("=" * 60)


if __name__ == "__main__":
    main()