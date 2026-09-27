# Methodology — Amazon ML Challenge 2026 (Business Entity Resolution)

## 1. Problem Framing

We are given three independent sources of business records (S1, S2, S3) with
noisy, partially overlapping fields (`business_name`, `business_address`,
`country`). Source 1 is the deduplicated reference source; the goal is to
find, for every Source 1 record, the set of Source 2 / Source 3 records that
refer to the same real-world business.

Because the evaluation metric is a **macro-averaged F_0.5** (precision
weighted 2× over recall, singletons included), we frame the problem as a
**learning-to-rank** task where the model must place true matches above
distractors, and a threshold tuned for F_0.5 decides which candidates
actually become matches.

## 2. Pipeline Overview

```
Raw TSVs
   │
   ▼
Data Loading (data_loader.py)
   │
   ▼
Normalisation + Blocking (preprocessing.py)
   │
   ▼
Candidate generation (multi-key inverted index)
   │
   ▼
Feature engineering (feature_engineering.py)
   │
   ▼
LightGBM lambdarank (model.py)
   │
   ▼
Threshold tuning on held-out Source-1 split (utils.py)
   │
   ▼
Output TSVs (matching_results.tsv, candidate_pairs.tsv)
```

## 3. Candidate Generation / Blocking

Blocking is the recall ceiling of the whole system, so we use **multiple
complementary blocking keys** and union their buckets:

1. **Name prefix** — first 4 compact characters of the normalised name.
2. **Address prefix** — first 4 compact characters of the normalised address.
3. **Metaphone of the first name token** — handles spelling variants
   (e.g., *Wipro* / *Vipro*).
4. **Soundex of the first name token** — a second phonetic signal.
5. **Sorted token key** — the first 3 name tokens sorted alphabetically,
   which is robust to word-order transpositions.
6. **Country-scoped name prefix** — prevents cross-country collisions.
7. **Postal-code block** — first 6 digits of the address digits.

Records sharing *any* key become candidates. This gives high recall while
keeping the candidate set tractable.

`candidate_pairs.tsv` records the **final** candidate list per Source 1
entity — i.e., exactly what the LightGBM model scores. Every ID in
`matching_results.tsv` is a subset of this list.

## 4. Feature Engineering

For each `(S1, S2/3)` candidate pair we compute 23 features:

**Name features (11):**
- Jaccard over word tokens
- Jaccard over character 2-grams and 3-grams
- Normalised Levenshtein similarity
- Jaro–Winkler similarity
- TF-IDF cosine similarity (char 2–4 grams, `char_wb`)
- Metaphone equality, Soundex equality
- First-token equality, last-token equality
- Length ratio

**Address features (6):**
- Jaccard over word tokens
- Jaccard over character 2-grams and 3-grams
- Normalised Levenshtein similarity
- TF-IDF cosine similarity
- Length ratio

**Numeric / categorical (2):**
- Digit-group overlap (PIN / ZIP / street numbers)
- Country equality

**Combined / structural (4):**
- Weighted name combo (0.7·TF-IDF + 0.3·Jaro–Winkler)
- Weighted address combo (0.6·TF-IDF + 0.4·token-Jaccard)
- Same source prefix indicator (S1 vs S2/S3 namespace sanity check)
- Implicit group size handled by the ranker

All text is normalised first: lowercased, accents stripped, abbreviations
expanded (`corp → corporation`, `rd → road`, `pvt → private`, …), punctuation
removed, common stopwords dropped. Country labels are treated as an **open
set** — we never hard-code `{US, India}`, so the pipeline generalises to the
unseen France test records.

## 5. Model

- **Algorithm:** LightGBM with `objective=lambdarank`, `metric=ndcg`.
- **Grouping:** one group per Source 1 entity (its candidate list).
- **Labels:** 1 for true matches (from `train_ground_truth.tsv`), 0 otherwise.
- **Hyperparameters:** 63 leaves, LR 0.05, 500 rounds, early stopping
  (50 rounds) on a held-out 20% Source-1 validation split, feature
  fraction 0.8, bagging fraction 0.8.
- **Why lambdarank:** the metric rewards ordering correct matches above
  distractors; a ranker learns exactly that objective, which pure
  binary classification does not.

To avoid leakage, the train/validation split is done at the **Source 1
entity level**, so a Source-1 record's candidates never appear in both splits
as the same query.

## 6. Threshold Tuning for F_0.5

After training, we sweep a grid of thresholds (0.05 … 0.95 in steps of 0.05)
on the validation split, compute macro F_0.5 per threshold, and pick the
threshold that maximises it. This directly optimises the competition metric,
including the singleton case: predicting an empty list for a Source-1 record
with no true matches earns a full 1.0 on that entity, so a high threshold is
usually preferred.

## 7. Reproducibility

```bash
# Install
pip install -r requirements.txt

# Run the full pipeline (trains + predicts on test)
python -m code.business_entity_resolution.run_pipeline --mode full

# Validate submission format
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The pipeline writes `output/matching_results.tsv`, `output/candidate_pairs.tsv`,
and `output/lgbm_model.txt` (the trained model). All randomness is seeded
with `RANDOM_SEED = 42`.

## 8. Why This Wins on F_0.5

- **Precision-first:** lambdarank + F_0.5-tuned threshold strongly
  penalises false merges.
- **Singleton aware:** high threshold produces empty lists for records with
  no plausible match, capturing full credit on singletons.
- **High recall ceiling:** seven complementary blocking keys mean true
  matches are almost never dropped before scoring.
- **Country-agnostic:** open-set country handling ensures the unseen France
  test records flow through the same pipeline without code changes.
- **Reproducible:** deterministic seeds, a single `run_pipeline.py` entry
  point, and MIT-licensed dependencies only.