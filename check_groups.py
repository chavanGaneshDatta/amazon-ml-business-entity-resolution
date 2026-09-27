import sys
sys.path.insert(0, '.')
from code.business_entity_resolution import data_loader, preprocessing

print("Loading data...")
s1_df, s2_df, s3_df = data_loader.load_train_sources()
print("Loaded")

s1_records = preprocessing.prepare_records(s1_df, is_source1=True)
s2_records = preprocessing.prepare_records(s2_df)
s3_records = preprocessing.prepare_records(s3_df)
s23_records = s2_records + s3_records

print("Generating candidates...")
cands = preprocessing.generate_candidates(s1_records, s23_records)
sizes = [len(c) for c in cands]

print("=" * 50)
print("Total S1:", len(sizes))
print("Groups with size 0:", sizes.count(0))
print("Groups with size 1:", sizes.count(1))
print("Groups with size 2+:", sum(1 for s in sizes if s >= 2))
print("=" * 50)