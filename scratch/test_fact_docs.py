import sys
import os
import json

sys.path.append("/Users/aishwaryashilpi/workspace/ragstack")
from scripts.run_rgb_eval import process_rgb_data

data_file = "/Users/aishwaryashilpi/workspace/ragstack/data/rgb/en_fact.json"
with open(data_file, "r") as f:
    instance = json.loads(f.readline())

print("--- Instance query:", instance['query'])
print("--- Instance answer:", instance['answer'])
print("--- Instance positive_wrong (len={}):".format(len(instance['positive_wrong'])), instance['positive_wrong'][0])

print("\n--- Running process_rgb_data with noise_rate=1.0, correct_rate=0.0 ---")
query, ans, docs = process_rgb_data(instance, noise_rate=1.0, passage_num=5, dataset_name="en_fact", correct_rate=0.0)
print("Docs generated (length={}):".format(len(docs)))
for i, d in enumerate(docs):
    print(f"  Doc {i+1}: {d[:100]}...")

print("\n--- Running process_rgb_data with noise_rate=0.0, correct_rate=0.0 ---")
query, ans, docs = process_rgb_data(instance, noise_rate=0.0, passage_num=5, dataset_name="en_fact", correct_rate=0.0)
print("Docs generated (length={}):".format(len(docs)))
for i, d in enumerate(docs):
    print(f"  Doc {i+1}: {d[:100]}...")
