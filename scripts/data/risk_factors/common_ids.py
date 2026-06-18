import json

FILE_1 = "dataset/zhai/v3/science/dev.jsonl"
FILE_2 = "dataset/zhai/v3/science/quick_dev.jsonl"

with open(FILE_1, "r", encoding="utf-8") as f:
    data_1 = [json.loads(line) for line in f.readlines()]

with open(FILE_2, "r", encoding="utf-8") as f:
    data_2 = [json.loads(line) for line in f.readlines()]

common_ids = set([item["id"] for item in data_1]) & set([item["id"] for item in data_2])
print(f"Number of common IDs: {len(common_ids)}")
print(f"Length of dataset 1: {len(data_1)}")
print(f"Length of dataset 2: {len(data_2)}")