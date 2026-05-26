import json
from scripts.data.generation.relevance_filter import food_insecurity_regex

c = 0
total = 0
with open('dataset/zhai/raw/sample_50000_with_tags_stratified_filtered.jsonl') as f:
    for line in f:
        total += 1
        r = json.loads(line)
        title = r.get('title', '')
        text = r.get('text', '')
        full = title + ' ' + text
        if food_insecurity_regex.search(full):
            c += 1

print(f"Matches: {c} / {total}")
