# /// script
# dependencies = ["datasets", "huggingface-hub"]
# ///

import json
import random
import re
from datasets import load_dataset

COMMON_ENGLISH = {"the", "and", "was", "that", "for", "with", "this", "from", "have", "has", "are", "were", "been", "not", "but", "they", "which"}
MIN_HITS = 3
MIN_LEN = 300
MAX_LEN = 3000
STREAM_LIMIT = 10_000
SAMPLE_SIZE = 100
SEED = 123
OUTPUT = "/Users/dome/work/worldbank/gliner2_fix/event-location-extraction/benchmark/data/baai_news_100.json"


def is_english(text: str) -> bool:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    hits = sum(1 for w in words if w in COMMON_ENGLISH)
    return hits >= MIN_HITS


def main():
    print("Loading dataset in streaming mode...")
    ds = load_dataset("BAAI/IndustryCorpus_news", split="train", streaming=True)

    candidates = []
    seen = 0
    for row in ds:
        seen += 1
        if seen > STREAM_LIMIT:
            break
        text = row.get("text") or row.get("content") or ""
        text = text.strip()
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            continue
        if not is_english(text):
            continue
        candidates.append(text)

    print(f"Streamed {seen} rows, {len(candidates)} passed English + length filter")

    random.seed(SEED)
    sample = random.sample(candidates, min(SAMPLE_SIZE, len(candidates)))

    records = []
    for i, text in enumerate(sample):
        records.append({
            "id": f"article_{i:03d}",
            "text": text,
            "text_length": len(text),
        })

    with open(OUTPUT, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Summary
    lengths = [r["text_length"] for r in records]
    print(f"\nSaved {len(records)} articles to {OUTPUT}")
    print(f"Length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
    print("\n--- First 3 article previews ---")
    for r in records[:3]:
        preview = r["text"][:200].replace("\n", " ")
        print(f"  [{r['id']}] ({r['text_length']} chars) {preview}...")


if __name__ == "__main__":
    main()
