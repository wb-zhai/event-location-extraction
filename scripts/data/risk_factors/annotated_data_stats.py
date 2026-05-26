import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Print statistics about annotated data."
    )
    arg_parser.add_argument(
        "input_path",
        type=Path,
    )
    args = arg_parser.parse_args()

    with args.input_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # count event roles, argument roles, and span texts
    event_roles: Counter[str] = Counter()
    argument_roles: Counter[str] = Counter()
    span_texts: Counter[str] = Counter()
    total_events = 0
    total_arguments = 0
    total_articles = len(records)
    articles_with_events = 0
    # count also the avg arguments per event

    for record in records:
        events = record.get("events", [])
        if events:
            articles_with_events += 1

        for event in events:
            event_type = str(event.get("event_type") or "").strip()
            if event_type:
                event_roles[event_type] += 1
                total_events += 1

            trigger_text = str(event.get("trigger_text") or "").strip()
            if trigger_text:
                span_texts[trigger_text] += 1

            arguments = event.get("arguments", [])
            for argument in arguments:
                role = str(argument.get("role") or "").strip()
                if role:
                    argument_roles[role] += 1
                    total_arguments += 1

    avg_events = total_events / total_articles if total_articles > 0 else 0
    avg_arguments = total_arguments / total_events if total_events > 0 else 0
    articles_no_events = total_articles - articles_with_events

    print(f"total articles: {total_articles}")
    print(f"articles with events: {articles_with_events}")
    print(f"articles without events: {articles_no_events}")
    print(f"total events: {total_events}")
    print(f"total arguments: {total_arguments}")
    print(f"average events per article: {avg_events:.2f}")
    print(f"average arguments per event: {avg_arguments:.2f}")

    # count the average number of windows each article is split into
    # info is in llm -> metadata -> long_document -> window_count
    total_windows = 0
    for record in records:
        llm_info: dict[str, Any] = record.get("llm", {})
        metadata: dict[str, Any] = llm_info.get("metadata", {})
        long_document: dict[str, Any] = metadata.get("long_document", {})
        window_count = long_document.get("window_count", 0)
        total_windows += window_count
    avg_windows = total_windows / total_articles if total_articles > 0 else 0
    print(f"average windows per article: {avg_windows:.2f}")

    # count the average number of chars in the article text
    total_chars = 0
    for record in records:
        text = record["source"].get("text", "")
        total_chars += len(text)
    avg_chars = total_chars / total_articles if total_articles > 0 else 0
    print(f"average chars per article: {avg_chars:.2f}")

    # count the average number of words in the article text
    total_words = 0
    for record in records:
        text = record["source"].get("text", "")
        total_words += len(text.split())
    avg_words = total_words / total_articles if total_articles > 0 else 0
    print(f"average words per article: {avg_words:.2f}")