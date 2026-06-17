import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import urlparse


def get_text(record: dict[str, Any]) -> str:
    """Return article text whether nested under "source" or at the top level."""
    source = record.get("source")
    if isinstance(source, dict) and "text" in source:
        return source.get("text", "") or ""
    return record.get("text", "") or ""


def get_source_url(record: dict[str, Any]) -> str:
    """Return source url whether nested under "source" or at the top level."""
    source = record.get("source")
    if isinstance(source, dict) and (source.get("source_url") or source.get("source_uri")):
        return source.get("source_url") or source.get("source_uri") or ""
    return record.get("source_url") or record.get("source_uri") or ""


def percentile(sorted_values: list[int], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def print_length_distribution(label: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"{label} length stats: none")
        return

    sorted_lengths = sorted(lengths)
    bucket_counts: Counter[str] = Counter()
    for length in lengths:
        if length <= 10:
            bucket_counts["1-10"] += 1
        elif length <= 20:
            bucket_counts["11-20"] += 1
        elif length <= 50:
            bucket_counts["21-50"] += 1
        elif length <= 100:
            bucket_counts["51-100"] += 1
        else:
            bucket_counts["101+"] += 1

    print(f"{label} length stats:")
    print(f"  count: {len(lengths)}")
    print(f"  min / median / avg / max: {sorted_lengths[0]} / {median(sorted_lengths):.1f} / {sum(lengths) / len(lengths):.1f} / {sorted_lengths[-1]}")
    print(
        "  p90 / p95 / p99: "
        f"{percentile(sorted_lengths, 0.90):.1f} / "
        f"{percentile(sorted_lengths, 0.95):.1f} / "
        f"{percentile(sorted_lengths, 0.99):.1f}"
    )
    print(
        "  buckets: "
        + ", ".join(
            f"{bucket}={bucket_counts.get(bucket, 0)}"
            for bucket in ("1-10", "11-20", "21-50", "51-100", "101+")
        )
    )


def print_longest_examples(label: str, spans: list[dict[str, Any]], limit: int = 5) -> None:
    print(f"longest {label}:")
    if not spans:
        print("  none")
        return

    seen: set[tuple[str, str]] = set()
    examples_printed = 0
    for span in sorted(
        spans,
        key=lambda item: (-item["length"], item["label"], item["text"], item["record_id"]),
    ):
        dedupe_key = (span["label"], span["text"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        print(
            f"  {span['length']:>3} chars | {span['label']} | "
            f"{span['record_id']} | {span['text']}"
        )
        examples_printed += 1
        if examples_printed >= limit:
            break


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
    trigger_lengths: list[int] = []
    argument_lengths: list[int] = []
    longest_spans: list[dict[str, Any]] = []
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
                trigger_lengths.append(len(trigger_text))
                longest_spans.append(
                    {
                        "length": len(trigger_text),
                        "kind": "trigger",
                        "label": event_type or "unknown",
                        "text": trigger_text,
                        "record_id": record.get("id", ""),
                    }
                )

            arguments = event.get("arguments", [])
            for argument in arguments:
                role = str(argument.get("role") or "").strip()
                if role:
                    argument_roles[role] += 1
                    total_arguments += 1

                argument_text = str(argument.get("text") or "").strip()
                if argument_text:
                    argument_lengths.append(len(argument_text))
                    longest_spans.append(
                        {
                            "length": len(argument_text),
                            "kind": "argument",
                            "label": role or "unknown",
                            "text": argument_text,
                            "record_id": record.get("id", ""),
                        }
                    )

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
        text = get_text(record)
        total_chars += len(text)
    avg_chars = total_chars / total_articles if total_articles > 0 else 0
    print(f"average chars per article: {avg_chars:.2f}")

    # count the average number of words in the article text
    total_words = 0
    for record in records:
        text = get_text(record)
        total_words += len(text.split())
    avg_words = total_words / total_articles if total_articles > 0 else 0
    print(f"average words per article: {avg_words:.2f}")

    domain_counts: Counter[str] = Counter()
    for record in records:
        url = get_source_url(record)
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0] or "(unknown)"
        domain_counts[domain] += 1
    print(f"domain distribution ({len(domain_counts)} unique):")
    for domain, count in domain_counts.most_common(20):
        pct = count / total_articles * 100
        print(f"  {count:>5} ({pct:5.1f}%)  {domain}")

    print_length_distribution("trigger span", trigger_lengths)
    print_length_distribution("argument span", argument_lengths)

    print_longest_examples(
        "trigger spans",
        [span for span in longest_spans if span["kind"] == "trigger"],
    )
    print_longest_examples(
        "argument spans",
        [span for span in longest_spans if span["kind"] == "argument"],
    )
