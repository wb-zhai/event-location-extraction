"""Small JSONL and path helpers for generation_v2 scripts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json_tolerant(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(cleaned)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line is not an object at {path}:{line_no}")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [normalize_input_record(row, index) for index, row in enumerate(iter_jsonl(path))]

    payload = load_json_tolerant(path)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("records", "samples", "articles", "data"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            rows = [payload]
    else:
        raise ValueError(f"Input JSON must contain object(s): {path}")

    return [normalize_input_record(row, index) for index, row in enumerate(rows)]


def normalize_input_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    source = record.get("source")
    source = source if isinstance(source, dict) else {}
    text = record.get("text") or record.get("body") or source.get("text") or ""
    return {
        **record,
        "id": str(record.get("id") or record.get("uri") or record.get("doc_id") or index),
        "title": str(record.get("title") or source.get("title") or ""),
        "text": str(text),
        "source_url": record.get("source_url") or record.get("url") or source.get("source_url"),
        "publish_date": record.get("publish_date") or record.get("published_at") or source.get("publish_date"),
    }


def completed_ids(path: Path, *, retry_failed: bool) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        record_id = row.get("id")
        if record_id is None:
            continue
        if retry_failed and row.get("status") == "error":
            continue
        done.add(str(record_id))
    return done


def completed_task_keys(path: Path, *, retry_failed: bool) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        if retry_failed and row.get("status") == "error":
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        key = metadata.get("task_key") or row.get("id")
        if key is not None:
            done.add(str(key))
    return done


def prune_retryable_error_rows(path: Path, *, retry_failed: bool) -> int:
    """Remove failed rows that will be retried before appending replacement rows."""
    if not retry_failed or not path.exists():
        return 0
    rows = list(iter_jsonl(path))
    kept_rows = [row for row in rows if row.get("status") != "error"]
    pruned_count = len(rows) - len(kept_rows)
    if pruned_count:
        write_jsonl(path, kept_rows, overwrite=True)
    return pruned_count


def has_completed_task_keys(path: Path, *, retry_failed: bool) -> bool:
    if not path.exists():
        return False
    for row in iter_jsonl(path):
        if retry_failed and row.get("status") == "error":
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("task_key"):
            return True
    return False


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
