"""Location extraction inference — folder of JSONL → folder of JSONL.

Reads every *.jsonl in input_dir (one JSON dict per line, must have a "text" field),
extracts location mentions with the chosen backend, and writes mirrored output files
to output_dir with a "locations" key added to each row.

Resumable: re-running on an existing output_dir skips already-processed rows
(matched by SHA-256 hash of "text"). Pass --overwrite to redo everything.

Usage:
  uv run scripts/location/run_inference.py \\
      --input_dir data/in \\
      --output_dir data/out \\
      --backend spacy              # or gliner2

  uv run scripts/location/run_inference.py \\
      --input_dir data/in \\
      --output_dir data/out \\
      --backend gliner2 \\
      --threshold 0.5 \\
      --coarse
"""
# /// script
# dependencies = [
#   "gliner2>=1.2.6,<1.3.0",
#   "semantic-text-splitter",
#   "spacy>=3.8.0,<4.0.0",
#   "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl",
#   "fire",
#   "tqdm",
# ]
# ///
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import fire

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation_v3.io_utils import append_jsonl_row, iter_jsonl  # noqa: E402
from scripts.location.extractors import build_extractor, coarsen             # noqa: E402


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_processed_keys(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for row in iter_jsonl(path):
        text = row.get("text") or ""
        if text:
            keys.add(_text_key(text))
    return keys


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main(
    input_dir: str,
    output_dir: str,
    backend: str = "gliner2",
    model: str | None = None,
    batch_size: int = 32,
    threshold: float = 0.5,
    coarse: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    """Extract location mentions from JSONL files.

    Args:
        input_dir:   Folder containing *.jsonl input files.
        output_dir:  Folder for mirrored output files.
        backend:     "gliner2" or "spacy".
        model:       Override the default model name for the chosen backend.
        batch_size:  Number of texts per inference batch.
        threshold:   Confidence threshold (gliner2 only).
        coarse:      Collapse all labels to "location".
        overwrite:   Re-process rows that already appear in output files.
        limit:       Process at most this many rows total (for testing).
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No *.jsonl files found in {input_dir}")
        return

    # Build extractor kwargs — only pass non-None / relevant values.
    extractor_kwargs: dict = {"batch_size": batch_size}
    if model:
        extractor_kwargs["model"] = model
    if backend == "gliner2":
        extractor_kwargs["threshold"] = threshold

    print(f"Loading {backend} extractor…")
    extractor = build_extractor(backend, **extractor_kwargs)
    print(f"Loaded {extractor.name}.\n")

    total_rows = 0
    total_mentions = 0
    t0 = time.perf_counter()

    for file_idx, in_file in enumerate(jsonl_files):
        out_file = output_path / in_file.name
        done_keys = set() if overwrite else _load_processed_keys(out_file)

        rows = list(iter_jsonl(in_file))
        pending = [r for r in rows if _text_key(r.get("text") or "") not in done_keys]

        if limit is not None:
            remaining = limit - total_rows
            pending = pending[:remaining]

        skipped = len(rows) - len(pending)
        print(
            f"[{file_idx + 1}/{len(jsonl_files)}] {in_file.name} — "
            f"{len(pending)} to process, {skipped} skipped"
        )

        processed = 0
        for batch in _batched(pending, batch_size):
            texts = [r.get("text") or "" for r in batch]
            results = extractor.extract_batch(texts)

            for row, mentions in zip(batch, results):
                if coarse:
                    mentions = coarsen(mentions)
                row["locations"] = [m.to_dict() for m in mentions]
                append_jsonl_row(out_file, row)
                total_mentions += len(mentions)

            processed += len(batch)
            total_rows += len(batch)
            print(
                f"\r    [{processed}/{len(pending)}] rows done",
                end="",
                flush=True,
            )

            if limit is not None and total_rows >= limit:
                break

        print()  # newline after \r progress

        if limit is not None and total_rows >= limit:
            print(f"Reached row limit ({limit}), stopping early.")
            break

    elapsed = time.perf_counter() - t0
    dps = total_rows / elapsed if elapsed > 0 else 0
    print(
        f"\nDone — {total_rows} rows, {total_mentions} mentions, "
        f"{elapsed:.1f}s ({dps:.0f} docs/sec)"
    )


if __name__ == "__main__":
    fire.Fire(main)
