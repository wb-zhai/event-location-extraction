"""vLLM batch relevance classification, streaming from GCS.

Input:  a manifest JSONL, one `{"id": <uri>, "gcs_path": "gs://bucket/uri.json"}` per line
        (produced by build_manifest.py). Each gcs_path points at the full article JSON
        (EventRegistry shape: top-level title/body, or nested source.title/source.text).
Output: a headerless CSV, one `id,label` per line, where label is `relevant` / `not relevant`.

The whole pipeline is streaming and memory-flat so it scales to millions of rows per shard:
a bounded pool of threads GETs article objects from GCS, texts are accumulated into batches
and handed to vLLM's pooling classifier, and results are appended to the CSV as each batch
finishes. Re-running on an existing output resumes (already-labeled ids are skipped).

This script is deliberately self-contained — it defines its own text builder, GCS reader,
and vLLM classifier wrapper and imports nothing from the rest of the repo.

Usage (single process, e.g. the timing test):
    python vertexai/inference/relevance/relevance_vllm_infer.py \
        --model_name_or_path /local/model \
        --input manifest-000.jsonl \
        --output /tmp/relevance.csv \
        --limit 10000

Pass --title_only to classify on the title alone (article body is ignored entirely,
overriding --max_chars); falls back to the text before the first blank line when an
article has no title.

Requires: fire, google-cloud-storage, vllm.
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import fire
from tqdm import tqdm

FALLBACK_ID2LABEL = {0: "irrelevant", 1: "relevant"}


# ---------------------------------------------------------------------------
# Text building + label mapping (self-contained)
# ---------------------------------------------------------------------------


def build_text(article: dict, max_chars: int, title_only: bool = False) -> str:
    """Title + body, mirroring how the classifier was trained. Handles both the raw
    EventRegistry shape (top-level title/body) and the repo's source.title/source.text.

    With title_only, only the title is used as model input (article text is ignored
    entirely). If the article has no title, falls back to the text before the first
    blank line and prints a warning."""
    source = article.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(article.get("title") or source.get("title") or "")
    if title_only:
        if title:
            return title
        text = str(article.get("body") or article.get("text") or source.get("text") or "")
        fallback_title, _, _ = text.partition("\n\n")
        print(
            f"--title-only: article {article.get('id')!r} has no 'title' field; "
            f"falling back to the text before the first blank line as the title: "
            f"{fallback_title[:120]!r}",
            file=sys.stderr,
        )
        return fallback_title
    text = str(article.get("body") or article.get("text") or source.get("text") or "")
    text = text[:max_chars]
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def to_output_label(model_label: str) -> str:
    """Map the model's positive class to `relevant`, everything else to `not relevant`."""
    return "relevant" if model_label == "relevant" else "not relevant"


# ---------------------------------------------------------------------------
# GCS reading (self-contained)
# ---------------------------------------------------------------------------


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


class ArticleFetcher:
    """Thread-safe article reader. One google.cloud.storage client, reused across threads."""

    def __init__(self, max_chars: int, title_only: bool = False, pool_size: int = 64):
        from google.cloud import storage
        from requests.adapters import HTTPAdapter

        self._client = storage.Client()
        # requests.adapters.HTTPAdapter defaults to a 10-connection pool, far below
        # gcs_read_concurrency worker threads sharing this one client -- past 10
        # concurrent requests, urllib3 can't reuse pooled connections and opens a
        # fresh one (full TLS handshake) per request instead, which made higher
        # concurrency *slower*. Size the pool to match so connections are reused.
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self._client._http.mount("https://", adapter)
        self._client._http.mount("http://", adapter)
        self.max_chars = max_chars
        self.title_only = title_only

    def fetch(self, row: dict) -> tuple[str, str] | None:
        """Return (id, text) or None if the object is missing / unreadable / empty."""
        article_id = str(row.get("id") or "")
        gcs_path = row.get("gcs_path")
        if not article_id or not gcs_path:
            return None
        # build_manifest.py includes title directly in the manifest -- in --title_only
        # mode that's the only field we need, so skip the GCS round trip entirely for
        # rows that already have one. Rows without one (older manifests, or a null
        # title in the DB) fall through to the normal per-article GCS fetch below.
        if self.title_only and row.get("title"):
            return article_id, str(row["title"])
        try:
            bucket_name, blob_name = _parse_gcs_uri(gcs_path)
            raw = self._client.bucket(bucket_name).blob(blob_name).download_as_text()
            article = json.loads(raw)
        except Exception:
            return None
        article.setdefault("id", article_id)
        text = build_text(article, self.max_chars, self.title_only)
        if not text.strip():
            return None
        return article_id, text


# ---------------------------------------------------------------------------
# vLLM classifier wrapper (self-contained)
# ---------------------------------------------------------------------------


class RelevanceClassifier:
    def __init__(self, model_name_or_path: str, max_length: int,
                 gpu_memory_utilization: float, tensor_parallel_size: int):
        from vllm import LLM

        self.max_length = max_length
        self.llm = LLM(
            model=model_name_or_path,
            runner="pooling",
            dtype="auto",
            max_model_len=max_length,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.id2label = self._load_id2label(model_name_or_path)

    @staticmethod
    def _load_id2label(model_name_or_path: str) -> dict[int, str]:
        config_path = Path(model_name_or_path) / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("id2label"):
                return {int(k): v for k, v in config["id2label"].items()}
        return dict(FALLBACK_ID2LABEL)

    def classify(self, texts: list[str]) -> list[str]:
        # Let vLLM tokenize internally instead of pre-tokenizing the whole batch in
        # Python (which serialized CPU tokenization ahead of every classify() call
        # and left the GPU idle). tokenization_kwargs still guarantees truncation to
        # max_length, so an oversized article can't make classify() raise
        # VLLMValidationError (char-based --max-chars doesn't bound token count).
        outputs = self.llm.classify(
            texts, tokenization_kwargs={"truncation": True, "max_length": self.max_length}
        )
        labels = []
        for output in outputs:
            probs = list(output.outputs.probs)
            pred_id = max(range(len(probs)), key=lambda i: probs[i])
            labels.append(to_output_label(self.id2label.get(pred_id, str(pred_id))))
        return labels


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def read_manifest(path: Path, shard_index: int, num_shards: int, limit: int | None) -> Iterator[dict]:
    """Stream manifest rows for this shard (line k kept when k % num_shards == shard_index).
    With num_shards == 1 every row is kept (the entrypoint hands each task its own manifest)."""
    yielded = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if num_shards > 1 and lineno % num_shards != shard_index:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            yielded += 1
            if limit and yielded >= limit:
                return


def load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if row:
                done.add(row[0])
    return done


def stream_fetched(rows: Iterator[dict], fetcher: ArticleFetcher, concurrency: int
                   ) -> Iterator[tuple[str, str]]:
    """Fetch articles concurrently with a bounded number of in-flight requests, so we
    never enqueue the whole shard at once. Yields (id, text) as each GET completes."""
    max_inflight = max(concurrency * 2, concurrency + 1)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        inflight: set = set()
        exhausted = False
        while True:
            while not exhausted and len(inflight) < max_inflight:
                try:
                    inflight.add(pool.submit(fetcher.fetch, next(rows)))
                except StopIteration:
                    exhausted = True
            if not inflight:
                break
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                result = fut.result()
                if result is not None:
                    yield result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def relevance_infer(
    model_name_or_path: str,
    input: str,
    output: str,
    shard_index: int = 0,
    num_shards: int = 1,
    max_chars: int = 4000,
    max_length: int = 4096,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    gcs_read_concurrency: int = 64,
    batch_size: int | None = 2000,
    limit: int | None = None,
    title_only: bool = False,
):
    """Classify articles from a manifest and write id,label CSV rows.

    batch_size=None disables batching (the whole input is classified in one batch).
    title_only=True uses only the article title as model input, ignoring the body
    entirely (overrides --max_chars)."""
    input_path = Path(input)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title_only:
        print("Preview truncation mode: title-only (article text is ignored)", file=sys.stderr)

    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} ids already labeled in {output_path}", file=sys.stderr)

    classifier = RelevanceClassifier(
        model_name_or_path, max_length, gpu_memory_utilization, tensor_parallel_size
    )
    fetcher = ArticleFetcher(max_chars, title_only, pool_size=gcs_read_concurrency)

    rows = (r for r in read_manifest(input_path, shard_index, num_shards, limit)
            if str(r.get("id") or "") not in done_ids)

    processed = 0
    pbar = tqdm(unit="article", desc=f"shard {shard_index}")

    with output_path.open("a", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        batch_ids: list[str] = []
        batch_texts: list[str] = []

        def flush():
            nonlocal processed
            if not batch_texts:
                return
            labels = classifier.classify(batch_texts)
            writer.writerows(zip(batch_ids, labels))
            out_f.flush()
            processed += len(batch_ids)
            pbar.update(len(batch_ids))
            batch_ids.clear()
            batch_texts.clear()

        for article_id, text in stream_fetched(rows, fetcher, gcs_read_concurrency):
            batch_ids.append(article_id)
            batch_texts.append(text)
            if batch_size is not None and len(batch_texts) >= batch_size:
                flush()
        flush()

    pbar.close()
    print(f"Done. {processed} labeled → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    fire.Fire(relevance_infer)
