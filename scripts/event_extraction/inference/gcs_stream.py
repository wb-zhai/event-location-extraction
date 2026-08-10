"""Stream articles from GCS for manifest-driven batch inference.

A manifest line (built by vertexai/inference/event-extraction/build_manifest.py) carries
only DB metadata:

    {"id": <uri>, "gcs_path": "gs://bucket/uri.json",
     "publish_date": "2017-01-07 19:57:00", "language": "eng"}

`fetch()` turns one of those into the row shape vllm_infer.py expects, pulling the article
text from its GCS object (one GET per article -- the object holds title and body together):

    {"id": ..., "language": ..., "source": {"text": ..., "publish_date": ...}}

The reader is streaming and memory-flat so it scales to millions of rows per shard: a
bounded pool of threads GETs article objects while the main thread is busy in vLLM, and
rows are yielded as each GET completes.

Requires: google-cloud-storage.
"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlparse


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def read_manifest(
    path: Path, shard_index: int = 0, num_shards: int = 1, limit: int | None = None
) -> Iterator[dict]:
    """Stream manifest rows for this shard (line k kept when k % num_shards == shard_index).

    With num_shards == 1 every row is kept -- that is the Cloud Batch case, where the
    entrypoint hands each task its own manifest file. The modulo path is for local runs
    that shard a single file across processes.
    """
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


class ArticleFetcher:
    """Thread-safe article reader. One google.cloud.storage client, reused across threads."""

    def __init__(self, pool_size: int = 64):
        from google.cloud import storage
        from requests.adapters import HTTPAdapter

        self._client = storage.Client()
        # requests.adapters.HTTPAdapter defaults to a 10-connection pool, far below the
        # worker threads sharing this one client -- past 10 concurrent requests, urllib3
        # can't reuse pooled connections and opens a fresh one (full TLS handshake) per
        # request instead, which makes higher concurrency *slower*. Size the pool to match.
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self._client._http.mount("https://", adapter)
        self._client._http.mount("http://", adapter)

    def fetch(self, row: dict) -> dict | None:
        """Return an inference row, or None if the object is missing / unreadable / empty."""
        article_id = str(row.get("id") or "")
        gcs_path = row.get("gcs_path")
        if not article_id or not gcs_path:
            return None
        try:
            bucket_name, blob_name = _parse_gcs_uri(gcs_path)
            raw = self._client.bucket(bucket_name).blob(blob_name).download_as_text()
            article = json.loads(raw)
        except Exception:
            return None

        # Handles both the raw EventRegistry shape (top-level body/text) and the repo's
        # nested source.text.
        source = article.get("source") or {}
        if not isinstance(source, dict):
            source = {}
        text = str(article.get("body") or article.get("text") or source.get("text") or "")
        if not text.strip():
            return None

        # publish_date and language come from the manifest (i.e. from the DB), not from the
        # GCS object, so they match the format the model was trained on.
        return {
            "id": article_id,
            "language": str(row.get("language") or "eng"),
            "source": {
                "text": text,
                "publish_date": str(row.get("publish_date") or ""),
            },
        }


def stream_fetched(
    rows: Iterable[dict], fetcher: ArticleFetcher, concurrency: int, max_inflight: int | None = None
) -> Iterator[dict]:
    """Fetch articles concurrently with a bounded number of in-flight requests, so we never
    enqueue the whole shard at once. Yields rows as each GET completes.

    max_inflight should be at least a batch's worth: the consumer blocks for minutes inside
    llm.generate(), and anything already submitted keeps downloading during that window, so
    the next batch is ready the moment the GPU frees up. Defaults to 2x concurrency.
    """
    rows = iter(rows)
    if max_inflight is None:
        max_inflight = concurrency * 2
    max_inflight = max(max_inflight, concurrency + 1)

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
