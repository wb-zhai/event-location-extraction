import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_ONTOLOGY = REPO_ROOT / "ontologies/zhai/risk.label.description.training.json"
DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_MAX_TEXT_CHARS = 8000
# https://ai.google.dev/gemini-api/docs/pricing — verify before use
PRICE_PER_1K_TOKENS = {"gemini-embedding-001": 0.00015}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _embed(
    client: genai.Client,
    texts: list[str],
    task_type: str,
    model: str,
    batch_size: int,
    workers: int = 1,
) -> np.ndarray:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def _call(batch: list[str]) -> np.ndarray:
        resp = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_call, batches))
    return np.concatenate(results, axis=0)


def _text_windows(text: str, window_size: int) -> list[str]:
    if len(text) <= window_size:
        return [text]
    return [
        text[i : i + window_size]
        for i in range(0, len(text), window_size)
        if text[i : i + window_size]
    ]


def _embed_per_window(
    client: genai.Client,
    records: list[dict],
    model: str,
    batch_size: int,
    workers: int,
    max_text_chars: int,
) -> list[np.ndarray]:
    """Embed each record as one vector per text window; return list of (n_windows, dim) arrays."""
    window_texts: list[str] = []
    window_record_idx: list[int] = []
    for r_idx, r in enumerate(records):
        for w in _text_windows(r.get("text", ""), max_text_chars):
            window_texts.append(w)
            window_record_idx.append(r_idx)

    all_vecs = _embed(
        client, window_texts, "RETRIEVAL_QUERY", model, batch_size, workers
    )

    per_record: list[list[np.ndarray]] = [[] for _ in records]
    for vec, r_idx in zip(all_vecs, window_record_idx):
        per_record[r_idx].append(vec)

    return [np.stack(vecs) for vecs in per_record]


def build_index(
    client: genai.Client,
    ontology_path: Path,
    index_path: Path,
    model: str,
    batch_size: int,
):
    events = json.loads(ontology_path.read_text())["events"]
    labels = list(events.keys())
    texts = [f"{name}: {desc}" for name, desc in events.items()]
    LOGGER.info("Building label index (%d labels)...", len(labels))
    index_tokens = client.models.count_tokens(model=model, contents=texts).total_tokens
    vectors = _embed(client, texts, "RETRIEVAL_DOCUMENT", model, batch_size)
    np.savez(index_path, labels=np.array(labels), vectors=vectors)
    LOGGER.info("Saved index to %s", index_path)
    return np.array(labels), vectors, index_tokens


def load_index(index_path: Path):
    LOGGER.info("Loading index from %s", index_path)
    data = np.load(index_path, allow_pickle=True)
    return data["labels"], data["vectors"]


def main():
    parser = argparse.ArgumentParser(description="Top-k label retriever for articles")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY, type=Path)
    parser.add_argument(
        "--index", required=True, type=Path, help="Path to .npz label vector cache"
    )
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument(
        "--workers", default=1, type=int, help="Parallel Gemini API calls"
    )
    parser.add_argument(
        "--max-text-chars",
        default=DEFAULT_MAX_TEXT_CHARS,
        type=int,
        help="Max characters of article text passed to the embedding model",
    )
    parser.add_argument(
        "--query-mode",
        default="full_doc",
        choices=["full_doc", "per_window"],
        help="full_doc: embed truncated article once; per_window: embed each chunk and take max score",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Ignore existing output and start over"
    )
    args = parser.parse_args()

    load_env_file(REPO_ROOT / ".env")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    index_tokens = 0
    if args.index.exists():
        labels, label_vectors = load_index(args.index)
    else:
        labels, label_vectors, index_tokens = build_index(
            client, args.ontology, args.index, args.model, args.batch_size
        )

    records = []
    with args.input.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    already_done = 0
    write_mode = "w"
    if not args.overwrite and args.output.exists():
        with args.output.open() as f:
            already_done = sum(1 for line in f if line.strip())
        if already_done:
            LOGGER.info(
                "Resuming: skipping %d already-processed records.", already_done
            )
            write_mode = "a"

    pending = records[already_done:]
    if not pending:
        LOGGER.info("Nothing to do — all %d records already processed.", len(records))
        return

    LOGGER.info(
        "Embedding %d articles in %s mode (workers=%d)...",
        len(pending),
        args.query_mode,
        args.workers,
    )
    with args.output.open(write_mode) as out:
        if args.query_mode == "per_window":
            per_record_vecs = _embed_per_window(
                client,
                pending,
                args.model,
                args.batch_size,
                args.workers,
                args.max_text_chars,
            )
            all_windows = [
                w
                for r in pending
                for w in _text_windows(r.get("text", ""), args.max_text_chars)
            ]
            article_tokens = (
                client.models.count_tokens(
                    model=args.model, contents=all_windows
                ).total_tokens
                if all_windows
                else 0
            )
            for record, window_vecs in tqdm(
                zip(pending, per_record_vecs), total=len(pending)
            ):
                # scores shape: (n_windows, n_labels) → take max across windows
                scores = (label_vectors @ window_vecs.T).max(axis=1)
                top_indices = np.argsort(scores)[::-1][: args.top_k]
                record["candidates"] = [labels[i] for i in top_indices]
                out.write(json.dumps(record) + "\n")
        else:
            texts = [r.get("text", "")[: args.max_text_chars] for r in pending]
            article_tokens = client.models.count_tokens(
                model=args.model, contents=texts
            ).total_tokens
            article_vectors = _embed(
                client,
                texts,
                "RETRIEVAL_QUERY",
                args.model,
                args.batch_size,
                args.workers,
            )
            for record, qvec in tqdm(zip(pending, article_vectors), total=len(pending)):
                scores = label_vectors @ qvec
                top_indices = np.argsort(scores)[::-1][: args.top_k]
                record["candidates"] = [labels[i] for i in top_indices]
                out.write(json.dumps(record) + "\n")

    total_tokens = index_tokens + article_tokens
    price = PRICE_PER_1K_TOKENS.get(args.model, 0.0)
    cost = total_tokens / 1000 * price
    LOGGER.info(
        "\n--- Summary ---\n"
        "  Records     : %d\n"
        "  Tokens      : %s\n"
        "  Cost        : $%.6f\n"
        "  Output      : %s",
        len(pending),
        f"{total_tokens:,}",
        cost,
        args.output,
    )


if __name__ == "__main__":
    main()
