import argparse
import csv
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from tqdm import tqdm
from vllm import LLM

from src.index.inmemory import InMemoryIndexer

QUERY_POSSIBLE_KEY_NAMES = ["query", "question", "prompt", "input", "text"]


def extract_query_and_metadata(obj):
    if isinstance(obj, str):
        return obj.strip(), {}
    if isinstance(obj, dict):
        for key in QUERY_POSSIBLE_KEY_NAMES:
            if key in obj:
                if "metadata" in obj:
                    metadata = obj["metadata"]
                else:
                    metadata = {k: v for k, v in obj.items() if k != key}
                return str(obj[key]).strip(), metadata
        # fallback: check nested source.text
        if "source" in obj and isinstance(obj["source"], dict) and "text" in obj["source"]:
            metadata = {k: v for k, v in obj.items() if k != "source"}
            return str(obj["source"]["text"]).strip(), metadata
    return "", {}


def make_output_row(record):
    original_obj = record["original_obj"]
    if isinstance(original_obj, dict):
        return dict(original_obj)
    return {"query": record["query"]}


def load_prompt_from_st_config(
    model_name: str, prompt_name: str | None
) -> str | None:
    """Return the literal prompt string from config_sentence_transformers.json.

    Tries the HF local cache first (local_files_only), then a local directory
    path. Uses ``prompt_name`` to look up a specific key; falls back to
    ``default_prompt_name`` when ``prompt_name`` is None. Returns None if the
    config is unavailable or no matching prompt is found.
    """
    config: dict | None = None

    # Try HF hub cache (no network call)
    try:
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config_sentence_transformers.json",
            local_files_only=True,
        )
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (EntryNotFoundError, LocalEntryNotFoundError, Exception):
        pass

    # Try local directory path
    if config is None:
        local = Path(model_name) / "config_sentence_transformers.json"
        if local.exists():
            with open(local, encoding="utf-8") as f:
                config = json.load(f)

    if config is None:
        return None

    prompts: dict = config.get("prompts", {})
    key = prompt_name or config.get("default_prompt_name")
    if key:
        return prompts.get(key)
    return None


def apply_query_prompt(query: str, query_prompt: str | None) -> str:
    """Prepend a literal prompt string to a query before encoding.

    If the prompt contains a ``{query}`` placeholder it is formatted,
    otherwise it is used as a prefix (e.g. ``"query: "``).
    """
    if not query_prompt:
        return query
    if "{query}" in query_prompt:
        return query_prompt.format(query=query)
    return f"{query_prompt}{query}"


def load_query_records(queries: list[str]) -> list[dict]:
    query_records: list[dict] = []

    # If a single argument is provided and it's an existing file, treat it as a queries file
    if len(queries) == 1 and Path(queries[0]).exists():
        path = Path(queries[0])
        suffix = path.suffix.lower()

        if suffix == ".txt":
            query_records = [
                {"query": line.strip(), "original_obj": {"query": line.strip()}}
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        elif suffix == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                items = data
            elif (
                isinstance(data, dict)
                and "queries" in data
                and isinstance(data["queries"], list)
            ):
                items = data["queries"]
            else:
                raise ValueError(
                    "JSON must be a list of queries or a dict with key 'queries'"
                )
            for item in items:
                q, query_metadata = extract_query_and_metadata(item)
                if q:
                    query_records.append(
                        {
                            "query": q,
                            "query_metadata": query_metadata,
                            "original_obj": item,
                        }
                    )
        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    q, query_metadata = extract_query_and_metadata(obj)
                    if not q:
                        raise ValueError(
                            "Each JSONL line must be a string or object with one of: "
                            + ", ".join(QUERY_POSSIBLE_KEY_NAMES)
                        )
                    query_records.append(
                        {
                            "query": q,
                            "query_metadata": query_metadata,
                            "original_obj": obj,
                        }
                    )
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("CSV must include a header row")
                query_col = next(
                    (
                        name
                        for name in QUERY_POSSIBLE_KEY_NAMES
                        if name in reader.fieldnames
                    ),
                    None,
                )
                if not query_col:
                    raise ValueError(
                        "CSV header must include one of: "
                        + ", ".join(QUERY_POSSIBLE_KEY_NAMES)
                    )
                for row in reader:
                    q = str(row.get(query_col, "")).strip()
                    if q:
                        query_metadata = {
                            key: value for key, value in row.items() if key != query_col
                        }
                        query_records.append(
                            {
                                "query": q,
                                "query_metadata": query_metadata,
                                "original_obj": row,
                            }
                        )
        else:
            raise ValueError("Unsupported file type. Use .txt, .json, .jsonl, or .csv")
    else:
        query_records = [
            {"query": query, "original_obj": {"query": query}} for query in queries
        ]

    return query_records


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Retrieve passages for queries using a vLLM pooling (embedding) model"
    )
    arg_parser.add_argument(
        "--queries", type=str, nargs="+", help="List of queries to test", required=True
    )
    arg_parser.add_argument(
        "--index", type=str, help="Name of the index to use", required=True
    )
    arg_parser.add_argument(
        "--output",
        type=str,
        help="Output file to save retrieved passages",
        required=True,
    )
    arg_parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the vLLM pooling/embedding model to use for retrieval",
    )
    arg_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top passages to retrieve for each query",
    )
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of queries per search/write batch (vLLM batches encoding internally)",
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to host the index and run the search on (e.g., 'cpu', 'cuda')",
    )
    arg_parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Whether to L2-normalize the query embeddings before searching (default: False)",
    )
    arg_parser.add_argument(
        "--query-prompt-name",
        type=str,
        default=None,
        help=(
            "Name of a prompt registered in the model's config_sentence_transformers.json "
            "(e.g. 'web_search_query'). Looked up automatically from the local HF cache "
            "or a local model directory. When omitted, 'default_prompt_name' from the "
            "config is used if present. Mutually exclusive with --query-prompt."
        ),
    )
    arg_parser.add_argument(
        "--query-prompt",
        type=str,
        default=None,
        help=(
            "Literal prompt string applied to each query before encoding. "
            "If it contains a '{query}' placeholder it is formatted with the query, "
            "otherwise it is used as a prefix (e.g. 'query: '). "
            "Mutually exclusive with --query-prompt-name."
        ),
    )
    arg_parser.add_argument(
        "--save-ids-only",
        action="store_true",
        help="Whether to save only passage IDs instead of text",
    )
    # vLLM engine options
    arg_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism in vLLM",
    )
    arg_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM is allowed to use",
    )
    arg_parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length for the vLLM model (defaults to the model config)",
    )
    arg_parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for the vLLM model weights (e.g., 'auto', 'float16', 'bfloat16')",
    )
    arg_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow vLLM to execute custom modeling code from the model repo",
    )

    args = arg_parser.parse_args()

    if args.query_prompt and args.query_prompt_name:
        arg_parser.error("--query-prompt and --query-prompt-name are mutually exclusive")

    # Resolve the literal prompt string: explicit > inferred from ST config > None.
    query_prompt: str | None = args.query_prompt
    if query_prompt is None:
        query_prompt = load_prompt_from_st_config(args.model_name, args.query_prompt_name)
        if query_prompt is not None:
            key = args.query_prompt_name or "(default_prompt_name)"
            print(f"Using query prompt '{key}' from config_sentence_transformers.json")

    indexer = InMemoryIndexer.from_pretrained(args.index, device=args.device)

    # Load the embedding model with vLLM in pooling mode.
    llm = LLM(
        model=args.model_name,
        runner="pooling",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    query_records = load_query_records(args.queries)

    # Encode all queries in a single call; vLLM handles batching internally.
    # https://docs.vllm.ai/en/stable/models/pooling_models/#llmencode
    prompts = [
        apply_query_prompt(record["query"], query_prompt)
        for record in query_records
    ]
    pooling_outputs = llm.encode(prompts, pooling_task="embed")
    query_embeddings = torch.stack(
        [output.outputs.data.to(torch.float32).cpu() for output in pooling_outputs]
    )

    if args.normalize_embeddings:
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)

    # write the retrieved passages to the output file as jsonl with fields "query" and "candidates"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        with tqdm(total=len(query_records), desc="Retrieving", unit="query") as pbar:
            for i in range(0, len(query_records), args.batch_size):
                batch_records = query_records[i : i + args.batch_size]
                batch_embeddings = query_embeddings[i : i + args.batch_size]

                results = indexer.search(batch_embeddings, args.top_k)

                for record, passages in zip(batch_records, results):
                    output_row = make_output_row(record)

                    if args.save_ids_only:
                        output_row["candidates"] = [
                            p["document"]["metadata"]["estimate_id"] for p in passages
                        ]
                    else:
                        output_row["candidates"] = [
                            p["document"]["text"] for p in passages
                        ]

                    json_line = json.dumps(output_row, ensure_ascii=False)
                    f.write(json_line + "\n")

                pbar.update(len(batch_records))
