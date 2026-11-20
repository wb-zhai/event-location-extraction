import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from src.index.inmemory import InMemoryIndexer
from src.retriever.sentence_tr_retriever import SentenceTransformersRetriever

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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Test SentenceTransformersRetriever on sample queries"
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
        help="Name of the SentenceTransformers model to use for retrieval",
    )
    arg_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top passages to retrieve for each query",
    )
    arg_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for retrieval"
    )
    arg_parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for retrieval"
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run retrieval on (e.g., 'cpu', 'cuda')",
    )
    arg_parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Whether to L2-normalize the output embeddings (default: False)",
    )
    arg_parser.add_argument(
        "--query-prompt-name",
        type=str,
        default=None,
        help=(
            "Optional name of a prompt template to apply to queries before encoding. "
            "The template must be registered in the retriever's model under the key 'query_prompt_name'."
        ),
    )
    arg_parser.add_argument(
        "--save-ids-only",
        action="store_true",
        help="Whether to save only passage IDs instead of text",
    )

    args = arg_parser.parse_args()

    indexer = InMemoryIndexer.from_pretrained(args.index, device=args.device)
    retriever = SentenceTransformersRetriever(
        model=args.model_name,
        indexer=indexer,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        query_prompt_name=args.query_prompt_name,
    )

    # check if queries is a file path or a list of queries
    query_records = []

    # If a single argument is provided and it's an existing file, treat it as a queries file
    if len(args.queries) == 1 and Path(args.queries[0]).exists():
        path = Path(args.queries[0])
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
                for item in data:
                    q, query_metadata = extract_query_and_metadata(item)
                    if q:
                        query_records.append(
                            {
                                "query": q,
                                "query_metadata": query_metadata,
                                "original_obj": item,
                            }
                        )
            elif (
                isinstance(data, dict)
                and "queries" in data
                and isinstance(data["queries"], list)
            ):
                for item in data["queries"]:
                    q, query_metadata = extract_query_and_metadata(item)
                    if q:
                        query_records.append(
                            {
                                "query": q,
                                "query_metadata": query_metadata,
                                "original_obj": item,
                            }
                        )
            else:
                raise ValueError(
                    "JSON must be a list of queries or a dict with key 'queries'"
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
            {"query": query, "original_obj": {"query": query}} for query in args.queries
        ]

    # write the retrieved passages to the output file as jsonl with fields "query" and "passages"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        with tqdm(total=len(query_records), desc="Retrieving", unit="query") as pbar:
            for i in range(0, len(query_records), args.batch_size):
                batch_records = query_records[i : i + args.batch_size]
                batch_queries = [record["query"] for record in batch_records]
                results = retriever.retrieve(
                    batch_queries,
                    k=args.top_k,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    progress_bar=False,
                )

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
