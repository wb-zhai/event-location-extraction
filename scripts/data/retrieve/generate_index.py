import argparse
import json
from pathlib import Path

from src.common.log import get_logger
from src.index.documents import DocumentStore
from src.index.inmemory import InMemoryIndexer
from src.retriever.gemini import GeminiRetriever
from src.retriever.hf_retriever import HuggingFaceRetriever
from src.retriever.sentence_tr_retriever import SentenceTransformersRetriever

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_path", type=Path, help="Path to the source JSONL file")
    parser.add_argument(
        "output_folder", type=Path, help="Path to the destination JSONL file"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the Gemini/Hugging Face/Sentence Transformer model to use for encoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for encoding documents (default: 128)",
    )
    parser.add_argument(
        "--output-dimensionality",
        type=int,
        default=1536,
        help="Dimensionality of the output vectors (default: 1536)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for indexing (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (default: 'cpu', e.g., 'cuda:0' for GPU)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Precision for model computations (default: '32', e.g., '16' for mixed precision)",
    )
    parser.add_argument(
        "--encode-concurrency",
        type=int,
        default=None,
        help=(
            "Number of concurrent encode requests. "
            "Defaults to --num-workers when omitted."
        ),
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Whether to L2-normalize the output embeddings (default: False)",
    )
    parser.add_argument(
        "--sentence-transformers",
        action="store_true",
        help="Whether to use Sentence Transformers instead of Hugging Face Transformers (default: False)",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default=None,
        help=(
            "Optional name of a prompt template to apply to passages before encoding. "
            "The template must be registered in the retriever's model under the key 'passage_prompt_name'."
        ),
    )
    args = parser.parse_args()

    index_path = args.index_path
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    # check if the index path exists
    if not index_path.exists():
        raise FileNotFoundError(f"Index path {index_path} does not exist.")

    with open(index_path, "r") as f:
        events = json.load(f)

    # convert events to documents
    documents = []
    for i, (event, desc) in enumerate(events["events"].items()):
        doc = {
            "id": i,
            "text": event,
            "metadata": {
                "description": desc,
            },
        }
        documents.append(doc)

    logger.info(f"Loading documents from {index_path}...")
    documents = DocumentStore.from_dict(documents)

    metadata_fields = ["description"]

    indexer = InMemoryIndexer(
        documents=documents,
        metadata_fields=metadata_fields,
        separator=": ",
        add_metadata_keys_to_text=False,
    )

    if "gemini" in args.model_name.lower():
        logger.info(f"Indexing {len(documents)} documents with Gemini embeddings...")
        retriever = GeminiRetriever(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=args.output_dimensionality,
        )
    else:
        logger.info(
            f"Indexing {len(documents)} documents with Hugging Face embeddings using model {
                args.model_name
            }..."
        )
        if args.sentence_transformers:
            retriever = SentenceTransformersRetriever(
                model=args.model_name,
                device=args.device,
                passage_prompt_name=args.prompt_name,
                normalize_embeddings=args.normalize_embeddings,
                model_kwargs={"dtype": "auto"},
            )
        else:
            logger.info(
                f"Indexing {len(documents)} documents with Hugging Face embeddings using model {args.model_name}..."
            )
            retriever = HuggingFaceRetriever(
                model=args.model_name,
                device=args.device,
                precision=args.precision,
            )

    indexer.index(
        retriever,
        batch_size=100,
        num_workers=args.num_workers,
        encode_concurrency=args.encode_concurrency,
        device=args.device,
    )

    indexer.save_pretrained(output_folder, mmap=True)

    logger.info(f"Wrote {len(documents)} records with vectors to {output_folder}")
