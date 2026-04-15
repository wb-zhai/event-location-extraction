import argparse
import json
import sys
from pathlib import Path

from gliner2 import GLiNER2
from gliner2.training.data import TrainingDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.eval import EventArgumentExtractionEvaluatorGliNER2


DEFAULT_MODEL_ID = "fastino/gliner2-base-v1"
DEFAULT_DATA_PATH = ROOT / "dataset/rams/processed/gliner/dev.jsonl"


def resolve_model_path(model_ref: str) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path)

    if "/" not in model_ref:
        return model_ref

    namespace, model_name = model_ref.split("/", maxsplit=1)
    cache_root = (
        Path.home() / ".cache/huggingface/hub" / f"models--{namespace}--{model_name}"
    )
    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])

    return model_ref


def collect_schema(dataset: TrainingDataset) -> tuple[list[str], list[str]]:
    event_types = sorted(
        {event_type for example in dataset for event_type in example.entities.keys()}
    )
    argument_types = sorted({relation.name for example in dataset for relation in example.relations})
    return event_types, argument_types


def load_jsonl_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def drop_empty_values(data: dict) -> dict:
    return {key: value for key, value in data.items() if value}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test EventArgumentExtractionEvaluatorGliNER2 on RAMS."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Model id or local snapshot path (default: fastino/gliner2-base-v1).",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the GLiNER-formatted RAMS dev split.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index inside the dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size passed to model.batch_extract.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers passed to model.batch_extract.",
    )
    args = parser.parse_args()

    dataset = TrainingDataset.load(args.data_file)
    raw_records = load_jsonl_records(args.data_file)
    subset_examples = dataset.examples[args.offset : args.offset + args.num_samples]
    subset_records = raw_records[args.offset : args.offset + args.num_samples]
    if not subset_examples:
        raise ValueError(
            f"No samples selected from {args.data_file} with offset={args.offset} "
            f"and num_samples={args.num_samples}"
        )

    subset = TrainingDataset(subset_examples)
    event_types, argument_types = collect_schema(dataset)

    model_path = resolve_model_path(args.model)
    model = GLiNER2.from_pretrained(
        model_path, local_files_only=Path(model_path).exists()
    )

    evaluator = EventArgumentExtractionEvaluatorGliNER2(
        event_types=event_types,
        argument_types=argument_types,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    schema = model.create_schema().entities(event_types).relations(argument_types)
    texts = [example.text for example in subset]
    results = model.batch_extract(
        texts,
        schema,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_confidence=True,
        include_spans=True,
    )

    event_preds = [sample["entities"] for sample in results]
    argument_preds = [sample["relation_extraction"] for sample in results]
    event_golds = [sample.entities for sample in subset]
    argument_golds = [sample.relations for sample in subset]

    event_metrics = evaluator.eval_events(texts, event_preds, event_golds)
    argument_metrics = evaluator.eval_arguments(texts, argument_preds, argument_golds)
    metrics = {
        "event_span_f1": event_metrics["span_f1"],
        "event_span_precision": event_metrics["span_precision"],
        "event_span_recall": event_metrics["span_recall"],
        "event_f1": event_metrics["f1"],
        "event_precision": event_metrics["precision"],
        "event_recall": event_metrics["recall"],
        "arg_i_precision": argument_metrics["arg_i_precision"],
        "arg_i_recall": argument_metrics["arg_i_recall"],
        "arg_i_f1": argument_metrics["arg_i_f1"],
        "arg_c_precision": argument_metrics["arg_c_precision"],
        "arg_c_recall": argument_metrics["arg_c_recall"],
        "arg_c_f1": argument_metrics["arg_c_f1"],
    }

    sample_ids = [record.get("metadata", {}).get("id") for record in subset_records]
    samples = []
    for sample_id, text, raw_record, result in zip(sample_ids, texts, subset_records, results):
        output = raw_record.get("output", {})
        samples.append(
            {
                "id": sample_id,
                "text": text,
                "gold": {
                    "entities": output.get("entities", {}),
                    "relations": output.get("relations", []),
                },
                "prediction": {
                    "entities": drop_empty_values(result.get("entities", {})),
                    "relations": drop_empty_values(
                        result.get("relation_extraction", {})
                    ),
                },
            }
        )

    print(
        json.dumps(
            {
                "offset": args.offset,
                "num_samples": len(subset),
                "sample_ids": sample_ids,
                "metrics": metrics,
                "samples": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
