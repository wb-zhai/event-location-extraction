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
DEFAULT_DATA_PATH = ROOT / "dataset/rams/processed/gliner/dev.macro.jsonl"


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
    argument_types = sorted(
        {relation.name for example in dataset for relation in example.relations}
    )
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


def build_gold_event_predictions(text: str, gold_entities: dict) -> dict:
    event_preds = {}
    for event_type, mentions in (gold_entities or {}).items():
        spans = []
        for mention in mentions or []:
            for (
                start,
                end,
            ) in EventArgumentExtractionEvaluatorGliNER2._find_exact_matches(
                text, mention
            ):
                spans.append({"start": start, "end": end, "text": text[start:end]})
        if spans:
            event_preds[event_type] = spans
    return event_preds


def build_gold_argument_predictions(text: str, gold_relations) -> dict:
    relation_preds = {}
    for relation in gold_relations or []:
        if not isinstance(relation, dict):
            continue

        for role, payload in relation.items():
            if not isinstance(payload, dict):
                continue

            event_text = payload.get("event")
            argument_text = payload.get("argument")
            if not event_text or not argument_text:
                continue

            event_spans = EventArgumentExtractionEvaluatorGliNER2._find_exact_matches(
                text, event_text
            )
            argument_spans = (
                EventArgumentExtractionEvaluatorGliNER2._find_exact_matches(
                    text, argument_text
                )
            )
            for event_start, event_end in event_spans:
                for arg_start, arg_end in argument_spans:
                    relation_preds.setdefault(role, []).append(
                        {
                            "event": {
                                "start": event_start,
                                "end": event_end,
                                "text": text[event_start:event_end],
                            },
                            "argument": {
                                "start": arg_start,
                                "end": arg_end,
                                "text": text[arg_start:arg_end],
                            },
                        }
                    )
    return relation_preds


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
        "--prediction-source",
        choices=["model", "gold"],
        default="model",
        help=(
            "Use model outputs or gold-converted span outputs as predictions. "
            "'gold' is a sanity mode that should return perfect arg/event metrics "
            "when exact-match mapping succeeds."
        ),
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
    parser.add_argument(
        "--ontology-file",
        type=Path,
        help="Path to the ontology file defining event and argument types. If not provided, the schema will be automatically collected from the dataset.",
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

    if args.ontology_file:
        with open(args.ontology_file, "r") as f:
            schema = json.load(f)
        event_types = schema["macro_trigger_types"]
        argument_types = schema["roles"]
    else:
        event_types, argument_types = collect_schema(dataset)

    model_path = resolve_model_path(args.model)
    model = GLiNER2.from_pretrained(
        model_path, local_files_only=Path(model_path).exists(), map_location="cpu"
    )

    evaluator = EventArgumentExtractionEvaluatorGliNER2(
        event_types=event_types,
        argument_types=argument_types,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    schema = model.create_schema().entities(event_types).relations(argument_types)

    texts = [example.text for example in subset]

    if args.prediction_source == "model":
        model_path = resolve_model_path(args.model)
        model = GLiNER2.from_pretrained(
            model_path, local_files_only=Path(model_path).exists()
        )
        schema = model.create_schema().entities(event_types).relations(argument_types)
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
    else:
        results = []
        event_preds = []
        argument_preds = []
        for text, raw_record in zip(texts, subset_records):
            output = raw_record.get("output", {})
            gold_entities = output.get("entities", {})
            gold_relations = output.get("relations", [])
            event_pred = build_gold_event_predictions(text, gold_entities)
            argument_pred = build_gold_argument_predictions(text, gold_relations)

            event_preds.append(event_pred)
            argument_preds.append(argument_pred)
            results.append(
                {
                    "entities": event_pred,
                    "relation_extraction": argument_pred,
                }
            )
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

    # sample_ids = [record.get("metadata", {}).get("id") for record in subset_records]
    samples = []
    for text, raw_record, result in zip(texts, subset_records, results):
        output = raw_record.get("output", {})
        samples.append(
            {
                # "id": sample_id,
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
                # "sample_ids": sample_ids,
                "metrics": metrics,
                "samples": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
