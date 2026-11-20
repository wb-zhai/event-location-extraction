"""Score a location-extraction backend against a gold dataset.

Compares the *set* of predicted location strings to the *set* of gold
location strings per document (no character offsets required), reporting
micro- and macro-averaged precision/recall/F1. Always scores coarse
("location") labels; pass --fine to additionally break down by native label
where the dataset's schema is directly comparable to the backend's
(currently: spaCy GPE/LOC/FAC vs. OntoNotes 5).

Usage:
  uv run scripts/location/eval/run_eval.py \\
      --dataset zhai --backend spacy --limit 200

  uv run scripts/location/eval/run_eval.py \\
      --dataset ontonotes5 --backend spacy --fine --limit 200

  uv run scripts/location/eval/run_eval.py \\
      --dataset wnut17 --backend gliner2 --match fuzzy --limit 100
"""
# /// script
# dependencies = [
#   "gliner2>=1.2.6,<1.3.0",
#   "semantic-text-splitter",
#   "spacy>=3.8.0,<4.0.0",
#   "en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl",
#   "datasets",
#   "fire",
# ]
# ///
from __future__ import annotations

import json
import pathlib
import sys
import time

import fire

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.location.extractors import build_extractor, coarsen  # noqa: E402
from scripts.location.eval.loaders import DATASET_REGISTRY, BACKEND_FINE_SCHEMA  # noqa: E402
from scripts.location.eval.metrics import Accumulator, MATCH_TIERS, dedupe_locations, score_sets  # noqa: E402


def _load_docs(dataset: str, dataset_path: str | None, split: str, limit: int | None):
    loader = DATASET_REGISTRY.get(dataset)
    if loader is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {sorted(DATASET_REGISTRY)}.")
    if dataset == "zhai":
        return loader(path=dataset_path, limit=limit)
    return loader(split=split, limit=limit)


def main(
    dataset: str,
    backend: str,
    match: str = "normalized",
    fine: bool = False,
    limit: int | None = None,
    dataset_path: str | None = None,
    split: str = "test",
    model: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    report_dir: str = "scripts/location/eval/results",
) -> None:
    """Run one (dataset, backend) evaluation and write a results JSON.

    Args:
        dataset:      One of DATASET_REGISTRY keys ("zhai", "conll2003", "wnut17", "ontonotes5").
        backend:      "spacy" or "gliner2".
        match:        Matching tier — "exact", "normalized" (default), or "fuzzy".
        fine:         Also score per native label where schemas align (see BACKEND_FINE_SCHEMA).
        limit:        Evaluate at most this many documents.
        dataset_path: Override input path for the "zhai" dataset.
        split:        HF dataset split for benchmark datasets (default "test").
        model:        Override the default model name for the chosen backend.
        threshold:    Confidence threshold (gliner2 only).
        batch_size:   Texts per inference batch (spacy only).
        report_dir:   Directory for the results JSON.
    """
    if match not in MATCH_TIERS:
        raise ValueError(f"--match must be one of {MATCH_TIERS}, got {match!r}")

    print(f"Loading dataset '{dataset}'…")
    docs = _load_docs(dataset, dataset_path, split, limit)
    print(f"Loaded {len(docs)} documents.")

    extractor_kwargs: dict = {}
    if model:
        extractor_kwargs["model"] = model
    if backend == "spacy":
        extractor_kwargs["batch_size"] = batch_size
    if backend == "gliner2":
        extractor_kwargs["threshold"] = threshold

    print(f"Loading '{backend}' extractor…")
    extractor = build_extractor(backend, **extractor_kwargs)

    fine_schema = BACKEND_FINE_SCHEMA.get(backend)
    fine_docs = [d for d in docs if fine_schema and d.fine_schema == fine_schema]
    fine_supported = fine and bool(fine_docs)
    if fine and not fine_supported:
        print(
            f"Fine-grained skipped: backend '{backend}' has no label schema "
            f"comparable to dataset '{dataset}' (native fine schema: "
            f"{fine_schema or 'none'} vs. dataset schema: "
            f"{docs[0].fine_schema if docs else 'none'})."
        )

    t0 = time.perf_counter()
    predictions = extractor.extract_batch([d.text for d in docs])
    elapsed = time.perf_counter() - t0

    coarse_acc = Accumulator()
    fine_accs: dict[str, Accumulator] = {}
    errors: list[dict] = []

    for doc, mentions in zip(docs, predictions):
        pred_coarse = dedupe_locations([m.text for m in coarsen(mentions)])
        tp, fp, fn = score_sets(pred_coarse, doc.locations, tier=match)
        coarse_acc.add(tp, fp, fn)
        if (fp or fn) and len(errors) < 20:
            errors.append({
                "id": doc.id,
                "gold": doc.locations,
                "pred": pred_coarse,
                "tp": tp, "fp": fp, "fn": fn,
            })

        if fine_supported and doc.fine_schema == fine_schema:
            for label, gold_spans in doc.fine.items():
                pred_label = dedupe_locations([m.text for m in mentions if m.label == label])
                ltp, lfp, lfn = score_sets(pred_label, gold_spans, tier=match)
                fine_accs.setdefault(label, Accumulator()).add(ltp, lfp, lfn)

    coarse_micro = coarse_acc.micro()
    coarse_macro = coarse_acc.macro()

    print(f"\n{dataset} / {backend}  (match={match}, n={len(docs)} docs, {elapsed:.1f}s)")
    print(
        f"  coarse micro  P={coarse_micro['precision']:.3f} "
        f"R={coarse_micro['recall']:.3f} F1={coarse_micro['f1']:.3f} "
        f"(tp={coarse_micro['tp']} fp={coarse_micro['fp']} fn={coarse_micro['fn']})"
    )
    print(
        f"  coarse macro  P={coarse_macro['precision']:.3f} "
        f"R={coarse_macro['recall']:.3f} F1={coarse_macro['f1']:.3f}"
    )

    fine_report: dict[str, dict] = {}
    if fine_supported:
        for label, acc in sorted(fine_accs.items()):
            micro = acc.micro()
            fine_report[label] = {"micro": micro, "macro": acc.macro()}
            print(
                f"  fine[{label:<4}] micro  P={micro['precision']:.3f} "
                f"R={micro['recall']:.3f} F1={micro['f1']:.3f}"
            )

    report_path = pathlib.Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    out_file = report_path / f"{dataset}_{backend}_{match}.json"
    out_file.write_text(json.dumps({
        "dataset": dataset,
        "backend": backend,
        "match": match,
        "n_docs": len(docs),
        "elapsed_s": round(elapsed, 2),
        "coarse": {"micro": coarse_micro, "macro": coarse_macro},
        "fine": fine_report,
        "sample_errors": errors,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    fire.Fire(main)
