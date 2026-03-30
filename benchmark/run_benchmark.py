# /// script
# dependencies = ["mlx-lm", "sentence-transformers"]
# ///

"""
Modular benchmarking framework for event-location extraction.

Usage:
  uv run benchmark/run_benchmark.py [OPTIONS]

Options:
  --models MODEL1,MODEL2    Comma-separated model names (default: all Qwen3.5)
  --dataset PATH            Path to annotated dataset JSON (default: BAAI gold)
  --max-articles N          Max articles to evaluate (default: all)
  --match-mode MODE         strict|relaxed (default: relaxed)
  --output DIR              Results output directory (default: benchmark/results)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from eval_metrics import (
    MetricsAccumulator, match_entity_sets, match_pair_sets,
    combined_similarity, semantic_similarity, make_match_fn,
)
from model_adapter import (
    MLXAdapter, ExtractionResult, get_mlx_models_to_benchmark,
    get_cross_family_models, get_2026_models, get_wave3_models,
    get_wave4_models, parse_extraction_output,
)


def load_dataset(path: Path) -> list[dict]:
    """Load annotated dataset. Expects list of {id, text, entities, event_location_pairs}."""
    with open(path) as f:
        data = json.load(f)
    return data


def evaluate_model(
    adapter,
    dataset: list[dict],
    score_fn,
    threshold: float,
    max_articles: int | None = None,
) -> dict:
    """Run a model on a dataset and compute metrics."""

    ent_acc = MetricsAccumulator()
    pair_acc = MetricsAccumulator()
    total_time = 0.0
    article_results = []
    errors = 0

    items = dataset[:max_articles] if max_articles else dataset

    for i, item in enumerate(items):
        aid = item.get("id", f"article_{i}")
        text = item.get("text", "")

        gold_entities = [(e[0].lower(), e[1].lower()) for e in item.get("entities", [])]
        gold_pairs = [(p[0].lower(), p[1].lower()) for p in item.get("event_location_pairs", [])]

        print(f"\r    [{i+1:3d}/{len(items)}] {aid}", end="", flush=True)

        try:
            result = adapter.extract(text)
        except Exception as e:
            print(f" ERROR: {e}")
            errors += 1
            continue

        total_time += result.inference_time_s

        # Entity evaluation
        e_tp, e_fp, e_fn = match_entity_sets(
            result.entities, gold_entities,
            score_fn=score_fn, threshold=threshold,
        )
        ent_acc.add(e_tp, e_fp, e_fn)

        # Pair evaluation
        p_tp, p_fp, p_fn = match_pair_sets(
            result.event_location_pairs, gold_pairs,
            score_fn=score_fn, threshold=threshold,
        )
        pair_acc.add(p_tp, p_fp, p_fn)

        article_results.append({
            "id": aid,
            "inference_time_s": round(result.inference_time_s, 3),
            "pred_entities": len(result.entities),
            "pred_pairs": len(result.event_location_pairs),
            "gold_entities": len(gold_entities),
            "gold_pairs": len(gold_pairs),
            "entity_tp": e_tp, "entity_fp": e_fp, "entity_fn": e_fn,
            "pair_tp": p_tp, "pair_fp": p_fp, "pair_fn": p_fn,
        })

    print()

    articles_evaluated = len(items) - errors
    avg_time = total_time / articles_evaluated if articles_evaluated > 0 else 0

    return {
        "model": adapter.name,
        "articles_evaluated": articles_evaluated,
        "errors": errors,
        "total_inference_time_s": round(total_time, 2),
        "avg_inference_time_s": round(avg_time, 3),
        "entity_metrics": ent_acc.to_dict(),
        "pair_metrics": pair_acc.to_dict(),
        "per_article": article_results,
    }


def generate_markdown_report(all_results: list[dict], output_dir: Path, match_mode: str) -> str:
    """Generate a markdown report comparing all models."""

    lines = [
        "# Event-Location Extraction Benchmark Report",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Match mode**: {match_mode}",
        f"**Articles evaluated**: {all_results[0]['articles_evaluated'] if all_results else 0}",
        "",
        "## Entity Extraction (micro-averaged)",
        "",
        "| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in sorted(all_results, key=lambda x: x["entity_metrics"]["F1"], reverse=True):
        m = r["entity_metrics"]
        t = r["avg_inference_time_s"]
        lines.append(
            f"| {r['model']} | {m['TP']} | {m['FP']} | {m['FN']} | "
            f"{m['Precision']:.3f} | {m['Recall']:.3f} | **{m['F1']:.3f}** | "
            f"{m['F0.5']:.3f} | {m['F2']:.3f} | {m['Jaccard']:.3f} | {t:.2f}s |"
        )

    lines += [
        "",
        "## Event-Location Linking (micro-averaged)",
        "",
        "| Model | TP | FP | FN | Prec | Rec | **F1** | F0.5 | F2 | Jaccard | Avg Time |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in sorted(all_results, key=lambda x: x["pair_metrics"]["F1"], reverse=True):
        m = r["pair_metrics"]
        t = r["avg_inference_time_s"]
        lines.append(
            f"| {r['model']} | {m['TP']} | {m['FP']} | {m['FN']} | "
            f"{m['Precision']:.3f} | {m['Recall']:.3f} | **{m['F1']:.3f}** | "
            f"{m['F0.5']:.3f} | {m['F2']:.3f} | {m['Jaccard']:.3f} | {t:.2f}s |"
        )

    # Best model summary
    if all_results:
        best_ent = max(all_results, key=lambda x: x["entity_metrics"]["F1"])
        best_pair = max(all_results, key=lambda x: x["pair_metrics"]["F1"])
        fastest = min(all_results, key=lambda x: x["avg_inference_time_s"])
        best_pair_f1 = best_pair["pair_metrics"]["F1"]

        # Efficiency score: F1 * (1 / log(time+1)) — rewards both quality and speed
        import math
        for r in all_results:
            pf1 = r["pair_metrics"]["F1"]
            t = r["avg_inference_time_s"]
            r["efficiency_score"] = round(pf1 / math.log2(t + 2), 4) if pf1 > 0 else 0
        best_eff = max(all_results, key=lambda x: x.get("efficiency_score", 0))

        lines += [
            "",
            "## Summary",
            "",
            f"- **Best entity extraction**: {best_ent['model']} (F1={best_ent['entity_metrics']['F1']:.3f})",
            f"- **Best event-location linking**: {best_pair['model']} (F1={best_pair_f1:.3f})",
            f"- **Fastest inference**: {fastest['model']} ({fastest['avg_inference_time_s']:.2f}s/article)",
            f"- **Best efficiency** (F1/log2(time)): {best_eff['model']} (score={best_eff['efficiency_score']:.4f})",
            "",
            "## Inference Speed",
            "",
            "| Model | Total Time | Avg/Article | Articles/Min |",
            "|---|---|---|---|",
        ]
        for r in sorted(all_results, key=lambda x: x["avg_inference_time_s"]):
            t = r["avg_inference_time_s"]
            apm = 60 / t if t > 0 else float("inf")
            lines.append(f"| {r['model']} | {r['total_inference_time_s']:.1f}s | {t:.2f}s | {apm:.1f} |")

    report = "\n".join(lines)

    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Event-Location Extraction Benchmark")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to benchmark (default: all)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to annotated dataset JSON")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Max articles to evaluate")
    parser.add_argument("--match-mode", choices=["strict", "relaxed", "semantic"], default="semantic",
                        help="Matching mode: strict|relaxed|semantic (default: semantic)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    base = Path(__file__).parent

    # Dataset
    if args.dataset:
        ds_path = Path(args.dataset)
    else:
        ds_path = base / "data" / "baai_news_100_gold.json"
    if not ds_path.exists():
        print(f"ERROR: Dataset not found: {ds_path}")
        sys.exit(1)

    dataset = load_dataset(ds_path)
    print(f"Dataset: {ds_path.name} ({len(dataset)} articles)")

    # Match function
    score_fn, threshold = make_match_fn(args.match_mode)
    print(f"Match mode: {args.match_mode} (threshold={threshold})")

    # Output dir
    output_dir = Path(args.output) if args.output else base / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models — combine all registries
    all_models = get_mlx_models_to_benchmark() + get_cross_family_models() + get_2026_models() + get_wave3_models() + get_wave4_models()
    # Deduplicate by name (not repo_id — allows retries with different settings)
    seen_names = set()
    deduped = []
    for m in all_models:
        if m["name"] not in seen_names:
            seen_names.add(m["name"])
            deduped.append(m)
    all_models = deduped

    if args.models:
        requested = set(args.models.split(","))
        all_models = [m for m in all_models if m["name"] in requested or m["repo_id"] in requested]

    if not all_models:
        print("ERROR: No models selected")
        sys.exit(1)

    print(f"Models to benchmark: {[m['name'] for m in all_models]}")
    print(f"Max articles: {args.max_articles or len(dataset)}")
    print()

    # Run benchmarks
    all_results = []

    for model_info in all_models:
        print(f"{'=' * 60}")
        print(f"Benchmarking: {model_info['name']} ({model_info['params']}B params)")
        print(f"{'=' * 60}")

        adapter = MLXAdapter(
            repo_id=model_info["repo_id"],
            name=model_info["name"],
            max_tokens=model_info.get("max_tokens", 2048),
        )

        try:
            adapter.warmup()
        except Exception as e:
            print(f"  FAILED to load model: {e}")
            continue

        result = evaluate_model(
            adapter, dataset, score_fn, threshold,
            max_articles=args.max_articles,
        )
        all_results.append(result)

        # Print quick summary
        em = result["entity_metrics"]
        pm = result["pair_metrics"]
        print(f"  Entity F1: {em['F1']:.3f} (P={em['Precision']:.3f} R={em['Recall']:.3f})")
        print(f"  Pair   F1: {pm['F1']:.3f} (P={pm['Precision']:.3f} R={pm['Recall']:.3f})")
        print(f"  Avg time:  {result['avg_inference_time_s']:.2f}s/article")
        print()

        # Save per-model results
        with open(output_dir / f"{model_info['name']}_results.json", "w") as f:
            json.dump(result, f, indent=2)

        adapter.cleanup()

    # Save combined results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate report
    report = generate_markdown_report(all_results, output_dir, args.match_mode)
    print("\n" + report)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
