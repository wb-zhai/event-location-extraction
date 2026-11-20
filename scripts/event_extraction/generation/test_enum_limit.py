"""
Test how many enum values the Gemini API accepts in a response schema.

Runs two sweeps (pydantic Literal and raw dict) across increasing enum sizes,
printing pass/fail and the inferred limit.

Usage:
    python scripts/event_extraction/generation/test_enum_limit.py
    python scripts/event_extraction/generation/test_enum_limit.py --model gemini-3.1-pro-preview
    python scripts/event_extraction/generation/test_enum_limit.py --sizes 5 10 20 50 100 159
"""

import argparse
import json
import os
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import _transformers as genai_transformers
from google.genai import types as genai_types
from pydantic import BaseModel, create_model

REPO_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_PATH = (
    REPO_ROOT / "ontologies" / "zhai" / "risk.label.description.training.json"
)

PROBE_TEXT = (
    "A flood devastated Bangladesh in 2020, causing widespread food insecurity."
)


def load_env_file(path: Path) -> None:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ── label loading ──────────────────────────────────────────────────────────────


def all_labels() -> list[str]:
    with open(ONTOLOGY_PATH) as f:
        data = json.load(f)
    return list(data["events"].keys())


# ── schema builders ────────────────────────────────────────────────────────────


def pydantic_schema(labels: list[str]) -> dict:
    """Build schema via pydantic Literal (same path as generate.py)."""
    Model = create_model(
        "Event",
        event_type=(Literal[tuple(labels)], ...),
        location=(str, ...),
    )
    return genai_transformers.t_schema(None, Model).model_dump(exclude_none=True)


def raw_dict_schema(labels: list[str]) -> dict:
    """Build schema as a plain dict, bypassing pydantic entirely."""
    return {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": labels},
            "location": {"type": "string"},
        },
        "required": ["event_type", "location"],
    }


# ── probe ──────────────────────────────────────────────────────────────────────


def probe(
    client: genai.Client, model: str, schema: dict, label: str
) -> tuple[bool, str]:
    """Returns (ok, detail). detail is the response text or the error message."""
    try:
        resp = client.models.generate_content(
            model=model,
            contents=PROBE_TEXT,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return True, resp.text or ""
    except Exception as e:
        return False, str(e)[:120]


# ── sweep ──────────────────────────────────────────────────────────────────────


def sweep(
    client: genai.Client,
    model: str,
    labels: list[str],
    sizes: list[int],
    builder,
    name: str,
) -> None:
    print(f"\n{'─'*60}")
    print(f"  {name}  (model: {model})")
    print(f"{'─'*60}")
    last_ok = 0
    for n in sorted(sizes):
        subset = labels[:n]
        schema = builder(subset)
        ok, detail = probe(client, model, schema, name)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  n={n:4d}  {detail[:80]}")
        if ok:
            last_ok = n
        else:
            break  # stop at first failure — remaining will also fail
    print(f"  → last passing size: {last_ok}")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary-search Gemini enum size limit")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5, 10, 20, 30, 50, 75, 100, 125, 150, 159],
        help="Enum sizes to probe (in order)",
    )
    parser.add_argument(
        "--mode",
        choices=["pydantic", "raw", "both"],
        default="both",
        help="Which schema-building path to test",
    )
    args = parser.parse_args()

    load_env_file(".env")

    labels = all_labels()
    print(f"Loaded {len(labels)} labels from ontology.")
    print(f"Sizes to probe: {args.sizes}")

    client = genai.Client()

    if args.mode in ("pydantic", "both"):
        sweep(
            client, args.model, labels, args.sizes, pydantic_schema, "pydantic Literal"
        )

    if args.mode in ("raw", "both"):
        sweep(client, args.model, labels, args.sizes, raw_dict_schema, "raw dict")


if __name__ == "__main__":
    main()
