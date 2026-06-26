"""
Produce two CSVs from a .geo.jsonl predictions file (or a folder of them):
  - risk_matches.csv:  article_uri, risk_id
  - locations.csv:     article_uri, adm_code
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_risk_factors(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            mapping[row["name"]] = row["id"]
    return mapping


def load_valid_adm_codes(path: Path) -> set[str]:
    with open(path) as f:
        return {row["adm_code"] for row in csv.DictReader(f)}


def process(
    jsonl_path: Path,
    risk_factors: dict[str, str],
    valid_adm_codes: set[str],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    risk_matches: set[tuple[str, str]] = set()
    locations: set[tuple[str, str]] = set()

    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            obj = json.loads(line)
            article_uri: str = obj["id"]

            for pred in obj.get("predictions", []):
                event_type: str = pred["event_type"]
                if event_type not in risk_factors:
                    print(
                        f"Error: unknown event_type '{event_type}' "
                        f"(line {lineno}, article {article_uri})",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                risk_id = risk_factors[event_type]
                risk_matches.add((article_uri, risk_id))

                for geo in pred.get("geotaxonomy", []):
                    adm_code = geo.get("adm_code") or "NULL"
                    if adm_code != "NULL" and adm_code not in valid_adm_codes:
                        print(
                            f"Error: unknown adm_code '{adm_code}' "
                            f"(line {lineno}, article {article_uri})",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    locations.add((article_uri, adm_code))

    return risk_matches, locations


def write_csv(path: Path, header: list[str], rows: set[tuple[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sorted(rows))


def resolve_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.geo.jsonl"))
        if not files:
            print(f"Error: no *.geo.jsonl files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        return files
    return [input_path]


def main() -> None:
    res_dir = Path(__file__).parent / "res"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .geo.jsonl file or a folder containing *.geo.jsonl files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to input path's directory)",
    )
    parser.add_argument(
        "--risk-factors",
        type=Path,
        default=res_dir / "risk_factors.csv",
    )
    parser.add_argument(
        "--geo-taxonomy",
        type=Path,
        default=res_dir / "geo_taxonomy.csv",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir or (
        args.input if args.input.is_dir() else args.input.parent
    )

    risk_factors = load_risk_factors(args.risk_factors)
    valid_adm_codes = load_valid_adm_codes(args.geo_taxonomy)

    input_files = resolve_inputs(args.input)

    all_risk_matches: set[tuple[str, str]] = set()
    all_locations: set[tuple[str, str]] = set()

    for path in input_files:
        print(f"Processing {path} ...")
        risk_matches, locations = process(path, risk_factors, valid_adm_codes)
        all_risk_matches |= risk_matches
        all_locations |= locations

    write_csv(out_dir / "risk_matches.csv", ["article_uri", "risk_id"], all_risk_matches)
    write_csv(out_dir / "locations.csv", ["article_uri", "adm_code"], all_locations)

    print(f"risk_matches.csv: {len(all_risk_matches)} rows")
    print(f"locations.csv:    {len(all_locations)} rows")


if __name__ == "__main__":
    main()
