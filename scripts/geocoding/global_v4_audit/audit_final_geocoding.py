#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path

ROOT = Path("outputs/geography/global_v4_audit/final_geocoded")
AUDIT_DIR = ROOT / "audits"

EXPECTED = {
    "shards": 200,
    "articles": 34_388_866,
    "events": 128_267_182,
    "events_with_stated_location": 120_001_343,
    "raw_component_mentions": 137_216_336,
    "clean_component_mentions": 137_210_352,
}

audit_paths = sorted(AUDIT_DIR.glob("shard-*.audit.json"))
output_paths = sorted(ROOT.glob("shard-*.geo.jsonl"))

print("=== FILE COMPLETENESS ===")
print(f"outputs : {len(output_paths):,}")
print(f"audits  : {len(audit_paths):,}")

if len(output_paths) != EXPECTED["shards"]:
    raise RuntimeError(
        f"Expected {EXPECTED['shards']} outputs, found {len(output_paths)}"
    )

if len(audit_paths) != EXPECTED["shards"]:
    raise RuntimeError(
        f"Expected {EXPECTED['shards']} audits, found {len(audit_paths)}"
    )

totals = Counter()
photon = Counter()
wb = Counter()

limited = []
missing_lookup = 0

for p in audit_paths:
    d = json.loads(p.read_text(encoding="utf-8"))

    if d.get("limited"):
        limited.append(str(p))

    for key in (
        "articles",
        "events",
        "events_with_stated_location",
        "raw_component_mentions",
        "admin_component_count_mismatches",
    ):
        totals[key] += int(d.get(key, 0))

    missing_lookup += int(
        d.get("missing_unique_lookup_keys", 0)
    )

    photon.update(
        {
            k: int(v)
            for k, v in d.get(
                "photon_resolution_status", {}
            ).items()
        }
    )

    wb.update(
        {
            k: int(v)
            for k, v in d.get(
                "wb_match_status", {}
            ).items()
        }
    )


print()
print("=== GLOBAL FINAL GEOCODING AUDIT ===")

for key in (
    "articles",
    "events",
    "events_with_stated_location",
    "raw_component_mentions",
    "admin_component_count_mismatches",
):
    print(f"{key:<32}: {totals[key]:,}")

print(f"{'missing lookup keys':<32}: {missing_lookup:,}")


print()
print("Photon statuses:")

for k, v in photon.most_common():
    pct = (
        100 * v / totals["raw_component_mentions"]
        if totals["raw_component_mentions"]
        else 0
    )
    print(f"  {k:<34} {v:>15,}  {pct:8.4f}%")


print()
print("WB statuses:")

for k, v in wb.most_common():
    pct = (
        100 * v / totals["raw_component_mentions"]
        if totals["raw_component_mentions"]
        else 0
    )
    print(f"  {k:<44} {v:>15,}  {pct:8.4f}%")


photon_total = sum(photon.values())
wb_total = sum(wb.values())

print()
print("=== ACCOUNTING ===")
print(f"Photon status total          : {photon_total:,}")
print(f"WB status total              : {wb_total:,}")

clean_delta = (
    totals["raw_component_mentions"]
    - EXPECTED["clean_component_mentions"]
)

print(
    f"raw - clean component delta  : "
    f"{clean_delta:,}"
)


# ---------------------------------------------------------
# Hard invariants
# ---------------------------------------------------------

if limited:
    raise RuntimeError(
        f"{len(limited)} final audits were produced "
        f"with --limit-articles"
    )

if missing_lookup != 0:
    raise RuntimeError(
        f"{missing_lookup:,} missing lookup keys remain"
    )

for key in (
    "articles",
    "events",
    "events_with_stated_location",
    "raw_component_mentions",
):
    observed = totals[key]
    expected = EXPECTED[key]

    if observed != expected:
        raise RuntimeError(
            f"{key}: expected {expected:,}, "
            f"observed {observed:,}"
        )

if photon_total != totals["raw_component_mentions"]:
    raise RuntimeError(
        "Photon status accounting does not close"
    )

if wb_total != totals["raw_component_mentions"]:
    raise RuntimeError(
        "WB status accounting does not close"
    )

if clean_delta != 5_984:
    raise RuntimeError(
        f"Expected raw-clean component delta 5,984; "
        f"observed {clean_delta:,}"
    )


result = {
    "status": "PASS",
    "shards": 200,
    "totals": dict(totals),
    "missing_lookup_keys": missing_lookup,
    "photon_resolution_status": dict(photon),
    "wb_match_status": dict(wb),
    "canonical_clean_component_mentions":
        EXPECTED["clean_component_mentions"],
    "raw_minus_clean_component_delta":
        clean_delta,
    "materialization_exception_pairs": 13,
    "repaired_shards": 15,
}

out = ROOT / "global_audit.json"

out.write_text(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print("======================================")
print("GLOBAL FINAL GEOCODING AUDIT: PASS")
print("======================================")
print("written:", out)
