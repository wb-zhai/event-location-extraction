from __future__ import annotations

import json
from pathlib import Path

from src.data.balance_event_dataset import (
    choose_target_count,
    choose_self_balanced_target_count,
    event_type_counts,
    main,
    select_balancing_subset,
    select_self_balanced_subset,
)


def _record(record_id: str, event_types: list[str]) -> dict:
    return {
        "id": record_id,
        "status": "ok",
        "source": {"title": record_id, "text": ""},
        "events": [{"event_type": event_type} for event_type in event_types],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_choose_target_count_uses_reference_max_by_default() -> None:
    counts = event_type_counts(
        [
            _record("a", ["attack", "flood"]),
            _record("b", ["attack"]),
            _record("c", ["attack", "injure"]),
        ]
    )

    assert choose_target_count(counts, strategy="max", explicit_target=None) == 3


def test_choose_self_balanced_target_count_uses_pool_min_by_default() -> None:
    counts = event_type_counts(
        [
            _record("a", ["attack", "flood"]),
            _record("b", ["attack"]),
            _record("c", ["attack", "injure"]),
        ]
    )

    assert choose_self_balanced_target_count(
        counts, strategy="min", explicit_target=None
    ) == 1


def test_select_balancing_subset_fills_underrepresented_labels() -> None:
    reference = [
        _record("r1", ["attack"]),
        _record("r2", ["attack"]),
        _record("r3", ["attack", "flood"]),
        _record("r4", ["injure"]),
    ]
    pool = [
        _record("p1", ["injure"]),
        _record("p2", ["injure", "attack"]),
        _record("p3", ["flood"]),
        _record("p4", ["attack"]),
    ]

    selected, report = select_balancing_subset(reference, pool, target_count=3, seed=7)

    assert {record["id"] for record in selected} == {"p1", "p2", "p3"}
    assert report["combined_event_type_counts"] == {
        "attack": 4,
        "injure": 3,
        "flood": 2,
    }
    assert report["remaining_deficits"] == {"flood": 1}


def test_main_writes_subset_and_report(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.jsonl"
    pool_path = tmp_path / "pool.jsonl"
    output_path = tmp_path / "selected.jsonl"
    report_path = tmp_path / "selected.report.json"

    _write_jsonl(reference_path, [_record("r1", ["attack"]), _record("r2", ["injure"])])
    _write_jsonl(pool_path, [_record("p1", ["attack"]), _record("p2", ["injure"])])

    exit_code = main(
        [
            "--reference",
            str(reference_path),
            "--pool",
            str(pool_path),
            "--output",
            str(output_path),
            "--report-output",
            str(report_path),
            "--target-count",
            "2",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert report_path.exists()
    assert len(output_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_select_self_balanced_subset_balances_from_pool_only() -> None:
    pool = [
        _record("p1", ["attack"]),
        _record("p2", ["attack"]),
        _record("p3", ["injure"]),
        _record("p4", ["flood"]),
    ]

    selected, report = select_self_balanced_subset(pool, target_count=1, seed=7)

    assert {record["id"] for record in selected} == {"p1", "p3", "p4"}
    assert report["selected_event_type_counts"] == {
        "attack": 1,
        "injure": 1,
        "flood": 1,
    }
    assert report["remaining_deficits"] == {}
    assert report["overshoot_by_label"] == {}


def test_select_self_balanced_subset_enforces_hard_caps() -> None:
    pool = [
        _record("p1", ["attack", "injure"]),
        _record("p2", ["attack"]),
        _record("p3", ["injure"]),
        _record("p4", ["flood"]),
    ]

    selected, report = select_self_balanced_subset(pool, target_count=1, seed=7)

    assert {record["id"] for record in selected} == {"p1", "p4"}
    assert report["selected_event_type_counts"] == {
        "attack": 1,
        "injure": 1,
        "flood": 1,
    }
    assert report["overshoot_by_label"] == {}


def test_main_self_balances_without_reference(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.jsonl"
    output_path = tmp_path / "selected.jsonl"
    report_path = tmp_path / "selected.report.json"

    _write_jsonl(
        pool_path,
        [
            _record("p1", ["attack"]),
            _record("p2", ["attack"]),
            _record("p3", ["injure"]),
            _record("p4", ["flood"]),
        ],
    )

    exit_code = main(
        [
            "--pool",
            str(pool_path),
            "--output",
            str(output_path),
            "--report-output",
            str(report_path),
            "--target-strategy",
            "min",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "self_balanced"
    assert report["selected_event_type_counts"] == {
        "attack": 1,
        "injure": 1,
        "flood": 1,
    }
