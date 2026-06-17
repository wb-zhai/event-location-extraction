from __future__ import annotations

import pytest

from src.train.eval_sft_event import evaluate_events


def _row(events: list[dict]) -> dict:
    return {"question": "conflict and flooding", "answer": {"events": events}}


def _event(event_type: str) -> dict:
    return {"event_type": event_type, "trigger": {"text": event_type}}


def test_evaluate_events_ignores_malformed_prediction_event_items() -> None:
    metrics = evaluate_events(
        [_row([_event("military conflict")])],
        [{"prediction": {"events": [_event("military conflict"), "military conflict"]}}],
    )

    assert metrics["window"]["precision"] == pytest.approx(1.0)
    assert metrics["window"]["recall"] == pytest.approx(1.0)
    assert metrics["window"]["f1"] == pytest.approx(1.0)
    assert metrics["document"]["precision"] == pytest.approx(1.0)
    assert metrics["document"]["recall"] == pytest.approx(1.0)
    assert metrics["document"]["f1"] == pytest.approx(1.0)
