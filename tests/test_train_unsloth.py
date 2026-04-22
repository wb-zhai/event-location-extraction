from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.data.dataset import DEFAULT_CANDIDATE_SAMPLING_SEED, load_candidate_ontology
from src.train import train_unsloth


class DummyTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}" for message in messages
        )


def build_sft_row() -> dict:
    return {
        "id": "doc-1",
        "question": "bombing in baghdad injured civilians",
        "event_labels": ["attack", "injure"],
        "argument_labels": ["place", "victim"],
        "answer": {
            "events": [
                {
                    "event_type": "attack",
                    "start": 0,
                    "end": 7,
                    "arguments": [
                        {"role": "place", "start": 11, "end": 18},
                    ],
                },
                {
                    "event_type": "injure",
                    "start": 19,
                    "end": 26,
                    "arguments": [
                        {"role": "victim", "start": 27, "end": 36},
                    ],
                },
            ]
        },
    }


def write_candidate_ontology(path: Path) -> Path:
    path.write_text(
        (
            '{"event_labels":["attack","injure","disaster","protest","arrest"],'
            '"argument_labels":["place","victim","agent","time","weapon"]}'
        ),
        encoding="utf-8",
    )
    return path


def test_format_row_includes_prompt_candidate_labels() -> None:
    text = train_unsloth._format_row(build_sft_row(), DummyTokenizer())["text"]

    assert 'Select event labels from the following set: ["attack", "injure"]' in text
    assert "Extract all events." in text
    assert "Select argument role labels" not in text


def test_candidate_fill_expands_requested_totals(tmp_path: Path) -> None:
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))

    event_labels, argument_labels = train_unsloth._resolve_candidate_labels(
        build_sft_row(),
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
        is_training=False,
        candidate_shuffle_probability=0.5,
        gold_candidate_dropout_probability=0.05,
        random_seed=DEFAULT_CANDIDATE_SAMPLING_SEED,
        candidate_rng=random.Random(7),
        index=0,
    )

    assert len(event_labels) == 4
    assert len(argument_labels) == 4
    assert {"attack", "injure"}.issubset(event_labels)
    assert {"place", "victim"}.issubset(argument_labels)
    assert len(set(event_labels)) == len(event_labels)
    assert len(set(argument_labels)) == len(argument_labels)


def test_train_sampling_applies_dropout_and_shuffle(tmp_path: Path) -> None:
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))
    row = build_sft_row()

    base_event_labels, base_argument_labels = train_unsloth._resolve_candidate_labels(
        row,
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
        is_training=False,
        candidate_shuffle_probability=0.0,
        gold_candidate_dropout_probability=0.0,
        random_seed=7,
        candidate_rng=random.Random(7),
        index=0,
    )
    train_event_labels, train_argument_labels = train_unsloth._resolve_candidate_labels(
        row,
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
        is_training=True,
        candidate_shuffle_probability=1.0,
        gold_candidate_dropout_probability=1.0,
        random_seed=7,
        candidate_rng=random.Random(7),
        index=0,
    )

    assert train_event_labels != base_event_labels
    assert train_argument_labels != base_argument_labels
    assert len(set(train_event_labels) & {"attack", "injure"}) >= 1
    assert len(set(train_event_labels) & {"attack", "injure"}) < 2
    assert len(set(train_argument_labels) & {"place", "victim"}) >= 1
    assert len(set(train_argument_labels) & {"place", "victim"}) < 2


def test_eval_candidate_lists_are_deterministic(tmp_path: Path) -> None:
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))
    row = build_sft_row()

    first = train_unsloth._resolve_candidate_labels(
        row,
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
        is_training=False,
        candidate_shuffle_probability=0.25,
        gold_candidate_dropout_probability=0.1,
        random_seed=99,
        candidate_rng=random.Random(99),
        index=0,
    )
    second = train_unsloth._resolve_candidate_labels(
        row,
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
        is_training=False,
        candidate_shuffle_probability=0.25,
        gold_candidate_dropout_probability=0.1,
        random_seed=99,
        candidate_rng=random.Random(99),
        index=0,
    )

    assert first == second


def test_main_requires_ontology_when_sampling_enabled(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="--ontology_file is required when candidate sampling is enabled"
    ):
        train_unsloth.main(
            [
                "--train_file",
                str(tmp_path / "train.jsonl"),
                "--model_name",
                "dummy-model",
                "--output_dir",
                str(tmp_path / "output"),
                "--num_event_candidates",
                "4",
            ]
        )


def test_format_row_rejects_missing_prompt_labels() -> None:
    row = build_sft_row()
    del row["event_labels"]

    with pytest.raises(ValueError, match="event_labels"):
        train_unsloth._format_row(row, DummyTokenizer())


def test_format_row_uses_gold_label_fallback_when_sampling_enabled(
    tmp_path: Path,
) -> None:
    row = build_sft_row()
    del row["event_labels"]
    del row["argument_labels"]
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))

    text = train_unsloth._format_row(
        row,
        DummyTokenizer(),
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
    )["text"]

    assert 'Select event labels from the following set: ["attack", "injure"' in text
    assert "Select argument role labels" not in text
