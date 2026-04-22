from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.inference import sft_infer
from src.sft_prompt import SYSTEM_PROMPT, render_chat


class DummyTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        prompt = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}" for message in messages
        )
        if add_generation_prompt:
            return prompt + "\n\nASSISTANT:\n"
        return prompt


def write_candidate_ontology(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "event_labels": ["attack", "injure", "transport"],
                "argument_labels": ["place", "victim"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prompt_matches_training_contract() -> None:
    text = render_chat(
        DummyTokenizer(),
        "bombing in baghdad injured civilians",
        ["attack", "injure"],
        add_generation_prompt=True,
    )

    assert SYSTEM_PROMPT in text
    assert "Extract all events." in text
    assert 'Select event labels from the following set: ["attack", "injure"]' in text
    assert "Select argument role labels" not in text
    assert text.endswith("ASSISTANT:\n")


def test_load_event_labels_from_ontology(tmp_path: Path) -> None:
    args = argparse.Namespace(
        event_labels=None,
        ontology_file=str(write_candidate_ontology(tmp_path / "ontology.json")),
    )

    assert sft_infer._load_event_labels(args) == ["attack", "injure", "transport"]


def test_load_event_labels_prefers_inline_override(tmp_path: Path) -> None:
    args = argparse.Namespace(
        event_labels=["custom_a", "custom_b"],
        ontology_file=str(write_candidate_ontology(tmp_path / "ontology.json")),
    )

    assert sft_infer._load_event_labels(args) == ["custom_a", "custom_b"]


def test_load_event_labels_requires_source() -> None:
    args = argparse.Namespace(event_labels=None, ontology_file=None)

    with pytest.raises(
        ValueError, match="Either --ontology_file or --event_labels must be provided"
    ):
        sft_infer._load_event_labels(args)


def test_parse_args_accepts_interactive_without_text() -> None:
    args = sft_infer.parse_args(
        [
            "--model_path",
            "dummy-model",
            "--event_labels",
            "attack",
            "--interactive",
        ]
    )

    assert args.interactive is True
    assert args.text is None
    assert args.text_file is None


def test_interactive_should_stop() -> None:
    assert sft_infer._interactive_should_stop("exit") is True
    assert sft_infer._interactive_should_stop(" Quit ") is True
    assert sft_infer._interactive_should_stop("bombing in baghdad") is False


def test_extract_first_json_object_skips_wrapper_text() -> None:
    parsed = sft_infer._extract_first_json_object(
        'Here you go:\n{"events":[{"event_type":"attack","text":"bombing"}]}\nThanks.'
    )

    assert parsed == {"events": [{"event_type": "attack", "text": "bombing"}]}


def test_parse_prediction_text_returns_empty_events_on_malformed_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parsed = sft_infer._parse_prediction_text("not valid json")

    assert parsed == {"events": []}
    assert "failed to parse model output as JSON" in capsys.readouterr().err


def test_normalize_prediction_keeps_valid_offsets() -> None:
    document = "bombing in baghdad injured civilians"
    prediction = {
        "events": [
            {"event_type": "attack", "start": 0, "end": 7, "text": "bombing"},
        ]
    }

    assert sft_infer._normalize_prediction(document, prediction) == {
        "events": [
            {"event_type": "attack", "start": 0, "end": 7, "text": "bombing"}
        ]
    }


def test_normalize_prediction_repairs_offsets_with_text_anchor() -> None:
    document = "bombing in baghdad injured civilians"
    prediction = {
        "events": [
            {"event_type": "attack", "start": 99, "end": 101, "text": "baghdad"},
        ]
    }

    assert sft_infer._normalize_prediction(document, prediction) == {
        "events": [
            {"event_type": "attack", "start": 11, "end": 18, "text": "baghdad"}
        ]
    }


def test_normalize_prediction_drops_unmatched_events() -> None:
    document = "bombing in baghdad injured civilians"
    prediction = {
        "events": [
            {"event_type": "attack", "text": "tehran"},
        ]
    }

    assert sft_infer._normalize_prediction(document, prediction) == {"events": []}


def test_normalize_prediction_dedupes_grounded_events() -> None:
    document = "bombing in baghdad injured civilians"
    prediction = {
        "events": [
            {"event_type": "attack", "start": 0, "end": 7, "text": "bombing"},
            {"event_type": "attack", "text": "bombing"},
        ]
    }

    assert sft_infer._normalize_prediction(document, prediction) == {
        "events": [
            {"event_type": "attack", "start": 0, "end": 7, "text": "bombing"}
        ]
    }


def test_generate_prediction_text_passes_prompt_as_text_keyword() -> None:
    class DummyTensor:
        def __init__(self, values):
            self.values = values
            self.shape = (1, len(values))

        def to(self, _device):
            return self

        def __getitem__(self, item):
            if isinstance(item, slice):
                return self.values[item]
            return self.values[item]

    class DummyInputs(dict):
        pass

    class ProcessorLikeTokenizer:
        def __init__(self) -> None:
            self.called_with_text = None

        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            prompt = "\n\n".join(
                f"{message['role'].upper()}:\n{message['content']}" for message in messages
            )
            if add_generation_prompt:
                return prompt + "\n\nASSISTANT:\n"
            return prompt

        def __call__(self, *args, **kwargs):
            assert args == ()
            self.called_with_text = kwargs["text"]
            return DummyInputs({"input_ids": DummyTensor([1, 2, 3])})

        def decode(self, tokens, *, skip_special_tokens: bool):
            assert skip_special_tokens is True
            assert tokens == [4, 5]
            return '{"events":[]}'

    class DummyOutputRow:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, item):
            if isinstance(item, slice):
                return self.values[item]
            return self.values[item]

    class DummyOutputs:
        def __getitem__(self, item):
            assert item == 0
            return DummyOutputRow([1, 2, 3, 4, 5])

    class DummyModel:
        device = "cpu"

        def generate(self, **kwargs):
            assert "input_ids" in kwargs
            return DummyOutputs()

    tokenizer = ProcessorLikeTokenizer()
    text = sft_infer._generate_prediction_text(
        DummyModel(),
        tokenizer,
        document="bombing in baghdad injured civilians",
        event_labels=["attack", "injure"],
        max_new_tokens=32,
        temperature=0.0,
    )

    assert "Extract all events." in tokenizer.called_with_text
    assert text == '{"events":[]}'
