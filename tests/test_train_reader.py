import json
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, TrainingArguments

from src.data.dataset import (
    ARGUMENT_MARKER,
    EVENT_MARKER,
    EventReaderCollator,
    EventReaderDataset,
    encode_sample,
    load_normalized_jsonl,
    normalize_record,
)
from src.modeling.model import (
    EventReaderConfig,
    EventReader,
    apply_relation_budget,
    decode_single_label_relations,
)
from src.train.train import EventArgumentTrainer


def build_record(sample_id: str = "doc-1") -> dict:
    return {
        "id": sample_id,
        "tokens": ["bombing", "in", "baghdad", "injured", "civilians"],
        "event_labels": ["attack", "injure"],
        "argument_labels": ["place", "victim"],
        "events": [
            {"start": 0, "end": 0, "label": "attack"},
            {"start": 3, "end": 3, "label": "injure"},
        ],
        "arguments": [
            {"start": 2, "end": 2},
            {"start": 4, "end": 4},
        ],
        "relations": [
            {"event_idx": 0, "argument_idx": 0, "label": "place"},
            {"event_idx": 1, "argument_idx": 1, "label": "victim"},
        ],
        "metadata": {"source": "synthetic"},
    }


def build_second_record(sample_id: str = "doc-2") -> dict:
    return {
        "id": sample_id,
        "tokens": ["earthquake", "damaged", "rome", "and", "injured", "tourists"],
        "event_labels": ["disaster", "injure"],
        "argument_labels": ["place", "victim"],
        "events": [
            {"start": 0, "end": 0, "label": "disaster"},
            {"start": 4, "end": 4, "label": "injure"},
        ],
        "arguments": [
            {"start": 2, "end": 2},
            {"start": 5, "end": 5},
        ],
        "relations": [
            {"event_idx": 0, "argument_idx": 0, "label": "place"},
            {"event_idx": 1, "argument_idx": 1, "label": "victim"},
        ],
        "metadata": {"source": "synthetic"},
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def build_test_tokenizer(tmp_path: Path) -> BertTokenizerFast:
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        EVENT_MARKER,
        ARGUMENT_MARKER,
        "bombing",
        "in",
        "baghdad",
        "injured",
        "civilians",
        "attack",
        "injure",
        "place",
        "victim",
        "earthquake",
        "damaged",
        "rome",
        "and",
        "tourists",
        "disaster",
    ]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [EVENT_MARKER, ARGUMENT_MARKER]}
    )
    return tokenizer


def build_test_model(tokenizer: BertTokenizerFast) -> EventReader:
    encoder = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
        )
    )
    model = EventReader(
        EventReaderConfig(model_name="test", projection_dim=16),
        encoder=encoder,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def test_normalized_schema_validation_rejects_duplicate_pair() -> None:
    record = build_record()
    record["relations"].append({"event_idx": 0, "argument_idx": 0, "label": "victim"})
    with pytest.raises(ValueError, match="multiple labels"):
        normalize_record(record)


def test_preprocessing_preserves_word_spans_and_marker_positions(
    tmp_path: Path,
) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    sample = normalize_record(build_record())
    encoded = encode_sample(sample, tokenizer, max_length=64)

    assert len(encoded["event_marker_positions"]) == 2
    assert len(encoded["argument_marker_positions"]) == 2
    assert encoded["token_to_word"][encoded["gold_event_token_starts"][0]] == 0
    assert encoded["token_to_word"][encoded["gold_event_token_ends"][1]] == 3
    assert encoded["token_to_word"][encoded["gold_argument_token_starts"][1]] == 4


def test_marker_state_extraction() -> None:
    hidden_states = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        ]
    )
    positions = torch.tensor([[1, 3]])
    gathered = EventReader._gather_positions(hidden_states, positions)
    assert gathered.shape == (1, 2, 2)
    assert torch.equal(gathered[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.equal(gathered[0, 1], torch.tensor([3.0, 3.0]))


def test_event_label_dot_product_typing() -> None:
    model = build_test_model(build_test_tokenizer(Path("/tmp")))
    span_repr = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    label_repr = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    positions = torch.tensor([[0, 1]])
    logits = model._compute_event_type_logits(span_repr, label_repr, positions)
    assert logits.argmax(dim=-1).tolist() == [[0, 1]]


def test_relation_tensor_construction_shape(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    model = build_test_model(tokenizer)
    hidden = model.encoder.config.hidden_size
    logits = model.compute_relation_logits(
        torch.randn(1, 2, hidden * 2),
        torch.randn(1, 3, hidden * 2),
        torch.randn(1, 4, hidden),
    )
    assert logits.shape == (1, 2, 3, 4, 2)


def test_thresholded_single_label_relation_decode() -> None:
    scores = torch.tensor(
        [
            [
                [0.20, 0.85, 0.70],
                [0.10, 0.30, 0.25],
            ]
        ]
    )
    decoded = decode_single_label_relations(scores, threshold=0.5)
    assert len(decoded) == 1
    assert decoded[0]["event_idx"] == 0
    assert decoded[0]["argument_idx"] == 0
    assert decoded[0]["label_idx"] == 1
    assert decoded[0]["score"] == pytest.approx(0.85)


def test_relation_budget_keeps_highest_confidence_spans() -> None:
    events = [{"score": 0.9}, {"score": 0.8}, {"score": 0.1}]
    arguments = [{"score": 0.95}, {"score": 0.7}, {"score": 0.2}]
    kept_events, kept_arguments = apply_relation_budget(
        events, arguments, pair_budget=4
    )
    assert len(kept_events) * len(kept_arguments) <= 4
    assert [item["score"] for item in kept_events] == [0.9, 0.8]
    assert [item["score"] for item in kept_arguments] == [0.95, 0.7]


def test_trainer_smoke_runs_on_tiny_synthetic_dataset(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(train_path, [build_record(), build_second_record()])
    write_jsonl(eval_path, [build_second_record(), build_record("doc-3")])

    tokenizer = build_test_tokenizer(tmp_path)
    model = build_test_model(tokenizer)
    train_dataset = EventReaderDataset(load_normalized_jsonl(train_path), tokenizer, 64)
    eval_dataset = EventReaderDataset(load_normalized_jsonl(eval_path), tokenizer, 64)
    trainer = EventArgumentTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "output"),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            max_steps=1,
            num_train_epochs=1,
            save_strategy="no",
            eval_strategy="no",
            remove_unused_columns=False,
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=EventReaderCollator(tokenizer.pad_token_id),
    )

    trainer.train()
    metrics = trainer.evaluate()

    assert "eval_event_span_f1" in metrics
    assert "eval_event_type_f1" in metrics
    assert "eval_argument_span_f1" in metrics
    assert "eval_relation_identification_f1" in metrics
    assert "eval_relation_classification_f1" in metrics
