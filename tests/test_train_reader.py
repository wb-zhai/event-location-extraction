import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
    TrainingArguments,
)

from src.train import train as train_module
from src.data.dataset import (
    ARGUMENT_MARKER,
    EVENT_MARKER,
    EventReaderCollator,
    EventReaderDataset,
    RelationAnnotation,
    encode_sample,
    load_candidate_ontology,
    load_normalized_jsonl,
    normalize_record,
)
from src.modeling.model import (
    DEFAULT_RELATION_THRESHOLD,
    EventReaderConfig,
    EventReader,
    apply_relation_budget,
    decode_multi_label_relations,
)
from src.train.train import EventArgumentTrainer, resolve_precision_flags
from src.train.train import (
    _prediction_with_relation_threshold,
    tune_relation_threshold,
)


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


def build_zero_role_record(sample_id: str = "doc-zero") -> dict:
    return {
        "id": sample_id,
        "tokens": ["bombing", "injured", "civilians"],
        "event_labels": ["attack"],
        "argument_labels": [],
        "events": [{"start": 0, "end": 0, "label": "attack"}],
        "arguments": [{"start": 2, "end": 2}],
        "relations": [],
        "metadata": {"source": "synthetic"},
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_candidate_ontology(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "event_labels": [
                    "attack",
                    "injure",
                    "disaster",
                    "protest",
                    "arrest",
                ],
                "argument_labels": [
                    "place",
                    "victim",
                    "agent",
                    "time",
                    "weapon",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


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


def build_subword_test_tokenizer(tmp_path: Path) -> BertTokenizerFast:
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        EVENT_MARKER,
        ARGUMENT_MARKER,
        "injured",
        "attack",
    ]
    vocab_path = tmp_path / "subword_vocab.txt"
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


def build_test_encoder(vocab_size: int) -> BertModel:
    return BertModel(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
        )
    )


def build_test_model(
    tokenizer: BertTokenizerFast,
    *,
    model_name: str = "test",
    encoder_vocab_size: int | None = None,
) -> EventReader:
    encoder = build_test_encoder(encoder_vocab_size or len(tokenizer))
    model = EventReader(
        EventReaderConfig(model_name=model_name, projection_dim=16),
        encoder=encoder,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def test_normalized_schema_allows_multiple_labels_for_same_pair() -> None:
    record = build_record()
    record["relations"].append({"event_idx": 0, "argument_idx": 0, "label": "victim"})

    sample = normalize_record(record)

    assert sample.relations == [
        RelationAnnotation(event_idx=0, argument_idx=0, label="place"),
        RelationAnnotation(event_idx=1, argument_idx=1, label="victim"),
        RelationAnnotation(event_idx=0, argument_idx=0, label="victim"),
    ]


def test_normalized_schema_merges_duplicate_argument_spans() -> None:
    record = build_record()
    record["arguments"] = [
        {"start": 2, "end": 2},
        {"start": 2, "end": 2},
        {"start": 4, "end": 4},
    ]
    record["relations"] = [
        {"event_idx": 0, "argument_idx": 0, "label": "place"},
        {"event_idx": 0, "argument_idx": 1, "label": "victim"},
        {"event_idx": 1, "argument_idx": 1, "label": "place"},
    ]

    sample = normalize_record(record)

    assert [(span.start, span.end) for span in sample.arguments] == [(2, 2), (4, 4)]
    assert sample.relations == [
        RelationAnnotation(event_idx=0, argument_idx=0, label="place"),
        RelationAnnotation(event_idx=0, argument_idx=0, label="victim"),
        RelationAnnotation(event_idx=1, argument_idx=0, label="place"),
    ]


def test_normalized_schema_rejects_exact_duplicate_relation_after_argument_merge() -> None:
    record = build_record()
    record["arguments"] = [
        {"start": 2, "end": 2},
        {"start": 2, "end": 2},
        {"start": 4, "end": 4},
    ]
    record["relations"] = [
        {"event_idx": 0, "argument_idx": 0, "label": "place"},
        {"event_idx": 0, "argument_idx": 1, "label": "place"},
    ]

    with pytest.raises(ValueError, match="duplicate relation"):
        normalize_record(record)


def test_normalized_schema_allows_duplicate_tokens() -> None:
    record = build_record()
    record["tokens"] = ["the", "attack", "in", "the", "city"]

    sample = normalize_record(record)

    assert sample.tokens == record["tokens"]


def test_normalized_schema_accepts_empty_argument_labels() -> None:
    sample = normalize_record(build_zero_role_record())
    assert sample.argument_labels == []
    assert sample.relations == []


def test_normalized_schema_rejects_relations_without_argument_labels() -> None:
    record = build_zero_role_record()
    record["relations"] = [{"event_idx": 0, "argument_idx": 0, "label": "victim"}]
    with pytest.raises(
        ValueError, match="must be empty when 'argument_labels' is empty"
    ):
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


def test_preprocessing_supports_zero_role_samples(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    sample = normalize_record(build_zero_role_record())
    encoded = encode_sample(sample, tokenizer, max_length=64)

    assert encoded["argument_marker_positions"] == []
    assert encoded["relation_targets"] == []
    assert len(encoded["gold_argument_token_starts"]) == 1


def test_candidate_augmentation_fills_requested_totals(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))
    sample = normalize_record(build_record())

    dataset = EventReaderDataset(
        [sample],
        tokenizer,
        64,
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
        random_seed=7,
    )

    candidate_sample = dataset.samples[0]
    assert len(candidate_sample.event_labels) == 4
    assert len(candidate_sample.argument_labels) == 4
    assert set(sample.event_labels).issubset(candidate_sample.event_labels)
    assert set(sample.argument_labels).issubset(candidate_sample.argument_labels)
    assert len(set(candidate_sample.event_labels)) == len(candidate_sample.event_labels)
    assert len(set(candidate_sample.argument_labels)) == len(
        candidate_sample.argument_labels
    )


def test_candidate_augmentation_rejects_requested_total_below_gold_count(
    tmp_path: Path,
) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))

    with pytest.raises(ValueError, match="requested 1 event candidates"):
        EventReaderDataset(
            [normalize_record(build_record())],
            tokenizer,
            64,
            ontology=ontology,
            num_event_candidates=1,
        )


def test_training_candidate_shuffle_reorders_labels(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))
    dataset = EventReaderDataset(
        [normalize_record(build_record())],
        tokenizer,
        64,
        ontology=ontology,
        num_event_candidates=5,
        num_relation_candidates=5,
        is_training=True,
        candidate_shuffle_probability=1.0,
        gold_candidate_dropout_probability=0.0,
        random_seed=7,
    )

    encoded = dataset[0]

    assert (
        encoded["event_label_texts"] != dataset.samples[0].event_labels
        or encoded["argument_label_texts"] != dataset.samples[0].argument_labels
    )


def test_training_gold_dropout_refills_and_ignores_dropped_event_types(
    tmp_path: Path,
) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))
    sample = normalize_record(build_record())
    dataset = EventReaderDataset(
        [sample],
        tokenizer,
        64,
        ontology=ontology,
        num_event_candidates=3,
        num_relation_candidates=3,
        is_training=True,
        candidate_shuffle_probability=0.0,
        gold_candidate_dropout_probability=1.0,
        random_seed=7,
    )

    encoded = dataset[0]
    remaining_event_golds = set(encoded["event_label_texts"]) & set(sample.event_labels)
    remaining_relation_golds = set(encoded["argument_label_texts"]) & set(
        sample.argument_labels
    )

    assert len(encoded["event_label_texts"]) == 3
    assert len(encoded["argument_label_texts"]) == 3
    assert 1 <= len(remaining_event_golds) < len(sample.event_labels)
    assert 1 <= len(remaining_relation_golds) < len(sample.argument_labels)
    assert -100 in encoded["gold_event_type_labels"]
    assert len(encoded["relation_targets"]) < len(sample.relations)


def test_eval_candidate_lists_are_deterministic(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))
    dataset = EventReaderDataset(
        [normalize_record(build_record())],
        tokenizer,
        64,
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=4,
        is_training=False,
        random_seed=7,
    )

    first = dataset[0]
    second = dataset[0]

    assert first["event_label_texts"] == dataset.samples[0].event_labels
    assert first["argument_label_texts"] == dataset.samples[0].argument_labels
    assert second["event_label_texts"] == first["event_label_texts"]
    assert second["argument_label_texts"] == first["argument_label_texts"]


def test_candidate_banks_still_respect_max_length_truncation(tmp_path: Path) -> None:
    tokenizer = build_subword_test_tokenizer(tmp_path)
    ontology = load_candidate_ontology(write_candidate_ontology(tmp_path / "ontology.json"))
    sample = normalize_record(
        {
            "id": "doc-truncate-candidates",
            "tokens": ["can't", "injured"],
            "event_labels": ["attack"],
            "argument_labels": [],
            "events": [{"start": 0, "end": 0, "label": "attack"}],
            "arguments": [{"start": 1, "end": 1}],
            "relations": [],
            "metadata": {"source": "synthetic"},
        }
    )

    dataset = EventReaderDataset(
        [sample],
        tokenizer,
        21,
        ontology=ontology,
        num_event_candidates=4,
        num_relation_candidates=3,
        random_seed=7,
    )
    encoded = dataset[0]

    assert encoded["reference"]["tokens"] == ["can't"]
    assert len(encoded["event_label_texts"]) == 4
    assert len(encoded["argument_label_texts"]) == 3


def test_preprocessing_aligns_multi_subword_words(tmp_path: Path) -> None:
    tokenizer = build_subword_test_tokenizer(tmp_path)
    sample = normalize_record(
        {
            "id": "doc-subword",
            "tokens": ["can't", "injured"],
            "event_labels": ["attack"],
            "argument_labels": [],
            "events": [{"start": 0, "end": 0, "label": "attack"}],
            "arguments": [{"start": 0, "end": 0}],
            "relations": [],
            "metadata": {"source": "synthetic"},
        }
    )

    encoded = encode_sample(sample, tokenizer, max_length=64)

    assert encoded["gold_event_token_starts"] == [1]
    assert encoded["gold_event_token_ends"] == [3]
    assert encoded["gold_argument_token_starts"] == [1]
    assert encoded["gold_argument_token_ends"] == [3]
    assert encoded["token_to_word"][1:5] == [0, 0, 0, 1]
    assert encoded["word_start_mask"][1:5] == [1, 0, 0, 1]
    assert encoded["word_end_mask"][1:5] == [0, 0, 1, 1]


def test_preprocessing_truncates_by_whole_words(tmp_path: Path) -> None:
    tokenizer = build_subword_test_tokenizer(tmp_path)
    sample = normalize_record(
        {
            "id": "doc-truncate",
            "tokens": ["can't", "injured"],
            "event_labels": ["attack"],
            "argument_labels": [],
            "events": [{"start": 0, "end": 0, "label": "attack"}],
            "arguments": [{"start": 1, "end": 1}],
            "relations": [],
            "metadata": {"source": "synthetic"},
        }
    )

    encoded = encode_sample(sample, tokenizer, max_length=9)

    assert encoded["reference"]["tokens"] == ["can't"]
    assert encoded["reference"]["metadata"]["truncated"] is True
    assert encoded["gold_event_token_starts"] == [1]
    assert encoded["gold_event_token_ends"] == [3]
    assert encoded["gold_argument_token_starts"] == []
    assert encoded["gold_argument_token_ends"] == []


def test_preprocessing_requires_fast_tokenizer_alignment(tmp_path: Path) -> None:
    fast_tokenizer = build_test_tokenizer(tmp_path)

    class SlowTokenizerProxy:
        is_fast = False

        def __init__(self, base_tokenizer: BertTokenizerFast):
            self.base_tokenizer = base_tokenizer

        def __call__(self, *args, **kwargs):
            return self.base_tokenizer(*args, **kwargs)

        def convert_tokens_to_ids(self, *args, **kwargs):
            return self.base_tokenizer.convert_tokens_to_ids(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.base_tokenizer, name)

    tokenizer = SlowTokenizerProxy(fast_tokenizer)
    sample = normalize_record(build_zero_role_record())

    with pytest.raises(ValueError, match="fast tokenizer with word_ids\\(\\) support"):
        encode_sample(sample, tokenizer, max_length=64)


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


def test_mask_invalid_columns_is_fp16_safe() -> None:
    logits = torch.tensor(
        [[[0.2, 0.5, -0.3], [0.1, -0.4, 0.7]]],
        dtype=torch.float16,
    )
    positions = torch.tensor([[0, -1, 2]], dtype=torch.long)

    masked = EventReader._mask_invalid_columns(logits, positions)

    expected_fill = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16)
    assert masked.dtype == torch.float16
    assert masked[0, 0, 1] == expected_fill
    assert masked[0, 1, 1] == expected_fill


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


def test_forward_aligns_half_hidden_states_with_float_heads(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    model = build_test_model(tokenizer)
    hidden_size = model.encoder.config.hidden_size

    def fake_encode(
        input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.randn(
            input_ids.shape[0],
            input_ids.shape[1],
            hidden_size,
            dtype=torch.float16,
        )

    model._encode = fake_encode  # type: ignore[assignment]

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        event_marker_positions=torch.tensor([[0]], dtype=torch.long),
        argument_marker_positions=torch.tensor([[0]], dtype=torch.long),
        word_end_mask=torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
        gold_event_token_starts=torch.tensor([[1]], dtype=torch.long),
        gold_event_token_ends=torch.tensor([[2]], dtype=torch.long),
        gold_argument_token_starts=torch.tensor([[1]], dtype=torch.long),
        gold_argument_token_ends=torch.tensor([[2]], dtype=torch.long),
        event_end_labels=torch.tensor([[-100, 0, 1, 0]], dtype=torch.long),
        argument_end_labels=torch.tensor([[-100, 0, 1, 0]], dtype=torch.long),
    )

    expected_dtype = model.event_start_head[1].weight.dtype
    assert outputs["event_start_logits"].dtype == expected_dtype
    assert outputs["event_end_logits"].dtype == expected_dtype
    assert outputs["argument_start_logits"].dtype == expected_dtype
    assert outputs["argument_end_logits"].dtype == expected_dtype


def test_thresholded_multi_label_relation_decode() -> None:
    scores = torch.tensor(
        [
            [
                [0.20, 0.85, 0.70],
                [0.10, 0.30, 0.25],
            ]
        ]
    )
    decoded = decode_multi_label_relations(scores, threshold=0.5)
    assert decoded == [
        {
            "event_idx": 0,
            "argument_idx": 0,
            "label_idx": 1,
            "score": pytest.approx(0.85),
        },
        {
            "event_idx": 0,
            "argument_idx": 0,
            "label_idx": 2,
            "score": pytest.approx(0.70),
        },
    ]


def test_relation_budget_keeps_highest_confidence_spans() -> None:
    events = [{"score": 0.9}, {"score": 0.8}, {"score": 0.1}]
    arguments = [{"score": 0.95}, {"score": 0.7}, {"score": 0.2}]
    kept_events, kept_arguments = apply_relation_budget(
        events, arguments, pair_budget=4
    )
    assert len(kept_events) * len(kept_arguments) <= 4
    assert [item["score"] for item in kept_events] == [0.9, 0.8]
    assert [item["score"] for item in kept_arguments] == [0.95, 0.7]


def test_conditioned_end_decoding_recovers_multi_token_spans() -> None:
    model = build_test_model(build_test_tokenizer(Path("/tmp")))
    hidden_states = torch.randn(5, model.encoder.config.hidden_size)
    start_logits = torch.tensor(
        [
            [4.0, -4.0],
            [-4.0, 4.0],
            [-4.0, 4.0],
            [-4.0, 4.0],
            [4.0, -4.0],
        ]
    )
    word_start_mask = torch.tensor([0, 1, 1, 1, 0], dtype=torch.long)
    word_end_mask = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long)
    token_to_word = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.long)

    def fake_conditioned_end_logits(
        hidden_states: torch.Tensor,
        start_positions: torch.Tensor,
        end_head: torch.nn.Module,
    ) -> torch.Tensor:
        logits = torch.full((1, 3, 5, 2), -5.0)
        logits[..., 0] = 5.0
        logits[0, 0, 3, 1] = 9.0
        logits[0, 1, 4, 1] = 9.0
        logits[0, 2, 4, 1] = 8.0
        return logits

    model._compute_conditioned_end_logits = fake_conditioned_end_logits  # type: ignore[assignment]

    spans = model._decode_spans(
        hidden_states,
        torch.softmax(start_logits, dim=-1)[..., 1],
        start_logits,
        model.event_end_head,
        word_start_mask,
        word_end_mask,
        token_to_word,
    )

    assert [(span["start"], span["end"]) for span in spans] == [(0, 2), (1, 3), (2, 3)]


def test_relation_threshold_projection_preserves_event_labels() -> None:
    prediction = {
        "events": [{"start": 0, "end": 0, "label": "attack", "score": 0.9}],
        "arguments": [{"start": 1, "end": 1, "score": 0.8}],
        "relations": [],
        "relation_candidates": [
            {"event_idx": 0, "argument_idx": 0, "label": "place", "score": 0.6}
        ],
    }

    thresholded = _prediction_with_relation_threshold(prediction, 0.5)

    assert thresholded["events"] == prediction["events"]
    assert thresholded["events"][0]["label"] == "attack"
    assert thresholded["relations"] == prediction["relation_candidates"]


def test_relation_threshold_search_picks_better_value_than_default() -> None:
    predictions = [
        {
            "events": [{"start": 0, "end": 0, "label": "attack", "score": 0.9}],
            "arguments": [{"start": 1, "end": 1, "score": 0.8}],
            "relations": [],
            "relation_candidates": [
                {"event_idx": 0, "argument_idx": 0, "label": "place", "score": 0.45},
                {"event_idx": 0, "argument_idx": 0, "label": "victim", "score": 0.40},
            ],
        }
    ]
    references = [
        {
            "events": [{"start": 0, "end": 0, "label": "attack"}],
            "arguments": [{"start": 1, "end": 1}],
            "relations": [{"event_idx": 0, "argument_idx": 0, "label": "place"}],
        }
    ]

    best_threshold, metrics = tune_relation_threshold(predictions, references)

    assert best_threshold == pytest.approx(0.45)
    assert best_threshold != DEFAULT_RELATION_THRESHOLD
    assert metrics["relation_classification_f1"] == pytest.approx(1.0)


def test_relation_budgeting_does_not_trim_returned_spans(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    model = build_test_model(tokenizer)
    hidden_states = torch.randn(5, model.encoder.config.hidden_size)
    event_predictions = [
        {"token_start": 1, "token_end": 1, "start": 0, "end": 0, "score": 0.9},
        {"token_start": 2, "token_end": 2, "start": 1, "end": 1, "score": 0.8},
        {"token_start": 3, "token_end": 3, "start": 2, "end": 2, "score": 0.7},
    ]
    argument_predictions = [
        {"token_start": 1, "token_end": 1, "start": 0, "end": 0, "score": 0.9},
        {"token_start": 2, "token_end": 2, "start": 1, "end": 1, "score": 0.8},
        {"token_start": 3, "token_end": 3, "start": 2, "end": 2, "score": 0.7},
    ]

    calls = {"count": 0}

    def fake_decode_spans(*args, **kwargs):
        calls["count"] += 1
        return event_predictions if calls["count"] == 1 else argument_predictions

    model._decode_spans = fake_decode_spans  # type: ignore[assignment]
    model.compute_relation_logits = lambda *args, **kwargs: torch.zeros(1, 2, 2, 1, 2)  # type: ignore[assignment]

    decoded = model._decode_document(
        hidden_states=hidden_states,
        event_start_logits=torch.zeros(5, 2),
        argument_start_logits=torch.zeros(5, 2),
        word_start_mask=torch.tensor([0, 1, 1, 1, 0], dtype=torch.long),
        word_end_mask=torch.tensor([0, 1, 1, 1, 0], dtype=torch.long),
        token_to_word=torch.tensor([-1, 0, 1, 2, -1], dtype=torch.long),
        event_marker_positions=torch.tensor([4], dtype=torch.long),
        argument_marker_positions=torch.tensor([4], dtype=torch.long),
        event_label_texts=["attack"],
        argument_label_texts=["place"],
        relation_threshold=0.5,
        relation_pair_budget=4,
    )

    assert len(decoded["events"]) == 3
    assert len(decoded["arguments"]) == 3


def test_event_reader_save_load_round_trip_preserves_resized_embeddings(
    tmp_path: Path,
) -> None:
    tokenizer = build_test_tokenizer(tmp_path)
    encoder_dir = tmp_path / "base_encoder"
    build_test_encoder(len(tokenizer) - 2).save_pretrained(encoder_dir)
    model = build_test_model(
        tokenizer,
        model_name=str(encoder_dir),
        encoder_vocab_size=len(tokenizer) - 2,
    )
    save_dir = tmp_path / "reader_checkpoint"

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    reloaded_model = EventReader.from_pretrained(save_dir)
    reloaded_tokenizer = BertTokenizerFast.from_pretrained(save_dir)

    assert model.config.encoder_vocab_size == len(tokenizer)
    assert reloaded_model.get_input_embeddings() is not None
    assert reloaded_model.get_input_embeddings().num_embeddings == len(
        reloaded_tokenizer
    )
    assert reloaded_model.config.encoder_vocab_size == len(reloaded_tokenizer)
    added_vocab = reloaded_tokenizer.get_added_vocab()
    assert EVENT_MARKER in added_vocab
    assert ARGUMENT_MARKER in added_vocab


def test_resolve_precision_flags_auto_uses_fp32_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_precision_flags("auto") == ("fp32", False, False)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_resolve_precision_flags_explicit_half_precision_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_precision_flags(precision)


def test_resolve_precision_flags_auto_prefers_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert resolve_precision_flags("auto") == ("bf16", False, True)


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


def test_trainer_smoke_runs_with_zero_role_samples(tmp_path: Path) -> None:
    train_path = tmp_path / "train_zero.jsonl"
    eval_path = tmp_path / "eval_zero.jsonl"
    write_jsonl(train_path, [build_zero_role_record(), build_record("doc-mixed")])
    write_jsonl(eval_path, [build_zero_role_record("doc-zero-eval")])

    tokenizer = build_test_tokenizer(tmp_path)
    model = build_test_model(tokenizer)
    train_dataset = EventReaderDataset(load_normalized_jsonl(train_path), tokenizer, 64)
    eval_dataset = EventReaderDataset(load_normalized_jsonl(eval_path), tokenizer, 64)
    trainer = EventArgumentTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "output_zero"),
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
    prediction = trainer.predict(eval_dataset)

    assert prediction["predictions"][0]["relations"] == []
    assert "test_relation_classification_f1" in prediction["metrics"]


def test_main_accepts_and_forwards_resume_from_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(train_path, [build_record()])
    write_jsonl(eval_path, [build_second_record()])

    tokenizer = build_test_tokenizer(tmp_path)
    encoder_dir = tmp_path / "encoder"
    build_test_encoder(len(tokenizer)).save_pretrained(encoder_dir)
    resume_dir = tmp_path / "checkpoint-7"
    resume_dir.mkdir()

    captured: dict[str, object] = {}

    class DummyTrainer:
        def __init__(
            self,
            *,
            model: EventReader,
            args: TrainingArguments,
            train_dataset: EventReaderDataset,
            eval_dataset: EventReaderDataset,
            data_collator: EventReaderCollator,
        ) -> None:
            captured["model"] = model
            captured["args"] = args
            captured["train_dataset"] = train_dataset
            captured["eval_dataset"] = eval_dataset
            captured["data_collator"] = data_collator

        def train(self, resume_from_checkpoint: str | None = None) -> None:
            captured["resume_from_checkpoint"] = resume_from_checkpoint

        def evaluate(self) -> dict[str, float]:
            return {
                "eval_event_span_f1": 0.0,
                "eval_best_relation_threshold": 0.7,
            }

        def save_model(self, output_dir: str) -> None:
            captured["save_model_output_dir"] = output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(train_module, "EventArgumentTrainer", DummyTrainer)
    monkeypatch.setattr(train_module, "build_tokenizer", lambda model_name: tokenizer)

    output_dir = tmp_path / "output"
    train_module.main(
        [
            "--model_name",
            str(encoder_dir),
            "--train_file",
            str(train_path),
            "--eval_file",
            str(eval_path),
            "--output_dir",
            str(output_dir),
            "--resume_from_checkpoint",
            str(resume_dir),
            "--precision",
            "fp32",
            "--batch_size",
            "1",
            "--num_epochs",
            "1",
            "--dataloader_num_workers",
            "0",
        ]
    )

    final_output_dir = output_dir / "final"
    saved_tokenizer = BertTokenizerFast.from_pretrained(final_output_dir)

    assert captured["resume_from_checkpoint"] == str(resume_dir)
    assert captured["save_model_output_dir"] == str(final_output_dir)
    assert final_output_dir.exists()
    assert captured["model"].config.relation_threshold == pytest.approx(0.7)
    added_vocab = saved_tokenizer.get_added_vocab()
    assert EVENT_MARKER in added_vocab
    assert ARGUMENT_MARKER in added_vocab


def test_main_forwards_candidate_sampling_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    ontology_path = write_candidate_ontology(tmp_path / "ontology.json")
    write_jsonl(train_path, [build_record()])
    write_jsonl(eval_path, [build_second_record()])

    tokenizer = build_test_tokenizer(tmp_path)
    encoder_dir = tmp_path / "encoder"
    build_test_encoder(len(tokenizer)).save_pretrained(encoder_dir)

    captured: dict[str, object] = {}

    class DummyTrainer:
        def __init__(
            self,
            *,
            model: EventReader,
            args: TrainingArguments,
            train_dataset: EventReaderDataset,
            eval_dataset: EventReaderDataset,
            data_collator: EventReaderCollator,
        ) -> None:
            captured["train_dataset"] = train_dataset
            captured["eval_dataset"] = eval_dataset

        def train(self, resume_from_checkpoint: str | None = None) -> None:
            captured["resume_from_checkpoint"] = resume_from_checkpoint

        def evaluate(self) -> dict[str, float]:
            return {
                "eval_event_span_f1": 0.0,
                "eval_best_relation_threshold": 0.5,
            }

        def save_model(self, output_dir: str) -> None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(train_module, "EventArgumentTrainer", DummyTrainer)
    monkeypatch.setattr(train_module, "build_tokenizer", lambda model_name: tokenizer)

    train_module.main(
        [
            "--model_name",
            str(encoder_dir),
            "--train_file",
            str(train_path),
            "--eval_file",
            str(eval_path),
            "--output_dir",
            str(tmp_path / "output"),
            "--precision",
            "fp32",
            "--batch_size",
            "1",
            "--num_epochs",
            "1",
            "--dataloader_num_workers",
            "0",
            "--ontology_file",
            str(ontology_path),
            "--num_event_candidates",
            "4",
            "--num_relation_candidates",
            "4",
            "--train_candidate_shuffle_prob",
            "0.25",
            "--train_gold_candidate_dropout_prob",
            "0.1",
            "--candidate_sampling_seed",
            "99",
        ]
    )

    train_dataset = captured["train_dataset"]
    eval_dataset = captured["eval_dataset"]
    assert isinstance(train_dataset, EventReaderDataset)
    assert isinstance(eval_dataset, EventReaderDataset)
    assert train_dataset.is_training is True
    assert eval_dataset.is_training is False
    assert train_dataset.num_event_candidates == 4
    assert train_dataset.num_relation_candidates == 4
    assert train_dataset.candidate_shuffle_probability == pytest.approx(0.25)
    assert train_dataset.gold_candidate_dropout_probability == pytest.approx(0.1)
    assert train_dataset.random_seed == 99
    assert train_dataset.ontology is not None
    assert len(train_dataset.samples[0].event_labels) == 4
    assert len(eval_dataset.samples[0].argument_labels) == 4


def test_maven_reader_converter_writes_zero_role_document(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "reader.jsonl"
    sample = {
        "id": "maven-doc-1",
        "title": "Synthetic",
        "tokens": [["An", "explosion", "happened", "yesterday"], ["Witnesses", "ran"]],
        "sentences": ["An explosion happened yesterday", "Witnesses ran"],
        "events": [
            {
                "id": "EVENT_1",
                "type": "Attack",
                "mention": [
                    {
                        "id": "mention-1",
                        "trigger_word": "explosion",
                        "sent_id": 0,
                        "offset": [1, 2],
                    }
                ],
            }
        ],
        "TIMEX": [],
        "temporal_relations": {},
        "causal_relations": {},
        "subevent_relations": [],
    }
    write_jsonl(input_path, [sample])

    subprocess.run(
        [
            sys.executable,
            "scripts/data/preprocess/reader/maven_to_reader.py",
            str(input_path),
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    converted = load_normalized_jsonl(output_path)
    assert len(converted) == 1
    assert converted[0].sample_id == "maven-doc-1"
    assert converted[0].argument_labels == []
    assert converted[0].relations == []
    assert converted[0].arguments[0].start == 1
    assert converted[0].arguments[0].end == 1


def test_maven_arg_reader_converter_supports_multi_relation_arguments(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "maven_arg.jsonl"
    output_path = tmp_path / "reader.jsonl"
    sample = {
        "id": "maven-arg-doc-1",
        "title": "Synthetic",
        "document": "Protesters attacked the embassy in Rome",
        "entities": [
            {
                "id": "ENTITY_1",
                "type": "Location",
                "mention": [
                    {
                        "id": "mention-1",
                        "mention": "Rome",
                        "offset": [35, 39],
                    }
                ],
            }
        ],
        "events": [
            {
                "id": "EVENT_1",
                "type": "Attack",
                "mention": [
                    {
                        "id": "event-mention-1",
                        "trigger_word": "attacked",
                        "offset": [11, 19],
                    }
                ],
                "argument": {
                    "Location": [{"entity_id": "ENTITY_1"}],
                    "Target": [{"entity_id": "ENTITY_1"}],
                },
            },
            {
                "id": "EVENT_2",
                "type": "Movement",
                "mention": [
                    {
                        "id": "event-mention-2",
                        "trigger_word": "Rome",
                        "offset": [35, 39],
                    }
                ],
                "argument": {
                    "Destination": [{"entity_id": "ENTITY_1"}],
                },
            },
        ],
        "negative_triggers": [],
    }
    write_jsonl(input_path, [sample])

    subprocess.run(
        [
            sys.executable,
            "scripts/data/preprocess/reader/maven_arg_to_reader.py",
            str(input_path),
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    converted = load_normalized_jsonl(output_path)
    assert len(converted) == 1
    assert [(span.start, span.end) for span in converted[0].arguments] == [(5, 5)]
    assert {
        (relation.event_idx, relation.argument_idx, relation.label)
        for relation in converted[0].relations
    } == {
        (0, 0, "location"),
        (0, 0, "target"),
        (1, 0, "destination"),
    }
