from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_RELATION_THRESHOLD = 0.5
INTERNAL_RELATION_PAIR_BUDGET = 256


def compute_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def apply_relation_budget(
    events: list[dict[str, Any]],
    arguments: list[dict[str, Any]],
    pair_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not events or not arguments or len(events) * len(arguments) <= pair_budget:
        return events, arguments

    kept_events = sorted(events, key=lambda item: item["score"], reverse=True)
    kept_arguments = sorted(arguments, key=lambda item: item["score"], reverse=True)
    while (
        kept_events
        and kept_arguments
        and len(kept_events) * len(kept_arguments) > pair_budget
    ):
        if len(kept_events) >= len(kept_arguments) and len(kept_events) > 1:
            kept_events.pop()
        elif len(kept_arguments) > 1:
            kept_arguments.pop()
        else:
            break
    return kept_events, kept_arguments


def decode_multi_label_relations(
    role_scores: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    if role_scores.numel() == 0:
        return decoded
    for event_idx in range(role_scores.shape[0]):
        for argument_idx in range(role_scores.shape[1]):
            for label_idx in range(role_scores.shape[2]):
                score = float(role_scores[event_idx, argument_idx, label_idx].item())
                if score < threshold:
                    continue
                decoded.append(
                    {
                        "event_idx": event_idx,
                        "argument_idx": argument_idx,
                        "label_idx": label_idx,
                        "score": score,
                    }
                )
    return decoded


def compute_reader_metrics(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    event_span_tp = event_span_fp = event_span_fn = 0
    event_type_tp = event_type_fp = event_type_fn = 0
    argument_span_tp = argument_span_fp = argument_span_fn = 0
    relation_id_tp = relation_id_fp = relation_id_fn = 0
    relation_cls_tp = relation_cls_fp = relation_cls_fn = 0

    for prediction, reference in zip(predictions, references):
        pred_event_spans = {
            (item["start"], item["end"]) for item in prediction["events"]
        }
        gold_event_spans = {
            (item["start"], item["end"]) for item in reference["events"]
        }

        pred_event_typed = {
            (item["start"], item["end"], item["label"]) for item in prediction["events"]
        }
        gold_event_typed = {
            (item["start"], item["end"], item["label"]) for item in reference["events"]
        }

        pred_argument_spans = {
            (item["start"], item["end"]) for item in prediction["arguments"]
        }
        gold_argument_spans = {
            (item["start"], item["end"]) for item in reference["arguments"]
        }

        pred_relation_id: set[tuple[int, int, int, int]] = set()
        pred_relation_cls: set[tuple[int, int, int, int, str]] = set()
        for relation in prediction["relations"]:
            event_span = prediction["events"][relation["event_idx"]]
            argument_span = prediction["arguments"][relation["argument_idx"]]
            relation_id = (
                event_span["start"],
                event_span["end"],
                argument_span["start"],
                argument_span["end"],
            )
            pred_relation_id.add(relation_id)
            pred_relation_cls.add((*relation_id, relation["label"]))

        gold_relation_id: set[tuple[int, int, int, int]] = set()
        gold_relation_cls: set[tuple[int, int, int, int, str]] = set()
        for relation in reference["relations"]:
            event_span = reference["events"][relation["event_idx"]]
            argument_span = reference["arguments"][relation["argument_idx"]]
            relation_id = (
                event_span["start"],
                event_span["end"],
                argument_span["start"],
                argument_span["end"],
            )
            gold_relation_id.add(relation_id)
            gold_relation_cls.add((*relation_id, relation["label"]))

        event_span_tp += len(pred_event_spans & gold_event_spans)
        event_span_fp += len(pred_event_spans - gold_event_spans)
        event_span_fn += len(gold_event_spans - pred_event_spans)

        event_type_tp += len(pred_event_typed & gold_event_typed)
        event_type_fp += len(pred_event_typed - gold_event_typed)
        event_type_fn += len(gold_event_typed - pred_event_typed)

        argument_span_tp += len(pred_argument_spans & gold_argument_spans)
        argument_span_fp += len(pred_argument_spans - gold_argument_spans)
        argument_span_fn += len(gold_argument_spans - pred_argument_spans)

        relation_id_tp += len(pred_relation_id & gold_relation_id)
        relation_id_fp += len(pred_relation_id - gold_relation_id)
        relation_id_fn += len(gold_relation_id - pred_relation_id)

        relation_cls_tp += len(pred_relation_cls & gold_relation_cls)
        relation_cls_fp += len(pred_relation_cls - gold_relation_cls)
        relation_cls_fn += len(gold_relation_cls - pred_relation_cls)

    metrics = {}
    for prefix, scores in (
        ("event_span", compute_prf(event_span_tp, event_span_fp, event_span_fn)),
        ("event_type", compute_prf(event_type_tp, event_type_fp, event_type_fn)),
        (
            "argument_span",
            compute_prf(argument_span_tp, argument_span_fp, argument_span_fn),
        ),
        (
            "relation_identification",
            compute_prf(relation_id_tp, relation_id_fp, relation_id_fn),
        ),
        (
            "relation_classification",
            compute_prf(relation_cls_tp, relation_cls_fp, relation_cls_fn),
        ),
    ):
        for metric_name, value in scores.items():
            metrics[f"{prefix}_{metric_name}"] = value
    return metrics


class EventReaderConfig(PretrainedConfig):
    model_type = "event_argument_reader"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        encoder_vocab_size: int | None = None,
        projection_dim: int = 256,
        hidden_dropout_prob: float = 0.1,
        relation_threshold: float = DEFAULT_RELATION_THRESHOLD,
        relation_pair_budget: int = INTERNAL_RELATION_PAIR_BUDGET,
        relation_loss_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.encoder_vocab_size = encoder_vocab_size
        self.projection_dim = projection_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.relation_threshold = relation_threshold
        self.relation_pair_budget = relation_pair_budget
        self.relation_loss_weight = relation_loss_weight


class EventReader(PreTrainedModel):
    config_class = EventReaderConfig

    def __init__(
        self,
        config: EventReaderConfig,
        encoder: PreTrainedModel | None = None,
    ) -> None:
        super().__init__(config)
        self.all_tied_weights_keys = {}
        default_device = getattr(torch, "get_default_device", lambda: torch.device("cpu"))()
        load_encoder_from_config = (
            encoder is None
            and isinstance(default_device, torch.device)
            and default_device.type == "meta"
        )
        encoder_from_config = None
        if encoder is None and load_encoder_from_config:
            encoder_config = AutoConfig.from_pretrained(config.model_name)
            if config.encoder_vocab_size is not None:
                encoder_config.vocab_size = config.encoder_vocab_size
            encoder_from_config = AutoModel.from_config(encoder_config)
        self.encoder = (
            encoder
            if encoder is not None
            else (
                encoder_from_config
                if encoder_from_config is not None
                else AutoModel.from_pretrained(config.model_name, dtype=torch.float32)
            )
        )
        input_embeddings = self.get_input_embeddings()
        if input_embeddings is None:
            raise ValueError("EventReader encoder must define input embeddings")
        current_vocab_size = input_embeddings.num_embeddings
        if config.encoder_vocab_size is None:
            config.encoder_vocab_size = current_vocab_size
        elif current_vocab_size != config.encoder_vocab_size:
            self.resize_token_embeddings(config.encoder_vocab_size)
        hidden_size = self.encoder.config.hidden_size
        projection_dim = config.projection_dim

        self.event_start_head = self._make_classifier(
            hidden_size, 2, config.hidden_dropout_prob
        )
        self.event_end_head = self._make_classifier(
            hidden_size * 2, 2, config.hidden_dropout_prob
        )
        self.argument_start_head = self._make_classifier(
            hidden_size, 2, config.hidden_dropout_prob
        )
        self.argument_end_head = self._make_classifier(
            hidden_size * 2, 2, config.hidden_dropout_prob
        )
        self.event_span_projector = self._make_mlp(
            hidden_size * 2,
            projection_dim,
            config.hidden_dropout_prob,
        )
        self.event_label_projector = self._make_mlp(
            hidden_size,
            projection_dim,
            config.hidden_dropout_prob,
        )
        self.argument_span_projector = self._make_mlp(
            hidden_size * 2,
            projection_dim,
            config.hidden_dropout_prob,
        )
        self.relation_event_projector = self._make_mlp(
            hidden_size * 2,
            projection_dim,
            config.hidden_dropout_prob,
        )
        self.relation_argument_projector = self._make_mlp(
            hidden_size * 2,
            projection_dim,
            config.hidden_dropout_prob,
        )
        self.relation_role_projector = self._make_mlp(
            hidden_size,
            projection_dim,
            config.hidden_dropout_prob,
        )
        self.relation_classifier = self._make_classifier(
            projection_dim,
            2,
            config.hidden_dropout_prob,
        )

        self._init_linear_layers(
            [
                self.event_start_head,
                self.event_end_head,
                self.argument_start_head,
                self.argument_end_head,
                self.event_span_projector,
                self.event_label_projector,
                self.argument_span_projector,
                self.relation_event_projector,
                self.relation_argument_projector,
                self.relation_role_projector,
                self.relation_classifier,
            ]
        )

    def _init_linear_layers(self, modules: list[nn.Module]) -> None:
        for parent_module in modules:
            for module in parent_module.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _make_mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _make_classifier(
        input_dim: int, output_dim: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        input_embeddings = self.encoder.resize_token_embeddings(
            new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        self.config.encoder_vocab_size = input_embeddings.num_embeddings
        return input_embeddings

    def get_input_embeddings(self) -> nn.Module | None:
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.encoder.set_input_embeddings(value)
        num_embeddings = getattr(value, "num_embeddings", None)
        if isinstance(num_embeddings, int):
            self.config.encoder_vocab_size = num_embeddings

    @staticmethod
    def _gather_positions(
        hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        safe_positions = positions.clamp(min=0)
        gathered = hidden_states.gather(
            1,
            safe_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
        )
        gathered = gathered.masked_fill((positions < 0).unsqueeze(-1), 0.0)
        return gathered

    @staticmethod
    def _pool_token_spans(
        hidden_states: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        token_indices = torch.arange(
            sequence_length,
            device=hidden_states.device,
        ).view(1, 1, -1)
        valid_spans = (start_positions >= 0) & (end_positions >= 0)
        span_mask = (
            valid_spans.unsqueeze(-1)
            & (token_indices >= start_positions.unsqueeze(-1))
            & (token_indices <= end_positions.unsqueeze(-1))
        )
        weights = span_mask.to(hidden_states.dtype)
        pooled = torch.einsum("blt,bth->blh", weights, hidden_states)
        counts = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return pooled / counts

    def _build_label_representations(
        self,
        hidden_states: torch.Tensor,
        marker_positions: torch.Tensor,
        label_token_starts: torch.Tensor | None,
        label_token_ends: torch.Tensor | None,
    ) -> torch.Tensor:
        if label_token_starts is None or label_token_ends is None:
            return self._gather_positions(hidden_states, marker_positions)
        return self._pool_token_spans(
            hidden_states,
            label_token_starts,
            label_token_ends,
        )

    @staticmethod
    def _mask_invalid_columns(
        logits: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        mask_value = torch.finfo(logits.dtype).min
        return logits.masked_fill((positions < 0).unsqueeze(1), mask_value)

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def _align_custom_head_dtype(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = hidden_states.dtype
        for module in self.event_start_head.modules():
            if isinstance(module, nn.Linear):
                target_dtype = module.weight.dtype
                break
        if hidden_states.dtype == target_dtype:
            return hidden_states
        return hidden_states.to(target_dtype)

    def _compute_conditioned_end_logits(
        self,
        hidden_states: torch.Tensor,
        start_positions: torch.Tensor,
        end_head: nn.Module,
    ) -> torch.Tensor:
        start_states = self._gather_positions(hidden_states, start_positions)
        token_states = hidden_states.unsqueeze(1).expand(
            -1, start_positions.shape[1], -1, -1
        )
        conditioned_features = torch.cat(
            [
                start_states.unsqueeze(2).expand(-1, -1, hidden_states.shape[1], -1),
                token_states,
            ],
            dim=-1,
        )
        return end_head(conditioned_features)

    def _build_conditioned_end_labels(
        self,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
        word_end_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, span_count = start_positions.shape
        sequence_length = word_end_mask.shape[1]
        labels = torch.full(
            (batch_size, span_count, sequence_length),
            -100,
            dtype=torch.long,
            device=start_positions.device,
        )
        token_indices = torch.arange(sequence_length, device=start_positions.device)
        valid_starts = start_positions >= 0
        valid_ends = word_end_mask.unsqueeze(1).bool()
        end_after_start = token_indices.view(1, 1, -1) >= start_positions.unsqueeze(-1)
        valid_positions = valid_starts.unsqueeze(-1) & valid_ends & end_after_start
        labels = labels.masked_fill(valid_positions, 0)

        valid_gold = valid_starts & (end_positions >= 0)
        if valid_gold.any():
            batch_indices, span_indices = valid_gold.nonzero(as_tuple=True)
            labels[batch_indices, span_indices, end_positions[batch_indices, span_indices]] = 1
        return labels

    def _build_span_features(
        self,
        hidden_states: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        start_states = self._gather_positions(hidden_states, starts)
        end_states = self._gather_positions(hidden_states, ends)
        return torch.cat([start_states, end_states], dim=-1)

    def _compute_event_type_logits(
        self,
        span_representations: torch.Tensor,
        label_representations: torch.Tensor,
        label_positions: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.bmm(
            span_representations,
            label_representations.transpose(1, 2),
        )
        return self._mask_invalid_columns(logits, label_positions)

    def compute_relation_logits(
        self,
        event_span_features: torch.Tensor,
        argument_span_features: torch.Tensor,
        role_label_features: torch.Tensor,
    ) -> torch.Tensor:
        event_features = self.relation_event_projector(event_span_features)
        argument_features = self.relation_argument_projector(argument_span_features)
        role_features = self.relation_role_projector(role_label_features)
        interaction = torch.einsum(
            "bnd,bmd,brd->bnmrd",
            event_features,
            argument_features,
            role_features,
        )
        return self.relation_classifier(interaction)

    def _loss_or_zero(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros(())
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def _decode_spans(
        self,
        hidden_states: torch.Tensor,
        start_scores: torch.Tensor,
        start_logits: torch.Tensor,
        end_head: nn.Module,
        word_start_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
        token_to_word: torch.Tensor,
    ) -> list[dict[str, Any]]:
        start_predictions = start_logits.argmax(dim=-1)
        candidate_starts = [
            index
            for index, prediction in enumerate(start_predictions.tolist())
            if prediction == 1 and int(word_start_mask[index].item()) == 1
        ]
        decoded: list[dict[str, Any]] = []
        seen_word_spans: set[tuple[int, int]] = set()
        if not candidate_starts:
            return decoded

        start_positions = torch.tensor(
            [candidate_starts],
            device=hidden_states.device,
            dtype=torch.long,
        )
        conditioned_end_logits = self._compute_conditioned_end_logits(
            hidden_states.unsqueeze(0),
            start_positions,
            end_head,
        )[0]
        conditioned_end_scores = torch.softmax(conditioned_end_logits, dim=-1)[..., 1]

        for start_idx in candidate_starts:
            start_offset = candidate_starts.index(start_idx)
            end_scores = conditioned_end_scores[start_offset]
            valid_end_mask = word_end_mask.bool() & (
                torch.arange(word_end_mask.shape[0], device=word_end_mask.device)
                >= start_idx
            )
            if not valid_end_mask.any():
                continue
            masked_end_scores = end_scores.masked_fill(
                ~valid_end_mask, torch.finfo(end_scores.dtype).min
            )
            end_idx = int(masked_end_scores.argmax().item())
            start_word = int(token_to_word[start_idx].item())
            end_word = int(token_to_word[end_idx].item())
            if start_word < 0 or end_word < start_word:
                continue
            word_span = (start_word, end_word)
            if word_span in seen_word_spans:
                continue
            seen_word_spans.add(word_span)
            decoded.append(
                {
                    "token_start": start_idx,
                    "token_end": end_idx,
                    "start": start_word,
                    "end": end_word,
                    "score": math.sqrt(
                        float(start_scores[start_idx].item())
                        * float(end_scores[end_idx].item())
                    ),
                }
            )
        return sorted(decoded, key=lambda item: item["score"], reverse=True)

    def _decode_document(
        self,
        hidden_states: torch.Tensor,
        event_start_logits: torch.Tensor,
        argument_start_logits: torch.Tensor,
        word_start_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
        token_to_word: torch.Tensor,
        event_marker_positions: torch.Tensor,
        argument_marker_positions: torch.Tensor,
        event_label_token_starts: torch.Tensor,
        event_label_token_ends: torch.Tensor,
        argument_label_token_starts: torch.Tensor,
        argument_label_token_ends: torch.Tensor,
        event_label_texts: list[str],
        argument_label_texts: list[str],
        relation_threshold: float,
        relation_pair_budget: int,
    ) -> dict[str, Any]:
        event_start_probs = torch.softmax(event_start_logits, dim=-1)[..., 1]
        argument_start_probs = torch.softmax(argument_start_logits, dim=-1)[..., 1]
        event_predictions = self._decode_spans(
            hidden_states,
            event_start_probs,
            event_start_logits,
            self.event_end_head,
            word_start_mask,
            word_end_mask,
            token_to_word,
        )
        argument_predictions = self._decode_spans(
            hidden_states,
            argument_start_probs,
            argument_start_logits,
            self.argument_end_head,
            word_start_mask,
            word_end_mask,
            token_to_word,
        )

        if event_predictions and event_label_texts:
            event_starts = torch.tensor(
                [[item["token_start"] for item in event_predictions]],
                device=hidden_states.device,
            )
            event_ends = torch.tensor(
                [[item["token_end"] for item in event_predictions]],
                device=hidden_states.device,
            )
            event_span_features = self._build_span_features(
                hidden_states.unsqueeze(0),
                event_starts,
                event_ends,
            )
            event_label_repr = self.event_label_projector(
                self._build_label_representations(
                    hidden_states.unsqueeze(0),
                    event_marker_positions.unsqueeze(0),
                    event_label_token_starts.unsqueeze(0),
                    event_label_token_ends.unsqueeze(0),
                )
            )
            type_logits = self._compute_event_type_logits(
                self.event_span_projector(event_span_features),
                event_label_repr,
                event_marker_positions.unsqueeze(0),
            )[0]
            type_probs = torch.softmax(type_logits, dim=-1)
            for event_index, prediction in enumerate(event_predictions):
                label_index = int(type_probs[event_index].argmax().item())
                prediction["label"] = event_label_texts[label_index]
                prediction["score"] *= float(
                    type_probs[event_index, label_index].item()
                )
        else:
            for prediction in event_predictions:
                prediction["label"] = (
                    event_label_texts[0] if event_label_texts else "event"
                )

        budgeted_events, budgeted_arguments = apply_relation_budget(
            [
                {**item, "original_idx": index}
                for index, item in enumerate(event_predictions)
            ],
            [
                {**item, "original_idx": index}
                for index, item in enumerate(argument_predictions)
            ],
            relation_pair_budget,
        )

        relation_candidates: list[dict[str, Any]] = []
        if budgeted_events and budgeted_arguments and argument_label_texts:
            event_starts = torch.tensor(
                [[item["token_start"] for item in budgeted_events]],
                device=hidden_states.device,
            )
            event_ends = torch.tensor(
                [[item["token_end"] for item in budgeted_events]],
                device=hidden_states.device,
            )
            argument_starts = torch.tensor(
                [[item["token_start"] for item in budgeted_arguments]],
                device=hidden_states.device,
            )
            argument_ends = torch.tensor(
                [[item["token_end"] for item in budgeted_arguments]],
                device=hidden_states.device,
            )
            role_positions = argument_marker_positions.unsqueeze(0)
            relation_logits = self.compute_relation_logits(
                self._build_span_features(
                    hidden_states.unsqueeze(0),
                    event_starts,
                    event_ends,
                ),
                self._build_span_features(
                    hidden_states.unsqueeze(0),
                    argument_starts,
                    argument_ends,
                ),
                self._build_label_representations(
                    hidden_states.unsqueeze(0),
                    role_positions,
                    argument_label_token_starts.unsqueeze(0),
                    argument_label_token_ends.unsqueeze(0),
                ),
            )[0]
            invalid_role_mask = (argument_marker_positions < 0).view(1, 1, -1, 1)
            relation_logits = relation_logits.masked_fill(
                invalid_role_mask,
                torch.finfo(relation_logits.dtype).min,
            )
            positive_scores = torch.softmax(relation_logits, dim=-1)[..., 1]
            decoded_relations = decode_multi_label_relations(positive_scores, 0.0)
            for relation in decoded_relations:
                if relation["label_idx"] >= len(argument_label_texts):
                    continue
                relation_candidates.append(
                    {
                        "event_idx": budgeted_events[relation["event_idx"]][
                            "original_idx"
                        ],
                        "argument_idx": budgeted_arguments[relation["argument_idx"]][
                            "original_idx"
                        ],
                        "label": argument_label_texts[relation["label_idx"]],
                        "score": relation["score"],
                    }
                )

        relation_predictions = [
            relation
            for relation in relation_candidates
            if relation["score"] >= relation_threshold
        ]

        return {
            "events": [
                {
                    "start": item["start"],
                    "end": item["end"],
                    "label": item["label"],
                    "score": item["score"],
                }
                for item in event_predictions
            ],
            "arguments": [
                {"start": item["start"], "end": item["end"], "score": item["score"]}
                for item in argument_predictions
            ],
            "relations": relation_predictions,
            "relation_candidates": relation_candidates,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_marker_positions: torch.Tensor,
        argument_marker_positions: torch.Tensor,
        event_label_token_starts: torch.Tensor | None = None,
        event_label_token_ends: torch.Tensor | None = None,
        argument_label_token_starts: torch.Tensor | None = None,
        argument_label_token_ends: torch.Tensor | None = None,
        word_start_mask: torch.Tensor | None = None,
        word_end_mask: torch.Tensor | None = None,
        token_to_word: torch.Tensor | None = None,
        event_start_labels: torch.Tensor | None = None,
        event_end_labels: torch.Tensor | None = None,
        argument_start_labels: torch.Tensor | None = None,
        argument_end_labels: torch.Tensor | None = None,
        gold_event_token_starts: torch.Tensor | None = None,
        gold_event_token_ends: torch.Tensor | None = None,
        gold_event_type_labels: torch.Tensor | None = None,
        gold_argument_token_starts: torch.Tensor | None = None,
        gold_argument_token_ends: torch.Tensor | None = None,
        relation_labels: torch.Tensor | None = None,
        event_label_texts: list[list[str]] | None = None,
        argument_label_texts: list[list[str]] | None = None,
        decode_predictions: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        hidden_states = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        aligned_hidden_states = self._align_custom_head_dtype(hidden_states)

        event_start_logits = self.event_start_head(aligned_hidden_states)
        argument_start_logits = self.argument_start_head(aligned_hidden_states)
        event_end_logits = None
        argument_end_logits = None

        outputs: dict[str, Any] = {
            "event_start_logits": event_start_logits,
            "event_end_logits": event_end_logits,
            "argument_start_logits": argument_start_logits,
            "argument_end_logits": argument_end_logits,
        }

        losses: list[tuple[torch.Tensor, float]] = []
        if event_start_labels is not None:
            event_start_loss = self._loss_or_zero(
                event_start_logits, event_start_labels
            )
            outputs["loss_event_start"] = event_start_loss
            losses.append((event_start_loss, 1.0))
        if event_end_labels is not None:
            if gold_event_token_starts is None or gold_event_token_ends is None:
                raise ValueError(
                    "Conditioned event end loss requires gold start and end positions"
                )
            conditioned_event_end_logits = self._compute_conditioned_end_logits(
                aligned_hidden_states,
                gold_event_token_starts,
                self.event_end_head,
            )
            conditioned_event_end_labels = self._build_conditioned_end_labels(
                gold_event_token_starts,
                gold_event_token_ends,
                word_end_mask if word_end_mask is not None else (event_end_labels != -100),
            )
            outputs["event_end_logits"] = conditioned_event_end_logits
            event_end_loss = self._loss_or_zero(
                conditioned_event_end_logits, conditioned_event_end_labels
            )
            outputs["loss_event_end"] = event_end_loss
            losses.append((event_end_loss, 1.0))
        if argument_start_labels is not None:
            argument_start_loss = self._loss_or_zero(
                argument_start_logits, argument_start_labels
            )
            outputs["loss_argument_start"] = argument_start_loss
            losses.append((argument_start_loss, 1.0))
        if argument_end_labels is not None:
            if gold_argument_token_starts is None or gold_argument_token_ends is None:
                raise ValueError(
                    "Conditioned argument end loss requires gold start and end positions"
                )
            conditioned_argument_end_logits = self._compute_conditioned_end_logits(
                aligned_hidden_states,
                gold_argument_token_starts,
                self.argument_end_head,
            )
            conditioned_argument_end_labels = self._build_conditioned_end_labels(
                gold_argument_token_starts,
                gold_argument_token_ends,
                word_end_mask
                if word_end_mask is not None
                else (argument_end_labels != -100),
            )
            outputs["argument_end_logits"] = conditioned_argument_end_logits
            argument_end_loss = self._loss_or_zero(
                conditioned_argument_end_logits, conditioned_argument_end_labels
            )
            outputs["loss_argument_end"] = argument_end_loss
            losses.append((argument_end_loss, 1.0))

        event_label_states = self._build_label_representations(
            aligned_hidden_states,
            event_marker_positions,
            event_label_token_starts,
            event_label_token_ends,
        )
        argument_label_states = self._build_label_representations(
            aligned_hidden_states,
            argument_marker_positions,
            argument_label_token_starts,
            argument_label_token_ends,
        )

        if (
            gold_event_token_starts is not None
            and gold_event_token_ends is not None
            and gold_event_type_labels is not None
            and gold_event_token_starts.shape[1] > 0
        ):
            gold_event_features = self._build_span_features(
                aligned_hidden_states,
                gold_event_token_starts,
                gold_event_token_ends,
            )
            event_type_logits = self._compute_event_type_logits(
                self.event_span_projector(gold_event_features),
                self.event_label_projector(event_label_states),
                event_marker_positions,
            )
            outputs["event_type_logits"] = event_type_logits
            event_type_loss = self._loss_or_zero(
                event_type_logits, gold_event_type_labels
            )
            outputs["loss_event_type"] = event_type_loss
            losses.append((event_type_loss, 1.0))
        else:
            gold_event_features = None

        if (
            gold_argument_token_starts is not None
            and gold_argument_token_ends is not None
            and gold_argument_token_starts.shape[1] > 0
        ):
            gold_argument_features = self._build_span_features(
                aligned_hidden_states,
                gold_argument_token_starts,
                gold_argument_token_ends,
            )
        else:
            gold_argument_features = None

        if (
            gold_event_token_starts is not None
            and gold_argument_token_starts is not None
            and relation_labels is not None
            and gold_event_features is not None
            and gold_argument_features is not None
            and argument_marker_positions.shape[1] > 0
        ):
            relation_logits = self.compute_relation_logits(
                gold_event_features,
                gold_argument_features,
                argument_label_states,
            )
            outputs["relation_logits"] = relation_logits
            relation_loss = self._loss_or_zero(relation_logits, relation_labels)
            outputs["loss_relation"] = relation_loss
            losses.append((relation_loss, self.config.relation_loss_weight))

        if losses:
            weighted_losses = torch.stack([loss * weight for loss, weight in losses])
            total_weight = sum(weight for _, weight in losses)
            outputs["loss"] = weighted_losses.sum() / max(total_weight, 1e-8)

        if decode_predictions:
            if (
                word_start_mask is None
                or word_end_mask is None
                or token_to_word is None
                or event_label_texts is None
                or argument_label_texts is None
            ):
                raise ValueError(
                    "Decoding requires word masks, token-to-word map, and label texts"
                )
            outputs["decoded_predictions"] = [
                self._decode_document(
                    hidden_states=aligned_hidden_states[batch_index],
                    event_start_logits=event_start_logits[batch_index],
                    argument_start_logits=argument_start_logits[batch_index],
                    word_start_mask=word_start_mask[batch_index],
                    word_end_mask=word_end_mask[batch_index],
                    token_to_word=token_to_word[batch_index],
                    event_marker_positions=event_marker_positions[batch_index],
                    argument_marker_positions=argument_marker_positions[batch_index],
                    event_label_token_starts=event_label_token_starts[batch_index],
                    event_label_token_ends=event_label_token_ends[batch_index],
                    argument_label_token_starts=argument_label_token_starts[batch_index],
                    argument_label_token_ends=argument_label_token_ends[batch_index],
                    event_label_texts=event_label_texts[batch_index],
                    argument_label_texts=argument_label_texts[batch_index],
                    relation_threshold=self.config.relation_threshold,
                    relation_pair_budget=self.config.relation_pair_budget,
                )
                for batch_index in range(input_ids.shape[0])
            ]

        return outputs
