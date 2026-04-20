from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_EVENT_THRESHOLD = 0.5
DEFAULT_ARGUMENT_THRESHOLD = 0.5
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


def decode_single_label_relations(
    role_scores: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    if role_scores.numel() == 0:
        return decoded
    best_scores, best_labels = role_scores.max(dim=-1)
    for event_idx in range(role_scores.shape[0]):
        for argument_idx in range(role_scores.shape[1]):
            score = float(best_scores[event_idx, argument_idx].item())
            if score < threshold:
                continue
            decoded.append(
                {
                    "event_idx": event_idx,
                    "argument_idx": argument_idx,
                    "label_idx": int(best_labels[event_idx, argument_idx].item()),
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
        projection_dim: int = 256,
        hidden_dropout_prob: float = 0.1,
        event_threshold: float = DEFAULT_EVENT_THRESHOLD,
        argument_threshold: float = DEFAULT_ARGUMENT_THRESHOLD,
        relation_threshold: float = DEFAULT_RELATION_THRESHOLD,
        relation_pair_budget: int = INTERNAL_RELATION_PAIR_BUDGET,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.event_threshold = event_threshold
        self.argument_threshold = argument_threshold
        self.relation_threshold = relation_threshold
        self.relation_pair_budget = relation_pair_budget


class EventReader(PreTrainedModel):
    config_class = EventReaderConfig

    def __init__(
        self,
        config: EventReaderConfig,
        encoder: PreTrainedModel | None = None,
    ) -> None:
        super().__init__(config)
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(config.model_name)
        )
        hidden_size = self.encoder.config.hidden_size
        projection_dim = config.projection_dim

        self.event_start_head = self._make_classifier(
            hidden_size, 2, config.hidden_dropout_prob
        )
        self.event_end_head = self._make_classifier(
            hidden_size, 2, config.hidden_dropout_prob
        )
        self.argument_start_head = self._make_classifier(
            hidden_size, 2, config.hidden_dropout_prob
        )
        self.argument_end_head = self._make_classifier(
            hidden_size, 2, config.hidden_dropout_prob
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

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        return self.encoder.resize_token_embeddings(new_num_tokens)

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
    def _mask_invalid_columns(
        logits: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return logits.masked_fill((positions < 0).unsqueeze(1), -1e9)

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def _span_probabilities(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_probs = torch.softmax(start_logits, dim=-1)[..., 1]
        end_probs = torch.softmax(end_logits, dim=-1)[..., 1]
        return start_probs, end_probs

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
        start_scores: torch.Tensor,
        end_scores: torch.Tensor,
        word_start_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
        token_to_word: torch.Tensor,
        threshold: float,
    ) -> list[dict[str, Any]]:
        candidate_starts = [
            index
            for index, score in enumerate(start_scores.tolist())
            if score >= threshold and int(word_start_mask[index].item()) == 1
        ]
        candidate_ends = [
            index
            for index, score in enumerate(end_scores.tolist())
            if score >= threshold and int(word_end_mask[index].item()) == 1
        ]
        decoded: list[dict[str, Any]] = []
        seen_word_spans: set[tuple[int, int]] = set()
        for start_idx in candidate_starts:
            valid_ends = [end_idx for end_idx in candidate_ends if end_idx >= start_idx]
            if not valid_ends:
                continue
            end_idx = max(
                valid_ends, key=lambda candidate: float(end_scores[candidate].item())
            )
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
        event_start_probs: torch.Tensor,
        event_end_probs: torch.Tensor,
        argument_start_probs: torch.Tensor,
        argument_end_probs: torch.Tensor,
        word_start_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
        token_to_word: torch.Tensor,
        event_marker_positions: torch.Tensor,
        argument_marker_positions: torch.Tensor,
        event_label_texts: list[str],
        argument_label_texts: list[str],
        event_threshold: float,
        argument_threshold: float,
        relation_threshold: float,
        relation_pair_budget: int,
    ) -> dict[str, Any]:
        event_predictions = self._decode_spans(
            event_start_probs,
            event_end_probs,
            word_start_mask,
            word_end_mask,
            token_to_word,
            event_threshold,
        )
        argument_predictions = self._decode_spans(
            argument_start_probs,
            argument_end_probs,
            word_start_mask,
            word_end_mask,
            token_to_word,
            argument_threshold,
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
                self._gather_positions(
                    hidden_states.unsqueeze(0), event_marker_positions.unsqueeze(0)
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

        event_predictions, argument_predictions = apply_relation_budget(
            event_predictions,
            argument_predictions,
            relation_pair_budget,
        )

        relation_predictions: list[dict[str, Any]] = []
        if event_predictions and argument_predictions and argument_label_texts:
            event_starts = torch.tensor(
                [[item["token_start"] for item in event_predictions]],
                device=hidden_states.device,
            )
            event_ends = torch.tensor(
                [[item["token_end"] for item in event_predictions]],
                device=hidden_states.device,
            )
            argument_starts = torch.tensor(
                [[item["token_start"] for item in argument_predictions]],
                device=hidden_states.device,
            )
            argument_ends = torch.tensor(
                [[item["token_end"] for item in argument_predictions]],
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
                self._gather_positions(hidden_states.unsqueeze(0), role_positions),
            )[0]
            positive_scores = torch.softmax(relation_logits, dim=-1)[..., 1]
            decoded_relations = decode_single_label_relations(
                positive_scores, relation_threshold
            )
            for relation in decoded_relations:
                relation_predictions.append(
                    {
                        "event_idx": relation["event_idx"],
                        "argument_idx": relation["argument_idx"],
                        "label": argument_label_texts[relation["label_idx"]],
                        "score": relation["score"],
                    }
                )

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
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_marker_positions: torch.Tensor,
        argument_marker_positions: torch.Tensor,
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

        event_start_logits = self.event_start_head(hidden_states)
        event_end_logits = self.event_end_head(hidden_states)
        argument_start_logits = self.argument_start_head(hidden_states)
        argument_end_logits = self.argument_end_head(hidden_states)

        outputs: dict[str, Any] = {
            "event_start_logits": event_start_logits,
            "event_end_logits": event_end_logits,
            "argument_start_logits": argument_start_logits,
            "argument_end_logits": argument_end_logits,
        }

        losses: list[torch.Tensor] = []
        if event_start_labels is not None:
            losses.append(self._loss_or_zero(event_start_logits, event_start_labels))
        if event_end_labels is not None:
            losses.append(self._loss_or_zero(event_end_logits, event_end_labels))
        if argument_start_labels is not None:
            losses.append(
                self._loss_or_zero(argument_start_logits, argument_start_labels)
            )
        if argument_end_labels is not None:
            losses.append(self._loss_or_zero(argument_end_logits, argument_end_labels))

        event_label_states = self._gather_positions(
            hidden_states, event_marker_positions
        )
        argument_label_states = self._gather_positions(
            hidden_states, argument_marker_positions
        )

        if (
            gold_event_token_starts is not None
            and gold_event_token_ends is not None
            and gold_event_type_labels is not None
            and gold_event_token_starts.shape[1] > 0
        ):
            gold_event_features = self._build_span_features(
                hidden_states,
                gold_event_token_starts,
                gold_event_token_ends,
            )
            event_type_logits = self._compute_event_type_logits(
                self.event_span_projector(gold_event_features),
                self.event_label_projector(event_label_states),
                event_marker_positions,
            )
            outputs["event_type_logits"] = event_type_logits
            losses.append(self._loss_or_zero(event_type_logits, gold_event_type_labels))
        else:
            gold_event_features = None

        if (
            gold_argument_token_starts is not None
            and gold_argument_token_ends is not None
            and gold_argument_token_starts.shape[1] > 0
        ):
            gold_argument_features = self._build_span_features(
                hidden_states,
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
            losses.append(self._loss_or_zero(relation_logits, relation_labels))

        if losses:
            outputs["loss"] = torch.stack(losses).mean()

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
            event_start_probs, event_end_probs = self._span_probabilities(
                event_start_logits, event_end_logits
            )
            argument_start_probs, argument_end_probs = self._span_probabilities(
                argument_start_logits, argument_end_logits
            )
            outputs["decoded_predictions"] = [
                self._decode_document(
                    hidden_states=hidden_states[batch_index],
                    event_start_probs=event_start_probs[batch_index],
                    event_end_probs=event_end_probs[batch_index],
                    argument_start_probs=argument_start_probs[batch_index],
                    argument_end_probs=argument_end_probs[batch_index],
                    word_start_mask=word_start_mask[batch_index],
                    word_end_mask=word_end_mask[batch_index],
                    token_to_word=token_to_word[batch_index],
                    event_marker_positions=event_marker_positions[batch_index],
                    argument_marker_positions=argument_marker_positions[batch_index],
                    event_label_texts=event_label_texts[batch_index],
                    argument_label_texts=argument_label_texts[batch_index],
                    event_threshold=self.config.event_threshold,
                    argument_threshold=self.config.argument_threshold,
                    relation_threshold=self.config.relation_threshold,
                    relation_pair_budget=self.config.relation_pair_budget,
                )
                for batch_index in range(input_ids.shape[0])
            ]

        return outputs
