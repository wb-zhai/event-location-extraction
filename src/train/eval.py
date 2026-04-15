from typing import Any

from gliner2 import GLiNER2
from gliner2.training.data import TrainDataInput, TrainingDataset


class EventArgumentExtractionEvaluatorGliNER2:
    def __init__(
        self,
        event_types: list[str],
        argument_types: list[str],
        threshold: float = 0.5,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.event_types = event_types
        self.argument_types = argument_types
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def _find_exact_matches(text: str, mention: str) -> set[tuple[int, int]]:
        if not mention:
            return set()

        text_lower = text.lower()
        mention_lower = mention.lower()
        matches = set()
        start = 0

        while True:
            match_start = text_lower.find(mention_lower, start)
            if match_start == -1:
                break
            matches.add((match_start, match_start + len(mention)))
            start = match_start + 1

        return matches

    @staticmethod
    def _compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def _iter_relations(doc):
        if not doc:
            return

        if isinstance(doc, dict):
            for role, relations in doc.items():
                relations_list = relations if isinstance(relations, list) else [relations]
                for relation in relations_list:
                    if relation is not None:
                        yield role, relation
            return

        if isinstance(doc, list):
            for relation in doc:
                if hasattr(relation, "to_dict"):
                    relation = relation.to_dict()

                if not isinstance(relation, dict):
                    continue

                for role, payload in relation.items():
                    if payload is not None:
                        yield role, payload

    @staticmethod
    def _extract_span(value) -> tuple[int, int] | None:
        if not isinstance(value, dict):
            return None

        start = value.get("start")
        end = value.get("end")
        if start is None or end is None:
            return None

        return start, end

    def __call__(self, model: GLiNER2, data: TrainingDataset) -> dict[str, Any]:
        metrics = {}

        schema = (
            model.create_schema()
            .entities(self.event_types)
            .relations(self.argument_types)
        )

        texts = [sample.text for sample in data]

        results = model.batch_extract(
            texts,
            schema,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            include_confidence=True,
            include_spans=True,
        )

        event_preds = [sample["entities"] for sample in results]
        argument_preds = [sample["relation_extraction"] for sample in results]

        event_golds = [sample.entities for sample in data]
        argument_golds = [sample.relations for sample in data]

        event_metrics = self.eval_events(texts, event_preds, event_golds)
        argument_metrics = self.eval_arguments(texts, argument_preds, argument_golds)

        # Event extraction metrics
        metrics["event_span_f1"] = event_metrics["span_f1"]
        metrics["event_span_precision"] = event_metrics["span_precision"]
        metrics["event_span_recall"] = event_metrics["span_recall"]

        metrics["event_f1"] = event_metrics["f1"]
        metrics["event_precision"] = event_metrics["precision"]
        metrics["event_recall"] = event_metrics["recall"]

        # Argument extraction metrics
        metrics["arg_i_precision"] = argument_metrics["arg_i_precision"]
        metrics["arg_i_recall"] = argument_metrics["arg_i_recall"]
        metrics["arg_i_f1"] = argument_metrics["arg_i_f1"]

        metrics["arg_c_precision"] = argument_metrics["arg_c_precision"]
        metrics["arg_c_recall"] = argument_metrics["arg_c_recall"]
        metrics["arg_c_f1"] = argument_metrics["arg_c_f1"]

        return metrics

    def eval_events(self, texts, preds, golds):
        """
        Evaluate event extraction performance (span-level and type-level).

        Span-level: it measure whether the predicted event mentions match the gold mentions, regardless of the event type.
        Type-level: it measure whether the predicted event types match the gold event types, together with the span match
        (if the span does not match, it is counted as incorrect regardless of the type prediction).

        Args:
            texts: List of input texts.
            preds: List of predicted events, where each event is represented as a dictionary with the event type as key, 
                and the value is a list of predicted event mentions (spans) in the format {"start": int, "end": int, "text": str, "confidence": float}.
            golds: List of gold events, where each event is represented as a dictionary with the type as key, and the value is a list of gold event mentions (spans) strings, e.g. {"EventType": ["event mention 1", "event mention 2"]}.
        
        Returns:
            A dictionary containing the evaluation metrics for event extraction, including span-level and type-level precision, recall, and F1 score.
        """
        span_tp = span_fp = span_fn = 0
        typed_tp = typed_fp = typed_fn = 0

        for text, pred_doc, gold_doc in zip(texts, preds, golds):
            pred_doc = pred_doc or {}
            gold_doc = gold_doc or {}

            pred_typed_spans = set()
            pred_spans = set()
            for event_type, mentions in pred_doc.items():
                for mention in mentions or []:
                    start = mention.get("start")
                    end = mention.get("end")
                    if start is None or end is None:
                        continue
                    span = (start, end)
                    pred_spans.add(span)
                    pred_typed_spans.add((event_type, start, end))

            gold_typed_spans = set()
            gold_spans = set()
            for event_type, mentions in gold_doc.items():
                for mention in mentions or []:
                    for start, end in self._find_exact_matches(text, mention):
                        gold_spans.add((start, end))
                        gold_typed_spans.add((event_type, start, end))

            span_tp += len(pred_spans & gold_spans)
            span_fp += len(pred_spans - gold_spans)
            span_fn += len(gold_spans - pred_spans)

            typed_tp += len(pred_typed_spans & gold_typed_spans)
            typed_fp += len(pred_typed_spans - gold_typed_spans)
            typed_fn += len(gold_typed_spans - pred_typed_spans)

        span_precision, span_recall, span_f1 = self._compute_prf(
            span_tp, span_fp, span_fn
        )
        precision, recall, f1 = self._compute_prf(typed_tp, typed_fp, typed_fn)

        return {
            "span_precision": span_precision,
            "span_recall": span_recall,
            "span_f1": span_f1,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def eval_arguments(self, texts, preds, golds):
        """
        Evaluate argument extraction performance. Argument extraction is evaluated as relation extraction,
        where the predicted and gold arguments are represented as relations between
        event mentions and argument mentions.

        """
        arg_i_tp = arg_i_fp = arg_i_fn = 0
        arg_c_tp = arg_c_fp = arg_c_fn = 0

        for text, pred_doc, gold_doc in zip(texts, preds, golds):
            pred_doc = pred_doc or {}
            gold_doc = gold_doc or {}

            pred_arg_i_relations = set()
            pred_arg_c_relations = set()

            for role, relation in self._iter_relations(pred_doc):
                if not isinstance(relation, dict):
                    continue

                event_span = self._extract_span(relation.get("event"))
                argument_span = self._extract_span(relation.get("argument"))

                if event_span and argument_span:
                    candidate_pairs = [(event_span, argument_span)]
                else:
                    head_span = self._extract_span(relation.get("head"))
                    tail_span = self._extract_span(relation.get("tail"))

                    if not head_span or not tail_span:
                        continue

                    # GLiNER2 relation outputs commonly use head/tail rather than
                    # explicit event/argument keys, so evaluate both orientations.
                    candidate_pairs = [(head_span, tail_span), (tail_span, head_span)]

                for event_span, argument_span in candidate_pairs:
                    event_start, event_end = event_span
                    arg_start, arg_end = argument_span
                    pred_arg_i_relations.add(
                        (event_start, event_end, arg_start, arg_end)
                    )
                    pred_arg_c_relations.add(
                        (role, event_start, event_end, arg_start, arg_end)
                    )

            gold_arg_i_relations = set()
            gold_arg_c_relations = set()

            for role, relation in self._iter_relations(gold_doc):
                if not isinstance(relation, dict):
                    continue

                event_text = relation.get("event")
                argument_text = relation.get("argument")

                if not event_text or not argument_text:
                    continue

                event_spans = self._find_exact_matches(text, event_text)
                argument_spans = self._find_exact_matches(text, argument_text)

                for event_start, event_end in event_spans:
                    for arg_start, arg_end in argument_spans:
                        gold_arg_i_relations.add(
                            (event_start, event_end, arg_start, arg_end)
                        )
                        gold_arg_c_relations.add(
                            (role, event_start, event_end, arg_start, arg_end)
                        )

            arg_i_tp += len(pred_arg_i_relations & gold_arg_i_relations)
            arg_i_fp += len(pred_arg_i_relations - gold_arg_i_relations)
            arg_i_fn += len(gold_arg_i_relations - pred_arg_i_relations)

            arg_c_tp += len(pred_arg_c_relations & gold_arg_c_relations)
            arg_c_fp += len(pred_arg_c_relations - gold_arg_c_relations)
            arg_c_fn += len(gold_arg_c_relations - pred_arg_c_relations)

        arg_i_precision, arg_i_recall, arg_i_f1 = self._compute_prf(
            arg_i_tp, arg_i_fp, arg_i_fn
        )
        arg_c_precision, arg_c_recall, arg_c_f1 = self._compute_prf(
            arg_c_tp, arg_c_fp, arg_c_fn
        )

        return {
            "arg_i_precision": arg_i_precision,
            "arg_i_recall": arg_i_recall,
            "arg_i_f1": arg_i_f1,
            "arg_c_precision": arg_c_precision,
            "arg_c_recall": arg_c_recall,
            "arg_c_f1": arg_c_f1,
        }
