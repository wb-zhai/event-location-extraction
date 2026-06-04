"""Data contracts for generation_v2 event-location annotations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


LocationType = Literal["country", "state", "county", "city", "other"]
ArgumentRole = Literal["location", "source_location", "target_location"]


class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_offsets(self) -> "Span":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self

    def validate_against(self, article_text: str) -> None:
        if self.end > len(article_text):
            raise ValueError("span end exceeds article text length")
        if article_text[self.start : self.end] != self.text:
            raise ValueError("span text does not match article text offsets")


class LocationMention(Span):
    location_type: LocationType


class EventArgument(LocationMention):
    role: ArgumentRole


class EventMention(Span):
    event_type: str = Field(min_length=1)
    arguments: list[EventArgument] = Field(default_factory=list)


class NegativeMetadata(BaseModel):
    has_target_event: bool
    negative_reason: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    thoughts_tokens: int = 0


class PromptSettings(BaseModel):
    temperature: float | None = None
    reasoning_effort: str | int | None = None
    max_tokens: int | None = None
    save_thought_summaries: bool | None = None


class AnnotationMetadata(BaseModel):
    annotation_model: str
    pipeline_version: str = "generation_v2"
    source_bucket: str | None = None
    verifier_model: str | None = None
    api_mode: Literal["interactive", "batch"] | None = None
    run_id: str | None = None
    task_key: str | None = None
    window_indices: list[int] = Field(default_factory=list)
    prompt_settings: dict[str, PromptSettings] = Field(default_factory=dict)
    usage: dict[str, TokenUsage] = Field(default_factory=dict)
    batch_job_name: str | None = None
    raw_response_path: str | None = None
    verifier_decision: str | None = None
    thought_summaries: dict[str, list[str]] = Field(default_factory=dict)
    recovery_status: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AnnotationRecord(BaseModel):
    id: str
    text: str = Field(min_length=1)
    events: list[EventMention] = Field(default_factory=list)
    locations: list[LocationMention] = Field(default_factory=list)
    negatives: NegativeMetadata
    metadata: AnnotationMetadata

    @model_validator(mode="after")
    def validate_spans_against_text(self) -> "AnnotationRecord":
        for event in self.events:
            event.validate_against(self.text)
            for argument in event.arguments:
                argument.validate_against(self.text)
        for location in self.locations:
            location.validate_against(self.text)
        if self.events and not self.negatives.has_target_event:
            raise ValueError("records with events must set has_target_event=true")
        return self


def validate_record_against_ontology(
    record: AnnotationRecord, ontology: dict
) -> None:
    """Validate event types, roles, and location types against ontology.json."""
    event_types = set((ontology.get("events") or {}).keys())
    location_types = set((ontology.get("location_types") or {}).keys())
    role_by_event = ontology.get("event_argument_roles") or {}

    for event in record.events:
        if event.event_type not in event_types:
            raise ValueError(f"unknown event_type: {event.event_type}")
        allowed_roles = set(role_by_event.get(event.event_type, []))
        for argument in event.arguments:
            if argument.role not in allowed_roles:
                raise ValueError(
                    f"role {argument.role} is not allowed for {event.event_type}"
                )
            if argument.location_type not in location_types:
                raise ValueError(f"unknown location_type: {argument.location_type}")

    for location in record.locations:
        if location.location_type not in location_types:
            raise ValueError(f"unknown location_type: {location.location_type}")


def validate_no_overlapping_token_spans(record: AnnotationRecord) -> None:
    """Reject overlapping event/location spans for token classification."""
    spans: list[tuple[int, int, str]] = []
    spans.extend((event.start, event.end, "event") for event in record.events)
    spans.extend((location.start, location.end, "location") for location in record.locations)
    spans.sort()

    for previous, current in zip(spans, spans[1:]):
        previous_start, previous_end, previous_kind = previous
        current_start, current_end, current_kind = current
        if current_start < previous_end:
            raise ValueError(
                "overlapping token spans: "
                f"{previous_kind}({previous_start},{previous_end}) and "
                f"{current_kind}({current_start},{current_end})"
            )
