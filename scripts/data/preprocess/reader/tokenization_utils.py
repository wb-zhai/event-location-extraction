from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast


@dataclass(frozen=True)
class TokenizedDocument:
    text: str
    input_ids: list[int]
    tokenizer_tokens: list[str]
    offsets: list[tuple[int, int]]
    word_char_spans: list[tuple[int, int]] | None = None


def build_fast_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            f"Tokenizer '{tokenizer_name}' must be a fast tokenizer for offset mapping"
        )
    # Suppress length warnings since we window the tokens later
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def tokenize_text(tokenizer: PreTrainedTokenizerFast, text: str) -> TokenizedDocument:
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = encoding["input_ids"]
    offsets = [tuple(offset) for offset in encoding["offset_mapping"]]
    if not input_ids:
        raise ValueError("Tokenizer produced no document pieces")
    return TokenizedDocument(
        text=text,
        input_ids=list(input_ids),
        tokenizer_tokens=tokenizer.convert_ids_to_tokens(input_ids),
        offsets=offsets,
    )


def tokenize_words(
    tokenizer: PreTrainedTokenizerFast,
    words: list[str],
) -> TokenizedDocument:
    text_parts: list[str] = []
    word_char_spans: list[tuple[int, int]] = []
    position = 0
    for index, word in enumerate(words):
        if index > 0:
            text_parts.append(" ")
            position += 1
        start = position
        text_parts.append(word)
        position += len(word)
        word_char_spans.append((start, position))
    text = "".join(text_parts)
    tokenized = tokenize_text(tokenizer, text)
    return TokenizedDocument(
        text=tokenized.text,
        input_ids=tokenized.input_ids,
        tokenizer_tokens=tokenized.tokenizer_tokens,
        offsets=tokenized.offsets,
        word_char_spans=word_char_spans,
    )


def char_span_to_piece_span(
    start: Any,
    end: Any,
    offsets: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None

    token_start: int | None = None
    token_end: int | None = None
    for token_index, (token_char_start, token_char_end) in enumerate(offsets):
        if token_char_start == token_char_end:
            continue
        if token_char_end <= start:
            continue
        if token_char_start >= end:
            break
        if token_start is None:
            token_start = token_index
        token_end = token_index

    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def word_span_to_piece_span(
    start_word: Any,
    end_word: Any,
    tokenized: TokenizedDocument,
) -> tuple[int, int] | None:
    if tokenized.word_char_spans is None:
        raise ValueError("word_char_spans are required for word-span conversion")
    if (
        not isinstance(start_word, int)
        or not isinstance(end_word, int)
        or start_word < 0
        or end_word < start_word
        or end_word >= len(tokenized.word_char_spans)
    ):
        return None
    start_char = tokenized.word_char_spans[start_word][0]
    end_char = tokenized.word_char_spans[end_word][1]
    return char_span_to_piece_span(start_char, end_char, tokenized.offsets)
