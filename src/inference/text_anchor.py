from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

TOKEN_PATTERN = re.compile(r"\S+")


class AnchorStatus:
    MATCH_EXACT = "match_exact"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"
    MATCH_CONTEXT = "match_context"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class AnchorConfig:
    fuzzy_threshold: float = 0.75
    fuzzy_min_density: float = 1.0 / 3.0
    enable_fuzzy: bool = True
    accept_match_lesser: bool = True
    window_padding_tokens: int = 4


@dataclass(frozen=True)
class AnchorMatch:
    start: int | None
    end: int | None
    score: float
    status: str
    matched_text: str | None = None


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int


class TextAnchorResolver:
    """Resolve a quote into a character span in a source document.

    Matching policy mirrors LangExtract-style anchoring at a high level:
    1) exact string match
    2) optional tolerant "lesser" exact match for whitespace/case variation
    3) fuzzy token alignment fallback
    """

    def __init__(self, config: AnchorConfig | None = None):
        self.config = config or AnchorConfig()

    def resolve(self, source_text: str, quote: str) -> AnchorMatch:
        if not source_text or not quote:
            return AnchorMatch(
                start=None,
                end=None,
                score=0.0,
                status=AnchorStatus.NOT_FOUND,
                matched_text=None,
            )

        exact_start = source_text.find(quote)
        if exact_start != -1:
            exact_end = exact_start + len(quote)
            return AnchorMatch(
                start=exact_start,
                end=exact_end,
                score=1.0,
                status=AnchorStatus.MATCH_EXACT,
                matched_text=source_text[exact_start:exact_end],
            )

        if self.config.accept_match_lesser:
            lesser = self._find_lesser_match(source_text, quote)
            if lesser is not None:
                start, end = lesser
                return AnchorMatch(
                    start=start,
                    end=end,
                    score=1.0,
                    status=AnchorStatus.MATCH_LESSER,
                    matched_text=source_text[start:end],
                )

        if self.config.enable_fuzzy:
            fuzzy = self._find_fuzzy_match(source_text, quote)
            if fuzzy is not None:
                start, end, score = fuzzy
                return AnchorMatch(
                    start=start,
                    end=end,
                    score=score,
                    status=AnchorStatus.MATCH_FUZZY,
                    matched_text=source_text[start:end],
                )

        return AnchorMatch(
            start=None,
            end=None,
            score=0.0,
            status=AnchorStatus.NOT_FOUND,
            matched_text=None,
        )

    def resolve_many(self, source_text: str, quotes: list[str]) -> list[AnchorMatch]:
        return [self.resolve(source_text, quote) for quote in quotes]

    def resolve_with_context(
        self,
        source_text: str,
        quote: str | None = None,
        *,
        left_context: str | None = None,
        right_context: str | None = None,
        start_hint: int | None = None,
        end_hint: int | None = None,
    ) -> AnchorMatch:
        if not source_text:
            return AnchorMatch(
                start=None,
                end=None,
                score=0.0,
                status=AnchorStatus.NOT_FOUND,
                matched_text=None,
            )

        # Fast path exact matches
        if quote is not None:
            # 1. Try exact match at hints
            if start_hint is not None and end_hint is not None:
                # Allow a small window around the hint in case of minor offsets
                window_start = max(0, start_hint - 20)
                window_end = min(len(source_text), end_hint + 20)
                idx = source_text.find(quote, window_start, window_end)
                if idx != -1:
                    return AnchorMatch(
                        start=idx,
                        end=idx + len(quote),
                        score=1.0,
                        status=AnchorStatus.MATCH_EXACT,
                        matched_text=quote,
                    )

            # 2. Try exact match using context built together
            if left_context or right_context:
                l_ctx = left_context or ""
                r_ctx = right_context or ""
                combined = l_ctx + quote + r_ctx
                idx = source_text.find(combined)
                if idx != -1:
                    start_pos = idx + len(l_ctx)
                    end_pos = start_pos + len(quote)
                    return AnchorMatch(
                        start=start_pos,
                        end=end_pos,
                        score=1.0,
                        status=AnchorStatus.MATCH_EXACT,
                        matched_text=quote,
                    )
        else:
            # quote is None
            if left_context and right_context:
                # Try to find exactly left_context + right_context
                combined = left_context + right_context
                idx = source_text.find(combined)
                if idx != -1:
                    start_pos = idx + len(left_context)
                    return AnchorMatch(
                        start=start_pos,
                        end=start_pos,
                        score=1.0,
                        status=AnchorStatus.MATCH_EXACT,
                        matched_text="",
                    )

                # Try finding them independently
                l_idx = source_text.find(left_context)
                if l_idx != -1:
                    r_idx = source_text.find(right_context, l_idx + len(left_context))
                    if r_idx != -1:
                        st = l_idx + len(left_context)
                        en = r_idx
                        return AnchorMatch(
                            start=st,
                            end=en,
                            score=1.0,
                            status=AnchorStatus.MATCH_EXACT,
                            matched_text=source_text[st:en],
                        )

        # Fallback to bounded context resolution
        normalized_quote = quote if isinstance(quote, str) and quote.strip() else None
        normalized_left = (
            left_context if isinstance(left_context, str) and left_context else None
        )
        normalized_right = (
            right_context if isinstance(right_context, str) and right_context else None
        )

        if normalized_left is None and normalized_right is None:
            if normalized_quote is None:
                return AnchorMatch(
                    start=None,
                    end=None,
                    score=0.0,
                    status=AnchorStatus.NOT_FOUND,
                    matched_text=None,
                )
            return self.resolve(source_text, normalized_quote)

        bounded = self._resolve_with_bounded_context(
            source_text,
            normalized_quote,
            left_context=normalized_left,
            right_context=normalized_right,
            start_hint=start_hint,
            end_hint=end_hint,
        )
        if bounded is not None:
            return bounded

        if normalized_quote is not None:
            return self.resolve(source_text, normalized_quote)

        return AnchorMatch(
            start=None,
            end=None,
            score=0.0,
            status=AnchorStatus.NOT_FOUND,
            matched_text=None,
        )

    def _resolve_with_bounded_context(
        self,
        source_text: str,
        quote: str | None,
        *,
        left_context: str | None,
        right_context: str | None,
        start_hint: int | None,
        end_hint: int | None,
    ) -> AnchorMatch | None:
        bounds = self._candidate_bounds(
            source_text,
            left_context=left_context,
            right_context=right_context,
        )
        if not bounds:
            return None

        best_match: AnchorMatch | None = None
        best_rank: tuple[float, float, float] | None = None

        for bound_start, bound_end, context_score in bounds:
            if quote is not None:
                bounded_match = self._resolve_in_window(
                    source_text,
                    quote,
                    bound_start,
                    bound_end,
                )
                if bounded_match is None:
                    continue
                match = AnchorMatch(
                    start=bounded_match.start,
                    end=bounded_match.end,
                    score=max(bounded_match.score, context_score),
                    status=bounded_match.status,
                    matched_text=bounded_match.matched_text,
                )
            else:
                if bound_end <= bound_start:
                    continue
                match = AnchorMatch(
                    start=bound_start,
                    end=bound_end,
                    score=context_score,
                    status=AnchorStatus.MATCH_CONTEXT,
                    matched_text=source_text[bound_start:bound_end],
                )

            hint_score = self._hint_score(
                match.start,
                match.end,
                start_hint=start_hint,
                end_hint=end_hint,
            )
            span_length = (match.end or 0) - (match.start or 0)
            rank = (hint_score, context_score, -float(span_length))
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_match = match

        return best_match

    def _candidate_bounds(
        self,
        source_text: str,
        *,
        left_context: str | None,
        right_context: str | None,
    ) -> list[tuple[int, int, float]]:
        left_matches = (
            [
                (m.start(), m.end())
                for m in re.finditer(re.escape(left_context), source_text)
            ]
            if left_context
            else [(0, 0)]
        )
        right_matches = (
            [
                (m.start(), m.end())
                for m in re.finditer(re.escape(right_context), source_text)
            ]
            if right_context
            else [(len(source_text), len(source_text))]
        )

        candidates: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for left_start, left_end in left_matches:
            for right_start, right_end in right_matches:
                if left_context and right_context and left_end > right_start:
                    continue
                start = left_end if left_context else 0
                end = right_start if right_context else len(source_text)
                if end < start:
                    continue
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                context_score = 1.0 if left_context and right_context else 0.8
                candidates.append((start, end, context_score))

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates

    def _resolve_in_window(
        self,
        source_text: str,
        quote: str,
        start: int,
        end: int,
    ) -> AnchorMatch | None:
        if end <= start:
            return None
        window = source_text[start:end]
        match = self.resolve(window, quote)
        if match.start is None or match.end is None or match.matched_text is None:
            return None
        return AnchorMatch(
            start=start + match.start,
            end=start + match.end,
            score=match.score,
            status=match.status,
            matched_text=match.matched_text,
        )

    @staticmethod
    def _hint_score(
        start: int | None,
        end: int | None,
        *,
        start_hint: int | None,
        end_hint: int | None,
    ) -> float:
        if start is None or end is None:
            return 0.0
        if start_hint is None or end_hint is None:
            return 0.0
        hint_length = max(1, end_hint - start_hint)
        start_delta = abs(start - start_hint)
        end_delta = abs(end - end_hint)
        return -((start_delta + end_delta) / hint_length)

    def _find_lesser_match(
        self, source_text: str, quote: str
    ) -> tuple[int, int] | None:
        quote_tokens = quote.split()
        if not quote_tokens:
            return None

        pattern = (
            r"\b" + r"\s+".join(re.escape(token) for token in quote_tokens) + r"\b"
        )
        match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if match is None:
            return None
        return (match.start(), match.end())

    def _find_fuzzy_match(
        self,
        source_text: str,
        quote: str,
    ) -> tuple[int, int, float] | None:
        source_tokens = self._tokenize(source_text)
        quote_tokens = self._tokenize(quote)
        if not source_tokens or not quote_tokens:
            return None

        source_norm = [self._normalize_token(token.text) for token in source_tokens]
        quote_norm = [self._normalize_token(token.text) for token in quote_tokens]

        q_len = len(quote_norm)
        min_window = max(1, q_len - self.config.window_padding_tokens)
        max_window = min(len(source_norm), q_len + self.config.window_padding_tokens)

        best_score = -1.0
        best_span: tuple[int, int] | None = None

        matcher = SequenceMatcher(autojunk=False)
        for window_size in range(min_window, max_window + 1):
            for start_idx in range(0, len(source_norm) - window_size + 1):
                candidate = source_norm[start_idx : start_idx + window_size]
                matcher.set_seqs(candidate, quote_norm)
                matching_blocks = matcher.get_matching_blocks()
                matches = sum(block.size for block in matching_blocks)
                if q_len == 0:
                    continue
                coverage = matches / q_len
                density = matches / window_size
                if coverage < self.config.fuzzy_threshold:
                    continue
                if density < self.config.fuzzy_min_density:
                    continue

                score = matcher.ratio()
                if score <= best_score:
                    continue
                best_score = score
                best_span = (start_idx, start_idx + window_size - 1)

        if best_span is None:
            return None

        start_token = source_tokens[best_span[0]]
        end_token = source_tokens[best_span[1]]
        return (start_token.start, end_token.end, best_score)

    @staticmethod
    def _tokenize(text: str) -> list[_Token]:
        tokens: list[_Token] = []
        for match in TOKEN_PATTERN.finditer(text):
            start, end = match.span()
            tokens.append(_Token(text=match.group(0), start=start, end=end))
        return tokens

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = token.lower()
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        return token
