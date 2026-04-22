from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


TOKEN_PATTERN = re.compile(r"\S+")


class AnchorStatus:
    MATCH_EXACT = "match_exact"
    MATCH_LESSER = "match_lesser"
    MATCH_FUZZY = "match_fuzzy"
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
