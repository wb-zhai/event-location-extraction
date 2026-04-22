from src.inference.text_anchor import (
    AnchorConfig,
    AnchorStatus,
    TextAnchorResolver,
)


def test_resolve_exact_match():
    resolver = TextAnchorResolver()
    text = "Nintendo has pricing power in this market."
    quote = "pricing power"

    match = resolver.resolve(text, quote)

    assert match.status == AnchorStatus.MATCH_EXACT
    assert match.start is not None and match.end is not None
    assert text[match.start : match.end] == quote


def test_resolve_lesser_match_case_and_whitespace():
    resolver = TextAnchorResolver()
    text = "Nintendo has pricing power in this market."
    quote = "nintendo   has   pricing"

    match = resolver.resolve(text, quote)

    assert match.status == AnchorStatus.MATCH_LESSER
    assert match.start is not None and match.end is not None
    assert "Nintendo has pricing" == text[match.start : match.end]


def test_resolve_fuzzy_match():
    resolver = TextAnchorResolver(
        AnchorConfig(
            fuzzy_threshold=0.6,
            fuzzy_min_density=0.25,
            enable_fuzzy=True,
            accept_match_lesser=False,
            window_padding_tokens=2,
        )
    )
    text = "Nintendo can set the price unchallenged in their market segment."
    quote = "Nintendo can set prices without competition"

    match = resolver.resolve(text, quote)

    assert match.status == AnchorStatus.MATCH_FUZZY
    assert match.start is not None and match.end is not None


def test_resolve_not_found():
    resolver = TextAnchorResolver(
        AnchorConfig(enable_fuzzy=False, accept_match_lesser=False)
    )
    text = "Nintendo has pricing power in this market."
    quote = "Sony dominates"

    match = resolver.resolve(text, quote)

    assert match.status == AnchorStatus.NOT_FOUND
    assert match.start is None
    assert match.end is None
