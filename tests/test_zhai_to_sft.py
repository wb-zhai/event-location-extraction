from __future__ import annotations

from scripts.data.preprocess.sft.zhai_to_sft import convert_to_sft_records


def test_convert_to_sft_records_preserves_doc_id_in_window_metadata() -> None:
    records = convert_to_sft_records(
        {
            "id": "article-123",
            "source": {"text": "one two three four"},
            "events": [],
        },
        window_size=2,
        window_stride=2,
        include_offsets=False,
    )

    assert [record["metadata"]["doc_id"] for record in records] == [
        "article-123",
        "article-123",
    ]
