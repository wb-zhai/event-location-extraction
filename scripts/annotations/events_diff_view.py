"""Gradio UI to browse an exported event-annotation JSONL file (the output of
`events_argilla.py export`) and compare each record's human-corrected events
against the model's original suggestion, field by field.

Usage:
    python -m scripts.annotations.events_diff_view --input dataset/manual/event-extraction/matrix_5M.sample_1000.3.1pro.2label_prompt.extracted.bona.v4.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

import gradio as gr

EVENT_FIELDS = [
    "event_type",
    "grounding_quote",
    "event_location_text",
    "event_location",
    "event_location_admin_level",
    "event_time_text",
    "event_time",
    "time_status",
    "severity",
]
DIFF_FIELDS = [f for f in EVENT_FIELDS if f != "grounding_quote"]

TABLE_COLUMNS = ["#", "id", "title", "annotated_by", "completed", "changed", "human events", "original events"]


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_id(r: dict[str, Any]) -> str:
    return str(r.get("id") or "")


def record_title(r: dict[str, Any]) -> str:
    return str(r.get("title") or "")


def record_text(r: dict[str, Any]) -> str:
    return str(r.get("text") or "")


def human_events(r: dict[str, Any]) -> list[dict[str, Any]]:
    return (r.get("annotation") or {}).get("events") or []


def original_events(r: dict[str, Any]) -> list[dict[str, Any]]:
    return ((r.get("annotation") or {}).get("original_annotation") or {}).get("events") or []


def annotated_by(r: dict[str, Any]) -> str | None:
    return (r.get("annotation") or {}).get("annotated_by")


def canonicalize(events: list[dict[str, Any]]) -> list[tuple]:
    return sorted(tuple(sorted(ev.items())) for ev in events if isinstance(ev, dict))


def is_changed(r: dict[str, Any]) -> bool:
    return canonicalize(human_events(r)) != canonicalize(original_events(r))


def index_by_quote(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {ev.get("grounding_quote", ""): ev for ev in events if isinstance(ev, dict)}


def event_status(ev: dict[str, Any], other_by_quote: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Status of `ev` relative to the other side's events, keyed by grounding_quote."""
    other = other_by_quote.get(ev.get("grounding_quote", ""))
    if other is None:
        return "unmatched", []
    diffs = [f for f in DIFF_FIELDS if ev.get(f) != other.get(f)]
    return ("modified" if diffs else "unchanged"), diffs


def format_events_side(events: list[dict[str, Any]], other_by_quote: dict[str, dict[str, Any]], unmatched_label: str) -> str:
    if not events:
        return "_(no events)_"
    parts = []
    for i, ev in enumerate(events, 1):
        status, diffs = event_status(ev, other_by_quote)
        marker = {"unchanged": "", "modified": "✏️ ", "unmatched": f"🔶 {unmatched_label} "}[status]
        parts.append(f"{i}. {marker}**{ev.get('event_type', 'n/a')}** — _{ev.get('grounding_quote', '')}_")
        for field in ["event_location", "event_location_admin_level", "event_time", "time_status", "severity"]:
            value = ev.get(field, "n/a")
            if field in diffs:
                parts.append(f"   - **{field}: {value}**")
            else:
                parts.append(f"   - {field}: {value}")
    return "\n".join(parts)


def diff_counts(r: dict[str, Any]) -> tuple[int, int, int]:
    """(unchanged, modified, added-or-removed) counts for a record."""
    orig_by_quote = index_by_quote(original_events(r))
    human_by_quote = index_by_quote(human_events(r))
    unchanged = modified = touched = 0
    for q, oev in orig_by_quote.items():
        hev = human_by_quote.get(q)
        if hev is None:
            touched += 1
        elif any(oev.get(f) != hev.get(f) for f in DIFF_FIELDS):
            modified += 1
        else:
            unchanged += 1
    added = sum(1 for q in human_by_quote if q not in orig_by_quote)
    return unchanged, modified, touched + added


def build_app(records: list[dict[str, Any]]):
    users = sorted({u for r in records if (u := annotated_by(r))})

    def matching_indices(user_filter, changed_filter, completed_filter, search):
        idxs = []
        needle = search.strip().lower()
        for i, r in enumerate(records):
            user = annotated_by(r)
            if user_filter != "All" and user != user_filter:
                continue
            completed = user is not None
            if completed_filter == "Completed only" and not completed:
                continue
            if completed_filter == "Not completed only" and completed:
                continue
            changed = completed and is_changed(r)
            if changed_filter == "Changed only" and not changed:
                continue
            if changed_filter == "Unchanged only" and (not completed or changed):
                continue
            if needle and needle not in record_title(r).lower() and needle not in record_text(r).lower():
                continue
            idxs.append(i)
        return idxs

    def row_for(i: int) -> list:
        r = records[i]
        user = annotated_by(r)
        completed = user is not None
        return [
            i,
            record_id(r),
            record_title(r)[:120],
            user or "",
            "yes" if completed else "no",
            "yes" if (completed and is_changed(r)) else ("" if not completed else "no"),
            len(human_events(r)),
            len(original_events(r)),
        ]

    def refresh_table(user_filter, changed_filter, completed_filter, search):
        idxs = matching_indices(user_filter, changed_filter, completed_filter, search)
        rows = [row_for(i) for i in idxs]
        count = f"**{len(idxs)} / {len(records)} records**"
        return rows, idxs, count, "", "", "", ""

    def show_detail(evt: gr.SelectData, idxs):
        row = evt.index[0]
        if row >= len(idxs):
            return "", "", "", ""
        r = records[idxs[row]]
        orig = original_events(r)
        human = human_events(r)
        orig_by_quote = index_by_quote(orig)
        human_by_quote = index_by_quote(human)

        unchanged, modified, touched = diff_counts(r)
        title_md = (
            f"### {record_title(r)}\n\n"
            f"**ID:** {record_id(r)}  \n"
            f"**Annotated by:** {annotated_by(r) or '_(not completed)_'}  \n"
            f"**Diff:** {unchanged} unchanged · {modified} modified · {touched} added/removed"
        )
        original_md = f"#### Original (model)\n\n{format_events_side(orig, human_by_quote, 'removed by human')}"
        human_md = f"#### Human corrected\n\n{format_events_side(human, orig_by_quote, 'added by human')}"
        return title_md, original_md, human_md, record_text(r)

    with gr.Blocks(title="Event annotation diff viewer") as demo:
        gr.Markdown("## Compare human-corrected events against the model's original annotation")
        with gr.Row():
            user_filter = gr.Dropdown(["All", *users], value="All", label="Annotated by")
            changed_filter = gr.Radio(
                ["All", "Changed only", "Unchanged only"], value="All", label="Correction status"
            )
            completed_filter = gr.Radio(
                ["All", "Completed only", "Not completed only"], value="Completed only", label="Completion"
            )
            search = gr.Textbox(label="Search title/text")

        count_label = gr.Markdown()
        idxs_state = gr.State(list(range(len(records))))
        table = gr.Dataframe(
            headers=TABLE_COLUMNS,
            datatype=["number", "str", "str", "str", "str", "str", "number", "number"],
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
        )

        gr.Markdown("### Selected record")
        detail_title = gr.Markdown()
        with gr.Row():
            detail_original = gr.Markdown()
            detail_human = gr.Markdown()
        detail_text = gr.Textbox(label="Full text", lines=15, max_lines=30)

        filter_inputs = [user_filter, changed_filter, completed_filter, search]
        table_outputs = [table, idxs_state, count_label, detail_title, detail_original, detail_human, detail_text]
        detail_outputs = [detail_title, detail_original, detail_human, detail_text]

        for comp in filter_inputs:
            comp.change(refresh_table, inputs=filter_inputs, outputs=table_outputs)

        demo.load(refresh_table, inputs=filter_inputs, outputs=table_outputs)
        table.select(show_detail, inputs=[idxs_state], outputs=detail_outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=str, help="Exported event-annotation JSONL file")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    demo = build_app(records)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
