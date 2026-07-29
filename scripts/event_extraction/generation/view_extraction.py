import argparse
import json
from pathlib import Path

import gradio as gr

COLUMNS = ["#", "id", "title", "label", "adm0", "risk_factors", "status", "decision", "events"]


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_title(r: dict) -> str:
    return str((r.get("source") or {}).get("title", ""))


def record_text(r: dict) -> str:
    return str((r.get("source") or {}).get("text", ""))


def get_events(r: dict) -> list[dict]:
    return (r.get("annotation") or {}).get("events") or []


def get_decision(r: dict) -> str:
    rel = r.get("relevance") or {}
    return rel.get("decision") or ("error" if "error" in rel else "")


def get_risk_factors(r: dict) -> list[str]:
    return r.get("risk_factors") or []


def format_events(events: list[dict]) -> str:
    if not events:
        return "\n\n_No events extracted._"
    parts = [f"\n\n### Events ({len(events)})"]
    for j, ev in enumerate(events, 1):
        parts.append(
            f"\n{j}. **{ev.get('event_type', 'n/a')}**\n"
            f"   - Grounding quote: _{ev.get('grounding_quote', '')}_\n"
            f"   - Event location: {ev.get('event_location', 'n/a')}\n"
            f"   - Event location text: {ev.get('event_location_text', 'n/a')}\n"
            f"   - Event time: {ev.get('event_time', 'n/a')}\n"
            f"   - Event time text: {ev.get('event_time_text', 'n/a')}\n"
            f"   - Time status: {ev.get('time_status', 'n/a')}\n"
            f"   - Severity: {ev.get('severity', 'n/a')}"
        )
    return "\n".join(parts)


def get_label_choices(records: list[dict]) -> list[str]:
    return sorted({str(r.get("label") or "") for r in records if r.get("label")})


def get_adm0_choices(records: list[dict]) -> list[str]:
    return sorted({str(r.get("adm0_code") or "") for r in records if r.get("adm0_code")})


def get_risk_factor_choices(records: list[dict]) -> list[str]:
    choices = set()
    for r in records:
        choices.update(get_risk_factors(r))
    return sorted(choices)


def get_status_choices(records: list[dict]) -> list[str]:
    return sorted({str(r.get("status") or "") for r in records if r.get("status")})


def get_decision_choices(records: list[dict]) -> list[str]:
    return sorted({get_decision(r) for r in records if get_decision(r)})


def record_row(i: int, r: dict) -> list:
    return [
        i,
        str(r.get("id") or ""),
        record_title(r)[:100],
        r.get("label") or "",
        r.get("adm0_code") or "",
        ", ".join(get_risk_factors(r))[:60],
        r.get("status") or "",
        get_decision(r),
        len(get_events(r)),
    ]


def build_app(records: list[dict]):
    def matching_indices(label_filter, adm0_filter, risk_filter, status_filter, decision_filter, min_events, search):
        idxs = []
        needle = search.strip().lower()
        for i, r in enumerate(records):
            if label_filter != "All" and (r.get("label") or "") != label_filter:
                continue
            if adm0_filter != "All" and (r.get("adm0_code") or "") != adm0_filter:
                continue
            if risk_filter != "All" and risk_filter not in get_risk_factors(r):
                continue
            if status_filter != "All" and (r.get("status") or "") != status_filter:
                continue
            if decision_filter != "All" and get_decision(r) != decision_filter:
                continue
            if len(get_events(r)) < min_events:
                continue
            if needle and needle not in record_title(r).lower() and needle not in record_text(r).lower():
                continue
            idxs.append(i)
        return idxs

    def refresh_table(label_filter, adm0_filter, risk_filter, status_filter, decision_filter, min_events, search):
        idxs = matching_indices(label_filter, adm0_filter, risk_filter, status_filter, decision_filter, min_events, search)
        rows = [record_row(i, records[i]) for i in idxs]
        count = f"**{len(idxs)} / {len(records)} records**"
        return rows, idxs, count, "", ""

    def show_detail(evt: gr.SelectData, idxs):
        row = evt.index[0]
        if row >= len(idxs):
            return "", ""
        r = records[idxs[row]]
        rel = r.get("relevance") or {}
        gen_llm = r.get("generation_llm") or {}
        meta = (
            f"### {record_title(r)}\n\n"
            f"**ID:** {r.get('id', 'n/a')}  \n"
            f"**Label:** {r.get('label', 'n/a')}  \n"
            f"**Adm0:** {r.get('adm0_code', 'n/a')}  \n"
            f"**Risk factors:** {', '.join(get_risk_factors(r)) or 'n/a'}  \n"
            f"**Quality score:** {r.get('quality_score', 'n/a')}  \n"
            f"**Status:** {r.get('status', 'n/a')}  \n"
            f"**Relevance decision:** {rel.get('decision') or rel.get('error', 'n/a')} "
            f"(confidence: {rel.get('confidence', 'n/a')})  \n"
            f"**Relevance reason:** {rel.get('reason') or rel.get('error') or ''}  \n"
            f"**Generation model:** {gen_llm.get('model', 'n/a')}"
            f"{format_events(get_events(r))}"
        )
        return meta, record_text(r)

    with gr.Blocks(title="Extraction annotation viewer") as demo:
        gr.Markdown("## Event extraction annotation viewer")
        with gr.Row():
            label_filter = gr.Dropdown(["All"] + get_label_choices(records), value="All", label="Label")
            adm0_filter = gr.Dropdown(["All"] + get_adm0_choices(records), value="All", label="Adm0")
            risk_filter = gr.Dropdown(["All"] + get_risk_factor_choices(records), value="All", label="Risk factor")
            status_filter = gr.Dropdown(["All"] + get_status_choices(records), value="All", label="Status")
            decision_filter = gr.Dropdown(["All"] + get_decision_choices(records), value="All", label="Relevance decision")
        with gr.Row():
            min_events = gr.Slider(0, 10, value=0, step=1, label="Min # events")
            search = gr.Textbox(label="Search title/text")

        count_label = gr.Markdown()
        idxs_state = gr.State(list(range(len(records))))
        table = gr.Dataframe(
            headers=COLUMNS,
            datatype=["number", "str", "str", "str", "str", "str", "str", "str", "number"],
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
        )

        gr.Markdown("### Selected record")
        detail_meta = gr.Markdown()
        detail_text = gr.Textbox(label="Full text", lines=20, max_lines=40)

        inputs = [label_filter, adm0_filter, risk_filter, status_filter, decision_filter, min_events, search]
        outputs = [table, idxs_state, count_label, detail_meta, detail_text]

        for comp in inputs:
            comp.change(refresh_table, inputs=inputs, outputs=outputs)

        demo.load(refresh_table, inputs=inputs, outputs=outputs)
        table.select(show_detail, inputs=[idxs_state], outputs=[detail_meta, detail_text])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Browse event-extraction annotated JSONL output.")
    parser.add_argument("--input", type=str, required=True, help="Path to an annotated extraction JSONL file")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    demo = build_app(records)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
