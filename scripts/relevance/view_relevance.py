import argparse
import json
from pathlib import Path

import gradio as gr

COLUMNS = ["#", "title", "decision", "confidence", "filtered", "reason"]


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_key(r: dict) -> str:
    return str(r.get("id") or r.get("url") or "")


def record_title(r: dict) -> str:
    return str(r.get("title") or (r.get("source") or {}).get("title", ""))


def record_text(r: dict) -> str:
    return str(r.get("text") or (r.get("source") or {}).get("text", ""))


def get_decision(r: dict) -> str:
    rel = r.get("relevance") or {}
    return rel.get("decision") or ("error" if "error" in rel else "")


def format_annotation(r: dict) -> str:
    ann = r.get("annotation")
    if not ann:
        return ""
    parts = ["\n\n---\n\n### Annotation"]
    events = ann.get("events") or []
    if events:
        parts.append(f"\n\n**Events ({len(events)}):**")
        for j, ev in enumerate(events, 1):
            parts.append(
                f"\n{j}. **{ev.get('event_type', 'n/a')}** — _{ev.get('grounding_quote', '')}_\n"
                f"   - Location: {ev.get('event_location', 'n/a')} ({ev.get('event_location_text', 'n/a')})\n"
                f"   - Time: {ev.get('event_time', 'n/a')} ({ev.get('event_time_text', 'n/a')}), status: {ev.get('time_status', 'n/a')}\n"
                f"   - Severity: {ev.get('severity', 'n/a')}"
            )
    return "\n".join(parts)


def get_decision_labels(records: list[dict]) -> list[str]:
    return sorted({get_decision(r) for r in records if get_decision(r)})


def get_model_name(records: list[dict]) -> str | None:
    for r in records:
        model = (r.get("relevance") or {}).get("model") or (r.get("llm") or {}).get("model")
        if model:
            return model
    return None


def record_row(i: int, r: dict) -> list:
    rel = r.get("relevance") or {}
    decision = get_decision(r)
    confidence = rel.get("confidence")
    return [
        i,
        record_title(r)[:120],
        decision,
        f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "",
        "yes" if rel.get("filtered", False) else "no",
        (rel.get("reason") or rel.get("error") or "")[:150],
    ]


def build_app(records: list[dict]):
    def matching_indices(decision_filter, filtered_filter, min_conf, search):
        idxs = []
        needle = search.strip().lower()
        for i, r in enumerate(records):
            rel = r.get("relevance") or {}
            decision = get_decision(r)
            if decision_filter != "All" and decision != decision_filter.lower():
                continue
            if filtered_filter == "Filtered only" and not rel.get("filtered", False):
                continue
            if filtered_filter == "Kept only" and rel.get("filtered", False):
                continue
            if (rel.get("confidence") or 0.0) < min_conf:
                continue
            if needle and needle not in record_title(r).lower() and needle not in record_text(r).lower():
                continue
            idxs.append(i)
        return idxs

    def refresh_table(decision_filter, filtered_filter, min_conf, search):
        idxs = matching_indices(decision_filter, filtered_filter, min_conf, search)
        rows = [record_row(i, records[i]) for i in idxs]
        count = f"**{len(idxs)} / {len(records)} records**"
        return rows, idxs, count, "", ""

    def show_detail(evt: gr.SelectData, idxs):
        row = evt.index[0]
        if row >= len(idxs):
            return "", ""
        r = records[idxs[row]]
        rel = r.get("relevance") or {}
        meta = (
            f"### {record_title(r)}\n\n"
            f"**Decision:** {rel.get('decision') or rel.get('error', 'n/a')}  \n"
            f"**Confidence:** {rel.get('confidence', 'n/a')}  \n"
            f"**Filtered:** {rel.get('filtered', False)}  \n"
            f"**Model:** {rel.get('model', 'n/a')}  \n"
            f"**Reason:** {rel.get('reason') or rel.get('error') or ''}"
            f"{format_annotation(r)}"
        )
        return meta, record_text(r)

    decision_choices = ["All"] + [label.capitalize() for label in get_decision_labels(records)]

    with gr.Blocks(title="Relevance filter viewer") as demo:
        gr.Markdown("## Relevance filter output viewer")
        with gr.Row():
            decision_filter = gr.Radio(
                decision_choices, value="All", label="Decision"
            )
            filtered_filter = gr.Radio(
                ["All", "Filtered only", "Kept only"], value="All", label="Filtered"
            )
            min_conf = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Min confidence")
            search = gr.Textbox(label="Search title/text")

        count_label = gr.Markdown()
        idxs_state = gr.State(list(range(len(records))))
        table = gr.Dataframe(
            headers=COLUMNS,
            datatype=["number", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
        )

        gr.Markdown("### Selected record")
        detail_meta = gr.Markdown()
        detail_text = gr.Textbox(label="Full text", lines=20, max_lines=40)

        inputs = [decision_filter, filtered_filter, min_conf, search]
        outputs = [table, idxs_state, count_label, detail_meta, detail_text]

        for comp in inputs:
            comp.change(refresh_table, inputs=inputs, outputs=outputs)

        demo.load(refresh_table, inputs=inputs, outputs=outputs)
        table.select(show_detail, inputs=[idxs_state], outputs=[detail_meta, detail_text])

    return demo


COMPARE_COLUMNS_BASE = ["#", "title", "agreement", "status"]


def build_compare_app(
    path_a: Path,
    path_b: Path,
    output_path: Path,
    a_name: str | None,
    b_name: str | None,
):
    records_a_list = load_records(path_a)
    records_b_list = load_records(path_b)
    records_a = {record_key(r): r for r in records_a_list}
    records_b = {record_key(r): r for r in records_b_list}

    a_name = a_name or get_model_name(records_a_list) or path_a.stem
    b_name = b_name or get_model_name(records_b_list) or path_b.stem

    common_ids = [k for k in records_a if k in records_b]
    only_a = [k for k in records_a if k not in records_b]
    only_b = [k for k in records_b if k not in records_a]
    if only_a or only_b:
        print(
            f"Warning: {len(only_a)} ids only in {path_a.name}, "
            f"{len(only_b)} ids only in {path_b.name}; comparing the {len(common_ids)} shared ids."
        )

    adjudications: dict[str, dict] = {}
    if output_path.exists():
        for r in load_records(output_path):
            adjudications[record_key(r)] = r
        print(f"Resuming: {len(adjudications)} adjudications already saved in {output_path}.")

    def write_output():
        with output_path.open("w", encoding="utf-8") as f:
            for k in common_ids:
                if k in adjudications:
                    f.write(json.dumps(adjudications[k], ensure_ascii=False) + "\n")

    def matching_indices(agreement_filter, status_filter, search):
        idxs = []
        needle = search.strip().lower()
        for i, k in enumerate(common_ids):
            ra, rb = records_a[k], records_b[k]
            agree = get_decision(ra) == get_decision(rb)
            if agreement_filter == "Agree only" and not agree:
                continue
            if agreement_filter == "Disagree only" and agree:
                continue
            done = k in adjudications
            if status_filter == "Pending only" and done:
                continue
            if status_filter == "Done only" and not done:
                continue
            if needle and needle not in record_title(ra).lower() and needle not in record_text(ra).lower():
                continue
            idxs.append(i)
        return idxs

    def row_for(i: int) -> list:
        k = common_ids[i]
        ra, rb = records_a[k], records_b[k]
        agree = get_decision(ra) == get_decision(rb)
        return [
            i,
            record_title(ra)[:120],
            "agree" if agree else "disagree",
            "done" if k in adjudications else "pending",
        ]

    def refresh_table(agreement_filter, status_filter, search):
        idxs = matching_indices(agreement_filter, status_filter, search)
        rows = [row_for(i) for i in idxs]
        count = f"**{len(idxs)} / {len(common_ids)} records**"
        return rows, idxs, count

    def load_id(k: str):
        ra, rb = records_a[k], records_b[k]
        rel_a, rel_b = ra.get("relevance") or {}, rb.get("relevance") or {}
        dec_a, dec_b = get_decision(ra), get_decision(rb)
        choice_a = f"A ({a_name}): {dec_a}"
        choice_b = f"B ({b_name}): {dec_b}"
        choices = [choice_a, choice_b, "Custom: relevant", "Custom: irrelevant"]

        existing = adjudications.get(k)
        if existing and existing.get("source") == "a":
            value = choice_a
        elif existing and existing.get("source") == "b":
            value = choice_b
        elif existing:
            value = f"Custom: {existing.get('decision', '')}"
        else:
            value = None

        meta_a = (
            f"#### A: {a_name}\n\n"
            f"**Decision:** {dec_a}  \n"
            f"**Confidence:** {rel_a.get('confidence', 'n/a')}  \n"
            f"**Reason:** {rel_a.get('reason') or rel_a.get('error') or ''}"
            f"{format_annotation(ra)}"
        )
        meta_b = (
            f"#### B: {b_name}\n\n"
            f"**Decision:** {dec_b}  \n"
            f"**Confidence:** {rel_b.get('confidence', 'n/a')}  \n"
            f"**Reason:** {rel_b.get('reason') or rel_b.get('error') or ''}"
            f"{format_annotation(rb)}"
        )
        title_md = f"### {record_title(ra)}"
        text = record_text(ra) or record_text(rb)
        note = existing.get("note", "") if existing else ""
        return (
            k,
            title_md,
            meta_a,
            meta_b,
            text,
            gr.update(choices=choices, value=value),
            note,
            "",
        )

    def show_detail(evt: gr.SelectData, idxs):
        row = evt.index[0]
        if row >= len(idxs):
            return (None, "", "", "", "", gr.update(choices=[], value=None), "", "")
        return load_id(common_ids[idxs[row]])

    def next_pending(current_id):
        start = common_ids.index(current_id) + 1 if current_id in common_ids else 0
        for offset in range(len(common_ids)):
            k = common_ids[(start + offset) % len(common_ids)]
            if k not in adjudications:
                return load_id(k)
        return load_id(current_id) if current_id else (None, "", "", "", "", gr.update(choices=[], value=None), "", "No pending records left.")

    def save(current_id, choice, note, agreement_filter, status_filter, search):
        if not current_id:
            rows, idxs, count = refresh_table(agreement_filter, status_filter, search)
            return rows, idxs, count, "No record selected."
        ra, rb = records_a[current_id], records_b[current_id]
        rel_a, rel_b = ra.get("relevance") or {}, rb.get("relevance") or {}
        if choice.startswith("A ("):
            source, decision = "a", get_decision(ra)
        elif choice.startswith("B ("):
            source, decision = "b", get_decision(rb)
        elif choice == "Custom: relevant":
            source, decision = "custom", "relevant"
        elif choice == "Custom: irrelevant":
            source, decision = "custom", "irrelevant"
        else:
            rows, idxs, count = refresh_table(agreement_filter, status_filter, search)
            return rows, idxs, count, "Pick a decision before saving."

        adjudications[current_id] = {
            "id": current_id,
            "title": record_title(ra),
            "decision": decision,
            "source": source,
            "note": note or "",
            "candidates": {
                a_name: {
                    "decision": get_decision(ra),
                    "confidence": rel_a.get("confidence"),
                    "reason": rel_a.get("reason"),
                },
                b_name: {
                    "decision": get_decision(rb),
                    "confidence": rel_b.get("confidence"),
                    "reason": rel_b.get("reason"),
                },
            },
        }
        write_output()
        rows, idxs, count = refresh_table(agreement_filter, status_filter, search)
        return rows, idxs, count, f"Saved ({len(adjudications)}/{len(common_ids)} done)."

    with gr.Blocks(title="Relevance comparison / adjudication") as demo:
        gr.Markdown(f"## Compare {a_name} vs {b_name}")
        with gr.Row():
            agreement_filter = gr.Radio(
                ["All", "Agree only", "Disagree only"], value="Disagree only", label="Agreement"
            )
            status_filter = gr.Radio(
                ["All", "Pending only", "Done only"], value="All", label="Status"
            )
            search = gr.Textbox(label="Search title/text")

        count_label = gr.Markdown()
        idxs_state = gr.State([])
        table = gr.Dataframe(
            headers=COMPARE_COLUMNS_BASE,
            datatype=["number", "str", "str", "str"],
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
        )

        gr.Markdown("### Selected record")
        current_id_state = gr.State(None)
        detail_title = gr.Markdown()
        with gr.Row():
            detail_meta_a = gr.Markdown()
            detail_meta_b = gr.Markdown()
        detail_text = gr.Textbox(label="Full text", lines=15, max_lines=30)

        choice = gr.Radio(choices=[], label="Adjudication", value=None)
        note = gr.Textbox(label="Note (optional)")
        with gr.Row():
            save_btn = gr.Button("Save adjudication", variant="primary")
            next_btn = gr.Button("Next pending")
        save_status = gr.Markdown()

        filter_inputs = [agreement_filter, status_filter, search]
        table_outputs = [table, idxs_state, count_label]
        detail_outputs = [current_id_state, detail_title, detail_meta_a, detail_meta_b, detail_text, choice, note, save_status]

        for comp in filter_inputs:
            comp.change(refresh_table, inputs=filter_inputs, outputs=table_outputs)

        demo.load(refresh_table, inputs=filter_inputs, outputs=table_outputs)
        table.select(show_detail, inputs=[idxs_state], outputs=detail_outputs)
        next_btn.click(next_pending, inputs=[current_id_state], outputs=detail_outputs)
        save_btn.click(
            save,
            inputs=[current_id_state, choice, note] + filter_inputs,
            outputs=table_outputs + [save_status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Browse relevance_filter.py JSONL output, or compare two prediction "
        "files side by side and adjudicate disagreements."
    )
    parser.add_argument("--input", type=str, help="Path to a single relevance_filter.py output JSONL (view mode)")
    parser.add_argument("--a", type=str, help="First predictions file (compare/adjudicate mode)")
    parser.add_argument("--b", type=str, help="Second predictions file (compare/adjudicate mode)")
    parser.add_argument("--a-name", type=str, default=None, help="Label for --a (default: model name from file)")
    parser.add_argument("--b-name", type=str, default=None, help="Label for --b (default: model name from file)")
    parser.add_argument(
        "--output", type=str, default=None, help="Where to write adjudications JSONL (required with --a/--b)"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.a and args.b:
        if not args.output:
            parser.error("Compare mode requires --output.")
        demo = build_compare_app(Path(args.a), Path(args.b), Path(args.output), args.a_name, args.b_name)
    elif args.a or args.b or args.input:
        path = args.input or args.a or args.b
        records = load_records(Path(path))
        demo = build_app(records)
    else:
        parser.error("Provide --input for view mode, or --a/--b/--output for compare mode.")

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
