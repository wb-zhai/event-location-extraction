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


def record_title(r: dict) -> str:
    return str(r.get("title") or (r.get("source") or {}).get("title", ""))


def record_text(r: dict) -> str:
    return str(r.get("text") or (r.get("source") or {}).get("text", ""))


def record_row(i: int, r: dict) -> list:
    rel = r.get("relevance") or {}
    decision = rel.get("decision") or ("error" if "error" in rel else "")
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
            decision = rel.get("decision") or ("error" if "error" in rel else "")
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
        )
        return meta, record_text(r)

    with gr.Blocks(title="Relevance filter viewer") as demo:
        gr.Markdown("## Relevance filter output viewer")
        with gr.Row():
            decision_filter = gr.Radio(
                ["All", "Relevant", "Irrelevant", "Error"], value="All", label="Decision"
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


def main():
    parser = argparse.ArgumentParser(
        description="Browse relevance_filter.py JSONL output in a Gradio app."
    )
    parser.add_argument("--input", required=True, type=str, help="Path to relevance_filter.py output JSONL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    demo = build_app(records)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
