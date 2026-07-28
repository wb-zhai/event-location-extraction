from __future__ import annotations

import argparse
import json
import logging
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any


from src.inference.text_anchor import AnchorStatus, TextAnchorResolver  # noqa: E402

DEFAULT_JSONL = (
    "dataset/risk-factor/run-15052025/sft/predictions/"
    "qwen3.5-4B-lora-sft-window-500-v4-response-events-only-omit_offsets-v3/"
    "dev.v4.sft.context.events.384.jsonl"
)
LOGGER = logging.getLogger("prediction_compare_ui")
ANCHOR_RESOLVER = TextAnchorResolver()


def resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            rows.append(row)
    return rows


def _valid_span(text: str, start: Any, end: Any) -> bool:
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    )


def _normalize_span(
    text: str,
    span_like: Any,
    *,
    start: Any = None,
    end: Any = None,
    text_value: Any = None,
) -> dict[str, Any] | None:
    nested = span_like if isinstance(span_like, dict) else {}
    span_start = nested.get("start", start)
    span_end = nested.get("end", end)
    span_text = nested.get("text", text_value)

    if _valid_span(text, span_start, span_end):
        actual_text = text[span_start:span_end]
        if not span_text or actual_text == span_text:
            return {
                "start": span_start,
                "end": span_end,
                "text": actual_text,
                "anchor_status": AnchorStatus.MATCH_EXACT,
            }

    if not isinstance(span_text, str) or not span_text.strip():
        span_text = None

    left_context = nested.get("left_context")
    right_context = nested.get("right_context")
    match = ANCHOR_RESOLVER.resolve_with_context(
        text,
        quote=span_text,
        left_context=left_context if isinstance(left_context, str) else None,
        right_context=right_context if isinstance(right_context, str) else None,
        start_hint=span_start if isinstance(span_start, int) else None,
        end_hint=span_end if isinstance(span_end, int) else None,
    )
    if match.start is None or match.end is None or match.matched_text is None:
        return None

    normalized = {
        "start": match.start,
        "end": match.end,
        "text": match.matched_text,
        "anchor_status": match.status,
    }
    if isinstance(left_context, str) and left_context:
        normalized["left_context"] = left_context
    if isinstance(right_context, str) and right_context:
        normalized["right_context"] = right_context
    return normalized


def normalize_events(text: str, raw_events: Any) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(raw_events, list):
        return [], 0

    normalized_events: list[dict[str, Any]] = []
    unresolved = 0

    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        event_type = raw_event.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            continue

        trigger = _normalize_span(
            text,
            raw_event.get("trigger"),
            start=raw_event.get("start"),
            end=raw_event.get("end"),
            text_value=raw_event.get("text"),
        )
        if trigger is None:
            unresolved += 1
            continue

        normalized_event = {
            "event_type": event_type,
            "start_char": trigger["start"],
            "end_char": trigger["end"],
            "trigger_text": trigger["text"],
            "anchor_status": trigger["anchor_status"],
            "arguments": [],
        }

        raw_arguments = raw_event.get("arguments")
        if isinstance(raw_arguments, list):
            for raw_argument in raw_arguments:
                if not isinstance(raw_argument, dict):
                    continue
                role = raw_argument.get("role")
                if not isinstance(role, str) or not role.strip():
                    continue
                normalized_argument = _normalize_span(
                    text,
                    raw_argument.get("span"),
                    start=raw_argument.get("start"),
                    end=raw_argument.get("end"),
                    text_value=raw_argument.get("text"),
                )
                if normalized_argument is None:
                    continue
                normalized_event["arguments"].append(
                    {
                        "role": role,
                        "start_char": normalized_argument["start"],
                        "end_char": normalized_argument["end"],
                        "text": normalized_argument["text"],
                        "anchor_status": normalized_argument["anchor_status"],
                    }
                )

        normalized_events.append(normalized_event)

    normalized_events.sort(
        key=lambda event: (
            event["start_char"],
            event["end_char"],
            event["event_type"],
        )
    )
    return normalized_events, unresolved


def event_key(event: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(event.get("event_type") or ""),
        int(event.get("start_char") or 0),
        int(event.get("end_char") or 0),
    )


def summarize_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    text = str(row.get("question") or "")
    gold_events, gold_unresolved = normalize_events(
        text, row.get("answer", {}).get("events")
    )
    prediction_events, prediction_unresolved = normalize_events(
        text, row.get("prediction", {}).get("events")
    )
    gold_keys = {event_key(event) for event in gold_events}
    prediction_keys = {event_key(event) for event in prediction_events}
    true_positive = len(gold_keys & prediction_keys)
    false_positive = len(prediction_keys - gold_keys)
    false_negative = len(gold_keys - prediction_keys)
    is_exact_match = false_positive == 0 and false_negative == 0
    preview = " ".join(text.split())
    if len(preview) > 110:
        preview = preview[:107] + "..."
    return {
        "index": index,
        "id": str(row.get("id") or f"record_{index}"),
        "preview": preview,
        "gold_count": len(gold_events),
        "prediction_count": len(prediction_events),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "gold_unresolved": gold_unresolved,
        "prediction_unresolved": prediction_unresolved,
        "is_exact_match": is_exact_match,
        "has_gold": bool(gold_events),
        "has_prediction": bool(prediction_events),
    }


def build_record_payload(row: dict[str, Any], index: int) -> dict[str, Any]:
    text = str(row.get("question") or "")
    gold_events, gold_unresolved = normalize_events(
        text, row.get("answer", {}).get("events")
    )
    prediction_events, prediction_unresolved = normalize_events(
        text, row.get("prediction", {}).get("events")
    )
    gold_keys = {event_key(event) for event in gold_events}
    prediction_keys = {event_key(event) for event in prediction_events}

    for event in gold_events:
        event["match_status"] = (
            "matched" if event_key(event) in prediction_keys else "missing"
        )
    for event in prediction_events:
        event["match_status"] = "matched" if event_key(event) in gold_keys else "extra"

    summary = summarize_record(row, index)
    return {
        "index": index,
        "id": summary["id"],
        "text": text,
        "summary": summary,
        "gold_events": gold_events,
        "prediction_events": prediction_events,
        "gold_unresolved": gold_unresolved,
        "prediction_unresolved": prediction_unresolved,
    }


class AppState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.loaded_path: str | None = None
        self.rows: list[dict[str, Any]] = []
        self.summaries: list[dict[str, Any]] = []


class PredictionCompareHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], state: AppState) -> None:
        self.state = state
        super().__init__(server_address, PredictionCompareHandler)

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class PredictionCompareHandler(BaseHTTPRequestHandler):
    server: PredictionCompareHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if self.path.startswith("/api/record"):
            self.handle_get_record()
            return
        if self.path == "/api/state":
            self.send_json(
                {
                    "default_path": DEFAULT_JSONL,
                    "loaded_path": self.server.state.loaded_path,
                    "count": len(self.server.state.summaries),
                    "records": self.server.state.summaries,
                }
            )
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/load":
            self.handle_load()
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def send_text(
        self,
        text: str,
        status: HTTPStatus = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_load(self) -> None:
        try:
            payload = self.read_json()
            path_text = str(payload.get("path") or "").strip()
            if not path_text:
                raise ValueError("Missing JSONL path.")
            path = resolve_local_path(path_text)
            rows = iter_jsonl(path)
            summaries = [summarize_record(row, index) for index, row in enumerate(rows)]
            self.server.state.loaded_path = str(path)
            self.server.state.rows = rows
            self.server.state.summaries = summaries
            first_record = build_record_payload(rows[0], 0) if rows else None
            self.send_json(
                {
                    "path": str(path),
                    "count": len(rows),
                    "records": summaries,
                    "record": first_record,
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_get_record(self) -> None:
        try:
            _, _, query = self.path.partition("?")
            params = {}
            if query:
                for piece in query.split("&"):
                    key, _, value = piece.partition("=")
                    params[key] = value
            index = int(params.get("index", "0"))
            rows = self.server.state.rows
            if index < 0 or index >= len(rows):
                raise ValueError(f"Record index out of range: {index}")
            self.send_json({"record": build_record_payload(rows[index], index)})
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prediction Compare UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1b2430;
      --muted: #6b7280;
      --line: #d9d2c3;
      --accent: #0f766e;
      --accent-soft: #d7f3ef;
      --gold: #d97706;
      --good: #15803d;
      --bad: #b42318;
      --extra: #7c3aed;
      --shadow: 0 8px 24px rgba(27, 36, 48, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 28%),
        linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      background: rgba(255, 253, 248, 0.92);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 750;
    }
    button, input, select {
      font: inherit;
    }
    button, input, select {
      border: 1px solid #c6bfaf;
      border-radius: 10px;
      min-height: 38px;
      background: #fffdf8;
      color: var(--ink);
    }
    button {
      padding: 8px 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    input, select {
      width: 100%;
      padding: 8px 10px;
    }
    main {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 72px);
    }
    .panel {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
    }
    .section {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .section:last-child { border-bottom: 0; }
    .field { margin-bottom: 12px; }
    .field:last-child { margin-bottom: 0; }
    label {
      display: block;
      margin-bottom: 5px;
      font-size: 12px;
      font-weight: 700;
      color: #475467;
    }
    .status, .meta {
      font-size: 12px;
      color: var(--muted);
    }
    .status.error { color: var(--bad); }
    .record-list {
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 320px);
      overflow: auto;
      padding-right: 2px;
    }
    .record-button {
      width: 100%;
      text-align: left;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fffdfa;
    }
    .record-button.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      box-shadow: inset 0 0 0 1px rgba(15, 118, 110, 0.08);
    }
    .record-button .counts {
      display: flex;
      gap: 8px;
      margin-top: 6px;
      font-size: 11px;
      color: var(--muted);
      flex-wrap: wrap;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 700;
    }
    .chip.match { background: #dcfce7; color: #166534; }
    .chip.mismatch { background: #fee2e2; color: #991b1b; }
    .chip.unresolved { background: #fef3c7; color: #92400e; }
    .workspace {
      display: grid;
      gap: 16px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      padding: 14px;
      border-radius: 14px;
      background: linear-gradient(180deg, #fffefb 0%, #f7f3ea 100%);
      border: 1px solid var(--line);
    }
    .metric strong {
      display: block;
      font-size: 22px;
      line-height: 1.1;
      margin-top: 6px;
    }
    .doc-header {
      padding: 18px 18px 0;
    }
    .doc-title {
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.3;
      font-weight: 760;
    }
    .compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      padding: 18px;
    }
    .column {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #fffefb;
    }
    .column-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #faf6ee;
    }
    .annotated-text {
      padding: 14px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
      line-height: 1.72;
      max-height: 420px;
      overflow: auto;
    }
    mark.event {
      padding: 2px 3px;
      border-radius: 6px;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }
    mark.event.matched {
      background: rgba(21, 128, 61, 0.18);
      outline: 1px solid rgba(21, 128, 61, 0.5);
    }
    mark.event.missing {
      background: rgba(185, 28, 28, 0.16);
      outline: 1px solid rgba(185, 28, 28, 0.45);
    }
    mark.event.extra {
      background: rgba(124, 58, 237, 0.16);
      outline: 1px solid rgba(124, 58, 237, 0.45);
    }
    .event-list {
      display: grid;
      gap: 8px;
      padding: 14px;
      border-top: 1px solid var(--line);
      max-height: 280px;
      overflow: auto;
    }
    .event-card {
      border: 1px solid var(--line);
      border-left: 5px solid var(--gold);
      border-radius: 12px;
      padding: 10px 11px;
      background: #fffdfa;
    }
    .event-card.matched { border-left-color: var(--good); }
    .event-card.missing { border-left-color: var(--bad); }
    .event-card.extra { border-left-color: var(--extra); }
    .event-card strong {
      display: block;
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
    @media (max-width: 1100px) {
      main, .compare-grid, .summary-grid {
        grid-template-columns: 1fr;
      }
      .record-list {
        max-height: 280px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Prediction Compare UI</h1>
    <div id="status" class="status">Ready</div>
  </header>
  <main>
    <aside class="panel">
      <div class="section">
        <div class="field">
          <label for="pathInput">Predictions JSONL</label>
          <input id="pathInput" value="__DEFAULT_JSONL__">
        </div>
        <button id="loadButton" class="primary" type="button">Load file</button>
      </div>
      <div class="section">
        <div class="field">
          <label for="filterSelect">Filter</label>
          <select id="filterSelect">
            <option value="all">All records</option>
            <option value="mismatch">Mismatches only</option>
            <option value="exact">Exact matches only</option>
            <option value="gold_only">Gold only</option>
            <option value="pred_only">Prediction only</option>
          </select>
        </div>
        <div class="field">
          <label for="searchInput">Search text</label>
          <input id="searchInput" placeholder="Filter by question preview">
        </div>
        <div class="meta" id="loadedMeta"></div>
      </div>
      <div class="section">
        <div class="record-list" id="recordList"></div>
      </div>
    </aside>

    <section class="workspace">
      <section class="panel">
        <div class="section">
          <div class="summary-grid" id="summaryGrid">
            <div class="metric"><div class="meta">Gold</div><strong>0</strong></div>
            <div class="metric"><div class="meta">Prediction</div><strong>0</strong></div>
            <div class="metric"><div class="meta">True positive</div><strong>0</strong></div>
            <div class="metric"><div class="meta">False positive</div><strong>0</strong></div>
            <div class="metric"><div class="meta">False negative</div><strong>0</strong></div>
            <div class="metric"><div class="meta">Unresolved</div><strong>0 / 0</strong></div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="doc-header">
          <h2 class="doc-title" id="docTitle">No record loaded</h2>
          <div class="meta" id="docMeta"></div>
        </div>
        <div class="compare-grid">
          <section class="column">
            <div class="column-header">
              <strong>Ground truth</strong>
              <span id="goldBadge" class="chip match">0 events</span>
            </div>
            <div class="annotated-text" id="goldText"></div>
            <div class="event-list" id="goldEvents"></div>
          </section>
          <section class="column">
            <div class="column-header">
              <strong>Prediction</strong>
              <span id="predictionBadge" class="chip mismatch">0 events</span>
            </div>
            <div class="annotated-text" id="predictionText"></div>
            <div class="event-list" id="predictionEvents"></div>
          </section>
        </div>
      </section>
    </section>
  </main>

  <script>
    const state = {
      records: [],
      filteredRecords: [],
      currentIndex: -1
    };

    const el = (id) => document.getElementById(id);

    function setStatus(message, isError = false) {
      el("status").textContent = message;
      el("status").classList.toggle("error", isError);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function annotateText(text, events) {
      const sorted = [...events].sort((a, b) => a.start_char - b.start_char || a.end_char - b.end_char);
      let cursor = 0;
      let html = "";
      for (const event of sorted) {
        const start = Number(event.start_char);
        const end = Number(event.end_char);
        if (!Number.isInteger(start) || !Number.isInteger(end)) continue;
        if (start < cursor || end <= start || end > text.length) continue;
        html += escapeHtml(text.slice(cursor, start));
        html += `<mark class="event ${escapeHtml(event.match_status)}">${escapeHtml(text.slice(start, end))}</mark>`;
        cursor = end;
      }
      html += escapeHtml(text.slice(cursor));
      return html || '<span class="empty">No text available.</span>';
    }

    function renderEventList(targetId, events, emptyLabel) {
      const root = el(targetId);
      if (!events.length) {
        root.innerHTML = `<div class="empty">${escapeHtml(emptyLabel)}</div>`;
        return;
      }
      root.innerHTML = events.map((event) => `
        <article class="event-card ${escapeHtml(event.match_status)}">
          <strong>${escapeHtml(event.event_type)}</strong>
          <div>${escapeHtml(event.trigger_text)}</div>
          <div class="meta">
            ${escapeHtml(event.match_status)} · ${event.start_char}-${event.end_char} · ${escapeHtml(event.anchor_status || "unknown")}
          </div>
        </article>
      `).join("");
    }

    function recordMatchesFilter(record) {
      const filter = el("filterSelect").value;
      const search = el("searchInput").value.trim().toLowerCase();
      const preview = String(record.preview || "").toLowerCase();
      if (search && !preview.includes(search)) return false;
      if (filter === "mismatch") return !record.is_exact_match;
      if (filter === "exact") return record.is_exact_match;
      if (filter === "gold_only") return record.has_gold && !record.has_prediction;
      if (filter === "pred_only") return !record.has_gold && record.has_prediction;
      return true;
    }

    function renderRecordList() {
      state.filteredRecords = state.records.filter(recordMatchesFilter);
      const list = el("recordList");
      if (!state.filteredRecords.length) {
        list.innerHTML = '<div class="empty">No records match the current filter.</div>';
        return;
      }
      list.innerHTML = state.filteredRecords.map((record) => `
        <button
          type="button"
          class="record-button ${record.index === state.currentIndex ? "active" : ""}"
          data-index="${record.index}"
        >
          <div>${escapeHtml(record.preview || `Record ${record.index}`)}</div>
          <div class="counts">
            <span class="chip ${record.is_exact_match ? "match" : "mismatch"}">${record.is_exact_match ? "exact" : "diff"}</span>
            <span>gold ${record.gold_count}</span>
            <span>pred ${record.prediction_count}</span>
            <span>fp ${record.fp}</span>
            <span>fn ${record.fn}</span>
          </div>
        </button>
      `).join("");
      for (const button of list.querySelectorAll("[data-index]")) {
        button.addEventListener("click", () => loadRecord(Number(button.dataset.index)));
      }
    }

    function renderSummary(summary) {
      el("summaryGrid").innerHTML = [
        ["Gold", summary.gold_count],
        ["Prediction", summary.prediction_count],
        ["True positive", summary.tp],
        ["False positive", summary.fp],
        ["False negative", summary.fn],
        ["Unresolved", `${summary.gold_unresolved} / ${summary.prediction_unresolved}`]
      ].map(([label, value]) => `
        <div class="metric">
          <div class="meta">${escapeHtml(label)}</div>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `).join("");
    }

    function renderRecord(record) {
      state.currentIndex = record.index;
      renderRecordList();
      renderSummary(record.summary);
      el("docTitle").textContent = record.summary.preview || `Record ${record.index}`;
      el("docMeta").textContent = `id ${record.id} · index ${record.index}`;
      el("goldBadge").textContent = `${record.gold_events.length} events`;
      el("goldBadge").className = `chip ${record.summary.fn ? "mismatch" : "match"}`;
      el("predictionBadge").textContent = `${record.prediction_events.length} events`;
      el("predictionBadge").className = `chip ${record.summary.fp ? "mismatch" : "match"}`;
      el("goldText").innerHTML = annotateText(record.text, record.gold_events);
      el("predictionText").innerHTML = annotateText(record.text, record.prediction_events);
      renderEventList("goldEvents", record.gold_events, "No gold events.");
      renderEventList("predictionEvents", record.prediction_events, "No predicted events.");
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      const payload = await response.json();
      if (!response.ok || payload.error) {
        throw new Error(payload.error || `Request failed with ${response.status}`);
      }
      return payload;
    }

    async function loadRecord(index) {
      try {
        setStatus(`Loading record ${index}...`);
        const payload = await fetchJson(`/api/record?index=${index}`);
        renderRecord(payload.record);
        setStatus(`Viewing record ${index}`);
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function loadFile() {
      try {
        setStatus("Loading JSONL...");
        const payload = await fetchJson("/api/load", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({path: el("pathInput").value})
        });
        state.records = payload.records || [];
        el("loadedMeta").textContent = `${payload.count} records loaded from ${payload.path}`;
        if (payload.record) {
          renderRecord(payload.record);
        } else {
          state.currentIndex = -1;
        }
        renderRecordList();
        setStatus(`Loaded ${payload.count} records`);
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    function installFilterHandlers() {
      el("filterSelect").addEventListener("change", renderRecordList);
      el("searchInput").addEventListener("input", renderRecordList);
      el("loadButton").addEventListener("click", loadFile);
    }

    async function init() {
      installFilterHandlers();
      el("pathInput").value = "__DEFAULT_JSONL__";
      await loadFile();
    }

    init();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_JSONL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    state = AppState(args)
    server = PredictionCompareHTTPServer((args.host, args.port), state)
    print(f"Prediction compare UI: http://{args.host}:{args.port}")
    print(f"Default file: {resolve_local_path(args.path)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
