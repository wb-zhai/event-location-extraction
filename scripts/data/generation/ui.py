from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.generation.gemini_event_gen import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    GeminiLLMClient,
    format_ontology,
    generate_one,
    iter_jsonl,
    load_env_file,
    load_json_tolerant,
    normalize_ontology,
)


DEFAULT_JSONL = (
    "dataset/gdelt/events/food_security_2020_2025_100.gemini2.5_flash.jsonl"
)
DEFAULT_ONTOLOGY = Path("ontologies/risk-factors/risk.cluster.description.json")
LOGGER = logging.getLogger("span_annotation_ui")


def resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_source(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source")
    if isinstance(source, dict):
        return source
    return record


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    source = get_source(record)
    title = str(source.get("title") or record.get("title") or "")
    text = str(source.get("text") or record.get("text") or "")
    spans = record.get("spans") if isinstance(record.get("spans"), list) else []
    record_id = str(record.get("id") or f"record_{index}")
    return {
        "id": record_id,
        "index": index,
        "status": str(record.get("status") or ""),
        "title": title,
        "text": text,
        "source_url": source.get("source_url") or record.get("source_url"),
        "publish_date": source.get("publish_date") or record.get("publish_date"),
        "spans": spans,
        "llm": record.get("llm") if isinstance(record.get("llm"), dict) else {},
        "error": record.get("error"),
    }


def summarize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = normalize_record(record, index)
    title = normalized["title"] or normalized["text"][:80].replace("\n", " ")
    if len(title) > 100:
        title = title[:97] + "..."
    return {
        "id": normalized["id"],
        "index": index,
        "title": title,
        "status": normalized["status"],
        "span_count": len(normalized["spans"]),
    }


class AppState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.records: list[dict[str, Any]] = []
        self.loaded_path: str | None = None
        self.ontology = normalize_ontology(load_json_tolerant(args.ontology))
        self.ontology_text = format_ontology(self.ontology)
        self.labels = set(self.ontology)
        if args.env_file:
            load_env_file(args.env_file)


class SpanUIHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], state: AppState) -> None:
        self.state = state
        super().__init__(server_address, SpanUIHandler)

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class SpanUIHandler(BaseHTTPRequestHandler):
    server: SpanUIHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/record":
            self.handle_get_record(parsed.query)
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/load-file":
            self.handle_load_file()
            return
        if parsed.path == "/api/infer":
            self.handle_infer()
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
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

    def handle_load_file(self) -> None:
        try:
            payload = self.read_json()
            path_text = str(payload.get("path") or "").strip()
            if not path_text:
                raise ValueError("Missing JSONL path.")
            path = resolve_local_path(path_text)
            records = iter_jsonl(path)
            self.server.state.records = records
            self.server.state.loaded_path = str(path)
            first = normalize_record(records[0], 0) if records else None
            self.send_json(
                {
                    "path": str(path),
                    "count": len(records),
                    "records": [
                        summarize_record(record, index)
                        for index, record in enumerate(records)
                    ],
                    "record": first,
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_get_record(self, query: str) -> None:
        try:
            params = parse_qs(query)
            index = int((params.get("index") or ["0"])[0])
            records = self.server.state.records
            if index < 0 or index >= len(records):
                raise ValueError(f"Record index out of range: {index}")
            self.send_json({"record": normalize_record(records[index], index)})
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_infer(self) -> None:
        try:
            payload = self.read_json()
            title = str(payload.get("title") or "")
            text = str(payload.get("text") or "")
            if not text.strip():
                raise ValueError("Text is required for inference.")

            state = self.server.state
            model = str(payload.get("model") or state.args.model)
            client = GeminiLLMClient(
                model_name=model,
                system_prompt=None,
                temperature=state.args.temperature,
                max_tokens=state.args.max_tokens,
                reasoning_effort=state.args.reasoning_effort,
                verbose=state.args.verbose,
            )
            result = asyncio.run(
                generate_one(
                    client=client,
                    record={
                        "id": str(payload.get("id") or "manual_input"),
                        "title": title,
                        "text": text,
                    },
                    ontology_text=state.ontology_text,
                    labels=state.labels,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    user_prompt_template=DEFAULT_USER_PROMPT,
                    max_retries=state.args.max_retries,
                    initial_backoff=state.args.initial_backoff,
                    max_backoff=state.args.max_backoff,
                    strict_offsets=state.args.strict_offsets,
                )
            )
            normalized = normalize_record(result, -1)
            self.send_json({"record": normalized, "result": result})
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Span Annotation UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d9ded6;
      --accent: #2563eb;
      --danger: #b42318;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 {
      font-size: 16px;
      line-height: 1.2;
      margin: 0;
      font-weight: 650;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid #b9c2cf;
      background: #ffffff;
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      padding: 6px 10px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    button:disabled {
      color: #98a2b3;
      background: #f2f4f7;
      cursor: not-allowed;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid #c7ced8;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      padding: 8px 9px;
    }
    textarea {
      min-height: 210px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      line-height: 1.5;
    }
    label {
      display: block;
      margin: 0 0 5px;
      color: #344054;
      font-size: 12px;
      font-weight: 650;
    }
    .layout {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr) 320px;
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 64px);
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .section {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .section:last-child { border-bottom: 0; }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .row > * { min-width: 0; }
    .field { margin-bottom: 12px; }
    .meta {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
    }
    .status.error { color: var(--danger); }
    .title {
      font-size: 19px;
      line-height: 1.25;
      margin: 0 0 8px;
      font-weight: 700;
    }
    .doc {
      padding: 18px;
    }
    .annotated-text {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
      line-height: 1.72;
    }
    mark.span {
      border-radius: 4px;
      padding: 2px 3px;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }
    .span-label {
      display: inline-block;
      margin-left: 4px;
      border-radius: 4px;
      padding: 0 3px;
      font-size: 10px;
      line-height: 1.4;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.72);
    }
    .span-list {
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 132px);
      overflow: auto;
      padding-right: 2px;
    }
    .span-card {
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 8px;
      padding: 9px;
      background: #ffffff;
    }
    .span-card strong {
      display: block;
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }
    .tabs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-bottom: 12px;
    }
    .tab {
      background: #f6f8fa;
    }
    .tab.active {
      border-color: var(--accent);
      color: var(--accent);
      background: #eff6ff;
      font-weight: 650;
    }
    .hidden { display: none; }
    @media (max-width: 1050px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .span-list {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Span Annotation UI</h1>
    <div class="status" id="status">Ready</div>
  </header>

  <main class="layout">
    <aside class="panel">
      <div class="section">
        <div class="tabs">
          <button id="fileTab" class="tab active" type="button">File</button>
          <button id="textTab" class="tab" type="button">Text</button>
        </div>

        <div id="fileMode">
          <div class="field">
            <label for="jsonlPath">JSONL path</label>
            <input id="jsonlPath" value="__DEFAULT_JSONL__">
          </div>
          <div class="row">
            <button id="loadFile" class="primary" type="button">Load</button>
            <button id="prevRecord" type="button" disabled>Prev</button>
            <button id="nextRecord" type="button" disabled>Next</button>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="recordSelect">Record</label>
            <select id="recordSelect" disabled></select>
          </div>
        </div>

        <div id="textMode" class="hidden">
          <div class="field">
            <label for="manualTitle">Title</label>
            <input id="manualTitle" placeholder="Optional title">
          </div>
          <div class="field">
            <label for="manualText">Article text</label>
            <textarea id="manualText" placeholder="Paste text to annotate"></textarea>
          </div>
        </div>
      </div>

      <div class="section">
        <div class="field">
          <label for="modelName">Model</label>
          <input id="modelName" value="__DEFAULT_MODEL__">
        </div>
        <button id="runInference" class="primary" type="button">Run inference</button>
        <div class="meta" style="margin-top: 10px;" id="loadedMeta"></div>
      </div>
    </aside>

    <section class="panel doc">
      <h2 class="title" id="docTitle">No record loaded</h2>
      <div class="meta" id="docMeta"></div>
      <hr style="border: 0; border-top: 1px solid var(--line); margin: 14px 0;">
      <div class="annotated-text" id="annotatedText">Load a JSONL file or paste text to run inference.</div>
    </section>

    <aside class="panel">
      <div class="section">
        <div class="row" style="justify-content: space-between;">
          <strong>Spans</strong>
          <span class="meta" id="spanCount">0</span>
        </div>
      </div>
      <div class="section">
        <div class="span-list" id="spanList"></div>
      </div>
    </aside>
  </main>

  <script>
    const palette = [
      "#fed7aa", "#bfdbfe", "#bbf7d0", "#fecdd3", "#ddd6fe", "#fde68a",
      "#a7f3d0", "#fbcfe8", "#bae6fd", "#e9d5ff", "#c7d2fe", "#d9f99d"
    ];
    const state = {
      records: [],
      currentIndex: -1,
      currentRecord: null,
      mode: "file"
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

    function labelColor(label) {
      let hash = 0;
      for (const ch of String(label)) {
        hash = ((hash << 5) - hash + ch.charCodeAt(0)) | 0;
      }
      return palette[Math.abs(hash) % palette.length];
    }

    function validSortedSpans(record) {
      const text = record?.text || "";
      const spans = Array.isArray(record?.spans) ? record.spans : [];
      const out = [];
      const seen = new Set();
      for (const span of spans) {
        const start = Number(span.start_char);
        const end = Number(span.end_char);
        const label = String(span.label || "");
        if (!Number.isInteger(start) || !Number.isInteger(end)) continue;
        if (start < 0 || end <= start || end > text.length) continue;
        const key = `${start}:${end}:${label}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({...span, start_char: start, end_char: end, label});
      }
      out.sort((a, b) => a.start_char - b.start_char || a.end_char - b.end_char);
      const nonOverlapping = [];
      let cursor = 0;
      for (const span of out) {
        if (span.start_char < cursor) continue;
        nonOverlapping.push(span);
        cursor = span.end_char;
      }
      return nonOverlapping;
    }

    function renderText(record) {
      const text = record?.text || "";
      const spans = validSortedSpans(record);
      let html = "";
      let cursor = 0;
      for (const span of spans) {
        const color = labelColor(span.label);
        html += escapeHtml(text.slice(cursor, span.start_char));
        html += `<mark class="span" style="background:${color}" title="${escapeHtml(span.label)}">`;
        html += escapeHtml(text.slice(span.start_char, span.end_char));
        html += `<span class="span-label">${escapeHtml(span.label)}</span></mark>`;
        cursor = span.end_char;
      }
      html += escapeHtml(text.slice(cursor));
      el("annotatedText").innerHTML = html || "No text to display.";
      el("spanCount").textContent = `${spans.length}`;
    }

    function renderSpanList(record) {
      const spans = validSortedSpans(record);
      if (!spans.length) {
        el("spanList").innerHTML = '<div class="meta">No valid spans.</div>';
        return;
      }
      el("spanList").innerHTML = spans.map((span) => {
        const color = labelColor(span.label);
        return `
          <div class="span-card" style="border-left-color:${color}">
            <strong>${escapeHtml(span.span_text || "")}</strong>
            <div class="meta">${escapeHtml(span.label)} · ${span.start_char}-${span.end_char}</div>
            <div class="meta">${escapeHtml(span.rationale || "")}</div>
          </div>
        `;
      }).join("");
    }

    function renderRecord(record) {
      state.currentRecord = record;
      el("docTitle").textContent = record?.title || "Untitled";
      const meta = [];
      if (record?.id) meta.push(record.id);
      if (record?.publish_date) meta.push(record.publish_date);
      if (record?.source_url) meta.push(record.source_url);
      if (record?.llm?.model) meta.push(`model: ${record.llm.model}`);
      if (record?.error) meta.push(`error: ${record.error}`);
      el("docMeta").textContent = meta.join(" · ");
      renderText(record);
      renderSpanList(record);
      if (state.mode === "file") {
        el("manualTitle").value = record?.title || "";
        el("manualText").value = record?.text || "";
      }
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: {"content-type": "application/json"},
        ...options
      });
      const payload = await response.json();
      if (!response.ok || payload.error) {
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return payload;
    }

    async function loadFile() {
      setStatus("Loading file...");
      const payload = await requestJson("/api/load-file", {
        method: "POST",
        body: JSON.stringify({path: el("jsonlPath").value})
      });
      state.records = payload.records || [];
      state.currentIndex = payload.record ? 0 : -1;
      el("recordSelect").innerHTML = state.records.map((record) => {
        const label = `${record.index + 1}. ${record.title} (${record.span_count})`;
        return `<option value="${record.index}">${escapeHtml(label)}</option>`;
      }).join("");
      el("recordSelect").disabled = !state.records.length;
      el("prevRecord").disabled = !state.records.length;
      el("nextRecord").disabled = !state.records.length;
      el("loadedMeta").textContent = `${payload.count} records · ${payload.path}`;
      if (payload.record) renderRecord(payload.record);
      setStatus(`Loaded ${payload.count} records`);
    }

    async function loadRecord(index) {
      if (index < 0 || index >= state.records.length) return;
      setStatus("Loading record...");
      const payload = await requestJson(`/api/record?index=${index}`);
      state.currentIndex = index;
      el("recordSelect").value = String(index);
      renderRecord(payload.record);
      setStatus(`Record ${index + 1} of ${state.records.length}`);
    }

    async function runInference() {
      const fromCurrent = state.mode === "file" && state.currentRecord;
      const title = state.mode === "text" ? el("manualTitle").value : state.currentRecord?.title || "";
      const text = state.mode === "text" ? el("manualText").value : state.currentRecord?.text || "";
      const id = fromCurrent ? state.currentRecord.id : "manual_input";
      setStatus("Running inference...");
      el("runInference").disabled = true;
      try {
        const payload = await requestJson("/api/infer", {
          method: "POST",
          body: JSON.stringify({
            id,
            title,
            text,
            model: el("modelName").value
          })
        });
        renderRecord(payload.record);
        setStatus(payload.record.status === "ok" ? "Inference complete" : "Inference returned an error", payload.record.status !== "ok");
      } finally {
        el("runInference").disabled = false;
      }
    }

    function setMode(mode) {
      state.mode = mode;
      el("fileMode").classList.toggle("hidden", mode !== "file");
      el("textMode").classList.toggle("hidden", mode !== "text");
      el("fileTab").classList.toggle("active", mode === "file");
      el("textTab").classList.toggle("active", mode === "text");
    }

    el("fileTab").addEventListener("click", () => setMode("file"));
    el("textTab").addEventListener("click", () => setMode("text"));
    el("loadFile").addEventListener("click", () => loadFile().catch((err) => setStatus(err.message, true)));
    el("recordSelect").addEventListener("change", (event) => loadRecord(Number(event.target.value)).catch((err) => setStatus(err.message, true)));
    el("prevRecord").addEventListener("click", () => loadRecord(state.currentIndex - 1).catch((err) => setStatus(err.message, true)));
    el("nextRecord").addEventListener("click", () => loadRecord(state.currentIndex + 1).catch((err) => setStatus(err.message, true)));
    el("runInference").addEventListener("click", () => runInference().catch((err) => setStatus(err.message, true)));
  </script>
</body>
</html>
""".replace("__DEFAULT_JSONL__", DEFAULT_JSONL).replace("__DEFAULT_MODEL__", DEFAULT_MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local browser UI for viewing and generating span annotations."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", default="disable")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--initial-backoff", type=float, default=2.0)
    parser.add_argument("--max-backoff", type=float, default=60.0)
    parser.add_argument(
        "--strict-offsets",
        action="store_true",
        help="Drop spans whose text cannot be exactly located in the article text.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log prompts through the LLM client."
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    state = AppState(args)
    server = SpanUIHTTPServer((args.host, args.port), state)
    url = f"http://{args.host}:{args.port}"
    print(f"Span annotation UI running at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
