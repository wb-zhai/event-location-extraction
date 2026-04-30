from __future__ import annotations

import argparse
import copy
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
    iter_jsonl,
    load_json_tolerant,
    normalize_argument_roles,
    normalize_event_argument_roles,
    normalize_ontology,
)


ZHAI_EVENTS_DIR = Path("dataset/zhai/events")
DEFAULT_ONTOLOGY = "ontologies/risk-factors/risk.cluster.description.json"
LOGGER = logging.getLogger("event_annotation_ui")


def resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def list_zhai_event_files() -> list[str]:
    root = REPO_ROOT / ZHAI_EVENTS_DIR
    if not root.exists():
        return []
    suffixes = {".jsonl", ".jsonlines"}
    return [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def list_ontology_files() -> list[str]:
    root = REPO_ROOT / "ontologies"
    if not root.exists():
        return [DEFAULT_ONTOLOGY]
    paths = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(root.rglob("*.json"))
        if path.is_file()
    ]
    if DEFAULT_ONTOLOGY not in paths:
        paths.insert(0, DEFAULT_ONTOLOGY)
    return paths


def get_source(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source")
    if isinstance(source, dict):
        return source
    return record


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    source = get_source(record)
    title = str(source.get("title") or record.get("title") or "")
    text = str(source.get("text") or record.get("text") or "")
    events = record.get("events") if isinstance(record.get("events"), list) else []
    return {
        "id": str(record.get("id") or f"record_{index}"),
        "index": index,
        "status": str(record.get("status") or ""),
        "title": title,
        "text": text,
        "source_url": source.get("source_url") or record.get("source_url"),
        "publish_date": source.get("publish_date") or record.get("publish_date"),
        "events": copy.deepcopy(events),
        "llm": record.get("llm") if isinstance(record.get("llm"), dict) else {},
        "error": record.get("error"),
    }


def summarize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    normalized = normalize_record(record, index)
    title = normalized["title"] or normalized["text"][:80].replace("\n", " ")
    if len(title) > 100:
        title = title[:97] + "..."
    event_count = len(normalized["events"])
    argument_count = sum(
        len(event.get("arguments") or [])
        for event in normalized["events"]
        if isinstance(event, dict)
    )
    return {
        "id": normalized["id"],
        "index": index,
        "title": title,
        "status": normalized["status"],
        "event_count": event_count,
        "argument_count": argument_count,
    }


def default_save_path(input_path: str | None) -> str:
    if not input_path:
        return ""
    path = Path(input_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}.fixed{path.suffix}"))
    return f"{input_path}.fixed.jsonl"


def validate_offset(start: Any, end: Any, text: str, context: str) -> tuple[int, int]:
    try:
        start_char = int(start)
        end_char = int(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} offsets must be integers.") from exc
    if start_char < 0 or end_char <= start_char or end_char > len(text):
        raise ValueError(
            f"{context} offsets must satisfy 0 <= start < end <= {len(text)}."
        )
    return start_char, end_char


class Ontology:
    def __init__(self, raw: Any) -> None:
        self.events = normalize_ontology(raw)
        self.argument_roles = normalize_argument_roles(raw)
        self.event_argument_roles = normalize_event_argument_roles(
            raw, set(self.events), set(self.argument_roles)
        )

    def as_payload(self, path: str | None = None) -> dict[str, Any]:
        payload = {
            "events": self.events,
            "argument_roles": self.argument_roles,
            "event_argument_roles": self.event_argument_roles,
        }
        if path is not None:
            payload["path"] = path
        return payload


def clean_events(
    events: Any,
    text: str,
    ontology: Ontology,
    record_label: str,
) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        raise ValueError(f"{record_label} events must be a list.")

    cleaned: list[dict[str, Any]] = []
    for event_index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(f"{record_label} event {event_index} must be an object.")
        event_type = str(event.get("event_type") or "").strip()
        if event_type not in ontology.events:
            raise ValueError(
                f"{record_label} event {event_index} has invalid event_type: {event_type!r}."
            )
        start_char, end_char = validate_offset(
            event.get("start_char"),
            event.get("end_char"),
            text,
            f"{record_label} event {event_index}",
        )

        clean_event = dict(event)
        clean_event["event_type"] = event_type
        clean_event["trigger_text"] = text[start_char:end_char]
        clean_event["start_char"] = start_char
        clean_event["end_char"] = end_char

        allowed_roles = set(
            ontology.event_argument_roles.get(event_type, ontology.argument_roles)
        )
        arguments = event.get("arguments") or []
        if not isinstance(arguments, list):
            raise ValueError(f"{record_label} event {event_index} arguments must be a list.")

        clean_arguments: list[dict[str, Any]] = []
        for argument_index, argument in enumerate(arguments, start=1):
            if not isinstance(argument, dict):
                raise ValueError(
                    f"{record_label} event {event_index} argument {argument_index} must be an object."
                )
            role = str(argument.get("role") or "").strip()
            if role not in allowed_roles:
                raise ValueError(
                    f"{record_label} event {event_index} argument {argument_index} "
                    f"has invalid role {role!r} for {event_type!r}."
                )
            arg_start, arg_end = validate_offset(
                argument.get("start_char"),
                argument.get("end_char"),
                text,
                f"{record_label} event {event_index} argument {argument_index}",
            )
            clean_argument = dict(argument)
            clean_argument["role"] = role
            clean_argument["text"] = text[arg_start:arg_end]
            clean_argument["start_char"] = arg_start
            clean_argument["end_char"] = arg_end
            clean_arguments.append(clean_argument)

        clean_event["arguments"] = clean_arguments
        cleaned.append(clean_event)

    return cleaned


class AppState:
    def __init__(self, ontology_path: str) -> None:
        self.records: list[dict[str, Any]] = []
        self.loaded_path: str | None = None
        self.ontology_path: str | None = None
        self.ontology = self.load_ontology(ontology_path)

    def load_ontology(self, path_text: str) -> Ontology:
        path = resolve_local_path(path_text)
        ontology = Ontology(load_json_tolerant(path))
        self.ontology_path = str(path)
        return ontology


class AnnotationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], state: AppState) -> None:
        self.state = state
        super().__init__(server_address, AnnotationHandler)

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/options":
            self.send_json(
                {
                    "datasets": list_zhai_event_files(),
                    "ontologies": list_ontology_files(),
                    "default_ontology": DEFAULT_ONTOLOGY,
                }
            )
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
        if parsed.path == "/api/load-ontology":
            self.handle_load_ontology()
            return
        if parsed.path == "/api/save":
            self.handle_save()
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
                    "save_path": default_save_path(str(path)),
                    "count": len(records),
                    "records": [
                        summarize_record(record, index)
                        for index, record in enumerate(records)
                    ],
                    "record": first,
                    "ontology": self.server.state.ontology.as_payload(
                        self.server.state.ontology_path
                    ),
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_load_ontology(self) -> None:
        try:
            payload = self.read_json()
            path_text = str(payload.get("path") or "").strip()
            if not path_text:
                raise ValueError("Missing ontology path.")
            ontology = self.server.state.load_ontology(path_text)
            self.send_json(ontology.as_payload(self.server.state.ontology_path))
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

    def handle_save(self) -> None:
        try:
            payload = self.read_json()
            path_text = str(payload.get("path") or "").strip()
            if not path_text:
                raise ValueError("Missing output path.")
            edits = payload.get("edits")
            if not isinstance(edits, dict):
                raise ValueError("Save payload must contain an edits object.")

            state = self.server.state
            output_records = copy.deepcopy(state.records)
            for raw_index, events in edits.items():
                try:
                    index = int(raw_index)
                except ValueError as exc:
                    raise ValueError(f"Invalid record index in edits: {raw_index!r}") from exc
                if index < 0 or index >= len(output_records):
                    raise ValueError(f"Record index out of range in edits: {index}")
                text = normalize_record(output_records[index], index)["text"]
                output_records[index]["events"] = clean_events(
                    events, text, state.ontology, f"record {index + 1}"
                )

            output_path = resolve_local_path(path_text)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                for record in output_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.send_json({"path": str(output_path), "count": len(output_records)})
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Event Annotation UI</title>
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
      --ok: #047857;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
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
    h1, h2, h3 { margin: 0; }
    h1 { font-size: 16px; line-height: 1.2; font-weight: 650; }
    h2 { font-size: 19px; line-height: 1.25; font-weight: 700; }
    h3 { font-size: 13px; line-height: 1.2; font-weight: 700; }
    button, input, select, textarea { font: inherit; }
    button {
      border: 1px solid #b9c2cf;
      background: #ffffff;
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      padding: 6px 10px;
      cursor: pointer;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: #ffffff; }
    button.danger { border-color: #f3b2ac; color: var(--danger); }
    button:disabled { color: #98a2b3; background: #f2f4f7; cursor: not-allowed; }
    input, select, textarea {
      width: 100%;
      border: 1px solid #c7ced8;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      padding: 8px 9px;
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
      grid-template-columns: 320px minmax(0, 1fr) 360px;
      gap: 14px;
      padding: 14px;
      flex: 1;
      min-height: 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
      min-height: 0;
    }
    .section { padding: 14px; border-bottom: 1px solid var(--line); }
    .section:last-child { border-bottom: 0; }
    .field { margin-bottom: 12px; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { min-width: 0; }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .meta, .status { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .status { min-height: 18px; }
    .status.error { color: var(--danger); }
    .status.ok { color: var(--ok); }
    .doc { padding: 18px; overflow-y: auto; }
    .doc-head { margin-bottom: 14px; }
    .annotated-text {
      position: relative;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 15px;
      line-height: 1.72;
      user-select: text;
    }
    mark.span {
      position: relative;
      border-radius: 4px;
      padding: 2px 3px;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
      cursor: pointer;
    }
    mark.event { outline: 2px solid rgba(31, 41, 51, 0.72); }
    mark.argument { border-bottom: 3px dashed rgba(31, 41, 51, 0.72); }
    mark.span.dimmed { background: #e5e7eb !important; color: #6b7280; }
    mark.span.selected { box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.34); z-index: 1; }
    mark.span.related { box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.20); z-index: 1; }
    mark.span.overlap { outline: 2px solid rgba(127, 86, 217, 0.72); }
    .span-label {
      display: inline-block;
      margin-left: 4px;
      border-radius: 4px;
      padding: 0 3px;
      font-size: 10px;
      line-height: 1.4;
      font-weight: 700;
      background: rgba(255, 255, 255, 0.72);
      user-select: none;
    }
    .span-label-stack {
      display: inline-flex;
      flex-wrap: wrap;
      gap: 3px;
      margin-left: 4px;
      vertical-align: baseline;
    }
    .span-label-stack .span-label {
      margin-left: 0;
      cursor: pointer;
    }
    .badge-kind {
      display: inline-block;
      margin-right: 3px;
      border-radius: 3px;
      padding: 0 3px;
      color: #ffffff;
      font-size: 9px;
      line-height: 1.35;
      letter-spacing: 0;
    }
    mark.event .badge-kind, .span-label[data-kind="event"] .badge-kind { background: #1f2933; }
    mark.argument .badge-kind, .span-label[data-kind="argument"] .badge-kind { background: #2563eb; }
    .list {
      display: grid;
      gap: 8px;
      max-height: 34vh;
      overflow: auto;
      padding-right: 2px;
    }
    .card {
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 8px;
      padding: 9px;
      background: #ffffff;
      cursor: pointer;
    }
    .card.selected { box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.22); border-color: var(--accent); }
    .card strong { display: block; margin-bottom: 4px; overflow-wrap: anywhere; }
    .event-group {
      display: grid;
      gap: 7px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 8px;
      background: #f8fafc;
    }
    .event-group.focused {
      border-color: rgba(37, 99, 235, 0.50);
      background: #eff6ff;
    }
    .argument-card {
      margin-left: 16px;
      background: #ffffff;
      border-left-style: dashed;
    }
    .list-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 2px;
    }
    .list-head button {
      min-height: 28px;
      padding: 3px 8px;
      font-size: 12px;
    }
    .toolbar { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .hidden { display: none; }
    @media (max-width: 1120px) {
      .layout { grid-template-columns: 1fr; }
      .list { max-height: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Event Annotation UI</h1>
    <div class="status" id="status">Ready</div>
  </header>

  <main class="layout">
    <aside class="panel">
      <div class="section">
        <div class="field">
          <label for="jsonlPath">JSONL file</label>
          <select id="jsonlPath"></select>
        </div>
        <div class="field">
          <label for="jsonlPathInput">Or file path</label>
          <input id="jsonlPathInput" placeholder="dataset/path/file.jsonl">
        </div>
        <div class="field">
          <label for="ontologyPath">Ontology</label>
          <select id="ontologyPath"></select>
        </div>
        <div class="toolbar">
          <button id="loadFile" class="primary" type="button">Load file</button>
          <button id="loadOntology" type="button">Load ontology</button>
        </div>
        <div class="meta" id="loadedMeta" style="margin-top: 10px;"></div>
      </div>

      <div class="section">
        <div class="row" style="margin-bottom: 12px;">
          <button id="prevRecord" type="button" disabled>Prev</button>
          <button id="nextRecord" type="button" disabled>Next</button>
        </div>
        <div class="field">
          <label for="recordSelect">Record</label>
          <select id="recordSelect" disabled></select>
        </div>
      </div>

      <div class="section">
        <div class="field">
          <label for="savePath">Save as</label>
          <input id="savePath" placeholder="Output JSONL path">
        </div>
        <button id="saveFile" class="primary" type="button" disabled>Save fixed copy</button>
      </div>
    </aside>

    <section class="panel doc">
      <div class="doc-head">
        <h2 id="docTitle">No record loaded</h2>
        <div class="meta" id="docMeta"></div>
        <div class="meta" id="selectionMeta" style="margin-top: 8px;">Select text to fill offsets.</div>
      </div>
      <hr style="border: 0; border-top: 1px solid var(--line); margin: 14px 0;">
      <div class="annotated-text" id="annotatedText">Load a JSONL file to begin.</div>
    </section>

    <aside class="panel">
      <div class="section">
        <div class="row" style="justify-content: space-between; margin-bottom: 10px;">
          <h3>Annotations</h3>
          <span class="meta" id="annotationCount">0</span>
        </div>
        <div class="toolbar">
          <button id="newEvent" type="button" disabled>Add event</button>
          <button id="newArgument" type="button" disabled>Add argument</button>
        </div>
      </div>

      <div class="section">
        <div class="list" id="annotationList"></div>
      </div>

      <div class="section">
        <h3 id="editorTitle">Editor</h3>
        <div class="meta" id="editorHint" style="margin: 8px 0 12px;">Select or add an annotation.</div>

        <div id="editorFields" class="hidden">
          <div class="field">
            <label for="kindField">Kind</label>
            <input id="kindField" disabled>
          </div>
          <div class="field" id="eventLinkField">
            <label for="eventLink">Connected event</label>
            <select id="eventLink"></select>
          </div>
          <div class="field" id="eventTypeField">
            <label for="eventType">Event label</label>
            <select id="eventType"></select>
          </div>
          <div class="field" id="roleField">
            <label for="argumentRole">Argument label</label>
            <select id="argumentRole"></select>
          </div>
          <div class="split">
            <div class="field">
              <label for="startChar">Start</label>
              <input id="startChar" type="number" min="0" step="1">
            </div>
            <div class="field">
              <label for="endChar">End</label>
              <input id="endChar" type="number" min="0" step="1">
            </div>
          </div>
          <div class="field">
            <label for="previewText">Text</label>
            <textarea id="previewText" rows="3" readonly></textarea>
          </div>
          <div class="toolbar">
            <button id="applyEdit" class="primary" type="button">Apply</button>
            <button id="removeAnnotation" class="danger" type="button">Remove</button>
          </div>
        </div>
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
      edits: {},
      currentIndex: -1,
      currentRecord: null,
      selected: null,
      selectedRange: null,
      ontology: {events: {}, argument_roles: {}, event_argument_roles: {}}
    };
    const el = (id) => document.getElementById(id);

    function setStatus(message, level = "") {
      el("status").textContent = message;
      el("status").classList.toggle("error", level === "error");
      el("status").classList.toggle("ok", level === "ok");
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

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: {"content-type": "application/json"},
        ...options
      });
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || `Request failed: ${response.status}`);
      return payload;
    }

    function currentEvents() {
      if (!state.currentRecord) return [];
      const edited = state.edits[state.currentIndex];
      return edited ? edited : (Array.isArray(state.currentRecord.events) ? state.currentRecord.events : []);
    }

    function setCurrentEvents(events) {
      state.edits[state.currentIndex] = events;
      state.currentRecord.events = events;
    }

    function eventLabels() {
      return Object.keys(state.ontology.events || {}).sort();
    }

    function allArgumentRoles() {
      return Object.keys(state.ontology.argument_roles || {}).sort();
    }

    function rolesForEvent(eventType) {
      const roles = state.ontology.event_argument_roles?.[eventType];
      return Array.isArray(roles) && roles.length ? roles : allArgumentRoles();
    }

    function optionHtml(values, selected = "") {
      return values.map((value) => {
        const isSelected = value === selected ? " selected" : "";
        return `<option value="${escapeHtml(value)}"${isSelected}>${escapeHtml(value)}</option>`;
      }).join("");
    }

    function pathOptions(values, selected = "", placeholder = "") {
      const options = placeholder ? [`<option value="">${escapeHtml(placeholder)}</option>`] : [];
      for (const value of values) {
        const isSelected = value === selected ? " selected" : "";
        options.push(`<option value="${escapeHtml(value)}"${isSelected}>${escapeHtml(value)}</option>`);
      }
      return options.join("");
    }

    function recordMeta(record) {
      const meta = [];
      if (record?.id) meta.push(record.id);
      if (record?.publish_date) meta.push(record.publish_date);
      if (record?.source_url) meta.push(record.source_url);
      if (record?.llm?.model) meta.push(`model: ${record.llm.model}`);
      if (record?.error) meta.push(`error: ${record.error}`);
      return meta.join(" · ");
    }

    function validSortedAnnotations(record) {
      const text = record?.text || "";
      const events = Array.isArray(record?.events) ? record.events : [];
      const out = [];
      events.forEach((event, eventIndex) => {
        if (!event || typeof event !== "object") return;
        const start = Number(event.start_char);
        const end = Number(event.end_char);
        const label = String(event.event_type || "");
        if (!Number.isInteger(start) || !Number.isInteger(end)) return;
        if (start < 0 || end <= start || end > text.length) return;
        out.push({
          start_char: start,
          end_char: end,
          label,
          kind: "event",
          event_index: eventIndex,
          span_text: event.trigger_text || text.slice(start, end)
        });
        const args = Array.isArray(event.arguments) ? event.arguments : [];
        args.forEach((argument, argumentIndex) => {
          const argStart = Number(argument?.start_char);
          const argEnd = Number(argument?.end_char);
          const role = String(argument?.role || "");
          if (!Number.isInteger(argStart) || !Number.isInteger(argEnd)) return;
          if (argStart < 0 || argEnd <= argStart || argEnd > text.length) return;
          out.push({
            start_char: argStart,
            end_char: argEnd,
            label: role,
            kind: "argument",
            event_index: eventIndex,
            argument_index: argumentIndex,
            span_text: argument.text || text.slice(argStart, argEnd)
          });
        });
      });
      out.sort((a, b) => a.start_char - b.start_char || a.end_char - b.end_char);
      return out;
    }

    function annotationGroups(annotations) {
      const groups = [];
      for (const annotation of annotations) {
        const last = groups[groups.length - 1];
        if (!last || annotation.start_char >= last.end_char) {
          groups.push({
            start_char: annotation.start_char,
            end_char: annotation.end_char,
            annotations: [annotation]
          });
        } else {
          last.end_char = Math.max(last.end_char, annotation.end_char);
          last.annotations.push(annotation);
        }
      }
      return groups;
    }

    function isSelectedAnnotation(annotation) {
      return state.selected
        && state.selected.kind === annotation.kind
        && state.selected.eventIndex === annotation.event_index
        && (annotation.kind === "event" || state.selected.argumentIndex === annotation.argument_index);
    }

    function annotationLabel(annotation, includeOffsets = false) {
      const relationId = `E${annotation.event_index + 1}`;
      const labelKind = annotation.kind === "argument" ? "ROLE" : "EVENT";
      const labelText = `${relationId}:${annotation.label}`;
      const offsets = includeOffsets ? ` ${annotation.start_char}-${annotation.end_char}` : "";
      return `<span class="span-label" data-kind="${annotation.kind}" data-event-index="${annotation.event_index}" data-argument-index="${annotation.argument_index ?? ""}"><span class="badge-kind">${labelKind}</span>${escapeHtml(labelText + offsets)}</span>`;
    }

    function renderText() {
      const record = state.currentRecord;
      const text = record?.text || "";
      const groups = annotationGroups(validSortedAnnotations(record));
      let html = "";
      let cursor = 0;
      for (const group of groups) {
        const first = group.annotations[0];
        const colorKey = first.kind === "argument" ? `event:${first.event_index}` : first.label;
        const color = labelColor(colorKey);
        const selected = group.annotations.some(isSelectedAnnotation);
        const related = state.selected
          && group.annotations.some((annotation) => annotation.event_index === state.selected.eventIndex)
          && !selected;
        const dimmed = state.selected
          && !group.annotations.some((annotation) => annotation.event_index === state.selected.eventIndex)
          ? " dimmed" : "";
        const selectedClass = selected ? " selected" : "";
        const relatedClass = related ? " related" : "";
        const overlapClass = group.annotations.length > 1 ? " overlap" : "";
        const kindClass = group.annotations.length === 1 ? ` ${first.kind}` : "";
        html += escapeHtml(text.slice(cursor, group.start_char));
        html += `<mark class="span${kindClass}${overlapClass}${dimmed}${selectedClass}${relatedClass}" data-kind="${first.kind}" data-event-index="${first.event_index}" data-argument-index="${first.argument_index ?? ""}" style="background:${color}">`;
        html += escapeHtml(text.slice(group.start_char, group.end_char));
        if (group.annotations.length === 1) {
          html += annotationLabel(first);
        } else {
          html += `<span class="span-label-stack">${group.annotations.map((annotation) => annotationLabel(annotation, true)).join("")}</span>`;
        }
        html += "</mark>";
        cursor = group.end_char;
      }
      html += escapeHtml(text.slice(cursor));
      el("annotatedText").innerHTML = html || "No text to display.";
    }

    function renderList() {
      const events = currentEvents();
      const focusedEventIndex = state.selected?.kind === "event" ? state.selected.eventIndex : null;
      let totalCount = events.length;
      for (const event of events) totalCount += Array.isArray(event.arguments) ? event.arguments.length : 0;
      if (!events.length) {
        el("annotationCount").textContent = "0";
        el("annotationList").innerHTML = '<div class="meta">No annotations.</div>';
        return;
      }
      const visibleEntries = focusedEventIndex === null
        ? events.map((event, eventIndex) => ({event, eventIndex}))
        : events
          .map((event, eventIndex) => ({event, eventIndex}))
          .filter((entry) => entry.eventIndex === focusedEventIndex);
      if (focusedEventIndex !== null) {
        const focusedArgs = visibleEntries[0]?.event?.arguments || [];
        el("annotationCount").textContent = `E${focusedEventIndex + 1}: ${1 + focusedArgs.length}/${totalCount}`;
      } else {
        el("annotationCount").textContent = String(totalCount);
      }
      const listHead = focusedEventIndex === null ? "" : `
        <div class="list-head">
          <span class="meta">Showing E${focusedEventIndex + 1} arguments</span>
          <button type="button" data-clear-selection>Show all</button>
        </div>
      `;
      el("annotationList").innerHTML = listHead + visibleEntries.map(({event, eventIndex}) => {
        const eventText = event.trigger_text || "";
        const selected = state.selected?.kind === "event" && state.selected.eventIndex === eventIndex ? " selected" : "";
        const color = labelColor(event.event_type || "");
        const args = Array.isArray(event.arguments) ? event.arguments : [];
        const argRows = args.map((argument, argumentIndex) => {
          const argSelected = state.selected?.kind === "argument"
            && state.selected.eventIndex === eventIndex
            && state.selected.argumentIndex === argumentIndex ? " selected" : "";
          return `
            <div class="card argument-card${argSelected}" data-kind="argument" data-event-index="${eventIndex}" data-argument-index="${argumentIndex}" style="border-left-color:${labelColor(`event:${eventIndex}`)}">
              <strong>ARG ${escapeHtml(argument.text || "")}</strong>
              <div class="meta">E${eventIndex + 1} · ${escapeHtml(argument.role || "")} · ${argument.start_char}-${argument.end_char}</div>
            </div>
          `;
        }).join("") || '<div class="meta" style="margin-left: 16px;">No arguments for this event.</div>';
        const focused = focusedEventIndex === eventIndex ? " focused" : "";
        return `
          <div class="event-group${focused}">
            <div class="card${selected}" data-kind="event" data-event-index="${eventIndex}" style="border-left-color:${color}">
              <strong>EVENT E${eventIndex + 1} ${escapeHtml(eventText)}</strong>
              <div class="meta">${escapeHtml(event.event_type || "")} · ${event.start_char}-${event.end_char}</div>
            </div>
            ${argRows}
          </div>
        `;
      }).join("");
    }

    function renderRecord(record) {
      state.currentRecord = record;
      if (state.edits[state.currentIndex]) {
        state.currentRecord.events = state.edits[state.currentIndex];
      }
      state.selected = null;
      el("docTitle").textContent = record?.title || "Untitled";
      el("docMeta").textContent = recordMeta(record);
      el("newEvent").disabled = !record;
      el("newArgument").disabled = !record;
      renderText();
      renderList();
      renderEditor();
    }

    function validateRange(start, end) {
      const text = state.currentRecord?.text || "";
      if (!Number.isInteger(start) || !Number.isInteger(end)) throw new Error("Offsets must be integers.");
      if (start < 0 || end <= start || end > text.length) {
        throw new Error(`Offsets must satisfy 0 <= start < end <= ${text.length}.`);
      }
      return text.slice(start, end);
    }

    function selectedTextRange() {
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;
      const root = el("annotatedText");
      const range = selection.getRangeAt(0);
      if (!root.contains(range.startContainer) || !root.contains(range.endContainer)) return null;
      const before = document.createRange();
      before.selectNodeContents(root);
      before.setEnd(range.startContainer, range.startOffset);
      const start = rangeTextWithoutLabels(before).length;
      const selectedLength = rangeTextWithoutLabels(range).length;
      const end = start + selectedLength;
      if (end <= start) return null;
      const text = state.currentRecord?.text || "";
      return {start, end, text: text.slice(start, end)};
    }

    function rangeTextWithoutLabels(range) {
      const fragment = range.cloneContents();
      for (const label of fragment.querySelectorAll(".span-label")) {
        label.remove();
      }
      return fragment.textContent || "";
    }

    function fillOffsetsFromSelection() {
      const range = selectedTextRange();
      if (!range) return;
      state.selectedRange = range;
      el("selectionMeta").textContent = `Selected ${range.start}-${range.end}: ${range.text}`;
      if (!el("editorFields").classList.contains("hidden")) {
        el("startChar").value = String(range.start);
        el("endChar").value = String(range.end);
        updatePreview();
      }
    }

    function selectAnnotation(kind, eventIndex, argumentIndex = null) {
      state.selected = {kind, eventIndex, argumentIndex};
      renderText();
      renderList();
      renderEditor();
    }

    function clearSelection() {
      state.selected = null;
      renderText();
      renderList();
      renderEditor();
    }

    function renderEditor() {
      const selected = state.selected;
      const hasSelected = Boolean(selected);
      el("editorFields").classList.toggle("hidden", !hasSelected);
      el("editorHint").classList.toggle("hidden", hasSelected);
      if (!hasSelected) return;

      const events = currentEvents();
      const event = events[selected.eventIndex];
      if (!event) {
        state.selected = null;
        renderEditor();
        return;
      }
      const isArgument = selected.kind === "argument";
      const argument = isArgument ? event.arguments?.[selected.argumentIndex] : null;
      el("kindField").value = isArgument ? "argument" : "event";
      el("editorTitle").textContent = isArgument ? "Argument" : "Event";
      el("eventLinkField").classList.toggle("hidden", !isArgument);
      el("eventTypeField").classList.toggle("hidden", isArgument);
      el("roleField").classList.toggle("hidden", !isArgument);

      el("eventType").innerHTML = optionHtml(eventLabels(), event.event_type || "");
      el("eventLink").innerHTML = events.map((candidate, index) => {
        const label = `E${index + 1} ${candidate.trigger_text || ""}`;
        const selectedText = index === selected.eventIndex ? " selected" : "";
        return `<option value="${index}"${selectedText}>${escapeHtml(label)}</option>`;
      }).join("");
      const roleEventType = events[Number(el("eventLink").value)]?.event_type || event.event_type || "";
      el("argumentRole").innerHTML = optionHtml(rolesForEvent(roleEventType), argument?.role || "");

      const source = isArgument ? argument : event;
      el("startChar").value = String(source?.start_char ?? "");
      el("endChar").value = String(source?.end_char ?? "");
      updatePreview();
    }

    function updatePreview() {
      const start = Number(el("startChar").value);
      const end = Number(el("endChar").value);
      if (!Number.isInteger(start) || !Number.isInteger(end) || !state.currentRecord) {
        el("previewText").value = "";
        return;
      }
      const text = state.currentRecord.text || "";
      el("previewText").value = start >= 0 && end > start && end <= text.length ? text.slice(start, end) : "";
    }

    function applyEdit() {
      if (!state.selected || !state.currentRecord) return;
      const events = copyEvents(currentEvents());
      const selected = state.selected;
      const start = Number(el("startChar").value);
      const end = Number(el("endChar").value);
      const text = validateRange(start, end);
      if (selected.kind === "event") {
        const eventType = el("eventType").value;
        if (!eventLabels().includes(eventType)) throw new Error("Invalid event label.");
        const event = events[selected.eventIndex];
        event.event_type = eventType;
        event.start_char = start;
        event.end_char = end;
        event.trigger_text = text;
      } else {
        const oldEvent = events[selected.eventIndex];
        const argument = oldEvent.arguments[selected.argumentIndex];
        const newEventIndex = Number(el("eventLink").value);
        const newEvent = events[newEventIndex];
        const role = el("argumentRole").value;
        if (!rolesForEvent(newEvent.event_type).includes(role)) throw new Error("Invalid argument label for connected event.");
        const editedArgument = {...argument, role, start_char: start, end_char: end, text};
        oldEvent.arguments.splice(selected.argumentIndex, 1);
        newEvent.arguments = Array.isArray(newEvent.arguments) ? newEvent.arguments : [];
        newEvent.arguments.push(editedArgument);
        selected.eventIndex = newEventIndex;
        selected.argumentIndex = newEvent.arguments.length - 1;
      }
      setCurrentEvents(events);
      renderText();
      renderList();
      renderEditor();
      setStatus("Applied edit", "ok");
    }

    function removeAnnotation() {
      if (!state.selected) return;
      const events = copyEvents(currentEvents());
      const selected = state.selected;
      if (selected.kind === "event") {
        events.splice(selected.eventIndex, 1);
      } else {
        events[selected.eventIndex].arguments.splice(selected.argumentIndex, 1);
      }
      state.selected = null;
      setCurrentEvents(events);
      renderText();
      renderList();
      renderEditor();
      setStatus("Removed annotation", "ok");
    }

    function copyEvents(events) {
      return JSON.parse(JSON.stringify(events || []));
    }

    function addEvent() {
      if (!state.currentRecord) return;
      const range = state.selectedRange || {start: 0, end: Math.min(1, (state.currentRecord.text || "").length)};
      const start = range.start;
      const end = range.end;
      const label = eventLabels()[0] || "";
      if (!label) throw new Error("Load an ontology before adding events.");
      const text = validateRange(start, end);
      const events = copyEvents(currentEvents());
      events.push({event_type: label, trigger_text: text, start_char: start, end_char: end, arguments: []});
      setCurrentEvents(events);
      selectAnnotation("event", events.length - 1);
      setStatus("Added event", "ok");
    }

    function addArgument() {
      if (!state.currentRecord) return;
      const events = copyEvents(currentEvents());
      if (!events.length) throw new Error("Add an event before adding an argument.");
      const eventIndex = state.selected?.kind === "event" || state.selected?.kind === "argument" ? state.selected.eventIndex : 0;
      const event = events[eventIndex];
      const roles = rolesForEvent(event.event_type);
      if (!roles.length) throw new Error("No argument roles are available for the selected event.");
      const range = state.selectedRange || {start: Number(event.start_char), end: Number(event.end_char)};
      const text = validateRange(range.start, range.end);
      event.arguments = Array.isArray(event.arguments) ? event.arguments : [];
      event.arguments.push({role: roles[0], text, start_char: range.start, end_char: range.end});
      setCurrentEvents(events);
      selectAnnotation("argument", eventIndex, event.arguments.length - 1);
      setStatus("Added argument", "ok");
    }

    async function loadOntology() {
      setStatus("Loading ontology...");
      const path = el("ontologyPath").value;
      if (!path) throw new Error("Select an ontology.");
      const payload = await requestJson("/api/load-ontology", {
        method: "POST",
        body: JSON.stringify({path})
      });
      state.ontology = payload;
      renderEditor();
      setStatus(`Loaded ontology with ${eventLabels().length} event labels`, "ok");
    }

    async function loadFile() {
      setStatus("Loading file...");
      const typedPath = el("jsonlPathInput").value.trim();
      const path = typedPath || el("jsonlPath").value;
      if (!path) throw new Error("Select or enter a JSONL file path.");
      const payload = await requestJson("/api/load-file", {
        method: "POST",
        body: JSON.stringify({path})
      });
      state.records = payload.records || [];
      state.edits = {};
      state.ontology = payload.ontology || state.ontology;
      state.currentIndex = payload.record ? 0 : -1;
      el("savePath").value = payload.save_path || "";
      el("recordSelect").innerHTML = state.records.map((record) => {
        const label = `${record.index + 1}. ${record.title} (${record.event_count}/${record.argument_count})`;
        return `<option value="${record.index}">${escapeHtml(label)}</option>`;
      }).join("");
      const hasRecords = state.records.length > 0;
      el("recordSelect").disabled = !hasRecords;
      el("prevRecord").disabled = !hasRecords;
      el("nextRecord").disabled = !hasRecords;
      el("saveFile").disabled = !hasRecords;
      el("loadedMeta").textContent = `${payload.count} records · ${payload.path}`;
      renderRecord(payload.record);
      setStatus(`Loaded ${payload.count} records`, "ok");
    }

    async function loadOptions() {
      setStatus("Loading options...");
      const payload = await requestJson("/api/options");
      el("jsonlPath").innerHTML = pathOptions(payload.datasets || [], "", "Select a file");
      el("ontologyPath").innerHTML = pathOptions(
        payload.ontologies || [],
        payload.default_ontology || "",
        "Select an ontology"
      );
      if (el("ontologyPath").value) await loadOntology();
      setStatus("Select a JSONL file", "ok");
    }

    async function loadRecord(index) {
      if (index < 0 || index >= state.records.length) return;
      setStatus("Loading record...");
      const payload = await requestJson(`/api/record?index=${index}`);
      state.currentIndex = index;
      el("recordSelect").value = String(index);
      renderRecord(payload.record);
      setStatus(`Record ${index + 1} of ${state.records.length}`, "ok");
    }

    async function saveFile() {
      if (!state.records.length) return;
      const edits = {...state.edits};
      if (state.currentRecord && state.currentIndex >= 0) edits[state.currentIndex] = currentEvents();
      setStatus("Saving...");
      const payload = await requestJson("/api/save", {
        method: "POST",
        body: JSON.stringify({path: el("savePath").value, edits})
      });
      setStatus(`Saved ${payload.count} records to ${payload.path}`, "ok");
    }

    function reportError(err) {
      setStatus(err?.message || String(err), "error");
    }

    el("loadFile").addEventListener("click", () => loadFile().catch(reportError));
    el("loadOntology").addEventListener("click", () => loadOntology().catch(reportError));
    el("recordSelect").addEventListener("change", (event) => loadRecord(Number(event.target.value)).catch(reportError));
    el("prevRecord").addEventListener("click", () => loadRecord(state.currentIndex - 1).catch(reportError));
    el("nextRecord").addEventListener("click", () => loadRecord(state.currentIndex + 1).catch(reportError));
    el("saveFile").addEventListener("click", () => saveFile().catch(reportError));
    el("newEvent").addEventListener("click", () => { try { addEvent(); } catch (err) { reportError(err); } });
    el("newArgument").addEventListener("click", () => { try { addArgument(); } catch (err) { reportError(err); } });
    el("applyEdit").addEventListener("click", () => { try { applyEdit(); } catch (err) { reportError(err); } });
    el("removeAnnotation").addEventListener("click", removeAnnotation);
    el("startChar").addEventListener("input", updatePreview);
    el("endChar").addEventListener("input", updatePreview);
    el("eventLink").addEventListener("change", () => {
      const targetEvent = currentEvents()[Number(el("eventLink").value)];
      el("argumentRole").innerHTML = optionHtml(rolesForEvent(targetEvent?.event_type || ""), el("argumentRole").value);
    });
    el("annotatedText").addEventListener("mouseup", fillOffsetsFromSelection);
    el("annotatedText").addEventListener("keyup", fillOffsetsFromSelection);
    el("annotatedText").addEventListener("click", (event) => {
      const target = event.target.closest(".span-label[data-kind], mark.span");
      if (!target) return;
      const kind = target.dataset.kind;
      const eventIndex = Number(target.dataset.eventIndex);
      const argumentIndex = target.dataset.argumentIndex === "" ? null : Number(target.dataset.argumentIndex);
      selectAnnotation(kind, eventIndex, argumentIndex);
    });
    el("annotationList").addEventListener("click", (event) => {
      const clearButton = event.target.closest("[data-clear-selection]");
      if (clearButton) {
        clearSelection();
        return;
      }
      const target = event.target.closest(".card");
      if (!target) return;
      const kind = target.dataset.kind;
      const eventIndex = Number(target.dataset.eventIndex);
      const argumentIndex = target.dataset.argumentIndex === undefined ? null : Number(target.dataset.argumentIndex);
      selectAnnotation(kind, eventIndex, argumentIndex);
    });

    loadOptions().catch(reportError);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local browser UI for manually fixing event annotations."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
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
    state = AppState(args.ontology)
    server = AnnotationHTTPServer((args.host, args.port), state)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Event annotation UI running at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
