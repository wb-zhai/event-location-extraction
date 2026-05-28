from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sft_prompt import render_chat  # noqa: E402


DEFAULT_ONTOLOGY = "ontologies/risk-factors/risk.label.description.training.json"
LOGGER = logging.getLogger("sft_infer_ui")


def get_sft_infer_module():
    return importlib.import_module("src.inference.sft_infer")


def resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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


@dataclass
class SamplingConfig:
    max_new_tokens: int
    temperature: float | None
    min_p: float | None
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    max_seq_length: int
    events_only: bool
    omit_offsets: bool
    description: bool


class AppState:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.model_name: str | None = None
        self.adapter_path: str | None = None
        self.load_in_4bit: bool = True
        self.max_seq_length: int = 8192


def _coerce_optional_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _coerce_optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def _build_load_args(payload: dict[str, Any]) -> argparse.Namespace:
    model_name = str(payload.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("model_name is required.")

    adapter_path = str(payload.get("adapter_path") or "").strip() or None
    max_seq_length = int(payload.get("max_seq_length") or 8192)
    load_in_4bit = bool(payload.get("load_in_4bit", True))

    return argparse.Namespace(
        model_name=model_name,
        adapter_path=str(resolve_local_path(adapter_path)) if adapter_path else None,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )


def _load_ontology_payload(
    ontology_file: str,
    *,
    description: bool,
) -> tuple[list[str] | dict[str, str], list[str] | dict[str, str], list[str] | dict[str, str]]:
    sft_infer = get_sft_infer_module()
    return sft_infer._load_ontology_file(
        str(resolve_local_path(ontology_file)), description
    )


def _build_sampling_config(payload: dict[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        max_new_tokens=int(payload.get("max_new_tokens") or 1024),
        temperature=_coerce_optional_float(payload.get("temperature")),
        min_p=_coerce_optional_float(payload.get("min_p")),
        top_k=_coerce_optional_int(payload.get("top_k")),
        top_p=_coerce_optional_float(payload.get("top_p")),
        repetition_penalty=_coerce_optional_float(payload.get("repetition_penalty")),
        max_seq_length=int(payload.get("max_seq_length") or 8192),
        events_only=bool(payload.get("events_only", False)),
        omit_offsets=bool(payload.get("omit_offsets", False)),
        description=bool(payload.get("description", True)),
    )


def run_single_inference(
    *,
    model: Any,
    tokenizer: Any,
    document: str,
    ontology_file: str,
    sampling: SamplingConfig,
) -> dict[str, Any]:
    if not document.strip():
        raise ValueError("document is required.")

    event_labels, argument_roles, location_types = _load_ontology_payload(
        ontology_file,
        description=sampling.description,
    )

    prompt_text = render_chat(
        tokenizer,
        document,
        event_labels,
        argument_roles,
        location_types,
        add_generation_prompt=True,
        events_only=sampling.events_only,
        omit_offsets=sampling.omit_offsets,
    )
    sft_infer = get_sft_infer_module()
    raw_output = sft_infer._generate_prediction_text(
        model,
        tokenizer,
        document=document,
        event_labels=event_labels,
        argument_roles=argument_roles,
        location_types=location_types,
        max_new_tokens=sampling.max_new_tokens,
        temperature=sampling.temperature,
        min_p=sampling.min_p,
        top_k=sampling.top_k,
        top_p=sampling.top_p,
        repetition_penalty=sampling.repetition_penalty,
        max_input_length=sampling.max_seq_length,
        events_only=sampling.events_only,
        omit_offsets=sampling.omit_offsets,
    )
    parsed = sft_infer._parse_prediction_text(raw_output)
    normalized = sft_infer._normalize_prediction(document, parsed)

    return {
        "prompt": prompt_text,
        "raw_output": raw_output,
        "parsed_prediction": parsed,
        "normalized_prediction": normalized,
    }


class SftInferHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], state: AppState) -> None:
        self.state = state
        super().__init__(server_address, SftInferHandler)

    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class SftInferHandler(BaseHTTPRequestHandler):
    server: SftInferHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/state":
            self.send_json(
                {
                    "loaded_model": {
                        "model_name": self.server.state.model_name,
                        "adapter_path": self.server.state.adapter_path,
                        "load_in_4bit": self.server.state.load_in_4bit,
                        "max_seq_length": self.server.state.max_seq_length,
                    }
                    if self.server.state.model is not None
                    else None,
                    "ontology_files": list_ontology_files(),
                    "default_ontology": DEFAULT_ONTOLOGY,
                }
            )
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/load-model":
            self.handle_load_model()
            return
        if self.path == "/api/infer":
            self.handle_infer()
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

    def handle_load_model(self) -> None:
        try:
            payload = self.read_json()
            args = _build_load_args(payload)
            sft_infer = get_sft_infer_module()
            model, tokenizer = sft_infer.load_inference_model(args)
            self.server.state.model = model
            self.server.state.tokenizer = tokenizer
            self.server.state.model_name = args.model_name
            self.server.state.adapter_path = args.adapter_path
            self.server.state.load_in_4bit = args.load_in_4bit
            self.server.state.max_seq_length = args.max_seq_length
            self.send_json(
                {
                    "ok": True,
                    "loaded_model": {
                        "model_name": args.model_name,
                        "adapter_path": args.adapter_path,
                        "load_in_4bit": args.load_in_4bit,
                        "max_seq_length": args.max_seq_length,
                    },
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_infer(self) -> None:
        try:
            payload = self.read_json()
            if self.server.state.model is None or self.server.state.tokenizer is None:
                raise ValueError("No model is loaded.")

            ontology_file = str(payload.get("ontology_file") or DEFAULT_ONTOLOGY).strip()
            sampling = _build_sampling_config(payload)
            document = str(payload.get("document") or "")
            result = run_single_inference(
                model=self.server.state.model,
                tokenizer=self.server.state.tokenizer,
                document=document,
                ontology_file=ontology_file,
                sampling=sampling,
            )
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SFT Inference UI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f4ef;
      --panel: #fffdf7;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d8d3c7;
      --accent: #1d4ed8;
      --danger: #b42318;
      --shadow: 0 10px 28px rgba(31, 41, 55, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.08), transparent 24%),
        linear-gradient(180deg, #f9f8f2 0%, var(--bg) 100%);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      background: rgba(255, 253, 247, 0.92);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 760; }
    .status { font-size: 12px; color: var(--muted); }
    .status.error { color: var(--danger); }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
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
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    label {
      display: block;
      margin-bottom: 5px;
      font-size: 12px;
      font-weight: 700;
      color: #475467;
    }
    input, select, textarea, button {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid #c5c0b3;
      border-radius: 10px;
      padding: 8px 10px;
      background: #fffdfa;
      color: var(--ink);
    }
    textarea {
      min-height: 200px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      line-height: 1.5;
    }
    button {
      border: 1px solid #b7bfd1;
      background: #ffffff;
      color: var(--ink);
      border-radius: 10px;
      min-height: 38px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    .meta {
      font-size: 12px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .results {
      display: grid;
      gap: 16px;
    }
    .result-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    pre {
      margin: 0;
      padding: 14px;
      background: #fcfbf7;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 360px;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    @media (max-width: 1100px) {
      main, .result-grid, .row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>SFT Inference UI</h1>
    <div id="status" class="status">Ready</div>
  </header>
  <main>
    <aside class="panel">
      <div class="section">
        <div class="field">
          <label for="modelName">Base model name</label>
          <input id="modelName" placeholder="Hugging Face model name, e.g. unsloth/Qwen3-4B">
        </div>
        <div class="field">
          <label for="adapterPath">Adapter path</label>
          <input id="adapterPath" placeholder="Optional LoRA adapter">
        </div>
        <div class="row">
          <div class="field">
            <label for="maxSeqLength">Max seq length</label>
            <input id="maxSeqLength" type="number" value="8192">
          </div>
          <div class="field">
            <label for="loadIn4Bit">Load mode</label>
            <select id="loadIn4Bit">
              <option value="true">4-bit</option>
              <option value="false">16-bit</option>
            </select>
          </div>
        </div>
        <button id="loadModelButton" class="primary" type="button">Load model</button>
        <div id="loadedModelMeta" class="meta" style="margin-top:10px;"></div>
      </div>

      <div class="section">
        <div class="field">
          <label for="ontologySelect">Ontology</label>
          <select id="ontologySelect"></select>
        </div>
        <div class="field">
          <label for="ontologyPath">Ontology path override</label>
          <input id="ontologyPath">
        </div>
        <div class="row">
          <div class="field">
            <label for="description">Descriptions in prompt</label>
            <select id="description">
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>
          <div class="field">
            <label for="eventsOnly">Events only</label>
            <select id="eventsOnly">
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>
        </div>
        <div class="field">
          <label for="omitOffsets">Offset strategy</label>
          <select id="omitOffsets">
            <option value="true">Omit offsets in generation</option>
            <option value="false">Ask for explicit offsets</option>
          </select>
        </div>
      </div>

      <div class="section">
        <div class="row">
          <div class="field">
            <label for="maxNewTokens">Max new tokens</label>
            <input id="maxNewTokens" type="number" value="1024">
          </div>
          <div class="field">
            <label for="temperature">Temperature</label>
            <input id="temperature" placeholder="e.g. 0.0">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="minP">Min-p</label>
            <input id="minP" placeholder="Optional">
          </div>
          <div class="field">
            <label for="topK">Top-k</label>
            <input id="topK" placeholder="Optional">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="topP">Top-p</label>
            <input id="topP" placeholder="Optional">
          </div>
          <div class="field">
            <label for="repetitionPenalty">Repetition penalty</label>
            <input id="repetitionPenalty" placeholder="Optional">
          </div>
        </div>
      </div>
    </aside>

    <section class="results">
      <section class="panel">
        <div class="section">
          <div class="field">
            <label for="document">Document</label>
            <textarea id="document" placeholder="Paste the text you want to run through the SFT model"></textarea>
          </div>
          <button id="runInferenceButton" class="primary" type="button">Run inference</button>
        </div>
      </section>

      <section class="panel">
        <div class="section">
          <strong>Normalized prediction</strong>
        </div>
        <div class="section">
          <pre id="normalizedOutput">{}</pre>
        </div>
      </section>

      <section class="result-grid">
        <section class="panel">
          <div class="section"><strong>Parsed JSON</strong></div>
          <div class="section"><pre id="parsedOutput">{}</pre></div>
        </section>
        <section class="panel">
          <div class="section"><strong>Raw model output</strong></div>
          <div class="section"><pre id="rawOutput"></pre></div>
        </section>
      </section>

      <section class="panel">
        <div class="section"><strong>Rendered prompt</strong></div>
        <div class="section"><pre id="promptOutput"></pre></div>
      </section>
    </section>
  </main>

  <script>
    const el = (id) => document.getElementById(id);

    function setStatus(message, isError = false) {
      el("status").textContent = message;
      el("status").classList.toggle("error", isError);
    }

    function boolValue(id) {
      return el(id).value === "true";
    }

    async function fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      const payload = await response.json();
      if (!response.ok || payload.error) {
        throw new Error(payload.error || `Request failed with ${response.status}`);
      }
      return payload;
    }

    function currentOntologyPath() {
      return el("ontologyPath").value.trim() || el("ontologySelect").value;
    }

    async function init() {
      const payload = await fetchJson("/api/state");
      const ontologyFiles = payload.ontology_files || [];
      el("ontologySelect").innerHTML = ontologyFiles.map((path) => (
        `<option value="${path}">${path}</option>`
      )).join("");
      el("ontologySelect").value = payload.default_ontology;
      el("ontologyPath").value = payload.default_ontology;
      el("ontologySelect").addEventListener("change", () => {
        el("ontologyPath").value = el("ontologySelect").value;
      });
      if (payload.loaded_model) {
        renderLoadedModel(payload.loaded_model);
      }
    }

    function renderLoadedModel(model) {
      el("loadedModelMeta").textContent = [
        `model ${model.model_name}`,
        model.adapter_path ? `adapter ${model.adapter_path}` : "no adapter",
        model.load_in_4bit ? "4-bit" : "16-bit",
        `max_seq_length ${model.max_seq_length}`
      ].join(" · ");
    }

    async function loadModel() {
      try {
        setStatus("Loading model...");
        const payload = await fetchJson("/api/load-model", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({
            model_name: el("modelName").value,
            adapter_path: el("adapterPath").value,
            max_seq_length: Number(el("maxSeqLength").value || 8192),
            load_in_4bit: boolValue("loadIn4Bit")
          })
        });
        renderLoadedModel(payload.loaded_model);
        setStatus("Model loaded");
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function runInference() {
      try {
        setStatus("Running inference...");
        const payload = await fetchJson("/api/infer", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({
            document: el("document").value,
            ontology_file: currentOntologyPath(),
            description: boolValue("description"),
            events_only: boolValue("eventsOnly"),
            omit_offsets: boolValue("omitOffsets"),
            max_seq_length: Number(el("maxSeqLength").value || 8192),
            max_new_tokens: Number(el("maxNewTokens").value || 1024),
            temperature: el("temperature").value.trim(),
            min_p: el("minP").value.trim(),
            top_k: el("topK").value.trim(),
            top_p: el("topP").value.trim(),
            repetition_penalty: el("repetitionPenalty").value.trim()
          })
        });
        el("promptOutput").textContent = payload.prompt || "";
        el("rawOutput").textContent = payload.raw_output || "";
        el("parsedOutput").textContent = JSON.stringify(payload.parsed_prediction || {}, null, 2);
        el("normalizedOutput").textContent = JSON.stringify(payload.normalized_prediction || {}, null, 2);
        setStatus("Inference complete");
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    el("loadModelButton").addEventListener("click", loadModel);
    el("runInferenceButton").addEventListener("click", runInference);
    init().catch((error) => setStatus(error.message, true));
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    state = AppState()
    server = SftInferHTTPServer((args.host, args.port), state)
    print(f"SFT inference UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
