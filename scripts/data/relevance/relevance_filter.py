import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai import types as genai_types
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llms.llm_client import GeminiContentBlockedError, GeminiLLMClient, LLMClient

load_dotenv(dotenv_path=REPO_ROOT / ".env")

DEFAULT_RELEVANCE_SYSTEM_PROMPT = """<role>
You are a high-recall relevance gate for the extraction of risk events contributing to food crises.
</role>

<goal>
Decide whether the article is likely to discuss an explicit risk event or food crisis itself, thus worth sending to the full extraction pipeline.
</goal>

<event_categories>
The pipeline extracts events in these categories:
- agricultural issues
- conflict and security
- displacement and migration
- economic stress
- environmental issues
- food insecurity
- humanitarian disruption
- political instability
- public health
- weather and natural hazards
</event_categories>

<policy>
- Favor recall over precision. Relevance depends on whether a real, current, concrete instance of a category is reported at all in the article -- NOT on whether it is the article's main subject. A brief side mention of an actual event (e.g. a football report noting "amid extreme rainfall affecting the region") makes the article relevant.
- What disqualifies a mention is not its prominence but its nature: category language used only as a quote, historical anecdote, hypothetical, or rhetorical comparison (e.g. "prices rose like the famine of the 1980s") does NOT describe a real, current occurrence, and does not make the article relevant on its own.
- Opinion/analysis pieces are relevant if they factually reference a concrete, current event or situation in one of the categories, even briefly, and not relevant if they only use category language rhetorically (e.g. domestic politics, culture, sports, entertainment, personal profiles that merely borrow a related term or metaphor).
- If the article is borderline, ambiguous, or only partially visible in the preview, mark it relevant.
- Mark it irrelevant only when the title and preview give no indication of any real, current, concrete instance of any event category -- i.e. all category-related language is rhetorical, historical, hypothetical, or quoted without describing a genuine current occurrence.
- Use only the provided title and article preview.
</policy>
"""

DEFAULT_RELEVANCE_USER_PROMPT = """<context>
<title>
{title}
</title>

<article_preview>
{text}
</article_preview>
</context>

<task>
Return whether this article should proceed to the full food-security risk/event extraction pipeline.
</task>

<decision_rule>
- is_relevant=true if the article reports a real, current, concrete instance of at least one of the following event categories, even as a brief or secondary detail within an article mainly about something else: agricultural issues, conflict and security, displacement and migration, economic stress, environmental issues, food insecurity, humanitarian disruption, political instability, public health, or weather and natural hazards.
- A historical anecdote, hypothetical, or rhetorical comparison that uses category-related language without describing a real current occurrence does not count as evidence (prominence in the article does not matter; whether it describes something real and current does).
- If uncertain whether the article's content describes a real, current in-scope occurrence, return is_relevant=true.
- is_relevant=false only when the article is clearly unrelated to all of the above categories, or when all category-related language is rhetorical, historical, hypothetical, or quoted rather than describing a genuine current occurrence.
</decision_rule>
"""

DEFAULT_RELEVANCE_SYSTEM_PROMPT_FR = """<role>
Vous êtes un filtre de pertinence à haut rappel pour l'extraction d'événements à risque contribuant aux crises alimentaires.
</role>

<goal>
Déterminez si l'article est susceptible de traiter d'un événement à risque explicite ou d'une crise alimentaire elle-même, et mérite donc d'être transmis au pipeline d'extraction complet.
</goal>

<event_categories>
Le pipeline extrait des événements dans les catégories suivantes :
- agricultural issues
- conflict and security
- displacement and migration
- economic stress
- environmental issues
- food insecurity
- humanitarian disruption
- political instability
- public health
- weather and natural hazards
</event_categories>

<policy>
- Privilégiez le rappel plutôt que la précision. La pertinence dépend du fait qu'une instance réelle, actuelle et concrète d'une catégorie soit rapportée dans l'article -- PAS du fait qu'il s'agisse du sujet principal de l'article. Une brève mention incidente d'un événement réel (par exemple, un article sportif notant « en pleines pluies extrêmes touchant la région ») rend l'article pertinent.
- Ce qui disqualifie une mention n'est pas sa proéminence mais sa nature : un langage de catégorie utilisé seulement comme citation, anecdote historique, hypothèse ou comparaison rhétorique (par exemple « les prix ont grimpé comme lors de la famine des années 1980 ») ne décrit PAS une occurrence réelle et actuelle, et ne rend pas l'article pertinent à lui seul.
- Les articles d'opinion/analyse sont pertinents s'ils font référence factuellement à un événement ou une situation concrète et actuelle relevant d'une des catégories, même brièvement, et non pertinents s'ils n'utilisent le langage de catégorie que de manière rhétorique (par exemple politique intérieure, culture, sport, divertissement, profils personnels qui empruntent seulement un terme ou une métaphore connexe).
- Si l'article est limite, ambigu, ou seulement partiellement visible dans l'aperçu, marquez-le comme pertinent.
- Ne marquez l'article comme non pertinent que lorsque le titre et l'aperçu ne donnent aucune indication d'une instance réelle, actuelle et concrète d'une catégorie d'événement -- c'est-à-dire que tout le langage lié à une catégorie est rhétorique, historique, hypothétique, ou cité sans décrire une occurrence réelle actuelle.
- Utilisez uniquement le titre et l'aperçu de l'article fournis.
</policy>
"""

DEFAULT_RELEVANCE_USER_PROMPT_FR = """<context>
<title>
{title}
</title>

<article_preview>
{text}
</article_preview>
</context>

<task>
Indiquez si cet article doit être transmis au pipeline complet d'extraction des risques/événements de sécurité alimentaire.
</task>

<decision_rule>
- is_relevant=true si l'article rapporte une instance réelle, actuelle et concrète d'au moins une des catégories d'événements suivantes, même comme détail bref ou secondaire dans un article traitant principalement d'autre chose : agricultural issues, conflict and security, displacement and migration, economic stress, environmental issues, food insecurity, humanitarian disruption, political instability, public health, or weather and natural hazards.
- Une anecdote historique, une hypothèse, ou une comparaison rhétorique qui utilise un langage lié à une catégorie sans décrire une occurrence réelle actuelle ne compte pas comme preuve (la proéminence dans l'article n'a pas d'importance ; ce qui compte est de savoir si cela décrit quelque chose de réel et actuel).
- En cas d'incertitude quant à savoir si le contenu de l'article décrit une occurrence réelle, actuelle et dans le périmètre, retournez is_relevant=true.
- is_relevant=false uniquement lorsque l'article est clairement sans rapport avec toutes les catégories ci-dessus, ou lorsque tout le langage lié à une catégorie est rhétorique, historique, hypothétique, ou cité plutôt que de décrire une occurrence réelle actuelle.
</decision_rule>
"""


food_insecurity_regex = re.compile(
    r"\b(?:"
    r"food insecurity|acute food insecurity|food security crisis|food crisis|"
    r"hunger crisis|acute hunger|hunger|famine|malnutrition|undernourishment|"
    r"food scarcity|food shortage(?:s)?|lack of food|"
    r"food access|food availability|food affordability|"
    r"food aid|food assistance|emergency food aid|humanitarian food assistance|"
    r"rising food prices|high food prices|food price inflation|"
    r"cereal prices|wheat prices|maize prices|rice prices|"
    r"fertilizer shortage|fertilizer prices|"
    r"crop failure|harvest failure|"
    r"drought|flood(?:ing)?|climate shock(?:s)?|"
    r"conflict|displacement"
    r")\b",
    re.IGNORECASE,
)

exhaustive_food_insecurity_regex = re.compile(
    r"\b(?:"
    r"food insecurity|acute food insecurity|chronic food insecurity|severe food insecurity|"
    r"food security|food security crisis|food crisis|nutrition crisis|"
    r"hunger crisis|acute hunger|chronic hunger|hunger|famine|near famine|"
    r"malnutrition|acute malnutrition|severe acute malnutrition|child malnutrition|"
    r"undernourishment|undernutrition|stunting|wasting|"
    r"food scarcity|food shortage(?:s)?|grain shortage(?:s)?|lack of food|"
    r"food access|food availability|food affordability|food consumption|"
    r"food aid|food assistance|emergency food aid|humanitarian food assistance|"
    r"cash assistance|nutrition assistance|school feeding|"
    r"rising food prices|high food prices|food price inflation|food inflation|"
    r"cereal prices|wheat prices|maize prices|rice prices|bread prices|"
    r"fertilizer shortage|fertilizer prices|input costs|"
    r"crop failure|harvest failure|poor harvest|failed harvest|yield loss(?:es)?|"
    r"livestock deaths|pasture shortage|water shortage(?:s)?|"
    r"drought|dry spell(?:s)?|heatwave(?:s)?|flood(?:ing)?|flash flood(?:s)?|"
    r"storm(?:s)?|cyclone(?:s)?|hurricane(?:s)?|typhoon(?:s)?|landslide(?:s)?|"
    r"climate shock(?:s)?|weather shock(?:s)?|el nino|la nina|"
    r"conflict|armed conflict|violence|insecurity|displacement|forced displacement|"
    r"refugee(?:s)?|internally displaced|idp(?:s)?|"
    r"locust(?:s)?|desert locust(?:s)?|fall armyworm|crop pest(?:s)?|livestock disease(?:s)?|"
    r"cholera outbreak(?:s)?|market disruption(?:s)?|supply chain disruption(?:s)?"
    r")\b",
    re.IGNORECASE,
)


class RelevanceDecision(BaseModel):
    reason: str = Field(default="")
    confidence: float = Field(
        default=0.0, description="Confidence from 0.0 to 1.0 in the relevance decision."
    )
    is_relevant: bool = Field(
        ..., description="True when the article should proceed to full extraction."
    )


def clean_relevance_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    if not isinstance(parsed, dict):
        parsed = {}

    is_relevant = bool(parsed.get("is_relevant", True))
    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    reason = str(parsed.get("reason", "")).strip()
    return {
        "is_relevant": is_relevant,
        "confidence": confidence,
        "reason": reason,
    }


def should_filter_by_relevance(
    decision: dict[str, Any], confidence_threshold: float
) -> bool:
    return (
        not bool(decision.get("is_relevant", True))
        and float(decision.get("confidence", 0.0) or 0.0) >= confidence_threshold
    )


async def classify_article_relevance(
    client: LLMClient,
    title: str,
    text: str,
    record_id: str,
    max_chars: int,
    confidence_threshold: float,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    verbose: bool = False,
    override_settings: dict[str, Any] | None = None,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> dict[str, Any]:
    from scripts.data.generation.gemini_event_gen import (
        log_llm_call,
        response_to_dict,
        truncate_text,
    )

    if system_prompt is None:
        system_prompt = DEFAULT_RELEVANCE_SYSTEM_PROMPT
    if user_prompt_template is None:
        user_prompt_template = DEFAULT_RELEVANCE_USER_PROMPT

    preview_text = truncate_text(text, max_chars)
    prompt = user_prompt_template.format(title=title, text=preview_text)
    log_llm_call(
        enabled=verbose,
        record_id=record_id,
        step="relevance_filter",
        call_type="relevance",
        system_prompt=system_prompt,
        prompt=prompt,
    )
    reasoning_effort = (
        "low"
        if "pro" in client.model_name
        else "minimal" if "3" in client.model_name else None
    )
    response_format = {"reason": str, "is_relevant": bool, "confidence": float}

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = None
            async for candidate in client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                override_settings=(
                    override_settings
                    if override_settings is not None
                    else {"temperature": 0.0, "max_output_tokens": 8192}
                ),
                response_format=response_format,
                add_cot_field=False,
                reasoning_effort=reasoning_effort,
            ):
                response = candidate
                break
            if response is None:
                raise RuntimeError("Gemini returned no relevance response.")

            raw_answer = (
                response_to_dict(response.parsed) if response.parsed else response.text
            )
            log_llm_call(
                enabled=verbose,
                record_id=record_id,
                step="relevance_filter",
                call_type="relevance",
                system_prompt=system_prompt,
                prompt=prompt,
                answer=raw_answer,
            )
            parsed = (
                raw_answer if isinstance(raw_answer, dict) else json.loads(raw_answer)
            )
            decision = clean_relevance_decision(parsed)
            decision_label = "relevant" if decision["is_relevant"] else "irrelevant"
            return {
                "decision": decision_label,
                "is_relevant": decision["is_relevant"],
                "confidence": decision["confidence"],
                "reason": decision["reason"],
                "filtered": should_filter_by_relevance(decision, confidence_threshold),
                "threshold": confidence_threshold,
                "model": client.model_name,
                "max_chars": max_chars,
                "text_chars_used": len(preview_text),
                "metadata": response.metadata,
            }
        except GeminiContentBlockedError as exc:
            # Retrying the same content will hit the same block again, so fail fast
            # instead of burning attempts. Default to relevant (favor recall, per the
            # gate's own policy) rather than silently dropping a blocked article.
            print(
                f"[relevance_filter] record={record_id} blocked by Gemini "
                f"(block_reason={exc.block_reason}); defaulting to relevant."
            )
            return {
                "decision": "relevant",
                "is_relevant": True,
                "confidence": 0.0,
                "reason": f"Gemini blocked the response (block_reason={exc.block_reason}); "
                "defaulted to relevant to favor recall.",
                "filtered": False,
                "threshold": confidence_threshold,
                "model": client.model_name,
                "max_chars": max_chars,
                "text_chars_used": len(preview_text),
                "blocked": True,
                "block_reason": str(exc.block_reason),
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                print(
                    f"[relevance_filter] record={record_id} attempt={attempt}/"
                    f"{max_attempts} failed ({exc}); retrying..."
                )
                await asyncio.sleep(retry_backoff_seconds * attempt)

    assert last_error is not None
    raise last_error


async def process_record(client, record, args):
    source = record.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    title = str(record.get("title") or source.get("title", ""))
    text = str(record.get("text") or source.get("text", ""))
    record_id = str(record.get("id", record.get("url", "")))

    try:
        relevance_info = {}
        if args.use_regex:
            full_text = title + " " + text
            regex_match = bool(food_insecurity_regex.search(full_text))

            if not regex_match:
                relevance_info = {
                    "decision": "irrelevant",
                    "is_relevant": False,
                    "confidence": 1.0,
                    "reason": "Regex filter mismatch",
                    "filtered": True,
                    "threshold": 1.0,
                    "model": "regex",
                }
            else:
                relevance_info = {
                    "decision": "relevant",
                    "is_relevant": True,
                    "confidence": 1.0,
                    "reason": "Regex match",
                    "filtered": False,
                    "threshold": 1.0,
                    "model": "regex",
                }
        elif args.use_keywords:
            text_lower = (title + " " + text).lower()
            keyword_match = any(kw.lower() in text_lower for kw in args.keywords)
            if not keyword_match:
                relevance_info = {
                    "decision": "irrelevant",
                    "is_relevant": False,
                    "confidence": 1.0,
                    "reason": "Keyword filter mismatch",
                    "filtered": True,
                    "threshold": 1.0,
                    "model": "keyword",
                }
            else:
                relevance_info = {
                    "decision": "relevant",
                    "is_relevant": True,
                    "confidence": 1.0,
                    "reason": "Keyword match",
                    "filtered": False,
                    "threshold": 1.0,
                    "model": "keyword",
                }

        if args.use_llm and (not relevance_info.get("filtered", False)):
            relevance_info = await classify_article_relevance(
                client=client,
                title=title,
                text=text,
                record_id=record_id,
                max_chars=args.max_chars,
                confidence_threshold=args.confidence_threshold,
                system_prompt=(
                    DEFAULT_RELEVANCE_SYSTEM_PROMPT_FR if args.french else None
                ),
                user_prompt_template=(
                    DEFAULT_RELEVANCE_USER_PROMPT_FR if args.french else None
                ),
                verbose=args.verbose,
                override_settings=getattr(args, "override_settings", None),
                max_attempts=args.max_attempts,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )

        record["relevance"] = relevance_info
        if "metadata" in relevance_info and relevance_info.get("model") not in (
            None,
            "regex",
            "keyword",
        ):
            record["llm"] = {
                "model": relevance_info["model"],
                "metadata": relevance_info["metadata"],
            }
    except Exception as e:
        record["relevance"] = {"error": str(e), "filtered": False}
        print(f"Error processing record {record_id}: {e}")

    return record


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("id") or record.get("url") or "")


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------

_RELEVANCE_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reason": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "is_relevant": {"type": "BOOLEAN"},
    },
    "required": ["reason", "confidence", "is_relevant"],
}


@dataclass
class _BatchTask:
    key: str
    record: dict[str, Any]
    prompt: str


def _build_batch_request(
    task: "_BatchTask", system_prompt: str, response_schema: dict[str, Any]
) -> dict[str, Any]:
    return {
        "key": task.key,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": task.prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 8192,
                "temperature": 0.0,
                "responseSchema": response_schema,
            },
        },
    }


async def _poll_batch_job(client: GeminiLLMClient, name: str, interval: int) -> Any:
    done = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    job = await asyncio.to_thread(client.client.batches.get, name=name)
    while job.state.name not in done:
        print(f"  batch={name} state={job.state.name}")
        await asyncio.sleep(interval)
        job = await asyncio.to_thread(client.client.batches.get, name=name)
    return job


def _batch_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("text"), str):
        return response["text"]
    for candidate in response.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            text = part.get("text")
            if isinstance(text, str) and not part.get("thought"):
                return text
    raise ValueError("Batch response contains no generated text.")


def _batch_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage_metadata") or response.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": int(
            usage.get("prompt_token_count") or usage.get("promptTokenCount") or 0
        ),
        "completion_tokens": int(
            usage.get("candidates_token_count")
            or usage.get("candidatesTokenCount")
            or 0
        ),
        "cached_tokens": int(
            usage.get("cached_content_token_count")
            or usage.get("cachedContentTokenCount")
            or 0
        ),
        "thoughts_token_count": int(
            usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount") or 0
        ),
    }


def _batch_result_key(line: dict[str, Any]) -> str:
    if line.get("key") is not None:
        return str(line["key"])
    return str((line.get("metadata") or {}).get("key") or "")


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _execute_batch_chunk(
    client: GeminiLLMClient,
    tasks: list["_BatchTask"],
    system_prompt: str,
    model: str,
    output_path: Path,
    chunk_index: int,
    poll_interval: int,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    response_schema = _RELEVANCE_RESPONSE_SCHEMA
    request_path = output_path.with_suffix(
        f".batch.part-{chunk_index:04d}.requests.jsonl"
    )
    result_path = output_path.with_suffix(
        f".batch.part-{chunk_index:04d}.results.jsonl"
    )

    request_path.write_text(
        "".join(
            json.dumps(
                _build_batch_request(t, system_prompt, response_schema),
                ensure_ascii=False,
            )
            + "\n"
            for t in tasks
        ),
        encoding="utf-8",
    )

    uploaded = await asyncio.to_thread(
        client.client.files.upload,
        file=str(request_path),
        config=genai_types.UploadFileConfig(
            display_name=request_path.stem,
            mime_type="jsonl",
        ),
    )
    job = await asyncio.to_thread(
        client.client.batches.create,
        model=model,
        src=uploaded.name,
        config={"display_name": f"{output_path.stem}-part-{chunk_index:04d}"},
    )
    print(
        f"Chunk {chunk_index}: batch job {job.name} submitted ({len(tasks)} records)."
    )
    job = await _poll_batch_job(client, job.name, poll_interval)

    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Chunk {chunk_index}: batch job {job.name} ended with {job.state.name}.")
        return [
            _error_batch_record(
                t.record, f"Batch job ended with {job.state.name}", model
            )
            for t in tasks
        ]

    if not job.dest or not job.dest.file_name:
        return [
            _error_batch_record(
                t.record, "Batch job succeeded without a result file.", model
            )
            for t in tasks
        ]

    data = await asyncio.to_thread(
        client.client.files.download, file=job.dest.file_name
    )
    result_path.write_bytes(data)

    task_by_key = {t.key: t for t in tasks}
    seen: set[str] = set()
    results: list[dict[str, Any]] = []

    with result_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            key = _batch_result_key(line)
            task = task_by_key.get(key)
            if task is None:
                continue
            try:
                response = line.get("response")
                if not isinstance(response, dict):
                    raise ValueError(
                        str(line.get("error") or "Missing batch response.")
                    )
                parsed = json.loads(_batch_response_text(response))
                decision = clean_relevance_decision(parsed)
                decision_label = "relevant" if decision["is_relevant"] else "irrelevant"
                metadata = _batch_response_metadata(response)
                relevance_info = {
                    "decision": decision_label,
                    "is_relevant": decision["is_relevant"],
                    "confidence": decision["confidence"],
                    "reason": decision["reason"],
                    "filtered": should_filter_by_relevance(
                        decision, confidence_threshold
                    ),
                    "threshold": confidence_threshold,
                    "model": model,
                    "metadata": metadata,
                }
                rec = dict(task.record)
                rec["relevance"] = relevance_info
                rec["llm"] = {"model": model, "metadata": metadata}
            except Exception as exc:
                rec = dict(task.record)
                rec["relevance"] = {"error": str(exc), "filtered": False}
            results.append(rec)
            seen.add(key)

    for task in tasks:
        if task.key not in seen:
            results.append(
                _error_batch_record(task.record, "Batch result missing.", model)
            )

    request_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    return results


def _error_batch_record(
    record: dict[str, Any], error: str, model: str
) -> dict[str, Any]:
    rec = dict(record)
    rec["relevance"] = {"error": error, "filtered": False, "model": model}
    return rec


async def run_batch_relevance(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_path: Path,
    append: bool,
) -> list[dict[str, Any]]:
    from scripts.data.generation_v3.costs import aggregate, report

    client = GeminiLLMClient(model_name=args.model, system_prompt=None)
    default_system_prompt = (
        DEFAULT_RELEVANCE_SYSTEM_PROMPT_FR if args.french else DEFAULT_RELEVANCE_SYSTEM_PROMPT
    )
    system_prompt = (
        args.system_prompt
        if hasattr(args, "system_prompt") and args.system_prompt
        else default_system_prompt
    )
    user_prompt_template = (
        DEFAULT_RELEVANCE_USER_PROMPT_FR if args.french else DEFAULT_RELEVANCE_USER_PROMPT
    )

    tasks = [
        _BatchTask(
            key=_record_key(r) or str(i),
            record=r,
            prompt=user_prompt_template.format(
                title=str(r.get("title") or (r.get("source") or {}).get("title", "")),
                text=(str(r.get("text") or (r.get("source") or {}).get("text", "")))[
                    : args.max_chars
                ],
            ),
        )
        for i, r in enumerate(records)
    ]

    chunks = list(enumerate(_chunked(tasks, args.batch_size), start=1))
    all_results: list[dict[str, Any]] = []

    for chunk_index, chunk in chunks:
        print(f"Processing chunk {chunk_index}/{len(chunks)} ({len(chunk)} records)...")
        chunk_results = await _execute_batch_chunk(
            client=client,
            tasks=chunk,
            system_prompt=system_prompt,
            model=args.model,
            output_path=output_path,
            chunk_index=chunk_index,
            poll_interval=args.batch_poll_interval_seconds,
            confidence_threshold=args.confidence_threshold,
        )
        all_results.extend(chunk_results)

        write_mode = "a" if append or chunk_index > 1 else "w"
        with output_path.open(write_mode, encoding="utf-8") as fh:
            for r in chunk_results:
                if args.filter_only and r.get("relevance", {}).get("filtered", False):
                    continue
                fh.write(json.dumps(r) + "\n")

    totals = aggregate(all_results)
    report(output_path.name, totals, batch=True)
    return all_results


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


async def process_file(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(
            f"Output file {output_path} already exists. Use --resume to continue "
            "or --overwrite to replace it."
        )

    # Load already-done keys when resuming. Records that previously errored are
    # dropped from the output so they get retried rather than skipped.
    done_keys: set[str] = set()
    if args.resume and output_path.exists():
        kept_lines = []
        error_count = 0
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("relevance", {}).get("error"):
                    error_count += 1
                    continue
                done_keys.add(_record_key(rec))
                kept_lines.append(line)

        if error_count:
            with output_path.open("w", encoding="utf-8") as f:
                for line in kept_lines:
                    f.write(line + "\n")
            print(
                f"Resuming: {error_count} previously errored records removed from "
                "output and will be retried in this run."
            )

        print(f"Resuming: {len(done_keys)} records already done, skipping.")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if _record_key(rec) not in done_keys:
                    records.append(rec)
            if args.limit and len(records) >= args.limit:
                break

    append = args.resume and output_path.exists()

    if args.batch_api:
        results = await run_batch_relevance(args, records, output_path, append)
    else:
        client = None
        if args.use_llm:
            client = GeminiLLMClient(model_name=args.model, system_prompt=None)

        sem = asyncio.Semaphore(args.concurrency)

        async def bounded_process(record):
            async with sem:
                return await process_record(client, record, args)

        # Optional progress bar if tqdm is available
        try:
            from tqdm.asyncio import tqdm

            tasks = [bounded_process(r) for r in records]
            results = []
            for f in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), desc="Filtering"
            ):
                results.append(await f)
        except ImportError:
            results = await asyncio.gather(*(bounded_process(r) for r in records))

        # Append when resuming, overwrite otherwise
        write_mode = "a" if append else "w"
        with open(output_path, write_mode, encoding="utf-8") as f:
            for r in results:
                if args.filter_only and r.get("relevance", {}).get("filtered", False):
                    continue
                f.write(json.dumps(r) + "\n")

        if args.use_llm:
            from scripts.data.generation_v3.costs import aggregate, report

            totals = aggregate(results)
            report(output_path.name, totals, batch=False)

    # print some summary stats
    total = len(results)
    filtered = sum(1 for r in results if r.get("relevance", {}).get("filtered", False))
    errors = [r for r in results if r.get("relevance", {}).get("error")]
    print(f"Total records: {total}")
    if total:
        print(f"Filtered out: {filtered} ({filtered/total:.2%})")

    if errors:
        error_counts: dict[str, int] = {}
        for r in errors:
            msg = str(r["relevance"]["error"])
            error_counts[msg] = error_counts.get(msg, 0) + 1
        print(
            f"Errors: {len(errors)} ({len(errors)/total:.2%})"
            if total
            else f"Errors: {len(errors)}"
        )
        for msg, count in sorted(error_counts.items(), key=lambda kv: -kv[1]):
            print(f"  [{count}x] {msg}")


DEFAULT_KEYWORDS = [
    "food insecurity",
    "famine",
    "starvation",
    "malnutrition",
    "undernourished",
    "hunger",
    "drought",
    "crop failure",
    "food shortage",
    "starve",
    "food price",
    "food crisis",
    "acute food",
    "food assistance",
    "food aid",
    "food rationing",
    "locust",
    "flood",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run relevance gate on articles standalone."
    )
    parser.add_argument("--input", required=True, type=str, help="Input JSONL file")
    parser.add_argument("--output", required=True, type=str, help="Output JSONL file")
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash", help="Model name"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max characters to use for relevance",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold to filter",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrent requests"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts per record for LLM relevance calls before giving up "
        "(only applies with --use-llm, non-batch mode).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds between retry attempts (multiplied by attempt "
        "number).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Do not write filtered records to output",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="List of keywords to filter by (case-insensitive). Only used when --use-keywords is provided.",
    )
    parser.add_argument(
        "--use-keywords",
        action="store_true",
        help="Filter articles by keyword match (see --keywords) before any other processing. If not provided, keyword filtering is skipped entirely.",
    )
    parser.add_argument(
        "--use-regex",
        action="store_true",
        help="Use the precompiled food_insecurity_regex instead of the keyword list. This is faster.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for relevance classification.",
    )
    parser.add_argument(
        "--french",
        action="store_true",
        help="Use the French translation of the relevance system/user prompts "
        "(event categories and output format stay in English).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records already present in the output file and append new results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file. Required if the output file "
        "already exists and --resume is not set.",
    )
    parser.add_argument(
        "--batch-api",
        action="store_true",
        help="Use Gemini Batch API instead of streaming (~50%% off, but async/slower).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records per batch job chunk (default: 1000).",
    )
    parser.add_argument(
        "--batch-poll-interval-seconds",
        type=int,
        default=30,
        help="Seconds between batch job status polls (default: 30).",
    )
    # parser.add_argument(
    #     "--thinking-level",
    #     type=str,
    #     default="default",
    #     help="Level of thinking to use for relevance classification.",
    # )
    args = parser.parse_args()

    # Import log_llm_call if needed globally, but it's handled inside classify_article_relevance
    asyncio.run(process_file(args))


if __name__ == "__main__":
    main()
