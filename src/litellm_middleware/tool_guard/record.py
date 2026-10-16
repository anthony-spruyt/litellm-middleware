"""Turns tool-guard verdicts for a request's new tool results into a LiteLLM guardrail record."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any

from .tool_results import FLAGGED, TOO_BIG, UNCHECKED, Identify, ToolResult, identifier

GUARDRAIL_NAME = "tool-guard"
PROVIDER = "llm-tool-guard"
SCANNER_ERROR = "scanner-error"
MAX_TEXT_BYTES = 32 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_IDENTITY_BYTES = 256
_SLACK_BYTES = 256
_FAILURES = (SCANNER_ERROR, UNCHECKED, TOO_BIG)
_FAILURE_LABELS = {SCANNER_ERROR: "scanner error", UNCHECKED: "unchecked", TOO_BIG: "too big"}


@dataclass(frozen=True)
class Record:
    status: str
    response: dict[str, Any]
    confidence_score: float | None


def build(
    items: list,
    call_type_format: str,
    results: list[ToolResult],
    verdicts: dict[str, str],
    scores: dict[str, float],
) -> Record | None:
    """Returns the record for the request's new tool results, or None when it has none."""
    new = [r for r in results if r.new]
    if not new:
        return None
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    identify = identifier(items, call_type_format)
    for result in new:
        if result.hash in seen or result.hash not in verdicts:
            continue
        seen.add(result.hash)
        entries.append(_entry(identify, result, verdicts[result.hash], scores.get(result.hash)))
    flagged = [e["score"] for e in entries if e["verdict"] == FLAGGED]
    if flagged:
        status = "guardrail_flagged"
    elif entries:
        status = "guardrail_failed_to_respond"
    else:
        status = "success"
    response = {"summary": _summary(entries, len(new)), "results": entries}
    return Record(status, _fit(response), max(flagged, default=None))


def write(data: dict, record: Record, started: float, ended: float) -> None:
    from litellm.types.guardrails import GuardrailEventHooks

    _guardrail().add_standard_logging_guardrail_information_to_request_data(
        guardrail_json_response=record.response,
        request_data=data,
        guardrail_status=record.status,
        start_time=started,
        end_time=ended,
        duration=ended - started,
        guardrail_provider=PROVIDER,
        event_type=GuardrailEventHooks.pre_call,
        tracing_detail=None if record.confidence_score is None else {"confidence_score": record.confidence_score},
    )


@functools.cache
def _guardrail() -> Any:
    from litellm.integrations.custom_guardrail import CustomGuardrail

    return CustomGuardrail(guardrail_name=GUARDRAIL_NAME)


def _entry(identify: Identify, result: ToolResult, verdict: str, score: float | None) -> dict[str, Any]:
    if verdict == FLAGGED and score is None:
        verdict = SCANNER_ERROR
    name, call_id = identify(result)
    return {
        "tool_name": None if name is None else _cut(name, MAX_IDENTITY_BYTES),
        "tool_call_id": None if call_id is None else _cut(call_id, MAX_IDENTITY_BYTES),
        "verdict": verdict,
        "score": score,
        "text": result.text if verdict == FLAGGED else None,
    }


def _summary(entries: list[dict[str, Any]], new_results: int) -> str:
    if not entries:
        return f"no suspected injection in {new_results} new tool result{'s' * (new_results != 1)}"
    parts = []
    flagged = [e for e in entries if e["verdict"] == FLAGGED]
    if flagged:
        parts.append(f"{len(flagged)} suspected injection{'s' * (len(flagged) != 1)}{_detail(flagged)}")
    for verdict in _FAILURES:
        count = sum(e["verdict"] == verdict for e in entries)
        if count:
            plural = "s" * (count != 1) if verdict == SCANNER_ERROR else ""
            parts.append(f"{count} {_FAILURE_LABELS[verdict]}{plural}")
    return ", ".join(parts)


def _detail(flagged: list[dict[str, Any]]) -> str:
    if len(flagged) > 1:
        return f" (max {max(e['score'] for e in flagged):.2f})"
    name, score = flagged[0]["tool_name"], flagged[0]["score"]
    return f" ({name}, {score:.2f})" if name else f" ({score:.2f})"


def _cut(text: str, limit: int) -> str:
    """Keeps the head and tail of text within limit UTF-8 bytes, naming the bytes cut between them."""
    raw = text.encode("utf-8", "replace")
    if len(raw) <= limit:
        return raw.decode("utf-8")
    keep = max(limit - len(f"...[{len(raw)} bytes cut]..."), 0)
    head = raw[: keep // 2].decode("utf-8", "ignore")
    tail = raw[len(raw) - (keep - keep // 2) :].decode("utf-8", "ignore")
    cut = len(raw) - len(head.encode()) - len(tail.encode())
    return f"{head}...[{cut} bytes cut]...{tail}"


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8", "surrogatepass"))


def _fit(response: dict[str, Any]) -> dict[str, Any]:
    """Cuts flagged texts to MAX_TEXT_BYTES, then shrinks them and drops trailing results to fit MAX_RESPONSE_BYTES."""
    entries = response["results"]
    full = {id(e): e["text"] for e in entries if e["text"] is not None}
    _recut(entries, full, MAX_TEXT_BYTES)
    if _size(response) <= MAX_RESPONSE_BYTES:
        return response
    budget = MAX_RESPONSE_BYTES - _size({**response, "results": [], "omitted": len(entries)}) - _SLACK_BYTES
    kept: list[dict[str, Any]] = []
    for entry in entries:
        floor = None if id(entry) not in full else _cut(full[id(entry)], 0)
        used = _size({**entry, "text": floor}) + 2
        if used > budget:
            break
        budget -= used
        kept.append(entry)
    out = {**response, "results": kept}
    if len(kept) < len(entries):
        out["omitted"] = len(entries) - len(kept)
    limit = max(budget // max(sum(id(e) in full for e in kept), 1), 0)
    while True:
        _recut(kept, full, limit)
        if limit <= 0 or _size(out) <= MAX_RESPONSE_BYTES:
            return out
        limit //= 2


def _recut(entries: list[dict[str, Any]], full: dict[int, str], limit: int) -> None:
    for entry in entries:
        if id(entry) in full:
            entry["text"] = _cut(full[id(entry)], limit)
