"""Turns tool-guard verdicts for a request's new tool results into a LiteLLM guardrail record."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any

from .tool_results import (
    BLOCKED,
    CLEARED,
    FLAGGED,
    MAX_IDENTITY_BYTES,
    MAX_PART_BYTES,
    STRICTNESS,
    TOO_BIG,
    UNCHECKED,
    Identify,
    ToolResult,
    cut,
    identifier,
)

GUARDRAIL_NAME = "tool-guard"
PROVIDER = "llm-tool-guard"
DETECTION_METHOD = "prompt-guard"
SCANNER_ERROR = "scanner-error"
BLOCK = "BLOCK"
FLAG = "FLAG"
SKIP = "SKIP"
GUARDRAIL_ACTION = "FLAGGED"
BLOCKED_ACTION = "BLOCKED"
UNNAMED_TOOL = "tool"
ARROW = "\u2192"
MAX_RESPONSE_BYTES = 64 * 1024
_SLACK_BYTES = 256
_FAILURES = (SCANNER_ERROR, UNCHECKED, TOO_BIG)
_FAILURE_LABELS = {SCANNER_ERROR: "scanner error", UNCHECKED: "unchecked", TOO_BIG: "too big"}
_SCORED = (BLOCKED, FLAGGED, CLEARED)
_VERDICTS = (*_SCORED, *_FAILURES)
_ROW_ACTIONS = {BLOCKED: BLOCK, FLAGGED: FLAG, CLEARED: FLAG}
_SUMMARY_LABELS = {BLOCKED: "blocked", FLAGGED: "suspected injection", CLEARED: "jev-cleared"}


@dataclass(frozen=True)
class Record:
    status: str
    response: dict[str, Any]
    detail: dict[str, Any]


@dataclass(frozen=True)
class _Entry:
    tool_name: str | None
    verdict: str
    score: float | None
    jev: float | None
    tool_input: str | None
    text: str | None


def build(  # noqa: PLR0913 - one argument for each part of the scanner's answer
    items: list,
    call_type_format: str,
    results: list[ToolResult],
    verdicts: dict[str, str],
    scores: dict[str, float],
    *,
    jev: dict[str, float] | None = None,
) -> Record | None:
    """Returns the record for the request's new tool results, or None when it has none."""
    new = [r for r in results if r.new]
    if not new:
        return None
    jev = jev or {}
    entries: list[_Entry] = []
    seen: set[str] = set()
    identify = identifier(items, call_type_format)
    for result in new:
        if result.hash in seen or result.hash not in verdicts:
            continue
        seen.add(result.hash)
        entries.append(_entry(identify, result, verdicts[result.hash], scores.get(result.hash), jev.get(result.hash)))
    scored = [e for e in entries if e.verdict in _SCORED]
    if any(e.verdict == BLOCKED for e in scored):
        status = "guardrail_intervened"
    elif scored:
        status = "guardrail_flagged"
    elif entries:
        status = "guardrail_failed_to_respond"
    else:
        status = "success"
    response = {"summary": _summary(entries, len(new)), "counts": {v: _count(entries, v) for v in _VERDICTS}}
    detail: dict[str, Any] = {"detection_method": DETECTION_METHOD}
    if scored:
        detail.update(_scored_detail(scored))
    return Record(status, *_fit(response, detail, entries))


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
        tracing_detail=record.detail,
    )


@functools.cache
def _guardrail() -> Any:
    from litellm.integrations.custom_guardrail import CustomGuardrail

    return CustomGuardrail(guardrail_name=GUARDRAIL_NAME)


def _entry(identify: Identify, result: ToolResult, verdict: str, score: float | None, jev: float | None) -> _Entry:
    if verdict in (BLOCKED, CLEARED) and jev is None:
        verdict = FLAGGED
    if verdict in _SCORED and score is None:
        verdict = SCANNER_ERROR
    name, _, tool_input = identify(result)
    return _Entry(
        None if name is None else cut(name, MAX_IDENTITY_BYTES),
        verdict,
        score,
        jev,
        tool_input,
        result.text if verdict in _SCORED else None,
    )


def _row(entry: _Entry) -> dict[str, str]:
    """One Match Details row: the scored ones show the tool input and its output, the others only the input."""
    tool_input = entry.tool_input or ""
    if entry.text is None:
        action, category, snippet = SKIP, entry.verdict, tool_input
    else:
        output = cut(entry.text, MAX_PART_BYTES)
        action, category = _ROW_ACTIONS[entry.verdict], _category(entry)
        snippet = f"{tool_input} {ARROW} {output}" if tool_input else output
    return {
        "type": entry.tool_name or UNNAMED_TOOL,
        "detection_method": DETECTION_METHOD,
        "action_taken": action,
        "category": category,
        "snippet": snippet,
    }


def _category(entry: _Entry) -> str:
    if entry.verdict == BLOCKED:
        return f"{entry.score:.2f}, jev {entry.jev:.2f}"
    if entry.verdict == CLEARED:
        return f"{entry.score:.2f}, {CLEARED} {entry.jev:.2f}"
    return f"{entry.score:.2f}"


def _reason(entry: _Entry) -> str:
    name = entry.tool_name or UNNAMED_TOOL
    if entry.verdict == BLOCKED:
        return f"{name} output removed, scored {entry.score:.2f} and jev {entry.jev:.2f}"
    if entry.verdict == CLEARED:
        return f"{name} output scored {entry.score:.2f}, jev cleared it at {entry.jev:.2f}"
    return f"{name} output scored {entry.score:.2f}"


def _scored_detail(scored: list[_Entry]) -> dict[str, Any]:
    top = max(scored, key=lambda e: (STRICTNESS[e.verdict], e.score))
    blocked = any(e.verdict == BLOCKED for e in scored)
    return {
        "confidence_score": top.score,
        "classification": {"category": FLAGGED, "confidence": top.score, "reason": _reason(top)},
        "violation_categories": [FLAGGED],
        "guardrail_action": BLOCKED_ACTION if blocked else GUARDRAIL_ACTION,
    }


def _count(entries: list[_Entry], verdict: str) -> int:
    return sum(e.verdict == verdict for e in entries)


def _summary(entries: list[_Entry], new_results: int) -> str:
    if not entries:
        return f"no suspected injection in {new_results} new tool result{'s' * (new_results != 1)}"
    parts = []
    for verdict in _SCORED:
        scored = [e for e in entries if e.verdict == verdict]
        if scored:
            plural = "s" * (verdict == FLAGGED and len(scored) != 1)
            parts.append(f"{len(scored)} {_SUMMARY_LABELS[verdict]}{plural}{_detail(scored)}")
    for verdict in _FAILURES:
        count = _count(entries, verdict)
        if count:
            plural = "s" * (count != 1) if verdict == SCANNER_ERROR else ""
            parts.append(f"{count} {_FAILURE_LABELS[verdict]}{plural}")
    return ", ".join(parts)


def _detail(scored: list[_Entry]) -> str:
    if len(scored) > 1:
        return f" (max {max(e.score for e in scored):.2f})"
    entry = scored[0]
    parts = [entry.tool_name] if entry.tool_name else []
    parts.append(f"{entry.score:.2f}")
    if entry.verdict != FLAGGED:
        parts.append(f"jev {entry.jev:.2f}")
    return f" ({', '.join(parts)})"


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8", "surrogatepass"))


def _fit(
    response: dict[str, Any], detail: dict[str, Any], entries: list[_Entry]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Builds the leading Match Details rows that fit MAX_RESPONSE_BYTES, and counts the others in `omitted`."""
    if not entries:
        return response, detail
    base = _size({"response": response, "detail": {**detail, "match_details": []}, "omitted": len(entries)})
    budget = MAX_RESPONSE_BYTES - base - _SLACK_BYTES
    kept: list[dict[str, str]] = []
    for entry in entries:
        row = _row(entry)
        budget -= _size(row) + 2
        if budget < 0:
            break
        kept.append(row)
    if len(kept) == len(entries):
        return response, {**detail, "match_details": kept}
    return {**response, "omitted": len(entries) - len(kept)}, {**detail, "match_details": kept}
