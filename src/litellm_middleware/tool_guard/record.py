"""Turns tool-guard verdicts for a request's new tool results into a LiteLLM guardrail record."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any

from .tool_results import FLAGGED, TOO_BIG, UNCHECKED, Identify, ToolResult, identifier

GUARDRAIL_NAME = "tool-guard"
PROVIDER = "llm-tool-guard"
DETECTION_METHOD = "prompt-guard"
SCANNER_ERROR = "scanner-error"
FLAG = "FLAG"
SKIP = "SKIP"
GUARDRAIL_ACTION = "FLAGGED"
UNNAMED_TOOL = "tool"
ARROW = "\u2192"
MAX_PART_BYTES = 300
MAX_RESPONSE_BYTES = 64 * 1024
MAX_IDENTITY_BYTES = 256
_SLACK_BYTES = 256
_FAILURES = (SCANNER_ERROR, UNCHECKED, TOO_BIG)
_FAILURE_LABELS = {SCANNER_ERROR: "scanner error", UNCHECKED: "unchecked", TOO_BIG: "too big"}
_VERDICTS = (FLAGGED, *_FAILURES)


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
    tool_input: str | None
    text: str | None


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
    entries: list[_Entry] = []
    seen: set[str] = set()
    identify = identifier(items, call_type_format)
    for result in new:
        if result.hash in seen or result.hash not in verdicts:
            continue
        seen.add(result.hash)
        entries.append(_entry(identify, result, verdicts[result.hash], scores.get(result.hash)))
    flagged = [e for e in entries if e.verdict == FLAGGED]
    if flagged:
        status = "guardrail_flagged"
    elif entries:
        status = "guardrail_failed_to_respond"
    else:
        status = "success"
    response = {"summary": _summary(entries, len(new)), "counts": {v: _count(entries, v) for v in _VERDICTS}}
    detail: dict[str, Any] = {"detection_method": DETECTION_METHOD}
    if entries:
        detail["match_details"] = [_row(e) for e in entries]
    if flagged:
        detail.update(_flagged_detail(flagged))
    return Record(status, *_fit(response, detail))


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


def _entry(identify: Identify, result: ToolResult, verdict: str, score: float | None) -> _Entry:
    if verdict == FLAGGED and score is None:
        verdict = SCANNER_ERROR
    name, _, tool_input = identify(result)
    return _Entry(
        None if name is None else _cut(name, MAX_IDENTITY_BYTES),
        verdict,
        score,
        tool_input,
        result.text if verdict == FLAGGED else None,
    )


def _row(entry: _Entry) -> dict[str, str]:
    """One Match Details row: the flagged ones show the tool input and its output, the others only the input."""
    tool_input = "" if entry.tool_input is None else _cut(entry.tool_input, MAX_PART_BYTES)
    if entry.text is None or entry.score is None:
        action, category, snippet = SKIP, entry.verdict, tool_input
    else:
        output = _cut(entry.text, MAX_PART_BYTES)
        action, category = FLAG, f"{entry.score:.2f}"
        snippet = f"{tool_input} {ARROW} {output}" if tool_input else output
    return {
        "type": entry.tool_name or UNNAMED_TOOL,
        "detection_method": DETECTION_METHOD,
        "action_taken": action,
        "category": category,
        "snippet": snippet,
    }


def _flagged_detail(flagged: list[_Entry]) -> dict[str, Any]:
    top = max(flagged, key=_score)
    return {
        "confidence_score": top.score,
        "classification": {
            "category": FLAGGED,
            "confidence": top.score,
            "reason": f"{top.tool_name or UNNAMED_TOOL} output scored {top.score:.2f}",
        },
        "violation_categories": [FLAGGED],
        "guardrail_action": GUARDRAIL_ACTION,
    }


def _score(entry: _Entry) -> float:
    return entry.score if entry.score is not None else 0.0


def _count(entries: list[_Entry], verdict: str) -> int:
    return sum(e.verdict == verdict for e in entries)


def _summary(entries: list[_Entry], new_results: int) -> str:
    if not entries:
        return f"no suspected injection in {new_results} new tool result{'s' * (new_results != 1)}"
    parts = []
    flagged = [e for e in entries if e.verdict == FLAGGED]
    if flagged:
        parts.append(f"{len(flagged)} suspected injection{'s' * (len(flagged) != 1)}{_detail(flagged)}")
    for verdict in _FAILURES:
        count = _count(entries, verdict)
        if count:
            plural = "s" * (count != 1) if verdict == SCANNER_ERROR else ""
            parts.append(f"{count} {_FAILURE_LABELS[verdict]}{plural}")
    return ", ".join(parts)


def _detail(flagged: list[_Entry]) -> str:
    if len(flagged) > 1:
        return f" (max {max(_score(e) for e in flagged):.2f})"
    name, score = flagged[0].tool_name, _score(flagged[0])
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


def _fit(response: dict[str, Any], detail: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keeps the leading Match Details rows that fit MAX_RESPONSE_BYTES and counts the others in `omitted`."""
    rows = detail.get("match_details", [])
    base = _size({"response": response, "detail": {**detail, "match_details": []}, "omitted": len(rows)})
    budget = MAX_RESPONSE_BYTES - base - _SLACK_BYTES
    kept: list[dict[str, str]] = []
    for row in rows:
        budget -= _size(row) + 2
        if budget < 0:
            break
        kept.append(row)
    if len(kept) == len(rows):
        return response, detail
    return {**response, "omitted": len(rows) - len(kept)}, {**detail, "match_details": kept}
