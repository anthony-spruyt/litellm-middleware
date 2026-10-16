"""Asks a scanner which tool results look like prompt injection and marks those, and any left unchecked, untrusted."""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Callable, Iterable
from typing import Any

from litellm._logging import verbose_proxy_logger

from ..pipeline import MiddlewarePipeline
from .batches import Batch, batches
from .record import build as build_record
from .record import write as write_record
from .tool_results import FLAGGED, TOO_BIG, UNCHECKED, ToolResult, find, rewrite

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_TEXT_BYTES = 256 * 1024
DEFAULT_MAX_BATCH_BYTES = 4 * 1024 * 1024
_log_warning = MiddlewarePipeline._log_warning
_REJECTED = "This endpoint is not available for this key."
_REJECTIONS = {
    "_aresponses_websocket": "Responses WebSocket mode is not available for this key; use the HTTP Responses API.",
}
# Bodies that never carry tool results into a model; enforced keys get any other unscanned call type rejected.
_TOOL_FREE_CALL_TYPES = frozenset(
    {
        "embedding",
        "aembedding",
        "moderation",
        "amoderation",
        "image_generation",
        "aimage_generation",
        "call_mcp_tool",
    }
)
_CALL_TYPES = {
    "anthropic_messages": ("messages", "anthropic"),
    "acompletion": ("messages", "chat"),
    "completion": ("messages", "chat"),
    "aresponses": ("input", "responses"),
    "responses": ("input", "responses"),
    "acompact_responses": ("input", "responses"),
    "agenerate_content": ("contents", "gemini"),
    "agenerate_content_stream": ("contents", "gemini"),
}


class UnsupportedCallTypeError(ValueError):
    # LiteLLM reads status_code off pre-call exceptions to pick the HTTP status.
    status_code = 400


class ToolGuardMiddleware:
    def __init__(  # noqa: PLR0913 - keyword-only tuning knobs
        self,
        url: str | None,
        *,
        key_aliases: Iterable[str] = (),
        team_ids: Iterable[str] = (),
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES,
        max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/") if url else None
        self.key_aliases = frozenset(key_aliases)
        self.team_ids = frozenset(team_ids)
        self.timeout = timeout
        self.max_text_bytes = max_text_bytes
        self.max_batch_bytes = max_batch_bytes
        self._factory = client_factory or _default_client
        self._client: Any = None

    @property
    def enabled(self) -> bool:
        return self.url is not None

    # async_* names mirror LiteLLM's CustomLogger hooks; callers await them.
    async def async_pre_call_hook(self, user_api_key_dict, _cache, data: dict, call_type: str):  # NOSONAR
        if not self.enabled or not self._enforced(user_api_key_dict):
            return data
        target = _CALL_TYPES.get(call_type)
        if target is None:
            if call_type in _TOOL_FREE_CALL_TYPES:
                return data
            return UnsupportedCallTypeError(_REJECTIONS.get(call_type, _REJECTED))
        field, fmt = target
        value = data.get(field)
        guarded = await self._guard(value, fmt, data)
        if guarded is not value:
            data[field] = guarded
        return data

    async def _guard(self, items: Any, fmt: str, data: dict) -> Any:
        if not isinstance(items, list):
            return items
        results = find(items, fmt)
        if not results:
            return items
        started = time.time()
        oversized = {r.hash for r in results if len(r.text.encode("utf-8", "surrogatepass")) > self.max_text_bytes}
        scannable = [r for r in results if r.hash not in oversized]
        verdicts, scores = await self._scan(scannable) if scannable else ({}, {})
        verdicts.update(dict.fromkeys(oversized, TOO_BIG))
        self._record(data, items, fmt, (results, verdicts, scores), started)
        hits = [r for r in results if r.hash in verdicts]
        marked = [(r, verdicts[r.hash]) for r in hits if r.wrappable]
        if hits:
            flagged = sum(verdicts[r.hash] == FLAGGED for r in hits)
            too_big = sum(verdicts[r.hash] == TOO_BIG for r in hits)
            _log_info(
                "tool guard flagged %d, could not check %d and found %d too big tool result(s), marked %d",
                flagged,
                len(hits) - flagged - too_big,
                too_big,
                len(marked),
            )
        return rewrite(items, marked) if marked else items

    def _enforced(self, user_api_key_dict: Any) -> bool:
        alias = getattr(user_api_key_dict, "key_alias", None)
        team = getattr(user_api_key_dict, "team_id", None)
        return (isinstance(alias, str) and alias in self.key_aliases) or (
            isinstance(team, str) and team in self.team_ids
        )

    def _record(self, data: dict, items: list, fmt: str, outcome: tuple, started: float) -> None:
        try:
            made = build_record(items, fmt, *outcome)
            if made is not None:
                write_record(data, made, started, time.time())
        except Exception as exc:  # noqa: BLE001 - a lost UI record must not fail the request
            _log_warning("tool guard record error: %s", type(exc).__name__)

    async def _scan(self, results: list[ToolResult]) -> tuple[dict[str, str], dict[str, float]]:
        """Returns the verdict to mark each result hash with, and the scanner's score for each model flag.

        Results the scanner cleared are left out of the verdicts.
        """
        texts: dict[str, str] = {}
        for result in reversed(results):
            texts.setdefault(result.hash, result.text)
        new = {r.hash for r in results if r.new}
        known = [h for h in texts if h not in new]
        flagged: set[str] = set()
        scores: dict[str, float] = {}
        unchecked = set(texts)
        try:
            async with asyncio.timeout(self.timeout):
                first = batches(((h, texts[h]) for h in texts if h in new), known, self.max_batch_bytes)
                unknown = set(await self._send(first, flagged, unchecked, scores))
                unchecked -= unknown & new
                retry = [h for h in texts if h in unknown and h not in new]
                follow_up = batches(((h, texts[h]) for h in retry), [], self.max_batch_bytes)
                await self._send(follow_up, flagged, unchecked, scores)
        except Exception as exc:  # noqa: BLE001
            _log_warning("tool guard scan error: %s", type(exc).__name__)
        verdicts = dict.fromkeys(unchecked, UNCHECKED)
        verdicts.update(dict.fromkeys(flagged, FLAGGED))
        return verdicts, scores

    async def _send(
        self, to_send: Iterable[Batch], flagged: set[str], unchecked: set[str], scores: dict[str, float]
    ) -> list[str]:
        """Posts each batch in turn; answered hashes leave unchecked, unknown ones stay until re-sent."""
        unknown: list[str] = []
        for batch in to_send:
            hits, missing, scored = await self._post(batch)
            flagged.update(hits)
            scores.update(scored)
            unknown.extend(missing)
            unchecked.difference_update(set(batch.hashes) - set(missing))
        return unknown

    async def _post(self, batch: Batch) -> tuple[list[str], list[str], dict[str, float]]:
        response = await self._connection().post(
            f"{self.url}/v1/scan", content=batch.body, headers={"content-type": "application/json"}
        )
        response.raise_for_status()
        body = response.json()
        flagged = _hashes(body["flagged"], "flagged")
        return flagged, _hashes(body.get("unknown", []), "unknown"), _scores(body.get("scores"), flagged)

    def _connection(self) -> Any:
        if self._client is None:
            self._client = self._factory()
        return self._client


def _hashes(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(h, str) for h in value):
        raise TypeError(f"{name} is not a list of hashes")
    return value


def _scores(value: Any, flagged: list[str]) -> dict[str, float]:
    """Scores for flagged hashes; a scanner that sends none, or junk, leaves its flags unscored."""
    if not isinstance(value, dict):
        return {}
    return {
        h: float(s)
        for h, s in value.items()
        if h in flagged and isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s)
    }


def _default_client() -> Any:
    import httpx

    # The asyncio.timeout around scanner calls sets the time limit, not httpx's per-call default.
    return httpx.AsyncClient(timeout=None)


def _log_info(message: str, *args: Any) -> None:
    try:
        verbose_proxy_logger.info(message, *args)
    except Exception:  # noqa: BLE001 - logging should never affect request handling
        return


def _csv(name: str) -> frozenset[str]:
    return frozenset(v.strip() for v in os.environ.get(name, "").split(",") if v.strip())


def _positive_from_env(name: str, default: Any, parse: Callable[[str], Any]) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = parse(raw)
    except ValueError:
        value = 0
    if value > 0:
        return value
    _log_warning("%s is not a positive number; using %s", name, default)
    return default


def tool_guard_from_env() -> ToolGuardMiddleware:
    return ToolGuardMiddleware(
        os.environ.get("TOOL_GUARD_URL") or None,
        key_aliases=_csv("TOOL_GUARD_KEY_ALIASES"),
        team_ids=_csv("TOOL_GUARD_TEAM_IDS"),
        timeout=_positive_from_env("TOOL_GUARD_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, float),
        max_text_bytes=_positive_from_env("TOOL_GUARD_MAX_TEXT_BYTES", DEFAULT_MAX_TEXT_BYTES, int),
        max_batch_bytes=_positive_from_env("TOOL_GUARD_MAX_BATCH_BYTES", DEFAULT_MAX_BATCH_BYTES, int),
    )


tool_guard = tool_guard_from_env()
