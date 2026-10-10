"""Asks a scanner which tool results look like prompt injection and marks those as untrusted data."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

from litellm._logging import verbose_proxy_logger

from ..pipeline import MiddlewarePipeline
from .tool_results import ToolResult, find, rewrite

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_TEXT_BYTES = 256 * 1024
_log_warning = MiddlewarePipeline._log_warning
_FRAME_FIELD = "first_message"
_CALL_TYPES = {
    "anthropic_messages": ("messages", "anthropic"),
    "acompletion": ("messages", "chat"),
    "completion": ("messages", "chat"),
    "aresponses": ("input", "responses"),
    "responses": ("input", "responses"),
    "acompact_responses": ("input", "responses"),
    "_aresponses_websocket": (_FRAME_FIELD, "responses"),
    "agenerate_content": ("contents", "gemini"),
    "agenerate_content_stream": ("contents", "gemini"),
}


class ToolGuardMiddleware:
    def __init__(  # noqa: PLR0913 - keyword-only tuning knobs
        self,
        url: str | None,
        *,
        key_aliases: Iterable[str] = (),
        team_ids: Iterable[str] = (),
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/") if url else None
        self.key_aliases = frozenset(key_aliases)
        self.team_ids = frozenset(team_ids)
        self.timeout = timeout
        self.max_text_bytes = max_text_bytes
        self._factory = client_factory or _default_client
        self._client: Any = None

    @property
    def enabled(self) -> bool:
        return self.url is not None

    # async_* names mirror LiteLLM's CustomLogger hooks; callers await them.
    async def async_pre_call_hook(self, user_api_key_dict, _cache, data: dict, call_type: str):  # NOSONAR
        target = _CALL_TYPES.get(call_type)
        if not self.enabled or target is None or not self._enforced(user_api_key_dict):
            return data
        field, fmt = target
        value = data.get(field)
        guarded = await (self._guard_frame(value, fmt) if field == _FRAME_FIELD else self._guard(value, fmt))
        if guarded is not value:
            data[field] = guarded
        return data

    async def _guard(self, items: Any, fmt: str) -> Any:
        if not isinstance(items, list):
            return items
        results = find(items, fmt)
        if not results:
            return items
        oversized = {r.hash for r in results if len(r.text.encode("utf-8", "surrogatepass")) > self.max_text_bytes}
        scannable = [r for r in results if r.hash not in oversized]
        flagged = await self._scan(scannable) if scannable else frozenset()
        hits = [r for r in results if r.hash in flagged or r.hash in oversized]
        marked = [r for r in hits if r.wrappable]
        if hits:
            _log_info("tool guard flagged %d tool result(s), marked %d as untrusted", len(hits), len(marked))
        return rewrite(items, marked) if marked else items

    async def _guard_frame(self, frame: Any, fmt: str) -> Any:
        try:
            event = json.loads(frame) if isinstance(frame, str) else None
        except ValueError:
            return frame
        if not isinstance(event, dict):
            return frame
        items = event.get("input")
        guarded = await self._guard(items, fmt)
        if guarded is items:
            return frame
        return json.dumps({**event, "input": guarded})

    def _enforced(self, user_api_key_dict: Any) -> bool:
        alias = getattr(user_api_key_dict, "key_alias", None)
        team = getattr(user_api_key_dict, "team_id", None)
        return (isinstance(alias, str) and alias in self.key_aliases) or (
            isinstance(team, str) and team in self.team_ids
        )

    async def _scan(self, results: list[ToolResult]) -> frozenset[str]:
        texts: dict[str, str] = {}
        new: dict[str, str] = {}
        for result in results:
            texts.setdefault(result.hash, result.text)
            if result.new:
                new.setdefault(result.hash, result.text)
        known = [h for h in texts if h not in new]
        flagged: set[str] = set()
        try:
            async with asyncio.timeout(self.timeout):
                hits, unknown = await self._post(new, known)
                flagged.update(hits)
                retry = {h: texts[h] for h in unknown if h in texts and h not in new}
                if retry:
                    hits, _ = await self._post(retry, [])
                    flagged.update(hits)
        except Exception as exc:  # noqa: BLE001 - a scanner fault must never fail the request
            _log_warning("tool guard scan error: %s", type(exc).__name__)
        return frozenset(flagged)

    async def _post(self, new: dict[str, str], known: list[str]) -> tuple[list[str], list[str]]:
        payload = {"new": [{"hash": h, "text": t} for h, t in new.items()], "known": known}
        response = await self._connection().post(f"{self.url}/v1/scan", json=payload)
        response.raise_for_status()
        body = response.json()
        return _hashes(body["flagged"], "flagged"), _hashes(body.get("unknown", []), "unknown")

    def _connection(self) -> Any:
        if self._client is None:
            self._client = self._factory()
        return self._client


def _hashes(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(h, str) for h in value):
        raise TypeError(f"{name} is not a list of hashes")
    return value


def _default_client() -> Any:
    import httpx

    return httpx.AsyncClient()


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
    )


tool_guard = tool_guard_from_env()
