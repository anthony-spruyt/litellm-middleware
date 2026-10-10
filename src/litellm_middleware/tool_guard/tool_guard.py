"""Asks a scanner which tool results look like prompt injection and marks those as untrusted data."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterable
from typing import Any

from litellm._logging import verbose_proxy_logger

from ..pipeline import MiddlewarePipeline
from .tool_results import ToolResult, find, rewrite

DEFAULT_TIMEOUT_SECONDS = 3.0
_log_warning = MiddlewarePipeline._log_warning
_CALL_TYPES = {
    "anthropic_messages": ("messages", "anthropic"),
    "acompletion": ("messages", "chat"),
    "completion": ("messages", "chat"),
    "aresponses": ("input", "responses"),
    "responses": ("input", "responses"),
}


class ToolGuardMiddleware:
    def __init__(
        self,
        url: str | None,
        *,
        key_aliases: Iterable[str] = (),
        team_ids: Iterable[str] = (),
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/") if url else None
        self.key_aliases = frozenset(key_aliases)
        self.team_ids = frozenset(team_ids)
        self.timeout = timeout
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
        items = data.get(field)
        if not isinstance(items, list):
            return data
        results = find(items, fmt)
        if not results:
            return data
        flagged = await self._scan(results)
        hits = [r for r in results if r.hash in flagged]
        if hits:
            data[field] = rewrite(items, hits)
            _log_info("tool guard marked %d tool result(s) as untrusted", len(hits))
        return data

    def _enforced(self, user_api_key_dict: Any) -> bool:
        alias = getattr(user_api_key_dict, "key_alias", None)
        team = getattr(user_api_key_dict, "team_id", None)
        return (isinstance(alias, str) and alias in self.key_aliases) or (
            isinstance(team, str) and team in self.team_ids
        )

    async def _scan(self, results: list[ToolResult]) -> frozenset[str]:
        new: dict[str, str] = {}
        for result in results:
            if result.new:
                new.setdefault(result.hash, result.text)
        known = list(dict.fromkeys(r.hash for r in results if not r.new and r.hash not in new))
        payload = {"new": [{"hash": h, "text": t} for h, t in new.items()], "known": known}
        try:
            async with asyncio.timeout(self.timeout):
                response = await self._connection().post(f"{self.url}/v1/scan", json=payload)
            response.raise_for_status()
            flagged = response.json()["flagged"]
            if not isinstance(flagged, list) or not all(isinstance(h, str) for h in flagged):
                raise TypeError("flagged is not a list of hashes")
        except Exception as exc:  # noqa: BLE001 - a scanner fault must never fail the request
            _log_warning("tool guard scan failed open: %s", type(exc).__name__)
            return frozenset()
        return frozenset(flagged)

    def _connection(self) -> Any:
        if self._client is None:
            self._client = self._factory()
        return self._client


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


def _timeout_from_env() -> float:
    raw = os.environ.get("TOOL_GUARD_TIMEOUT_SECONDS")
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value > 0:
        return value
    _log_warning("TOOL_GUARD_TIMEOUT_SECONDS is not a positive number; using %s", DEFAULT_TIMEOUT_SECONDS)
    return DEFAULT_TIMEOUT_SECONDS


def tool_guard_from_env() -> ToolGuardMiddleware:
    return ToolGuardMiddleware(
        os.environ.get("TOOL_GUARD_URL") or None,
        key_aliases=_csv("TOOL_GUARD_KEY_ALIASES"),
        team_ids=_csv("TOOL_GUARD_TEAM_IDS"),
        timeout=_timeout_from_env(),
    )


tool_guard = tool_guard_from_env()
