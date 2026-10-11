"""Drop client credentials from calls to the API-key fallback model groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_GROUP_PREFIX = "anthropic-api/"
_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})
_OAUTH_BETA_PREFIX = "oauth-"


class FallbackAuthMiddleware:
    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type: Any = None) -> dict | None:  # NOSONAR
        try:
            is_fallback = _is_fallback_group(kwargs)
        except Exception:  # noqa: BLE001 - an unidentifiable deployment is left untouched
            return None
        if not is_fallback:
            return None

        try:
            return _scrubbed(kwargs)
        except Exception:  # noqa: BLE001 - any scrub failure must still drop the client headers
            return {**kwargs, "headers": {}, "extra_headers": {}, "provider_specific_header": None}


def _scrubbed(kwargs: dict) -> dict | None:
    changes: dict[str, Any] = {}
    for key in ("headers", "extra_headers"):
        cleaned = _scrub(kwargs.get(key))
        if cleaned is not None:
            changes[key] = cleaned

    scoped = kwargs.get("provider_specific_header")
    if isinstance(scoped, Mapping):
        cleaned_entry = _scrub_entry(scoped)
        if cleaned_entry is not None:
            changes["provider_specific_header"] = cleaned_entry
    elif isinstance(scoped, (list, tuple)):
        entries = [_scrub_entry(entry) for entry in scoped]
        if any(entry is not None for entry in entries):
            changes["provider_specific_header"] = [
                new if new is not None else old for new, old in zip(entries, scoped, strict=True)
            ]

    return {**kwargs, **changes} if changes else None


def _is_fallback_group(kwargs: dict) -> bool:
    buckets = [b for b in (kwargs.get("litellm_metadata"), kwargs.get("metadata")) if isinstance(b, Mapping)]
    for field in ("deployment_model_name", "model_group"):
        for metadata in buckets:
            name = metadata.get(field)
            if isinstance(name, str):
                return name.startswith(_GROUP_PREFIX)
    return False


def _scrub_entry(entry: Any) -> dict | None:
    cleaned = _scrub(entry.get("extra_headers")) if isinstance(entry, Mapping) else None
    return None if cleaned is None else {**entry, "extra_headers": cleaned}


def _scrub(headers: Any) -> dict | None:
    if not isinstance(headers, Mapping):
        return None

    cleaned: dict = {}
    for name, value in headers.items():
        lowered = name.lower() if isinstance(name, str) else name
        if lowered in _CREDENTIAL_HEADERS:
            continue
        if lowered == "anthropic-beta" and isinstance(value, str):
            remaining = _without_oauth_tokens(value)
            if remaining:
                cleaned[name] = remaining
            continue
        cleaned[name] = value
    return cleaned if cleaned != headers else None


def _without_oauth_tokens(beta: str) -> str:
    tokens = [token.strip() for token in beta.split(",")]
    if not any(_is_oauth_token(t) for t in tokens):
        return beta
    return ",".join(t for t in tokens if t and not _is_oauth_token(t))


def _is_oauth_token(token: str) -> bool:
    return token.lower().startswith(_OAUTH_BETA_PREFIX)


fallback_auth = FallbackAuthMiddleware()
