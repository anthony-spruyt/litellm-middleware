"""Drop client credentials from calls to the API-key fallback model groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..pipeline import MiddlewarePipeline

_log_warning = MiddlewarePipeline._log_warning

_GROUP_PREFIX = "anthropic-api/"
_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})
_OAUTH_BETA_PREFIX = "oauth-"
_LITELLM_METADATA_CALLS = frozenset({"anthropic_messages", "aanthropic_messages", "aresponses", "responses"})
_METADATA_CALLS = frozenset({"acompletion", "completion", "atext_completion", "text_completion"})


class FallbackAuthMiddleware:
    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type: Any = None) -> dict | None:  # NOSONAR
        try:
            is_fallback = _is_fallback_group(kwargs, call_type)
        except Exception as exc:  # noqa: BLE001
            _log_warning("fallback auth could not identify the deployment: %s", type(exc).__name__)
            return None
        if not is_fallback:
            return None

        try:
            return _scrubbed(kwargs)
        except Exception as exc:  # noqa: BLE001 - any scrub failure must still drop the client headers
            _log_warning("fallback auth scrub failed, dropping client headers: %s", type(exc).__name__)
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


def _is_fallback_group(kwargs: dict, call_type: Any) -> bool:
    call = getattr(call_type, "value", call_type)
    if call in _LITELLM_METADATA_CALLS:
        bucket = kwargs.get("litellm_metadata")
    elif call in _METADATA_CALLS:
        bucket = kwargs.get("metadata")
    else:
        return False
    if not isinstance(bucket, Mapping):
        return False
    name = bucket.get("deployment_model_name")
    return isinstance(name, str) and name.startswith(_GROUP_PREFIX)


def _scrub_entry(entry: Any) -> dict | None:
    cleaned = _scrub(entry.get("extra_headers")) if isinstance(entry, Mapping) else None
    return None if cleaned is None else {**entry, "extra_headers": cleaned}


def _scrub(headers: Any) -> dict | None:
    if not isinstance(headers, Mapping):
        return None

    cleaned: dict = {}
    for name, value in headers.items():
        lowered = _lowered(name)
        if lowered in _CREDENTIAL_HEADERS:
            continue
        if lowered == "anthropic-beta" and isinstance(value, (str, bytes, list, tuple)):
            remaining = _without_oauth_tokens(value)
            if remaining:
                cleaned[name] = remaining
            continue
        cleaned[name] = value
    return cleaned if cleaned != headers else None


def _lowered(name: Any) -> Any:
    if isinstance(name, bytes):
        name = name.decode("latin-1")
    return name.lower() if isinstance(name, str) else name


def _without_oauth_tokens(beta: str | bytes | list | tuple) -> str | bytes | list | tuple:
    if isinstance(beta, (str, bytes)):
        return _without_oauth_tokens_scalar(beta)
    parts = [_without_oauth_tokens_scalar(part) if isinstance(part, (str, bytes)) else part for part in beta]
    return type(beta)(part for part in parts if part)


def _without_oauth_tokens_scalar(beta: str | bytes) -> str | bytes:
    if isinstance(beta, bytes):
        return _without_oauth_tokens_str(beta.decode("latin-1")).encode("latin-1")
    return _without_oauth_tokens_str(beta)


def _without_oauth_tokens_str(beta: str) -> str:
    tokens = [token.strip() for token in beta.split(",")]
    if not any(_is_oauth_token(t) for t in tokens):
        return beta
    return ",".join(t for t in tokens if t and not _is_oauth_token(t))


def _is_oauth_token(token: str) -> bool:
    return token.lower().startswith(_OAUTH_BETA_PREFIX)


fallback_auth = FallbackAuthMiddleware()
