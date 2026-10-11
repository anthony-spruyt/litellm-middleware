"""Drop client credentials from calls to the API-key fallback model groups."""

from __future__ import annotations

from typing import Any

_GROUP_PREFIX = "anthropic-api/"
_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})
_OAUTH_BETA_PREFIX = "oauth-"


class FallbackAuthMiddleware:
    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type: Any = None) -> dict | None:  # NOSONAR
        if not _is_fallback_group(kwargs):
            return None

        changes: dict[str, Any] = {}
        for key in ("headers", "extra_headers"):
            cleaned = _scrub(kwargs.get(key))
            if cleaned is not None:
                changes[key] = cleaned

        scoped = kwargs.get("provider_specific_header")
        if isinstance(scoped, dict):
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
    for key in ("litellm_metadata", "metadata"):
        metadata = kwargs.get(key)
        group = metadata.get("model_group") if isinstance(metadata, dict) else None
        if isinstance(group, str):
            return group.startswith(_GROUP_PREFIX)
    return False


def _scrub_entry(entry: Any) -> dict | None:
    cleaned = _scrub(entry.get("extra_headers")) if isinstance(entry, dict) else None
    return None if cleaned is None else {**entry, "extra_headers": cleaned}


def _scrub(headers: Any) -> dict | None:
    """Returns a cleaned copy, or None when there is nothing to remove."""
    if not isinstance(headers, dict):
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
    if not any(t.startswith(_OAUTH_BETA_PREFIX) for t in tokens):
        return beta
    return ",".join(t for t in tokens if t and not t.startswith(_OAUTH_BETA_PREFIX))


fallback_auth = FallbackAuthMiddleware()
