"""Test-only custom auth: gives one key an alias without a database; every other key falls through to LiteLLM."""

from litellm.proxy._types import UserAPIKeyAuth

GUARDED_KEY = "it-tool-guard-key"
GUARDED_ALIAS = "it-tool-guard"


async def user_api_key_auth(request, api_key: str) -> UserAPIKeyAuth:
    if api_key.removeprefix("Bearer ").strip() != GUARDED_KEY:
        # custom_auth_settings.mode "auto" hands any non-ProxyException to LiteLLM's own auth.
        raise ValueError("not the guarded test key")
    return UserAPIKeyAuth(api_key=GUARDED_KEY, key_alias=GUARDED_ALIAS)
