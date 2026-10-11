from conftest import API_KEY_MODEL, FALLBACK_API_KEY, MASTER_KEY, SUBSCRIPTION_MODEL

OAUTH_TOKEN = "sk-ant-oat01-fake"


def _messages(proxy, model: str) -> list[dict]:
    """Sends a /v1/messages call as a subscription client and returns the requests the upstream saw."""
    before = len(proxy.upstream_received())
    resp = proxy.client.post(
        "/v1/messages",
        headers={
            "x-litellm-api-key": f"Bearer {MASTER_KEY}",
            "authorization": f"Bearer {OAUTH_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
            "anthropic-version": "2023-06-01",
        },
        json={"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text
    return [r for r in proxy.upstream_received()[before:] if r["path"].startswith("/v1/messages")]


def test_direct_subscription_call_still_forwards_the_client_bearer(proxy):
    first = _messages(proxy, SUBSCRIPTION_MODEL)[0]["headers"]

    assert first["authorization"] == f"Bearer {OAUTH_TOKEN}"
    assert "oauth-2025-04-20" in first["anthropic-beta"]


def test_fallback_call_uses_the_deployment_api_key(proxy):
    seen = _messages(proxy, SUBSCRIPTION_MODEL)

    assert seen[0]["headers"]["authorization"] == f"Bearer {OAUTH_TOKEN}", "first attempt is the subscription call"
    fallback = seen[-1]["headers"]
    assert fallback["x-api-key"] == FALLBACK_API_KEY
    assert "authorization" not in fallback
    assert "oauth" not in fallback.get("anthropic-beta", "")
    assert "fine-grained-tool-streaming-2025-05-14" in fallback["anthropic-beta"]


def test_client_credentials_are_dropped_on_a_direct_api_key_group_call(proxy):
    seen = _messages(proxy, API_KEY_MODEL)

    assert len(seen) == 1
    assert seen[0]["headers"]["x-api-key"] == FALLBACK_API_KEY
    assert "authorization" not in seen[0]["headers"]
