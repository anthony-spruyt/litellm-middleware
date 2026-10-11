import json

import pytest
from conftest import ALIASED_API_KEY_MODEL, API_KEY_MODEL, FALLBACK_API_KEY, MASTER_KEY, SUBSCRIPTION_MODEL

OAUTH_TOKEN = "sk-ant-oat01-fake"


CALLS = {
    "messages": ("/v1/messages", False, {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
    "messages-stream": ("/v1/messages", True, {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
    "chat": ("/v1/chat/completions", False, {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
    "chat-stream": ("/v1/chat/completions", True, {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
    "responses": ("/v1/responses", False, {"max_output_tokens": 16, "input": "hi"}),
    "responses-stream": ("/v1/responses", True, {"max_output_tokens": 16, "input": "hi"}),
}
endpoints = pytest.mark.parametrize("call", list(CALLS.values()), ids=list(CALLS))


def _send(proxy, call, model: str) -> list[dict]:
    """Sends the call as a subscription client and returns the requests the upstream saw."""
    path, stream, body = call
    before = len(proxy.upstream_received())
    with proxy.client.stream(
        "POST",
        path,
        headers={
            "x-litellm-api-key": f"Bearer {MASTER_KEY}",
            "authorization": f"Bearer {OAUTH_TOKEN}",
            "anthropic-beta": "oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
            "anthropic-version": "2023-06-01",
        },
        json={"model": model, "stream": stream, **body},
    ) as resp:
        resp.read()
    assert resp.status_code == 200, resp.text
    return [r for r in proxy.upstream_received()[before:] if r["path"].startswith("/v1/messages")]


@endpoints
def test_direct_subscription_call_still_forwards_the_client_bearer(proxy, call):
    first = _send(proxy, call, SUBSCRIPTION_MODEL)[0]["headers"]

    assert first["authorization"] == f"Bearer {OAUTH_TOKEN}"
    assert "oauth-2025-04-20" in first["anthropic-beta"]


@endpoints
def test_fallback_call_uses_the_deployment_api_key(proxy, call):
    seen = _send(proxy, call, SUBSCRIPTION_MODEL)

    assert seen[0]["headers"]["authorization"] == f"Bearer {OAUTH_TOKEN}", "first attempt is the subscription call"
    fallback = seen[-1]["headers"]
    assert fallback["x-api-key"] == FALLBACK_API_KEY
    assert "authorization" not in fallback
    assert "oauth" not in fallback.get("anthropic-beta", "")


def test_fallback_call_keeps_the_other_beta_tokens(proxy):
    fallback = _send(proxy, CALLS["messages"], SUBSCRIPTION_MODEL)[-1]["headers"]

    assert "fine-grained-tool-streaming-2025-05-14" in fallback["anthropic-beta"]


@endpoints
def test_client_credentials_are_dropped_on_a_direct_api_key_group_call(proxy, call):
    seen = _send(proxy, call, API_KEY_MODEL)

    assert len(seen) == 1
    assert seen[0]["headers"]["x-api-key"] == FALLBACK_API_KEY
    assert "authorization" not in seen[0]["headers"]
    assert "oauth" not in seen[0]["headers"].get("anthropic-beta", "")


@endpoints
def test_client_credentials_are_dropped_when_an_alias_resolves_to_an_api_key_group(proxy, call):
    seen = _send(proxy, call, ALIASED_API_KEY_MODEL)

    assert len(seen) == 1
    assert seen[0]["headers"]["x-api-key"] == FALLBACK_API_KEY
    assert "authorization" not in seen[0]["headers"]
    assert "oauth" not in seen[0]["headers"].get("anthropic-beta", "")


_SYNC_RESPONSES = """
import json
import litellm

if __WITH_PIPELINE__:
    from litellm_middleware.pipeline_plugin import pipeline_middleware

    litellm.callbacks = [pipeline_middleware]

resp = litellm.responses(
    model="openai/gpt-it",
    input="hi",
    api_base="http://127.0.0.1:8099/v1",
    api_key="it",
    extra_headers={"authorization": "Bearer __TOKEN__", "x-keep": "1"},
)
print(json.dumps({"status": resp.status}))
"""


def _sync_responses(proxy, *, pipeline: bool) -> dict:
    """Runs litellm.responses in the container and returns the request the upstream saw."""
    before = len(proxy.upstream_received())
    code = _SYNC_RESPONSES.replace("__WITH_PIPELINE__", repr(pipeline)).replace("__TOKEN__", OAUTH_TOKEN)
    assert json.loads(proxy.python(code))["status"] == "completed"
    seen = [r for r in proxy.upstream_received()[before:] if r["path"].startswith("/v1/responses")]
    assert len(seen) == 1
    return seen[0]


def test_sync_responses_through_the_pipeline_matches_a_call_without_it(proxy):
    without = _sync_responses(proxy, pipeline=False)
    with_pipeline = _sync_responses(proxy, pipeline=True)

    assert with_pipeline["body"] == without["body"]
    assert with_pipeline["headers"] == without["headers"]
    assert with_pipeline["headers"]["x-keep"] == "1"
