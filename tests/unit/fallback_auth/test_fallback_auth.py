import importlib
import sys

import pytest


@pytest.fixture
def module():
    sys.modules.pop("litellm_middleware.fallback_auth.fallback_auth", None)
    return importlib.import_module("litellm_middleware.fallback_auth.fallback_auth")


async def _hook(module, kwargs):
    return await module.fallback_auth.async_pre_call_deployment_hook(kwargs, "anthropic_messages")


def _kwargs(group="anthropic-api/claude-x", **extra):
    return {"model": "anthropic/claude-x", "litellm_metadata": {"model_group": group}, **extra}


async def test_drops_client_credentials_from_headers_on_api_fallback_group(module):
    kwargs = _kwargs(
        headers={
            "Authorization": "Bearer sk-ant-oat01-x",
            "X-Api-Key": "sk-ant-api03-client",
            "anthropic-version": "2023-06-01",
        }
    )

    out = await _hook(module, kwargs)

    assert out["headers"] == {"anthropic-version": "2023-06-01"}


async def test_drops_oauth_beta_tokens_and_keeps_others(module):
    kwargs = _kwargs(
        headers={"anthropic-beta": "claude-code-20250219, oauth-2025-04-20 ,interleaved-thinking-2025-05-14"}
    )

    out = await _hook(module, kwargs)

    assert out["headers"] == {"anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14"}


async def test_drops_beta_header_when_only_oauth_tokens_remain(module):
    out = await _hook(module, _kwargs(headers={"Anthropic-Beta": "oauth-2025-04-20", "x-other": "1"}))

    assert out["headers"] == {"x-other": "1"}


async def test_scrubs_extra_headers_and_provider_specific_headers(module):
    kwargs = _kwargs(
        extra_headers={"authorization": "Bearer sk-ant-oat01-x", "x-keep": "1"},
        provider_specific_header=[
            {"custom_llm_provider": "anthropic,bedrock", "extra_headers": {"anthropic-beta": "oauth-2025-04-20,foo"}},
            {"custom_llm_provider": "anthropic", "extra_headers": {"authorization": "Bearer sk-ant-oat01-x"}},
        ],
    )

    out = await _hook(module, kwargs)

    assert out["extra_headers"] == {"x-keep": "1"}
    assert out["provider_specific_header"] == [
        {"custom_llm_provider": "anthropic,bedrock", "extra_headers": {"anthropic-beta": "foo"}},
        {"custom_llm_provider": "anthropic", "extra_headers": {}},
    ]


async def test_scrubs_single_provider_specific_header_dict(module):
    kwargs = _kwargs(
        provider_specific_header={"custom_llm_provider": "anthropic", "extra_headers": {"x-api-key": "k", "a": "b"}}
    )

    out = await _hook(module, kwargs)

    assert out["provider_specific_header"] == {"custom_llm_provider": "anthropic", "extra_headers": {"a": "b"}}


async def test_does_not_mutate_the_original_kwargs(module):
    headers = {"authorization": "Bearer sk-ant-oat01-x"}
    psh = [{"custom_llm_provider": "anthropic", "extra_headers": {"authorization": "Bearer sk-ant-oat01-x"}}]
    kwargs = _kwargs(headers=headers, provider_specific_header=psh)

    await _hook(module, kwargs)

    assert headers == {"authorization": "Bearer sk-ant-oat01-x"}
    assert psh[0]["extra_headers"] == {"authorization": "Bearer sk-ant-oat01-x"}
    assert kwargs["headers"] is headers


async def test_reads_model_group_from_metadata_when_litellm_metadata_is_absent(module):
    kwargs = {"metadata": {"model_group": "anthropic-api/claude-x"}, "headers": {"authorization": "Bearer t"}}

    out = await _hook(module, kwargs)

    assert out["headers"] == {}


@pytest.mark.parametrize("group", ["anthropic/claude-x", "claude-x", "anthropic-apix/claude-x", None, 5])
async def test_leaves_other_model_groups_untouched(module, group):
    kwargs = _kwargs(
        group=group, headers={"authorization": "Bearer sk-ant-oat01-x", "anthropic-beta": "oauth-2025-04-20"}
    )

    assert await _hook(module, kwargs) is None


async def test_returns_none_when_nothing_to_remove(module):
    assert await _hook(module, _kwargs(headers={"anthropic-beta": "foo, bar"})) is None
    assert await _hook(module, _kwargs(headers={"anthropic-version": "2023-06-01", "anthropic-beta": "foo"})) is None
    assert await _hook(module, _kwargs()) is None


async def test_tolerates_missing_or_non_dict_metadata_and_headers(module):
    assert await _hook(module, {"model": "m"}) is None
    assert await _hook(module, {"litellm_metadata": None, "metadata": "x"}) is None
    assert await _hook(module, _kwargs(headers=None, extra_headers="x", provider_specific_header=None)) is None
