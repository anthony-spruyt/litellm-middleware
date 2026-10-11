import importlib
import sys
from collections.abc import Mapping

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
            "X-Api-Key": "client-api-key",
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


async def test_decides_on_the_selected_deployment_not_the_requested_name(module):
    kwargs = {
        "litellm_metadata": {"model_group": "anthropic-api/claude-x", "deployment_model_name": "anthropic/*"},
        "headers": {"authorization": "Bearer t"},
    }

    assert await _hook(module, kwargs) is None


async def test_strips_when_the_selected_deployment_is_an_api_key_group_behind_another_name(module):
    kwargs = {
        "litellm_metadata": {"model_group": "claude-alias", "deployment_model_name": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    out = await _hook(module, kwargs)

    assert out["headers"] == {}


async def test_reads_deployment_model_name_from_metadata_too(module):
    kwargs = {
        "metadata": {"model_group": "claude-alias", "deployment_model_name": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    out = await _hook(module, kwargs)

    assert out["headers"] == {}


async def test_drops_oauth_beta_tokens_regardless_of_case(module):
    out = await _hook(module, _kwargs(headers={"anthropic-beta": "OAuth-2025-04-20,Foo-Bar"}))

    assert out["headers"] == {"anthropic-beta": "Foo-Bar"}


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


class _ReadOnlyMapping(Mapping):
    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


async def test_scrubs_headers_held_in_a_non_dict_mapping(module):
    kwargs = _kwargs(
        headers=_ReadOnlyMapping({"authorization": "Bearer t", "x-keep": "1"}),
        extra_headers=_ReadOnlyMapping({"x-api-key": "k"}),
        provider_specific_header=_ReadOnlyMapping(
            {"custom_llm_provider": "anthropic", "extra_headers": _ReadOnlyMapping({"authorization": "Bearer t"})}
        ),
    )

    out = await _hook(module, kwargs)

    assert out["headers"] == {"x-keep": "1"}
    assert out["extra_headers"] == {}
    assert out["provider_specific_header"] == {"custom_llm_provider": "anthropic", "extra_headers": {}}


async def test_reads_deployment_name_from_a_non_dict_mapping_bucket(module):
    kwargs = {
        "litellm_metadata": _ReadOnlyMapping({"deployment_model_name": "anthropic-api/claude-x"}),
        "headers": {"authorization": "Bearer t"},
    }

    out = await _hook(module, kwargs)

    assert out["headers"] == {}


async def test_deployment_name_in_metadata_wins_over_model_group_in_litellm_metadata(module):
    kwargs = {
        "litellm_metadata": {"model_group": "claude-alias"},
        "metadata": {"deployment_model_name": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    out = await _hook(module, kwargs)

    assert out["headers"] == {}


async def test_deployment_name_in_litellm_metadata_wins_over_model_group_in_metadata(module):
    kwargs = {
        "litellm_metadata": {"deployment_model_name": "anthropic/*"},
        "metadata": {"model_group": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    assert await _hook(module, kwargs) is None


async def test_deployment_name_in_metadata_wins_over_a_matching_model_group_in_litellm_metadata(module):
    kwargs = {
        "litellm_metadata": {"model_group": "anthropic-api/claude-x"},
        "metadata": {"deployment_model_name": "anthropic/*"},
        "headers": {"authorization": "Bearer t"},
    }

    assert await _hook(module, kwargs) is None


def _credentialed_kwargs():
    return _kwargs(
        headers={"Authorization": "Bearer t", "x-api-key": "k", "anthropic-beta": "oauth-2025-04-20", "x-keep": "1"},
        extra_headers={"authorization": "Bearer t", "x-keep": "1"},
        provider_specific_header=[
            {"custom_llm_provider": "anthropic", "extra_headers": {"authorization": "Bearer t", "x-keep": "1"}}
        ],
    )


async def test_drops_all_client_header_containers_if_scrubbing_fails(module, monkeypatch):
    def boom(_headers):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_scrub", boom)
    kwargs = _credentialed_kwargs()
    original = dict(kwargs)
    snapshot = repr(kwargs)

    out = await _hook(module, kwargs)

    assert out["headers"] == {}
    assert out["extra_headers"] == {}
    assert not out["provider_specific_header"]
    rendered = repr(out).lower()
    for needle in ("authorization", "x-api-key", "oauth-", "bearer"):
        assert needle not in rendered
    assert out["model"] == kwargs["model"]
    assert repr(kwargs) == snapshot
    assert kwargs["headers"] is original["headers"]


async def test_leaves_other_groups_untouched_if_scrubbing_would_fail(module, monkeypatch):
    def boom(_headers):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_scrub", boom)

    assert await _hook(module, _kwargs(group="anthropic/claude-x", headers={"authorization": "Bearer t"})) is None


async def test_leaves_kwargs_untouched_if_the_deployment_lookup_fails(module):
    class Exploding(_ReadOnlyMapping):
        def get(self, key, default=None):
            raise RuntimeError("boom")

    kwargs = {"litellm_metadata": Exploding({}), "headers": {"authorization": "Bearer t"}}

    assert await _hook(module, kwargs) is None
