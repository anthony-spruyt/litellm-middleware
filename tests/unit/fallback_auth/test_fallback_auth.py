import enum
import importlib
import sys
import types
from collections.abc import Mapping

import pytest


@pytest.fixture(autouse=True)
def fake_litellm(monkeypatch):
    litellm = types.ModuleType("litellm")
    logging = types.ModuleType("litellm._logging")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    custom_logger.CustomLogger = type("CustomLogger", (), {})

    class Logger:
        def __init__(self):
            self.messages = []

        def warning(self, message, *args):
            self.messages.append(message % args)

    logging.verbose_proxy_logger = Logger()
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setitem(sys.modules, "litellm._logging", logging)
    monkeypatch.setitem(sys.modules, "litellm.integrations", types.ModuleType("litellm.integrations"))
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger", custom_logger)
    return logging


@pytest.fixture
def warnings(fake_litellm):
    return fake_litellm.verbose_proxy_logger.messages


@pytest.fixture
def module():
    sys.modules.pop("litellm_middleware.fallback_auth.fallback_auth", None)
    sys.modules.pop("litellm_middleware.pipeline", None)
    return importlib.import_module("litellm_middleware.fallback_auth.fallback_auth")


async def _hook(module, kwargs, call_type="anthropic_messages"):
    return await module.fallback_auth.async_pre_call_deployment_hook(kwargs, call_type)


def _kwargs(group="anthropic-api/claude-x", **extra):
    return {"model": "anthropic/claude-x", "litellm_metadata": {"deployment_model_name": group}, **extra}


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


ROUTER_BUCKETS = [
    ("anthropic_messages", "litellm_metadata", "metadata"),
    ("aanthropic_messages", "litellm_metadata", "metadata"),
    ("aresponses", "litellm_metadata", "metadata"),
    ("responses", "litellm_metadata", "metadata"),
    ("acompletion", "metadata", "litellm_metadata"),
    ("completion", "metadata", "litellm_metadata"),
    ("atext_completion", "metadata", "litellm_metadata"),
]


@pytest.mark.parametrize(("call_type", "bucket", "other"), ROUTER_BUCKETS)
async def test_reads_the_deployment_name_from_the_bucket_the_router_writes_for_the_call_type(
    module, call_type, bucket, other
):
    kwargs = {bucket: {"deployment_model_name": "anthropic-api/claude-x"}, "headers": {"authorization": "Bearer t"}}

    out = await _hook(module, kwargs, call_type)

    assert out["headers"] == {}


@pytest.mark.parametrize(("call_type", "bucket", "other"), ROUTER_BUCKETS)
async def test_ignores_a_client_supplied_deployment_name_in_the_other_bucket(module, call_type, bucket, other):
    kwargs = {
        bucket: {"deployment_model_name": "anthropic/*"},
        other: {"deployment_model_name": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    assert await _hook(module, kwargs, call_type) is None


@pytest.mark.parametrize(("call_type", "bucket", "other"), ROUTER_BUCKETS)
async def test_a_deployment_name_only_in_the_other_bucket_is_not_used(module, call_type, bucket, other):
    kwargs = {other: {"deployment_model_name": "anthropic-api/claude-x"}, "headers": {"authorization": "Bearer t"}}

    assert await _hook(module, kwargs, call_type) is None


@pytest.mark.parametrize(("call_type", "bucket", "other"), ROUTER_BUCKETS)
async def test_a_requested_model_group_never_decides(module, call_type, bucket, other):
    kwargs = {
        bucket: {"model_group": "anthropic-api/claude-x"},
        other: {"model_group": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    assert await _hook(module, kwargs, call_type) is None


@pytest.mark.parametrize("call_type", [None, "", "aembedding", "something_new", 5])
async def test_leaves_kwargs_untouched_when_the_call_type_has_no_known_bucket(module, call_type):
    kwargs = {
        "litellm_metadata": {"deployment_model_name": "anthropic-api/claude-x"},
        "metadata": {"deployment_model_name": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    assert await _hook(module, kwargs, call_type) is None


async def test_accepts_a_call_types_enum_member(module):
    class CallTypes(enum.StrEnum):
        acompletion = "acompletion"

    kwargs = {
        "metadata": {"deployment_model_name": "anthropic-api/claude-x"},
        "headers": {"authorization": "Bearer t"},
    }

    out = await _hook(module, kwargs, CallTypes.acompletion)

    assert out["headers"] == {}


async def test_drops_oauth_beta_tokens_regardless_of_case(module):
    out = await _hook(module, _kwargs(headers={"anthropic-beta": "OAuth-2025-04-20,Foo-Bar"}))

    assert out["headers"] == {"anthropic-beta": "Foo-Bar"}


async def test_scrubs_header_names_given_as_bytes(module):
    kwargs = _kwargs(
        headers={
            b"Authorization": "Bearer t",
            b"X-Api-Key": "k",
            b"anthropic-beta": "oauth-2025-04-20,foo",
            b"x-keep": "1",
        }
    )

    out = await _hook(module, kwargs)

    assert out["headers"] == {b"anthropic-beta": "foo", b"x-keep": "1"}


async def test_scrubs_oauth_tokens_from_a_bytes_beta_value(module):
    kwargs = _kwargs(headers={"anthropic-beta": b"oauth-2025-04-20,foo", "x-keep": b"1"})

    out = await _hook(module, kwargs)

    assert out["headers"] == {"anthropic-beta": b"foo", "x-keep": b"1"}
    assert type(out["headers"]["anthropic-beta"]) is bytes


async def test_drops_a_bytes_beta_value_holding_only_oauth_tokens(module):
    out = await _hook(module, _kwargs(headers={b"Anthropic-Beta": b"oauth-2025-04-20", "x-keep": "1"}))

    assert out["headers"] == {"x-keep": "1"}


async def test_leaves_a_bytes_beta_value_without_oauth_tokens(module):
    assert await _hook(module, _kwargs(headers={"anthropic-beta": b"foo,bar"})) is None


@pytest.mark.parametrize("container", [list, tuple])
async def test_scrubs_bytes_items_in_sequence_beta_values(module, container):
    out = await _hook(module, _kwargs(headers={"anthropic-beta": container([b"oauth-2025-04-20", b"foo"])}))

    assert out["headers"] == {"anthropic-beta": container([b"foo"])}


@pytest.mark.parametrize("container", [list, tuple])
async def test_drops_oauth_tokens_from_sequence_beta_values(module, container):
    beta = container(["oauth-2025-04-20", "claude-code-20250219", "OAuth-x, foo"])

    out = await _hook(module, _kwargs(headers={"anthropic-beta": beta}))

    assert out["headers"] == {"anthropic-beta": container(["claude-code-20250219", "foo"])}
    assert type(out["headers"]["anthropic-beta"]) is container


@pytest.mark.parametrize("container", [list, tuple])
async def test_drops_the_beta_header_when_a_sequence_value_holds_only_oauth_tokens(module, container):
    out = await _hook(module, _kwargs(headers={"anthropic-beta": container(["oauth-2025-04-20"]), "x-keep": "1"}))

    assert out["headers"] == {"x-keep": "1"}


async def test_leaves_a_sequence_beta_value_without_oauth_tokens(module):
    assert await _hook(module, _kwargs(headers={"anthropic-beta": ["foo", "bar"]})) is None


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


def _credentialed_kwargs():
    return _kwargs(
        headers={"Authorization": "Bearer t", "x-api-key": "k", "anthropic-beta": "oauth-2025-04-20", "x-keep": "1"},
        extra_headers={"authorization": "Bearer t", "x-keep": "1"},
        provider_specific_header=[
            {"custom_llm_provider": "anthropic", "extra_headers": {"authorization": "Bearer t", "x-keep": "1"}}
        ],
    )


async def test_drops_all_client_header_containers_if_scrubbing_fails(module, monkeypatch, warnings):
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
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0]
    for value in ("Bearer", "sk-ant", "oauth-", "client-api-key", "x-keep"):
        assert value not in warnings[0]


async def test_leaves_other_groups_untouched_if_scrubbing_would_fail(module, monkeypatch):
    def boom(_headers):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_scrub", boom)

    assert await _hook(module, _kwargs(group="anthropic/claude-x", headers={"authorization": "Bearer t"})) is None


async def test_leaves_kwargs_untouched_and_warns_if_the_deployment_lookup_fails(module, warnings):
    class Exploding(_ReadOnlyMapping):
        def get(self, key, default=None):
            raise RuntimeError("boom")

    kwargs = {"litellm_metadata": Exploding({}), "headers": {"authorization": "Bearer t"}}

    assert await _hook(module, kwargs) is None
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0]
    assert "Bearer" not in warnings[0]


async def test_does_not_warn_when_nothing_fails(module, warnings):
    await _hook(module, _kwargs(headers={"authorization": "Bearer t"}))
    await _hook(module, _kwargs(group="anthropic/claude-x", headers={"authorization": "Bearer t"}))

    assert warnings == []
