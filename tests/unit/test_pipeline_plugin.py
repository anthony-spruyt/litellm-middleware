import importlib
import sys
import types

import pytest


@pytest.fixture
def fake_pipeline_and_registry(monkeypatch):
    litellm = types.ModuleType("litellm")
    logging = types.ModuleType("litellm._logging")
    pipeline_mod = types.ModuleType("litellm_middleware.pipeline")
    registry_mod = types.ModuleType("litellm_middleware.registry")

    class Logger:
        def warning(self, *args, **kwargs):
            pass

    class MiddlewarePipeline:
        def __init__(self, middlewares):
            self.middlewares = middlewares

    sentinel_middlewares = (object(),)
    logging.verbose_proxy_logger = Logger()
    pipeline_mod.MiddlewarePipeline = MiddlewarePipeline
    registry_mod.load_default_middlewares = lambda logger=None: sentinel_middlewares

    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setitem(sys.modules, "litellm._logging", logging)
    monkeypatch.setitem(sys.modules, "litellm_middleware.pipeline", pipeline_mod)
    monkeypatch.setitem(sys.modules, "litellm_middleware.registry", registry_mod)

    return sentinel_middlewares


def test_pipeline_plugin_exposes_configured_pipeline(fake_pipeline_and_registry):
    sys.modules.pop("litellm_middleware.pipeline_plugin", None)

    plugin = importlib.import_module("litellm_middleware.pipeline_plugin")

    assert plugin.pipeline_middleware.middlewares is fake_pipeline_and_registry
