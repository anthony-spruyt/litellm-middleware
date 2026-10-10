import importlib
import sys
import types

import pytest


@pytest.fixture
def production_import_shape(monkeypatch):
    litellm = types.ModuleType("litellm")
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")
    logging = types.ModuleType("litellm._logging")

    class CustomLogger:
        pass

    class Logger:
        def debug(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    custom_logger.CustomLogger = CustomLogger
    logging.verbose_proxy_logger = Logger()

    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setitem(sys.modules, "litellm.integrations", integrations)
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger", custom_logger)
    monkeypatch.setitem(sys.modules, "litellm._logging", logging)


def test_production_dotted_imports_resolve(production_import_shape):
    modules = [
        "litellm_middleware.pipeline",
        "litellm_middleware.registry",
        "litellm_middleware.pipeline_plugin",
        "litellm_middleware.secret_masking.shared_fakes",
        "litellm_middleware.secret_masking.secret_masking",
        "litellm_middleware.ratelimit_headers.ratelimit_headers",
        "litellm_middleware.tool_guard.tool_results",
        "litellm_middleware.tool_guard.tool_guard",
    ]

    for module in modules:
        sys.modules.pop(module, None)

    for module in modules:
        importlib.import_module(module)


def test_production_pipeline_loads_secret_masking(production_import_shape):
    for module in [
        "litellm_middleware.secret_masking.secret_masking",
        "litellm_middleware.registry",
        "litellm_middleware.pipeline_plugin",
    ]:
        sys.modules.pop(module, None)

    plugin = importlib.import_module("litellm_middleware.pipeline_plugin")
    masking = importlib.import_module("litellm_middleware.secret_masking.secret_masking")

    assert masking.secret_masking in plugin.pipeline_middleware.middlewares


def test_production_pipeline_loads_ratelimit_headers(production_import_shape):
    for module in [
        "litellm_middleware.ratelimit_headers.ratelimit_headers",
        "litellm_middleware.registry",
        "litellm_middleware.pipeline_plugin",
    ]:
        sys.modules.pop(module, None)

    plugin = importlib.import_module("litellm_middleware.pipeline_plugin")
    restorer = importlib.import_module("litellm_middleware.ratelimit_headers.ratelimit_headers")

    assert restorer.ratelimit_headers in plugin.pipeline_middleware.middlewares


def test_production_pipeline_loads_tool_guard(production_import_shape):
    for module in [
        "litellm_middleware.tool_guard.tool_guard",
        "litellm_middleware.registry",
        "litellm_middleware.pipeline_plugin",
    ]:
        sys.modules.pop(module, None)

    plugin = importlib.import_module("litellm_middleware.pipeline_plugin")
    guard = importlib.import_module("litellm_middleware.tool_guard.tool_guard")

    assert guard.tool_guard in plugin.pipeline_middleware.middlewares
