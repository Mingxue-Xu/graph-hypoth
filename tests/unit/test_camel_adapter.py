from __future__ import annotations

import builtins
import sys
import types

import pytest

from src.camel_adapter import (
    _create_camel_model_backend,
    _extract_json_object,
    backend_json,
)
from src.config import AgentConfig, ModelConfig


class _JsonFakeBackend:
    def __init__(self, content: str | Exception) -> None:
        self._content = content
        self.calls: list[object] = []

    def run(self, messages: object) -> dict:
        self.calls.append(messages)
        if isinstance(self._content, Exception):
            raise self._content
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.mark.parametrize(
    ("content", "kwargs", "expected"),
    [
        ('{"a": 1}', {}, {"a": 1}),
        ("prefix {\"a\": 1} suffix", {}, {"a": 1}),
        ("", {}, {}),
        ("not json at all", {}, {}),
        (RuntimeError("boom"), {"catch_backend_errors": True}, {}),
    ],
)
def test_backend_json_parses_or_degrades(content, kwargs, expected):
    backend = _JsonFakeBackend(content)
    assert backend_json(backend, "sys", "user", **kwargs) == expected


def test_backend_json_sends_system_and_user_messages_verbatim() -> None:
    backend = _JsonFakeBackend('{"a": 1}')

    backend_json(backend, "sys", "user")

    assert backend.calls == [
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]
    ]


def test_backend_json_propagates_backend_error_by_default() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        backend_json(_JsonFakeBackend(RuntimeError("boom")), "sys", "user")


def test_backend_json_return_response_yields_data_and_response() -> None:
    backend = _JsonFakeBackend('{"a": 1}')

    data, response = backend_json(
        backend, "sys", "user", return_response=True
    )

    assert data == {"a": 1}
    assert response["choices"][0]["message"]["content"] == '{"a": 1}'


def test_extract_json_object_errors_are_backend_neutral() -> None:
    with pytest.raises(ValueError, match="model response did not contain JSON"):
        _extract_json_object("not json")


def test_camel_model_factory_receives_configured_model(monkeypatch) -> None:
    captured: dict[str, object] = {}
    backend = object()

    class FakeModelFactory:
        @staticmethod
        def create(**kwargs: object) -> object:
            captured.update(kwargs)
            return backend

    camel_module = types.ModuleType("camel")
    models_module = types.ModuleType("camel.models")
    models_module.ModelFactory = FakeModelFactory
    monkeypatch.setitem(sys.modules, "camel", camel_module)
    monkeypatch.setitem(sys.modules, "camel.models", models_module)
    monkeypatch.setenv("MODEL_API_KEY", "secret")
    agent = AgentConfig(
        temperature=0.2,
        model=ModelConfig(
            provider="openrouter",
            model_id="vendor/model",
            api_key_env="MODEL_API_KEY",
            base_url="https://example.test/v1",
            max_tokens=1234,
            timeout_seconds=45,
        ),
    )

    assert _create_camel_model_backend(agent, role_name="critic_panel") is backend
    assert captured == {
        "model_platform": "openrouter",
        "model_type": "vendor/model",
        "model_config_dict": {"temperature": 0.2, "max_tokens": 1234},
        "api_key": "secret",
        "url": "https://example.test/v1",
        "timeout": 45.0,
    }


def test_missing_camel_models_dependency_has_clear_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "camel.models":
            raise ImportError("blocked camel.models import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    agent = AgentConfig(
        temperature=0.0,
        model=ModelConfig(provider="openai", model_id="gpt-test"),
    )

    with pytest.raises(RuntimeError, match="CAMEL is required"):
        _create_camel_model_backend(agent, role_name="extractor")


def test_model_factory_errors_redact_configured_api_key(monkeypatch) -> None:
    class FakeModelFactory:
        @staticmethod
        def create(**kwargs: object) -> object:
            raise ValueError(f"bad credential {kwargs['api_key']}")

    camel_module = types.ModuleType("camel")
    models_module = types.ModuleType("camel.models")
    models_module.ModelFactory = FakeModelFactory
    monkeypatch.setitem(sys.modules, "camel", camel_module)
    monkeypatch.setitem(sys.modules, "camel.models", models_module)
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")
    agent = AgentConfig(
        temperature=0.0,
        model=ModelConfig(
            provider="anthropic",
            model_id="claude-test",
            api_key_env="MODEL_API_KEY",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _create_camel_model_backend(agent, role_name="research_synthesist")

    assert "secret-value" not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)
