"""Standalone-run intermediate-artifact logging at the backend boundary.

The hook records every LLM seam's raw request and response in the audit log, so
the standalone setting preserves the same intermediate signals as CLI-backed
model calls.
"""
from __future__ import annotations

from src.camel_adapter import _install_logging_backend_hook
from src.graph_state_runtime import _logging_make_backend
from src.log_store import SQLiteLogStore


class _FakeBackend:
    """A minimal CAMEL-shaped backend: ``run(messages)`` returns a ChatCompletion-like dict."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list = []

    def run(self, messages, **kwargs):
        self.calls.append(messages)
        return {"choices": [{"message": {"content": self._content}}]}


def test_install_logging_hook_logs_request_and_response_then_delegates():
    recorded: list = []
    fake = _FakeBackend("RESPONSE-TEXT")
    wrapped = _install_logging_backend_hook(
        fake, lambda role, msgs, resp: recorded.append((role, msgs, resp)), role="proposer"
    )
    out = wrapped.run([{"role": "user", "content": "hi"}])
    # delegates: returns the inner response unchanged
    assert out == {"choices": [{"message": {"content": "RESPONSE-TEXT"}}]}
    assert len(recorded) == 1
    role, msgs, resp = recorded[0]
    assert role == "proposer"
    assert msgs == [{"role": "user", "content": "hi"}]
    assert resp["choices"][0]["message"]["content"] == "RESPONSE-TEXT"


def test_logging_hook_is_idempotent():
    fake = _FakeBackend("R")
    sink_calls: list = []
    _install_logging_backend_hook(fake, lambda *a: sink_calls.append("first"), role="x")
    _install_logging_backend_hook(fake, lambda *a: sink_calls.append("second"), role="x")  # no-op
    fake.run([])
    assert sink_calls == ["first"]  # second hook not installed


def test_logging_hook_never_breaks_the_run_on_sink_error():
    fake = _FakeBackend("R")

    def boom(*_a):
        raise RuntimeError("sink exploded")

    wrapped = _install_logging_backend_hook(fake, boom, role="x")
    out = wrapped.run([])  # must NOT raise
    assert out["choices"][0]["message"]["content"] == "R"


def test_logging_make_backend_records_every_seam_call(tmp_path):
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    def base_factory(_agent_config, *, role_name):
        return _FakeBackend(f"resp-for-{role_name}")

    make = _logging_make_backend(base_factory, store, run_id="run1", thread_id="t1")
    synth = make(object(), role_name="research_synthesist")
    critic = make(object(), role_name="critic_panel")

    synth.run([{"role": "user", "content": "mine this"}])
    critic.run([{"role": "user", "content": "judge this"}])

    events = store.list_events("run1")
    assert len(events) == 2
    by_role = {e.sender_role: e for e in events}
    assert set(by_role) == {"research_synthesist", "critic_panel"}
    # the raw request + response are persisted in structured_payload
    synth_payload = by_role["research_synthesist"].structured_payload
    assert synth_payload["raw_response"] == "resp-for-research_synthesist"
    assert synth_payload["request_messages"] == [{"role": "user", "content": "mine this"}]
    assert by_role["critic_panel"].structured_payload["raw_response"] == "resp-for-critic_panel"


def test_logging_make_backend_unique_keys_for_repeated_role(tmp_path):
    """Multiple calls of the SAME role (e.g. a 3-judge panel) must not collide on idempotency key."""
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    def base_factory(_c, *, role_name):
        return _FakeBackend("x")

    make = _logging_make_backend(base_factory, store, run_id="r", thread_id="t")
    judge = make(object(), role_name="critic_panel")
    judge.run([{"role": "user", "content": "1"}])
    judge.run([{"role": "user", "content": "2"}])  # same role, second call

    events = store.list_events("r")
    assert len(events) == 2  # both recorded, no conflict


def test_logging_make_backend_persists_codex_response_metadata(tmp_path):
    store = SQLiteLogStore(tmp_path / "events.sqlite")
    store.setup()

    class _CodexBackend:
        def run(self, messages, **kwargs):
            del messages, kwargs
            return {
                "choices": [{"message": {"content": "result"}}],
                "info": {
                    "provider": "codex-cli",
                    "model": "gpt-test",
                    "role": "builder",
                    "latency_ms": 42,
                    "termination_reason": "stop",
                    "usage": {"input_tokens": 7, "output_tokens": 3},
                    "codex_thread_id": "thread-codex",
                },
            }

    make = _logging_make_backend(
        lambda _config, *, role_name: _CodexBackend(),
        store,
        run_id="run-codex",
        thread_id="thread",
    )
    make(object(), role_name="builder").run(
        [{"role": "user", "content": "request"}]
    )

    [event] = store.list_events("run-codex")
    assert event.provider == "codex-cli"
    assert event.model == "gpt-test"
    assert event.latency_ms == 42
    assert event.termination_reason == "stop"
    assert event.token_usage == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert (
        event.structured_payload["response_metadata"]["codex_thread_id"]
        == "thread-codex"
    )
