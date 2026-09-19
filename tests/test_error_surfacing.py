"""Regression tests: LLM/loop failures must surface as `error` SSE events,
never as a torn-down stream.

Context: an early version let a `RuntimeError` from DeepSeekChatModel
(401 / timeout / transport error) escape `AgentLoop.run`, which killed
the SSE response mid-flight. The Dashboard rendered an empty event panel
and the user could not tell what went wrong. These tests pin the fix.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_platform.api.app import Runtime, create_app
from agent_platform.approvals import ApprovalStore
from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import Settings
from agent_platform.config_store import AgentConfig, AgentConfigStore
from agent_platform.core.agent_loop import AgentLoop
from agent_platform.core.events import EventType
from agent_platform.core.llm import ChatModel, MockChatModel
from agent_platform.core.providers import create_chat_model
from agent_platform.secrets_store import SecretStore
from agent_platform.store.agent_manager import AgentManager
from agent_platform.tools import build_default_registry


async def _read_sse(resp):
    """Yield (event_name, data_dict) from an SSE streaming response."""
    event_name = None
    async for line in resp.aiter_lines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event_name is not None:
            try:
                yield event_name, json.loads(line.split(":", 1)[1])
            except json.JSONDecodeError:
                pass
            event_name = None


async def _drain(agent: AgentLoop, **kwargs):
    out = []
    async for ev in agent.run(**kwargs):
        out.append(ev)
    return out


# ----- loop-level: exception -> error event ---------------------------------


@pytest.mark.asyncio
async def test_llm_failure_emits_error_event_not_exception(ckpt_store) -> None:
    """A ChatModel that always raises must NOT propagate out of run()."""

    class ExplodingModel(ChatModel):
        async def ainvoke(self, messages, tools):
            raise RuntimeError("deepseek 401: Authentication Fails")

    s = Settings(
        llm_max_retries=1,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
        compress_trigger_tokens=100_000,
        checkpoint_ttl_seconds=60,
    )
    agent = AgentLoop(
        llm=ExplodingModel(),
        tools=build_default_registry(s),
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="boom",
    )
    # No pytest.raises: the call must return normally.
    events = await _drain(agent, thread_id="boom", user_id="u", user_message="hi")
    types = [e.type for e in events]
    assert EventType.START in types
    assert EventType.ERROR in types
    err = next(e for e in events if e.type == EventType.ERROR)
    assert err.data["reason"] == "internal_error"
    assert err.data["error_type"] == "RuntimeError"
    assert "401" in err.data["message"]


@pytest.mark.asyncio
async def test_transport_timeout_also_surfaces(ckpt_store) -> None:
    """Same guarantee for a transport-style error."""

    class TimingOutModel(ChatModel):
        async def ainvoke(self, messages, tools):
            raise TimeoutError("deepseek timeout after 30.0s")

    s = Settings(
        llm_max_retries=1,
        llm_retry_base_delay=0.0,
        compress_trigger_tokens=100_000,
        checkpoint_ttl_seconds=60,
    )
    agent = AgentLoop(
        llm=TimingOutModel(),
        tools=build_default_registry(s),
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="boom",
    )
    events = await _drain(agent, thread_id="to", user_id="u", user_message="hi")
    err = next(e for e in events if e.type == EventType.ERROR)
    assert err.data["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_successful_run_emits_no_error_event(agent) -> None:
    """The happy path must remain error-free (no over-eager catching)."""
    events = await _drain(agent, thread_id="ok", user_id="u", user_message="hello")
    assert EventType.ERROR not in [e.type for e in events]
    assert EventType.FINISH in [e.type for e in events]


# ----- api-level: SSE stream keeps its terminal frame -----------------------


@pytest_asyncio.fixture
async def runtime() -> Runtime:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    s = Settings(
        llm_max_retries=1,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
        checkpoint_ttl_seconds=60,
    )
    redis = fakeredis.aioredis.FakeRedis()
    ckpt = CheckpointStore(redis, ttl_seconds=60)
    cfg_store = AgentConfigStore(redis)
    secret_store = SecretStore(redis)
    tools = build_default_registry(s)
    llm = MockChatModel()
    approvals = ApprovalStore(redis, ttl_seconds=s.approval_grant_ttl_seconds)
    agents = AgentManager(
        llm=llm,
        tools=tools,
        checkpoint=ckpt,
        settings=s,
        config_store=cfg_store,
        secret_store=secret_store,
        approvals=approvals,
        llm_factory=lambda m, k: create_chat_model(m, s, api_key_override=k),
    )
    yield Runtime(
        settings=s,
        checkpoint=ckpt,
        config_store=cfg_store,
        secret_store=secret_store,
        tools=tools,
        llm=llm,
        agents=agents,
        approval_store=approvals,
    )
    await redis.aclose()


@pytest.mark.asyncio
async def test_admin_chat_unknown_provider_returns_400(runtime: Runtime) -> None:
    """An unroutable model string should be a clean HTTP error, not a
    half-open SSE stream."""
    await runtime.config_store.upsert(
        AgentConfig(agent_id="bad", model="not-a-real-provider")
    )
    app = create_app(runtime)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        r = await ac.post(
            "/admin/api/chat", json={"agent_id": "bad", "content": "hi"}
        )
        assert r.status_code == 400
        body = r.json()
        assert "not-a-real-provider" in str(body["detail"])


@pytest.mark.asyncio
async def test_admin_chat_bad_key_streams_error_frame(runtime: Runtime) -> None:
    """With a bogus key the stream must still terminate with an `error`
    event the Dashboard can render — not a silent close."""
    from datetime import datetime, timezone

    from agent_platform.secrets_store import SecretEntry

    await runtime.secret_store.set(
        SecretEntry(
            provider="deepseek",
            name="api_key",
            value="sk-definitely-invalid",
            updated_at=datetime.now(timezone.utc),
        )
    )
    await runtime.config_store.upsert(
        AgentConfig(agent_id="ds", model="deepseek-flash")
    )
    app = create_app(runtime)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        async with ac.stream(
            "POST",
            "/admin/api/chat",
            json={"agent_id": "ds", "content": "hi"},
        ) as resp:
            assert resp.status_code == 200
            events = [(n, d) async for n, d in _read_sse(resp)]
    names = [n for n, _ in events]
    assert names[0] == "start"
    assert "error" in names, f"expected an error frame, got {names}"
    assert names[-1] == "error", f"error must be terminal, got {names}"
    err = next(d for n, d in events if n == "error")
    assert err["reason"] == "internal_error"


# ----- retry policy: only retry failures that can plausibly clear ----------


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried(ckpt_store) -> None:
    """A 400 (unknown model name, bad request) fails identically every time.

    Regression: the loop used to retry it three times with backoff, so the
    operator waited 3.5s to be told something the first response already
    said.
    """
    from agent_platform.core.llm import LLMError

    calls = {"n": 0}

    class BadModel(ChatModel):
        async def ainvoke(self, messages, tools):
            calls["n"] += 1
            raise LLMError("deepseek 400: unknown model", retryable=False)

    s = Settings(
        llm_max_retries=3,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
        compress_trigger_tokens=100_000,
        checkpoint_ttl_seconds=60,
    )
    agent = AgentLoop(
        llm=BadModel(),
        tools=build_default_registry(s),
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="no-retry",
    )
    events = await _drain(agent, thread_id="nr", user_id="u", user_message="hi")
    assert calls["n"] == 1, "a non-retryable error must be tried exactly once"
    err = next(e for e in events if e.type == EventType.ERROR)
    assert "unknown model" in err.data["message"]


@pytest.mark.asyncio
async def test_retryable_error_is_retried_to_the_budget(ckpt_store) -> None:
    from agent_platform.core.llm import LLMError

    calls = {"n": 0}

    class FlakyModel(ChatModel):
        async def ainvoke(self, messages, tools):
            calls["n"] += 1
            raise LLMError("deepseek 503: upstream busy", retryable=True)

    s = Settings(
        llm_max_retries=3,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
        compress_trigger_tokens=100_000,
        checkpoint_ttl_seconds=60,
    )
    agent = AgentLoop(
        llm=FlakyModel(),
        tools=build_default_registry(s),
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="retry",
    )
    await _drain(agent, thread_id="r", user_id="u", user_message="hi")
    assert calls["n"] == 3, "a retryable error should use the whole budget"


@pytest.mark.asyncio
async def test_unknown_exception_defaults_to_retryable(ckpt_store) -> None:
    """An exception with no `retryable` attribute is assumed transient.

    Over-retrying an unknown error beats turning a blip into a hard failure.
    """
    calls = {"n": 0}

    class MysteryModel(ChatModel):
        async def ainvoke(self, messages, tools):
            calls["n"] += 1
            raise RuntimeError("something we did not anticipate")

    s = Settings(
        llm_max_retries=2,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
        compress_trigger_tokens=100_000,
        checkpoint_ttl_seconds=60,
    )
    agent = AgentLoop(
        llm=MysteryModel(),
        tools=build_default_registry(s),
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="mystery",
    )
    await _drain(agent, thread_id="m", user_id="u", user_message="hi")
    assert calls["n"] == 2

