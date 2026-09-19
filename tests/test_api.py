"""FastAPI integration tests using fakeredis-backed Runtime."""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_platform.api.app import Runtime, create_app
from agent_platform.approvals import ApprovalStore
from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import Settings
from agent_platform.config_store import AgentConfigStore
from agent_platform.core.llm import MockChatModel
from agent_platform.store.agent_manager import AgentManager
from agent_platform.tools import build_default_registry


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
    tools = build_default_registry(s)
    llm = MockChatModel()
    approvals = ApprovalStore(redis, ttl_seconds=s.approval_grant_ttl_seconds)
    agents = AgentManager(
        llm=llm, tools=tools, checkpoint=ckpt, settings=s, approvals=approvals
    )
    from agent_platform.secrets_store import SecretStore

    yield Runtime(
        settings=s,
        checkpoint=ckpt,
        config_store=AgentConfigStore(redis),
        secret_store=SecretStore(redis),
        tools=tools,
        llm=llm,
        agents=agents,
        approval_store=approvals,
    )
    await redis.aclose()


async def _read_sse(resp):
    """Yield (event_name, data_dict) pairs from an SSE response."""
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


@pytest.mark.asyncio
async def test_healthz(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_session_lifecycle(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/sessions", json={"agent_id": "demo", "user_id": "u1"})
        assert r.status_code == 200
        sid = r.json()["thread_id"]

        r = await ac.get(f"/v1/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["thread_id"] == sid


@pytest.mark.asyncio
async def test_chat_sse_stream(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/sessions", json={"agent_id": "demo", "user_id": "u1"})
        sid = r.json()["thread_id"]

        async with ac.stream(
            "POST",
            f"/v1/sessions/{sid}/chat",
            json={"content": "please echo abc"},
        ) as resp:
            assert resp.status_code == 200
            events = [name async for name, _ in _read_sse(resp)]
        assert "start" in events
        assert "tool_call" in events
        assert "tool_result" in events
        assert "finish" in events


@pytest.mark.asyncio
async def test_hitl_full_round_trip(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/v1/sessions", json={"agent_id": "demo", "user_id": "u1"})
        sid = r.json()["thread_id"]

        async with ac.stream(
            "POST",
            f"/v1/sessions/{sid}/chat",
            json={"content": "write a file"},
        ) as resp:
            assert resp.status_code == 200
            captured = [(name, data) async for name, data in _read_sse(resp)]
        assert any(name == "hitl_required" for name, _ in captured)
        approval_id = next(
            data["approval_id"]
            for name, data in captured
            if name == "hitl_required"
        )
        assert approval_id is not None

        r = await ac.post(
            f"/v1/sessions/{sid}/hitl/approve",
            json={"approval_id": approval_id},
        )
        assert r.status_code == 200

        async with ac.stream(
            "POST",
            f"/v1/sessions/{sid}/chat",
            json={"content": "", "approval_id": approval_id},
        ) as resp:
            assert resp.status_code == 200
            captured2 = [name async for name, _ in _read_sse(resp)]
        assert "hitl_resolved" in captured2
        assert "finish" in captured2
