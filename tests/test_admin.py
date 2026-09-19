"""Tests for AgentConfigStore + admin API."""

from __future__ import annotations

from datetime import UTC

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_platform.api.app import Runtime, create_app
from agent_platform.approvals import ApprovalStore
from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import Settings
from agent_platform.config_store import AgentConfig, AgentConfigStore
from agent_platform.core.llm import MockChatModel
from agent_platform.store.agent_manager import AgentManager
from agent_platform.tools import build_default_registry

# ----- AgentConfigStore -----------------------------------------------------


@pytest_asyncio.fixture
async def cfg_store() -> AgentConfigStore:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    redis = fakeredis.aioredis.FakeRedis()
    yield AgentConfigStore(redis)
    await redis.aclose()


@pytest.mark.asyncio
async def test_upsert_get_roundtrip(cfg_store: AgentConfigStore) -> None:
    cfg = AgentConfig(agent_id="a", system_prompt="hi", tools=["echo"])
    await cfg_store.upsert(cfg)
    loaded = await cfg_store.get("a")
    assert loaded is not None
    assert loaded.system_prompt == "hi"
    assert loaded.tools == ["echo"]


@pytest.mark.asyncio
async def test_get_missing_returns_none(cfg_store: AgentConfigStore) -> None:
    assert await cfg_store.get("missing") is None


@pytest.mark.asyncio
async def test_list_ids_sorted(cfg_store: AgentConfigStore) -> None:
    await cfg_store.upsert(AgentConfig(agent_id="b"))
    await cfg_store.upsert(AgentConfig(agent_id="a"))
    await cfg_store.upsert(AgentConfig(agent_id="c"))
    assert await cfg_store.list_ids() == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_delete(cfg_store: AgentConfigStore) -> None:
    await cfg_store.upsert(AgentConfig(agent_id="x"))
    assert await cfg_store.delete("x") is True
    assert await cfg_store.delete("x") is False


def test_to_loop_config_shape() -> None:
    cfg = AgentConfig(
        agent_id="x",
        system_prompt="s",
        model="openai:gpt-4o",
        temperature=0.3,
        tools=["a", "b"],
        sensitive_tools=["b"],
    )
    out = cfg.to_loop_config()
    assert out["system_prompt"] == "s"
    assert out["model"] == "openai:gpt-4o"
    assert out["tools"] == ["a", "b"]
    assert out["sensitive_tools"] == ["b"]


# ----- admin API ------------------------------------------------------------


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
    from agent_platform.core.providers import create_chat_model
    from agent_platform.secrets_store import SecretStore

    secret_store = SecretStore(redis)
    approvals = ApprovalStore(redis, ttl_seconds=s.approval_grant_ttl_seconds)
    tools = build_default_registry(s)
    llm = MockChatModel()
    agents = AgentManager(
        llm=llm,
        tools=tools,
        checkpoint=ckpt,
        settings=s,
        config_store=cfg_store,
        secret_store=secret_store,
        approvals=approvals,
        llm_factory=lambda model, key: create_chat_model(model, s, api_key_override=key),
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
async def test_list_configs_empty(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/admin/api/configs")
        assert r.status_code == 200
        assert r.json() == []


@pytest.mark.asyncio
async def test_upsert_then_get_config(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.put(
            "/admin/api/configs/qa",
            json={
                "system_prompt": "you are a QA bot",
                "model": "mock",
                "temperature": 0.2,
                "max_tokens": 1024,
                "tools": ["echo"],
                "sensitive_tools": [],
                "skills": [],
                "metadata": {"team": "qa"},
            },
        )
        assert r.status_code == 200, r.text
        cfg = r.json()
        assert cfg["agent_id"] == "qa"
        assert cfg["system_prompt"] == "you are a QA bot"

        r = await ac.get("/admin/api/configs/qa")
        assert r.status_code == 200
        assert r.json()["system_prompt"] == "you are a QA bot"

        r = await ac.get("/admin/api/configs")
        ids = [c["agent_id"] for c in r.json()]
        assert "qa" in ids


@pytest.mark.asyncio
async def test_partial_update_preserves_other_fields(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.put(
            "/admin/api/configs/x",
            json={"system_prompt": "p1", "temperature": 0.5, "tools": ["echo"]},
        )
        # Update only temperature.
        await ac.put("/admin/api/configs/x", json={"temperature": 0.9})
        r = await ac.get("/admin/api/configs/x")
        cfg = r.json()
        assert cfg["system_prompt"] == "p1"
        assert cfg["temperature"] == 0.9
        assert cfg["tools"] == ["echo"]


@pytest.mark.asyncio
async def test_delete_config(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.put("/admin/api/configs/z", json={"system_prompt": "x"})
        r = await ac.delete("/admin/api/configs/z")
        assert r.status_code == 200
        assert r.json() == {"deleted": True}
        r = await ac.get("/admin/api/configs/z")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_unknown_config_404(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/admin/api/configs/nope")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_list_tools(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/admin/api/tools")
        assert r.status_code == 200
        names = {t["name"] for t in r.json()}
        assert "echo" in names
        assert "http_get" in names
        assert "write_file" in names
        sensitive = {t["name"] for t in r.json() if t["sensitive"]}
        assert sensitive == {"write_file"}


@pytest.mark.asyncio
async def test_checkpoints_list_and_get(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Empty.
        r = await ac.get("/admin/api/checkpoints")
        assert r.status_code == 200
        assert r.json() == []

        # Create one via debug chat.
        async with ac.stream(
            "POST",
            "/admin/api/chat",
            json={"agent_id": "demo", "content": "please echo hi"},
        ) as resp:
            assert resp.status_code == 200
            tid = resp.headers.get("x-thread-id")
            assert tid
            async for _ in resp.aiter_lines():
                pass

        r = await ac.get("/admin/api/checkpoints")
        rows = r.json()
        assert len(rows) >= 1
        assert any(r["thread_id"] == tid for r in rows)

        r = await ac.get(f"/admin/api/checkpoints/{tid}")
        assert r.status_code == 200
        snap = r.json()
        assert snap["thread_id"] == tid
        assert snap["status"] == "finished"


@pytest.mark.asyncio
async def test_hitl_pending_listing(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger HITL.
        async with ac.stream(
            "POST",
            "/admin/api/chat",
            json={"agent_id": "demo", "content": "write a file"},
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_lines():
                pass

        r = await ac.get("/admin/api/hitl/pending")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "write_file"


@pytest.mark.asyncio
async def test_admin_index_serves_html(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/admin/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Agent Platform" in r.text


# ----- Secrets (Providers tab) -----


@pytest.mark.asyncio
async def test_secrets_list_when_empty(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/admin/api/secrets")
        assert r.status_code == 200
        rows = r.json()
        # Known slots always returned, even when empty.
        assert any(s["provider"] == "deepseek" and s["name"] == "api_key" for s in rows)
        deepseek = next(s for s in rows if s["provider"] == "deepseek")
        assert deepseek["has_value"] is False
        assert deepseek["source"] is None


@pytest.mark.asyncio
async def test_secrets_upsert_then_list_shows_masked(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.put(
            "/admin/api/secrets/deepseek/api_key",
            json={"value": "sk-secret-1234567890abcdef", "note": "primary"},
        )
        assert r.status_code == 200
        assert r.json()["saved"] is True

        r = await ac.get("/admin/api/secrets")
        deepseek = next(s for s in r.json() if s["provider"] == "deepseek")
        assert deepseek["has_value"] is True
        assert deepseek["source"] == "store"
        # Never the raw value, only the masked prefix/suffix.
        assert "1234567890abcdef" not in deepseek["masked"]
        assert deepseek["masked"].startswith("sk-s")
        assert deepseek["masked"].endswith("cdef")
        assert deepseek["note"] == "primary"


@pytest.mark.asyncio
async def test_secrets_delete_clears(runtime: Runtime) -> None:
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.put(
            "/admin/api/secrets/deepseek/api_key",
            json={"value": "sk-abc", "note": ""},
        )
        r = await ac.delete("/admin/api/secrets/deepseek/api_key")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        r = await ac.get("/admin/api/secrets")
        deepseek = next(s for s in r.json() if s["provider"] == "deepseek")
        assert deepseek["has_value"] is False


@pytest.mark.asyncio
async def test_secrets_upsert_invalidates_cached_agent(runtime: Runtime) -> None:
    """Changing the key should drop any cached AgentLoop for that provider."""
    # Seed a real key first so DeepSeekChatModel can be built.
    from datetime import datetime

    from agent_platform.config_store import AgentConfig
    from agent_platform.core.providers import DeepSeekChatModel
    from agent_platform.secrets_store import SecretEntry

    await runtime.secret_store.set(
        SecretEntry(
            provider="deepseek",
            name="api_key",
            value="sk-initial",
            updated_at=datetime.now(UTC),
        )
    )

    await runtime.config_store.upsert(
        AgentConfig(agent_id="d", model="deepseek:deepseek-chat")
    )
    # Populate the cache.
    a = await runtime.agents.get_or_create("d")
    first_llm = a.llm
    assert isinstance(first_llm, DeepSeekChatModel)

    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.put(
            "/admin/api/secrets/deepseek/api_key",
            json={"value": "sk-rotated", "note": "rotated"},
        )
        assert r.status_code == 200
        # The upsert endpoint should have invalidated the cached loop.
        assert "d" in r.json()["invalidated_agents"]

    b = await runtime.agents.get_or_create("d")
    assert b is not a
    assert isinstance(b.llm, DeepSeekChatModel)


@pytest.mark.asyncio
async def test_secrets_env_only_when_no_store_value(runtime: Runtime) -> None:
    """Without a stored secret, the source pill should show 'env' if the
    env var is set, otherwise None."""
    # runtime fixture has no env key by default.
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/admin/api/secrets")
        rows = r.json()
        deepseek = next(s for s in rows if s["provider"] == "deepseek")
        # No env key on the test settings, no store value -> source=None.
        assert deepseek["source"] is None
