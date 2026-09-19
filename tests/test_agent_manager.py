"""Tests for AgentManager's per-agent model routing via llm_factory."""

from __future__ import annotations

import pytest
import pytest_asyncio

from agent_platform.config.settings import Settings
from agent_platform.domain.llm import ChatModel, LLMResponse, MockChatModel
from agent_platform.infra.agent_manager import AgentManager
from agent_platform.infra.checkpoint_store import CheckpointStore
from agent_platform.infra.config_store import AgentConfig, AgentConfigStore
from agent_platform.infra.providers import DeepSeekChatModel
from agent_platform.tools import build_default_registry


class _FixedModel(ChatModel):
    """Identifiable mock LLM for assertions in factory tests."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    async def ainvoke(self, messages, tools):
        return LLMResponse(content=f"hello from {self.tag}")


@pytest_asyncio.fixture
async def mgr():
    import fakeredis.aioredis  # type: ignore[import-untyped]

    s = Settings(
        deepseek_api_key="",
        checkpoint_ttl_seconds=60,
        llm_max_retries=1,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
    )
    redis = fakeredis.aioredis.FakeRedis()
    ckpt = CheckpointStore(redis, ttl_seconds=60)
    cfg_store = AgentConfigStore(redis)
    tools = build_default_registry(s)
    llm = MockChatModel()
    factory_calls: list = []

    def factory(model_string: str, api_key: str | None = None) -> ChatModel:
        factory_calls.append((model_string, api_key))
        if model_string.startswith("deepseek"):
            return DeepSeekChatModel(
                api_key=api_key or "sk-test",
                base_url=s.deepseek_base_url,
                model=s.deepseek_model,
                timeout=s.deepseek_timeout,
            )
        if model_string.startswith("mock"):
            return _FixedModel("mock-via-factory")
        return _FixedModel(model_string)

    agents = AgentManager(
        llm=llm,
        tools=tools,
        checkpoint=ckpt,
        settings=s,
        config_store=cfg_store,
        llm_factory=factory,
    )
    yield {
        "agents": agents,
        "cfg_store": cfg_store,
        "factory_calls": factory_calls,
    }
    await redis.aclose()


@pytest.mark.asyncio
async def test_default_llm_used_when_no_config(mgr) -> None:
    agents = mgr["agents"]
    agent = await agents.get_or_create("ad-hoc")
    # No config stored -> no model field -> default LLM, no factory call.
    assert agent.llm is agents._llm
    assert mgr["factory_calls"] == []


@pytest.mark.asyncio
async def test_factory_used_when_model_field_set(mgr) -> None:
    await mgr["cfg_store"].upsert(
        AgentConfig(agent_id="ds", model="deepseek:deepseek-chat")
    )
    agent = await mgr["agents"].get_or_create("ds")
    assert isinstance(agent.llm, DeepSeekChatModel)
    assert mgr["factory_calls"][0][0] == "deepseek:deepseek-chat"


@pytest.mark.asyncio
async def test_factory_routes_by_provider_prefix(mgr) -> None:
    await mgr["cfg_store"].upsert(
        AgentConfig(agent_id="mockagent", model="mock")
    )
    agent = await mgr["agents"].get_or_create("mockagent")
    # Routed through factory -> distinguishable instance.
    assert isinstance(agent.llm, _FixedModel)
    assert agent.llm.tag == "mock-via-factory"


@pytest.mark.asyncio
async def test_invalidate_drops_loop_so_new_model_picks_up(mgr) -> None:
    await mgr["cfg_store"].upsert(
        AgentConfig(agent_id="switchable", model="mock")
    )
    first = await mgr["agents"].get_or_create("switchable")
    first_id = id(first.llm)

    # Switch the config to deepseek and invalidate the cached loop.
    await mgr["cfg_store"].upsert(
        AgentConfig(agent_id="switchable", model="deepseek:deepseek-chat")
    )
    mgr["agents"].invalidate("switchable")
    second = await mgr["agents"].get_or_create("switchable")
    assert id(second.llm) != first_id
    assert isinstance(second.llm, DeepSeekChatModel)


@pytest.mark.asyncio
async def test_no_factory_with_explicit_model_raises(mgr) -> None:
    """A typo'd model field without a factory should fail loud."""
    # Construct a manager WITHOUT a factory.
    import fakeredis.aioredis  # type: ignore[import-untyped]

    s = Settings(checkpoint_ttl_seconds=60)
    redis = fakeredis.aioredis.FakeRedis()
    ckpt = CheckpointStore(redis, ttl_seconds=60)
    cfg_store = AgentConfigStore(redis)
    await cfg_store.upsert(AgentConfig(agent_id="x", model="deepseek"))
    agents = AgentManager(
        llm=MockChatModel(),
        tools=build_default_registry(s),
        checkpoint=ckpt,
        settings=s,
        config_store=cfg_store,
        # no llm_factory
    )
    with pytest.raises(RuntimeError, match="llm_factory"):
        await agents.get_or_create("x")
    await redis.aclose()
