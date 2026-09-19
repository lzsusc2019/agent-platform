"""Test fixtures: fakeredis-backed CheckpointStore, default AgentLoop.

Tests must be hermetic: they may not read the developer's config files or
ambient credentials. An autouse fixture below points the YAML layers at
nonexistent paths and clears the key environment variables, so a local
`config/platform.local.yaml` can never change a test outcome.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import (
    DEEPSEEK_KEY_ENV_VARS,
    LOCAL_YAML_ENV,
    PLATFORM_YAML_ENV,
    Settings,
)
from agent_platform.core.agent_loop import AgentLoop
from agent_platform.core.llm import MockChatModel
from agent_platform.tools import build_default_registry


@pytest.fixture(autouse=True)
def _hermetic_config(monkeypatch, tmp_path_factory):
    """Detach every test from the developer's real configuration.

    Without this, a `config/platform.local.yaml` on the machine running the
    suite leaks into `Settings()` and produces failures that reproduce on
    exactly one computer.
    """
    sandbox = tmp_path_factory.mktemp("no-config")
    monkeypatch.setenv(PLATFORM_YAML_ENV, str(sandbox / "absent-platform.yaml"))
    monkeypatch.setenv(LOCAL_YAML_ENV, str(sandbox / "absent-local.yaml"))
    for var in DEEPSEEK_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Keep the sensitive tool's writes out of the repo. Without this, any test
    # that approves a write_file call drops a `workspace/` directory next to
    # pyproject.toml.
    monkeypatch.setenv(
        "AGENT_PLATFORM_TOOL_WRITE_FILE_ROOT",
        str(tmp_path_factory.mktemp("workspace")),
    )
    yield


@pytest.fixture
def settings(tmp_path_factory) -> Settings:
    """Tuned for fast tests: tiny compression trigger, low retry budget.

    The write_file root points at a temp directory so a test that exercises
    the sensitive tool never drops files into the repo.
    """
    return Settings(
        llm_max_retries=2,
        empty_response_max_retries=2,
        llm_retry_base_delay=0.0,
        compress_trigger_tokens=100,  # force compression in compression test
        compress_keep_recent_turns=2,
        checkpoint_ttl_seconds=60,
        tool_write_file_root=str(tmp_path_factory.mktemp("workspace")),
    )


@pytest_asyncio.fixture
async def ckpt_store() -> CheckpointStore:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    redis = fakeredis.aioredis.FakeRedis()
    yield CheckpointStore(redis, ttl_seconds=60)
    await redis.aclose()


@pytest.fixture
def tools(settings: Settings):
    return build_default_registry(settings)


@pytest.fixture
def llm() -> MockChatModel:
    return MockChatModel()


@pytest.fixture
def agent(ckpt_store, tools, llm, settings) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        tools=tools,
        checkpoint=ckpt_store,
        system_prompt="You are a test agent.",
        settings=settings,
        agent_id="test-agent",
    )


async def drain(agent: AgentLoop, **kwargs):
    """Collect all events from a Loop run into a list."""
    out = []
    async for ev in agent.run(**kwargs):
        out.append(ev)
    return out
