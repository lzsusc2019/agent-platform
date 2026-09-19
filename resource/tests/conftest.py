"""Test fixtures: fakeredis-backed CheckpointStore, default AgentLoop.

Tests must be hermetic: they may not read the developer's config files or
ambient credentials. An autouse fixture below points the YAML layers at
nonexistent paths and clears the key environment variables, so a local
`config/platform.local.yaml` can never change a test outcome.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from agent_platform.config.settings import (
    DEEPSEEK_KEY_ENV_VARS,
    LOCAL_YAML_ENV,
    PLATFORM_YAML_ENV,
    Settings,
    project_root,
)
from agent_platform.domain.agent_loop import AgentLoop
from agent_platform.domain.llm import MockChatModel
from agent_platform.infra.checkpoint_store import CheckpointStore
from agent_platform.tools import build_default_registry


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, resolved exactly the way the application does.

    Deliberately NOT `Path(__file__).parent.parent`. That form encodes how
    deep the test file happens to sit, so moving the suite — which happened,
    when it went under resource/ — silently repointed every path at the wrong
    directory. The symptom is a missing config file, which reads as a broken
    checkout rather than a stale assumption.

    `project_root()` walks up looking for pyproject.toml, so it survives any
    future move, and using it here means tests and production agree on where
    the repository is by construction.
    """
    root = project_root()
    assert root is not None and isinstance(root, Path), (
        "project_root() found no pyproject.toml; tests must run from a "
        "source checkout"
    )
    return root


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
