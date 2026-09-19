"""Agent seed archive: config/agents.yaml loading and precedence."""

from __future__ import annotations

import pytest

from agent_platform.config.seed import AgentSeedError, load_agent_seeds
from agent_platform.config.settings import Settings
from agent_platform.infra.approval_store import ApprovalStore


def _write(tmp_path, body: str):
    p = tmp_path / "agents.yaml"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_seeding_can_be_disabled() -> None:
    assert load_agent_seeds(Settings(agent_seed_file="")) == []


def test_missing_file_yields_no_agents(tmp_path) -> None:
    s = Settings(agent_seed_file=str(tmp_path / "nope.yaml"))
    assert load_agent_seeds(s) == []


def test_minimal_entry_inherits_agent_defaults(tmp_path) -> None:
    path = _write(
        tmp_path,
        "agents:\n"
        "  - agent_id: tiny\n"
        "    model: deepseek-flash\n",
    )
    s = Settings(
        agent_seed_file=path,
        agent_default_system_prompt="Platform prompt.",
        agent_default_temperature=0.3,
        agent_default_max_tokens=256,
    )
    (cfg,) = load_agent_seeds(s)
    assert cfg.agent_id == "tiny"
    assert cfg.model == "deepseek-flash"
    assert cfg.system_prompt == "Platform prompt."
    assert cfg.temperature == 0.3
    assert cfg.max_tokens == 256


def test_explicit_fields_beat_defaults(tmp_path) -> None:
    path = _write(
        tmp_path,
        "agents:\n"
        "  - agent_id: custom\n"
        "    model: mock\n"
        "    system_prompt: Custom.\n"
        "    temperature: 0.05\n"
        "    tools: [echo]\n"
        "    sensitive_tools: []\n",
    )
    (cfg,) = load_agent_seeds(Settings(agent_seed_file=path))
    assert cfg.model == "mock"
    assert cfg.system_prompt == "Custom."
    assert cfg.temperature == 0.05
    assert cfg.tools == ["echo"]


def test_multiple_agents_and_block_scalars(tmp_path) -> None:
    path = _write(
        tmp_path,
        "agents:\n"
        "  - agent_id: a\n"
        "    model: mock\n"
        "    system_prompt: |\n"
        "      line one\n"
        "      line two\n"
        "  - agent_id: b\n"
        "    model: deepseek-flash\n"
        "    metadata:\n"
        "      owner: team-b\n",
    )
    a, b = load_agent_seeds(Settings(agent_seed_file=path))
    assert a.agent_id == "a"
    assert a.system_prompt == "line one\nline two\n"
    assert b.metadata == {"owner": "team-b"}


def test_empty_agents_list_is_fine(tmp_path) -> None:
    path = _write(tmp_path, "agents: []\n")
    assert load_agent_seeds(Settings(agent_seed_file=path)) == []


def test_invalid_yaml_raises_seed_error(tmp_path) -> None:
    path = _write(tmp_path, "agents: [unclosed\n")
    with pytest.raises(AgentSeedError, match="invalid YAML"):
        load_agent_seeds(Settings(agent_seed_file=path))


def test_missing_agent_id_raises_seed_error(tmp_path) -> None:
    path = _write(tmp_path, "agents:\n  - model: mock\n")
    with pytest.raises(AgentSeedError, match="missing `agent_id`"):
        load_agent_seeds(Settings(agent_seed_file=path))


def test_non_mapping_root_raises_seed_error(tmp_path) -> None:
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(AgentSeedError, match="expected a mapping"):
        load_agent_seeds(Settings(agent_seed_file=path))


def test_agents_not_a_list_raises_seed_error(tmp_path) -> None:
    path = _write(tmp_path, "agents: nope\n")
    with pytest.raises(AgentSeedError, match="must be a list"):
        load_agent_seeds(Settings(agent_seed_file=path))


def test_bad_field_type_raises_seed_error(tmp_path) -> None:
    path = _write(
        tmp_path, "agents:\n  - agent_id: x\n    temperature: not-a-number\n"
    )
    with pytest.raises(AgentSeedError):
        load_agent_seeds(Settings(agent_seed_file=path))


# ----- the shipped archive --------------------------------------------------


def test_shipped_agents_yaml_is_valid(repo_root) -> None:
    path = repo_root / "config" / "agents.yaml"
    assert path.exists(), "config/agents.yaml is missing from the repo"
    seeds = load_agent_seeds(Settings(agent_seed_file=str(path)))
    ids = [a.agent_id for a in seeds]
    assert "demo" in ids


def test_shipped_demo_agent_uses_deepseek_flash(repo_root) -> None:
    path = repo_root / "config" / "agents.yaml"
    seeds = load_agent_seeds(Settings(agent_seed_file=str(path)))
    demo = next(a for a in seeds if a.agent_id == "demo")
    assert demo.model == "deepseek-flash"
    assert "write_file" in demo.tools
    assert "write_file" in demo.sensitive_tools


# ----- seeding into the runtime store ---------------------------------------


@pytest.mark.asyncio
async def test_seed_applies_only_missing_agents(tmp_path) -> None:
    """A restart must not clobber agents edited in the Dashboard."""
    import fakeredis.aioredis  # type: ignore[import-untyped]

    from agent_platform.api.app import Runtime
    from agent_platform.domain.llm import MockChatModel
    from agent_platform.infra.agent_manager import AgentManager
    from agent_platform.infra.checkpoint_store import CheckpointStore
    from agent_platform.infra.config_store import AgentConfig, AgentConfigStore
    from agent_platform.infra.providers import create_chat_model
    from agent_platform.infra.secrets_store import SecretStore
    from agent_platform.tools import build_default_registry

    path = _write(
        tmp_path,
        "agents:\n"
        "  - agent_id: demo\n"
        "    model: deepseek-flash\n"
        "  - agent_id: fresh\n"
        "    model: mock\n",
    )
    s = Settings(agent_seed_file=path, checkpoint_ttl_seconds=60)
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
    rt = Runtime(
        settings=s,
        checkpoint=ckpt,
        config_store=cfg_store,
        secret_store=secret_store,
        tools=tools,
        llm=llm,
        agents=agents,
        approval_store=approvals,
    )

    # Pre-existing entry, as if the operator had edited it in the Dashboard.
    await cfg_store.upsert(AgentConfig.with_defaults("demo", s, model="mock"))

    await rt.seed_defaults()

    demo = await cfg_store.get("demo")
    fresh = await cfg_store.get("fresh")
    assert demo is not None and demo.model == "mock", "existing entry must win"
    assert fresh is not None and fresh.model == "mock"
    assert (await cfg_store.list_ids()) == ["demo", "fresh"]
    await redis.aclose()
