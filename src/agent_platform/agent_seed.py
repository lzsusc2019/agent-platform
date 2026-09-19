"""Load agent definitions from a YAML archive.

Agent definitions live in `config/agents.yaml` rather than in environment
variables: they are structured (nested tool lists, prompt blocks, metadata),
they are worth reviewing in a diff, and there are usually several of them.
Env vars stay for scalars and secrets.

Schema:

    agents:
      - agent_id: demo
        model: deepseek-flash
        system_prompt: |
          You are a helpful assistant.
        temperature: 0.7
        max_tokens: 2048
        tools: [echo, http_get, write_file]
        sensitive_tools: [write_file]
        skills: []
        metadata: {owner: platform-team}

Omitted fields fall back to the `agent_default_*` settings, so a minimal
entry is just `{agent_id: x, model: y}`.
"""

from __future__ import annotations

import logging

import yaml
from pydantic import BaseModel, Field, ValidationError

from agent_platform.config import Settings, resolve_config_path
from agent_platform.config_store import AgentConfig

log = logging.getLogger(__name__)


class AgentSeedFile(BaseModel):
    """Top-level shape of config/agents.yaml."""

    agents: list[AgentConfig] = Field(default_factory=list)


class AgentSeedError(Exception):
    """The seed file exists but could not be used."""


def load_agent_seeds(settings: Settings) -> list[AgentConfig]:
    """Read the agent seed file and fill in defaults.

    Returns an empty list — with a log line explaining why — when seeding is
    disabled or the file is absent. A file that exists but is malformed or
    fails validation raises `AgentSeedError`: that is an operator mistake
    worth failing loudly for, not something to silently skip.
    """
    raw_path = (settings.agent_seed_file or "").strip()
    if not raw_path:
        log.info("agent_seed.disabled (AGENT_PLATFORM_AGENT_SEED_FILE is empty)")
        return []

    path = resolve_config_path(raw_path)
    if not path.exists():
        log.warning(
            "agent_seed.file_missing path=%s — starting with no seeded agents",
            path,
        )
        return []

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise AgentSeedError(f"{path}: invalid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise AgentSeedError(
            f"{path}: expected a mapping with an `agents:` key, got "
            f"{type(raw).__name__}"
        )

    # Merge each entry over the configured defaults so a seed file only has
    # to state what differs.
    entries = raw.get("agents") or []
    if not isinstance(entries, list):
        raise AgentSeedError(f"{path}: `agents` must be a list")

    defaults = AgentConfig.with_defaults("__seed__", settings)
    default_fields = defaults.model_dump(exclude={"agent_id"})

    merged: list[dict] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AgentSeedError(f"{path}: agents[{i}] must be a mapping")
        if not entry.get("agent_id"):
            raise AgentSeedError(f"{path}: agents[{i}] is missing `agent_id`")
        merged.append({**default_fields, **entry})

    try:
        parsed = AgentSeedFile.model_validate({"agents": merged})
    except ValidationError as e:
        raise AgentSeedError(f"{path}: {e}") from e

    log.info(
        "agent_seed.loaded path=%s agents=%s",
        path,
        [a.agent_id for a in parsed.agents],
    )
    return parsed.agents
