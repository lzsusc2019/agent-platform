"""AgentConfigStore — the agent_id -> business-config directory.

Each AgentConfig describes:
- system_prompt: instructions for the LLM
- model: which LLM provider/model to use (e.g. "mock", "deepseek:deepseek-flash")
- temperature: sampling parameter
- max_tokens: per-response cap
- tools: list of tool names allowed for this agent
- skills: list of skill ids (MCP or local)
- sensitive_tools: subset of `tools` that need HITL approval
- metadata: free-form (owner team, version, etc.)

The store is Redis-backed: key = `agent:{agent_id}`, JSON-serialized
AgentConfig. There is no "create" operation — `upsert` covers both first
write and update. Listing reads all keys with a single KEYS scan (acceptable
for MVP; tens of agents).

The per-AgentConfig defaults come from `Settings.agent_default_*` so an
operator can retune them without touching code. Agent *definitions* — the
seed archive applied at startup — live in `config/agents.yaml` and are
loaded by `agent_platform.agent_seed`.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import Settings

log = logging.getLogger(__name__)


def agent_key(agent_id: str) -> str:
    return f"agent:{agent_id}"


class AgentConfig(BaseModel):
    """The business config for one Agent, keyed by agent_id."""

    agent_id: str
    system_prompt: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    sensitive_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def with_defaults(cls, agent_id: str, settings: Settings, **overrides: Any) -> AgentConfig:
        """Build a config seeded from `settings.agent_default_*`.

        Used by the Admin API when creating a brand-new agent, and by the
        demo seed. `overrides` win over the settings defaults.
        """
        base: dict[str, Any] = {
            "agent_id": agent_id,
            "system_prompt": settings.agent_default_system_prompt,
            "model": settings.agent_default_model,
            "temperature": settings.agent_default_temperature,
            "max_tokens": settings.agent_default_max_tokens,
            "tools": [],
            "skills": [],
            "sensitive_tools": [],
            "metadata": {},
        }
        base.update(overrides)
        return cls(**base)

    def to_loop_config(self) -> dict[str, Any]:
        """Render the dict shape AgentLoop expects."""
        return {
            "system_prompt": self.system_prompt,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": list(self.tools),
            "skills": list(self.skills),
            "sensitive_tools": list(self.sensitive_tools),
            "metadata": dict(self.metadata),
        }


class AgentConfigStore:
    """Thin async wrapper over Redis for AgentConfig CRUD."""

    def __init__(self, redis: Any, ttl_seconds: int | None = None) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, ttl_seconds: int | None = None) -> AgentConfigStore:
        import redis.asyncio as aioredis

        return cls(aioredis.from_url(url), ttl_seconds=ttl_seconds)

    async def get(self, agent_id: str) -> AgentConfig | None:
        raw = await self._redis.get(agent_key(agent_id))
        if raw is None:
            return None
        try:
            return AgentConfig.model_validate_json(raw)
        except Exception as e:
            log.warning(
                "agent_config.deserialize_failed agent_id=%s error=%s",
                agent_id,
                e,
            )
            return None

    async def upsert(self, cfg: AgentConfig) -> None:
        payload = cfg.model_dump_json()
        if self._ttl is None:
            await self._redis.set(agent_key(cfg.agent_id), payload)
        else:
            await self._redis.set(agent_key(cfg.agent_id), payload, ex=self._ttl)

    async def delete(self, agent_id: str) -> bool:
        n = await self._redis.delete(agent_key(agent_id))
        return n > 0

    async def list_ids(self) -> list[str]:
        keys = await self._redis.keys("agent:*")
        return sorted(k.decode().removeprefix("agent:") for k in keys)

    async def list_all(self) -> list[AgentConfig]:
        ids = await self.list_ids()
        out: list[AgentConfig] = []
        for aid in ids:
            cfg = await self.get(aid)
            if cfg is not None:
                out.append(cfg)
        return out

