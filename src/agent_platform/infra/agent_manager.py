"""AgentManager — registry of live AgentLoop instances, keyed by agent_id.

This is the MVP equivalent of Agent中台.md's `agentMap`. Hot-reload (polling
threads, atomic Map swap, retired-instance queue) is intentionally not
implemented; see ADR-001.

The Manager consults an `AgentConfigStore` on `get_or_create`. If the store
has a config for the requested agent_id, its `system_prompt` is passed to
the new AgentLoop, AND the `model` field is used to construct the right
ChatModel via the supplied `llm_factory`. If no config exists, a default
loop with the built-in system prompt and the default LLM is created.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from agent_platform.config.settings import Settings
from agent_platform.domain.agent_loop import AgentLoop
from agent_platform.domain.llm import ChatModel
from agent_platform.domain.tool import ToolRegistry
from agent_platform.infra.approval_store import ApprovalStore
from agent_platform.infra.checkpoint_store import CheckpointStore
from agent_platform.infra.config_store import AgentConfigStore
from agent_platform.infra.secrets_store import SecretStore

log = logging.getLogger(__name__)


# Type alias for the LLM factory. Given an AgentConfig.model string
# and an optional resolved API key (from SecretStore), returns a ChatModel
# instance. See core/providers.py for the default.
LLMFactory = Callable[[str, str | None], ChatModel]


class AgentManager:
    """Per-process registry mapping `agent_id` -> `AgentLoop`."""

    def __init__(
        self,
        *,
        llm: ChatModel,
        tools: ToolRegistry,
        checkpoint: CheckpointStore,
        settings: Settings,
        config_store: AgentConfigStore | None = None,
        secret_store: SecretStore | None = None,
        default_system_prompt: str | None = None,
        llm_factory: LLMFactory | None = None,
        approvals: ApprovalStore | None = None,
    ) -> None:
        # `llm` is the fallback used when a config has no `model` field or
        # the model string is the built-in default.
        self._llm = llm
        self._tools = tools
        self._checkpoint = checkpoint
        self._settings = settings
        self._configs = config_store
        self._secret_store = secret_store
        # Fall back to the configured default rather than a literal, so the
        # platform-wide prompt lives in one place (Settings).
        self._default_system_prompt = (
            default_system_prompt
            if default_system_prompt is not None
            else settings.agent_default_system_prompt
        )
        self._llm_factory = llm_factory
        self._approvals = approvals
        self._agents: dict[str, AgentLoop] = {}
        self._lock = asyncio.Lock()
        # Track the (updated_at, provider) pairs we've seen so we can
        # invalidate cached loops when a secret changes. Keyed by
        # (provider, name).
        self._secret_versions: dict[tuple[str, str], str] = {}

    @property
    def config_store(self) -> AgentConfigStore | None:
        return self._configs

    @property
    def secret_store(self) -> SecretStore | None:
        return self._secret_store

    def get(self, agent_id: str) -> AgentLoop:
        try:
            return self._agents[agent_id]
        except KeyError as e:
            raise KeyError(f"Agent '{agent_id}' not registered") from e

    async def get_or_create(
        self, agent_id: str, config_override: dict[str, Any] | None = None
    ) -> AgentLoop:
        # Fast path.
        if agent_id in self._agents:
            return self._agents[agent_id]
        async with self._lock:
            if agent_id in self._agents:
                return self._agents[agent_id]

            # Resolve config: caller override > stored config > default.
            stored = await self._configs.get(agent_id) if self._configs else None
            system_prompt = self._default_system_prompt
            loop_config: dict[str, Any] = {"agent_id": agent_id}
            if stored is not None:
                system_prompt = stored.system_prompt
                loop_config.update(stored.to_loop_config())
            if config_override:
                loop_config.update(config_override)

            # Pick the LLM. If a stored config has a `model` string, route
            # through the factory so per-agent model selection works. If
            # not, fall back to the manager's default LLM.
            model_string = str(loop_config.get("model", "")).strip()
            llm = await self._resolve_llm(model_string)

            agent = AgentLoop(
                llm=llm,
                tools=self._tools,
                checkpoint=self._checkpoint,
                system_prompt=system_prompt,
                settings=self._settings,
                config=loop_config,
                agent_id=agent_id,
                approvals=self._approvals,
            )
            self._agents[agent_id] = agent
            log.info(
                "agent.registered agent_id=%s has_config=%s model=%s",
                agent_id,
                stored is not None,
                model_string or "(default)",
            )
            return agent

    async def _resolve_llm(self, model_string: str) -> ChatModel:
        """Map a config.model string to a ChatModel instance.

        Also consults the SecretStore for the provider's API key,
        falling back to `settings.deepseek_api_key`.
        """
        if not model_string:
            return self._llm
        if self._llm_factory is None:
            raise RuntimeError(
                f"AgentConfig.model='{model_string}' requires a llm_factory, "
                f"but AgentManager was constructed without one. Use "
                f"core.providers.create_chat_model in production."
            )
        # Resolve the provider id and look up its key in the SecretStore.
        from agent_platform.infra.providers import parse_model_string

        provider, _ = parse_model_string(model_string)
        api_key = await self._resolve_provider_key(provider)
        return self._llm_factory(model_string, api_key)

    async def _resolve_provider_key(self, provider: str) -> str | None:
        """Look up a provider's API key with override-then-env precedence.

        If a secret is found, also records its version so subsequent
        get_or_create calls can detect rotation.
        """
        # We only know the key shape for the built-in deepseek provider.
        # Other providers can override via their own provider class.
        if provider != "deepseek":
            return None
        if self._secret_store is not None:
            entry = await self._secret_store.get("deepseek", "api_key")
            if entry is not None:
                self._secret_versions[("deepseek", "api_key")] = (
                    entry.updated_at.isoformat()
                )
                # A value written before the write-time validator existed
                # (or edited straight in Redis) can still be hostile. Reject
                # it here with the same actionable message rather than
                # letting httpx raise an opaque UnicodeEncodeError later.
                from agent_platform.infra.providers import validate_api_key

                candidate = entry.value.strip()
                try:
                    validate_api_key(candidate)
                except ValueError as e:
                    raise ValueError(
                        f"stored secret 'deepseek/api_key' is unusable: {e}. "
                        "Re-save it from the Dashboard Providers tab."
                    ) from e
                return candidate
        return self._settings.deepseek_api_key or None

    async def invalidate_stale_secrets(self) -> list[str]:
        """Invalidate cached AgentLoops whose provider key has rotated.

        Iterates over EVERY known secret slot in the store, comparing
        the current `updated_at` (or absence) against what we've recorded.
        If a secret has changed — or appeared for the first time since
        the loop was built (the case after a fresh key is saved through
        the Dashboard) — drop the loop so the next chat rebuilds it
        with the new key.

        Returns the list of invalidated agent_ids. Call this from the
        secrets PUT/DELETE endpoints (or on a timer) to make secret
        changes take effect without a process restart.
        """
        from agent_platform.infra.providers import parse_model_string

        invalidated: list[str] = []
        if self._secret_store is None:
            return invalidated
        # Walk every secret the store currently knows about. This
        # catches the "loop was built when there was no key, now there
        # is one" case that the per-seen-version tracking misses.
        seen_keys: set[tuple[str, str]] = set()
        for entry in await self._secret_store.list():
            slot = (entry.provider, entry.name)
            seen_keys.add(slot)
            current_version = entry.updated_at.isoformat()
            prior = self._secret_versions.get(slot, "")
            if current_version != prior:
                for agent_id, loop in list(self._agents.items()):
                    model = str(loop.config.get("model", ""))
                    if not model:
                        continue
                    p, _ = parse_model_string(model)
                    if p == entry.provider:
                        self._agents.pop(agent_id, None)
                        invalidated.append(agent_id)
                self._secret_versions[slot] = current_version
        # Also catch deletions: any slot we previously saw but no
        # longer exists should also trigger invalidation (so the next
        # chat picks up the env-var fallback path or just goes mock).
        for slot in list(self._secret_versions.keys()):
            if slot in seen_keys:
                continue
            provider, _ = slot
            current_version = ""
            prior = self._secret_versions[slot]
            if current_version != prior:
                for agent_id, loop in list(self._agents.items()):
                    model = str(loop.config.get("model", ""))
                    if not model:
                        continue
                    p, _ = parse_model_string(model)
                    if p == provider:
                        self._agents.pop(agent_id, None)
                        invalidated.append(agent_id)
                self._secret_versions[slot] = current_version
        return invalidated

    def register(self, agent_id: str, agent: AgentLoop) -> None:
        """Pre-register an instance (useful in tests)."""
        self._agents[agent_id] = agent

    def invalidate(self, agent_id: str) -> bool:
        """Drop the cached AgentLoop so the next get_or_create rebuilds it
        from the latest AgentConfig. Returns True if an instance was removed.
        """
        return self._agents.pop(agent_id, None) is not None

    # TODO: hot-reload loop. Per Agent中台.md: a thread polls config every 90s,
    # builds new instances in parallel, atomically swaps the Map, queues the
    # old instance, and another thread drains the queue every 60s.
