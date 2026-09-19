"""FastAPI app factory + dependency wiring.

The app takes a `runtime` object on construction so tests can inject a
fakeredis-backed runtime without going through env vars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import FastAPI

from agent_platform.agent_seed import AgentSeedError, load_agent_seeds
from agent_platform.api.routes import router as api_router
from agent_platform.approvals import ApprovalStore
from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import Settings
from agent_platform.config_store import AgentConfigStore
from agent_platform.core.llm import ChatModel, MockChatModel
from agent_platform.core.tool import ToolRegistry
from agent_platform.secrets_store import SecretStore
from agent_platform.store.agent_manager import AgentManager
from agent_platform.tools import build_default_registry

log = logging.getLogger(__name__)

__all__ = ["Runtime", "create_app", "AgentSeedError"]


@dataclass
class Runtime:
    """All long-lived singletons the API needs.

    Bundled so tests can swap them out in one shot.
    """

    settings: Settings
    checkpoint: CheckpointStore
    config_store: AgentConfigStore
    secret_store: SecretStore
    tools: ToolRegistry
    llm: ChatModel
    agents: AgentManager
    # Where human approvals live. Required, not optional: without it the HITL
    # gate has nothing to consult and an approval_id in the request becomes the
    # only signal, which is exactly the hole this closes.
    approval_store: ApprovalStore

    @classmethod
    def default(cls, settings: Settings | None = None) -> "Runtime":
        s = settings or Settings()
        if s.use_fake_redis:
            import fakeredis.aioredis  # type: ignore[import-untyped]

            redis = fakeredis.aioredis.FakeRedis()
            ckpt = CheckpointStore(redis, ttl_seconds=s.checkpoint_ttl_seconds)
            cfg_store = AgentConfigStore(redis)
            secret_store = SecretStore(redis)
            approvals = ApprovalStore(redis, ttl_seconds=s.approval_grant_ttl_seconds)
        else:
            ckpt = CheckpointStore.from_url(
                s.redis_url, ttl_seconds=s.checkpoint_ttl_seconds
            )
            cfg_store = AgentConfigStore.from_url(s.redis_url)
            secret_store = SecretStore.from_url(s.redis_url)
            approvals = ApprovalStore.from_url(
                s.redis_url, ttl_seconds=s.approval_grant_ttl_seconds
            )
        tools = build_default_registry(s)
        llm = MockChatModel()  # default fallback when no model field is set
        from agent_platform.core.providers import create_chat_model

        # The factory signature is (model_string, api_key_override). The
        # AgentManager calls it with the key it just resolved from the
        # SecretStore, so we don't need to look it up again here.
        agents = AgentManager(
            llm=llm,
            tools=tools,
            checkpoint=ckpt,
            settings=s,
            config_store=cfg_store,
            secret_store=secret_store,
            approvals=approvals,
            llm_factory=lambda model, key: create_chat_model(
                model, s, api_key_override=key
            ),
        )
        return cls(
            settings=s,
            checkpoint=ckpt,
            config_store=cfg_store,
            secret_store=secret_store,
            tools=tools,
            llm=llm,
            agents=agents,
            approval_store=approvals,
        )

    def log_configuration(self) -> None:
        """Report what was injected, once, at startup.

        Answers two questions the operator would otherwise have to grep for:
        which LLM is actually live, and where the credential came from. The
        key itself is never printed — only its source and a masked hint.
        """
        s = self.settings
        key = s.deepseek_api_key
        masked = ""
        if key:
            masked = (
                key[:7] + "..." + key[-4:] if len(key) > 14 else "***"
            )
        log.info(
            "config.llm provider=%s model=%s key=%s source=%s",
            type(self.llm).__name__,
            s.deepseek_model,
            masked or "<unset>",
            s.deepseek_key_source(),
        )
        for warning in s.credential_warnings():
            log.warning("config.warning %s", warning)

    async def seed_defaults(self) -> None:
        """Apply the agent archive from config/agents.yaml.

        Writes each archived agent whose `agent_id` is not already in the
        runtime store, so a restart never clobbers a Dashboard edit. A
        malformed file raises `AgentSeedError` — better to refuse to start
        than to boot with agents silently missing.

        Call this from the FastAPI startup hook (or right after creating the
        Runtime). Synchronous callers can use `asyncio.run(...)`.
        """
        seeds = load_agent_seeds(self.settings)
        for cfg in seeds:
            existing = await self.config_store.get(cfg.agent_id)
            if existing is not None:
                log.info(
                    "agent_seed.skipped agent_id=%s (already in the store)",
                    cfg.agent_id,
                )
                continue
            await self.config_store.upsert(cfg)
            log.info(
                "agent_seed.applied agent_id=%s model=%s tools=%s",
                cfg.agent_id,
                cfg.model,
                cfg.tools,
            )


def create_app(runtime: Runtime | None = None) -> FastAPI:
    rt = runtime or Runtime.default()
    app = FastAPI(
        title="Agent Platform",
        version="0.1.0",
        description="MVP skeleton for the 灵枢 Agent Middleware Platform.",
    )
    app.state.runtime = rt
    app.include_router(api_router, prefix="/v1")

    # Admin / Dashboard routes. Imported lazily to avoid an import cycle.
    from agent_platform.api.admin import router as admin_router

    app.include_router(admin_router)

    @app.on_event("startup")
    async def _startup() -> None:
        # Report the effective configuration, then apply the agent archive.
        # Both are idempotent and safe on every boot.
        rt.log_configuration()
        await rt.seed_defaults()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    return app
