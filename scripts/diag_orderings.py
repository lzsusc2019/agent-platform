"""Diagnostic: which operation orderings leave the agent stuck on MockChatModel?

Simulates the Dashboard workflow (set key / set model / send chat) in
different orders and prints the resulting ChatModel class.
"""
from __future__ import annotations

import asyncio
import logging

logging.disable(logging.WARNING)

import fakeredis.aioredis  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from agent_platform.api.app import Runtime, create_app  # noqa: E402
from agent_platform.checkpoint.store import CheckpointStore  # noqa: E402
from agent_platform.config import Settings  # noqa: E402
from agent_platform.config_store import AgentConfigStore  # noqa: E402
from agent_platform.core.llm import MockChatModel  # noqa: E402
from agent_platform.core.providers import create_chat_model  # noqa: E402
from agent_platform.secrets_store import SecretStore  # noqa: E402
from agent_platform.store.agent_manager import AgentManager  # noqa: E402
from agent_platform.tools import build_default_registry  # noqa: E402


def build():
    # Point the provider at a closed port: the scenarios below care about
    # *which* ChatModel class gets resolved, never about what it answers. Left
    # alone this would fire five real requests at the configured endpoint with
    # the throwaway key below, which is noise at best.
    s = Settings(deepseek_base_url="http://127.0.0.1:9")
    redis = fakeredis.aioredis.FakeRedis()
    cfg = AgentConfigStore(redis)
    sec = SecretStore(redis)
    ck = CheckpointStore(redis, ttl_seconds=60)
    tools = build_default_registry(s)
    llm = MockChatModel()
    agents = AgentManager(
        llm=llm, tools=tools, checkpoint=ck, settings=s,
        config_store=cfg, secret_store=sec,
        llm_factory=lambda m, k: create_chat_model(m, s, api_key_override=k),
    )
    rt = Runtime(
        settings=s, checkpoint=ck, config_store=cfg, secret_store=sec,
        tools=tools, llm=llm, agents=agents,
    )
    return rt, agents


LAST_EVENTS: list[tuple[str, dict]] = []


async def do_chat(ac):
    """Send a chat turn and record the events. Errors are captured rather
    than raised — the point of the diagnostic is to see what the *client*
    would receive."""
    LAST_EVENTS.clear()
    async with ac.stream(
        "POST", "/admin/api/chat", json={"agent_id": "demo", "content": "hi"}
    ) as r:
        import json as _json

        ev = None
        async for line in r.aiter_lines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and ev:
                try:
                    LAST_EVENTS.append((ev, _json.loads(line.split(":", 1)[1])))
                except Exception:
                    LAST_EVENTS.append((ev, {}))
                ev = None


async def do_set_model(ac):
    await ac.put("/admin/api/configs/demo", json={"model": "deepseek-flash"})


async def do_set_key(ac):
    await ac.put("/admin/api/secrets/deepseek/api_key", json={"value": "sk-test"})


async def scenario(name: str, steps) -> None:
    rt, agents = build()
    await rt.seed_defaults()
    app = create_app(rt)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        for step in steps:
            await step(ac)
        loop = await agents.get_or_create("demo")
        cls = type(loop.llm).__name__
        mark = "OK  " if cls == "DeepSeekChatModel" else "MOCK"
        ev_names = [n for n, _ in LAST_EVENTS] or ["(no chat)"]
        err = next((d for n, d in LAST_EVENTS if n == "error"), None)
        err_str = ""
        if err:
            err_str = f"  err={err.get('error_type')}: {str(err.get('message'))[:40]}"
        print(f"{mark} {name:35s} llm={cls:18s} events={ev_names}{err_str}")


async def main() -> None:
    await scenario("A  key -> model -> chat", [do_set_key, do_set_model, do_chat])
    await scenario("B  model -> key -> chat", [do_set_model, do_set_key, do_chat])
    await scenario("C  chat -> key -> model -> chat", [do_chat, do_set_key, do_set_model, do_chat])
    await scenario("D  chat -> model -> key -> chat", [do_chat, do_set_model, do_set_key, do_chat])
    await scenario("E  model -> chat(no key) -> key -> chat", [do_set_model, do_chat, do_set_key, do_chat])


if __name__ == "__main__":
    asyncio.run(main())
