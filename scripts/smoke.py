"""Smoke test: drive the FastAPI app via ASGITransport (no real network).

Exercises the full HTTP surface end-to-end — healthz, session creation,
SSE chat, the tool round-trip, HITL suspend, approval, and resume — using
the same `Runtime.default()` wiring the server uses, with an in-memory
Redis and the deterministic MockChatModel. Needs no external services and
makes no outbound calls: the seeded agent is pinned to `model: mock` below.

Run: .venv/bin/python scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from agent_platform.api.app import Runtime, create_app
from agent_platform.config import Settings


async def read_sse(resp):
    """Yield (event_name, data_dict) from an httpx streaming response."""
    event_name = None
    async for line in resp.aiter_lines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event_name is not None:
            with contextlib.suppress(json.JSONDecodeError):
                yield event_name, json.loads(line.split(":", 1)[1])
            event_name = None


async def main() -> None:
    # The approval-gated write_file tool defaults to ./workspace, which would
    # drop notes/demo.txt into the repo on every run. Point it at a temp dir.
    write_root = Path(tempfile.mkdtemp(prefix="smoke-workspace-"))
    settings = Settings(
        use_fake_redis=True,
        checkpoint_ttl_seconds=60,
        llm_max_retries=1,
        llm_retry_base_delay=0.0,
        empty_response_max_retries=1,
        tool_write_file_root=str(write_root),
    )
    runtime = Runtime.default(settings)
    await runtime.seed_defaults()
    app = create_app(runtime)

    from agent_platform.agent_seed import load_agent_seeds

    seeds = load_agent_seeds(settings)
    assert seeds, f"no agents seeded from {settings.agent_seed_file}"
    agent_id = seeds[0].agent_id
    seeded_model = seeds[0].model

    # This script exercises the HTTP surface and the Loop, not the provider.
    # The seed says `deepseek-flash`, which would send it to the real API
    # whenever a key happens to be configured — so both the outcome and the
    # cost would depend on whose machine it runs on. Pin the seeded agent to
    # the deterministic mock; scripts/live_verify.py covers the real API.
    cfg = await runtime.config_store.get(agent_id)
    assert cfg is not None, agent_id
    cfg.model = "mock"
    await runtime.config_store.upsert(cfg)
    agent_model = "mock"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # healthz
        r = await ac.get("/healthz")
        assert r.status_code == 200, r.text
        print("healthz:", r.json())

        # session
        r = await ac.post(
            "/v1/sessions", json={"agent_id": agent_id, "user_id": "smoke"}
        )
        assert r.status_code == 200, r.text
        sid = r.json()["thread_id"]
        print("session:", sid)

        # chat: echo -> tool round-trip
        async with ac.stream(
            "POST",
            f"/v1/sessions/{sid}/chat",
            json={"content": "please echo hello"},
        ) as resp:
            assert resp.status_code == 200
            echo_events = [(n, d) async for n, d in read_sse(resp)]
        names = [n for n, _ in echo_events]
        print("echo events:", names)
        assert "finish" in names, names

        # The start event must tell us which LLM actually ran.
        start = next(d for n, d in echo_events if n == "start")
        print("  start:", {k: start.get(k) for k in ("agent_id", "model", "llm")})

        # chat: write intent -> HITL
        async with ac.stream(
            "POST",
            f"/v1/sessions/{sid}/chat",
            json={"content": "save a note to notes/smoke.txt"},
        ) as resp:
            assert resp.status_code == 200
            hitl_events = [(n, d) async for n, d in read_sse(resp)]
        names = [n for n, _ in hitl_events]
        print("write events:", names)
        assert "hitl_required" in names, names
        approval_id = next(
            d["approval_id"] for n, d in hitl_events if n == "hitl_required"
        )

        # approve
        r = await ac.post(
            f"/v1/sessions/{sid}/hitl/approve",
            json={"approval_id": approval_id},
        )
        assert r.status_code == 200, r.text
        print("approve:", r.json())

        # resume
        async with ac.stream(
            "POST",
            f"/v1/sessions/{sid}/chat",
            json={"content": "", "approval_id": approval_id},
        ) as resp:
            assert resp.status_code == 200
            resume_events = [(n, d) async for n, d in read_sse(resp)]
        names = [n for n, _ in resume_events]
        print("resume events:", names)
        assert "hitl_resolved" in names, names
        assert "finish" in names, names

        # admin surface
        r = await ac.get("/admin/api/configs")
        cfgs = {c["agent_id"]: c["model"] for c in r.json()}
        print("admin configs:", cfgs)
        assert cfgs.get(agent_id) == agent_model, cfgs

        r = await ac.get("/admin/api/tools")
        print("admin tools:", [t["name"] for t in r.json()])

        written = write_root / "notes" / "demo.txt"
        print("approved write:", written.read_text(encoding="utf-8"))

        r = await ac.get("/admin/api/secrets")
        ds = next(s for s in r.json() if s["provider"] == "deepseek")
        print("admin deepseek key:", ds["masked"] or "(unset)", "/ source:", ds["source"])

    print(
        f"OK — smoke test passed (agent={agent_id} seeded={seeded_model} "
        f"pinned={agent_model})"
    )


if __name__ == "__main__":
    asyncio.run(main())
