"""Live verification against the real DeepSeek API.

Two things this proves that unit tests cannot:

1. The assistant/tool messages we put on the wire are accepted by the real
   endpoint. (A previous revision sent `tool_calls` in our own internal
   shape and got back a 422 \"missing field 'type'\" on the *second* turn,
   once a tool result had to be replayed.)
2. A sensitive tool survives the whole HITL arc against a real model:
   model asks for the tool -> we suspend -> approve -> resume -> the file
   actually lands on disk.

Costs a real API call. Run:
    .venv/bin/python scripts/live_verify.py
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
    event_name = None
    async for line in resp.aiter_lines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event_name is not None:
            with contextlib.suppress(json.JSONDecodeError):
                yield event_name, json.loads(line.split(":", 1)[1])
            event_name = None


def show(events, label):
    print(f"  {label}:")
    for name, data in events:
        keys = ("tool", "name", "error", "message", "approval_id", "content")
        brief = {k: data[k] for k in keys if k in data}
        if "content" in brief and isinstance(brief["content"], str):
            brief["content"] = brief["content"][:90]
        print(f"    {name:<14} {brief}")


async def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="live-verify-"))
    settings = Settings(
        use_fake_redis=True,
        checkpoint_ttl_seconds=120,
        llm_max_retries=2,
        llm_retry_base_delay=0.5,
        empty_response_max_retries=1,
        tool_write_file_root=str(workdir),
    )

    print(f"write root : {workdir}")
    print(f"key source : {settings.deepseek_key_source()}")
    print(f"key        : {settings.redacted().get('deepseek_api_key')}")
    if settings.deepseek_key_source() == "none":
        print("\nNo DeepSeek key configured — nothing to verify against.")
        return 2

    runtime = Runtime.default(settings)
    await runtime.seed_defaults()
    app = create_app(runtime)

    failures: list[str] = []
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=90) as ac:
        r = await ac.post("/v1/sessions", json={"agent_id": "demo", "user_id": "live"})
        r.raise_for_status()
        sid = r.json()["thread_id"]
        print(f"session    : {sid}\n")

        # --- turn 1: plain tool round-trip; this is the turn that 422'd ------
        print("[1] echo round-trip (real model decides to call a tool)")
        async with ac.stream(
            "POST", f"/v1/sessions/{sid}/chat",
            json={"content": "Call the echo tool with the word 'ping'. Then tell me what it returned."},
        ) as resp:
            assert resp.status_code == 200, resp.status_code
            turn1 = [(n, d) async for n, d in read_sse(resp)]
        show(turn1, "events")
        names = [n for n, _ in turn1]
        if any(n == "error" for n in names):
            failures.append("turn 1 emitted an error event")
        if "tool_call" not in names:
            failures.append("model did not call a tool on turn 1")
        if "finish" not in names:
            failures.append("turn 1 never finished")

        # --- turn 2: the reported bug -------------------------------------
        # The complaint was "获取东莞天气工具调用失败了". The tool call itself
        # succeeded and the weather came back; what blew up was the *follow-up*
        # request replaying that tool result (422, missing field 'type').
        # Reproducing that exact shape is the point of this step.
        print("\n[2] http_get weather (the originally reported failure)")
        async with ac.stream(
            "POST", f"/v1/sessions/{sid}/chat",
            json={"content": (
                "Use the http_get tool to fetch https://wttr.in/Dongguan?format=3 "
                "and tell me what the weather in Dongguan is right now."
            )},
        ) as resp:
            assert resp.status_code == 200, resp.status_code
            weather = [(n, d) async for n, d in read_sse(resp)]
        show(weather, "events")
        names = [n for n, _ in weather]
        if "error" in names:
            failures.append("weather turn emitted an error event")
        if "tool_call" not in names:
            failures.append("model did not reach for http_get on the weather turn")
        if "finish" not in names:
            failures.append("weather turn never finished")
        answer = next((d.get("content") or "" for n, d in weather if n == "assistant"), "")
        if answer:
            print(f"  answer: {answer[:140]}")

        # --- turn 3: the sensitive tool + approval -------------------------
        print("\n[3] write_file round-trip (sensitive -> HITL)")
        async with ac.stream(
            "POST", f"/v1/sessions/{sid}/chat",
            json={"content": (
                "Use the write_file tool to save the text 'live verification ok' "
                "into notes/live.txt, then confirm you did it."
            )},
        ) as resp:
            assert resp.status_code == 200, resp.status_code
            turn2 = [(n, d) async for n, d in read_sse(resp)]
        show(turn2, "events")
        names = [n for n, _ in turn2]
        if "hitl_required" not in names:
            failures.append("write_file did not trigger HITL")
        approval_id = next((d.get("approval_id") for n, d in turn2 if n == "hitl_required"), None)

        target = workdir / "notes" / "live.txt"
        if approval_id:
            r = await ac.post(f"/v1/sessions/{sid}/hitl/approve", json={"approval_id": approval_id})
            print(f"\n  approve -> {r.status_code} {r.json()}")

            print("\n[4] resume after approval")
            async with ac.stream(
                "POST", f"/v1/sessions/{sid}/chat",
                json={"content": "", "approval_id": approval_id},
            ) as resp:
                assert resp.status_code == 200, resp.status_code
                turn3 = [(n, d) async for n, d in read_sse(resp)]
            show(turn3, "events")
            names = [n for n, _ in turn3]
            if "hitl_resolved" not in names:
                failures.append("resume did not emit hitl_resolved")
            if "finish" not in names:
                failures.append("resume never finished")
            if not target.exists():
                failures.append(f"file was not written: {target}")
            else:
                print(f"\n  wrote {target} -> {target.read_text(encoding='utf-8')!r}")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — real-API tool round-trip and HITL approval both clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
