"""Verify what one human approval actually covers, against a running server.

An approval used to be a field in the resume request: /hitl/approve wrote
nothing, the Loop only checked whether an approval_id was present, and one
approval therefore waved through every sensitive call in the run. Skipping the
approve call entirely worked.

Now an approval is a recorded grant scoped to one target with a TTL. These
checks pin the boundaries in both directions — too broad and a prompt-injected
agent writes files nobody reviewed; too narrow and the platform nags on every
turn.

Usage (server must be running):
    .venv/bin/python scripts/check_approval_scope.py [base_url] [agent_id]
"""

from __future__ import annotations

import asyncio
import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
AGENT = sys.argv[2] if len(sys.argv) > 2 else "demo"

FAILURES: list[str] = []


def say(title: str) -> None:
    print("\n\033[1m== " + title + "\033[0m")


def ok(msg: str) -> None:
    print("   \033[32mOK\033[0m   " + msg)


def bad(msg: str) -> None:
    print("   \033[31mFAIL\033[0m " + msg)
    FAILURES.append(msg)


async def read_sse(resp) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    name = None
    async for line in resp.aiter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name is not None:
            try:
                out.append((name, json.loads(line.split(":", 1)[1])))
            except json.JSONDecodeError:
                out.append((name, {}))
            name = None
    return out


def names(events) -> list[str]:
    return [n for n, _ in events]


def approval_of(events) -> str | None:
    for n, d in events:
        if n == "hitl_required":
            return d.get("approval_id")
    return None


async def new_session(ac: httpx.AsyncClient, user: str) -> str:
    r = await ac.post("/v1/sessions", json={"agent_id": AGENT, "user_id": user})
    r.raise_for_status()
    return r.json()["thread_id"]


async def write_request(ac: httpx.AsyncClient, sid: str, path: str) -> list:
    async with ac.stream(
        "POST",
        "/v1/sessions/" + sid + "/chat",
        json={
            "content": "Use the write_file tool to save hello into " + path
        },
    ) as resp:
        return await read_sse(resp)


async def resume(ac: httpx.AsyncClient, sid: str, approval_id: str) -> list:
    async with ac.stream(
        "POST",
        "/v1/sessions/" + sid + "/chat",
        json={"content": "", "approval_id": approval_id},
    ) as resp:
        return await read_sse(resp)


async def main() -> int:
    async with httpx.AsyncClient(
        base_url=BASE, timeout=60, trust_env=False
    ) as ac:
        say("1. a write asks for approval, and the tool does not run first")
        sid = await new_session(ac, "scope-check")
        events = await write_request(ac, sid, "notes/scope-a.txt")
        first = approval_of(events)
        if first is None:
            bad("no hitl_required: " + str(names(events)))
            return 1
        ok("hitl_required emitted")
        if "tool_result" in names(events):
            bad("the tool ran before approval")
        else:
            ok("the tool did not run before approval")

        say("2. approving records a grant scoped to that file")
        r = await ac.post(
            "/v1/sessions/" + sid + "/hitl/approve",
            json={"approval_id": first},
        )
        if r.status_code != 200:
            bad("approve returned " + str(r.status_code) + ": " + r.text)
            return 1
        granted = r.json().get("granted", {})
        if str(granted.get("scope", "")).endswith("scope-a.txt"):
            ok("grant scope = " + str(granted.get("scope")).rsplit("/", 1)[-1])
        else:
            bad("grant scope looks wrong: " + str(granted))
        if granted.get("expires_in_seconds"):
            ok("grant expires in " + str(granted["expires_in_seconds"]) + "s")

        say("3. the resume executes")
        after = await resume(ac, sid, first)
        if "tool_result" in names(after) and "finish" in names(after):
            ok("tool ran and the turn finished")
        else:
            bad("resume did not complete: " + str(names(after)))

        say("4. the SAME file inside the TTL does not ask again")
        sid2 = await new_session(ac, "scope-check")
        again = await write_request(ac, sid2, "notes/scope-a.txt")
        if "hitl_required" in names(again):
            bad("the same file asked for approval again within the TTL")
        else:
            ok("no approval prompt")
        if "tool_result" in names(again):
            ok("and it executed")
        else:
            bad("same-file write did not execute: " + str(names(again)))

        say("5. a DIFFERENT file asks again")
        sid3 = await new_session(ac, "scope-check")
        other = await write_request(ac, sid3, "notes/scope-b.txt")
        if "hitl_required" in names(other):
            ok("a different file is not covered by the first approval")
        else:
            bad("a different file was silently covered: " + str(names(other)))

        say("6. resuming without approving executes nothing")
        pending = approval_of(other)
        if pending is None:
            bad("could not reach HITL on the second file")
        else:
            sneaky = await resume(ac, sid3, pending)
            if "tool_result" in names(sneaky):
                bad("the tool ran with no recorded approval")
            else:
                ok("nothing executed")
            reason = next(
                (d.get("reason") for n, d in sneaky if n == "hitl_required"), None
            )
            if reason == "grant_missing_or_expired":
                ok("parked again with reason=" + reason)
            else:
                bad("expected a grant_missing_or_expired re-park, got " + str(names(sneaky)))

        say("7. a forged approval_id is rejected")
        r = await ac.post(
            "/v1/sessions/" + sid3 + "/hitl/approve",
            json={"approval_id": "appr_" + sid3 + "_MADE_UP"},
        )
        if r.status_code == 400:
            ok("400 as expected")
        else:
            bad("forged id returned " + str(r.status_code))

        say("8. another user does not inherit the grant")
        sid4 = await new_session(ac, "someone-else")
        foreign = await write_request(ac, sid4, "notes/scope-a.txt")
        if "hitl_required" in names(foreign):
            ok("a different principal is asked separately")
        else:
            bad("another user inherited the grant: " + str(names(foreign)))

        say("9. revoking the grant brings the prompt back")
        grants = (await ac.get("/admin/api/approvals")).json()
        revoked = 0
        for g in grants:
            r = await ac.post(
                "/admin/api/approvals/revoke",
                json={
                    "agent_id": g["agent_id"],
                    "user_id": g["user_id"],
                    "tool_name": g["tool_name"],
                    "scope": g["scope"],
                },
            )
            revoked += 1 if r.json().get("revoked") else 0
        if revoked:
            ok("revoked " + str(revoked) + " grant(s)")
            sid5 = await new_session(ac, "scope-check")
            back = await write_request(ac, sid5, "notes/scope-a.txt")
            if "hitl_required" in names(back):
                ok("the prompt is back after revoking")
            else:
                bad("a revoked grant still authorised a write")
        else:
            bad("no grants to revoke")

    print()
    if FAILURES:
        print("\033[31m" + str(len(FAILURES)) + " check(s) failed.\033[0m")
        return 1
    print("\033[32mAll approval-scope checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
