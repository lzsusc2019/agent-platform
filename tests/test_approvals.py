r"""Approval grants: scope, TTL, and the fact that they are actually recorded.

Before this existed, "approved" meant "the resume request carried an
approval_id". /hitl/approve wrote nothing, the Loop only checked
\`approval_id is None\`, and one approval therefore waved through every
sensitive call in the run. Resuming without approving at all worked.

The semantics these tests pin down:

* An approval covers ONE target. Approving notes/a.txt must not authorise
  notes/b.txt.
* It expires. After the TTL the same call asks again.
* It is issued to a principal. Another user's turn does not inherit it.
* It cannot be conjured by asserting an id in a request.
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_platform.api.app import Runtime, create_app
from agent_platform.approvals import ApprovalGrant, ApprovalStore
from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import Settings
from agent_platform.config_store import AgentConfigStore
from agent_platform.core.agent_loop import AgentLoop
from agent_platform.core.llm import ChatModel, LLMResponse
from agent_platform.core.messages import ToolCall
from agent_platform.secrets_store import SecretStore
from agent_platform.store.agent_manager import AgentManager
from agent_platform.tools import build_default_registry


class _WriteOnce(ChatModel):
    """Asks to write one path, then stops once it sees a tool result."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.emitted = 0

    async def ainvoke(self, messages, tools):  # type: ignore[no-untyped-def]
        if messages and messages[-1].get("role") == "tool":
            return LLMResponse(content="done", tool_calls=[])
        self.emitted += 1
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.emitted}",
                    name="write_file",
                    arguments={"path": self.path, "content": "x"},
                )
            ],
        )


def _loop(
    ckpt: CheckpointStore,
    settings: Settings,
    approvals: ApprovalStore | None,
    path: str = "notes/a.txt",
) -> AgentLoop:
    return AgentLoop(
        llm=_WriteOnce(path),
        tools=build_default_registry(settings),
        checkpoint=ckpt,
        system_prompt="test",
        settings=settings,
        agent_id="demo",
        approvals=approvals,
    )


async def _run(loop: AgentLoop, **kw) -> list:
    return [ev async for ev in loop.run(**kw)]


def _kinds(events) -> list[str]:
    # .value, not str(): EventType is a (str, Enum), and str() on one of those
    # yields "EventType.HITL_REQUIRED" rather than "hitl_required".
    return [ev.type.value for ev in events]


@pytest_asyncio.fixture
async def redis():
    r = fakeredis.aioredis.FakeRedis()
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def ckpt(redis):
    return CheckpointStore(redis, ttl_seconds=60)


@pytest_asyncio.fixture
async def approvals(redis):
    return ApprovalStore(redis, ttl_seconds=3600)


async def _grant_for_file(
    approvals: ApprovalStore, settings: Settings, path: str, *, user_id: str = "u1"
) -> str:
    """Record a grant the way /hitl/approve would, and return its scope."""
    tools = build_default_registry(settings)
    scope = tools.get("write_file").approval_scope({"path": path})
    await approvals.grant(
        ApprovalGrant(
            agent_id="demo",
            user_id=user_id,
            tool_name="write_file",
            scope=scope,
            approval_id="appr_test",
        )
    )
    return scope


# --------------------------------------------------------------------------- #
# scope: one approval, one file
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_first_write_asks_for_approval(ckpt, settings, approvals) -> None:
    loop = _loop(ckpt, settings, approvals)
    events = await _run(loop, thread_id="t", user_id="u1", user_message="write it")
    assert "hitl_required" in _kinds(events)


@pytest.mark.asyncio
async def test_same_file_does_not_ask_twice(ckpt, settings, approvals) -> None:
    await _grant_for_file(approvals, settings, "notes/a.txt")
    loop = _loop(ckpt, settings, approvals, path="notes/a.txt")
    events = await _run(loop, thread_id="t1", user_id="u1", user_message="write it")
    kinds = _kinds(events)
    assert "hitl_required" not in kinds, kinds
    assert "tool_result" in kinds, kinds


@pytest.mark.asyncio
async def test_a_different_file_asks_again(ckpt, settings, approvals) -> None:
    """The whole point: approving one file must not authorise another."""
    await _grant_for_file(approvals, settings, "notes/a.txt")
    loop = _loop(ckpt, settings, approvals, path="notes/b.txt")
    events = await _run(loop, thread_id="t2", user_id="u1", user_message="write it")
    assert "hitl_required" in _kinds(events)


@pytest.mark.asyncio
async def test_path_spellings_of_one_file_share_a_grant(ckpt, settings, approvals) -> None:
    """`./notes/../notes/a.txt` is the same file and must not re-prompt."""
    await _grant_for_file(approvals, settings, "notes/a.txt")
    loop = _loop(ckpt, settings, approvals, path="./notes/../notes/a.txt")
    events = await _run(loop, thread_id="t3", user_id="u1", user_message="write it")
    assert "hitl_required" not in _kinds(events)


@pytest.mark.asyncio
async def test_grant_scoped_to_a_user_does_not_cover_another(
    ckpt, settings, approvals
) -> None:
    """An approval is issued to a principal, not to the file in the abstract."""
    await _grant_for_file(approvals, settings, "notes/a.txt", user_id="alice")
    loop = _loop(ckpt, settings, approvals, path="notes/a.txt")
    events = await _run(loop, thread_id="t4", user_id="bob", user_message="write it")
    assert "hitl_required" in _kinds(events)


@pytest.mark.asyncio
async def test_grant_scoped_to_another_tool_does_not_apply(
    ckpt, settings, approvals
) -> None:
    await approvals.grant(
        ApprovalGrant(
            agent_id="demo",
            user_id="u1",
            tool_name="http_get",
            scope="",
            approval_id="appr_other",
        )
    )
    loop = _loop(ckpt, settings, approvals, path="notes/a.txt")
    events = await _run(loop, thread_id="t5", user_id="u1", user_message="write it")
    assert "hitl_required" in _kinds(events)


# --------------------------------------------------------------------------- #
# TTL
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_expired_grant_asks_again(ckpt, settings, redis) -> None:
    """An approval is time-boxed; the store's TTL is the authority."""
    short = ApprovalStore(redis, ttl_seconds=1)
    await _grant_for_file(short, settings, "notes/a.txt")

    fresh = _loop(ckpt, settings, short, path="notes/a.txt")
    assert "hitl_required" not in _kinds(
        await _run(fresh, thread_id="ttl1", user_id="u1", user_message="write")
    )

    await asyncio.sleep(1.2)

    expired = _loop(ckpt, settings, short, path="notes/a.txt")
    events = await _run(expired, thread_id="ttl2", user_id="u1", user_message="write")
    assert "hitl_required" in _kinds(events), "a grant outlived its TTL"


def test_store_rejects_a_nonpositive_ttl(redis) -> None:
    with pytest.raises(ValueError):
        ApprovalStore(redis, ttl_seconds=0)


@pytest.mark.asyncio
async def test_list_grants_reports_what_is_currently_permitted(
    settings, approvals
) -> None:
    await _grant_for_file(approvals, settings, "notes/a.txt")
    grants = await approvals.list_grants()
    assert len(grants) == 1
    assert grants[0].tool_name == "write_file"
    assert grants[0].scope.endswith("notes/a.txt")
    assert grants[0].ttl_seconds is not None and grants[0].ttl_seconds > 0


@pytest.mark.asyncio
async def test_revoke_removes_the_grant(settings, approvals) -> None:
    scope = await _grant_for_file(approvals, settings, "notes/a.txt")
    assert await approvals.is_granted(
        agent_id="demo", user_id="u1", tool_name="write_file", scope=scope
    )
    await approvals.revoke(
        agent_id="demo", user_id="u1", tool_name="write_file", scope=scope
    )
    assert not await approvals.is_granted(
        agent_id="demo", user_id="u1", tool_name="write_file", scope=scope
    )


# --------------------------------------------------------------------------- #
# no store wired: the pre-grant behaviour is preserved
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_without_a_store_an_approval_id_still_resumes(ckpt, settings) -> None:
    loop = _loop(ckpt, settings, None)
    events = await _run(loop, thread_id="n1", user_id="u1", user_message="write")
    approval_id = next(
        ev.data["approval_id"]
        for ev in events
        if ev.type.value == "hitl_required"
    )
    resumed = await _run(
        loop, thread_id="n1", user_id="u1", user_message="", approval_id=approval_id
    )
    kinds = _kinds(resumed)
    assert "hitl_resolved" in kinds, kinds
    assert "tool_result" in kinds, kinds


# --------------------------------------------------------------------------- #
# through the API
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture
async def rt(redis, settings):
    ck = CheckpointStore(redis, ttl_seconds=60)
    approvals = ApprovalStore(redis, ttl_seconds=settings.approval_grant_ttl_seconds)
    agents = AgentManager(
        llm=_WriteOnce("notes/a.txt"),
        tools=build_default_registry(settings),
        checkpoint=ck,
        settings=settings,
        approvals=approvals,
    )
    return Runtime(
        settings=settings,
        checkpoint=ck,
        config_store=AgentConfigStore(redis),
        secret_store=SecretStore(redis),
        tools=build_default_registry(settings),
        llm=_WriteOnce("notes/a.txt"),
        agents=agents,
        approval_store=approvals,
    )


@pytest_asyncio.fixture
async def client(rt):
    app = create_app(rt)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as ac:
        yield ac


async def _sse(resp) -> list[tuple[str, dict]]:
    import json

    out: list[tuple[str, dict]] = []
    name = None
    async for line in resp.aiter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            try:
                out.append((name, json.loads(line.split(":", 1)[1])))
            except json.JSONDecodeError:
                out.append((name, {}))
            name = None
    return out


async def _start_and_park(client) -> tuple[str, str]:
    r = await client.post("/v1/sessions", json={"agent_id": "demo", "user_id": "u1"})
    sid = r.json()["thread_id"]
    async with client.stream(
        "POST", f"/v1/sessions/{sid}/chat", json={"content": "write it"}
    ) as resp:
        events = await _sse(resp)
    approval_id = next(
        d["approval_id"] for n, d in events if n == "hitl_required"
    )
    return sid, approval_id


@pytest.mark.asyncio
async def test_approve_records_a_grant_for_that_file(client, rt) -> None:
    sid, approval_id = await _start_and_park(client)
    r = await client.post(
        f"/v1/sessions/{sid}/hitl/approve", json={"approval_id": approval_id}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["granted"]["tool"] == "write_file"
    assert body["granted"]["scope"].endswith("notes/a.txt")
    assert body["granted"]["expires_in_seconds"] == 3600

    grants = await rt.approval_store.list_grants()
    assert [g.scope.rsplit("/", 1)[-1] for g in grants] == ["a.txt"]


@pytest.mark.asyncio
async def test_resume_after_approving_succeeds(client) -> None:
    sid, approval_id = await _start_and_park(client)
    await client.post(
        f"/v1/sessions/{sid}/hitl/approve", json={"approval_id": approval_id}
    )
    async with client.stream(
        "POST",
        f"/v1/sessions/{sid}/chat",
        json={"content": "", "approval_id": approval_id},
    ) as resp:
        names = [n for n, _ in await _sse(resp)]
    assert "hitl_resolved" in names, names
    assert "tool_result" in names, names


@pytest.mark.asyncio
async def test_resume_without_approving_parks_again(client) -> None:
    """The hole this closes: skipping /hitl/approve used to just work."""
    sid, approval_id = await _start_and_park(client)
    async with client.stream(
        "POST",
        f"/v1/sessions/{sid}/chat",
        json={"content": "", "approval_id": approval_id},
    ) as resp:
        events = await _sse(resp)
    names = [n for n, _ in events]
    assert "tool_result" not in names, "the tool ran without any recorded approval"
    assert "hitl_required" in names, names
    reason = next(d for n, d in events if n == "hitl_required").get("reason")
    assert reason == "grant_missing_or_expired", reason


@pytest.mark.asyncio
async def test_approve_rejects_a_forged_approval_id(client) -> None:
    sid, _ = await _start_and_park(client)
    r = await client.post(
        f"/v1/sessions/{sid}/hitl/approve",
        json={"approval_id": f"appr_{sid}_TOTALLY_MADE_UP"},
    )
    assert r.status_code == 400, r.text


@pytest.mark.asyncio
async def test_reject_grants_nothing(client, rt) -> None:
    sid, approval_id = await _start_and_park(client)
    r = await client.post(
        f"/v1/sessions/{sid}/hitl/reject", json={"approval_id": approval_id}
    )
    assert r.status_code == 200
    assert await rt.approval_store.list_grants() == []
