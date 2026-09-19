r"""The tool-call ordering invariant, and the paths that used to break it.

OpenAI-compatible endpoints enforce one rule absolutely:

    an assistant message carrying tool_calls must be followed immediately by
    one tool message per tool_call_id.

Violating it is a hard 400 that names a message index and nothing else, so it
reads as a wire-format bug when the real cause is a history that some other
path left inconsistent. This has now bitten three times:

  1. a resume carrying \`content: ""\` wedged an empty user turn in between
  2. context compression could cut between a call and its reply
  3. **rejecting an approval left the call permanently unanswered**, so the
     *next* turn on that thread replayed the broken history and 400'd

(3) is the one that reached production. The tests below pin all three, plus
the Loop-level seal that makes the invariant hold no matter which path broke it.
"""

from __future__ import annotations

import json

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent_platform.api.app import Runtime, create_app
from agent_platform.api.routes import router  # noqa: F401  (import-time check)
from agent_platform.approvals import ApprovalStore
from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config_store import AgentConfigStore
from agent_platform.core.agent_loop import AgentLoop
from agent_platform.core.checkpoint import CheckpointStatus
from agent_platform.core.llm import ChatModel, LLMResponse, MockChatModel
from agent_platform.core.messages import (
    Message,
    MessageRole,
    ToolCall,
    unanswered_tool_calls,
)
from agent_platform.secrets_store import SecretStore
from agent_platform.store.agent_manager import AgentManager
from agent_platform.tools import build_default_registry


def assert_ordering(messages: list[dict]) -> None:
    """The rule the provider enforces, checked on the outbound payload."""
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        want = [tc["id"] for tc in m["tool_calls"]]
        got, j = [], i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            got.append(messages[j].get("tool_call_id"))
            j += 1
        assert got == want, (
            f"messages[{i}] tool_calls={want} answered by {got}, "
            f"next={(messages[j].get('role') if j < len(messages) else 'EOF')}"
        )


class _EnforcingModel(ChatModel):
    """A stub that rejects a malformed history the way a real provider does.

    Without this, a stub accepts anything and the bug only shows up against
    the live API — which is exactly how it got out.
    """

    def __init__(self, *, write_path: str | None = "notes/r.txt") -> None:
        self.write_path = write_path
        self.seen: list[list[dict]] = []

    async def ainvoke(self, messages, tools):  # type: ignore[no-untyped-def]
        self.seen.append(messages)
        assert_ordering(messages)
        if self.write_path is None:
            return LLMResponse(content="ok", tool_calls=[])
        if messages and messages[-1].get("role") == "tool":
            return LLMResponse(content="done", tool_calls=[])
        return LLMResponse(
            content="writing",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={"path": self.write_path, "content": "x"},
                )
            ],
        )


class _RecordingModel(MockChatModel):
    """Accepts anything, but remembers what it was asked to send."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list[dict]] = []

    async def ainvoke(self, messages, tools):  # type: ignore[no-untyped-def]
        self.seen.append(messages)
        return await super().ainvoke(messages, tools)


# --------------------------------------------------------------------------- #
# the helper itself
# --------------------------------------------------------------------------- #


def test_helper_reports_calls_with_no_reply() -> None:
    tc = [ToolCall(id="c1", name="a"), ToolCall(id="c2", name="b")]
    messages = [Message(role=MessageRole.ASSISTANT, tool_calls=tc)]
    assert [t.id for t in unanswered_tool_calls(messages)] == ["c1", "c2"]

    messages.append(Message(role=MessageRole.TOOL, content="r", tool_call_id="c1"))
    assert [t.id for t in unanswered_tool_calls(messages)] == ["c2"]

    messages.append(Message(role=MessageRole.TOOL, content="r", tool_call_id="c2"))
    assert unanswered_tool_calls(messages) == []


def test_helper_ignores_a_settled_history() -> None:
    messages = [
        Message(role=MessageRole.USER, content="hi"),
        Message(role=MessageRole.ASSISTANT, content="hello"),
    ]
    assert unanswered_tool_calls(messages) == []


# --------------------------------------------------------------------------- #
# the Loop-level seal: the invariant holds no matter who broke it
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_loop_seals_a_hand_corrupted_history(ckpt_store, settings) -> None:
    """A dangling call seeded directly into the Checkpoint must not 400."""
    snap = ckpt_store.new_snapshot(
        thread_id="corrupt",
        messages=[
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="write it"),
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[ToolCall(id="c9", name="write_file", arguments={})],
            ),
        ],
        status=CheckpointStatus.FINISHED,
        turn=1,
        last_config={},
    )
    await ckpt_store.save(snap)

    model = _RecordingModel()
    loop = AgentLoop(
        llm=model,
        tools=build_default_registry(settings),
        checkpoint=ckpt_store,
        system_prompt="sys",
        settings=settings,
        agent_id="demo",
    )
    events = [
        ev
        async for ev in loop.run(
            thread_id="corrupt", user_id="u1", user_message="hello again"
        )
    ]

    # The model was actually called, and with a legal history.
    assert model.seen, "the model was never invoked"
    assert_ordering(model.seen[0])
    assert "error" not in [ev.type.value for ev in events], [
        ev.data for ev in events
    ]

    # And the repair is persisted, so it does not have to happen again.
    repaired = await ckpt_store.load("corrupt")
    assert repaired is not None
    sealed = [m for m in repaired.messages if m.meta.get("sealed")]
    assert [m.tool_call_id for m in sealed] == ["c9"]
    assert "never ran" in sealed[0].content


@pytest.mark.asyncio
async def test_loop_does_not_touch_a_healthy_history(ckpt_store, settings) -> None:
    loop = AgentLoop(
        llm=_RecordingModel(),
        tools=build_default_registry(settings),
        checkpoint=ckpt_store,
        system_prompt="sys",
        settings=settings,
        agent_id="demo",
    )
    events = [
        ev
        async for ev in loop.run(
            thread_id="healthy", user_id="u1", user_message="please echo hi"
        )
    ]
    assert "finish" in [ev.type.value for ev in events]
    snap = await ckpt_store.load("healthy")
    assert snap is not None
    assert not [m for m in snap.messages if m.meta.get("sealed")]


# --------------------------------------------------------------------------- #
# the reject path, through the API — the bug that reached production
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture
async def redis():
    r = fakeredis.aioredis.FakeRedis()
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def api(redis, settings):
    ck = CheckpointStore(redis, ttl_seconds=300)
    approvals = ApprovalStore(redis, ttl_seconds=3600)
    model = _EnforcingModel()
    tools = build_default_registry(settings)
    agents = AgentManager(
        llm=model, tools=tools, checkpoint=ck, settings=settings, approvals=approvals
    )
    rt = Runtime(
        settings=settings,
        checkpoint=ck,
        config_store=AgentConfigStore(redis),
        secret_store=SecretStore(redis),
        tools=tools,
        llm=model,
        agents=agents,
        approval_store=approvals,
    )
    async with AsyncClient(
        transport=ASGITransport(app=create_app(rt)), base_url="http://t"
    ) as ac:
        yield ac, rt


async def _sse(resp) -> list[tuple[str, dict]]:
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


async def _park(ac) -> tuple[str, str]:
    r = await ac.post("/v1/sessions", json={"agent_id": "demo", "user_id": "u1"})
    sid = r.json()["thread_id"]
    async with ac.stream(
        "POST", f"/v1/sessions/{sid}/chat", json={"content": "写点东西"}
    ) as resp:
        events = await _sse(resp)
    return sid, next(d["approval_id"] for n, d in events if n == "hitl_required")


@pytest.mark.asyncio
async def test_reject_answers_the_pending_tool_call(api) -> None:
    """The rejection itself was always fine; the *history* it left was not."""
    ac, rt = api
    sid, approval_id = await _park(ac)
    r = await ac.post(
        f"/v1/sessions/{sid}/hitl/reject", json={"approval_id": approval_id}
    )
    assert r.status_code == 200

    snap = await rt.checkpoint.load(sid)
    assert snap is not None
    assert snap.status == CheckpointStatus.FINISHED
    assert unanswered_tool_calls(snap.messages) == [], (
        "a rejected thread was left with an unanswered tool call; the next turn "
        "on it will 400"
    )
    denied = [m for m in snap.messages if m.meta.get("denied")]
    assert len(denied) == 1
    assert "denied approval" in denied[0].content
    # The spent approval must not linger on a finished thread.
    assert snap.approval_id is None


@pytest.mark.asyncio
async def test_a_new_turn_after_reject_works(api) -> None:
    """The user-visible symptom: continuing the conversation after a reject."""
    ac, _ = api
    sid, approval_id = await _park(ac)
    await ac.post(f"/v1/sessions/{sid}/hitl/reject", json={"approval_id": approval_id})

    async with ac.stream(
        "POST", f"/v1/sessions/{sid}/chat", json={"content": "算了 你好"}
    ) as resp:
        events = await _sse(resp)
    names = [n for n, _ in events]
    assert "error" not in names, [d for n, d in events if n == "error"]
    # The stub asks to write again, so a fresh park is the expected outcome —
    # what matters is that the request was accepted at all.
    assert names[0] == "start", names


@pytest.mark.asyncio
async def test_admin_chat_refuses_a_new_message_on_a_parked_thread(api) -> None:
    """The other door into the same malformed history."""
    ac, _ = api
    sid, _ = await _park(ac)
    r = await ac.post(
        "/admin/api/chat", json={"agent_id": "demo", "thread_id": sid, "content": "你好"}
    )
    assert r.status_code == 409, r.text
    assert "awaiting approval" in r.text


@pytest.mark.asyncio
async def test_admin_chat_still_resumes_with_an_approval_id(api) -> None:
    """The guard must not block the legitimate resume."""
    ac, _ = api
    sid, approval_id = await _park(ac)
    await ac.post(f"/v1/sessions/{sid}/hitl/approve", json={"approval_id": approval_id})
    async with ac.stream(
        "POST",
        "/admin/api/chat",
        json={
            "agent_id": "demo",
            "thread_id": sid,
            "content": "",
            "approval_id": approval_id,
        },
    ) as resp:
        names = [n for n, _ in await _sse(resp)]
    assert "hitl_resolved" in names, names
    assert "error" not in names, names
