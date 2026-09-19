"""Regression tests for the HITL-resume wire format.

A real DeepSeek run surfaced a second ordering bug after the `missing field
'type'` one was fixed:

    400: An assistant message with 'tool_calls' must be followed by tool
    messages responding to each 'tool_call_id'.

The dashboard and `scripts/smoke.py` both resume a suspended thread with
`{"content": "", "approval_id": ...}`. The empty string is not `None`, so the
loop treated it as real input and appended an empty USER turn directly after
the assistant's still-unanswered `tool_calls`. Every OpenAI-compatible endpoint
rejects that ordering.

The invariant these tests defend is narrow and worth stating plainly: *between
an assistant message carrying tool_calls and the tool messages answering it,
nothing else may appear.* Both the resume path and context compression can
violate it, so both are covered.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.config import Settings
from agent_platform.core.agent_loop import AgentLoop, HITLInterrupt
from agent_platform.core.messages import MessageRole
from agent_platform.core.providers import DeepSeekChatModel
from agent_platform.tools import build_default_registry

# --------------------------------------------------------------------------- #
# a validator shared by every test here
# --------------------------------------------------------------------------- #


def assert_valid_openai_order(messages: list[dict[str, Any]]) -> None:
    """Assert each assistant tool_calls message is answered immediately.

    This is the server-side rule, checked locally: we cannot run the real API
    in a unit test, but we can prove the payload would not be rejected.
    """
    for i, m in enumerate(messages):
        assert "role" in m, f"messages[{i}] has no role"
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        wanted = [tc["id"] for tc in m["tool_calls"]]
        answerers: list[str] = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            answerers.append(messages[j].get("tool_call_id"))
            j += 1
        assert answerers == wanted, (
            f"messages[{i}] has tool_calls {wanted} but is answered by "
            f"{answerers!r}; next message is {messages[j].get('role') if j < len(messages) else 'EOF'!r}"
        )


# --------------------------------------------------------------------------- #
# an httpx stand-in that captures outbound payloads and scripts the replies
# --------------------------------------------------------------------------- #


class _ScriptedClient:
    """Records every request body; replies from a per-call script."""

    # Class-level so the fixture and the model share one buffer; ClassVar keeps
    # the linter from reading these as mutable instance defaults.
    script: ClassVar[list[dict]] = []
    sent: ClassVar[list[dict]] = []

    def __init__(self, *args, **kwargs):
        pass

    async def post(self, path, json):
        type(self).sent.append(json)
        idx = min(len(type(self).sent) - 1, len(type(self).script) - 1)
        payload = type(self).script[idx]

        class _R:
            status_code = 200
            text = "{}"

            def json(self):
                return {"choices": [{"message": payload}]}

        return _R()

    async def aclose(self):
        pass


def _tool_call_response(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": __import__("json").dumps(arguments)},
            }
        ],
    }


@pytest.fixture
def script(monkeypatch):
    """Install the recording httpx client and reset its buffers."""
    import httpx

    _ScriptedClient.script = []
    _ScriptedClient.sent = []
    monkeypatch.setattr(httpx, "AsyncClient", _ScriptedClient)
    return _ScriptedClient


def _loop(ckpt_store: CheckpointStore, settings: Settings, script) -> AgentLoop:
    return AgentLoop(
        llm=DeepSeekChatModel(
            api_key="sk-test",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            timeout=30.0,
        ),
        tools=build_default_registry(settings),
        checkpoint=ckpt_store,
        system_prompt="You are a test agent.",
        settings=settings,
        agent_id="test-agent",
    )


async def _drive_to_interrupt(loop: AgentLoop, prompt: str = "save a note") -> str:
    """Run one turn until the sensitive tool suspends the loop.

    The loop catches HITLInterrupt internally, persists the Checkpoint as
    WAITING_APPROVAL, and reports it as a hitl_required event — so the
    suspension is observable on the event stream, not as an exception.
    """
    events = []
    try:
        async for ev in loop.run(thread_id="t1", user_id="u1", user_message=prompt):
            events.append(ev)
    except HITLInterrupt as exc:  # pragma: no cover - defensive
        return exc.approval_id
    for ev in events:
        if ev.type == "hitl_required":
            return ev.data["approval_id"]
    raise AssertionError(f"expected HITL, got {[str(e.type) for e in events]}")


# --------------------------------------------------------------------------- #
# 1. the actual regression
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_blank_content_resume_does_not_inject_a_user_turn(
    ckpt_store, settings, script
):
    """The exact bug: `content: ""` on resume used to break tool ordering."""
    script.script = [
        _tool_call_response("call_1", "write_file", {"path": "a.txt", "content": "hi"}),
        {"content": "Done — I wrote the file.", "tool_calls": None},
    ]
    loop = _loop(ckpt_store, settings, script)
    approval_id = await _drive_to_interrupt(loop)

    async for _ in loop.run(
        thread_id="t1", user_id="u1", user_message="", approval_id=approval_id
    ):
        pass

    assert len(script.sent) == 2, "resume should reach the model exactly once"
    payload = script.sent[1]["messages"]

    # The precise symptom: an empty USER turn wedged between the assistant's
    # tool_calls and its tool result.
    roles = [m["role"] for m in payload]
    assert "" not in [m.get("content") or "" for m in payload if m["role"] == "user"], (
        f"an empty user message was sent: {roles}"
    )
    for i, m in enumerate(payload):
        if m["role"] == "user":
            assert m["content"].strip(), f"messages[{i}] is a blank user turn"

    assert_valid_openai_order(payload)


@pytest.mark.asyncio
async def test_whitespace_only_content_is_also_treated_as_absent(
    ckpt_store, settings, script
):
    script.script = [
        _tool_call_response("call_1", "write_file", {"path": "b.txt", "content": "x"}),
        {"content": "ok", "tool_calls": None},
    ]
    loop = _loop(ckpt_store, settings, script)
    approval_id = await _drive_to_interrupt(loop)

    async for _ in loop.run(
        thread_id="t1", user_id="u1", user_message="   \n  ", approval_id=approval_id
    ):
        pass

    payload = script.sent[1]["messages"]
    assert all(m["content"].strip() for m in payload if m["role"] == "user")
    assert_valid_openai_order(payload)


@pytest.mark.asyncio
async def test_a_real_user_message_on_resume_still_lands(
    ckpt_store, settings, script
):
    """Normalising blanks must not swallow genuine input."""
    script.script = [
        _tool_call_response("call_1", "write_file", {"path": "c.txt", "content": "x"}),
        {"content": "ok", "tool_calls": None},
    ]
    loop = _loop(ckpt_store, settings, script)
    approval_id = await _drive_to_interrupt(loop)

    async for _ in loop.run(
        thread_id="t1",
        user_id="u1",
        user_message="and also say hello",
        approval_id=approval_id,
    ):
        pass

    payload = script.sent[1]["messages"]
    users = [m["content"] for m in payload if m["role"] == "user"]
    assert users[-1] == "and also say hello", users
    assert_valid_openai_order(payload)


# --------------------------------------------------------------------------- #
# 2. event hygiene
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_hitl_resolved_is_emitted_exactly_once(ckpt_store, settings, script):
    """It used to fire twice — once generically, once from the shortcut."""
    script.script = [
        _tool_call_response("call_1", "write_file", {"path": "d.txt", "content": "x"}),
        {"content": "ok", "tool_calls": None},
    ]
    loop = _loop(ckpt_store, settings, script)
    approval_id = await _drive_to_interrupt(loop)

    names = [
        ev.type
        async for ev in loop.run(
            thread_id="t1", user_id="u1", user_message="", approval_id=approval_id
        )
    ]
    assert names.count("hitl_resolved") == 1, names
    assert names[-1] == "finish", names


@pytest.mark.asyncio
async def test_resume_executes_the_approved_tool_and_finishes(
    ckpt_store, settings, script
):
    """The whole arc: suspend, approve, resume, write, finalize."""
    from pathlib import Path

    script.script = [
        _tool_call_response(
            "call_1", "write_file", {"path": "notes/x.txt", "content": "payload"}
        ),
        {"content": "Wrote it.", "tool_calls": None},
    ]
    loop = _loop(ckpt_store, settings, script)
    approval_id = await _drive_to_interrupt(loop)

    events = [
        ev
        async for ev in loop.run(
            thread_id="t1", user_id="u1", user_message="", approval_id=approval_id
        )
    ]
    kinds = [ev.type for ev in events]

    assert "tool_call" in kinds and "tool_result" in kinds, kinds
    assert Path(settings.tool_write_file_root, "notes", "x.txt").read_text(
        encoding="utf-8"
    ) == "payload"
    assert_valid_openai_order(script.sent[1]["messages"])


@pytest.mark.asyncio
async def test_replaying_a_finished_resume_does_not_crash(ckpt_store, settings, script):
    """A caller that retries a resume it already got a response for.

    Every tool call already has a DONE result, so the shortcut has nothing to
    execute. This path used to reference an unbound `results` local.
    """
    script.script = [
        _tool_call_response("call_1", "write_file", {"path": "e.txt", "content": "x"}),
        {"content": "ok", "tool_calls": None},
    ]
    loop = _loop(ckpt_store, settings, script)
    approval_id = await _drive_to_interrupt(loop)

    async for _ in loop.run(
        thread_id="t1", user_id="u1", user_message="", approval_id=approval_id
    ):
        pass
    calls_after_first = len(script.sent)

    # Replay the identical request.
    events = [
        ev
        async for ev in loop.run(
            thread_id="t1", user_id="u1", user_message="", approval_id=approval_id
        )
    ]
    assert [ev.type for ev in events][-1] == "finish"
    assert len(script.sent) == calls_after_first, "the replay hit the model again"


# --------------------------------------------------------------------------- #
# 3. compression must not orphan a tool result
# --------------------------------------------------------------------------- #


def test_compression_never_starts_with_an_orphan_tool_message() -> None:
    """A `tool` message whose assistant was summarized away is a hard 400.

    _split_recent() counts messages one at a time with no notion of the
    assistant/tool pairing, so it could cut between them.
    """
    from agent_platform.core.agent_loop import _maybe_compress
    from agent_platform.core.messages import Message, ToolCall

    messages = [Message(role=MessageRole.SYSTEM, content="sys")]
    # Build a long history, then force the boundary to land mid-pair by
    # tuning keep_recent.
    for i in range(6):
        messages.append(Message(role=MessageRole.USER, content=f"question {i} " * 30))
        messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"n": i})],
            )
        )
        messages.append(
            Message(role=MessageRole.TOOL, content=f"answer {i}", tool_call_id=f"c{i}")
        )
        messages.append(Message(role=MessageRole.ASSISTANT, content=f"reply {i} " * 30))

    # Sweep every boundary: none of them may produce an orphan.
    for keep in range(1, 12):
        compressed, did = _maybe_compress(
            messages, trigger_tokens=1, keep_recent=keep, chars_per_token=4
        )
        assert did
        assert_valid_openai_order([m.to_openai_dict() for m in compressed])


def test_compression_drops_both_halves_of_a_pair() -> None:
    """When the assistant is summarized away, its tool replies go too."""
    from agent_platform.core.agent_loop import _maybe_compress
    from agent_platform.core.messages import Message, ToolCall

    messages = [
        Message(role=MessageRole.SYSTEM, content="sys " * 50),
        Message(role=MessageRole.USER, content="u " * 50),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
        ),
        Message(role=MessageRole.TOOL, content="tool output", tool_call_id="c1"),
        Message(role=MessageRole.ASSISTANT, content="final " * 50),
    ]
    compressed, _ = _maybe_compress(
        messages, trigger_tokens=1, keep_recent=1, chars_per_token=4
    )
    assert_valid_openai_order([m.to_openai_dict() for m in compressed])
    # "final" is the newest non-system message, so it survives verbatim.
    assert compressed[-1].content.startswith("final")
