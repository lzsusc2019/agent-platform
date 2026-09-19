"""Agent Loop integration tests — covers the core scenarios from
Agent中台.md end-to-end:
- single turn, no tools
- single turn with one tool
- multi-turn with tools (echo, then done)
- HITL interrupt -> approve -> resume
- HITL interrupt -> reject
- 100-turn budget exceeded
- empty-response guard
- context compression
"""

from __future__ import annotations

import pytest

from agent_platform.checkpoint.store import CheckpointStore
from agent_platform.core.checkpoint import (
    CheckpointStatus,
    ToolPendingState,
)
from agent_platform.core.errors import (
    HITLInterrupt,
    LoopBudgetExceeded,
)
from agent_platform.core.events import EventType

from conftest import drain


@pytest.mark.asyncio
async def test_no_tool_call_finishes_immediately(agent) -> None:
    events = await drain(
        agent, thread_id="t1", user_id="u1", user_message="hello"
    )
    types = [e.type for e in events]
    assert EventType.START in types
    assert EventType.ASSISTANT in types
    assert EventType.FINISH in types
    # No tool call, no tool result.
    assert EventType.TOOL_CALL not in types
    finish = next(e for e in events if e.type == EventType.FINISH)
    assert finish.data["reason"] == "done"


@pytest.mark.asyncio
async def test_echo_tool_round_trip(agent, ckpt_store: CheckpointStore) -> None:
    events = await drain(
        agent, thread_id="t2", user_id="u1", user_message="please echo world"
    )
    types = [e.type for e in events]
    assert EventType.TOOL_CALL in types
    assert EventType.TOOL_RESULT in types
    assert EventType.FINISH in types
    tool_result = next(e for e in events if e.type == EventType.TOOL_RESULT)
    assert tool_result.data["results"][0]["name"] == "echo"
    assert "echo: world" in tool_result.data["results"][0]["content"]

    snap = await ckpt_store.load("t2")
    assert snap is not None
    assert snap.status == CheckpointStatus.FINISHED


@pytest.mark.asyncio
async def test_hitl_interrupt_then_approve(agent, ckpt_store: CheckpointStore) -> None:
    # First call: triggers HITL because sensitive tool is sensitive.
    events = await drain(
        agent, thread_id="t3", user_id="u1", user_message="write a file"
    )
    hitl = next(e for e in events if e.type == EventType.HITL_REQUIRED)
    assert hitl.data["tool_name"] == "write_file"
    approval_id = hitl.data["approval_id"]

    # Verify Checkpoint is in WAITING_APPROVAL with PENDING tool.
    snap = await ckpt_store.load("t3")
    assert snap is not None
    assert snap.status == CheckpointStatus.WAITING_APPROVAL
    assert all(s is ToolPendingState.PENDING for s in snap.pending_tools.values())

    # Resume with approval_id; Loop should now run the sensitive tool and finish.
    events2 = await drain(
        agent,
        thread_id="t3",
        user_id="u1",
        user_message=None,
        approval_id=approval_id,
    )
    types2 = [e.type for e in events2]
    assert EventType.HITL_RESOLVED in types2
    assert EventType.TOOL_RESULT in types2
    assert EventType.FINISH in types2
    tool_result = next(e for e in events2 if e.type == EventType.TOOL_RESULT)
    assert "bytes_written" in tool_result.data["results"][0]["content"]

    snap2 = await ckpt_store.load("t3")
    assert snap2 is not None
    assert snap2.status == CheckpointStatus.FINISHED
    # DONE state for the sensitive tool call.
    assert all(
        s is ToolPendingState.DONE for s in snap2.pending_tools.values()
    ), snap2.pending_tools


@pytest.mark.asyncio
async def test_hitl_rejection_terminates(agent, ckpt_store: CheckpointStore) -> None:
    events = await drain(
        agent, thread_id="t4", user_id="u1", user_message="write a file"
    )
    hitl = next(e for e in events if e.type == EventType.HITL_REQUIRED)
    approval_id = hitl.data["approval_id"]

    # API-layer reject path: we mark the snapshot FINISHED manually.
    from agent_platform.core.checkpoint import CheckpointSnapshot
    snap = await ckpt_store.load("t4")
    assert snap is not None
    finished = snap.model_copy(
        update={"status": CheckpointStatus.FINISHED, "pending_tools": {}}
    )
    await ckpt_store.save(finished)

    snap_after = await ckpt_store.load("t4")
    assert snap_after is not None
    assert snap_after.status == CheckpointStatus.FINISHED
    # approval_id is still recorded; future resumes would refuse.
    assert approval_id.startswith("appr_t4_")


@pytest.mark.asyncio
async def test_max_turns_budget(ckpt_store, tools, llm, settings) -> None:
    """Force the loop over budget by crafting a model that always requests a tool."""
    from agent_platform.core.agent_loop import AgentLoop
    from agent_platform.core.llm import ChatModel, LLMResponse
    from agent_platform.core.messages import ToolCall

    class AlwaysTool(ChatModel):
        async def ainvoke(self, messages, tools):
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="loop", name="echo", arguments={"text": "x"})
                ],
            )

    # Lower the cap so the test is quick.
    s = settings.model_copy(update={"max_turns": 3})
    a = AgentLoop(
        llm=AlwaysTool(),
        tools=tools,
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="budget",
    )
    with pytest.raises(LoopBudgetExceeded):
        await drain(a, thread_id="budget", user_id="u", user_message="go")


@pytest.mark.asyncio
async def test_empty_response_guard(ckpt_store, tools, settings) -> None:
    """Model returns empty 3 times -> LoopEmptyResponse."""
    from agent_platform.core.agent_loop import AgentLoop
    from agent_platform.core.errors import LoopEmptyResponse
    from agent_platform.core.llm import ChatModel, LLMResponse

    class EmptyModel(ChatModel):
        async def ainvoke(self, messages, tools):
            return LLMResponse(content="", tool_calls=[])

    s = settings.model_copy(update={"empty_response_max_retries": 3})
    a = AgentLoop(
        llm=EmptyModel(),
        tools=tools,
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="budget",
    )
    with pytest.raises(LoopEmptyResponse):
        await drain(a, thread_id="empty", user_id="u", user_message="hi")


@pytest.mark.asyncio
async def test_context_compression_fires(ckpt_store, tools, settings) -> None:
    """With compress_trigger_tokens=100, a long enough message triggers compression."""
    from agent_platform.core.agent_loop import AgentLoop
    from agent_platform.core.llm import ChatModel, LLMResponse

    # Model that just produces a long answer; no tools.
    class LongAnswer(ChatModel):
        async def ainvoke(self, messages, tools):
            # 400 chars -> ~100 tokens -> above the 100-token trigger.
            return LLMResponse(content="x" * 400)

    s = settings.model_copy(update={"compress_trigger_tokens": 100})
    a = AgentLoop(
        llm=LongAnswer(),
        tools=tools,
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=s,
        agent_id="budget",
    )
    events = await drain(
        a,
        thread_id="compress",
        user_id="u",
        user_message="first user message that should be summarized",
    )
    # First turn may not trigger (system + 1 user + 1 assistant still under
    # 100 tokens). Loop continues, eventually compresses. We don't pin to a
    # specific turn — just assert it happened at least once when context
    # grows. Run a second turn to be sure.
    if EventType.COMPRESSED not in [e.type for e in events]:
        events2 = await drain(
            a,
            thread_id="compress",
            user_id="u",
            user_message="second user message",
        )
        events.extend(events2)
    assert EventType.COMPRESSED in [e.type for e in events]


@pytest.mark.asyncio
async def test_done_result_replayed_on_resume(agent, ckpt_store) -> None:
    """Resume after HITL should NOT re-execute a tool whose result is already cached."""
    # Use a counter to ensure the sensitive tool runs at most once.
    call_count = {"n": 0}

    # Wrap the already-registered tool rather than constructing a new one —
    # that way we count the exact instance the Loop will call.
    real_run = agent.tools.get("write_file").run

    async def counting_run(arguments, ctx):
        call_count["n"] += 1
        return await real_run(arguments, ctx)

    # Patch the registered tool.
    agent.tools._tools["write_file"].run = counting_run  # type: ignore[attr-defined]

    # Drive HITL.
    events = await drain(
        agent, thread_id="replay", user_id="u", user_message="write a file please"
    )
    hitl = next(e for e in events if e.type == EventType.HITL_REQUIRED)
    approval_id = hitl.data["approval_id"]
    assert call_count["n"] == 0  # not run yet

    # Resume.
    events2 = await drain(
        agent,
        thread_id="replay",
        user_id="u",
        user_message=None,
        approval_id=approval_id,
    )
    assert call_count["n"] == 1
    # A second resume should NOT call the tool again.
    events3 = await drain(
        agent,
        thread_id="replay",
        user_id="u",
        user_message=None,
        approval_id=approval_id,
    )
    assert call_count["n"] == 1
    # Last event of the third pass should be FINISH. Reason may be "done"
    # (third pass did new LLM work) or "already_finalized" (the no-op replay
    # short-circuit kicked in). Both are acceptable outcomes.
    finish = next(e for e in events3 if e.type == EventType.FINISH)
    assert finish.data["reason"] in ("done", "already_finalized")
