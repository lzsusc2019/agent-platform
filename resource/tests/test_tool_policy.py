"""Per-agent tool policy: AgentConfig.tools and AgentConfig.sensitive_tools.

Both fields were dead. to_loop_config() handed them to AgentLoop, but the Loop
only ever read config["model"] — tool schemas came from the whole registry and
sensitivity from Tool.sensitive alone. An operator could set
tools: ["echo"], watch the model get offered all three tools anyway, and set
sensitive_tools: [] and watch write_file ask for approval regardless.

Two semantics are worth stating because they are choices, not defaults:

* An EMPTY tools list means "no restriction", not "no tools". Every existing
  config and every Dashboard-created agent has an empty list; reading it as
  "no tools" would have silently disarmed them all.
* sensitive_tools can only ADD to what the code declares. Dropping an approval
  gate is not something a config typo should be able to do.
"""

from __future__ import annotations

import pytest

from agent_platform.config.settings import Settings
from agent_platform.domain.agent_loop import AgentLoop
from agent_platform.domain.llm import ChatModel, LLMResponse, MockChatModel
from agent_platform.domain.messages import ToolCall
from agent_platform.infra.checkpoint_store import CheckpointStore
from agent_platform.tools import build_default_registry


class _RogueModel(ChatModel):
    """Calls a tool it was never offered, then stops.

    MockChatModel is well-behaved: it only reaches for tools present in the
    schemas it was handed, so it cannot exercise the case that matters. A real
    model can name any tool at all — through hallucination or a prompt
    injection in tool output — which is precisely why the schema list being
    advisory is not good enough.
    """

    def __init__(self, tool_name: str, arguments: dict | None = None) -> None:
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.calls = 0

    async def ainvoke(self, messages, tools):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls > 1:
            return LLMResponse(content="done", tool_calls=[])
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="rogue_1", name=self.tool_name, arguments=self.arguments)
            ],
        )


def _loop(
    ckpt_store: CheckpointStore,
    settings: Settings,
    *,
    llm: ChatModel | None = None,
    **config,
) -> AgentLoop:
    return AgentLoop(
        llm=llm or MockChatModel(),
        tools=build_default_registry(settings),
        checkpoint=ckpt_store,
        system_prompt="You are a test agent.",
        settings=settings,
        agent_id="policy-agent",
        config=config,
    )


async def _drain(loop: AgentLoop, **kwargs):
    return [ev async for ev in loop.run(**kwargs)]


# --------------------------------------------------------------------------- #
# which tools the agent is offered
# --------------------------------------------------------------------------- #


def test_empty_tools_list_means_no_restriction(ckpt_store, settings) -> None:
    """The backward-compatibility guarantee."""
    loop = _loop(ckpt_store, settings)
    offered = {s["name"] for s in loop.tools.schemas_for_llm(loop._allowed_tools)}
    assert offered == {"echo", "http_get", "write_file"}


def test_tools_list_narrows_what_the_model_sees(ckpt_store, settings) -> None:
    loop = _loop(ckpt_store, settings, tools=["echo"])
    offered = {
        s["name"] for s in loop.tools.schemas_for_llm(loop._allowed_tools)
    }
    assert offered == {"echo"}


def test_restriction_does_not_mutate_the_registry(ckpt_store, settings) -> None:
    """The registry is shared by every agent; narrowing one must not affect it."""
    registry = build_default_registry(settings)
    loop = AgentLoop(
        llm=MockChatModel(),
        tools=registry,
        checkpoint=ckpt_store,
        system_prompt="x",
        settings=settings,
        agent_id="a",
        config={"tools": ["echo"]},
    )
    assert loop._allowed_tools == {"echo"}
    assert set(registry.names()) == {"echo", "http_get", "write_file"}
    assert {s["name"] for s in registry.schemas_for_llm()} == {
        "echo",
        "http_get",
        "write_file",
    }


def test_unknown_name_in_tools_list_is_inert(ckpt_store, settings) -> None:
    """A typo'd or since-deleted tool must not crash the agent."""
    loop = _loop(ckpt_store, settings, tools=["echo", "no_such_tool"])
    offered = {
        s["name"] for s in loop.tools.schemas_for_llm(loop._allowed_tools)
    }
    assert offered == {"echo"}


# --------------------------------------------------------------------------- #
# enforcement, not just advertising
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_disallowed_tool_call_is_refused_at_execution(ckpt_store, settings) -> None:
    """The schema list is an offering. This is the gate.

    The prompt makes MockChatModel reach for 'echo', which this agent may not
    use. It must come back as an error result rather than run.
    """
    loop = _loop(
        ckpt_store,
        settings,
        llm=_RogueModel("echo", {"text": "hi"}),
        tools=["write_file"],
    )
    events = await _drain(
        loop, thread_id="t1", user_id="u1", user_message="anything"
    )
    kinds = [ev.type for ev in events]

    assert "tool_call" in kinds, kinds
    results = next(ev.data["results"] for ev in events if ev.type == "tool_result")
    (only,) = results
    assert only["name"] == "echo"
    assert only["is_error"] is True, only
    assert "not enabled for agent 'policy-agent'" in only["content"], only


@pytest.mark.asyncio
async def test_allowed_tool_still_runs(ckpt_store, settings) -> None:
    loop = _loop(ckpt_store, settings, tools=["echo"])
    events = await _drain(
        loop, thread_id="t1", user_id="u1", user_message="please echo hello"
    )
    results = next(ev.data["results"] for ev in events if ev.type == "tool_result")
    assert results[0]["is_error"] is False, results


@pytest.mark.asyncio
async def test_disallowed_sensitive_tool_does_not_raise_a_pointless_prompt(
    ckpt_store, settings
) -> None:
    """Asking a human to approve something the agent may not do is noise.

    The answer is 'no' regardless of what they say, so the call is refused
    outright instead of parking the thread.
    """
    loop = _loop(
        ckpt_store,
        settings,
        llm=_RogueModel("write_file", {"path": "a.txt", "content": "x"}),
        tools=["echo"],
    )
    events = await _drain(
        loop, thread_id="t1", user_id="u1", user_message="anything"
    )
    kinds = [ev.type for ev in events]
    assert "hitl_required" not in kinds, kinds
    results = next(ev.data["results"] for ev in events if ev.type == "tool_result")
    assert results[0]["is_error"] is True
    # And nothing was written on the way to being refused.
    from pathlib import Path

    assert not (Path(settings.tool_write_file_root) / "a.txt").exists()


# --------------------------------------------------------------------------- #
# sensitivity: config may add gates, never remove them
# --------------------------------------------------------------------------- #


def test_code_declared_sensitivity_applies_with_no_config(ckpt_store, settings) -> None:
    loop = _loop(ckpt_store, settings)
    assert loop.tool_is_sensitive("write_file")
    assert not loop.tool_is_sensitive("echo")
    assert not loop.tool_is_sensitive("http_get")


def test_config_cannot_un_gate_a_code_declared_sensitive_tool(
    ckpt_store, settings
) -> None:
    """An operator must not be able to switch off an approval gate by config."""
    loop = _loop(ckpt_store, settings, tools=["write_file"], sensitive_tools=[])
    assert loop.tool_is_sensitive("write_file"), (
        "write_file is sensitive on the class; an empty sensitive_tools list "
        "must not disarm it"
    )


def test_config_can_gate_an_otherwise_safe_tool(ckpt_store, settings) -> None:
    loop = _loop(ckpt_store, settings, sensitive_tools=["http_get"])
    assert loop.tool_is_sensitive("http_get")
    assert loop.tool_is_sensitive("write_file")  # union, not replacement


@pytest.mark.asyncio
async def test_config_gated_tool_triggers_hitl(ckpt_store, settings) -> None:
    """The new capability: HITL on a tool that code alone would let through."""
    loop = _loop(ckpt_store, settings, sensitive_tools=["echo"])
    events = await _drain(
        loop, thread_id="t1", user_id="u1", user_message="please echo hello"
    )
    kinds = [ev.type for ev in events]
    assert "hitl_required" in kinds, kinds
    hitl = next(ev for ev in events if ev.type == "hitl_required")
    assert hitl.data["tool_name"] == "echo"


# --------------------------------------------------------------------------- #
# the pending-approval record
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_approval_id_is_persisted_on_the_waiting_snapshot(
    ckpt_store, settings
) -> None:
    """The admin API lists pending approvals from the Checkpoint, so the id has
    to live there rather than being reconstructed from the thread_id."""
    from agent_platform.domain.checkpoint import CheckpointStatus

    loop = _loop(ckpt_store, settings)
    events = await _drain(
        loop, thread_id="t1", user_id="u1", user_message="save a file please"
    )
    announced = next(ev.data["approval_id"] for ev in events if ev.type == "hitl_required")

    snap = await ckpt_store.load("t1")
    assert snap is not None
    assert snap.status == CheckpointStatus.WAITING_APPROVAL
    assert snap.approval_id == announced


@pytest.mark.asyncio
async def test_approval_id_is_cleared_once_consumed(ckpt_store, settings) -> None:
    """Leaving it set would make a finished thread look approvable forever."""
    loop = _loop(ckpt_store, settings)
    events = await _drain(
        loop, thread_id="t1", user_id="u1", user_message="save a file please"
    )
    approval_id = next(
        ev.data["approval_id"] for ev in events if ev.type == "hitl_required"
    )
    await _drain(
        loop, thread_id="t1", user_id="u1", user_message="", approval_id=approval_id
    )
    snap = await ckpt_store.load("t1")
    assert snap is not None
    assert snap.approval_id is None
