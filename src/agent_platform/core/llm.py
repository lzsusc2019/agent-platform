"""LLM abstraction.

Mirrors the Tool pattern: a small interface that wraps whatever provider we
plug in. MVP ships only a deterministic mock used by tests + the default
CLI demo. Real OpenAI/Anthropic adapters are scaffolded but not wired.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from agent_platform.core.messages import ToolCall

__all__ = [
    "ChatModel",
    "LLMError",
    "LLMResponse",
    "MockChatModel",
]


class LLMError(RuntimeError):
    """A provider call failed.

    `retryable` tells the Agent Loop whether backing off and trying again
    could plausibly help. Getting this wrong is expensive in both
    directions:

    - Retrying a 400 (bad model name, malformed request) burns the full
      backoff budget — 3.5s of dead time — and then fails anyway.
    - Not retrying a timeout or a 503 turns a blip into a user-visible
      failure.

    So the provider decides, and the Loop just obeys.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMResponse(BaseModel):
    """What an LLM call returns — message + optional tool calls.

    `content` is the assistant's natural-language reply. If the LLM wanted to
    invoke tools, those go in `tool_calls`. Both can be present (e.g.
    "Let me check that for you." + [tool_call]).
    """

    content: str
    tool_calls: list[ToolCall] = []


class ChatModel(ABC):
    """The interface every provider must satisfy."""

    @abstractmethod
    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Send messages + tool schemas; return one assistant response."""


class MockChatModel(ChatModel):
    """Deterministic LLM stand-in for tests and the demo.

    Behaviour is driven by the message history:
    - First turn with a user message containing the word "echo": returns an
      echo tool call with the literal word that follows "echo".
    - First turn with a user message expressing a write intent: returns a
      write_file tool call (a sensitive tool, so HITL triggers).
    - Otherwise: returns a plain assistant message.

    Tests can subclass or replace this with their own scripted model.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        self.call_count += 1
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        user_text = (last_user or {}).get("content", "") or ""
        tool_names = {t["name"] for t in tools}
        # If the CURRENT turn already received a tool result (i.e. the most
        # recent assistant message had tool_calls and they've all been
        # answered), the agent is in the "finalize" phase: stop calling
        # tools and produce a final answer. This mirrors a real LLM that,
        # after seeing the tool response, decides the task is done.
        # We deliberately scope this to the latest turn, not the whole
        # history, so a follow-up user message can request new tools.
        if messages:
            last = messages[-1]
            if last.get("role") == "tool":
                # Walk back to the corresponding assistant tool_calls to know
                # whether all of them have been answered.
                tc_ids: set[str] = set()
                answered: set[str] = set()
                for m in reversed(messages):
                    if m.get("role") == "tool":
                        answered.add(m.get("tool_call_id", ""))
                    elif m.get("role") == "assistant":
                        for tc in m.get("tool_calls", []) or []:
                            tc_ids.add(tc.get("id", ""))
                        break  # only the most recent assistant turn
                    else:
                        break
                if tc_ids and tc_ids <= answered:
                    joined = " | ".join(
                        m.get("content", "")
                        for m in messages
                        if m.get("role") == "tool"
                    )
                    return LLMResponse(content=f"done: {joined}")

        # Sensitive tool trigger. Any write intent routes to write_file,
        # which is marked sensitive and therefore must clear HITL.
        low = user_text.lower()
        wants_write = any(t in low for t in ("write", "save", "create file")) or any(
            t in user_text for t in ("写", "保存", "写入")
        )
        if "write_file" in tool_names and wants_write:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call_{self.call_count}",
                        name="write_file",
                        arguments={
                            "path": "notes/demo.txt",
                            "content": "written by the demo agent",
                        },
                    )
                ],
            )

        # Echo tool trigger.
        if "echo" in tool_names and "echo" in user_text.lower():
            # Pull out the word after "echo" if any.
            payload = user_text.lower().split("echo", 1)[1].strip() or "hello"
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call_{self.call_count}",
                        name="echo",
                        arguments={"text": payload},
                    )
                ],
            )

        # Default: plain answer. Reference any tool results that came back so
        # tests can verify tool results flowed through.
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if tool_msgs:
            joined = " | ".join(m.get("content", "") for m in tool_msgs)
            return LLMResponse(content=f"done: {joined}")
        return LLMResponse(content="ok")
