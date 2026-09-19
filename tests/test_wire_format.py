"""Wire-format tests for outgoing provider requests.

These exist because a malformed `tool_calls` payload slipped through every
other test: the mock model never serializes a request, so nothing exercised
the OpenAI wire format until a real DeepSeek call returned

    422: messages[4]: missing field `type`

What we had was `{"id": ..., "name": ..., "args": {...}}`. What OpenAI and
DeepSeek accept is `{"id": ..., "type": "function", "function": {"name":
..., "arguments": "<json string>"}}` — the name and arguments are nested one
level deeper, `type` is mandatory, and `arguments` is a JSON-encoded string.
"""

from __future__ import annotations

import json

import pytest

from agent_platform.core.messages import Message, MessageRole, ToolCall
from agent_platform.core.providers import DeepSeekChatModel


class _CapturingClient:
    """Stands in for httpx.AsyncClient and records the request payload."""

    def __init__(self, *args, **kwargs):
        self.sent: list[dict] = []

    async def post(self, path, json):
        self.sent.append(json)

        class _R:
            status_code = 200
            text = "{}"

            def json(self):
                return {
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}]
                }

        return _R()

    async def aclose(self):
        pass


@pytest.fixture
def capture(monkeypatch):
    """Patch httpx so we can inspect exactly what would go on the wire."""
    import httpx

    clients: list[_CapturingClient] = []

    def _factory(*args, **kwargs):
        c = _CapturingClient(*args, **kwargs)
        clients.append(c)
        return c

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return clients


def _model() -> DeepSeekChatModel:
    return DeepSeekChatModel(
        api_key="sk-x",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        timeout=30.0,
    )


# ----- the shapes that were wrong ------------------------------------------


def test_assistant_tool_call_matches_openai_shape() -> None:
    """The exact bug: `type` present, namespaced under `function`."""
    msg = Message(
        role=MessageRole.ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})],
    )
    out = msg.to_openai_dict()

    assert out["role"] == "assistant"
    assert out["content"] == ""
    (tc,) = out["tool_calls"]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function", "missing `type` is a hard 422"
    assert tc["function"]["name"] == "echo"
    assert "name" not in tc, "name belongs under `function`, not beside it"
    assert "args" not in tc, "the key is `arguments`, not `args`"


def test_tool_arguments_are_a_json_string_not_an_object() -> None:
    msg = Message(
        role=MessageRole.ASSISTANT,
        tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "hi"})],
    )
    args = msg.to_openai_dict()["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str), "arguments must be a JSON-encoded string"
    assert json.loads(args) == {"text": "hi"}


def test_non_ascii_arguments_survive_round_trip() -> None:
    """ensure_ascii=False keeps Chinese arguments readable on the wire."""
    msg = Message(
        role=MessageRole.ASSISTANT,
        tool_calls=[ToolCall(id="c", name="echo", arguments={"city": "东莞"})],
    )
    args = msg.to_openai_dict()["tool_calls"][0]["function"]["arguments"]
    assert "东莞" in args
    assert json.loads(args) == {"city": "东莞"}


def test_tool_message_shape() -> None:
    msg = Message(role=MessageRole.TOOL, content="42", tool_call_id="call_1")
    assert msg.to_openai_dict() == {
        "role": "tool",
        "content": "42",
        "tool_call_id": "call_1",
    }


def test_plain_messages_shape() -> None:
    assert Message(role=MessageRole.SYSTEM, content="s").to_openai_dict() == {
        "role": "system",
        "content": "s",
    }
    assert Message(role=MessageRole.USER, content="u").to_openai_dict() == {
        "role": "user",
        "content": "u",
    }


# ----- the same shapes, end to end through a real request ------------------


@pytest.mark.asyncio
async def test_full_tool_round_trip_payload(capture) -> None:
    """A conversation containing a tool call must serialize cleanly.

    This is the exact history shape that produced the 422: system, user,
    assistant-with-tool-calls, tool-result.
    """
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful.").to_openai_dict(),
        Message(role=MessageRole.USER, content="东莞天气").to_openai_dict(),
        Message(
            role=MessageRole.ASSISTANT,
            tool_calls=[
                ToolCall(id="call_1", name="http_get", arguments={"url": "http://x"})
            ],
        ).to_openai_dict(),
        Message(
            role=MessageRole.TOOL, content="sunny", tool_call_id="call_1"
        ).to_openai_dict(),
    ]

    cm = _model()
    await cm.ainvoke(messages=messages, tools=[])

    sent = capture[0].sent[0]
    assert sent["model"] == "deepseek-flash"
    assert len(sent["messages"]) == 4

    assistant = sent["messages"][2]
    assert assistant["role"] == "assistant"
    (tc,) = assistant["tool_calls"]
    assert tc["type"] == "function"
    assert isinstance(tc["function"]["arguments"], str)
    assert tc["function"]["name"] == "http_get"

    tool_msg = sent["messages"][3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_every_message_has_a_role(capture) -> None:
    """No message may be sent without a role — the other 422 shape."""
    messages = [
        Message(role=MessageRole.SYSTEM, content="s").to_openai_dict(),
        Message(role=MessageRole.USER, content="u").to_openai_dict(),
        Message(
            role=MessageRole.ASSISTANT,
            content="thinking",
            tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
        ).to_openai_dict(),
        Message(role=MessageRole.TOOL, content="r", tool_call_id="c1").to_openai_dict(),
    ]
    cm = _model()
    await cm.ainvoke(messages=messages, tools=[])
    for i, m in enumerate(capture[0].sent[0]["messages"]):
        assert "role" in m, f"messages[{i}] has no role"


@pytest.mark.asyncio
async def test_tool_schemas_use_openai_function_format(capture) -> None:
    """The `tools` array must wrap each schema in {"type": "function"}."""
    cm = _model()
    await cm.ainvoke(
        messages=[{"role": "user", "content": "x"}],
        tools=[
            {
                "name": "echo",
                "description": "Echo.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    sent = capture[0].sent[0]
    assert sent["tool_choice"] == "auto"
    (entry,) = sent["tools"]
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "echo"
    assert "name" not in entry
