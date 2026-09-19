"""Tests for LLM provider routing and DeepSeekChatModel."""

from __future__ import annotations

import pytest

from agent_platform.config import Settings
from agent_platform.core.llm import MockChatModel
from agent_platform.core.providers import (
    DeepSeekChatModel,
    available_providers,
    create_chat_model,
    parse_model_string,
    register_provider,
)


def test_parse_model_string() -> None:
    assert parse_model_string("mock") == ("mock", None)
    assert parse_model_string("deepseek") == ("deepseek", None)
    assert parse_model_string("deepseek:deepseek-chat") == ("deepseek", "deepseek-chat")
    assert parse_model_string("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")


def test_available_providers_includes_mock() -> None:
    providers = available_providers()
    assert "mock" in providers


def test_create_chat_model_mock() -> None:
    s = Settings()
    cm = create_chat_model("mock", s)
    assert isinstance(cm, MockChatModel)


def test_create_chat_model_deepseek_missing_key_falls_back(caplog) -> None:
    s = Settings(deepseek_api_key="")  # no key
    cm = create_chat_model("deepseek:deepseek-chat", s)
    # Falls back to mock when key is missing and require_key is False.
    assert isinstance(cm, MockChatModel)


def test_create_chat_model_deepseek_missing_key_raises_when_required() -> None:
    s = Settings(deepseek_api_key="", deepseek_require_key=True)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        create_chat_model("deepseek", s)


def test_create_chat_model_deepseek_with_key() -> None:
    s = Settings(
        deepseek_api_key="sk-test",
        deepseek_model="deepseek-chat",
        deepseek_base_url="https://api.deepseek.com",
    )
    cm = create_chat_model("deepseek:deepseek-chat", s)
    assert isinstance(cm, DeepSeekChatModel)


def test_create_chat_model_deepseek_uses_default_model_when_unspecified() -> None:
    s = Settings(
        deepseek_api_key="sk-test",
        deepseek_model="deepseek-reasoner",
    )
    cm = create_chat_model("deepseek", s)  # no model suffix
    assert isinstance(cm, DeepSeekChatModel)
    assert cm._model == "deepseek-reasoner"


def test_create_chat_model_unknown_provider_raises() -> None:
    s = Settings()
    with pytest.raises(ValueError, match="unknown LLM provider"):
        create_chat_model("nonexistent-llm", s)


def test_register_provider_then_route() -> None:
    """Custom provider registration via the public API."""
    class _Fake(ChatModel := __import__("agent_platform.core.llm", fromlist=["ChatModel"]).ChatModel):
        async def ainvoke(self, messages, tools):
            from agent_platform.core.llm import LLMResponse
            return LLMResponse(content="fake")

    register_provider("fake-test", _Fake)
    try:
        cm = create_chat_model("fake-test", Settings())
        assert isinstance(cm, _Fake)
    finally:
        # Clean up so other tests aren't affected.
        from agent_platform.core import providers

        providers._PROVIDERS.pop("fake-test", None)


def test_deepseek_chatmodel_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        DeepSeekChatModel(api_key="", base_url="https://x", model="m", timeout=1.0)


@pytest.mark.asyncio
async def test_deepseek_chatmodel_sends_correct_request_shape(monkeypatch) -> None:
    """Verify the HTTP payload DeepSeekChatModel sends matches OpenAI's spec."""
    from agent_platform.core.providers import DeepSeekChatModel

    captured: dict = {}

    class _StubResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "hi from deepseek",
                            "tool_calls": None,
                        }
                    }
                ]
            }

    class _StubClient:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, path, json):
            captured["path"] = path
            captured["payload"] = json
            return _StubResponse()

        async def aclose(self):
            pass

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _StubClient)

    cm = DeepSeekChatModel(
        api_key="sk-x",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        timeout=30.0,
    )

    resp = await cm.ainvoke(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "echo", "description": "x", "parameters": {"type": "object"}}],
    )
    assert captured["path"] == "/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == "deepseek-chat"
    assert payload["stream"] is False
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == [
        {"type": "function", "function": {"name": "echo", "description": "x", "parameters": {"type": "object"}}}
    ]
    assert resp.content == "hi from deepseek"
    await cm.aclose()


@pytest.mark.asyncio
async def test_deepseek_chatmodel_handles_4xx_error(monkeypatch) -> None:
    from agent_platform.core.providers import DeepSeekChatModel

    class _ErrResponse:
        status_code = 401
        text = '{"error":{"message":"Invalid API key"}}'

        def json(self):
            return {"error": {"message": "Invalid API key"}}

    class _StubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def post(self, path, json):
            return _ErrResponse()

        async def aclose(self):
            pass

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _StubClient)
    cm = DeepSeekChatModel(
        api_key="bad", base_url="https://x", model="m", timeout=1.0
    )
    with pytest.raises(RuntimeError, match="Invalid API key"):
        await cm.ainvoke(messages=[{"role": "user", "content": "x"}], tools=[])
    await cm.aclose()


# ----- hyphen alias parsing (added after a "deepseek-flash" bug report) ----


def test_parse_model_string_hyphen_alias_deepseek() -> None:
    """`deepseek-flash` names provider deepseek AND the model deepseek-flash.

    Regression: an earlier version stripped the `deepseek-` prefix and sent
    the API `model=flash`, which 400s with "The supported API model names
    are deepseek-flash, deepseek-v4-pro, but you passed flash." DeepSeek's
    model ids include their provider prefix, so the whole string is the
    model name.
    """
    assert parse_model_string("deepseek-flash") == ("deepseek", "deepseek-flash")
    assert parse_model_string("deepseek-v4-pro") == ("deepseek", "deepseek-v4-pro")
    assert parse_model_string("deepseek-chat") == ("deepseek", "deepseek-chat")


def test_parse_model_string_hyphen_alias_does_not_swallow_unknown() -> None:
    """A hyphenated name we don't recognize stays as-is —
    let create_chat_model raise the 'unknown provider' error.
    """
    assert parse_model_string("claude-3-haiku") == ("claude-3-haiku", None)
    assert parse_model_string("gpt-4o") == ("gpt-4o", None)


def test_create_chat_model_hyphen_alias_works() -> None:
    """End-to-end: `deepseek-flash` builds a DeepSeekChatModel pointed at the
    full model id, rather than raising `unknown provider`."""
    s = Settings(deepseek_api_key="sk-test")
    cm = create_chat_model("deepseek-flash", s)
    assert isinstance(cm, DeepSeekChatModel)
    assert cm._model == "deepseek-flash"


def test_hyphen_shorthand_and_qualified_form_agree() -> None:
    """`deepseek-flash` and `deepseek:deepseek-flash` must send the same id."""
    s = Settings(deepseek_api_key="sk-test")
    shorthand = create_chat_model("deepseek-flash", s)
    qualified = create_chat_model("deepseek:deepseek-flash", s)
    assert shorthand._model == qualified._model == "deepseek-flash"


def test_create_chat_model_unknown_with_hyphen_still_raises() -> None:
    """A typo like 'claude-3-haiku' should raise a clear 'unknown provider'
    error rather than silently route to deepseek."""
    s = Settings(deepseek_api_key="sk-test")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        create_chat_model("claude-3-haiku", s)


def asyncio_run(coro):
    """Run a coroutine from a sync test."""
    import asyncio

    return asyncio.run(coro)



# ----- retry classification -------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_retryable"),
    [
        (400, False),  # unknown model name / bad request — retrying is pointless
        (401, False),  # bad key — retrying is pointless
        (403, False),
        (404, False),
        (422, False),
        (408, True),  # request timeout — transient
        (409, True),
        (429, True),  # rate limited — back off and retry
        (500, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
def test_http_status_retry_classification(monkeypatch, status, expected_retryable) -> None:
    from agent_platform.core.llm import LLMError

    class _Resp:
        status_code = status
        text = '{"error":{"message":"nope"}}'

        def json(self):
            return {"error": {"message": "nope"}}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def post(self, path, json):
            return _Resp()

        async def aclose(self):
            pass

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    cm = DeepSeekChatModel(
        api_key="sk-x", base_url="https://x", model="deepseek-flash", timeout=1.0
    )
    with pytest.raises(LLMError) as ei:
        asyncio_run(cm.ainvoke(messages=[{"role": "user", "content": "x"}], tools=[]))
    assert ei.value.retryable is expected_retryable, (
        f"HTTP {status} classified as retryable={ei.value.retryable}"
    )


@pytest.mark.asyncio
async def test_timeout_is_retryable(monkeypatch) -> None:
    from agent_platform.core.llm import LLMError

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def post(self, path, json):
            import httpx

            raise httpx.TimeoutException("too slow")

        async def aclose(self):
            pass

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    cm = DeepSeekChatModel(
        api_key="sk-x", base_url="https://x", model="deepseek-flash", timeout=1.0
    )
    with pytest.raises(LLMError) as ei:
        await cm.ainvoke(messages=[{"role": "user", "content": "x"}], tools=[])
    assert ei.value.retryable is True


@pytest.mark.asyncio
async def test_error_message_names_the_model(monkeypatch) -> None:
    """A 400 must say which model id we sent — that is the whole diagnosis."""
    from agent_platform.core.llm import LLMError

    class _Resp:
        status_code = 400
        text = '{"error":{"message":"The supported API model names are ..."}}'

        def json(self):
            return {
                "error": {"message": "The supported API model names are ..."}
            }

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def post(self, path, json):
            return _Resp()

        async def aclose(self):
            pass

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    cm = DeepSeekChatModel(
        api_key="sk-x",
        base_url="https://x",
        model="deepseek-flash",
        timeout=1.0,
    )
    with pytest.raises(LLMError, match="model=deepseek-flash"):
        await cm.ainvoke(messages=[{"role": "user", "content": "x"}], tools=[])

