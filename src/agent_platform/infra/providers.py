"""LLM provider implementations and a tiny factory for routing.

ChatModel is the abstract interface (see core/llm.py). Two concrete
implementations ship with the MVP:

- MockChatModel — deterministic, no network, used by tests and the default
  demo Agent. Defined in core/llm.py.
- DeepSeekChatModel — calls DeepSeek's OpenAI-compatible /chat/completions
  endpoint via httpx. The shape of the request follows the OpenAI chat
  completions spec exactly, with tool_calls support.

A new provider (OpenAI, Anthropic, local llama, ...) plugs in by
implementing ChatModel and registering a provider id with `register_provider`.

Routing rules:
- The AgentConfig stores a `model` string like "mock" or "deepseek:deepseek-chat".
  The format is "<provider>[:<model-name>]".
- `create_chat_model(config, settings)` parses that string and returns the
  matching ChatModel.
- If the requested provider has no key configured, we fall back to
  MockChatModel and log a warning — UNLESS the Settings says
  `deepseek_require_key=True`, in which case we raise.

This is deliberately simple: no streaming, no async batching, no
provider-side retries (the Loop has its own retry/backoff). See
`ADR-005-llm-providers.md` once we write it.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx

from agent_platform.config.settings import Settings
from agent_platform.domain.llm import ChatModel, LLMError, LLMResponse, MockChatModel
from agent_platform.domain.messages import ToolCall

log = logging.getLogger(__name__)


# Statuses worth trying again: the request was fine, the server or the
# network was not. Everything else in the 4xx range is our own fault
# (bad model name, bad arguments, bad key) and will fail identically on
# every retry.
_RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})


def _is_retryable_status(status: int) -> bool:
    return status >= 500 or status in _RETRYABLE_STATUSES


def validate_api_key(api_key: str) -> None:
    """Reject API keys that cannot go into an HTTP header.

    httpx encodes header values as ASCII. A key containing a full-width
    character, a smart quote, or a newline fails deep inside the client
    with `UnicodeEncodeError: ascii codec cannot encode characters in
    position N-M` — which tells you nothing about which credential is bad.
    We check up front so the operator gets a sentence they can act on.

    Raises ValueError with a human-readable reason.
    """
    if not api_key:
        raise ValueError("API key is empty")
    if not api_key.isprintable():
        raise ValueError(
            "API key contains control characters (a newline or tab, most "
            "likely from a multi-line paste). Paste only the key itself."
        )
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as e:
        offending = api_key[e.start : e.end]
        raise ValueError(
            f"API key contains non-ASCII characters at position "
            f"{e.start}-{e.end - 1} ({offending!r}). Keys are ASCII; this is "
            "usually a full-width character (e.g. a full-width colon) or a "
            "smart quote picked up during copy-paste."
        ) from e


class DeepSeekChatModel(ChatModel):
    """ChatModel backed by DeepSeek's OpenAI-compatible API.

    Endpoint: POST {base_url}/chat/completions
    Headers: Authorization: Bearer <api_key>

    Tools are passed in OpenAI function-calling format. The response's
    `tool_calls[i].function.arguments` is a JSON string which we parse
    into a dict before constructing our `ToolCall` model.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        chat_path: str = "/chat/completions",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """All transport parameters are required.

        They come from `Settings` via `create_chat_model`. Keeping them
        required (rather than defaulting here) means there is exactly one
        place to change a model name or endpoint, and a misconfigured
        provider cannot silently fall back to a library default.
        """
        if not api_key:
            raise ValueError("DeepSeekChatModel requires a non-empty api_key")
        validate_api_key(api_key)
        if not base_url:
            raise ValueError("DeepSeekChatModel requires a non-empty base_url")
        if not model:
            raise ValueError("DeepSeekChatModel requires a non-empty model")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._chat_path = chat_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Persistent client. Connection pooling is fine here — DeepSeek's
        # API is stateless across calls.
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if tools:
            # OpenAI function-calling format. Our `tools` list is already
            # in that shape (name / description / parameters).
            payload["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            payload["tool_choice"] = "auto"

        try:
            r = await self._client.post(self._chat_path, json=payload)
        except httpx.TimeoutException as e:
            raise LLMError(
                f"deepseek timeout after {self._timeout}s "
                f"(model={self._model})",
                retryable=True,
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(
                f"deepseek transport error (model={self._model}): {e}",
                retryable=True,
            ) from e

        if r.status_code >= 400:
            # DeepSeek returns a JSON body for both 4xx and 5xx; the message
            # is usually the most useful thing we can show an operator.
            try:
                err = r.json()
                msg = err.get("error", {}).get("message", r.text)
            except Exception:
                msg = r.text
            raise LLMError(
                f"deepseek {r.status_code} (model={self._model}): {msg}",
                retryable=_is_retryable_status(r.status_code),
            )

        data = r.json()
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(
                f"deepseek unexpected response shape (model={self._model}): {data}",
                retryable=False,
            ) from e

        content = msg.get("content") or ""
        tool_calls_out: list[ToolCall] = []
        for raw in msg.get("tool_calls", []) or []:
            fn = raw.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except json.JSONDecodeError:
                log.warning("deepseek.bad_tool_args %s", args_raw)
                args = {}
            tool_calls_out.append(
                ToolCall(
                    id=raw.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        return LLMResponse(content=content, tool_calls=tool_calls_out)


# ---- factory + registry ----------------------------------------------------


_PROVIDERS: dict[str, type[ChatModel]] = {"mock": MockChatModel}


def register_provider(provider_id: str, cls: type[ChatModel]) -> None:
    """Register a ChatModel implementation under a string id.

    Used by tests and by any future provider module. Built-ins are
    pre-registered; this is for extension.
    """
    _PROVIDERS[provider_id] = cls


def available_providers() -> list[str]:
    return sorted(_PROVIDERS.keys())


# Providers whose model ids are themselves prefixed with the provider name.
# DeepSeek names its models `deepseek-flash` and `deepseek-v4-pro`, so the
# bare string `deepseek-flash` is a valid shorthand for the explicitly
# qualified `deepseek:deepseek-flash` — provider `deepseek`, model
# `deepseek-flash` (the WHOLE string).
_SELF_NAMING_PROVIDERS = ("deepseek",)


def parse_model_string(model: str) -> tuple[str, str | None]:
    """Split a model string into ``(provider, model_name_or_None)``.

    Recognized forms, in priority order:

    - ``"provider:model_name"`` → ``("provider", "model_name")``
    - ``"deepseek-flash"`` → ``("deepseek", "deepseek-flash")``
      The provider prefix is detected, but the model name keeps the **full**
      string. DeepSeek's model ids include the ``deepseek-`` prefix;
      stripping it would send the API an unknown name and earn a 400.
    - ``"mock"`` (or any other registered id) → ``("mock", None)``, meaning
      "use the provider's default model".
    """
    if ":" in model:
        provider, name = model.split(":", 1)
        return provider, name
    for prefix in _SELF_NAMING_PROVIDERS:
        if model.startswith(prefix + "-"):
            return prefix, model
    return model, None


def create_chat_model(
    model_string: str,
    settings: Settings,
    *,
    api_key_override: str | None = None,
    require_key: bool | None = None,
) -> ChatModel:
    """Build a ChatModel from an `AgentConfig.model` string.

    `api_key_override` is the resolved DeepSeek key, computed by the
    caller (Runtime/AgentManager) by merging the SecretStore with
    `settings.deepseek_api_key`. This keeps the factory sync — secret
    store lookups happen in the async layer.

    Falls back to MockChatModel (with a warning) if the requested
    provider needs a key that isn't configured. Set
    `require_key=True` to make missing keys fatal instead.
    """
    provider, model_name = parse_model_string(model_string)

    # Unknown provider: explicit failure, no silent fallback. A typo
    # in the config should be loud, not masked.
    if provider not in _PROVIDERS and provider != "deepseek":
        raise ValueError(
            f"unknown LLM provider '{provider}'. "
            f"Available: {', '.join(sorted(set(_PROVIDERS) | {'deepseek'}))}"
        )

    if provider == "deepseek":
        effective_key = api_key_override or settings.deepseek_api_key
        if not effective_key:
            msg = (
                "deepseek requested but no API key is configured. Set "
                "AGENT_PLATFORM_DEEPSEEK_API_KEY env, or store a key via "
                "the Dashboard Providers tab. (Or set "
                "AGENT_PLATFORM_DEEPSEEK_REQUIRE_KEY=false to suppress this; "
                "default is graceful fallback to MockChatModel.)"
            )
            if require_key or settings.deepseek_require_key:
                raise RuntimeError(msg)
            log.warning("provider.fallback %s", msg)
            return MockChatModel()
        return DeepSeekChatModel(
            api_key=effective_key,
            base_url=settings.deepseek_base_url,
            model=model_name or settings.deepseek_model,
            timeout=settings.deepseek_timeout,
            chat_path=settings.deepseek_chat_path,
        )

    # Built-in registered providers (mock, ...).
    cls = _PROVIDERS[provider]
    return cls()
