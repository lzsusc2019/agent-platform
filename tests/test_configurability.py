"""Every tunable must actually take effect.

Extracting a literal into `Settings` is only half the job — the value has
to reach the code path that uses it. These tests assert the wiring, one
behaviour per setting, so a future refactor that reintroduces a hardcoded
default fails here rather than in production.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_platform.config.settings import Settings
from agent_platform.domain.agent_loop import _estimate_tokens, _maybe_compress
from agent_platform.domain.messages import Message, MessageRole
from agent_platform.domain.tool import ToolContext
from agent_platform.infra.checkpoint_store import CheckpointStore, snapshot_key
from agent_platform.infra.config_store import AgentConfig
from agent_platform.infra.providers import create_chat_model, validate_api_key
from agent_platform.tools import build_default_registry
from agent_platform.tools.builtins import HttpGetTool

# ----- layingering: kwargs > env > .env > platform.yaml > code default -----


def _point_at(tmp_path, platform: str = "", local: str = "") -> None:
    """Redirect the config layers at temp files via the path env overrides."""
    import os

    p = tmp_path / "platform.yaml"
    p.write_text(platform, encoding="utf-8")
    local_path = tmp_path / "platform.local.yaml"
    local_path.write_text(local, encoding="utf-8")
    os.environ["AGENT_PLATFORM_PLATFORM_YAML_FILE"] = str(p)
    os.environ["AGENT_PLATFORM_LOCAL_YAML_FILE"] = str(local_path)


@pytest.fixture(autouse=True)
def _isolate_config_env(monkeypatch):
    """Keep the developer's own env from leaking into these assertions."""
    for var in (
        "AGENT_PLATFORM_PLATFORM_YAML_FILE",
        "AGENT_PLATFORM_LOCAL_YAML_FILE",
        "AGENT_PLATFORM_MAX_TURNS",
        "AGENT_PLATFORM_DEEPSEEK_API_KEY",
        "AGENT_PLATFORM_DEEPSEEK_APIKEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def test_yaml_file_is_loaded(tmp_path, monkeypatch) -> None:
    """A value present in platform.yaml overrides the code default."""
    _point_at(tmp_path, platform="max_turns: 7\n")
    assert Settings().max_turns == 7


def test_local_overlay_beats_platform_yaml(tmp_path, monkeypatch) -> None:
    _point_at(tmp_path, platform="max_turns: 7\n", local="max_turns: 11\n")
    assert Settings().max_turns == 11


def test_env_beats_both_yaml_layers(tmp_path, monkeypatch) -> None:
    _point_at(tmp_path, platform="max_turns: 7\n", local="max_turns: 11\n")
    monkeypatch.setenv("AGENT_PLATFORM_MAX_TURNS", "13")
    assert Settings().max_turns == 13, "env must win over both yaml layers"


def test_kwargs_beat_everything(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PLATFORM_MAX_TURNS", "9")
    assert Settings(max_turns=11).max_turns == 11


def test_missing_yaml_files_are_not_fatal(tmp_path, monkeypatch) -> None:
    import os

    os.environ["AGENT_PLATFORM_PLATFORM_YAML_FILE"] = str(tmp_path / "nope.yaml")
    os.environ["AGENT_PLATFORM_LOCAL_YAML_FILE"] = str(tmp_path / "also-nope.yaml")
    # Falls back to the code default rather than raising.
    assert Settings().max_turns == 100


def test_shipped_platform_yaml_matches_the_documented_defaults() -> None:
    """The local config archive must not silently drift.

    If someone changes a code default, this fails until they either update
    config/platform.yaml or drop the key from it (letting it track the
    default). Either is fine; an unnoticed divergence is not.

    config/platform.yaml is **not version-controlled** — it holds the
    operator's credentials, and this repository is public. So a fresh clone
    legitimately has no such file, and the check is skipped rather than
    failing: there is nothing to compare against, and it is not a mistake.

    The consequence, worth stating plainly: the drift guarantee only holds on
    a machine that has the file. A code default can still move without anyone
    noticing. Keep this file in your own checkout to keep the check alive.
    """
    from pathlib import Path

    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "config" / "platform.yaml"
    if not cfg_path.exists():
        pytest.skip(
            "config/platform.yaml is operator-local (holds credentials, not "
            "tracked); nothing to compare against"
        )

    archived = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    assert archived, "config/platform.yaml is empty"

    defaults = Settings.model_construct()  # field defaults, no sources
    for key, value in archived.items():
        assert key in Settings.model_fields, f"{key} is not a Settings field"
        # Credentials are deployment-specific, not tunables; they legitimately
        # differ from the empty code default.
        if "api_key" in key or "apikey" in key:
            continue
        default = defaults.__dict__.get(key)
        assert value == default, (
            f"config/platform.yaml pins {key}={value!r} but the code default "
            f"is {default!r}; update one of them"
        )


# ----- credential redaction ------------------------------------------------


def test_redacted_masks_credentials_only() -> None:
    s = Settings(deepseek_api_key="sk-secret")
    out = s.redacted()
    assert out["deepseek_api_key"] == "<set>"
    # Numeric knobs whose names merely contain "token" must survive.
    assert out["compress_trigger_tokens"] == s.compress_trigger_tokens
    assert out["token_estimate_chars_per_token"] == s.token_estimate_chars_per_token


# ----- API key validation --------------------------------------------------


def test_validate_api_key_accepts_a_normal_key() -> None:
    validate_api_key("sk-abcdef1234567890")


def test_validate_api_key_rejects_non_ascii() -> None:
    with pytest.raises(ValueError, match="non-ASCII"):
        validate_api_key("sk-你的密钥")


def test_validate_api_key_rejects_full_width_colon() -> None:
    """The realistic paste accident: a full-width colon in the key."""
    with pytest.raises(ValueError, match="non-ASCII"):
        validate_api_key("sk-abc\uff1adef")


def test_validate_api_key_rejects_embedded_newline() -> None:
    with pytest.raises(ValueError, match="control characters"):
        validate_api_key("sk-abc\ndef")


def test_validate_api_key_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_api_key("")


def test_bad_key_fails_at_construction_not_at_request_time() -> None:
    """The whole point: a clear error where the key is set, not a
    UnicodeEncodeError from inside httpx later."""
    s = Settings(deepseek_api_key="sk-中文")
    with pytest.raises(ValueError, match="non-ASCII"):
        create_chat_model("deepseek-flash", s)


# ----- agent defaults -------------------------------------------------------


def test_agent_config_defaults_come_from_settings() -> None:
    s = Settings(
        agent_default_system_prompt="Platform prompt.",
        agent_default_model="mock",
        agent_default_temperature=0.25,
        agent_default_max_tokens=123,
    )
    cfg = AgentConfig.with_defaults("anything", s)
    assert cfg.system_prompt == "Platform prompt."
    assert cfg.model == "mock"
    assert cfg.temperature == 0.25
    assert cfg.max_tokens == 123


def test_with_defaults_lets_overrides_win() -> None:
    cfg = AgentConfig.with_defaults("x", Settings(), model="mock", temperature=0.9)
    assert cfg.model == "mock"
    assert cfg.temperature == 0.9


# ----- checkpoint TTL -------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_ttl_reaches_redis() -> None:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    redis = fakeredis.aioredis.FakeRedis()
    store = CheckpointStore(redis, ttl_seconds=4242)
    await store.save(store.new_snapshot("t", messages=[]))
    ttl = await redis.ttl(snapshot_key("t"))
    assert 0 < ttl <= 4242
    assert ttl > 4000, f"TTL {ttl} does not reflect the configured 4242"
    await redis.aclose()


# ----- tools ----------------------------------------------------------------


def test_http_get_scheme_allow_list_is_enforced() -> None:
    tool = HttpGetTool(Settings(tool_http_get_allowed_schemes=["https://"]))
    out = asyncio.run(
        tool.run({"url": "http://example.com"}, ToolContext(thread_id="t", user_id="u"))
    )
    assert "refusing" in out
    assert "https://" in out


def test_http_get_description_reflects_configured_timeout() -> None:
    tool = HttpGetTool(Settings(tool_http_get_timeout=12.5))
    assert "12.5 seconds" in tool.description


def test_registry_is_built_from_the_supplied_settings() -> None:
    reg = build_default_registry(Settings(tool_http_get_timeout=7.0))
    assert "7 seconds" in reg.get("http_get").description


# ----- context compression --------------------------------------------------


def test_token_estimate_honours_chars_per_token() -> None:
    msgs = [Message(role=MessageRole.USER, content="x" * 100)]
    assert _estimate_tokens(msgs, 4) == 25
    assert _estimate_tokens(msgs, 2) == 50
    # A zero/negative ratio must not divide by zero.
    assert _estimate_tokens(msgs, 0) == 100


def test_compression_trigger_scales_with_the_token_ratio() -> None:
    msgs = [
        Message(role=MessageRole.SYSTEM, content="sys"),
        Message(role=MessageRole.USER, content="u" * 400),
    ]
    _, did_default = _maybe_compress(msgs, 500, 1, chars_per_token=4)
    assert did_default is False
    _, did_coarse = _maybe_compress(msgs, 500, 1, chars_per_token=2)
    assert did_coarse is False
    _, did_small = _maybe_compress(msgs, 150, 1, chars_per_token=2)
    assert did_small is True


# ----- provider wiring ------------------------------------------------------


def test_deepseek_chat_path_is_used() -> None:
    s = Settings(
        deepseek_api_key="sk-x",
        deepseek_chat_path="/v1/gateway/chat",
        deepseek_model="deepseek-flash",
    )
    assert create_chat_model("deepseek", s)._chat_path == "/v1/gateway/chat"


def test_deepseek_model_setting_is_the_fallback_name() -> None:
    s = Settings(deepseek_api_key="sk-x", deepseek_model="custom-model")
    assert create_chat_model("deepseek", s)._model == "custom-model"


def test_explicit_model_suffix_beats_the_setting() -> None:
    s = Settings(deepseek_api_key="sk-x", deepseek_model="deepseek-flash")
    assert create_chat_model("deepseek:deepseek-v4-pro", s)._model == "deepseek-v4-pro"
