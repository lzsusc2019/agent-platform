"""DeepSeek API key injection at startup.

The key can arrive from four places; these tests pin the precedence and
confirm the startup report names the source without leaking the value.
"""

from __future__ import annotations

import pytest

from agent_platform.config.settings import (
    DEEPSEEK_KEY_ENV_VARS,
    Settings,
)

KEY_ENV_VARS = (*DEEPSEEK_KEY_ENV_VARS, "AGENT_PLATFORM_PLATFORM_YAML_FILE",
                "AGENT_PLATFORM_LOCAL_YAML_FILE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No ambient key or path overrides — each test states its own."""
    for var in KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_PLATFORM_PLATFORM_YAML_FILE", "/nonexistent/p.yaml")
    monkeypatch.setenv("AGENT_PLATFORM_LOCAL_YAML_FILE", "/nonexistent/l.yaml")
    yield


def _layer(tmp_path, platform: str = "", local: str = "") -> None:
    import os

    p = tmp_path / "platform.yaml"
    p.write_text(platform, encoding="utf-8")
    local_path = tmp_path / "platform.local.yaml"
    local_path.write_text(local, encoding="utf-8")
    os.environ["AGENT_PLATFORM_PLATFORM_YAML_FILE"] = str(p)
    os.environ["AGENT_PLATFORM_LOCAL_YAML_FILE"] = str(local_path)


# ----- the canonical field name -------------------------------------------


def test_key_from_platform_yaml(tmp_path) -> None:
    _layer(tmp_path, platform="deepseek_api_key: sk-from-platform\n")
    assert Settings().deepseek_api_key == "sk-from-platform"


def test_key_from_local_overlay(tmp_path) -> None:
    _layer(tmp_path, local="deepseek_api_key: sk-from-local\n")
    assert Settings().deepseek_api_key == "sk-from-local"


def test_local_overlay_key_beats_platform(tmp_path) -> None:
    _layer(
        tmp_path,
        platform="deepseek_api_key: sk-platform\n",
        local="deepseek_api_key: sk-local\n",
    )
    assert Settings().deepseek_api_key == "sk-local"


def test_env_beats_yaml_layers(tmp_path, monkeypatch) -> None:
    _layer(tmp_path, local="deepseek_api_key: sk-local\n")
    monkeypatch.setenv("AGENT_PLATFORM_DEEPSEEK_API_KEY", "sk-env")
    assert Settings().deepseek_api_key == "sk-env"


def test_kwarg_beats_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PLATFORM_DEEPSEEK_API_KEY", "sk-env")
    assert Settings(deepseek_api_key="sk-kwarg").deepseek_api_key == "sk-kwarg"


# ----- alias spellings ------------------------------------------------------


def test_yaml_alias_deepseek_apikey(tmp_path) -> None:
    """The spelling from the ops request works in YAML."""
    _layer(tmp_path, local="deepseek_apikey: sk-alias\n")
    assert Settings().deepseek_api_key == "sk-alias"


def test_kwarg_alias_deepseek_apikey() -> None:
    assert Settings(deepseek_apikey="sk-alias").deepseek_api_key == "sk-alias"


def test_canonical_spelling_wins_over_alias(tmp_path) -> None:
    _layer(
        tmp_path,
        local="deepseek_api_key: sk-canonical\ndeepseek_apikey: sk-alias\n",
    )
    assert Settings().deepseek_api_key == "sk-canonical"


def test_env_alias_spelling(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PLATFORM_DEEPSEEK_APIKEY", "sk-env-alias")
    assert Settings().deepseek_api_key == "sk-env-alias"


def test_sdk_env_var_is_accepted(monkeypatch) -> None:
    """An operator who already exports DEEPSEEK_API_KEY need not duplicate it."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-sdk-name")
    assert Settings().deepseek_api_key == "sk-sdk-name"


def test_canonical_env_wins_over_sdk_env(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-sdk-name")
    monkeypatch.setenv("AGENT_PLATFORM_DEEPSEEK_API_KEY", "sk-canonical")
    assert Settings().deepseek_api_key == "sk-canonical"


# ----- hygiene --------------------------------------------------------------


def test_surrounding_whitespace_is_trimmed() -> None:
    assert Settings(deepseek_api_key="  sk-padded  ").deepseek_api_key == "sk-padded"


def test_yaml_whitespace_is_trimmed(tmp_path) -> None:
    _layer(tmp_path, local='deepseek_api_key: "sk-padded   "\n')
    assert Settings().deepseek_api_key == "sk-padded"


def test_unset_by_default() -> None:
    assert Settings().deepseek_api_key == ""
    assert Settings().deepseek_key_source() == "unset"


# ----- the startup report ---------------------------------------------------


def test_key_source_names_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_PLATFORM_DEEPSEEK_API_KEY", "sk-env")
    assert Settings().deepseek_key_source() == (
        "env:AGENT_PLATFORM_DEEPSEEK_API_KEY"
    )


def test_key_source_names_local_overlay(tmp_path) -> None:
    _layer(tmp_path, local="deepseek_api_key: sk-local\n")
    assert Settings().deepseek_key_source().endswith("platform.local.yaml")


def test_key_source_names_platform_archive(tmp_path) -> None:
    _layer(tmp_path, platform="deepseek_api_key: sk-platform\n")
    assert Settings().deepseek_key_source().endswith("platform.yaml")
    assert "local" not in Settings().deepseek_key_source()


def test_platform_yaml_key_does_not_warn(tmp_path) -> None:
    """`config/platform.yaml` is the intended home for the key in this
    deployment, so loading one from there is not worth a warning."""
    _layer(tmp_path, platform="deepseek_api_key: sk-in-archive\n")
    assert Settings().credential_warnings() == []


def test_local_overlay_key_does_not_warn(tmp_path) -> None:
    _layer(tmp_path, local="deepseek_api_key: sk-fine\n")
    assert Settings().credential_warnings() == []


def test_use_fake_redis_warns() -> None:
    assert any(
        "use_fake_redis" in w for w in Settings(use_fake_redis=True).credential_warnings()
    )


# ----- the shipped archive parses and carries the demo agent ---------------


def test_shipped_platform_yaml_parses() -> None:
    from pathlib import Path

    import yaml

    from agent_platform.config.settings import PLATFORM_YAML

    # Use the module constant, not the helper: the autouse fixture redirects
    # the helper at a temp file.
    path = Path(__file__).resolve().parent.parent / PLATFORM_YAML
    assert path.exists(), "config/platform.yaml must ship in the repo"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(data, dict) and data, "config/platform.yaml is empty"
