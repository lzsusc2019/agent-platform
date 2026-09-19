"""Runtime configuration.

Four layers, later ones override earlier ones:

    1. code defaults              — this file
    2. config/platform.yaml       — the committed, reviewable config archive
    3. config/platform.local.yaml — gitignored per-operator overlay (secrets)
    4. environment                — AGENT_PLATFORM_* (and .env)

Explicit keyword arguments to `Settings(...)` beat all four, which is how
tests pin behaviour.

Nothing in the platform should hardcode a timeout, a TTL, a retry count, a
model name, or an agent id — if you find yourself typing a literal that an
operator might reasonably want to change, add it below (see ADR-005).

Agent *definitions* do not live here; they live in `config/agents.yaml`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Config file locations, relative to the working directory the process was
# started in. Both are overridable so a deployment can keep its config
# outside the source tree (and tests can point at a temp directory).
#
# NOTE: `config/platform.yaml` is committed. This deployment keeps its key
# there; move it to the gitignored local overlay if the repo is shared.
PLATFORM_YAML = "config/platform.yaml"
LOCAL_YAML = "config/platform.local.yaml"
PLATFORM_YAML_ENV = "AGENT_PLATFORM_PLATFORM_YAML_FILE"
LOCAL_YAML_ENV = "AGENT_PLATFORM_LOCAL_YAML_FILE"


def project_root() -> Path | None:
    """The repository root, located from this file rather than the CWD.

    Walks up looking for pyproject.toml rather than counting `..` segments.
    The counting version broke the moment this module moved a layer deeper
    (src/agent_platform/config.py -> src/agent_platform/config/settings.py),
    and a path helper that silently points one directory short is exactly the
    failure mode it exists to prevent.

    Returns None when the package is installed into site-packages, where no
    source tree exists.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def resolve_config_path(path: str) -> Path:
    """Resolve a config path, tolerating a working directory that isn't the root.

    An absolute path is taken as-is. A relative one is tried against the CWD
    first — that is what a CLI user running from the repo root gets — and then
    against the project root.

    That second attempt matters more than it looks. Every entry point here is
    relative ("config/platform.yaml"), so launching from the wrong directory
    used to drop the DeepSeek key silently and fall back to MockChatModel: the
    agent still answers, just with mock text, and nothing reports a problem.
    IDEs make this easy to hit — PyCharm's default working directory is the
    content root, which is not necessarily this project's directory.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    root = project_root()
    if root is not None and (root / candidate).exists():
        return root / candidate
    return candidate


def _yaml_layer_path(relative: str, env_var: str) -> str:
    """Path for one YAML layer, honouring an explicit env override verbatim.

    An override wins even when the file does not exist. The test suite points
    these at a nonexistent path to stay hermetic, and that must never fall
    through to a real file on the developer's machine.
    """
    override = os.environ.get(env_var, "").strip()
    if override:
        return override
    return str(resolve_config_path(relative))


def platform_yaml_path() -> str:
    """Where the committed config archive lives, allowing an env override."""
    return _yaml_layer_path(PLATFORM_YAML, PLATFORM_YAML_ENV)


def local_yaml_path() -> str:
    """Where the gitignored per-operator overlay lives."""
    return _yaml_layer_path(LOCAL_YAML, LOCAL_YAML_ENV)

# Everything a DeepSeek key might arrive as through the environment. The first
# is canonical; the others are conveniences so an operator who already exports
# DEEPSEEK_API_KEY (the name the official SDK uses) need not duplicate it.
DEEPSEEK_KEY_ENV_VARS = (
    "AGENT_PLATFORM_DEEPSEEK_API_KEY",
    "AGENT_PLATFORM_DEEPSEEK_APIKEY",
    "DEEPSEEK_API_KEY",
)

# Spellings accepted for the key when it comes from YAML or kwargs.
_KEY_ALIASES = ("deepseek_apikey", "deepseek-api-key", "deepseekApiKey")


def _yaml_defines_key(path: str) -> bool:
    """Does the top level of this YAML file set a DeepSeek key?

    Used only for reporting. A missing or malformed file is treated as a
    negative rather than raising: this must never be the reason startup
    fails.
    """
    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    if not target.exists():
        return False
    try:
        import yaml

        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if str(data.get("deepseek_api_key", "")).strip():
        return True
    return any(str(data.get(a, "")).strip() for a in _KEY_ALIASES)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_PLATFORM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Source wiring
    # ------------------------------------------------------------------

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """kwargs > env > .env > platform.local.yaml > platform.yaml > defaults.

        The local overlay sits above the committed archive so an operator can
        keep a secret in `config/platform.local.yaml` (gitignored) without
        ever touching the file that ships in the repo.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=local_yaml_path()),
            YamlConfigSettingsSource(settings_cls, yaml_file=platform_yaml_path()),
            file_secret_settings,
        )

    @model_validator(mode="before")
    @classmethod
    def _accept_key_spellings(cls, data: Any) -> Any:
        """Let `deepseek_apikey` and friends stand in for `deepseek_api_key`.

        Covers YAML and keyword arguments. Environment variables are handled
        in `_apply_env_key_fallbacks`, because declaring a `validation_alias`
        would bypass the `AGENT_PLATFORM_` prefix the canonical name relies on.
        """
        if not isinstance(data, dict):
            return data
        if data.get("deepseek_api_key"):
            return data
        for alias in _KEY_ALIASES:
            value = data.get(alias)
            if value:
                return {**data, "deepseek_api_key": value}
        return data

    @model_validator(mode="after")
    def _apply_env_key_fallbacks(self) -> Settings:
        """Honour alias spellings and the official SDK env var name."""
        if self.deepseek_api_key:
            stripped = self.deepseek_api_key.strip()
            if stripped != self.deepseek_api_key:
                object.__setattr__(self, "deepseek_api_key", stripped)
            return self
        for var in DEEPSEEK_KEY_ENV_VARS:
            raw = os.environ.get(var)
            if raw and raw.strip():
                object.__setattr__(self, "deepseek_api_key", raw.strip())
                break
        return self

    # ==================================================================
    # Storage
    # ==================================================================
    redis_url: str = "redis://localhost:6379/0"
    # Dev-only convenience: swap in an in-memory fakeredis. Never enable
    # in production — data is lost on restart and not shared across
    # processes. Runtime logs a loud warning when this is on.
    use_fake_redis: bool = False
    # How long a Checkpoint snapshot survives without writes. Per ADR-003
    # this has to outlive the longest realistic human approval.
    checkpoint_ttl_seconds: int = 30 * 60

    # How long a human approval stays valid. An approval is scoped to one
    # target — for write_file, one resolved path — so this reads "you may keep
    # editing THIS file for this long", not "you may keep writing files".
    #
    # Independent of checkpoint_ttl_seconds: that one bounds how long a parked
    # thread survives waiting for an answer, this one bounds how long the
    # answer keeps working once given.
    approval_grant_ttl_seconds: int = 3600

    # ==================================================================
    # Agent Loop
    # ==================================================================
    # Hard iteration cap per conversation turn, per ADR-001 / Agent中台.md.
    max_turns: int = 100
    # Transient-error retries around a single LLM call (exponential backoff).
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 0.5  # seconds
    # Consecutive empty responses tolerated before giving up on the turn.
    empty_response_max_retries: int = 3

    # ==================================================================
    # Context compression
    # ==================================================================
    # Tokens are estimated with a cheap chars/N heuristic — not an exact
    # tokenizer. Good enough for a fuzzy trigger threshold.
    token_estimate_chars_per_token: int = 4
    # ~70% of a 260k-token window, leaving room for the reply.
    compress_trigger_tokens: int = 180_000
    compress_keep_recent_turns: int = 30

    # ==================================================================
    # Agent defaults
    #
    # Applied when an AgentConfig omits a field, and when the Admin API
    # creates a brand-new agent. Entries in config/agents.yaml or in the
    # runtime store always win.
    # ==================================================================
    agent_default_system_prompt: str = (
        "You are a helpful assistant. Use tools when needed."
    )
    agent_default_model: str = "mock"
    agent_default_temperature: float = 0.7
    agent_default_max_tokens: int = 4096

    # ==================================================================
    # Agent seed
    #
    # Agent definitions are archived in YAML, not in this file. On startup
    # every agent in that file is written to the runtime store if (and only
    # if) its agent_id is not already present, so Dashboard edits are never
    # clobbered by a restart. Point at a different file, or set it to the
    # empty string to disable seeding entirely.
    # ==================================================================
    agent_seed_file: str = "config/agents.yaml"


    # --- DeepSeek ------------------------------------------------------
    # DeepSeek exposes an OpenAI-compatible API: POST {base_url}{chat_path}
    # with an `Authorization: Bearer` header. Without a key the factory falls
    # back to MockChatModel unless deepseek_require_key is set.
    #
    # Supply the key from any of these, highest precedence first:
    #   AGENT_PLATFORM_DEEPSEEK_API_KEY env var
    #   config/platform.local.yaml   (gitignored — per-machine override)
    #   config/platform.yaml         (the deployment archive)
    # The alias `deepseek_apikey` is accepted in YAML, and the environment
    # variable `DEEPSEEK_API_KEY` is read as a fallback.
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_timeout: float = 30.0  # seconds per request
    deepseek_require_key: bool = False
    # Path appended to base_url for chat completions. Overridable so a
    # gateway or proxy with a different route can be dropped in.
    deepseek_chat_path: str = "/chat/completions"
    # Path used by the Dashboard "test" button to validate a key.
    deepseek_models_path: str = "/models"

    # ==================================================================
    # Tools
    # ==================================================================
    # http_get
    tool_http_get_timeout: float = 5.0
    tool_http_get_max_body_chars: int = 500
    tool_http_get_allowed_schemes: list[str] = ["http://", "https://"]
    # write_file — the sensitive (HITL-gated) demo tool. Every write is
    # confined to this root; the check resolves symlinks and `..` before
    # comparing, so a path cannot escape it.
    tool_write_file_root: str = "workspace"
    tool_write_file_max_bytes: int = 65_536

    # ==================================================================
    # Admin / Dashboard
    # ==================================================================
    # Timeout for the provider-key connectivity check.
    admin_secret_test_timeout: float = 10.0


    # ==================================================================
    # Observability
    # ==================================================================
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def deepseek_key_source(self) -> str:
        """Best-effort label for which layer supplied the DeepSeek key.

        Printed once at startup so "did my key get injected?" is answerable
        from the log without grepping files or the environment.
        """
        if not self.deepseek_api_key:
            return "unset"
        for var in DEEPSEEK_KEY_ENV_VARS:
            if os.environ.get(var, "").strip():
                return "env:" + var
        if _yaml_defines_key(local_yaml_path()):
            return local_yaml_path()
        if _yaml_defines_key(platform_yaml_path()):
            return platform_yaml_path()
        return "unknown"

    def credential_warnings(self) -> list[str]:
        """Things worth shouting about at startup."""
        out: list[str] = []
        if self.use_fake_redis:
            out.append(
                "use_fake_redis=true: data is in-memory only, lost on "
                "restart. Do NOT use this in production."
            )
        return out

    def redacted(self) -> dict[str, Any]:
        """Dump for display/logging with credentials masked."""
        dumped = self.model_dump()
        for key in list(dumped):
            is_credential = (
                key.endswith("_api_key")
                or key.endswith("_password")
                or key.endswith("_secret")
                or key in {"api_key", "password", "secret", "token"}
            )
            if is_credential:
                dumped[key] = "<set>" if dumped[key] else "<unset>"
        return dumped
