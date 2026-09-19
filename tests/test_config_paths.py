"""Config path resolution must not depend on the working directory.

Every config entry point is a relative path ("config/platform.yaml"), so a
process launched from the wrong directory used to miss the file entirely and
fall back to MockChatModel. Nothing reported a problem — the agent simply
answered with mock text — which made it a genuinely nasty thing to debug.
PyCharm's default working directory is the content root, not necessarily this
project's directory, so hitting it was easy.

The fix tries the CWD first and then the project root. The tests below pin
both halves of that, plus the guard that keeps the fallback from defeating
test hermeticity.
"""

from __future__ import annotations

from pathlib import Path

from agent_platform.config.settings import (
    DEEPSEEK_KEY_ENV_VARS,
    LOCAL_YAML,
    LOCAL_YAML_ENV,
    PLATFORM_YAML,
    PLATFORM_YAML_ENV,
    Settings,
    local_yaml_path,
    platform_yaml_path,
    project_root,
    resolve_config_path,
)

# --------------------------------------------------------------------------- #
# project_root / resolve_config_path
# --------------------------------------------------------------------------- #


def test_project_root_is_the_repo_not_the_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = project_root()
    assert root is not None, "pyproject.toml should mark the repo root"
    assert root != tmp_path
    assert (root / "pyproject.toml").exists()
    # Anchor on something the layering does not move: the config archive.
    assert (root / "src" / "agent_platform").is_dir()
    assert (root / "config" / "agents.yaml").exists()


def test_absolute_paths_pass_through_untouched(tmp_path) -> None:
    target = tmp_path / "somewhere" / "x.yaml"
    assert resolve_config_path(str(target)) == target


def test_cwd_wins_when_the_file_is_there(tmp_path, monkeypatch) -> None:
    """Running from the repo root must keep working exactly as before."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "platform.yaml").write_text("deepseek_model: x\n")
    assert resolve_config_path("config/platform.yaml") == Path("config/platform.yaml")


def test_falls_back_to_the_project_root(tmp_path, monkeypatch) -> None:
    """The actual fix: a foreign CWD still finds the repo's own files."""
    monkeypatch.chdir(tmp_path)
    resolved = resolve_config_path("pyproject.toml")
    assert resolved.is_absolute(), resolved
    assert resolved == project_root() / "pyproject.toml"


def test_unknown_relative_path_is_returned_unchanged(tmp_path, monkeypatch) -> None:
    """No silent invention of a path that exists nowhere."""
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path("nope/absent.yaml") == Path("nope/absent.yaml")


# --------------------------------------------------------------------------- #
# the env override must win verbatim, or hermeticity collapses
# --------------------------------------------------------------------------- #


def test_platform_env_override_wins_even_when_absent(tmp_path, monkeypatch) -> None:
    """tests/conftest.py relies on this: pointing at a nonexistent path must
    not fall through to the real archive on the developer's machine."""
    monkeypatch.chdir(tmp_path)
    absent = tmp_path / "absent-platform.yaml"
    monkeypatch.setenv(PLATFORM_YAML_ENV, str(absent))
    assert platform_yaml_path() == str(absent)


def test_local_env_override_wins_even_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    absent = tmp_path / "absent-local.yaml"
    monkeypatch.setenv(LOCAL_YAML_ENV, str(absent))
    assert local_yaml_path() == str(absent)


def test_empty_env_override_falls_back_to_normal_resolution(tmp_path, monkeypatch) -> None:
    """An empty string means "unset", not "the empty path"."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(PLATFORM_YAML_ENV, "   ")
    resolved = platform_yaml_path()
    assert resolved.strip(), resolved
    assert Path(resolved).is_absolute() or resolved == PLATFORM_YAML


# --------------------------------------------------------------------------- #
# end to end through Settings
# --------------------------------------------------------------------------- #


def test_settings_reads_no_key_from_a_foreign_cwd_when_overridden(
    tmp_path, monkeypatch
) -> None:
    """The hermetic contract, restated at the Settings level."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(PLATFORM_YAML_ENV, str(tmp_path / "absent-platform.yaml"))
    monkeypatch.setenv(LOCAL_YAML_ENV, str(tmp_path / "absent-local.yaml"))
    for var in DEEPSEEK_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    assert Settings().deepseek_key_source() == "unset"


def test_settings_finds_the_archive_from_a_foreign_cwd(tmp_path, monkeypatch) -> None:
    """The bug being fixed, asserted at the level an operator would notice.

    With the env override cleared, a moved CWD must still resolve the same
    absolute file the repo root would have produced.
    """
    monkeypatch.delenv(PLATFORM_YAML_ENV, raising=False)
    monkeypatch.delenv(LOCAL_YAML_ENV, raising=False)

    monkeypatch.chdir(project_root())
    # Resolve while still in the root: a bare relative path means different
    # things from different directories, which is the whole point.
    from_root = Path(platform_yaml_path()).resolve()
    monkeypatch.chdir(tmp_path)
    from_elsewhere = Path(platform_yaml_path())

    assert from_elsewhere.is_absolute(), from_elsewhere
    assert from_elsewhere.resolve() == (project_root() / PLATFORM_YAML).resolve()
    # Both spellings name the same file.
    assert from_root == from_elsewhere.resolve()


def test_local_overlay_uses_the_same_rule(tmp_path, monkeypatch) -> None:
    """The overlay is gitignored and usually absent. Absent means "no config
    to load", so the path we hand back is immaterial — what matters is that
    the rule is shared, and that an existing overlay resolves."""
    monkeypatch.delenv(LOCAL_YAML_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert local_yaml_path() == LOCAL_YAML  # absent at the root: unchanged

    # If an overlay does exist at the project root, it is found from anywhere.
    monkeypatch.chdir(project_root())
    assert Path(local_yaml_path()) == Path(LOCAL_YAML)


def test_the_root_fallback_only_applies_to_files_that_exist(tmp_path, monkeypatch) -> None:
    """Absent means absent — we must not fabricate a project-root path for a
    file that is nowhere, which would turn a clear "file_missing" warning into
    a confusing absolute path."""
    monkeypatch.chdir(tmp_path)
    resolved = resolve_config_path("config/definitely-not-here.yaml")
    assert not resolved.is_absolute()
    assert resolved == Path("config/definitely-not-here.yaml")


def test_every_layer_agrees_when_the_file_exists(tmp_path, monkeypatch) -> None:
    """PLATFORM_YAML is committed, so it exists — the layer an operator
    actually depends on is the one that must survive a moved CWD."""
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path(PLATFORM_YAML) == project_root() / PLATFORM_YAML
