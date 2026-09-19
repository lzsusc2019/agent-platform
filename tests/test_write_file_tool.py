"""WriteFileTool — the approval-gated, workspace-confined demo tool."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_platform.config import Settings
from agent_platform.core.tool import ToolContext
from agent_platform.tools import build_default_registry
from agent_platform.tools.builtins import WriteFileTool


def _settings(root: Path, **kw) -> Settings:
    return Settings(tool_write_file_root=str(root), **kw)


def _ctx(approved: bool = True) -> ToolContext:
    return ToolContext(
        thread_id="t", user_id="u", approval_id="appr_x" if approved else None
    )


def _run(tool: WriteFileTool, args: dict, ctx: ToolContext) -> str:
    return asyncio.run(tool.run(args, ctx))


# ----- sensitivity ----------------------------------------------------------


def test_write_file_is_marked_sensitive(tmp_path) -> None:
    reg = build_default_registry(_settings(tmp_path))
    assert "write_file" in reg.sensitive_tools()
    assert "echo" not in reg.sensitive_tools()
    assert "http_get" not in reg.sensitive_tools()


def test_write_file_schema_tells_the_model_it_needs_approval(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path))
    assert "APPROVAL" in tool.description.upper()


def test_refuses_to_run_without_an_approval_id(tmp_path) -> None:
    """Defence in depth: a misconfigured Loop must fail loud, not write."""
    tool = WriteFileTool(_settings(tmp_path))
    with pytest.raises(RuntimeError, match="approval_id"):
        _run(tool, {"path": "a.txt", "content": "x"}, _ctx(approved=False))
    assert not (tmp_path / "a.txt").exists()


# ----- the happy path -------------------------------------------------------


def test_writes_a_file_inside_the_root(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path))
    out = _run(tool, {"path": "notes/a.txt", "content": "hello"}, _ctx())
    written = tmp_path / "notes" / "a.txt"
    assert written.read_text(encoding="utf-8") == "hello"
    payload = json.loads(out)
    assert payload["bytes_written"] == 5
    assert payload["approved_by"] == "appr_x"


def test_creates_missing_parent_directories(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path))
    _run(tool, {"path": "a/b/c/d.txt", "content": "deep"}, _ctx())
    assert (tmp_path / "a" / "b" / "c" / "d.txt").exists()


def test_overwrites_an_existing_file(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path))
    _run(tool, {"path": "a.txt", "content": "first"}, _ctx())
    _run(tool, {"path": "a.txt", "content": "second"}, _ctx())
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "second"


def test_non_ascii_content_round_trips(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path))
    _run(tool, {"path": "cn.txt", "content": "东莞天气-晴"}, _ctx())
    assert (tmp_path / "cn.txt").read_text(encoding="utf-8") == "东莞天气-晴"


# ----- confinement ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.txt",
        "../../escape.txt",
        "notes/../../escape.txt",
        "/etc/passwd",
        "/tmp/absolute.txt",
    ],
)
def test_refuses_paths_outside_the_root(tmp_path, bad_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    tool = WriteFileTool(_settings(root))
    out = _run(tool, {"path": bad_path, "content": "nope"}, _ctx())
    assert out.startswith("error:"), out
    # Nothing escaped.
    assert not (tmp_path / "escape.txt").exists()


def test_refuses_a_path_that_merely_shares_a_prefix(tmp_path) -> None:
    """`/tmp/root-evil` starts with `/tmp/root` as a string but is outside it.

    A naive `str.startswith` check would let this through; we compare
    resolved paths instead.
    """
    root = tmp_path / "root"
    root.mkdir()
    tool = WriteFileTool(_settings(root))
    out = _run(tool, {"path": "../root-evil/a.txt", "content": "x"}, _ctx())
    assert out.startswith("error:"), out
    assert not (tmp_path / "root-evil").exists()


def test_empty_path_is_rejected(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path))
    assert _run(tool, {"path": "  ", "content": "x"}, _ctx()).startswith("error:")


def test_writing_into_the_root_itself_is_allowed(tmp_path) -> None:
    """The root is inside the root — a legitimate top-level file."""
    tool = WriteFileTool(_settings(tmp_path))
    out = _run(tool, {"path": "top.txt", "content": "x"}, _ctx())
    assert not out.startswith("error:"), out
    assert (tmp_path / "top.txt").exists()


# ----- size limit -----------------------------------------------------------


def test_refuses_content_over_the_byte_limit(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path, tool_write_file_max_bytes=10))
    out = _run(tool, {"path": "big.txt", "content": "x" * 11}, _ctx())
    assert "over the 10-byte limit" in out
    assert not (tmp_path / "big.txt").exists()


def test_accepts_content_exactly_at_the_limit(tmp_path) -> None:
    tool = WriteFileTool(_settings(tmp_path, tool_write_file_max_bytes=10))
    out = _run(tool, {"path": "ok.txt", "content": "x" * 10}, _ctx())
    assert not out.startswith("error:"), out


def test_byte_limit_counts_utf8_bytes_not_characters(tmp_path) -> None:
    """Chinese is 3 bytes per character — a char count would be wrong."""
    tool = WriteFileTool(_settings(tmp_path, tool_write_file_max_bytes=8))
    # 3 chars * 3 bytes = 9 bytes > 8
    out = _run(tool, {"path": "cn.txt", "content": "东莞天"}, _ctx())
    assert "over the 8-byte limit" in out


# ----- registry wiring ------------------------------------------------------


def test_registry_exposes_the_three_builtin_tools(tmp_path) -> None:
    reg = build_default_registry(_settings(tmp_path))
    assert set(reg.names()) == {"echo", "http_get", "write_file"}
