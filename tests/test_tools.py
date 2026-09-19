"""Tool registry tests."""

from __future__ import annotations

from agent_platform.core.tool import Tool, ToolContext, ToolRegistry


def test_register_and_lookup() -> None:
    class _T(Tool):
        name = "x"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def run(self, arguments, ctx):
            return "ok"

    r = ToolRegistry()
    r.register(_T())
    assert r.get("x").name == "x"
    assert "x" in r.names()


def test_sensitive_set(settings) -> None:
    from agent_platform.tools import build_default_registry

    reg = build_default_registry(settings)
    assert "write_file" in reg.sensitive_tools()
    assert "echo" not in reg.sensitive_tools()
    assert "http_get" not in reg.sensitive_tools()


def test_duplicate_register_raises() -> None:
    class _T(Tool):
        name = "dup"
        description = ""
        parameters = {}

        async def run(self, arguments, ctx):
            return ""

    r = ToolRegistry()
    r.register(_T())
    with __import__("pytest").raises(ValueError):
        r.register(_T())


def test_unknown_tool_lookup_raises(settings) -> None:
    from agent_platform.tools import build_default_registry

    reg = build_default_registry(settings)
    with __import__("pytest").raises(KeyError):
        reg.get("nope")
