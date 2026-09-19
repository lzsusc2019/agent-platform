"""Built-in tools + helper to construct the default ToolRegistry.

The MVP ships three tools:
- echo: safe local echo, no side effects
- http_get: safe outbound HTTP (timeout + scheme allow-list from Settings)
- write_file: SENSITIVE — writes inside a workspace root, needs HITL approval

Replace this with an MCP-backed registry in the next iteration.
"""

from __future__ import annotations

from agent_platform.tools.builtins import (
    EchoTool,
    HttpGetTool,
    WriteFileTool,
    build_default_registry,
)

__all__ = ["EchoTool", "HttpGetTool", "WriteFileTool", "build_default_registry"]
