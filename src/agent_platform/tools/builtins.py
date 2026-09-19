"""Default tool implementations for the MVP.

Anything an operator might want to tune (timeouts, which URL schemes are
allowed, how much of a response body is fed back to the model, where file
writes are allowed to land) is read from `Settings` rather than baked into
the class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import httpx

from agent_platform.config.settings import Settings
from agent_platform.domain.tool import Tool, ToolContext, ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the supplied text. Useful for verifying the tool-call round-trip."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo back."}},
        "required": ["text"],
    }
    sensitive = False

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        return f"echo: {arguments.get('text', '')}"


class HttpGetTool(Tool):
    """Fetch a URL. Refuses anything outside the configured scheme allow-list.

    The description is rebuilt from settings so the LLM sees the real
    timeout rather than a stale hardcoded claim.
    """

    name = "http_get"
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."}
        },
        "required": ["url"],
    }
    sensitive = False

    def __init__(self, settings: Settings) -> None:
        self._timeout = settings.tool_http_get_timeout
        self._max_body_chars = settings.tool_http_get_max_body_chars
        self._allowed_schemes = tuple(settings.tool_http_get_allowed_schemes)
        self.description = (
            f"Fetch the body of a public URL. Times out after {self._timeout:g} seconds."
        )

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        url = arguments.get("url", "")
        if not url.startswith(self._allowed_schemes):
            allowed = ", ".join(self._allowed_schemes)
            return f"error: refusing to fetch '{url}' (allowed schemes: {allowed})"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(url)
            return f"status={r.status_code} body={r.text[: self._max_body_chars]}"


class WriteFileTool(Tool):
    """Write a file inside a configured workspace root. REQUIRES APPROVAL.

    This is the platform's example of a *side-effecting* tool — the kind
    that must not run without a human saying yes. Agent中台.md makes the same
    point with a credential-fetching tool; a file write is the same shape
    (irreversible, touches state outside the conversation) without needing a
    credential in the demo.

    Three independent guards stand between the model and the filesystem:

    1. `sensitive = True`, so the Loop raises HITLInterrupt before calling.
    2. An `approval_id` check here, so a misconfigured Loop fails loudly
       instead of silently writing.
    3. Path confinement to `tool_write_file_root`, so even an approved call
       cannot escape the workspace with `../` or an absolute path.
    """

    name = "write_file"
    description = (
        "Write text to a file inside the workspace. REQUIRES HUMAN APPROVAL "
        "before it runs."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path relative to the workspace root, e.g. 'notes/draft.md'."
                ),
            },
            "content": {"type": "string", "description": "Text to write."},
        },
        "required": ["path", "content"],
    }
    sensitive = True

    def __init__(self, settings: Settings) -> None:
        self._root = Path(settings.tool_write_file_root)
        self._max_bytes = settings.tool_write_file_max_bytes

    def approval_scope(self, arguments: dict[str, Any]) -> str:
        """The resolved target path: one approval covers one file.

        Resolving here rather than hashing the raw argument is what lets
        `notes/a.txt` and `./notes/../notes/a.txt` share a grant instead of
        each demanding its own approval.

        A path that cannot be resolved returns "" and so never matches a grant
        earned by a legitimate write — a call that is going to be refused
        anyway must not inherit someone else's permission.
        """
        target, _ = self._resolve(str(arguments.get("path", "")))
        return "" if target is None else str(target)

    def _resolve(self, raw_path: str) -> tuple[Path | None, str | None]:
        """Resolve `raw_path` under the workspace root, or explain why not."""
        if not raw_path.strip():
            return None, "error: 'path' is empty"
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return None, f"error: refusing absolute path '{raw_path}'"
        root = self._root.resolve()
        target = (root / candidate).resolve()
        # `resolve()` collapses `..`, and symlinks, so this comparison is the
        # real containment check — a string prefix test would not be.
        if target != root and root not in target.parents:
            return None, (
                f"error: refusing path '{raw_path}' — it escapes the "
                f"workspace root ({root})"
            )
        return target, None

    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        if ctx.approval_id is None:
            # Should never be reached: the Loop raises HITLInterrupt before
            # calling a sensitive tool. Kept so a misconfigured Agent fails
            # loud rather than writing unapproved files.
            raise RuntimeError(
                "WriteFileTool called without approval_id — the Loop should "
                "have raised HITLInterrupt first."
            )

        raw_path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))

        if len(content.encode("utf-8")) > self._max_bytes:
            return (
                f"error: content is {len(content.encode('utf-8'))} bytes, "
                f"over the {self._max_bytes}-byte limit"
            )

        target, problem = self._resolve(raw_path)
        if problem is not None:
            return problem
        assert target is not None  # for type checkers

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return json.dumps(
            {
                "path": str(target),
                "bytes_written": len(content.encode("utf-8")),
                "approved_by": ctx.approval_id,
            },
            ensure_ascii=False,
        )


def build_default_registry(settings: Settings) -> ToolRegistry:
    """Construct the built-in tool set for the given settings."""
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(HttpGetTool(settings))
    reg.register(WriteFileTool(settings))
    return reg
