"""Tool abstraction.

A Tool is anything the LLM can request via ReAct. In production this will
wrap MCP servers; in MVP we ship a local registry with a handful of demo tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class ToolSchema(BaseModel):
    """JSON-Schema-ish description of a tool's arguments.

    Rendered to the LLM as part of its function-calling prompt.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object
    # Sensitive tools must be HITL-approved before they run. Sensitive here
    # means "side-effecty" (e.g. acquire a credential, mutate an external
    # system) — exactly the cases described in Agent中台.md's HITL section.
    sensitive: bool = False


@dataclass
class ToolContext:
    """Per-invocation context passed to a tool."""

    thread_id: str
    user_id: str
    # Pre-approval token, set by the Loop after HITL resolves.
    approval_id: str | None = None


class Tool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    parameters: dict[str, Any]
    sensitive: bool = False

    @abstractmethod
    async def run(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        """Execute the tool and return a string result for the LLM.

        Raise on failure. The Loop catches and converts to a ToolResult with
        is_error=True; the LLM is then asked to replan.
        """

    def approval_scope(self, arguments: dict[str, Any]) -> str:
        """What does one human approval of this call cover?

        The default is the empty string, meaning "this tool, any target".
        A tool whose risk is tied to a specific resource should narrow it —
        otherwise approving one target silently approves every other.

        Must be deterministic for the same logical target: `notes/a.txt`,
        `./notes/a.txt` and `notes/../notes/a.txt` have to collapse to a
        single scope, or the same resource asks for approval twice.
        """
        return ""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            sensitive=self.sensitive,
        )


class ToolRegistry:
    """In-process registry of Tools, keyed by name.

    Replaced by an MCP-backed implementation in the next iteration (see ADR-001).
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as e:
            raise KeyError(f"Unknown tool '{name}'") from e

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def sensitive_tools(self) -> set[str]:
        return {name for name, t in self._tools.items() if t.sensitive}

    def schemas_for_llm(self, only: set[str] | None = None) -> list[dict[str, Any]]:
        """Render tool schemas in OpenAI-function-calling format.

        `only` narrows the result to that set of names; None means every
        registered tool. An agent's `tools` list uses this so a tool it may not
        use never even enters the model's vocabulary.

        This is an offering, not an enforcement: a model that hallucinates or
        gets prompt-injected can still name a tool it was never shown. The Loop
        re-checks the allow-list before executing — see `AgentLoop._execute_tools`.
        """
        return [
            {
                "name": t.schema().name,
                "description": t.schema().description,
                "parameters": t.schema().parameters,
            }
            for name, t in self._tools.items()
            if only is None or name in only
        ]
