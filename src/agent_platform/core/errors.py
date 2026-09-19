"""Loop-level exceptions.

These are domain errors, not transport errors. They signal that the Loop has
made a control-flow decision (pause, give up, refuse) rather than crashed.
"""

from __future__ import annotations


class AgentPlatformError(Exception):
    """Base class for all platform-raised errors."""


class HITLInterrupt(AgentPlatformError):
    """Raised inside the Loop when a sensitive tool needs human approval.

    The Loop catches this, persists a Checkpoint with status=waiting_approval,
    emits a hitl_required SSE event, and exits cleanly.

    Attributes:
        approval_id: opaque id the client uses to approve/reject.
        tool_name: which tool triggered the interrupt.
        tool_arguments: arguments the LLM wanted to pass.
        description: human-readable summary for the approval UI.
    """

    def __init__(
        self,
        *,
        approval_id: str,
        tool_name: str,
        tool_arguments: dict,
        description: str,
    ) -> None:
        super().__init__(f"HITL required for tool '{tool_name}' (approval={approval_id})")
        self.approval_id = approval_id
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        self.description = description


class LoopBudgetExceeded(AgentPlatformError):
    """Raised when the loop exceeds its configured iteration cap (default 100)."""

    def __init__(self, max_turns: int) -> None:
        super().__init__(f"Loop exceeded maximum of {max_turns} turns")
        self.max_turns = max_turns


class LoopEmptyResponse(AgentPlatformError):
    """Raised when the LLM returns no content AND no tool_calls N times in a row.

    After max retries we surface this so the Loop can finish with a graceful
    error event instead of looping forever.
    """

    def __init__(self, attempts: int) -> None:
        super().__init__(f"LLM returned empty response {attempts} times")
        self.attempts = attempts


class ToolPermissionDenied(AgentPlatformError):
    """Raised when a tool is not on the allow-list for the current thread."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is not permitted in this thread")
        self.tool_name = tool_name


class CheckpointVersionError(AgentPlatformError):
    """Raised when a Checkpoint snapshot's version doesn't match the current code."""

    def __init__(self, found: int, expected: int) -> None:
        super().__init__(
            f"Checkpoint version mismatch: found {found}, expected {expected}"
        )
        self.found = found
        self.expected = expected
