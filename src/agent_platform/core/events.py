"""Loop events — streamed to the API layer as SSE."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class EventType(str, Enum):
    # Lifecycle.
    START = "start"
    FINISH = "finish"
    ERROR = "error"
    # LLM.
    ASSISTANT = "assistant"           # partial assistant text
    TOOL_CALL = "tool_call"           # LLM requested a tool
    TOOL_RESULT = "tool_result"       # tool execution finished
    # Compression.
    COMPRESSED = "compressed"         # context was compressed this turn
    # HITL.
    HITL_REQUIRED = "hitl_required"   # loop is pausing for approval
    # Approval outcome (when the resumed Loop restarts).
    HITL_RESOLVED = "hitl_resolved"


class LoopEvent(BaseModel):
    """A single observable event from the Agent Loop."""

    type: EventType
    thread_id: str
    turn: int  # 0-based loop iteration
    data: dict[str, Any] = {}

    def to_sse(self) -> dict[str, str]:
        """Format as an SSE event payload."""
        import json

        return {
            "event": self.type.value,
            "data": json.dumps(
                {"thread_id": self.thread_id, "turn": self.turn, **self.data},
                ensure_ascii=False,
                default=str,
            ),
        }
