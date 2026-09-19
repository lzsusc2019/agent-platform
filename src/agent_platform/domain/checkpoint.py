"""Checkpoint data model."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.domain.errors import CheckpointVersionError
from agent_platform.domain.messages import Message

__all__ = [
    "CHECKPOINT_VERSION",
    "CheckpointSnapshot",
    "CheckpointStatus",
    "CheckpointVersionError",
    "ToolPendingState",
]


class CheckpointStatus(str, Enum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    FINISHED = "finished"


class ToolPendingState(str, Enum):
    """Tri-state for a tool call's idempotency record.

    - DONE: completed in a previous run; reuse result on resume.
    - PENDING: started but unknown outcome (timeout / Checkpoint write failed).
      MUST NOT be treated as failure. Re-attempt only after external confirmation.
    - FAILED: failed cleanly; eligible for retry.
    """

    DONE = "done"
    PENDING = "pending"
    FAILED = "failed"


class CheckpointSnapshot(BaseModel):
    """The persisted ReAct state for one thread.

    Versioned: bumps allow graceful handling of incompatible state schemas
    after deploys.
    """

    thread_id: str
    version: int = 1
    status: CheckpointStatus = CheckpointStatus.RUNNING
    messages: list[Message] = Field(default_factory=list)
    pending_tools: dict[str, ToolPendingState] = Field(default_factory=dict)
    # Idempotency key -> cached ToolResult content, so a DONE call on resume
    # can be replayed without re-executing the tool.
    done_results: dict[str, str] = Field(default_factory=dict)
    # Snapshot of agent config at write time. Resume code compares to detect
    # that config has changed under us.
    last_config: dict[str, Any] = Field(default_factory=dict)
    # Loop turn counter at the time of snapshot. Useful for "resume from turn N".
    turn: int = 0
    # Set while status is WAITING_APPROVAL: the id the client must echo back to
    # resume. Stored rather than re-derived, so the approval-id format stays the
    # Loop's business and the admin API can report pending approvals without
    # reconstructing it. Optional, so snapshots written before this field
    # existed still deserialize.
    approval_id: str | None = None
    # Who the pending approval is for. A grant is issued to a principal, so the
    # approver needs to know what it is being issued against.
    user_id: str | None = None
    # The call the human is actually being asked about. Stored so /hitl/approve
    # can record a grant for exactly that target without having to re-derive it
    # by parsing the approval-id format, which is the Loop's business.
    pending_tool_name: str | None = None
    pending_tool_arguments: dict[str, Any] = Field(default_factory=dict)


# Current schema version. Bump when CheckpointSnapshot changes shape in an
# incompatible way.
CHECKPOINT_VERSION = 1
