"""HTTP routes — sessions, chat (SSE), HITL approval."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent_platform.approvals import ApprovalGrant
from agent_platform.core.checkpoint import CheckpointStatus
from agent_platform.core.messages import Message, MessageRole, unanswered_tool_calls

log = logging.getLogger(__name__)
router = APIRouter()


# ---- schemas ---------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    agent_id: str
    user_id: str


class SessionInfo(BaseModel):
    thread_id: str
    agent_id: str
    user_id: str
    status: str
    turn: int


class ChatRequest(BaseModel):
    content: str
    # When resuming after HITL, the approval_id returned by the hitl_required
    # event. Optional for fresh conversations.
    approval_id: str | None = None


class ApprovalRequest(BaseModel):
    approval_id: str


# ---- session CRUD ----------------------------------------------------------


@router.post("/sessions", response_model=SessionInfo)
async def create_session(req: CreateSessionRequest, request: Request) -> SessionInfo:
    rt = request.app.state.runtime
    thread_id = uuid.uuid4().hex
    # Touch the registry so the Agent is constructed.
    await rt.agents.get_or_create(req.agent_id)
    # Initialise an empty Checkpoint snapshot. This isn't strictly required
    # (the Loop creates one on first save) but it gives the caller an explicit
    # "session exists" handle.
    snap = rt.checkpoint.new_snapshot(
        thread_id=thread_id,
        messages=[],
        status=CheckpointStatus.RUNNING,
        turn=0,
        last_config={"agent_id": req.agent_id, "user_id": req.user_id},
        user_id=req.user_id,
    )
    await rt.checkpoint.save(snap)
    return SessionInfo(
        thread_id=thread_id,
        agent_id=req.agent_id,
        user_id=req.user_id,
        status=snap.status.value,
        turn=snap.turn,
    )


@router.get("/sessions/{thread_id}", response_model=SessionInfo)
async def get_session(thread_id: str, request: Request) -> SessionInfo:
    rt = request.app.state.runtime
    snap = await rt.checkpoint.load(thread_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionInfo(
        thread_id=thread_id,
        agent_id=str(snap.last_config.get("agent_id", "default")),
        user_id=str(snap.user_id or snap.last_config.get("user_id") or "unknown"),
        status=snap.status.value,
        turn=snap.turn,
    )


# ---- chat (SSE) ------------------------------------------------------------


@router.post("/sessions/{thread_id}/chat")
async def chat(
    thread_id: str, req: ChatRequest, request: Request
) -> EventSourceResponse:
    rt = request.app.state.runtime
    snap = await rt.checkpoint.load(thread_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="session not found")
    agent_id = str(snap.last_config.get("agent_id", "default"))
    # Prefer the snapshot field; last_config is the legacy location and the
    # Loop overwrites it, so it is only good for the very first turn.
    user_id = str(snap.user_id or snap.last_config.get("user_id") or "unknown")
    if snap.status == CheckpointStatus.WAITING_APPROVAL and req.approval_id is None:
        raise HTTPException(
            status_code=409,
            detail="session is awaiting approval; supply approval_id",
        )

    agent = await rt.agents.get_or_create(agent_id)

    async def event_source() -> AsyncIterator[dict]:
        async for ev in agent.run(
            thread_id=thread_id,
            user_id=user_id,
            user_message=req.content,
            approval_id=req.approval_id,
        ):
            yield ev.to_sse()

    return EventSourceResponse(event_source())


# ---- HITL approvals --------------------------------------------------------


def _grant_scope(rt: object, snap: object) -> tuple[str, str]:
    """(tool_name, scope) for the call this thread is parked on."""
    name = getattr(snap, "pending_tool_name", None) or ""
    arguments = getattr(snap, "pending_tool_arguments", None) or {}
    tools = getattr(rt, "tools", None)
    if not name or tools is None or name not in tools.names():
        return name, ""
    return name, tools.get(name).approval_scope(arguments)


@router.post("/sessions/{thread_id}/hitl/approve")
async def approve(thread_id: str, req: ApprovalRequest, request: Request) -> dict:
    """Record the human decision as a grant, scoped to what was approved.

    This endpoint used to validate and return without writing anything, which
    made it decorative: the resume call carried an approval_id and the Loop
    believed it, so one approval waved through every sensitive call in the run
    and a client could approve by asserting an id it had merely read off the
    event stream.

    Now the decision is recorded, keyed by the target it covers and given a
    TTL. The resume re-checks that grant. Approving notes/a.txt therefore does
    not authorise notes/b.txt, and an hour later both ask again.

    We still don't run the Loop here — the frontend re-issues the chat call to
    actually resume.
    """
    rt = request.app.state.runtime
    snap = await rt.checkpoint.load(thread_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="session not found")
    if snap.status != CheckpointStatus.WAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"session is not awaiting approval (status={snap.status.value})",
        )
    # Exact match, not a prefix. The id we handed out identifies one specific
    # tool call; anything else was not what the human was shown.
    if not snap.approval_id or req.approval_id != snap.approval_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "approval_id does not match the approval this session is "
                "waiting on"
            ),
        )

    tool_name, scope = _grant_scope(rt, snap)
    if not tool_name:
        raise HTTPException(
            status_code=409,
            detail="this session has no pending tool call to approve",
        )

    await rt.approval_store.grant(
        ApprovalGrant(
            agent_id=str(snap.last_config.get("agent_id") or "?"),
            user_id=snap.user_id or "?",
            tool_name=tool_name,
            scope=scope,
            approval_id=req.approval_id,
        )
    )
    return {
        "thread_id": thread_id,
        "approval_id": req.approval_id,
        "decision": "approved",
        # What the approval actually covers, so a caller can tell a narrow
        # grant from a tool-wide one without guessing.
        "granted": {
            "tool": tool_name,
            "scope": scope or f"(any target of {tool_name})",
            "expires_in_seconds": rt.approval_store.ttl_seconds,
        },
    }


@router.post("/sessions/{thread_id}/hitl/reject")
async def reject(thread_id: str, req: ApprovalRequest, request: Request) -> dict:
    rt = request.app.state.runtime
    snap = await rt.checkpoint.load(thread_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="session not found")
    if snap.status != CheckpointStatus.WAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"session is not awaiting approval (status={snap.status.value})",
        )
    if not req.approval_id.startswith(f"appr_{thread_id}_"):
        raise HTTPException(status_code=400, detail="approval_id does not match session")
    # Answer the tool calls the model made before closing the thread.
    #
    # Without this the Checkpoint keeps an assistant message whose tool_calls
    # are never answered — and every OpenAI-compatible endpoint rejects that
    # outright. The rejection itself was fine; what broke was any *later* turn
    # on the same thread, which replayed the history and got
    # "an assistant message with 'tool_calls' must be followed by tool
    # messages". Recording the refusal is also the honest outcome: the model
    # learns its request was turned down, instead of the turn vanishing.
    messages = list(snap.messages)
    for tc in unanswered_tool_calls(messages):
        messages.append(
            Message(
                role=MessageRole.TOOL,
                content=(
                    "error: a human denied approval for this tool call. It did "
                    "not run. Do not retry it unless the user asks again."
                ),
                tool_call_id=tc.id,
                meta={"tool_name": tc.name, "denied": True, "is_error": True},
            )
        )
    # Mark Checkpoint FINISHED — the conversation ends on reject.
    finished = snap.model_copy(
        update={
            "status": CheckpointStatus.FINISHED,
            "pending_tools": {},
            "messages": messages,
            # The approval is spent either way; leaving it set would make a
            # finished thread look like it was still waiting on someone.
            "approval_id": None,
            "pending_tool_name": None,
            "pending_tool_arguments": {},
        }
    )
    await rt.checkpoint.save(finished)
    return {"thread_id": thread_id, "approval_id": req.approval_id, "decision": "rejected"}
