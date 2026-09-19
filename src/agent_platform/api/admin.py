"""Admin / Dashboard HTTP routes.

These power the /admin Dashboard page. They are intentionally separate from
the /v1 API surface so a future ops split (different auth, different
network exposure) is easy.

Endpoints:
- GET    /admin/                              -> serves the static dashboard HTML
- GET    /admin/api/configs                   -> list all agent configs
- GET    /admin/api/configs/{agent_id}        -> read one
- PUT    /admin/api/configs/{agent_id}        -> upsert (invalidates the cached AgentLoop)
- DELETE /admin/api/configs/{agent_id}        -> remove config + invalidate
- GET    /admin/api/checkpoints               -> list thread_ids with snapshot status
- GET    /admin/api/checkpoints/{thread_id}   -> full snapshot JSON
- DELETE /admin/api/checkpoints/{thread_id}   -> drop a snapshot (debug aid)
- GET    /admin/api/tools                     -> list registered tools + sensitive flag
- POST   /admin/api/chat                      -> debug: stream a chat turn (SSE), bypassing approval
"""

from __future__ import annotations

import json
import logging
from datetime import UTC
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent_platform.config_store import AgentConfig
from agent_platform.core.checkpoint import CheckpointStatus
from agent_platform.secrets_store import KNOWN_SECRETS, SecretEntry

log = logging.getLogger(__name__)
router = APIRouter()

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "static" / "admin.html"


# ---- pydantic request/response shapes ---------------------------------------


class UpsertConfigRequest(BaseModel):
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[str] | None = None
    skills: list[str] | None = None
    sensitive_tools: list[str] | None = None
    metadata: dict[str, Any] | None = None


# ---- dashboard HTML ---------------------------------------------------------


@router.get("/admin", include_in_schema=False)
@router.get("/admin/", include_in_schema=False)
async def admin_index() -> FileResponse:
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=503, detail="dashboard HTML missing")
    return FileResponse(DASHBOARD_PATH, media_type="text/html")


# ---- agent configs CRUD -----------------------------------------------------


@router.get("/admin/api/configs")
async def list_configs(request: Request) -> list[dict[str, Any]]:
    rt = request.app.state.runtime
    cfgs = await rt.config_store.list_all()
    return [c.model_dump() for c in cfgs]


@router.get("/admin/api/configs/{agent_id}")
async def get_config(agent_id: str, request: Request) -> dict[str, Any]:
    rt = request.app.state.runtime
    cfg = await rt.config_store.get(agent_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"config '{agent_id}' not found")
    return cfg.model_dump()


@router.put("/admin/api/configs/{agent_id}")
async def upsert_config(
    agent_id: str, body: UpsertConfigRequest, request: Request
) -> dict[str, Any]:
    rt = request.app.state.runtime
    existing = await rt.config_store.get(agent_id)
    # New agents start from the configured defaults, not from literals.
    base = (
        existing.model_dump()
        if existing
        else AgentConfig.with_defaults(agent_id, rt.settings).model_dump()
    )
    # Apply only provided fields.
    payload = body.model_dump(exclude_unset=True)
    base.update(payload)
    cfg = AgentConfig(**base)
    await rt.config_store.upsert(cfg)
    # Invalidate the cached AgentLoop so the next chat picks up the new
    # system_prompt / model / etc.
    rt.agents.invalidate(agent_id)
    return cfg.model_dump()


@router.delete("/admin/api/configs/{agent_id}")
async def delete_config(agent_id: str, request: Request) -> dict[str, bool]:
    rt = request.app.state.runtime
    removed = await rt.config_store.delete(agent_id)
    rt.agents.invalidate(agent_id)
    return {"deleted": removed}


# ---- checkpoints introspection ----------------------------------------------


@router.get("/admin/api/checkpoints")
async def list_checkpoints(request: Request) -> list[dict[str, Any]]:
    rt = request.app.state.runtime
    thread_ids = await rt.checkpoint.list_threads()
    out: list[dict[str, Any]] = []
    for tid in thread_ids:
        snap = await rt.checkpoint.load(tid)
        if snap is None:
            continue
        out.append(
            {
                "thread_id": tid,
                "status": snap.status.value,
                "turn": snap.turn,
                "messages": len(snap.messages),
                "last_config_agent": snap.last_config.get("agent_id", "?"),
            }
        )
    return out


@router.get("/admin/api/checkpoints/{thread_id}")
async def get_checkpoint(thread_id: str, request: Request) -> dict[str, Any]:
    rt = request.app.state.runtime
    snap = await rt.checkpoint.load(thread_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"checkpoint '{thread_id}' not found")
    return snap.model_dump(mode="json")


@router.delete("/admin/api/checkpoints/{thread_id}")
async def delete_checkpoint(thread_id: str, request: Request) -> dict[str, bool]:
    rt = request.app.state.runtime
    await rt.checkpoint.delete(thread_id)
    return {"deleted": True}


# ---- tools introspection ----------------------------------------------------


@router.get("/admin/api/tools")
async def list_tools(request: Request) -> list[dict[str, Any]]:
    rt = request.app.state.runtime
    schemas = rt.tools.schemas_for_llm()
    sensitive = rt.tools.sensitive_tools()
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
            "sensitive": s["name"] in sensitive,
        }
        for s in schemas
    ]


# ---- debug chat (SSE) -------------------------------------------------------


class DebugChatRequest(BaseModel):
    agent_id: str
    thread_id: str | None = None
    content: str = ""
    # For HITL resume from the dashboard.
    approval_id: str | None = None


@router.post("/admin/api/chat")
async def debug_chat(req: DebugChatRequest, request: Request) -> StreamingResponse:
    """Stream a chat turn. No auth, no quota — this is a debug endpoint."""
    import uuid

    rt = request.app.state.runtime
    thread_id = req.thread_id or uuid.uuid4().hex
    snap = await rt.checkpoint.load(thread_id)
    if snap is None:
        # Create an empty snapshot so the Loop has a place to start.
        snap = rt.checkpoint.new_snapshot(
            thread_id=thread_id,
            messages=[],
            status=CheckpointStatus.RUNNING,
            turn=0,
            last_config={"agent_id": req.agent_id, "user_id": "admin-debug"},
        )
        await rt.checkpoint.save(snap)

    # Mirror the guard /v1/sessions/{id}/chat already applies. Without it, a new
    # message typed into Debug Chat while a thread is parked appends a user turn
    # after the unanswered tool_calls — which every provider rejects. Better to
    # say so than to let the model 400 on a request the operator did not know
    # was malformed.
    if snap.status == CheckpointStatus.WAITING_APPROVAL and req.approval_id is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"thread {thread_id} is awaiting approval; approve or reject it "
                "first, or resend with approval_id to resume"
            ),
        )

    # get_or_create itself can raise (unknown provider, bad config).
    # Surface that as a proper error frame instead of a 500 HTML body,
    # since the caller is an SSE client that expects an event stream.
    try:
        agent = await rt.agents.get_or_create(req.agent_id)
    except Exception as e:
        log.exception("admin.chat.setup_failed agent_id=%s", req.agent_id)
        raise HTTPException(
            status_code=400,
            detail=f"failed to build agent '{req.agent_id}': {type(e).__name__}: {e}",
        ) from e

    # Speak as whoever the thread belongs to. A grant is issued to a principal,
    # so resuming as a different user would re-prompt for an approval the
    # operator just gave — and, worse, would let the Debug Chat act as an
    # identity the thread was not created under.
    debug_user = snap.user_id or "admin-debug"

    async def event_source():
        try:
            async for ev in agent.run(
                thread_id=thread_id,
                user_id=debug_user,
                user_message=req.content or None,
                approval_id=req.approval_id,
            ):
                # ev.to_sse() returns {"event": "...", "data": "..."}; render
                # as proper SSE text for StreamingResponse.
                sse = ev.to_sse()
                yield f"event: {sse['event']}\ndata: {sse['data']}\n\n"
        except Exception as e:
            # Safety net. The Loop converts its own failures into `error`
            # events, but LoopBudgetExceeded / LoopEmptyResponse still
            # propagate, and a middleware upstream of the loop could raise.
            # Without this the response body would just stop mid-stream and
            # the browser would show a truncated/empty event panel.
            log.exception("admin.chat.stream_failed thread_id=%s", thread_id)
            payload = json.dumps(
                {
                    "thread_id": thread_id,
                    "turn": -1,
                    "reason": "stream_error",
                    "error_type": type(e).__name__,
                    "message": str(e),
                },
                ensure_ascii=False,
                default=str,
            )
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-store",
            "x-thread-id": thread_id,
        },
    )


# ---- HITL introspection ----------------------------------------------------


@router.get("/admin/api/hitl/pending")
async def list_pending_hitl(request: Request) -> list[dict[str, Any]]:
    """List thread_ids whose Checkpoint is WAITING_APPROVAL."""
    rt = request.app.state.runtime
    thread_ids = await rt.checkpoint.list_threads()
    out: list[dict[str, Any]] = []
    for tid in thread_ids:
        snap = await rt.checkpoint.load(tid)
        if snap is None or snap.status != CheckpointStatus.WAITING_APPROVAL:
            continue
        # Find the most recent hitl_required tool name from messages.
        tool_name = None
        agent_id = str(snap.last_config.get("agent_id", "?"))
        for m in reversed(snap.messages):
            if m.role.value == "assistant" and m.tool_calls:
                tool_name = m.tool_calls[0].name
                break
        out.append(
            {
                "thread_id": tid,
                "agent_id": agent_id,
                "tool_name": tool_name,
                "turn": snap.turn,
                # Recorded by the Loop when it parked the thread. The Dashboard
                # needs it to offer an approve/reject action.
                "approval_id": snap.approval_id,
            }
        )
    return out


# ---- approval grants (Dashboard) -------------------------------------------


class RevokeGrantRequest(BaseModel):
    agent_id: str
    user_id: str
    tool_name: str
    scope: str = ""


@router.get("/admin/api/approvals")
async def list_approvals(request: Request) -> list[dict[str, Any]]:
    """Live approval grants, with the time each has left.

    Without this, "why did that write not ask me again?" is unanswerable from
    the UI — the grant is real state now, so it should be visible state.
    """
    rt = request.app.state.runtime
    return [g.public() for g in await rt.approval_store.list_grants()]


@router.post("/admin/api/approvals/revoke")
async def revoke_approval(request: Request, body: RevokeGrantRequest) -> dict[str, bool]:
    """Revoke one grant, so the next matching call asks again."""
    rt = request.app.state.runtime
    removed = await rt.approval_store.revoke(
        agent_id=body.agent_id,
        user_id=body.user_id,
        tool_name=body.tool_name,
        scope=body.scope,
    )
    return {"revoked": removed}


# ---- provider secrets (Dashboard Providers tab) ----------------------------


class UpsertSecretRequest(BaseModel):
    value: str
    note: str = ""


@router.get("/admin/api/secrets")
async def list_secrets(request: Request) -> list[dict[str, Any]]:
    """List known secret slots, with `has_value` indicating whether one
    is currently set (in store or env). The actual value is NEVER returned.
    """
    rt = request.app.state.runtime
    stored = {f"{e.provider}:{e.name}": e for e in await rt.secret_store.list()}
    out: list[dict[str, Any]] = []
    for provider, names in KNOWN_SECRETS.items():
        for name in names:
            entry = stored.get(f"{provider}:{name}")
            env_value = _env_value_for(provider, name, rt.settings)
            if entry is not None:
                out.append(entry.to_public() | {"source": "store"})
            elif env_value:
                out.append(
                    {
                        "provider": provider,
                        "name": name,
                        "masked": _mask(env_value),
                        "updated_at": None,
                        "note": "from environment variable",
                        "has_value": True,
                        "source": "env",
                    }
                )
            else:
                out.append(
                    {
                        "provider": provider,
                        "name": name,
                        "masked": "",
                        "updated_at": None,
                        "note": "",
                        "has_value": False,
                        "source": None,
                    }
                )
    return out


@router.put("/admin/api/secrets/{provider}/{name}")
async def upsert_secret(
    provider: str, name: str, body: UpsertSecretRequest, request: Request
) -> dict[str, Any]:
    """Set a provider secret. Overrides any env var for that provider."""
    from datetime import datetime

    if provider not in KNOWN_SECRETS or name not in KNOWN_SECRETS[provider]:
        raise HTTPException(
            status_code=404,
            detail=f"unknown secret slot '{provider}/{name}'",
        )
    if not body.value.strip():
        raise HTTPException(status_code=400, detail="value cannot be empty")

    # Catch header-hostile credentials at write time. Otherwise the failure
    # surfaces later as a UnicodeEncodeError from deep inside httpx, which
    # does not name the offending field.
    from agent_platform.core.providers import validate_api_key

    try:
        validate_api_key(body.value.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    rt = request.app.state.runtime
    entry = SecretEntry(
        provider=provider,
        name=name,
        value=body.value.strip(),
        note=body.note,
        updated_at=datetime.now(UTC),
    )
    await rt.secret_store.set(entry)
    # Drop any cached AgentLoops that depended on this provider's key.
    invalidated = await rt.agents.invalidate_stale_secrets()
    return {"saved": True, "invalidated_agents": invalidated}


@router.delete("/admin/api/secrets/{provider}/{name}")
async def delete_secret(provider: str, name: str, request: Request) -> dict[str, Any]:
    rt = request.app.state.runtime
    removed = await rt.secret_store.delete(provider, name)
    invalidated = await rt.agents.invalidate_stale_secrets()
    return {"deleted": removed, "invalidated_agents": invalidated}


@router.post("/admin/api/secrets/{provider}/{name}/test")
async def test_secret(provider: str, name: str, request: Request) -> dict[str, Any]:
    """Verify that a configured key actually works against the provider API.

    For DeepSeek this issues a cheap /models request. Returns
    `{"ok": bool, "status": int|None, "detail": str}`.
    """
    import httpx

    if provider != "deepseek" or name != "api_key":
        raise HTTPException(status_code=404, detail="test endpoint not implemented for this slot")
    rt = request.app.state.runtime
    # Resolve effective key: store first, env fallback.
    api_key: str | None = None
    if rt.secret_store is not None:
        entry = await rt.secret_store.get(provider, name)
        if entry is not None:
            api_key = entry.value
    if not api_key:
        api_key = rt.settings.deepseek_api_key or None
    if not api_key:
        return {"ok": False, "status": None, "detail": "no key configured"}

    base = rt.settings.deepseek_base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=rt.settings.admin_secret_test_timeout) as client:
            r = await client.get(
                base + rt.settings.deepseek_models_path,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        return {
            "ok": r.status_code == 200,
            "status": r.status_code,
            "detail": r.text[:200] if r.status_code != 200 else "ok",
        }
    except httpx.HTTPError as e:
        return {"ok": False, "status": None, "detail": f"transport: {e}"}


def _env_value_for(provider: str, name: str, settings: Any) -> str:
    """Return the env-var override for a known secret slot, if any."""
    if provider == "deepseek" and name == "api_key":
        return settings.deepseek_api_key
    return ""


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * 8}{value[-4:]}"
