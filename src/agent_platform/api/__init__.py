"""FastAPI HTTP layer.

Endpoints (all /v1):
- POST /sessions                -> create a session
- GET  /sessions/{sid}          -> read session + Checkpoint status
- POST /sessions/{sid}/chat     -> stream a chat turn (SSE)
- POST /sessions/{sid}/hitl/approve  -> approve a pending tool call
- POST /sessions/{sid}/hitl/reject   -> reject a pending tool call
- GET  /healthz                 -> liveness
"""

from agent_platform.api.app import create_app

__all__ = ["create_app"]
