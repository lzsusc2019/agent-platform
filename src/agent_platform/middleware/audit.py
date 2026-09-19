"""Audit middleware — placeholder.

A real implementation would write structured access logs to MySQL / ClickHouse
on every request, including the resolved user_id, agent_id, and the size of
the response. We expose the class so the FastAPI app can mount it once real
storage lands.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

log = logging.getLogger(__name__)


class AuditLogMiddleware:  # pragma: no cover - placeholder
    def __init__(self, app: Callable[[Request], Awaitable[Response]]) -> None:
        self.app = app

    async def __call__(self, request: Request) -> Response:
        response = await self.app(request)
        log.info(
            "http.audit method=%s path=%s status=%d",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response
