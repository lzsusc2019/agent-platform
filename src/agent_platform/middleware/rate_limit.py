"""Rate-limit middleware — placeholder.

Real implementation would be a Redis token-bucket keyed by user_id +
endpoint, with sliding-window semantics. The MVP just logs and passes
through; rate-limit policies are enforced at the gateway in production.
"""

from __future__ import annotations

from fastapi import Request, Response


class RateLimitMiddleware:  # pragma: no cover - placeholder
    async def __call__(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["x-ratelimit-placeholder"] = "true"
        return response
