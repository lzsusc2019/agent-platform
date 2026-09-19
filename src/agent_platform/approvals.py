"""Human approval grants.

An approval is a decision about a specific target, not a blank cheque. This
store is what makes that true.

Before this existed, "approved" meant nothing more than "the resume request
carried an approval_id": the /hitl/approve endpoint persisted nothing, the
Loop only checked `approval_id is None`, and one approval therefore waved
through every sensitive call in the run — including writes to files nobody
looked at. Anyone who could read the SSE stream could approve.

Now the decision is recorded, keyed by the resource it covers and given a TTL:

* **Per target.** `Tool.approval_scope()` decides what one approval covers.
  `write_file` returns the resolved path, so approving notes/a.txt does not
  authorise notes/b.txt.
* **Time-boxed.** A grant expires after `approval_grant_ttl_seconds` (1 hour by
  default). Resuming with a stale grant re-prompts rather than executing.
* **Recorded.** Who it was for, which tool, which target, and when — enough for
  the Dashboard to show what is currently permitted.

The Loop consults this store, not the request, which is what makes an approval
mean something the caller cannot simply assert.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["APPROVAL_KEY_PREFIX", "ApprovalGrant", "ApprovalStore"]

APPROVAL_KEY_PREFIX = "approval"


def _scope_digest(scope: str) -> str:
    """Hash the scope so arbitrary paths cannot inject Redis key separators."""
    return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]


@dataclass
class ApprovalGrant:
    """One recorded human decision, scoped to a single target."""

    agent_id: str
    user_id: str
    tool_name: str
    scope: str
    # The approval that produced it. Kept for traceability back to the SSE
    # event the human actually saw.
    approval_id: str
    granted_at: float = field(default_factory=time.time)
    # Seconds remaining, filled in when listing. Not part of the stored value.
    ttl_seconds: int | None = None

    def public(self) -> dict[str, Any]:
        """JSON-safe view for the admin API."""
        out = asdict(self)
        out["granted_at_iso"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.granted_at)
        )
        out["scope_label"] = self.scope or f"<any target of {self.tool_name}>"
        return out


class ApprovalStore:
    """Redis-backed grant store. The TTL lives in Redis, not in our code."""

    def __init__(self, redis: Any, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._redis = redis
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, *, ttl_seconds: int) -> ApprovalStore:
        import redis.asyncio as aioredis

        return cls(aioredis.from_url(url, decode_responses=True), ttl_seconds=ttl_seconds)

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    @staticmethod
    def grant_key(agent_id: str, user_id: str, tool_name: str, scope: str) -> str:
        return (
            f"{APPROVAL_KEY_PREFIX}:{agent_id}:{user_id}:{tool_name}:"
            f"{_scope_digest(scope)}"
        )

    async def grant(self, grant: ApprovalGrant) -> None:
        """Record an approval, expiring after the configured TTL."""
        key = self.grant_key(
            grant.agent_id, grant.user_id, grant.tool_name, grant.scope
        )
        payload = json.dumps(asdict(grant))
        await self._redis.set(key, payload, ex=self._ttl)
        log.info(
            "approval.granted agent_id=%s tool=%s scope=%s ttl=%ds approval_id=%s",
            grant.agent_id,
            grant.tool_name,
            grant.scope or "(tool-wide)",
            self._ttl,
            grant.approval_id,
        )

    async def is_granted(
        self, *, agent_id: str, user_id: str, tool_name: str, scope: str
    ) -> bool:
        key = self.grant_key(agent_id, user_id, tool_name, scope)
        return await self._redis.get(key) is not None

    async def revoke(
        self, *, agent_id: str, user_id: str, tool_name: str, scope: str
    ) -> bool:
        key = self.grant_key(agent_id, user_id, tool_name, scope)
        return bool(await self._redis.delete(key))

    async def list_grants(self) -> list[ApprovalGrant]:
        """Every live grant, with its remaining TTL.

        Used by the Dashboard so an operator can see what is currently
        permitted without reading Redis by hand.
        """
        out: list[ApprovalGrant] = []
        keys = await self._redis.keys(f"{APPROVAL_KEY_PREFIX}:*")
        for raw in keys:
            key = raw.decode() if isinstance(raw, bytes) else raw
            value = await self._redis.get(key)
            if value is None:
                continue
            try:
                data = json.loads(value)
            except (TypeError, ValueError):
                continue
            ttl = await self._redis.ttl(key)
            grant = ApprovalGrant(**{**data, "ttl_seconds": max(0, int(ttl))})
            out.append(grant)
        return sorted(out, key=lambda g: g.granted_at, reverse=True)
