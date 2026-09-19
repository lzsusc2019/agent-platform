"""Redis-backed CheckpointStore.

Snapshot model (see ADR-003):
- Key: `ckpt:{thread_id}`
- Value: JSON-serialized CheckpointSnapshot
- TTL: supplied by the caller (`Settings.checkpoint_ttl_seconds`)

Idempotency record:
- Key: `tool:{idempotency_key}`
- Value: JSON-serialized ToolExecutionRecord (mostly for debug; the snapshot's
  pending_tools map is the source of truth for resume logic)
- TTL: same as the snapshot

The store is deliberately tiny: no transactions, no Lua scripts. Cross-key
atomicity isn't required because we never read+write two keys atomically —
the snapshot's pending_tools map is always written together with the
snapshot in one SET.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Protocol

import redis.asyncio as aioredis

from agent_platform.core.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointSnapshot,
    CheckpointStatus,
    CheckpointVersionError,
)
from agent_platform.core.messages import Message

log = logging.getLogger(__name__)


class RedisLike(Protocol):
    """Subset of redis.asyncio.Redis we use. Lets tests pass fakeredis."""

    async def get(self, key: str) -> bytes | None: ...
    async def set(self, key: str, value: str | bytes, ex: int | None = None) -> bool | None: ...
    async def delete(self, *keys: str) -> int: ...
    async def keys(self, pattern: str) -> list[bytes]: ...
    async def ttl(self, key: str) -> int: ...


def snapshot_key(thread_id: str) -> str:
    return f"ckpt:{thread_id}"


class CheckpointStore:
    """Thin async wrapper over Redis. MVP-grade; not transactional."""

    def __init__(self, redis: RedisLike, *, ttl_seconds: int) -> None:
        """`ttl_seconds` is required.

        The TTL is a policy decision (see ADR-003) that belongs to config,
        not a library default — passing it explicitly keeps the expiry of
        every deployment auditable from `Settings`.
        """
        self._redis = redis
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, *, ttl_seconds: int) -> CheckpointStore:
        return cls(aioredis.from_url(url), ttl_seconds=ttl_seconds)

    # ----- snapshot CRUD -----

    async def load(self, thread_id: str) -> CheckpointSnapshot | None:
        raw = await self._redis.get(snapshot_key(thread_id))
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            snap = CheckpointSnapshot.model_validate(data)
        except Exception as e:
            log.warning("checkpoint.load.deserialize_failed thread_id=%s error=%s", thread_id, e)
            return None
        if snap.version != CHECKPOINT_VERSION:
            raise CheckpointVersionError(snap.version, CHECKPOINT_VERSION)
        return snap

    async def save(self, snap: CheckpointSnapshot) -> None:
        # version is set on construction; bump only on schema-incompatible
        # changes (not handled automatically).
        payload = snap.model_dump_json()
        await self._redis.set(snapshot_key(snap.thread_id), payload, ex=self._ttl)

    async def delete(self, thread_id: str) -> None:
        await self._redis.delete(snapshot_key(thread_id))

    async def exists(self, thread_id: str) -> bool:
        return await self._redis.get(snapshot_key(thread_id)) is not None

    async def list_threads(self) -> list[str]:
        keys = await self._redis.keys("ckpt:*")
        return [k.decode().removeprefix("ckpt:") for k in keys]

    # ----- helper builders -----

    def new_snapshot(
        self,
        thread_id: str,
        messages: list[Message],
        status: CheckpointStatus = CheckpointStatus.RUNNING,
        turn: int = 0,
        last_config: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> CheckpointSnapshot:
        return CheckpointSnapshot(
            thread_id=thread_id,
            version=CHECKPOINT_VERSION,
            status=status,
            messages=messages,
            turn=turn,
            last_config=last_config or {},
            # Carried on every snapshot, not just the initial one. It used to
            # live only in last_config, which the Loop overwrites with its own
            # config on every write — so the caller's identity survived exactly
            # one save and every later turn ran as "unknown".
            user_id=user_id,
        )


def new_idempotency_key(thread_id: str, tool_call_id: str) -> str:
    """Generate a globally-unique key per (thread, tool_call).

    The thread prefix means a resume on the same thread reuses the same key,
    which is exactly what we want for the PENDING -> DONE state machine.
    """
    # uuid4 namespace + tool_call_id is overkill but unambiguous.
    return f"{thread_id}:{tool_call_id}:{uuid.uuid4().hex[:8]}"
