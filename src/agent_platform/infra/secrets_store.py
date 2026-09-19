"""SecretStore — runtime-mutable provider credentials, Redis-backed.

The Dashboard's "Providers" tab writes here so you can rotate a DeepSeek
(or future OpenAI / Anthropic / ...) API key without restarting the
server. Env vars (AGENT_PLATFORM_DEEPSEEK_API_KEY) are still respected
as the initial seed; anything in the store overrides env at lookup time.

Keys are stored under `secret:<provider>:<name>` (e.g. `secret:deepseek:api_key`).
Values are never returned over the wire unmasked — the API returns
only `masked` (first 4 chars + "...") and metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


class SecretEntry(BaseModel):
    provider: str
    name: str
    # Never returned by the API. Use masked for UI display.
    value: str
    updated_at: datetime
    # Optional user-supplied note ("primary", "backup", "team-a")
    note: str = ""

    def masked(self) -> str:
        if not self.value:
            return ""
        if len(self.value) <= 8:
            return "*" * len(self.value)
        # Show prefix (typically the provider's scheme like "sk-"), mask the
        # middle, leave the last 4 for disambiguation when comparing keys.
        prefix = self.value[:4]
        suffix = self.value[-4:]
        return f"{prefix}{'*' * 8}{suffix}"

    def to_public(self) -> dict[str, Any]:
        """The safe-to-expose view for the Dashboard."""
        return {
            "provider": self.provider,
            "name": self.name,
            "masked": self.masked(),
            "updated_at": self.updated_at.isoformat(),
            "note": self.note,
            "has_value": bool(self.value),
        }


def _secret_key(provider: str, name: str) -> str:
    return f"secret:{provider}:{name}"


class SecretStore:
    """Async Redis wrapper for provider secret CRUD.

    The store is deliberately simple: one JSON blob per (provider, name).
    No encryption at rest — secrets are stored in plaintext in Redis.
    Production deployments should run Redis with at-rest encryption
    (e.g. AWS KMS-backed EBS) and access control, OR replace this store
    with a Vault/KMS client. See ADR-005.
    """

    def __init__(self, redis: Any, ttl_seconds: int | None = None) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, ttl_seconds: int | None = None) -> SecretStore:
        import redis.asyncio as aioredis

        return cls(aioredis.from_url(url), ttl_seconds=ttl_seconds)

    async def get(self, provider: str, name: str) -> SecretEntry | None:
        raw = await self._redis.get(_secret_key(provider, name))
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return SecretEntry.model_validate(data)
        except Exception as e:
            log.warning(
                "secret.deserialize_failed provider=%s name=%s error=%s",
                provider,
                name,
                e,
            )
            return None

    async def set(self, entry: SecretEntry) -> None:
        payload = entry.model_dump_json()
        if self._ttl is None:
            await self._redis.set(
                _secret_key(entry.provider, entry.name), payload
            )
        else:
            await self._redis.set(
                _secret_key(entry.provider, entry.name), payload, ex=self._ttl
            )

    async def delete(self, provider: str, name: str) -> bool:
        n = await self._redis.delete(_secret_key(provider, name))
        return n > 0

    async def list(self) -> list[SecretEntry]:
        keys = await self._redis.keys("secret:*")
        out: list[SecretEntry] = []
        for k in keys:
            raw = await self._redis.get(k)
            if raw is None:
                continue
            try:
                out.append(SecretEntry.model_validate_json(raw))
            except Exception as e:
                log.warning("secret.list.deserialize_failed key=%s error=%s", k, e)
        return out


# Well-known secret names. The Dashboard uses these as default inputs so
# users don't have to know the internal keying scheme.
KNOWN_SECRETS: dict[str, list[str]] = {
    "deepseek": ["api_key"],
    "openai": ["api_key"],  # not registered as a provider yet, but the
    # slot exists so the Dashboard doesn't 404 on a known name.
}
