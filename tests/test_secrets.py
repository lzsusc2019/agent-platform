"""Unit tests for SecretStore."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_platform.secrets_store import SecretEntry, SecretStore


@pytest.mark.asyncio
async def test_secret_roundtrip() -> None:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    redis = fakeredis.aioredis.FakeRedis()
    store = SecretStore(redis)
    entry = SecretEntry(
        provider="deepseek",
        name="api_key",
        value="sk-test-1234567890",
        note="primary",
        updated_at=datetime.now(UTC),
    )
    await store.set(entry)
    loaded = await store.get("deepseek", "api_key")
    assert loaded is not None
    assert loaded.value == "sk-test-1234567890"
    assert loaded.note == "primary"
    assert loaded.provider == "deepseek"
    await redis.aclose()


@pytest.mark.asyncio
async def test_secret_get_value_helper() -> None:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    redis = fakeredis.aioredis.FakeRedis()
    store = SecretStore(redis)
    assert await store.get("deepseek", "api_key") is None
    await store.set(
        SecretEntry(
            provider="deepseek",
            name="api_key",
            value="sk-x",
            updated_at=datetime.now(UTC),
        )
    )
    entry = await store.get("deepseek", "api_key")
    assert entry is not None and entry.value == "sk-x"
    await redis.aclose()


@pytest.mark.asyncio
async def test_secret_delete_returns_bool() -> None:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    redis = fakeredis.aioredis.FakeRedis()
    store = SecretStore(redis)
    assert await store.delete("deepseek", "api_key") is False
    await store.set(
        SecretEntry(
            provider="deepseek",
            name="api_key",
            value="x",
            updated_at=datetime.now(UTC),
        )
    )
    assert await store.delete("deepseek", "api_key") is True
    assert await store.delete("deepseek", "api_key") is False
    await redis.aclose()


@pytest.mark.asyncio
async def test_secret_list_returns_all_stored() -> None:
    import fakeredis.aioredis  # type: ignore[import-untyped]

    redis = fakeredis.aioredis.FakeRedis()
    store = SecretStore(redis)
    now = datetime.now(UTC)
    await store.set(SecretEntry(provider="deepseek", name="api_key", value="a", updated_at=now))
    await store.set(SecretEntry(provider="openai", name="api_key", value="b", updated_at=now))
    out = await store.list()
    providers = {e.provider for e in out}
    assert providers == {"deepseek", "openai"}
    await redis.aclose()


def test_mask_long_value() -> None:
    e = SecretEntry(
        provider="deepseek",
        name="api_key",
        value="sk-1234567890abcdef",
        updated_at=datetime.now(UTC),
    )
    masked = e.masked()
    # Prefix (4) + 8 stars + suffix (4).
    assert masked.startswith("sk-1")
    assert masked.endswith("cdef")
    assert "****" in masked
    assert "1234567890abcdef" not in masked


def test_mask_short_value() -> None:
    e = SecretEntry(
        provider="x",
        name="api_key",
        value="short",
        updated_at=datetime.now(UTC),
    )
    assert e.masked() == "*****"


def test_to_public_does_not_leak_value() -> None:
    e = SecretEntry(
        provider="deepseek",
        name="api_key",
        value="sk-supersecret-99999999",
        updated_at=datetime.now(UTC),
    )
    pub = e.to_public()
    assert "value" not in pub
    assert "supersecret" not in pub["masked"]
    assert pub["has_value"] is True
