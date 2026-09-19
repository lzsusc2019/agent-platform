"""Checkpoint store tests."""

from __future__ import annotations

import pytest

from agent_platform.domain.checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointSnapshot,
    CheckpointVersionError,
    ToolPendingState,
)
from agent_platform.domain.messages import Message, MessageRole
from agent_platform.infra.checkpoint_store import CheckpointStore, new_idempotency_key


@pytest.mark.asyncio
async def test_save_load_roundtrip(ckpt_store: CheckpointStore) -> None:
    snap = CheckpointSnapshot(
        thread_id="t1",
        messages=[Message(role=MessageRole.USER, content="hi")],
        pending_tools={"call_x": ToolPendingState.DONE},
        done_results={"call_x": "ok"},
        turn=2,
    )
    await ckpt_store.save(snap)
    loaded = await ckpt_store.load("t1")
    assert loaded is not None
    assert loaded.thread_id == "t1"
    assert loaded.version == CHECKPOINT_VERSION
    assert loaded.turn == 2
    assert loaded.messages[0].content == "hi"
    assert loaded.pending_tools == {"call_x": ToolPendingState.DONE}
    assert loaded.done_results == {"call_x": "ok"}


@pytest.mark.asyncio
async def test_load_missing_returns_none(ckpt_store: CheckpointStore) -> None:
    assert await ckpt_store.load("nope") is None


@pytest.mark.asyncio
async def test_version_mismatch_raises(ckpt_store: CheckpointStore) -> None:
    snap = CheckpointSnapshot(thread_id="t2", version=999)
    await ckpt_store.save(snap)
    with pytest.raises(CheckpointVersionError):
        await ckpt_store.load("t2")


@pytest.mark.asyncio
async def test_list_threads(ckpt_store: CheckpointStore) -> None:
    await ckpt_store.save(CheckpointSnapshot(thread_id="a"))
    await ckpt_store.save(CheckpointSnapshot(thread_id="b"))
    threads = await ckpt_store.list_threads()
    assert set(threads) == {"a", "b"}


def test_new_idempotency_key_is_unique() -> None:
    a = new_idempotency_key("t1", "call_x")
    b = new_idempotency_key("t1", "call_x")
    assert a != b
    assert a.startswith("t1:call_x:")
