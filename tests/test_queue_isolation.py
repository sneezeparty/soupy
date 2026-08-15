"""Tests for the split between the image queue and the chat queue.

Chat replies and image generation shared a single `SDQueue` for most of the
bot's life. Since that queue has one consumer which awaits each job to
completion, a running /flux generation blocked every reply bot-wide until it
finished — and because `process_chat_message` is what opens the typing
indicator, the channel showed no sign the bot had even noticed the message.

The tests here pin the two properties that fix depends on: chat and image work
run concurrently, and chat replies are still serialized among themselves.

pytest-asyncio isn't a dependency, so the consumers are driven inside a single
`asyncio.run` per test and cancelled at the end.
"""

from __future__ import annotations

import asyncio

import pytest

import soupy_remastered_stablediffusion as bot


async def _drain(queue, timeout=2.0):
    """Run `queue.process_queue` until it goes idle, then cancel it."""
    task = asyncio.create_task(queue.process_queue())
    deadline = asyncio.get_event_loop().time() + timeout
    while queue.qsize() > 0 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # Let the in-flight item finish.
    task.cancel()
    return task


# ---------------------------------------------------------------------------
# The regression: image work must not block chat
# ---------------------------------------------------------------------------


def test_chat_runs_while_an_image_is_generating(monkeypatch):
    """The bug report: /flux in flight, message arrives, channel stays silent."""
    events = []

    async def slow_image(item):
        events.append("image_start")
        await asyncio.sleep(0.3)  # Stands in for a local mflux generation.
        events.append("image_end")

    async def fake_chat(message, image_descriptions):
        events.append("chat_done")

    monkeypatch.setattr(bot, "process_chat_message", fake_chat)

    async def scenario():
        from soupy.cogs import flux as _flux

        monkeypatch.setattr(_flux, "process_flux_image", slow_image)

        sd_q, chat_q = bot.SDQueue(), bot.ChatQueue()
        sd_task = asyncio.create_task(sd_q.process_queue())
        chat_task = asyncio.create_task(chat_q.process_queue())

        await sd_q.put({"type": "flux", "interaction": None, "description": "x"})
        await asyncio.sleep(0.05)  # Image generation is now under way.
        await chat_q.put({"type": "chat", "message": object(), "image_descriptions": []})

        await asyncio.sleep(0.5)
        sd_task.cancel()
        chat_task.cancel()

    asyncio.run(scenario())

    # The reply must land before the image finishes. On the old shared queue
    # the order was image_start, image_end, chat_done.
    assert events == ["image_start", "chat_done", "image_end"]


def test_chat_replies_are_still_serialized():
    """One reply at a time — ordering, and no LM Studio stampede."""
    concurrent = 0
    peak = 0
    order = []

    async def fake_chat(message, image_descriptions):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.05)
        order.append(message)
        concurrent -= 1

    async def scenario():
        q = bot.ChatQueue()
        for i in range(4):
            await q.put({"type": "chat", "message": i, "image_descriptions": []})
        await _drain(q)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bot, "process_chat_message", fake_chat)
        asyncio.run(scenario())

    assert peak == 1, "chat replies must not overlap"
    assert order == [0, 1, 2, 3], "replies must stay in arrival order"


def test_a_failing_reply_does_not_kill_the_consumer(monkeypatch):
    """One bad message must not silence the bot until restart."""
    handled = []

    async def fake_chat(message, image_descriptions):
        if message == "boom":
            raise RuntimeError("LM Studio is down")
        handled.append(message)

    monkeypatch.setattr(bot, "process_chat_message", fake_chat)

    async def scenario():
        q = bot.ChatQueue()
        for msg in ("boom", "after"):
            await q.put({"type": "chat", "message": msg, "image_descriptions": []})
        await _drain(q)

    asyncio.run(scenario())
    assert handled == ["after"]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_sd_queue_ignores_chat_items(monkeypatch):
    """Chat routing moved off sd_queue; a stray chat item must not be processed."""
    called = []

    async def fake_chat(message, image_descriptions):
        called.append(message)

    monkeypatch.setattr(bot, "process_chat_message", fake_chat)

    async def scenario():
        q = bot.SDQueue()
        await q.put({"type": "chat", "message": "x", "image_descriptions": []})
        await _drain(q)

    asyncio.run(scenario())
    assert called == []


def test_the_two_queues_are_independent_instances():
    sd_q, chat_q = bot.SDQueue(), bot.ChatQueue()
    assert sd_q._queue is not chat_q._queue
    assert isinstance(sd_q, bot._WorkQueue) and isinstance(chat_q, bot._WorkQueue)


# ---------------------------------------------------------------------------
# Shared _WorkQueue mechanics
# ---------------------------------------------------------------------------


def test_qsize_tracks_pending_work():
    async def scenario():
        q = bot.ChatQueue()
        assert q.qsize() == 0
        await q.put({"type": "chat"})
        await q.put({"type": "chat"})
        assert q.qsize() == 2
        await q.get()
        assert q.qsize() == 1

    asyncio.run(scenario())


def test_shutdown_drains_pending_items():
    async def scenario():
        q = bot.ChatQueue()
        for _ in range(3):
            await q.put({"type": "chat"})
        await q.initiate_shutdown()
        assert q._queue.empty()
        assert q._shutdown is True

    asyncio.run(scenario())
