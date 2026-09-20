"""
One-at-a-time gate for the big LM Studio calls made from the bot process.

LM Studio loads the chat model with one context window, and a single request
can use all of it (verified with ``parallel=2``: a 22k-token prompt went
through on a 28k load). So two large prompts running at once — a chat reply
landing in the middle of a profile-build pass, say — can't both fit, and
hitting the window's ceiling takes LM Studio down with it. The profile builder
deliberately sizes its calls close to the window, which is what makes this
gate necessary rather than nice-to-have.

Who takes the gate:

* ``ChatQueue`` — for the whole reply, so a profile pass can't slip in between
  the reply's own LLM calls.
* The profile builder — per LLM call, and before each one it steps aside
  (:func:`wait_for_chat_to_clear`) while a chat reply is queued, running, or
  just finished. Without that the lock alternates fairly, so a burst of
  replies each waited behind a whole pass.
* SELF.MD reflection and the musings / dailypost / bluesky / search cogs —
  around their LLM calls.

Small calls (image descriptions, URL summaries, slash-command one-liners) stay
ungated; the builder's safety margin covers them.

Gotcha: the gate is re-entrant *per task* via a ContextVar. A caller already
holding it (a chat reply that ends up in a cog helper, for instance) passes
straight through instead of deadlocking on itself.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Callable, Optional

_lock: Optional[asyncio.Lock] = None
_lock_loop: Optional[asyncio.AbstractEventLoop] = None
_held: ContextVar[bool] = ContextVar("soupy_llm_gate_held", default=False)
_last_chat_activity: float = 0.0
_chat_pending: int = 0


def _get_lock() -> asyncio.Lock:
    # One lock per event loop: the bot only ever has one, but tests call
    # asyncio.run repeatedly and a lock bound to a dead loop can't be awaited.
    global _lock, _lock_loop
    loop = asyncio.get_running_loop()
    if _lock is None or _lock_loop is not loop:
        _lock, _lock_loop = asyncio.Lock(), loop
    return _lock


@asynccontextmanager
async def llm_turn() -> AsyncIterator[None]:
    """Hold the gate for the duration of the block (no-op if this task already holds it)."""
    if _held.get():
        yield
        return
    lock = _get_lock()
    await lock.acquire()
    token = _held.set(True)
    try:
        yield
    finally:
        _held.reset(token)
        lock.release()


def gate_busy() -> bool:
    """True while someone on the current event loop holds the gate."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    return _lock is not None and _lock_loop is loop and _lock.locked()


def note_chat_activity() -> None:
    """Record that a chat reply just ran; the profile builder shrinks its passes for a while after."""
    global _last_chat_activity
    _last_chat_activity = time.monotonic()


def seconds_since_chat_activity() -> Optional[float]:
    """Seconds since the last chat reply, or None if there hasn't been one in this process."""
    if _last_chat_activity <= 0:
        return None
    return time.monotonic() - _last_chat_activity


def note_chat_queued() -> None:
    """A chat reply was queued; background work waits until :func:`note_chat_done` balances it."""
    global _chat_pending
    _chat_pending += 1


def note_chat_done() -> None:
    """A queued chat reply finished (sent, failed, or dropped)."""
    global _chat_pending
    _chat_pending = max(0, _chat_pending - 1)
    note_chat_activity()


def chat_pending() -> int:
    """Chat replies queued or in progress."""
    return _chat_pending


async def wait_for_chat_to_clear(
    *,
    grace_seconds: float,
    max_wait_seconds: float,
    poll_seconds: float = 1.0,
    tick: Optional[Callable[[], None]] = None,
    tick_seconds: float = 30.0,
) -> float:
    """Wait until no chat reply is pending and none finished in the last ``grace_seconds``.

    Returns the seconds waited. Background callers use this before taking the
    gate so replies go first. It can't pre-empt a call already in flight — an
    aborted request may keep running inside LM Studio, and then the reply would
    overlap it — so the worst case for a reply is still one call.

    ``max_wait_seconds`` bounds a stuck counter or a reply that never returns.
    Giving up is safe: the caller still queues on the gate, so nothing overlaps.
    """
    start = time.monotonic()
    last_tick = start
    while True:
        since = seconds_since_chat_activity()
        if _chat_pending <= 0 and (since is None or since >= grace_seconds):
            break
        now = time.monotonic()
        if now - start >= max_wait_seconds:
            break
        if tick is not None and now - last_tick >= tick_seconds:
            tick()
            last_tick = now
        await asyncio.sleep(poll_seconds)
    return time.monotonic() - start
