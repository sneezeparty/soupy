"""
Shared scheduling primitives for the autonomous cogs.

``dailypost``, ``bluesky``, and ``musings`` all do the same two things: pick
random times inside a local-time window for "today", and persist the resulting
schedule to a small JSON file under ``data/`` so a restart doesn't reroll it.
Each cog used to carry its own copy of both. The copies drifted — two wrote
non-atomically, all three handled a bad hour pair differently — so the
primitives live here now.

Only the window math and the state-file I/O are shared. The *shape* of each
cog's schedule (one time for musings, two channel+slot events for dailypost, a
list of (time, action) pairs for bluesky) stays in that cog, because forcing
those three into one type would cost more than the duplication did.

Gotcha: these take pytz timezones (what ``settings.timezone`` yields). They
degrade gracefully on a stdlib ``zoneinfo`` tz, but the DST-correct paths
(``localize`` / ``normalize``) are pytz-only.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local-time windows
# ---------------------------------------------------------------------------


def _localize(tz, naive: datetime) -> datetime:
    """Attach ``tz`` to a naive datetime, preferring pytz's DST-aware path."""
    localize = getattr(tz, "localize", None)
    if localize is not None:
        return localize(naive)
    return naive.replace(tzinfo=tz)


def _normalize(tz, aware: datetime) -> datetime:
    """Re-resolve a tz-aware datetime after arithmetic (pytz DST correction)."""
    normalize = getattr(tz, "normalize", None)
    if normalize is not None:
        return normalize(aware)
    return aware


def _at_local_hour(tz, local_date, hour: int) -> datetime:
    """Wall-clock ``hour`` on ``local_date``, tz-aware. ``hour`` may be >= 24.

    Hours past 23 roll into the following day, so 24 means midnight at the end
    of ``local_date``. Building the result from a wall-clock hour (rather than
    adding a timedelta to midnight) is what keeps "20:00 local" meaning 20:00
    on a day containing a DST transition, where an absolute 20-hour offset
    would land at 19:00 or 21:00.
    """
    days, hour = divmod(hour, 24)
    base = local_date + timedelta(days=days)
    return _localize(tz, datetime.combine(base, datetime.min.time()).replace(hour=hour))


def window_bounds(tz, local_date, start_hour: int, end_hour: int) -> Tuple[datetime, datetime]:
    """Return the tz-aware ``[start, end)`` bounds of an hour window on ``local_date``.

    ``end_hour`` may be 24, meaning midnight at the end of the day. That case
    is the reason this exists: ``datetime.replace(hour=24)`` raises
    ``ValueError``, so every caller that let an operator configure the end of a
    window was one plausible config value away from crashing its own loop.
    """
    return _at_local_hour(tz, local_date, start_hour), _at_local_hour(tz, local_date, end_hour)


def random_time_in_window(tz, local_date, start_hour: int, end_hour: int) -> datetime:
    """Pick a uniformly random tz-aware time inside ``[start_hour, end_hour)``.

    Callers are expected to have already rejected an inverted hour pair, but
    the span is floored at one second regardless: a ``ValueError`` out of a
    scheduling helper kills the calling loop for every subsequent tick, which
    is a far worse failure than a degenerate window.
    """
    start, end = window_bounds(tz, local_date, start_hour, end_hour)
    span_seconds = max(1, int((end - start).total_seconds()))
    return _normalize(tz, start + timedelta(seconds=random.randrange(span_seconds)))


def parse_aware(raw: Any, tz) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into a tz-aware datetime, or None if unusable.

    Guards more than ``fromisoformat``'s ``ValueError``: a hand-edited or
    half-written state file can hold a number where a string belongs (which
    raises ``TypeError``), and a *naive* timestamp parses fine but then raises
    ``TypeError`` at the first comparison against an aware "now". Anything
    naive is assumed to already be in ``tz``.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return _localize(tz, parsed)
    return parsed.astimezone(tz)


# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------


def load_json_state(path: str) -> Dict:
    """Load a JSON state file. Returns ``{}`` if it's missing, corrupt, or not a dict."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_json_state(path: str, payload: Any) -> bool:
    """Atomically write a JSON state file. Returns True on success.

    Writes to a sibling ``.tmp`` and ``os.replace``s it into place so a crash
    mid-write can't leave a truncated file behind. The return value matters:
    callers that treat "written" as durable state need to know when the write
    failed, and a silently-swallowed failure here is how a once-a-day loop
    turns into a once-a-minute loop.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning("Failed to persist state file %s: %s", path, exc)
        return False
