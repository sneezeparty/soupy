"""Time awareness for chat replies: what "now" is, and how long ago things were said.

The chat prompt used to carry no clock, channel history arrived as one unbroken
conversation, and retrieved memory had either no date or a raw UTC timestamp.
In the archive, four replies in ten were written over history that reached back
six hours or more, read as if it were still going on.

Everything here is a pure function of the datetimes passed in, so it is tested
without a clock; :func:`now_local` is the one seam that reads the real time.

Ages are spelled out in words ("yesterday evening", "about 5 months ago") on
purpose. The 12b chat model is unreliable at date arithmetic, and a phrase gives
it nothing raw to copy into a reply.

Callers:

* ``process_chat_message`` — :func:`time_rules` at the end of the system prompt,
  :func:`label_history` over the recent messages, :func:`quiet_note` in the task
  fence.
* ``soupy_database.rag`` — :func:`describe_age` on retrieved conversations, from
  the message ids already stored (:func:`snowflake_time`), so no reindex.
* ``soupy_database.profile_document`` — :func:`describe_item_age` on profile and
  self-memory items in the chat render.

The whole feature is one runtime flag, ``time_awareness`` (see
``soupy_database.runtime_flags``). Off, every caller falls back to what it did
before.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone, tzinfo
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# A silence this long splits channel history into separate stretches, and makes
# the message being answered "new" rather than part of what came before.
HISTORY_GAP = timedelta(hours=1)

# What a leaked history marker looks like in a reply (for ``clean_response``).
HISTORY_LABEL_RE = re.compile(r"\[chat from [^\]\n]{1,60}\]", re.IGNORECASE)

_DISCORD_EPOCH_MS = 1420070400000

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _zone(name: str) -> tzinfo:
    if not name:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except (KeyError, ValueError, OSError):
        # Cached per name, so a bad TIMEZONE warns once rather than on every reply.
        logger.warning("timeaware: unknown TIMEZONE %r, using UTC", name)
        return timezone.utc


def local_tz() -> tzinfo:
    """The configured ``TIMEZONE``, or UTC when it is blank or unknown."""
    from soupy.settings import settings

    return _zone((settings.timezone or "").strip())


def now_local() -> datetime:
    """Current time in the configured timezone. A seam for tests."""
    return datetime.now(local_tz())


def snowflake_time(message_id: Any) -> Optional[datetime]:
    """When a Discord message id was minted (UTC), or None for an id that isn't a snowflake."""
    try:
        ms = (int(message_id) >> 22) + _DISCORD_EPOCH_MS
    except (TypeError, ValueError):
        return None
    if ms <= _DISCORD_EPOCH_MS:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _aware(dt: datetime) -> datetime:
    # discord.py and the archive are both UTC; a naive datetime from either is UTC too.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Phrases
# ---------------------------------------------------------------------------


def clock_time(dt: datetime) -> str:
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'am' if dt.hour < 12 else 'pm'}"


def part_of_day(hour: int) -> str:
    if hour < 5:
        return "middle of the night"
    if hour < 8:
        return "early morning"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def describe_now(now: datetime) -> str:
    """``tuesday, october 6, 2026, 6:14 am PDT (early morning)``"""
    zone = now.tzname() or ""
    if zone[:1] in ("+", "-"):
        zone = f"UTC{zone}"
    return (
        f"{_WEEKDAYS[now.weekday()]}, {_MONTHS[now.month - 1]} {now.day}, {now.year}, "
        f"{clock_time(now)}{' ' + zone if zone else ''} ({part_of_day(now.hour)})"
    )


def describe_age_days(days: int, year: Optional[int] = None) -> str:
    """How long ago, to the day. ``year`` is appended once the age is about a year or more."""
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 14:
        return "about a week ago"
    if days < 56:
        return f"about {round(days / 7)} weeks ago"
    if days < 335:
        return f"about {max(2, round(days / 30.44))} months ago"
    years = days / 365.25
    if years < 1.25:
        phrase = "about a year ago"
    elif years < 1.75:
        phrase = "about a year and a half ago"
    else:
        phrase = f"about {round(years)} years ago"
    return f"{phrase}, in {year}" if year else phrase


def _earlier_today(hour: int) -> str:
    if hour < 5:
        return "the early hours of this morning"
    if hour < 8:
        return "early this morning"
    if hour < 12:
        return "this morning"
    if hour < 17:
        return "this afternoon"
    return "this evening"


def _yesterday(hour: int) -> str:
    if hour < 5:
        return "the early hours of yesterday morning"
    if hour < 12:
        return "yesterday morning"
    if hour < 17:
        return "yesterday afternoon"
    if hour < 21:
        return "yesterday evening"
    return "last night"


def describe_age(then: datetime, now: datetime, *, with_year: bool = False) -> str:
    """How long before ``now`` something happened, on ``now``'s calendar.

    Under six hours it counts elapsed time, because "about 2 hours ago" is what a
    person says at 1 am about 11 pm. Past that it goes by the local calendar day.
    """
    then = _aware(then).astimezone(now.tzinfo)
    seconds = (now - then).total_seconds()
    if seconds < 10 * 60:
        return "the last few minutes"
    if seconds < 57.5 * 60:
        return f"about {5 * round(seconds / 300)} minutes ago"
    hours = round(seconds / 3600)
    if seconds < 6 * 3600:
        return "about an hour ago" if hours <= 1 else f"about {hours} hours ago"
    days = (now.date() - then.date()).days
    if days <= 0:
        return _earlier_today(then.hour)
    if days == 1:
        return _yesterday(then.hour)
    if days < 7:
        return f"{days} days ago, on {_WEEKDAYS[then.weekday()]}"
    return describe_age_days(days, then.year if with_year else None)


def describe_item_age(item_date: date, today: date) -> str:
    """How long ago a profile item came up.

    Item dates are the UTC date of the message, which is a day ahead of the local
    date for anything said in the evening — so under a week this stays vague
    rather than claim "today" or "yesterday".
    """
    days = (today - item_date).days
    if days <= 1:
        return "in the last day or two"
    if days < 7:
        return "a few days ago"
    return describe_age_days(days)


def describe_gap(gap: timedelta) -> str:
    """A length of silence: ``about 14 hours``, ``about 3 days``."""
    hours = gap.total_seconds() / 3600
    if hours < 1.5:
        return "about an hour"
    if hours < 23.5:
        return f"about {round(hours)} hours"
    days = hours / 24
    if days < 1.5:
        return "about a day"
    if days < 14:
        return f"about {round(days)} days"
    if days < 56:
        return f"about {round(days / 7)} weeks"
    if days < 335:
        return f"about {max(2, round(days / 30.44))} months"
    years = days / 365.25
    return "about a year" if years < 1.5 else f"about {round(years)} years"


# ---------------------------------------------------------------------------
# Prompt pieces
# ---------------------------------------------------------------------------


def time_rules(now: datetime) -> str:
    """The TIME block for the end of the chat system prompt.

    It goes last so everything before it stays byte-identical from reply to reply.
    It sits in the system prompt rather than next to the message being answered
    because a clock beside the task gets announced; here it is background.
    """
    return (
        "\n\nTIME:\n"
        f"Right now it is {describe_now(now)}. That is your own clock; people in the chat may be in other time zones.\n"
        "- Treat it like glancing at a clock. Use it when it matters: someone asks the time or the date, says good "
        "morning or good night, talks about today, tonight, tomorrow, yesterday or the weekend, or something has to "
        "be placed in time.\n"
        "- Do not bring up the time or the date on your own, and do not remark on what hour it is for anyone else.\n"
        "- Chat history may carry markers like [chat from yesterday evening]. Each one says when the messages under "
        "it were sent. Chat under an older marker is finished: it is background, so do not answer it or carry it on "
        "as if it were happening now.\n"
        "- The memory block may say how long ago something was said. Old memories may no longer be true, and when "
        "two disagree the more recent one wins.\n"
        "- Never write a [chat from ...] marker yourself.\n"
    )


def strip_timestamps(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """History entries without their ``ts`` — the LLM API takes role and content only."""
    return [{k: v for k, v in m.items() if k != "ts"} for m in messages]


def _stamps(messages: Sequence[Dict[str, Any]]) -> Optional[List[datetime]]:
    stamps = [m.get("ts") for m in messages]
    if not stamps or not all(isinstance(s, datetime) for s in stamps):
        return None
    return [_aware(s) for s in stamps]


def label_history(messages: Sequence[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
    """Mark each stretch of channel history with how long ago it was.

    A stretch is a run of messages with no silence of ``HISTORY_GAP`` or more.
    History that is one stretch ending within ``HISTORY_GAP`` of ``now`` comes
    back unmarked: that is the live conversation, and its prompt is unchanged.

    Markers are their own ``user`` entries so they never sit inside one of
    Soupy's turns, where the model would learn to write them. The caller's
    same-role merge glues them onto the neighbouring user text.
    """
    plain = strip_timestamps(messages)
    stamps = _stamps(messages)
    if stamps is None:
        return plain
    starts = [0] + [i for i in range(1, len(stamps)) if stamps[i] - stamps[i - 1] >= HISTORY_GAP]
    if len(starts) == 1 and now - stamps[-1] < HISTORY_GAP:
        return plain
    labelled: List[Dict[str, Any]] = []
    for start, end in zip(starts, starts[1:] + [len(stamps)], strict=True):
        labelled.append({"role": "user", "content": f"[chat from {describe_age(stamps[end - 1], now)}]"})
        labelled.extend(plain[start:end])
    return labelled


def quiet_note(messages: Sequence[Dict[str, Any]], sent_at: Any) -> str:
    """One line for the task fence when the message being answered broke a silence, else ``""``."""
    stamps = _stamps(messages)
    if stamps is None or not isinstance(sent_at, datetime):
        return ""
    gap = _aware(sent_at) - stamps[-1]
    if gap < HISTORY_GAP:
        return ""
    return f"This message is new. The channel had been quiet for {describe_gap(gap)} before it.\n"
