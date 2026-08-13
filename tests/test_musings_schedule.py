"""Tests for the musings cog's once-a-day scheduler.

The loop runs every 60 seconds and is the only thing standing between an LLM
outage and ~800 failed generation attempts in a day, so the cases here are
mostly about what happens when something goes wrong at the scheduled minute.

The cog is built with ``object.__new__`` and hand-populated rather than
constructed for real — instantiating it would start a discord.py task loop and
a warmup coroutine. ``_loop.coro`` is the undecorated function underneath
``@tasks.loop``. pytest-asyncio isn't a dependency, so async paths are driven
with ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
import pytz

from soupy.cogs import musings

TZ = pytz.timezone("America/Los_Angeles")
DAY = datetime(2026, 8, 1)


@pytest.fixture
def cog(tmp_path, monkeypatch):
    """A MusingsCog wired to a temp state file, a fake clock, and a stub channel."""
    monkeypatch.setattr(musings, "MUSING_DAILY_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        musings,
        "settings",
        SimpleNamespace(
            musing_enabled=True,
            musing_channel_id=123,
            musing_hour_min=6,
            musing_hour_max=20,
        ),
    )

    c = object.__new__(musings.MusingsCog)
    c.timezone = TZ
    c._daily_state_cache = None
    c._warned_bad_window = False

    channel = SimpleNamespace(guild=SimpleNamespace(id=999))
    c.bot = SimpleNamespace(get_channel=lambda _id: channel, _timer_state=None)

    c.posted = []

    async def fake_run_and_post(ch, guild_id, mode, trigger_label):
        c.posted.append((guild_id, mode, trigger_label))
        return "a thought"

    c._run_and_post = fake_run_and_post
    c._now_local = lambda: TZ.localize(DAY.replace(hour=12))
    return c


def tick(cog):
    """Run one iteration of the background loop."""
    asyncio.run(musings.MusingsCog._loop.coro(cog))


def at(hour, minute=0, second=0, day=1):
    return TZ.localize(DAY.replace(day=day, hour=hour, minute=minute, second=second))


# ---------------------------------------------------------------------------
# Hour-window config
# ---------------------------------------------------------------------------


def test_end_hour_24_does_not_raise(cog):
    cog._hour_window = lambda: (6, 24)
    start, end = cog._window_bounds(DAY.date())
    assert end == at(0, day=2)


def test_inverted_hour_pair_falls_back_to_defaults(cog, monkeypatch, caplog):
    # Setting MUSING_HOUR_MAX below MUSING_HOUR_MIN used to raise inside the
    # loop on every tick, silently killing musings until someone read the log.
    monkeypatch.setattr(
        musings,
        "settings",
        SimpleNamespace(
            musing_enabled=True, musing_channel_id=123, musing_hour_min=20, musing_hour_max=6
        ),
    )
    with caplog.at_level("WARNING", logger="soupy.cogs.musings"):
        assert cog._hour_window() == (musings.DEFAULT_HOUR_MIN, musings.DEFAULT_HOUR_MAX)
        # ...and repeated reads must not spam: this runs several times a minute.
        for _ in range(5):
            cog._hour_window()
    assert len([r for r in caplog.records if "must be greater than" in r.message]) == 1


# ---------------------------------------------------------------------------
# Corrupt state
# ---------------------------------------------------------------------------


def test_naive_scheduled_at_is_localized_not_fatal(cog):
    # A hand-edited state file with no UTC offset parses fine, then used to
    # raise TypeError on the first aware/naive comparison.
    state = {"scheduled_date": "2026-08-01", "scheduled_at": "2026-08-01T14:23:00"}
    scheduled = cog._scheduled_time_for(state, DAY.date())
    assert scheduled == at(14, 23)
    assert scheduled > cog._now_local() or scheduled < cog._now_local()


def test_non_string_scheduled_at_rerolls(cog):
    # fromisoformat(12345) raises TypeError, which a ValueError guard misses.
    state = {"scheduled_date": "2026-08-01", "scheduled_at": 12345}
    scheduled = cog._scheduled_time_for(state, DAY.date())
    start, end = cog._window_bounds(DAY.date())
    assert start <= scheduled < end


def test_yesterdays_schedule_is_not_reused(cog):
    state = {"scheduled_date": "2026-07-31", "scheduled_at": at(9, day=1).isoformat()}
    cog._scheduled_time_for(state, DAY.date())
    assert state["scheduled_date"] == "2026-08-01"


# ---------------------------------------------------------------------------
# Loop behaviour
# ---------------------------------------------------------------------------


def test_posts_once_then_never_again_that_day(cog):
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(11).isoformat()}
    tick(cog)
    assert len(cog.posted) == 1
    for _ in range(5):
        tick(cog)
    assert len(cog.posted) == 1
    assert cog._daily_state()["last_handled_status"] == "posted"


def test_generation_failure_still_burns_the_day(cog):
    # The regression this guards: without the finally, an LLM outage left
    # last_handled_date unwritten and the 60s loop retried until the window
    # closed — hundreds of failed calls.
    async def boom(*_a, **_kw):
        raise RuntimeError("LM Studio is down")

    cog._run_and_post = boom
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(11).isoformat()}

    tick(cog)
    state = cog._daily_state()
    assert state["last_handled_date"] == "2026-08-01"
    assert state["last_handled_status"] == "error"

    # And the next tick does not try again.
    calls = []
    cog._run_and_post = lambda *a, **k: calls.append(1)
    tick(cog)
    assert calls == []


def test_channel_outside_a_guild_burns_the_day_without_crashing(cog):
    cog.bot = SimpleNamespace(get_channel=lambda _id: SimpleNamespace(guild=None), _timer_state=None)
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(11).isoformat()}
    tick(cog)
    assert cog._daily_state()["last_handled_status"] == "not_a_guild_channel"


def test_musing_due_at_the_window_edge_still_fires(cog):
    # Scheduled 19:59, tick lands just after the 20:00 close. The window check
    # used to run before the schedule was even read, so this was dropped.
    cog._daily_state_cache = {
        "scheduled_date": "2026-08-01",
        "scheduled_at": at(19, 59).isoformat(),
    }
    cog._now_local = lambda: at(20, 0, 4)
    tick(cog)
    assert len(cog.posted) == 1


def test_boot_long_after_the_window_skips_the_day(cog):
    cog._now_local = lambda: at(21, 30)
    tick(cog)
    assert cog.posted == []
    assert cog._daily_state()["last_handled_status"] == "skipped_past_window"


def test_not_yet_time_does_nothing(cog):
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(18).isoformat()}
    tick(cog)
    assert cog.posted == []
    assert "last_handled_date" not in cog._daily_state()


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def test_state_is_read_from_disk_once(cog, monkeypatch):
    reads = []
    real_load = musings.load_json_state
    monkeypatch.setattr(
        musings, "load_json_state", lambda p: (reads.append(p), real_load(p))[1]
    )
    for _ in range(4):
        tick(cog)
    assert len(reads) == 1


def test_unwritable_state_still_stops_the_loop_reposting(cog, monkeypatch):
    # data/ read-only or the disk full. The old code re-read {} from disk every
    # tick, so it posted a musing every 60 seconds indefinitely.
    monkeypatch.setattr(musings, "save_json_state", lambda *_a, **_kw: False)
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(11).isoformat()}
    for _ in range(10):
        tick(cog)
    assert len(cog.posted) == 1


def test_state_survives_a_restart(cog):
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(11).isoformat()}
    tick(cog)

    cog._daily_state_cache = None  # Simulate a bot restart.
    cog.posted.clear()
    tick(cog)
    assert cog.posted == []


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_slot_is_populated(cog):
    timers = {"musings": {"last_run": None, "next_run": None, "interval": None, "enabled": False}}
    cog.bot = SimpleNamespace(
        get_channel=lambda _id: SimpleNamespace(guild=SimpleNamespace(id=999)),
        _timer_state=timers,
    )
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(18).isoformat()}

    tick(cog)  # Not due yet — should advertise the upcoming run.
    assert timers["musings"]["enabled"] is True
    assert timers["musings"]["interval"] == "1x/day, 06:00-20:00 local"
    assert timers["musings"]["next_run"] == at(18).astimezone(pytz.UTC).isoformat()
    assert timers["musings"]["last_run"] is None

    cog._now_local = lambda: at(18, 1)
    tick(cog)  # Now it has fired.
    assert timers["musings"]["next_run"] is None
    assert timers["musings"]["last_run"] is not None
    assert timers["musings"]["last_status"] == "posted"


def test_dashboard_write_never_breaks_the_loop(cog):
    cog.bot = SimpleNamespace(
        get_channel=lambda _id: SimpleNamespace(guild=SimpleNamespace(id=999)),
        _timer_state={"musings": "not a dict"},
    )
    cog._daily_state_cache = {"scheduled_date": "2026-08-01", "scheduled_at": at(11).isoformat()}
    tick(cog)
    assert len(cog.posted) == 1


def test_late_fire_grace_is_bounded(cog):
    # Just inside the grace fires; well past it does not.
    assert musings.LATE_FIRE_GRACE <= timedelta(minutes=30)
