"""Tests for ``soupy.timeaware`` — the clock line, age phrases, and history markers.

All pure: every function takes the datetimes it reasons about, so nothing here
reads the real clock.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from soupy import timeaware as ta
from soupy_database import runtime_flags

LA = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 10, 6, 6, 14, tzinfo=LA)  # a Tuesday
EVENING = datetime(2026, 10, 6, 18, 0, tzinfo=LA)


def _at(*args) -> datetime:
    return datetime(*args, tzinfo=LA)


# ---------------------------------------------------------------------------
# The clock line
# ---------------------------------------------------------------------------


def test_describe_now_reads_like_a_clock():
    assert ta.describe_now(NOW) == "tuesday, october 6, 2026, 6:14 am PDT (early morning)"


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(0, 5, "12:05 am"), (9, 0, "9:00 am"), (12, 0, "12:00 pm"), (13, 30, "1:30 pm"), (23, 59, "11:59 pm")],
)
def test_clock_time_is_twelve_hour(hour, minute, expected):
    assert ta.clock_time(_at(2026, 10, 6, hour, minute)) == expected


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (2, "middle of the night"),
        (5, "early morning"),
        (8, "morning"),
        (12, "afternoon"),
        (17, "evening"),
        (21, "night"),
    ],
)
def test_part_of_day(hour, expected):
    assert ta.part_of_day(hour) == expected


def test_a_zone_with_no_abbreviation_is_shown_as_a_utc_offset():
    now = datetime(2026, 1, 1, 9, 0, tzinfo=ZoneInfo("Asia/Dubai"))
    assert ta.describe_now(now) == "thursday, january 1, 2026, 9:00 am UTC+04 (morning)"


def test_unknown_or_blank_timezone_falls_back_to_utc():
    assert ta._zone("Not/A_Zone") is timezone.utc
    assert ta._zone("") is timezone.utc
    assert ta._zone("America/Los_Angeles") == LA


# ---------------------------------------------------------------------------
# Ages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("then", "expected"),
    [
        (NOW - timedelta(minutes=3), "the last few minutes"),
        (NOW - timedelta(minutes=20), "about 20 minutes ago"),
        (NOW - timedelta(minutes=58), "about an hour ago"),
        (NOW - timedelta(hours=3), "about 3 hours ago"),
        (NOW - timedelta(hours=5, minutes=50), "about 6 hours ago"),
        (_at(2026, 10, 6, 0, 10), "the early hours of this morning"),
        (_at(2026, 10, 5, 23, 14), "last night"),
        (_at(2026, 10, 5, 18, 0), "yesterday evening"),
        (_at(2026, 10, 5, 14, 0), "yesterday afternoon"),
        (_at(2026, 10, 5, 9, 0), "yesterday morning"),
        (_at(2026, 10, 5, 3, 0), "the early hours of yesterday morning"),
        (_at(2026, 10, 3, 15, 0), "3 days ago, on saturday"),
        (_at(2026, 9, 28, 12, 0), "about a week ago"),
        (_at(2026, 9, 15, 12, 0), "about 3 weeks ago"),
        (_at(2026, 5, 6, 12, 0), "about 5 months ago"),
        (_at(2025, 10, 1, 12, 0), "about a year ago"),
        (_at(2024, 4, 6, 12, 0), "about 2 years ago"),
    ],
)
def test_describe_age(then, expected):
    assert ta.describe_age(then, NOW) == expected


@pytest.mark.parametrize(
    ("then", "expected"),
    [
        (_at(2026, 10, 6, 6, 0), "early this morning"),
        (_at(2026, 10, 6, 9, 0), "this morning"),
        (_at(2026, 10, 6, 11, 59), "this morning"),
    ],
)
def test_describe_age_earlier_the_same_day(then, expected):
    assert ta.describe_age(then, EVENING) == expected


def test_a_few_hours_ago_stays_elapsed_time_across_midnight():
    assert ta.describe_age(_at(2026, 10, 5, 23, 0), _at(2026, 10, 6, 1, 0)) == "about 2 hours ago"


def test_ages_go_by_the_local_calendar_not_the_utc_one():
    # 02:00 UTC on the 6th is 7 pm on the 5th in Los Angeles: yesterday, though the UTC date is today's.
    then = datetime(2026, 10, 6, 2, 0, tzinfo=timezone.utc)
    assert ta.describe_age(then, NOW) == "yesterday evening"
    assert ta.describe_age(then.replace(tzinfo=None), NOW) == "yesterday evening", "naive means UTC"


def test_the_day_count_survives_the_end_of_daylight_saving():
    # 25 hours apart across the 2026-11-01 fall-back, but one calendar day.
    assert ta.describe_age(_at(2026, 10, 31, 20, 0), _at(2026, 11, 1, 20, 0)) == "yesterday evening"


def test_old_ages_can_carry_their_year():
    assert ta.describe_age(_at(2024, 4, 6, 12, 0), NOW, with_year=True) == "about 2 years ago, in 2024"
    assert ta.describe_age(_at(2026, 9, 15, 12, 0), NOW, with_year=True) == "about 3 weeks ago"


def test_a_timestamp_from_the_future_reads_as_now():
    assert ta.describe_age(NOW + timedelta(minutes=5), NOW) == "the last few minutes"


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (0, "today"),
        (1, "yesterday"),
        (6, "6 days ago"),
        (7, "about a week ago"),
        (13, "about a week ago"),
        (14, "about 2 weeks ago"),
        (55, "about 8 weeks ago"),
        (56, "about 2 months ago"),
        (334, "about 11 months ago"),
        (335, "about a year ago"),
        (500, "about a year and a half ago"),
        (640, "about 2 years ago"),
    ],
)
def test_describe_age_days(days, expected):
    assert ta.describe_age_days(days) == expected


def test_item_ages_stay_vague_inside_a_week():
    """Item dates are UTC message dates, so "today" could really be yesterday (or tomorrow)."""
    today = date(2026, 10, 6)
    assert ta.describe_item_age(date(2026, 10, 7), today) == "in the last day or two"
    assert ta.describe_item_age(date(2026, 10, 6), today) == "in the last day or two"
    assert ta.describe_item_age(date(2026, 10, 5), today) == "in the last day or two"
    assert ta.describe_item_age(date(2026, 10, 3), today) == "a few days ago"
    assert ta.describe_item_age(date(2026, 9, 6), today) == "about 4 weeks ago"
    assert ta.describe_item_age(date(2024, 3, 2), today) == "about 3 years ago"


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (timedelta(hours=1), "about an hour"),
        (timedelta(hours=14), "about 14 hours"),
        (timedelta(hours=30), "about a day"),
        (timedelta(days=3), "about 3 days"),
        (timedelta(days=21), "about 3 weeks"),
        (timedelta(days=100), "about 3 months"),
        (timedelta(days=400), "about a year"),
        (timedelta(days=800), "about 2 years"),
    ],
)
def test_describe_gap(gap, expected):
    assert ta.describe_gap(gap) == expected


def test_snowflake_time_reads_the_send_time_off_a_discord_id():
    sent = datetime(2026, 9, 15, 19, 0, tzinfo=timezone.utc)
    snowflake = (int(sent.timestamp() * 1000) - 1420070400000) << 22
    assert ta.snowflake_time(snowflake) == sent
    assert ta.snowflake_time(str(snowflake)) == sent


@pytest.mark.parametrize("not_a_snowflake", [12345, 0, "abc", None])
def test_snowflake_time_rejects_ids_that_are_not_snowflakes(not_a_snowflake):
    assert ta.snowflake_time(not_a_snowflake) is None


# ---------------------------------------------------------------------------
# History markers
# ---------------------------------------------------------------------------


def _msg(role, content, ts):
    return {"role": role, "content": content, "ts": ts}


def test_live_history_is_left_alone_apart_from_dropping_the_timestamps():
    history = [
        _msg("user", "ann: hi", NOW - timedelta(minutes=9)),
        _msg("assistant", "hey", NOW - timedelta(minutes=8)),
        _msg("user", "ann: you up", NOW - timedelta(minutes=1)),
    ]
    assert ta.label_history(history, NOW) == [
        {"role": "user", "content": "ann: hi"},
        {"role": "assistant", "content": "hey"},
        {"role": "user", "content": "ann: you up"},
    ]


def test_each_stretch_of_history_is_marked_with_its_age():
    history = [
        _msg("user", "glenn: anyone watching the game", _at(2026, 10, 5, 19, 0)),
        _msg("assistant", "nah", _at(2026, 10, 5, 19, 1)),
        _msg("user", "bo: back", NOW - timedelta(hours=3)),
        _msg("user", "ann: morning", NOW - timedelta(minutes=2)),
    ]
    assert ta.label_history(history, NOW) == [
        {"role": "user", "content": "[chat from yesterday evening]"},
        {"role": "user", "content": "glenn: anyone watching the game"},
        {"role": "assistant", "content": "nah"},
        {"role": "user", "content": "[chat from about 3 hours ago]"},
        {"role": "user", "content": "bo: back"},
        {"role": "user", "content": "[chat from the last few minutes]"},
        {"role": "user", "content": "ann: morning"},
    ]


def test_history_that_is_all_old_gets_one_marker():
    history = [
        _msg("assistant", "a musing nobody answered", _at(2026, 10, 4, 10, 0)),
        _msg("user", "ann: lol", _at(2026, 10, 4, 10, 5)),
    ]
    labelled = ta.label_history(history, NOW)
    assert labelled[0] == {"role": "user", "content": "[chat from 2 days ago, on sunday]"}
    assert [m["role"] for m in labelled] == ["user", "assistant", "user"], "a marker is never one of Soupy's turns"


def test_history_without_timestamps_passes_through():
    history = [{"role": "user", "content": "ann: hi"}, _msg("user", "ann: you up", NOW)]
    assert ta.label_history(history, NOW) == [
        {"role": "user", "content": "ann: hi"},
        {"role": "user", "content": "ann: you up"},
    ]
    assert ta.label_history([], NOW) == []


def test_strip_timestamps_keeps_everything_else():
    assert ta.strip_timestamps([_msg("user", "ann: hi", NOW)]) == [{"role": "user", "content": "ann: hi"}]


def test_every_marker_matches_the_pattern_the_reply_cleaner_strips():
    for hours in (0, 0.3, 2, 9, 20, 30, 80, 24 * 10, 24 * 40, 24 * 200, 24 * 400, 24 * 1000):
        marker = ta.label_history([_msg("user", "x", NOW - timedelta(hours=hours, minutes=61))], NOW)[0]["content"]
        assert ta.HISTORY_LABEL_RE.fullmatch(marker), marker


def test_quiet_note_only_when_the_message_broke_a_silence():
    history = [_msg("user", "glenn: night all", _at(2026, 10, 5, 19, 1))]
    assert ta.quiet_note(history, NOW) == (
        "This message is new. The channel had been quiet for about 11 hours before it.\n"
    )
    assert ta.quiet_note(history, _at(2026, 10, 5, 19, 30)) == ""
    assert ta.quiet_note(history, _at(2026, 10, 5, 18, 0)) == "", "history newer than the message is not a silence"
    assert ta.quiet_note([], NOW) == ""
    assert ta.quiet_note(history, "not a datetime") == ""


def test_time_rules_state_the_clock_and_forbid_writing_markers():
    rules = ta.time_rules(NOW)
    assert rules.startswith("\n\nTIME:\nRight now it is tuesday, october 6, 2026, 6:14 am PDT (early morning).")
    assert rules.endswith("Never write a [chat from ...] marker yourself.\n")


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


@pytest.fixture
def flags_file(tmp_path, monkeypatch):
    path = tmp_path / "runtime_flags.json"
    monkeypatch.setattr(runtime_flags, "flags_path", lambda: path)
    monkeypatch.setitem(runtime_flags._cache, "data", None)
    monkeypatch.setitem(runtime_flags._cache, "mtime", 0.0)
    return path


def test_time_awareness_is_on_until_switched_off(flags_file):
    assert runtime_flags.is_time_awareness_enabled(), "no flags file yet"
    flags_file.write_text('{"rag_enabled": true}', encoding="utf-8")
    assert runtime_flags.is_time_awareness_enabled(), "a flags file written before the feature existed"
    runtime_flags.write_runtime_flags({"time_awareness": False})
    assert not runtime_flags.is_time_awareness_enabled()
    assert runtime_flags.read_runtime_flags()["rag_enabled"] is True, "other flags are untouched"
    runtime_flags.write_runtime_flags({"time_awareness": True})
    assert runtime_flags.is_time_awareness_enabled()
