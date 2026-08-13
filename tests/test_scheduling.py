"""Tests for the shared scheduling primitives in ``soupy.scheduling``.

These cover the edge cases that used to crash the cogs' hand-rolled copies:
an end hour of 24, an inverted or degenerate window, a DST transition, and a
state file holding something other than an aware ISO timestamp.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytz

from soupy import scheduling

TZ = pytz.timezone("America/Los_Angeles")
A_DAY = date(2026, 8, 1)

# US spring-forward in 2026: 02:00 local does not exist, so the day is 23h long.
DST_FORWARD_DAY = date(2026, 3, 8)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_window_bounds_returns_wall_clock_hours():
    start, end = scheduling.window_bounds(TZ, A_DAY, 6, 20)
    assert (start.hour, start.minute) == (6, 0)
    assert (end.hour, end.minute) == (20, 0)
    assert start.tzinfo is not None and end.tzinfo is not None


def test_window_bounds_accepts_end_hour_24():
    # datetime.replace(hour=24) raises ValueError; 24 must mean next midnight.
    _, end = scheduling.window_bounds(TZ, A_DAY, 6, 24)
    assert end == TZ.localize(datetime(2026, 8, 2, 0, 0))


def test_window_bounds_keeps_wall_clock_across_dst():
    # A 20-hour *absolute* offset from midnight would land at 21:00 on a
    # spring-forward day. The window is defined in wall-clock terms.
    start, end = scheduling.window_bounds(TZ, DST_FORWARD_DAY, 6, 20)
    assert (start.hour, end.hour) == (6, 20)
    # ...and the real elapsed time is still 14h, since the gap is before 06:00.
    assert end - start == timedelta(hours=14)


def test_full_day_window_on_dst_day_is_23_hours():
    start, end = scheduling.window_bounds(TZ, DST_FORWARD_DAY, 0, 24)
    assert end - start == timedelta(hours=23)


def test_random_time_lands_inside_the_window():
    start, end = scheduling.window_bounds(TZ, A_DAY, 6, 20)
    for _ in range(200):
        picked = scheduling.random_time_in_window(TZ, A_DAY, 6, 20)
        assert start <= picked < end


def test_random_time_survives_a_degenerate_window():
    # randrange(0) would raise; a zero-width window must degrade, not crash.
    picked = scheduling.random_time_in_window(TZ, A_DAY, 9, 9)
    assert picked == scheduling.window_bounds(TZ, A_DAY, 9, 9)[0]


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def test_parse_aware_round_trips_an_aware_timestamp():
    original = TZ.localize(datetime(2026, 8, 1, 14, 23))
    assert scheduling.parse_aware(original.isoformat(), TZ) == original


def test_parse_aware_localizes_a_naive_timestamp():
    # A naive value parses fine, then explodes on the first aware comparison.
    parsed = scheduling.parse_aware("2026-08-01T14:23:00", TZ)
    assert parsed is not None and parsed.tzinfo is not None
    assert parsed < datetime.now(TZ) or parsed > datetime.now(TZ)  # comparable


def test_parse_aware_rejects_non_strings():
    # fromisoformat(12345) raises TypeError, which a ValueError guard misses.
    assert scheduling.parse_aware(12345, TZ) is None
    assert scheduling.parse_aware(None, TZ) is None
    assert scheduling.parse_aware({"scheduled_at": "x"}, TZ) is None


def test_parse_aware_rejects_garbage_strings():
    assert scheduling.parse_aware("not a timestamp", TZ) is None
    assert scheduling.parse_aware("", TZ) is None


# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------


def test_state_round_trip(tmp_path):
    path = str(tmp_path / "nested" / "state.json")
    assert scheduling.save_json_state(path, {"a": 1}) is True
    assert scheduling.load_json_state(path) == {"a": 1}


def test_load_missing_or_corrupt_state_returns_empty(tmp_path):
    assert scheduling.load_json_state(str(tmp_path / "absent.json")) == {}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert scheduling.load_json_state(str(corrupt)) == {}

    not_a_dict = tmp_path / "list.json"
    not_a_dict.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert scheduling.load_json_state(str(not_a_dict)) == {}


def test_save_reports_failure_instead_of_swallowing_it(tmp_path):
    # A directory where the file should be: the write can never succeed, and
    # the caller has to be able to tell.
    path = tmp_path / "state.json"
    path.mkdir()
    assert scheduling.save_json_state(str(path), {"a": 1}) is False


def test_save_serializes_datetimes(tmp_path):
    path = str(tmp_path / "state.json")
    stamp = TZ.localize(datetime(2026, 8, 1, 9, 0))
    assert scheduling.save_json_state(path, {"when": stamp}) is True
    assert scheduling.load_json_state(path)["when"] == str(stamp)
