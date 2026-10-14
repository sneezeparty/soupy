"""Retrieved memory says how long ago it was — when the chat path passes its clock.

Ages come from the Discord message ids already stored with every chunk and
row, so nothing here needs an index rebuild. Without a clock (time awareness
off, or any caller that isn't the chat path) the block is byte-for-byte what it
was.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from soupy_database import rag

NOW = datetime(2026, 10, 6, 6, 14, tzinfo=ZoneInfo("America/Los_Angeles"))


def _snowflake(sent: datetime) -> int:
    return (int(sent.timestamp() * 1000) - 1420070400000) << 22


# 02:00 UTC on Sep 16 is 7 pm on Sep 15 in Los Angeles — three weeks before NOW.
RECENT = _snowflake(datetime(2026, 9, 16, 2, 0, tzinfo=timezone.utc))
OLD = _snowflake(datetime(2024, 4, 6, 19, 0, tzinfo=timezone.utc))


def _hit(first_id: int, last_id: int):
    return (0.812, "[#general channel_id=5]\nAnn (user_id=1): we adopted a kitten", (first_id, last_id, "general"))


def _bundle(hits, now):
    return rag._assemble_rag_bundle("", "", hits, 5000, 2000, now=now)[0]


def test_related_conversations_say_how_long_ago_they_were():
    out = _bundle([_hit(RECENT - 1, RECENT)], NOW)
    assert out.startswith("--- related conversation 1, from about 3 weeks ago (relevance=0.812, channel #general, ")


def test_conversations_from_other_years_name_the_year():
    out = _bundle([_hit(RECENT, RECENT), _hit(OLD, OLD)], NOW)
    assert "--- related conversation 2, from about 2 years ago, in 2024 (relevance=" in out


def test_related_conversations_are_unchanged_without_a_clock():
    out = _bundle([_hit(RECENT - 1, RECENT)], None)
    assert out.startswith(f"--- related conversation 1 (relevance=0.812, channel #general, messages {RECENT - 1}-")


def test_an_id_that_is_not_a_snowflake_gets_no_age():
    assert rag._age_of_message(12345, NOW) == ""
    assert _bundle([_hit(12344, 12345)], NOW).startswith("--- related conversation 1 (relevance=0.812, ")


def test_keyword_lines_swap_the_utc_timestamp_for_an_age():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE messages (message_id INTEGER PRIMARY KEY, date TEXT, time TEXT, username TEXT, nickname TEXT, "
        "user_id INTEGER, message_content TEXT, channel_id INTEGER, channel_name TEXT, image_description TEXT, "
        "url_summary TEXT)"
    )
    conn.execute(
        "INSERT INTO messages VALUES (?, '2026-09-16', '02:00:00', 'ann', 'Ann', 1, 'we adopted a kitten', 5, "
        "'general', NULL, NULL)",
        (RECENT,),
    )
    question = "did i say anything about my kitten"
    try:
        raw = rag._keyword_supplement_block(conn, question, 1, question)
        aged = rag._keyword_supplement_block(conn, question, 1, question, now=NOW)
    finally:
        conn.close()
    assert raw.endswith(f"we adopted a kitten [message_id={RECENT}, #general, 2026-09-16 02:00:00]")
    assert aged.endswith(f"we adopted a kitten [message_id={RECENT}, #general, about 3 weeks ago]")
