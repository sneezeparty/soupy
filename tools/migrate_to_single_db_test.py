"""
Tests for tools/migrate_to_single_db.py.

These build synthetic per-guild SQLite fixtures, run the migration, and assert:
  - Row counts match per (table, guild_id).
  - Re-running the migration is idempotent (no duplicate rows, exit 0).
  - --dry-run does not touch the target file.
  - --verify catches a deliberately-injected mismatch.
  - rag_chunks WHERE guild_id = ? is index-driven (per the perf constraint).
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pytest

# Make tools/ importable when pytest is invoked from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import migrate_to_single_db as m  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _build_full_guild_db(path: Path, guild_id: int, n_messages: int) -> None:
    """Build a per-guild DB with every table populated, mirroring the live schema."""
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE channels (
            channel_id INTEGER PRIMARY KEY,
            channel_name TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            username TEXT NOT NULL,
            nickname TEXT,
            user_id INTEGER NOT NULL,
            message_content TEXT,
            channel_id INTEGER NOT NULL,
            channel_name TEXT NOT NULL,
            image_description TEXT,
            url_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
        );
        CREATE TABLE scan_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_scan_time TIMESTAMP NOT NULL,
            scan_type TEXT NOT NULL,
            messages_scanned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE guild_metadata (
            guild_id INTEGER PRIMARY KEY,
            guild_name TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            archive_scan_interval_minutes INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE rag_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_message_id INTEGER NOT NULL,
            last_message_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL,
            channel_name TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(first_message_id, last_message_id)
        );
        CREATE TABLE user_profile_summaries (
            user_id INTEGER PRIMARY KEY,
            nickname_hint TEXT,
            summary TEXT NOT NULL,
            source_message_count INTEGER NOT NULL DEFAULT 0,
            source_max_message_id INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_used TEXT,
            structured_json TEXT
        );
        CREATE TABLE profile_batch_jobs (
            guild_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'idle',
            user_ids_json TEXT NOT NULL DEFAULT '[]',
            next_index INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL DEFAULT 0,
            stats_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE profile_job_log_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id INTEGER NOT NULL,
            line TEXT NOT NULL
        );
        CREATE TABLE self_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            section TEXT NOT NULL DEFAULT '',
            chunk_text TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cur.execute(
        "INSERT INTO channels VALUES (?, ?, CURRENT_TIMESTAMP)",
        (1000 + guild_id, f"general-{guild_id}"),
    )
    cur.execute(
        "INSERT INTO channels VALUES (?, ?, CURRENT_TIMESTAMP)",
        (2000 + guild_id, f"random-{guild_id}"),
    )
    cur.execute(
        "INSERT INTO guild_metadata VALUES (?, ?, CURRENT_TIMESTAMP, 60)",
        (guild_id, f"Guild {guild_id}"),
    )

    for i in range(n_messages):
        cur.execute(
            """
            INSERT INTO messages (
                message_id, date, time, username, nickname, user_id,
                message_content, channel_id, channel_name,
                image_description, url_summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                guild_id * 1_000_000 + i,
                "2025-01-01", "00:00:00",
                f"user{i % 4}", f"nick{i % 4}", 7000 + (i % 4),
                f"hello from guild {guild_id} msg {i}",
                1000 + guild_id, f"general-{guild_id}",
                None, None,
            ),
        )

    cur.execute(
        "INSERT INTO scan_metadata (last_scan_time, scan_type, messages_scanned) VALUES (?, ?, ?)",
        ("2025-01-01T00:00:00", "initial", n_messages),
    )

    # rag_chunks: tiny synthetic 4-d embeddings packed as 4 floats == 16 bytes.
    import struct
    for i in range(min(5, n_messages)):
        emb = struct.pack("4f", 0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i)
        cur.execute(
            """
            INSERT INTO rag_chunks (
                first_message_id, last_message_id, channel_id, channel_name,
                chunk_text, embedding_dim, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                guild_id * 1_000_000 + i, guild_id * 1_000_000 + i,
                1000 + guild_id, f"general-{guild_id}",
                f"chunk text {i}", 4, emb,
            ),
        )

    cur.execute(
        """
        INSERT INTO user_profile_summaries
            (user_id, nickname_hint, summary, source_message_count, model_used)
        VALUES (?, ?, ?, ?, ?)
        """,
        (7000, "alice", f"Profile for guild {guild_id}", n_messages, "test-model"),
    )

    cur.execute(
        """
        INSERT INTO profile_batch_jobs
            (guild_id, status, user_ids_json, next_index, total)
        VALUES (?, 'idle', '[]', 0, 0)
        """,
        (guild_id,),
    )

    cur.execute(
        "INSERT INTO profile_job_log_lines (guild_id, line) VALUES (?, ?)",
        (guild_id, f"log line for guild {guild_id}"),
    )

    import struct
    emb = struct.pack("4f", 0.5, 0.5, 0.5, 0.5)
    cur.execute(
        """
        INSERT INTO self_chunks (source, section, chunk_text, embedding_dim, embedding)
        VALUES ('full', 'beliefs', ?, 4, ?)
        """,
        (f"self-knowledge for guild {guild_id}", emb),
    )

    conn.commit()
    conn.close()


def _build_minimal_guild_db(path: Path, guild_id: int) -> None:
    """A guild DB that is missing self_chunks and profile_batch_jobs.

    Mirrors a real-world install that pre-dates those features.
    """
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE channels (
            channel_id INTEGER PRIMARY KEY,
            channel_name TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            username TEXT NOT NULL,
            nickname TEXT,
            user_id INTEGER NOT NULL,
            message_content TEXT,
            channel_id INTEGER NOT NULL,
            channel_name TEXT NOT NULL,
            image_description TEXT,
            url_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE guild_metadata (
            guild_id INTEGER PRIMARY KEY,
            guild_name TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            archive_scan_interval_minutes INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    cur.execute(
        "INSERT INTO channels VALUES (?, ?, CURRENT_TIMESTAMP)",
        (3000 + guild_id, f"announce-{guild_id}"),
    )
    cur.execute(
        "INSERT INTO guild_metadata VALUES (?, ?, CURRENT_TIMESTAMP, 0)",
        (guild_id, f"Minimal Guild {guild_id}"),
    )
    cur.execute(
        """
        INSERT INTO messages (message_id, date, time, username, user_id,
                              channel_id, channel_name, message_content)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            guild_id * 1_000_000 + 1,
            "2025-02-02", "01:01:01",
            "bob", 9000, 3000 + guild_id, f"announce-{guild_id}",
            "minimal guild message",
        ),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def fixture_dirs(tmp_path: Path):
    """Return (source_dir, target_path)."""
    src = tmp_path / "source"
    src.mkdir()
    _build_full_guild_db(src / "guild_111.db", 111, n_messages=20)
    _build_minimal_guild_db(src / "guild_222.db", 222)
    # A non-matching file that must be ignored by the discovery pass.
    (src / "ignore-me.db").write_bytes(b"not a real db")
    target = tmp_path / "soupy.db"
    return src, target


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_discover_skips_non_guild_files(fixture_dirs):
    src, _ = fixture_dirs
    found = m.discover_guild_dbs(src)
    assert [gid for gid, _ in found] == [111, 222]


def test_dry_run_does_not_touch_target(fixture_dirs, caplog):
    src, target = fixture_dirs
    assert not target.exists()
    rc = m.main(["--source", str(src), "--target", str(target), "--dry-run"])
    assert rc == 0
    assert not target.exists(), "target file must not be created in --dry-run"


def test_real_migration_row_counts(fixture_dirs):
    src, target = fixture_dirs
    rc = m.main(["--source", str(src), "--target", str(target)])
    assert rc == 0
    assert target.exists()

    conn = sqlite3.connect(str(target))
    try:
        # Guild 111 (full schema, 20 messages, 5 rag chunks, 1 self chunk, etc.)
        for table, expected in [
            ("channels", 2),
            ("messages", 20),
            ("scan_metadata", 1),
            ("guild_metadata", 1),
            ("rag_chunks", 5),
            ("user_profile_summaries", 1),
            ("profile_batch_jobs", 1),
            ("profile_job_log_lines", 1),
            ("self_chunks", 1),
        ]:
            n = conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE guild_id = ?', (111,)
            ).fetchone()[0]
            assert n == expected, f"guild 111 / {table}: expected {expected}, got {n}"

        # Guild 222 (minimal schema — only 3 tables present)
        n_msgs = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE guild_id = ?", (222,)
        ).fetchone()[0]
        assert n_msgs == 1
        n_chan = conn.execute(
            "SELECT COUNT(*) FROM channels WHERE guild_id = ?", (222,)
        ).fetchone()[0]
        assert n_chan == 1
        n_meta = conn.execute(
            "SELECT COUNT(*) FROM guild_metadata WHERE guild_id = ?", (222,)
        ).fetchone()[0]
        assert n_meta == 1
        # No self_chunks rows for guild 222.
        n_self = conn.execute(
            "SELECT COUNT(*) FROM self_chunks WHERE guild_id = ?", (222,)
        ).fetchone()[0]
        assert n_self == 0
    finally:
        conn.close()


def test_migration_is_idempotent(fixture_dirs):
    src, target = fixture_dirs
    assert m.main(["--source", str(src), "--target", str(target)]) == 0

    conn = sqlite3.connect(str(target))
    try:
        before = {
            t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            for t in (
                "channels", "messages", "scan_metadata", "guild_metadata",
                "rag_chunks", "user_profile_summaries", "profile_batch_jobs",
                "profile_job_log_lines", "self_chunks",
            )
        }
    finally:
        conn.close()

    # Re-run.  Must succeed and must not duplicate rows.
    assert m.main(["--source", str(src), "--target", str(target)]) == 0

    conn = sqlite3.connect(str(target))
    try:
        after = {
            t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            for t in before
        }
    finally:
        conn.close()
    assert before == after, f"row counts changed after re-run: before={before} after={after}"


def test_verify_passes_after_clean_migration(fixture_dirs):
    src, target = fixture_dirs
    rc = m.main(["--source", str(src), "--target", str(target), "--verify"])
    assert rc == 0


def test_verify_catches_injected_mismatch(fixture_dirs):
    src, target = fixture_dirs
    assert m.main(["--source", str(src), "--target", str(target)]) == 0

    # Corrupt the merged DB by deleting a known guild's messages.
    conn = sqlite3.connect(str(target))
    try:
        conn.execute("DELETE FROM messages WHERE guild_id = 111")
        conn.commit()
    finally:
        conn.close()

    # Direct verify_migration call — this is what users see when they pass
    # --verify after running the migration tool.
    mismatches, messages = m.verify_migration(source_dir=src, target_path=target)
    assert mismatches >= 1
    assert any("FAIL" in msg and "messages" in msg for msg in messages), messages

    # Sanity: calling main with --verify reruns the migration, which heals the
    # deleted rows via INSERT OR IGNORE, so the second verify pass succeeds.
    # That's the documented self-heal behaviour.  The load-bearing assertion is
    # the verify_migration call above.
    rc = m.main(["--source", str(src), "--target", str(target), "--verify"])
    assert rc == 0


def test_filename_overrides_guild_id_column(fixture_dirs, tmp_path):
    """guild_metadata in source has guild_id; migration must use filename's id."""
    src = tmp_path / "src2"
    src.mkdir()
    db_path = src / "guild_555.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE guild_metadata (
            guild_id INTEGER PRIMARY KEY,
            guild_name TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            archive_scan_interval_minutes INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    # Intentionally store a stale/wrong guild_id inside the file.
    conn.execute(
        "INSERT INTO guild_metadata VALUES (999, 'wrong id row', CURRENT_TIMESTAMP, 0)"
    )
    conn.commit()
    conn.close()

    target = tmp_path / "soupy.db"
    rc = m.main(["--source", str(src), "--target", str(target)])
    assert rc == 0

    conn = sqlite3.connect(str(target))
    try:
        rows = conn.execute(
            "SELECT guild_id, guild_name FROM guild_metadata"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [(555, "wrong id row")], rows


def test_rag_chunks_query_uses_guild_id_index(tmp_path):
    """Performance constraint: WHERE guild_id = ? on rag_chunks must use an index.

    Inserts 100k chunks across 10 guilds and asserts EXPLAIN QUERY PLAN reports
    an index-driven SEARCH (not a full SCAN).
    """
    target = tmp_path / "soupy.db"
    conn = sqlite3.connect(str(target))
    try:
        m.apply_schema(conn)

        # Synthetic embeddings (4 floats == 16 bytes); the test does not depend on
        # the embedding contents, just on row volume.
        import struct
        emb = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
        rows_per_guild = 10_000
        guilds = list(range(1, 11))
        cur = conn.cursor()
        cur.execute("BEGIN")
        for gid in guilds:
            params = [
                (gid, gid * 10_000_000 + i, gid * 10_000_000 + i,
                 1000 + gid, f"chan-{gid}", f"text {i}", 4, emb)
                for i in range(rows_per_guild)
            ]
            cur.executemany(
                """
                INSERT INTO rag_chunks (
                    guild_id, first_message_id, last_message_id, channel_id,
                    channel_name, chunk_text, embedding_dim, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
        conn.commit()

        n = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        assert n == rows_per_guild * len(guilds), n

        plan_rows = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM rag_chunks WHERE guild_id = ?", (5,)
        ).fetchall()
        plan_text = " | ".join(str(r) for r in plan_rows).lower()
        assert "rag_chunks" in plan_text, plan_text
        # SQLite's planner reports index-driven access as "SEARCH ... USING INDEX"
        # or "SEARCH ... USING COVERING INDEX".  A full table scan would say "SCAN
        # rag_chunks" with no USING INDEX.  The assertion below catches that case.
        assert "using index" in plan_text, (
            f"rag_chunks WHERE guild_id = ? must use an index; plan was: {plan_text}"
        )

        # Also verify the (guild_id, channel_id) compound index is reachable for
        # the per-channel filter pattern used by index_new_messages().
        plan2 = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM rag_chunks "
            "WHERE guild_id = ? AND channel_id = ?",
            (5, 1005),
        ).fetchall()
        plan2_text = " | ".join(str(r) for r in plan2).lower()
        assert "using index" in plan2_text, plan2_text
    finally:
        conn.close()


def test_failure_in_one_guild_does_not_abort_others(tmp_path, caplog):
    """If one guild file is corrupt, the migration should log it and continue.

    The CLI exits 1 in that case, but every other guild's data still lands.
    """
    src = tmp_path / "src"
    src.mkdir()
    _build_full_guild_db(src / "guild_700.db", 700, n_messages=5)
    # Corrupt file: the SQLite header check will fail when we try to open it.
    (src / "guild_701.db").write_bytes(b"not a sqlite database, just garbage bytes" * 10)
    _build_full_guild_db(src / "guild_702.db", 702, n_messages=7)

    target = tmp_path / "soupy.db"
    with caplog.at_level(logging.INFO):
        rc = m.main(["--source", str(src), "--target", str(target)])
    # Some failure was logged; rc must be 1.
    assert rc == 1
    # But guild 700 and 702 made it through.
    conn = sqlite3.connect(str(target))
    try:
        n700 = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE guild_id = 700"
        ).fetchone()[0]
        n702 = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE guild_id = 702"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n700 == 5
    assert n702 == 7
