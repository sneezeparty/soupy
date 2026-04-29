"""
Background profile batch jobs: logging, pause/resume state in SQLite, worker loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .database import get_db_path

logger = logging.getLogger(__name__)

_PROFILE_TASKS: Dict[int, asyncio.Task] = {}


def _profile_log_max_stored() -> int:
    try:
        n = int(os.getenv("PROFILE_JOB_LOG_MAX_LINES", "500"))
    except ValueError:
        n = 500
    return max(50, min(n, 5000))


def profile_job_log_append(guild_id: int, line: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    formatted = f"[{ts}] {line}"
    logger.info("[profiles guild=%s] %s", guild_id, line)
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_profile_job_schema(conn)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO profile_job_log_lines (guild_id, line) VALUES (?, ?)",
            (guild_id, formatted),
        )
        max_keep = _profile_log_max_stored()
        cur.execute(
            """
            DELETE FROM profile_job_log_lines
            WHERE guild_id = ? AND id NOT IN (
                SELECT id FROM profile_job_log_lines
                WHERE guild_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (guild_id, guild_id, max_keep),
        )
        conn.commit()
    finally:
        conn.close()


def profile_job_log_clear(guild_id: int) -> None:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_profile_job_schema(conn)
        conn.execute("DELETE FROM profile_job_log_lines WHERE guild_id = ?", (guild_id,))
        conn.commit()
    finally:
        conn.close()


def profile_job_log_lines(guild_id: int, last: int = 300) -> List[str]:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return []
    if last <= 0:
        last = _profile_log_max_stored()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_profile_job_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT line FROM profile_job_log_lines
            WHERE guild_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (guild_id, max(1, last)),
        )
        rows = [r[0] for r in cur.fetchall()]
        return list(reversed(rows))
    finally:
        conn.close()


def ensure_profile_job_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profile_batch_jobs (
            guild_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'idle',
            user_ids_json TEXT NOT NULL DEFAULT '[]',
            next_index INTEGER NOT NULL DEFAULT 0,
            total INTEGER NOT NULL DEFAULT 0,
            stats_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profile_job_log_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guild_id INTEGER NOT NULL,
            line TEXT NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_profile_job_log_guild_id ON profile_job_log_lines(guild_id, id)"
    )
    conn.commit()


def get_job_row(conn: sqlite3.Connection, guild_id: int) -> Optional[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM profile_batch_jobs WHERE guild_id = ?", (guild_id,))
    return cur.fetchone()


def upsert_job(
    conn: sqlite3.Connection,
    guild_id: int,
    *,
    status: Optional[str] = None,
    user_ids_json: Optional[str] = None,
    next_index: Optional[int] = None,
    total: Optional[int] = None,
    stats_json: Optional[str] = None,
) -> None:
    cur = conn.cursor()
    row = get_job_row(conn, guild_id)
    if row is None:
        cur.execute(
            """
            INSERT INTO profile_batch_jobs (guild_id, status, user_ids_json, next_index, total, stats_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                guild_id,
                status if status is not None else "idle",
                user_ids_json if user_ids_json is not None else "[]",
                next_index if next_index is not None else 0,
                total if total is not None else 0,
                stats_json,
            ),
        )
    else:
        parts: List[str] = []
        params: List[Any] = []
        if status is not None:
            parts.append("status = ?")
            params.append(status)
        if user_ids_json is not None:
            parts.append("user_ids_json = ?")
            params.append(user_ids_json)
        if next_index is not None:
            parts.append("next_index = ?")
            params.append(next_index)
        if total is not None:
            parts.append("total = ?")
            params.append(total)
        if stats_json is not None:
            parts.append("stats_json = ?")
            params.append(stats_json)
        if not parts:
            return
        parts.append("updated_at = CURRENT_TIMESTAMP")
        params.append(guild_id)
        cur.execute(
            f"UPDATE profile_batch_jobs SET {', '.join(parts)} WHERE guild_id = ?",
            params,
        )
    conn.commit()


def delete_job(conn: sqlite3.Connection, guild_id: int) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM profile_batch_jobs WHERE guild_id = ?", (guild_id,))
    conn.commit()


async def cancel_profile_task(guild_id: int) -> None:
    t = _PROFILE_TASKS.pop(guild_id, None)
    if t is not None and not t.done():
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass


def spawn_profile_worker(
    guild_id: int,
    worker_coro_factory: Callable[[int], Any],
) -> None:
    """Start worker if not already running for this guild."""
    existing = _PROFILE_TASKS.get(guild_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(worker_coro_factory(guild_id))
    _PROFILE_TASKS[guild_id] = task


def forget_profile_task(guild_id: int) -> None:
    _PROFILE_TASKS.pop(guild_id, None)


def profile_task_running(guild_id: int) -> bool:
    t = _PROFILE_TASKS.get(guild_id)
    return t is not None and not t.done()
