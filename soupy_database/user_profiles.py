"""
Server-local member profiles built from archived messages (SQLite).

For each member with enough messages, a row in ``user_profile_summaries``
holds:

* ``structured_json`` — the version-2 profile document (see
  :mod:`soupy_database.profile_document`): an overview, a communication-style
  note, and dated items in sections like personality, lately, family, opinions.
* ``summary`` — the same profile as readable text with full dates (shown in the
  dashboard's profile browser).
* ``source_max_message_id`` — the last message folded into the profile. Every
  build pass saves, so this is also the resume point.

How a profile is built:

* The builder walks the member's messages oldest-first after
  ``source_max_message_id``, in passes sized to LM Studio's loaded context
  window (see :mod:`soupy_database.profile_llm`). Each pass asks the LLM for an
  edit list and applies it in code, then saves. A first build and a nightly
  update are the same loop; a first build just starts from an empty profile.
* A row that predates version 2 is copied to ``user_profile_summaries_v1`` and
  rebuilt from scratch.
* Every LLM call runs inside :func:`soupy.llm_gate.llm_turn`, so a pass never
  overlaps a chat reply.

Where it runs:

* Only inside the bot process. The dashboard's batch buttons write a job row;
  :func:`profile_jobs_loop` (started from the bot's ``on_ready``) picks it up,
  and also queues the nightly refresh. The web process never calls the LLM for
  profiles, because it can't see the bot's gate.

Cross-module:

* :func:`ensure_user_profile_schema` and :func:`_load_structured_profiles`
  are imported by ``soupy.cogs.dailypost`` for top-poster audience briefs.
* :func:`format_profile_prefix_for_rag` is called by ``soupy_database.rag``
  to put the relevant slice of a profile into a chat reply's context.

Gotchas:

* :func:`_migrate_profile_columns` silently ALTERs the table to add
  ``structured_json`` on first access — old DBs upgrade in-place without
  an explicit migration step. Removing this without a real migration
  would break every existing install.
* Per-guild SQLite path is resolved via ``soupy_database.database.get_db_path``
  — never hard-code paths here; the directory is settings-configurable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pytz

from soupy.llm_gate import llm_turn, seconds_since_chat_activity
from soupy.scheduling import load_json_state, save_json_state, window_bounds
from soupy.settings import _env_bool, _env_int, settings

from .database import get_db_path
from .profile_batch import (
    delete_job,
    ensure_profile_job_schema,
    get_job_row,
    profile_job_log_append,
    profile_job_log_clear,
    upsert_job,
)
from .profile_document import (
    SECTION_KEYS,
    EditStats,
    apply_edits,
    is_v2,
    item_count,
    new_document,
    normalize_document,
    parse_date,
    render_for_chat,
    render_for_prompt,
    render_summary,
)
from .profile_llm import (
    ContextOverflowError,
    build_system_prompt,
    build_user_prompt,
    compute_budget,
    observe_prompt_tokens,
    request_edits,
)

logger = logging.getLogger(__name__)

V1_BACKUP_TABLE = "user_profile_summaries_v1"
NIGHTLY_STATE_PATH = os.path.join("data", "profile_nightly_state.json")

_MESSAGE_BODY_MAX = 600
_IMAGE_DESCRIPTION_MAX = 160
# A pass that still fails at this many messages is skipped rather than split further.
_MIN_SPLIT_MESSAGES = 4
# A pass needs at least this much room for messages after the prompt's fixed parts.
_MIN_MESSAGE_CHARS = 1500
# Chat within this many seconds counts as "busy": passes read fewer messages so replies wait less.
_BUSY_WINDOW_SEC = 600
# A running job whose heartbeat is older than this is shown as waiting for the bot
# (never shorter than one slow LLM call, or a long pass would look like a dead bot).
_HEARTBEAT_STALE_MIN_SEC = 15 * 60
_NIGHTLY_START_WINDOW_HOURS = 3


def _migrate_profile_columns(conn: sqlite3.Connection) -> None:
    # WHY: silent in-place schema migration. Old installs only had `summary`;
    # `structured_json` was added later. Rather than ship a one-shot migration
    # script, every connection runs PRAGMA on connect and ALTERs if the
    # column is missing. The ALTER is cheap on a non-existent column and
    # idempotent if the column already exists.
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(user_profile_summaries)")
    cols = {row[1] for row in cur.fetchall()}
    if "structured_json" not in cols:
        cur.execute("ALTER TABLE user_profile_summaries ADD COLUMN structured_json TEXT")
    conn.commit()


def ensure_user_profile_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profile_summaries (
            user_id INTEGER PRIMARY KEY,
            nickname_hint TEXT,
            summary TEXT NOT NULL,
            source_message_count INTEGER NOT NULL DEFAULT 0,
            source_max_message_id INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_used TEXT,
            structured_json TEXT
        )
        """)
    conn.commit()
    _migrate_profile_columns(conn)
    ensure_profile_job_schema(conn)


def _loads_profile(raw: Optional[str]) -> Any:
    try:
        return json.loads(raw) if raw and raw.strip() else None
    except (TypeError, ValueError):
        return None


def _backup_v1_rows(conn: sqlite3.Connection, user_ids: Optional[Sequence[int]] = None) -> int:
    """Copy pre-v2 profile rows into ``user_profile_summaries_v1`` (first copy wins) before they are replaced."""
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {V1_BACKUP_TABLE} (
            user_id INTEGER PRIMARY KEY,
            nickname_hint TEXT,
            summary TEXT NOT NULL,
            source_message_count INTEGER NOT NULL DEFAULT 0,
            source_max_message_id INTEGER,
            updated_at TIMESTAMP,
            model_used TEXT,
            structured_json TEXT,
            backed_up_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    if user_ids is None:
        cur.execute("SELECT user_id, structured_json FROM user_profile_summaries")
    else:
        ids = [int(u) for u in user_ids]
        if not ids:
            return 0
        cur.execute(
            f"SELECT user_id, structured_json FROM user_profile_summaries WHERE user_id IN ({','.join('?' * len(ids))})",
            ids,
        )
    v1_ids = [int(r[0]) for r in cur.fetchall() if not is_v2(_loads_profile(r[1]))]
    if not v1_ids:
        return 0
    cur.execute(
        f"""
        INSERT OR IGNORE INTO {V1_BACKUP_TABLE} (
            user_id, nickname_hint, summary, source_message_count, source_max_message_id,
            updated_at, model_used, structured_json
        )
        SELECT user_id, nickname_hint, summary, source_message_count, source_max_message_id,
               updated_at, model_used, structured_json
        FROM user_profile_summaries WHERE user_id IN ({','.join('?' * len(v1_ids))})
        """,
        v1_ids,
    )
    conn.commit()
    return len(v1_ids)


# ---------------------------------------------------------------------------
# Archive reads
# ---------------------------------------------------------------------------

_USABLE_MESSAGE_SQL = "(coalesce(trim(message_content), '') != '' OR coalesce(trim(image_description), '') != '')"


def _usable_message_count(conn: sqlite3.Connection, user_id: int, after_message_id: Optional[int] = None) -> int:
    cur = conn.cursor()
    cur.execute(
        f"SELECT COUNT(*) FROM messages WHERE user_id = ? AND message_id > ? AND {_USABLE_MESSAGE_SQL}",
        (user_id, int(after_message_id or 0)),
    )
    return int(cur.fetchone()[0])


def _fetch_message_lines(
    conn: sqlite3.Connection,
    user_id: int,
    after_message_id: Optional[int],
) -> List[Tuple[int, str, str]]:
    """(message_id, date, prompt line) for this member's messages after the cursor, oldest first."""
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT message_id, date, channel_name, message_content, image_description
        FROM messages
        WHERE user_id = ? AND message_id > ? AND {_USABLE_MESSAGE_SQL}
        ORDER BY message_id ASC
        """,
        (user_id, int(after_message_id or 0)),
    )
    out: List[Tuple[int, str, str]] = []
    for mid, day, channel, content, image_desc in cur.fetchall():
        body = re.sub(r"\s+", " ", content or "").strip()
        if len(body) > _MESSAGE_BODY_MAX:
            body = body[: _MESSAGE_BODY_MAX - 1] + "…"
        line = f"[{day} #{channel or '?'}] {body}".rstrip()
        desc = re.sub(r"\s+", " ", image_desc or "").strip()
        if desc:
            if len(desc) > _IMAGE_DESCRIPTION_MAX:
                desc = desc[: _IMAGE_DESCRIPTION_MAX - 1] + "…"
            line += f" [shared an image: {desc}]"
        out.append((int(mid), str(day), line))
    return out


def _latest_name(conn: sqlite3.Connection, user_id: int) -> str:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(NULLIF(TRIM(nickname), ''), NULLIF(TRIM(username), ''))
        FROM messages WHERE user_id = ? ORDER BY message_id DESC LIMIT 1
        """,
        (user_id,),
    )
    row = cur.fetchone()
    return (row[0] or "").strip() if row else ""


def _top_member_directory(conn: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, COUNT(*) AS c FROM messages GROUP BY user_id ORDER BY c DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for uid, count in rows:
        uid = int(uid)
        out.append({"user_id": uid, "label": _latest_name(conn, uid) or f"user_{uid}", "message_count": int(count)})
    return out


def _interaction_hints(conn: sqlite3.Connection, user_id: int, names: Dict[int, str], limit: int = 12) -> List[str]:
    """Members whose messages most often sit right before or after this member's in the same channel."""
    cur = conn.cursor()
    cur.execute(
        """
        WITH seq AS (
            SELECT user_id,
                   LAG(user_id) OVER w AS prev_uid,
                   LEAD(user_id) OVER w AS next_uid
            FROM messages
            WINDOW w AS (PARTITION BY channel_id ORDER BY message_id)
        ),
        pairs AS (
            SELECT prev_uid AS oid FROM seq WHERE user_id = ? AND prev_uid != ?
            UNION ALL
            SELECT next_uid AS oid FROM seq WHERE user_id = ? AND next_uid != ?
        )
        SELECT oid, COUNT(*) AS c FROM pairs WHERE oid IS NOT NULL
        GROUP BY oid ORDER BY c DESC LIMIT ?
        """,
        (user_id, user_id, user_id, user_id, limit),
    )
    lines = []
    for oid, count in cur.fetchall():
        oid = int(oid)
        label = names.get(oid) or _latest_name(conn, oid) or str(oid)
        lines.append(f"- user_id={oid} ({label}): {int(count)} back-and-forth messages")
    return lines


def _upsert_profile(
    conn: sqlite3.Connection,
    user_id: int,
    nickname_hint: Optional[str],
    summary: str,
    structured_json: Optional[str],
    source_count: int,
    source_max_mid: Optional[int],
    model_used: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_profile_summaries (
            user_id, nickname_hint, summary, structured_json, source_message_count,
            source_max_message_id, updated_at, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            nickname_hint = excluded.nickname_hint,
            summary = excluded.summary,
            structured_json = excluded.structured_json,
            source_message_count = excluded.source_message_count,
            source_max_message_id = excluded.source_max_message_id,
            updated_at = CURRENT_TIMESTAMP,
            model_used = excluded.model_used
        """,
        (
            user_id,
            nickname_hint or "",
            summary,
            structured_json,
            source_count,
            source_max_mid,
            model_used,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Building one profile
# ---------------------------------------------------------------------------


async def refresh_user_profile(
    guild_id: int,
    user_id: int,
    *,
    progress: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    heartbeat: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Fold this member's unread messages into their profile, one saved pass at a time.

    ``should_stop`` is checked before each pass (pause, cancel, time limit);
    when it fires, the result has ``stopped=True`` and the next call resumes
    from the last saved pass.
    """

    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "error": "no database"}

    min_messages = _env_int("USER_PROFILE_MIN_MESSAGES", 8, minimum=1)
    pass_max_messages = _env_int("USER_PROFILE_PASS_MAX_MESSAGES", 150, minimum=10)
    busy_pass_messages = _env_int("USER_PROFILE_BUSY_PASS_MESSAGES", 40, minimum=5)
    lately_days = _env_int("USER_PROFILE_LATELY_DAYS", 90, minimum=0)
    directory_size = _env_int("USER_PROFILE_MEMBER_DIRECTORY", 45, minimum=0)
    summary_max = _env_int("USER_PROFILE_SUMMARY_MAX_CHARS", 20000, minimum=1000)
    model_name = os.getenv("LOCAL_CHAT", "")
    loop = asyncio.get_running_loop()

    def _prepare() -> Dict[str, Any]:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            ensure_user_profile_schema(conn)
            cur = conn.cursor()
            cur.execute(
                "SELECT structured_json, source_max_message_id FROM user_profile_summaries WHERE user_id = ?",
                (user_id,),
            )
            row = cur.fetchone()
            total = _usable_message_count(conn, user_id)
            if total < min_messages:
                return {"skip": "too_few_messages", "total": total}
            existing = _loads_profile(row[0]) if row else None
            if is_v2(existing):
                doc = normalize_document(existing)
                cursor = int(row[1]) if row[1] is not None else None
                mode = "update"
            else:
                if row:
                    _backup_v1_rows(conn, [user_id])
                doc, cursor, mode = new_document(), None, "rebuild"
            messages = _fetch_message_lines(conn, user_id, cursor)
            if not messages:
                return {"skip": "up_to_date", "total": total}
            directory = [m for m in _top_member_directory(conn, directory_size) if m["user_id"] != user_id]
            names = {m["user_id"]: m["label"] for m in directory}
            return {
                "doc": doc,
                "cursor": cursor,
                "mode": mode,
                "messages": messages,
                "total": total,
                "label": _latest_name(conn, user_id) or f"user_{user_id}",
                "directory": directory,
                "peers": _interaction_hints(conn, user_id, names),
            }
        finally:
            conn.close()

    plan = await loop.run_in_executor(None, _prepare)
    if plan.get("skip"):
        reason = plan["skip"]
        if reason == "too_few_messages":
            _p(f"Skip user_id={user_id}: only {plan['total']} usable messages (minimum {min_messages}).")
        return {"ok": True, "skipped": True, "reason": reason, "total_msgs": plan["total"]}

    doc: Dict[str, Any] = plan["doc"]
    messages: List[Tuple[int, str, str]] = plan["messages"]
    label: str = plan["label"]
    mode: str = plan["mode"]
    directory_names = {m["user_id"]: m["label"] for m in plan["directory"]}
    directory_lines = [
        f"- user_id={m['user_id']} name={m['label'][:80]} (msgs≈{m['message_count']})" for m in plan["directory"]
    ]
    peer_lines: List[str] = plan["peers"]
    system_prompt = build_system_prompt()

    def _save(snapshot: Dict[str, Any], cursor: int) -> None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            ensure_user_profile_schema(conn)
            _upsert_profile(
                conn,
                user_id,
                label,
                render_summary(snapshot, summary_max),
                json.dumps(snapshot, ensure_ascii=False),
                int(snapshot["coverage"]["messages"]),
                cursor,
                model_name,
            )
        finally:
            conn.close()

    def _user_prompt(lines: List[str]) -> str:
        return build_user_prompt(
            member_label=label,
            member_id=user_id,
            profile_render=render_for_prompt(doc),
            empty_sections=[k for k in SECTION_KEYS if k != "uncertain" and not doc["sections"].get(k)],
            directory_lines=directory_lines,
            peer_lines=peer_lines,
            message_lines=lines,
        )

    _p(
        f"user_id={user_id} ({label}) · {mode} · {len(messages)} message(s) to read "
        f"({messages[0][1]} → {messages[-1][1]}) · profile has {item_count(doc)} items"
    )

    pos = 0
    passes = 0
    skipped_messages = 0
    stopped = False
    while pos < len(messages):
        if should_stop and should_stop():
            stopped = True
            break
        if heartbeat:
            heartbeat()
        since_chat = seconds_since_chat_activity()
        busy = since_chat is not None and since_chat < _BUSY_WINDOW_SEC
        budget = await compute_budget()
        fixed_chars = len(system_prompt) + len(_user_prompt([]))
        available = budget.prompt_chars - fixed_chars
        if available < _MIN_MESSAGE_CHARS:
            raise RuntimeError(
                f"profile for user_id={user_id} leaves no room for messages ({budget.describe()}, "
                f"fixed prompt≈{fixed_chars} chars)"
            )

        limit = min(pass_max_messages, busy_pass_messages) if busy else pass_max_messages
        take, used = 0, 0
        while pos + take < len(messages) and take < limit:
            size = len(messages[pos + take][2]) + 1
            if take and used + size > available:
                break
            used += size
            take += 1

        passes += 1
        _p(
            f"  ▸ Pass {passes}: messages {pos + 1}–{pos + take} of {len(messages)} · {budget.describe()}"
            + (" · chat busy, smaller pass" if busy else "")
        )

        reply = None
        updated: Optional[Dict[str, Any]] = None
        chunk = messages[pos : pos + take]
        while True:
            chunk = messages[pos : pos + take]
            lines = [m[2] for m in chunk]
            user_prompt = _user_prompt(lines)
            failure = ""
            try:
                async with llm_turn():
                    if heartbeat:
                        heartbeat()
                    reply = await request_edits(
                        system_prompt, user_prompt, max_tokens=budget.max_output_tokens, progress=_p
                    )
            except ContextOverflowError:
                failure = "context overflow"
            except asyncio.TimeoutError:
                failure = "timeout"
            if reply is not None and not failure:
                observe_prompt_tokens(len(system_prompt) + len(user_prompt), reply.prompt_tokens)
                if reply.edits is None:
                    failure = "truncated output" if reply.finish_reason == "length" else "invalid JSON"
                else:
                    chunk_days = sorted(d for d in (parse_date(m[1]) for m in chunk) if d) or [date.today()]
                    updated, stats = apply_edits(
                        doc,
                        reply.edits,
                        window=(chunk_days[0], chunk_days[-1]),
                        source_text="\n".join(lines),
                        directory=directory_names,
                        lately_days=lately_days,
                    )
                    if not _looks_repetitive(stats):
                        break
                    # A small model can loop, re-emitting the same items until the output cap. Dedup
                    # absorbs the copies, but the few "new" ones can still push real items over a cap.
                    failure = f"repetitive output ({stats.merged_duplicates} duplicate items)"
            reply = None
            if take <= _MIN_SPLIT_MESSAGES:
                break
            take = max(1, take // 2)
            _p(f"    {failure} — retrying with the first {take} messages of this pass")

        if reply is None:
            skipped_messages += len(chunk)
            _p(f"    ❌ could not process messages {pos + 1}–{pos + len(chunk)} ({failure}); skipping them")
            pos += len(chunk)
            await loop.run_in_executor(None, _save, doc, chunk[-1][0])
            continue

        doc = updated
        coverage = doc["coverage"]
        coverage["first"] = coverage.get("first") or chunk[0][1]
        coverage["last"] = chunk[-1][1]
        coverage["messages"] = int(coverage.get("messages") or 0) + len(chunk)
        coverage["passes"] = int(coverage.get("passes") or 0) + 1
        pos += len(chunk)
        await loop.run_in_executor(None, _save, doc, chunk[-1][0])
        _p(
            f"  ◂ Pass {passes} saved: {chunk[0][1]} → {chunk[-1][1]} · {len(chunk)} msgs · "
            f"prompt {reply.prompt_tokens} tok · output {reply.completion_tokens} tok · {reply.elapsed:.0f}s · "
            f"edits {stats.summary()} · profile {item_count(doc)} items"
        )
        for note in stats.notes:
            _p(f"    note: {note}")

    logger.info(
        "user profile %s guild=%s user_id=%s passes=%s processed=%s/%s skipped=%s stopped=%s",
        mode,
        guild_id,
        user_id,
        passes,
        pos,
        len(messages),
        skipped_messages,
        stopped,
    )
    return {
        "ok": True,
        "user_id": user_id,
        "mode": mode,
        "passes": passes,
        "messages_processed": pos,
        "messages_remaining": len(messages) - pos,
        "messages_skipped": skipped_messages,
        "stopped": stopped,
        "items": item_count(doc),
    }


# ---------------------------------------------------------------------------
# Jobs: manual batches from the dashboard and the nightly refresh
# ---------------------------------------------------------------------------


def _looks_repetitive(stats: EditStats) -> bool:
    return stats.merged_duplicates >= 15 and stats.merged_duplicates > 2 * (stats.added + stats.updated)


def build_candidate_user_ids(guild_id: int) -> List[int]:
    """Members for a manual batch: the top posters with at least USER_PROFILE_MIN_MESSAGES."""
    max_users = _env_int("USER_PROFILES_BATCH_MAX_USERS", 80, minimum=1)
    min_msgs = _env_int("USER_PROFILE_MIN_MESSAGES", 8, minimum=1)
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT user_id, COUNT(*) AS c FROM messages WHERE {_USABLE_MESSAGE_SQL}
            GROUP BY user_id HAVING c >= ? ORDER BY c DESC LIMIT ?
            """,
            (min_msgs, max_users),
        )
        return [int(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()


def build_nightly_candidate_user_ids(
    guild_id: int, min_new_messages: int, exclude_user_ids: Iterable[int] = ()
) -> List[int]:
    """Members due a nightly update: enough new messages since their last pass, or no v2 profile yet.

    Updates come first (most new messages first); unbuilt profiles follow, so a
    long first build never crowds out the quick updates within the time limit.
    """
    min_msgs = _env_int("USER_PROFILE_MIN_MESSAGES", 8, minimum=1)
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute(
            f"SELECT user_id, COUNT(*) FROM messages WHERE {_USABLE_MESSAGE_SQL} GROUP BY user_id HAVING COUNT(*) >= ?",
            (min_msgs,),
        )
        excluded = {int(u) for u in exclude_user_ids}
        totals = {int(uid): int(c) for uid, c in cur.fetchall() if int(uid) not in excluded}
        cur.execute("SELECT user_id, structured_json, source_max_message_id FROM user_profile_summaries")
        profiles = {int(uid): (sj, mx) for uid, sj, mx in cur.fetchall()}
        ranked: List[Tuple[int, int, int]] = []
        for uid, total in totals.items():
            sj, cursor = profiles.get(uid, (None, None))
            if is_v2(_loads_profile(sj)):
                new = _usable_message_count(conn, uid, cursor)
                if new >= min_new_messages:
                    ranked.append((0, -new, uid))
            else:
                ranked.append((1, -total, uid))
        return [uid for _group, _n, uid in sorted(ranked)]
    finally:
        conn.close()


def clear_stored_profiles(guild_id: int) -> Dict[str, Any]:
    """Delete all rows in user_profile_summaries and the batch job row (pre-v2 rows are backed up first)."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_user_profile_schema(conn)
        _backup_v1_rows(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM user_profile_summaries")
        conn.commit()
        ensure_profile_job_schema(conn)
        delete_job(conn, guild_id)
    finally:
        conn.close()
    profile_job_log_clear(guild_id)
    return {"ok": True}


def _parse_batch_stats_json(raw: Optional[str]) -> Dict[str, int]:
    if not raw or not str(raw).strip():
        return {"saved": 0, "skipped": 0, "failed": 0}
    try:
        d = json.loads(raw)
        if not isinstance(d, dict):
            return {"saved": 0, "skipped": 0, "failed": 0}
        return {
            "saved": int(d.get("saved", 0) or 0),
            "skipped": int(d.get("skipped", 0) or 0),
            "failed": int(d.get("failed", 0) or 0),
        }
    except Exception:
        return {"saved": 0, "skipped": 0, "failed": 0}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _with_job_conn(db_path: str, fn: Callable[[sqlite3.Connection], Any]) -> Any:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_profile_job_schema(conn)
        return fn(conn)
    finally:
        conn.close()


def _job_status(guild_id: int) -> Optional[str]:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return None
    row = _with_job_conn(db_path, lambda c: get_job_row(c, guild_id))
    return str(row["status"]) if row is not None else None


async def _run_profile_job(
    guild_id: int,
    exclude_user_ids: Iterable[int] = (),
    between_members: Optional[Callable[[], Awaitable[None]]] = None,
) -> None:
    """Work through a running job's members until it finishes, is paused/cancelled, or hits its time limit.

    ``exclude_user_ids`` is Soupy's own account: the archive holds its messages
    too, and a profile of the bot would feed its own words back as a "member".
    ``between_members`` runs after each member, so a batch lasting hours doesn't
    starve the nightly scheduler.
    """
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return
    excluded = {int(u) for u in exclude_user_ids}
    while await _run_profile_job_step(guild_id, db_path, excluded):
        if between_members is not None:
            await between_members()


async def _run_profile_job_step(guild_id: int, db_path: str, excluded: Set[int]) -> bool:
    """Process the job's current member. Returns False once there is nothing more to do right now."""
    loop = asyncio.get_running_loop()
    row = await loop.run_in_executor(None, _with_job_conn, db_path, lambda c: get_job_row(c, guild_id))
    if row is None or str(row["status"] or "") != "running":
        return False
    uids: List[int] = json.loads(row["user_ids_json"] or "[]")
    i = int(row["next_index"] or 0)
    kind = str(row["kind"] or "manual")
    deadline = _parse_utc(row["deadline_at"])
    stats = _parse_batch_stats_json(row["stats_json"])

    def _finish(message: str) -> None:
        _with_job_conn(db_path, lambda c: upsert_job(c, guild_id, status="completed"))
        profile_job_log_append(guild_id, message)

    tally = f"saved={stats['saved']} · skipped={stats['skipped']} · failed={stats['failed']}"
    if deadline is not None and _utc_now() >= deadline:
        await loop.run_in_executor(
            None,
            _finish,
            f"━━━ {kind} run hit its time limit at member {i + 1}/{len(uids)} · {tally} · "
            "progress is saved and the rest continue next run ━━━",
        )
        return False
    if i >= len(uids):
        await loop.run_in_executor(None, _finish, f"━━━ {kind} run complete: {len(uids)} members · {tally} ━━━")
        return False

    uid = uids[i]
    if uid in excluded:
        stats["skipped"] += 1
        profile_job_log_append(guild_id, f"⏭ Skipping user_id={uid}: that's Soupy's own account.")
        stats_json = json.dumps(stats)
        await loop.run_in_executor(
            None, _with_job_conn, db_path, lambda c: upsert_job(c, guild_id, next_index=i + 1, stats_json=stats_json)
        )
        return True
    nick = await loop.run_in_executor(None, _with_job_conn, db_path, lambda c: _latest_name(c, uid))
    profile_job_log_append(
        guild_id,
        f"━━━ Member {i + 1}/{len(uids)}: user_id={uid}{f' ({nick})' if nick else ''} · {kind} · {tally} ━━━",
    )

    def _should_stop() -> bool:
        r = _with_job_conn(db_path, lambda c: get_job_row(c, guild_id))
        if r is None or str(r["status"] or "") != "running":
            return True
        return deadline is not None and _utc_now() >= deadline

    def _heartbeat() -> None:
        _with_job_conn(db_path, lambda c: upsert_job(c, guild_id, heartbeat_at=_utc_now().isoformat()))

    try:
        result = await refresh_user_profile(
            guild_id,
            uid,
            progress=lambda m: profile_job_log_append(guild_id, m),
            should_stop=_should_stop,
            heartbeat=_heartbeat,
        )
    except Exception as e:
        stats["failed"] += 1
        profile_job_log_append(guild_id, f"❌ ERROR user_id={uid}: {str(e).strip() or type(e).__name__}")
        logger.exception("profile job failed guild=%s user_id=%s", guild_id, uid)
        await asyncio.sleep(30)
    else:
        if result.get("stopped"):
            # Paused, cancelled or out of time: stay on this member; the saved cursor resumes it.
            profile_job_log_append(
                guild_id,
                f"⏸ Stopped mid-member after {result.get('passes', 0)} pass(es); "
                f"{result.get('messages_remaining', 0)} message(s) left for user_id={uid}.",
            )
            return True
        if result.get("skipped"):
            stats["skipped"] += 1
        elif result.get("ok"):
            stats["saved"] += 1
            profile_job_log_append(
                guild_id,
                f"✅ user_id={uid} · {result.get('mode')} · {result.get('passes')} pass(es) · "
                f"{result.get('messages_processed')} msgs · {result.get('items')} items",
            )

    stats_json = json.dumps(stats)
    await loop.run_in_executor(
        None,
        _with_job_conn,
        db_path,
        lambda c: upsert_job(c, guild_id, next_index=i + 1, stats_json=stats_json),
    )
    return True


def _nightly_run_date(now_local: datetime, tz: Any, hour: int) -> Optional[date]:
    """The local date whose nightly start window contains ``now_local`` (the window may cross midnight)."""
    for day in (now_local.date(), now_local.date() - timedelta(days=1)):
        start, end = window_bounds(tz, day, hour, hour + _NIGHTLY_START_WINDOW_HOURS)
        if start <= now_local < end:
            return day
    return None


async def _maybe_schedule_nightly(
    guild_ids: Sequence[int],
    timer: Optional[Dict[str, Any]] = None,
    exclude_user_ids: Iterable[int] = (),
) -> None:
    if not _env_bool("USER_PROFILE_NIGHTLY_ENABLED", True):
        if timer is not None:
            timer["enabled"] = False
        return
    hour = _env_int("USER_PROFILE_NIGHTLY_HOUR", 4, minimum=0, maximum=23)
    max_minutes = _env_int("USER_PROFILE_NIGHTLY_MAX_MINUTES", 180, minimum=10)
    min_new = _env_int("USER_PROFILE_NIGHTLY_MIN_NEW_MESSAGES", 10, minimum=1)
    tz = pytz.timezone(settings.timezone or "UTC")
    now_local = datetime.now(tz)
    state = load_json_state(NIGHTLY_STATE_PATH)

    if timer is not None:
        timer["enabled"] = True
        timer["interval"] = f"daily at {hour:02d}:00 local, up to {max_minutes} min"
        timer["last_run"] = state.get("scheduled_at")
        next_start = window_bounds(tz, now_local.date(), hour, hour)[0]
        if next_start <= now_local or state.get("last_run_date") == now_local.date().isoformat():
            next_start = window_bounds(tz, now_local.date() + timedelta(days=1), hour, hour)[0]
        timer["next_run"] = next_start.astimezone(timezone.utc).isoformat()

    run_date = _nightly_run_date(now_local, tz, hour)
    if run_date is None or state.get("last_run_date") == run_date.isoformat():
        return

    deadline = (_utc_now() + timedelta(minutes=max_minutes)).isoformat()
    for gid in guild_ids:
        db_path = get_db_path(gid)
        if not os.path.exists(db_path):
            continue
        status = _job_status(gid)
        if status in ("running", "paused"):
            profile_job_log_append(gid, f"Nightly refresh skipped: a {status} job is already queued.")
            continue
        uids = await asyncio.get_running_loop().run_in_executor(
            None, build_nightly_candidate_user_ids, gid, min_new, tuple(exclude_user_ids)
        )
        if not uids:
            continue

        def _queue(c: sqlite3.Connection, _gid: int = gid, _uids: List[int] = uids) -> None:
            upsert_job(
                c,
                _gid,
                status="running",
                user_ids_json=json.dumps(_uids),
                next_index=0,
                total=len(_uids),
                stats_json=json.dumps({"saved": 0, "skipped": 0, "failed": 0}),
                kind="nightly",
                deadline_at=deadline,
            )

        _with_job_conn(db_path, _queue)
        profile_job_log_clear(gid)
        profile_job_log_append(
            gid,
            f"━━━ Nightly refresh queued: {len(uids)} member(s) with new messages or no profile yet · "
            f"time limit {max_minutes} min ━━━",
        )

    save_json_state(
        NIGHTLY_STATE_PATH,
        {"last_run_date": run_date.isoformat(), "scheduled_at": _utc_now().isoformat()},
    )


async def profile_jobs_loop(
    get_guild_ids: Callable[[], Iterable[int]],
    *,
    get_excluded_user_ids: Callable[[], Iterable[int]] = tuple,
    timer: Optional[Dict[str, Any]] = None,
    poll_seconds: float = 20.0,
) -> None:
    """Bot-side worker: queues the nightly refresh when due and runs any running job. Never returns.

    ``get_excluded_user_ids`` returns accounts never to profile (Soupy itself).
    """
    while True:
        try:
            guild_ids = [int(g) for g in get_guild_ids()]
            excluded = tuple(int(u) for u in get_excluded_user_ids())
            async def _nightly_check(_ids: List[int] = guild_ids, _excluded: Tuple[int, ...] = excluded) -> None:
                await _maybe_schedule_nightly(_ids, timer, _excluded)

            await _nightly_check()
            for gid in guild_ids:
                if _job_status(gid) == "running":
                    await _run_profile_job(gid, excluded, between_members=_nightly_check)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("profile jobs loop tick failed")
        await asyncio.sleep(poll_seconds)


def start_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    """Queue a manual batch for the bot's profile worker (web process; no LLM calls happen here)."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    uids = build_candidate_user_ids(guild_id)
    if not uids:
        return {"ok": False, "message": "No users meet USER_PROFILE_MIN_MESSAGES threshold"}

    _with_job_conn(
        db_path,
        lambda c: upsert_job(
            c,
            guild_id,
            status="running",
            user_ids_json=json.dumps(uids),
            next_index=0,
            total=len(uids),
            stats_json=json.dumps({"saved": 0, "skipped": 0, "failed": 0}),
            kind="manual",
            deadline_at="",
        ),
    )
    profile_job_log_clear(guild_id)
    profile_job_log_append(
        guild_id,
        f"━━━ Batch queued: guild={guild_id} · {len(uids)} candidate member(s). The bot picks it up within "
        "about 20 seconds (it must be running). Profiles from before this version are rebuilt from scratch; "
        "current ones only read new messages. ━━━",
    )
    return {"ok": True, "total": len(uids), "user_ids": uids}


def pause_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    if _with_job_conn(db_path, lambda c: get_job_row(c, guild_id)) is None:
        return {"ok": False, "message": "no job"}
    _with_job_conn(db_path, lambda c: upsert_job(c, guild_id, status="paused"))
    profile_job_log_append(guild_id, "Pause requested — the bot stops after the current pass.")
    return {"ok": True, "status": "paused"}


def resume_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    if _with_job_conn(db_path, lambda c: get_job_row(c, guild_id)) is None:
        return {"ok": False, "message": "no job"}
    _with_job_conn(db_path, lambda c: upsert_job(c, guild_id, status="running"))
    profile_job_log_append(guild_id, "Resumed — the bot continues from the last saved pass.")
    return {"ok": True, "status": "running"}


async def cancel_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    db_path = get_db_path(guild_id)
    if os.path.exists(db_path):
        row = _with_job_conn(db_path, lambda c: get_job_row(c, guild_id))
        if row is not None:
            _with_job_conn(db_path, lambda c: upsert_job(c, guild_id, status="cancelled"))
    profile_job_log_append(guild_id, "Cancel requested — the bot stops after the current pass.")
    return {"ok": True, "status": "cancelled"}


def get_profile_batch_status(guild_id: int) -> Dict[str, Any]:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM user_profile_summaries")
        profile_row_count = int(cur.fetchone()["c"])
        row = get_job_row(conn, guild_id)
        if row is None:
            return {
                "ok": True,
                "guild_id": str(guild_id),
                "status": "idle",
                "next_index": 0,
                "total": 0,
                "profile_row_count": profile_row_count,
                "batch_stats": None,
                "task_running": False,
                "waiting_for_bot": False,
            }
        d = {k: row[k] for k in row.keys()}
        heartbeat = _parse_utc(d.get("heartbeat_at"))
        stale_after = max(_HEARTBEAT_STALE_MIN_SEC, _env_int("USER_PROFILE_LLM_TIMEOUT", 600, minimum=60) + 300)
        fresh = heartbeat is not None and (_utc_now() - heartbeat).total_seconds() < stale_after
        running = d.get("status") == "running"
        raw_stats = d.get("stats_json")
        return {
            "ok": True,
            "guild_id": str(guild_id),
            "status": d.get("status"),
            "kind": d.get("kind") or "manual",
            "deadline_at": d.get("deadline_at") or None,
            "heartbeat_at": d.get("heartbeat_at") or None,
            "next_index": int(d.get("next_index") or 0),
            "total": int(d.get("total") or 0),
            "user_ids_json": d.get("user_ids_json"),
            "profile_row_count": profile_row_count,
            "batch_stats": _parse_batch_stats_json(raw_stats) if raw_stats else None,
            "task_running": running and fresh,
            "waiting_for_bot": running and not fresh,
        }
    finally:
        conn.close()


def get_user_profile_stats(guild_id: int) -> Dict[str, Any]:
    """Row count and latest refresh time for stored summaries (creates table if missing)."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "exists": False, "message": "Database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM user_profile_summaries")
        c = int(cur.fetchone()[0])
        cur.execute("SELECT MAX(updated_at) AS mx FROM user_profile_summaries")
        mx = cur.fetchone()["mx"]
        return {
            "ok": True,
            "exists": True,
            "guild_id": str(guild_id),
            "profile_count": c,
            "latest_updated_at": mx if mx else None,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reading profiles for chat and daily posts
# ---------------------------------------------------------------------------


def _load_structured_profiles(conn: sqlite3.Connection, user_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """user_id -> {nickname_hint, structured, summary, age_days}"""
    ids = [int(u) for u in user_ids if u]
    if not ids:
        return {}
    cur = conn.cursor()
    qmarks = ",".join("?" * len(ids))
    cur.execute(
        f"SELECT user_id, nickname_hint, structured_json, summary, updated_at "
        f"FROM user_profile_summaries WHERE user_id IN ({qmarks})",
        ids,
    )
    out: Dict[int, Dict[str, Any]] = {}
    for uid, nickname_hint, structured_json, summary, updated_at in cur.fetchall():
        parsed = _loads_profile(structured_json)
        age_days: Optional[float] = None
        try:
            if updated_at:
                dt = datetime.fromisoformat(str(updated_at).replace(" ", "T"))
                age_days = (datetime.utcnow() - dt.replace(tzinfo=None)).total_seconds() / 86400
        except Exception:
            pass
        out[int(uid)] = {
            "nickname_hint": nickname_hint or "",
            "structured": parsed if isinstance(parsed, dict) else {},
            "summary": summary or "",
            "age_days": age_days,
        }
    return out


def _legacy_summary_excerpt(summary: str, max_chars: int) -> str:
    """Pre-v2 rows (not rebuilt yet): the stored plain summary, cut at a line boundary."""
    text = (summary or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    newline = cut.rfind("\n")
    return cut[:newline].rstrip() if newline > max_chars // 2 else cut.rstrip() + "…"


def format_profile_prefix_for_rag(
    conn: sqlite3.Connection,
    author_user_id: int,
    query_text: str,
    first_person_hint: Optional[str],
    subject_user_id: Optional[int] = None,
) -> str:
    """
    Build a query-aware profile prefix for the RAG bundle.
    Picks the profile items most relevant to the current message (see
    ``profile_document.render_for_chat``) instead of dumping the whole profile.
    If subject_user_id is provided, load that profile instead of re-resolving from tokens.
    """
    if os.getenv("USER_PROFILES_IN_RAG", "1").strip() in ("0", "false", "no"):
        return ""

    from .rag import (  # local import: rag fully loaded
        _extract_query_tokens,
        is_first_person_archive_query,
        normalize_rag_query_text,
    )

    ensure_user_profile_schema(conn)
    fp_src = (first_person_hint if first_person_hint is not None else query_text) or ""
    fp = is_first_person_archive_query(fp_src)
    # Use current message only for token extraction (avoid conversation history noise)
    _cur_msg = query_text or ""
    if "Current message:\n" in _cur_msg:
        _cur_msg = _cur_msg.split("Current message:\n")[-1].strip()
    _cur_msg = re.sub(r"^(?:user|assistant)\s*\([^)]*\)\s*:\s*", "", _cur_msg, flags=re.IGNORECASE).strip()
    qnorm = normalize_rag_query_text(_cur_msg or query_text or "")
    query_tokens = _extract_query_tokens(qnorm)
    # When asking about a specific other user, only include THAT user's profile
    # to prevent cross-contamination (LLM confuses asker's media list with subject's).
    if not fp and subject_user_id is not None and int(subject_user_id) != int(author_user_id):
        uids: Set[int] = {int(subject_user_id)}
    else:
        uids = {int(author_user_id)}

    profiles = _load_structured_profiles(conn, list(uids))
    if not profiles:
        return ""

    max_each = _env_int("RAG_PROFILE_MAX_CHARS_PER_USER", 3000, minimum=300)
    max_total = _env_int("RAG_PROFILE_MAX_CHARS", 4000, minimum=500)
    lately_days = _env_int("USER_PROFILE_LATELY_DAYS", 90, minimum=0)

    sketch_parts: List[str] = []
    for uid in sorted(uids):
        row = profiles.get(uid)
        if not row:
            continue
        nick = (row["nickname_hint"] or "").strip() or f"user_id {uid}"
        age_days = row["age_days"]
        age_str = ""
        if age_days is not None:
            if age_days < 1:
                age_str = " (updated today)"
            elif age_days < 2:
                age_str = " (updated yesterday)"
            else:
                age_str = f" (updated {int(age_days)}d ago)"

        if is_v2(row["structured"]):
            body = render_for_chat(
                row["structured"], query_tokens, max_each, today=date.today(), lately_days=lately_days
            )
        else:
            body = _legacy_summary_excerpt(row["summary"], max_each)
        if not body:
            continue
        sketch_parts.append(f"— {nick} (id {uid}){age_str}:\n{body}")

    if not sketch_parts:
        return ""

    header = (
        "Internal notes on this member, built from saved chat—not quotes or proof.\n"
        "Use them for tone, personal facts, and continuity. For what anyone actually said, use the excerpts below; "
        "they win if the notes disagree.\n"
        "(Mon YYYY) marks when something came up in chat. Older things may have changed—treat them as what you "
        "last heard, not as current fact.\n"
        "Answer in your normal voice. Never mention these notes, their dates, or that you keep a profile, and do "
        "not read them out like a list."
    )
    out = header + "\n\n" + "\n\n".join(sketch_parts)
    if len(out) > max_total:
        out = out[: max_total - 1] + "…"
    return "--- Member sketches (approximate; excerpts below win for facts) ---\n" + out + "\n--- end sketches ---\n\n"
