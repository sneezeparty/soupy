"""
Soupy's memory of itself, built the way member profiles are.

The old SELF.MD reflection rewrote one free-text document every night from the
last 60 replies. Its output hit the 4,000-token cap, so the document came back
truncated at the same length daily and lost every section but opinions. This
module replaces it with a version-2 document of the ``SELF`` kind (see
:mod:`soupy_database.profile_document`): dated, sectioned items the LLM edits
in passes over Soupy's own archived messages, each shown with what members said
just before it.

Grounding: items come only from what Soupy said. Members' lines are context.
Relationship items must name someone who is in the exchange.

Storage, per guild, in ``data/self_md/``:

* ``guild_<id>_self.json`` — the source of truth: ``{"bot_user_id", "cursor",
  "updated_at", "model", "doc"}``. ``cursor`` is the last message folded in.
* ``guild_<id>.md``, ``_core.md``, ``_anchor.md`` — views rendered after every
  pass, in the files the old reflection wrote, so their readers keep working:
  the anchor goes into every reply's system prompt, /soupyself shows the full
  view, and musings / bluesky read the core.
* ``v1_backup/`` — the old SELF.MD files, copied once before the first save.
  Nothing from them is imported.
* ``guild_<id>_self.sample.json`` — a preview build (see
  :func:`request_self_refresh`); never read by chat.

Each visible item is also embedded into the guild DB's ``self_chunks`` table
(``source = "item:<id>"``) when a build finishes, so chat can find the items
related to a message by meaning, not just shared words.

Where it runs: only in the bot process, through the profile worker in
:mod:`soupy_database.user_profiles` — at the end of every profile job (manual
or nightly) when Soupy has enough new messages, or when requested.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from soupy.scheduling import load_json_state, save_json_state
from soupy.settings import _env_int

from . import self_context
from .database import get_db_path
from .profile_document import (
    SELF,
    is_v2,
    item_count,
    new_document,
    normalize_document,
    render_for_prompt,
    render_self_core,
    render_self_for_chat,
    render_self_markdown,
    section_items,
    self_anchor,
)
from .profile_llm import build_self_system_prompt, build_self_user_prompt
from .self_context import (
    _cosine,
    _pack_embedding,
    _unpack_embedding,
    _write_file,
    ensure_self_chunks_schema,
    self_anchor_path,
    self_archive_path,
    self_core_path,
    self_md_path,
)

logger = logging.getLogger(__name__)


_SOUPY_BODY_MAX = 600
_CONTEXT_BODY_MAX = 300
# Members' messages shown before each of Soupy's: at most this many, from at most this long before.
_CONTEXT_LINES = 3
_CONTEXT_WINDOW = timedelta(minutes=15)
# Soupy messages this close together, with nothing in between, are one reply split into chunks.
_CONTINUATION_GAP = timedelta(minutes=5)
# Held back from each pass's budget for the batch-specific parts of the prompt.
_RELATIONSHIP_RESERVE_CHARS = 4000
_DIRECTORY_RESERVE_CHARS = 1500
_MENTION_RE = re.compile(r"<@!?(\d{15,20})>")
# Soupy's archived messages include slash-command output (search / weather / stock embeds, 8-ball and 9-ball
# answers). That's the command talking, not Soupy, so it's never read into the memory.
_SOUPY_VOICE_SQL = (
    "coalesce(trim(message_content), '') != '' "
    "AND message_content NOT LIKE '[Embed Title:%' "
    "AND message_content NOT LIKE '%-Ball says:%'"
)

# (last message_id covered, YYYY-MM-DD, prompt text) — the shape run_edit_passes reads.
Unit = Tuple[int, str, str]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def self_json_path(guild_id: int, *, sample: bool = False) -> Path:
    base = self_context.SELF_MD_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / (f"guild_{guild_id}_self.sample.json" if sample else f"guild_{guild_id}_self.json")


def v1_backup_dir() -> Path:
    return self_context.SELF_MD_DIR / "v1_backup"


def _requests_path() -> str:
    return str(self_context.SELF_MD_DIR / "refresh_requests.json")


def load_self_state(guild_id: int, *, sample: bool = False) -> Optional[Dict[str, Any]]:
    """The stored state with its document normalized, or None if there is no usable memory yet."""
    path = self_json_path(guild_id, sample=sample)
    if not path.exists():
        return None
    state = load_json_state(str(path))
    if not isinstance(state, dict) or not is_v2(state.get("doc")):
        return None
    state["doc"] = normalize_document(state["doc"], SELF)
    return state


def load_self_document(guild_id: int) -> Optional[Dict[str, Any]]:
    state = load_self_state(guild_id)
    return state["doc"] if state else None


def _backup_v1_files(guild_id: int) -> int:
    """Copy the old SELF.MD files into v1_backup/ (first copy wins). Returns how many were copied."""
    copied = 0
    for src in (
        self_md_path(guild_id),
        self_core_path(guild_id),
        self_anchor_path(guild_id),
        self_archive_path(guild_id),
    ):
        for path in (src, src.with_suffix(src.suffix + ".bak")):
            dest = v1_backup_dir() / path.name
            if path.exists() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                copied += 1
    return copied


def write_views(guild_id: int, doc: Dict[str, Any]) -> None:
    """Render the memory into the markdown files chat, /soupyself and the cogs read."""
    anchor_max = _env_int("SELF_MD_ANCHOR_MAX_CHARS", 600, minimum=100)
    _write_file(self_md_path(guild_id), render_self_markdown(doc) or "(empty)", backup=False)
    _write_file(self_core_path(guild_id), render_self_core(doc) or "", backup=False)
    _write_file(self_anchor_path(guild_id), self_anchor(doc, anchor_max) or "", backup=False)


def reset_self_memory(guild_id: int) -> bool:
    """Set the memory aside so the next refresh rebuilds it from the whole archive. Returns False if there was none."""
    path = self_json_path(guild_id)
    if not path.exists():
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    v1_backup_dir().mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(v1_backup_dir() / f"guild_{guild_id}_self.reset-{stamp}.json"))
    for view in (self_md_path(guild_id), self_core_path(guild_id), self_anchor_path(guild_id)):
        _write_file(view, "", backup=False)
    return True


# ---------------------------------------------------------------------------
# Reading Soupy's messages out of the archive
# ---------------------------------------------------------------------------


def _clip(text: Any, limit: int) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    return body if len(body) <= limit else body[: limit - 1] + "…"


def _parse_when(day: Any, tm: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(f"{str(day)[:10]} {str(tm or '00:00:00')[:8]}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _name_lookup(conn: sqlite3.Connection) -> Callable[[int], str]:
    from .user_profiles import _latest_name

    cache: Dict[int, str] = {}

    def lookup(uid: int) -> str:
        if uid not in cache:
            cache[uid] = _latest_name(conn, uid) or f"user_{uid}"
        return cache[uid]

    return lookup


def member_aliases(conn: sqlite3.Connection, names: Mapping[int, str], limit: int = 5) -> Dict[int, List[str]]:
    """Other names each member has gone by in the archive (usernames, older nicknames), newest first.

    The exchanges show everyone under their current nickname, but Soupy often used a username or an old
    nickname ("mr.keith7444" for sMURF0r), so matching memories to people needs all of them.
    """
    out: Dict[int, List[str]] = {}
    for uid, current in names.items():
        rows = conn.execute(
            "SELECT username, nickname FROM messages WHERE user_id = ? GROUP BY username, nickname "
            "ORDER BY MAX(message_id) DESC",
            (uid,),
        ).fetchall()
        seen = {current.strip().lower()}
        found: List[str] = []
        for username, nickname in rows:
            for name in (nickname, username):
                name = (name or "").strip()
                if name and name.lower() not in seen and len(found) < limit:
                    seen.add(name.lower())
                    found.append(name)
        if found:
            out[uid] = found
    return out


def _soupy_message_count(conn: sqlite3.Connection, bot_user_id: int, after_message_id: Optional[int] = None) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM messages WHERE user_id = ? AND message_id > ? AND {_SOUPY_VOICE_SQL}",
        (bot_user_id, int(after_message_id or 0)),
    ).fetchone()
    return int(row[0])


def fetch_self_units(
    conn: sqlite3.Connection, bot_user_id: int, after_message_id: Optional[int]
) -> Tuple[List[Unit], Dict[int, FrozenSet[int]], Dict[int, str]]:
    """Soupy's messages after the cursor as exchanges, oldest first.

    Returns ``(units, people, names)``: ``people`` maps a unit's message_id to
    the members in it (speakers and @-mentions), ``names`` those members' names.
    Each unit is Soupy's message (or a reply split over several messages) with
    up to three member messages from the 15 minutes before it.
    """
    names = _name_lookup(conn)
    rows = conn.execute(
        f"""
        SELECT message_id, date, time, channel_id, channel_name, message_content
        FROM messages
        WHERE user_id = ? AND message_id > ? AND {_SOUPY_VOICE_SQL}
        ORDER BY message_id ASC
        """,
        (bot_user_id, int(after_message_id or 0)),
    ).fetchall()

    units: List[Unit] = []
    people: Dict[int, FrozenSet[int]] = {}
    last_in_channel: Dict[int, Tuple[int, Optional[datetime], str]] = {}
    for mid, day, tm, channel_id, channel_name, content in rows:
        mid = int(mid)
        when = _parse_when(day, tm)
        tag = f"[{day} #{channel_name or '?'}]"
        before = conn.execute(
            """
            SELECT message_id, date, time, user_id, message_content, image_description
            FROM messages WHERE channel_id = ? AND message_id < ?
            ORDER BY message_id DESC LIMIT ?
            """,
            (channel_id, mid, _CONTEXT_LINES + 1),
        ).fetchall()
        body = _clip(content, _SOUPY_BODY_MAX)
        previous = last_in_channel.get(int(channel_id))

        if before and int(before[0][3]) == bot_user_id and previous and previous[0] == int(before[0][0]):
            # Nothing between this and Soupy's previous message in the channel.
            if previous[2] == body:
                # The archive sometimes stores the same message twice: skip it, but move the cursor past it.
                if units and units[-1][0] == previous[0]:
                    units[-1] = (mid, units[-1][1], units[-1][2])
                    people[mid] = people.pop(previous[0], frozenset())
                last_in_channel[int(channel_id)] = (mid, when, body)
                continue
            prev_when = previous[1]
            if when and prev_when and when - prev_when <= _CONTINUATION_GAP and units:
                last_mid, last_day, last_text = units[-1]
                if last_mid == previous[0]:
                    units[-1] = (mid, last_day, f"{last_text}\n{tag} SOUPY: {body}")
                    people[mid] = people.pop(last_mid, frozenset())
                    last_in_channel[int(channel_id)] = (mid, when, body)
                    continue

        context: List[str] = []
        present: Set[int] = set()
        for _bmid, bday, btm, buid, bcontent, bimage in before[:_CONTEXT_LINES]:
            buid = int(buid)
            if buid == bot_user_id:
                break
            bwhen = _parse_when(bday, btm)
            if when and bwhen and when - bwhen > _CONTEXT_WINDOW:
                break
            text = _clip(bcontent, _CONTEXT_BODY_MAX)
            if bimage:
                text = f"{text} [shared an image: {_clip(bimage, 120)}]".strip()
            if not text:
                continue
            present.add(buid)
            context.insert(0, f"[{bday} #{channel_name or '?'}] {names(buid)}: {text}")

        lines = context + [f"{tag} SOUPY: {body}"]
        text = "\n".join(lines)
        for m in _MENTION_RE.finditer(text):
            uid = int(m.group(1))
            if uid != bot_user_id:
                present.add(uid)

        def _mention(m: "re.Match[str]") -> str:
            uid = int(m.group(1))
            return "@soupy" if uid == bot_user_id else f"@{names(uid)}"

        units.append((mid, str(day), _MENTION_RE.sub(_mention, text)))
        people[mid] = frozenset(present)
        last_in_channel[int(channel_id)] = (mid, when, body)

    everyone = {uid for group in people.values() for uid in group}
    return units, people, {uid: names(uid) for uid in sorted(everyone)}


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def self_refresh_due(guild_id: int, bot_user_id: int, min_new_messages: int) -> bool:
    """True when the memory hasn't been built yet (and there's enough to build from) or has enough unread messages."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return False
    state = load_self_state(guild_id)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        if state is None:
            return _soupy_message_count(conn, bot_user_id) >= _env_int("USER_PROFILE_MIN_MESSAGES", 8, minimum=1)
        return _soupy_message_count(conn, bot_user_id, state.get("cursor")) >= min_new_messages
    finally:
        conn.close()


async def refresh_self_profile(
    guild_id: int,
    bot_user_id: int,
    *,
    progress: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    heartbeat: Optional[Callable[[], None]] = None,
    sample_passes: int = 0,
) -> Dict[str, Any]:
    """Fold Soupy's unread messages into its memory, one saved pass at a time.

    ``sample_passes`` > 0 builds a preview from the start of the archive into
    ``guild_<id>_self.sample.json`` and stops after that many passes, leaving
    the live memory, its views and the embeddings alone.
    """
    import asyncio

    from .user_profiles import run_edit_passes

    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "error": "no database"}
    sample = sample_passes > 0
    min_messages = _env_int("USER_PROFILE_MIN_MESSAGES", 8, minimum=1)
    pass_max_messages = _env_int("USER_PROFILE_PASS_MAX_MESSAGES", 150, minimum=10)
    busy_pass_messages = _env_int("USER_PROFILE_BUSY_PASS_MESSAGES", 40, minimum=5)
    lately_days = _env_int("USER_PROFILE_LATELY_DAYS", 90, minimum=0)
    model_name = os.getenv("LOCAL_CHAT", "")
    loop = asyncio.get_running_loop()

    state = None if sample else load_self_state(guild_id)
    doc = state["doc"] if state else new_document(SELF)
    cursor = state.get("cursor") if state else None
    mode = "sample" if sample else ("update" if state else "rebuild")

    def _prepare() -> Dict[str, Any]:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            total = _soupy_message_count(conn, bot_user_id)
            if total < min_messages:
                return {"skip": "too_few_messages", "total": total}
            units, people, names = fetch_self_units(conn, bot_user_id, cursor)
            aliases = member_aliases(conn, names)
            return {"units": units, "people": people, "names": names, "aliases": aliases, "total": total}
        finally:
            conn.close()

    plan = await loop.run_in_executor(None, _prepare)
    if plan.get("skip"):
        _p(f"Skip Soupy's memory: only {plan['total']} messages from Soupy (minimum {min_messages}).")
        return {"ok": True, "skipped": True, "reason": plan["skip"]}
    units: List[Unit] = plan["units"]
    if not units:
        return {"ok": True, "skipped": True, "reason": "up_to_date"}
    people: Dict[int, FrozenSet[int]] = plan["people"]
    names: Dict[int, str] = plan["names"]
    aliases: Dict[int, List[str]] = plan["aliases"]
    patterns = {
        uid: [re.compile(rf"(?<!\w){re.escape(n.lower())}(?!\w)") for n in [names[uid], *aliases.get(uid, ())] if n]
        for uid in names
    }

    def _user_prompt(chunk: Sequence[Unit], current: Dict[str, Any]) -> str:
        present: Set[int] = set()
        for mid, _day, _text in chunk:
            present.update(people.get(mid, ()))
        # Members Soupy names without them speaking ("i agree with ranc1d", "mr.keith7444") count too.
        chunk_lower = "\n".join(u[2] for u in chunk).lower()
        for uid, pats in patterns.items():
            if uid not in present and any(p.search(chunk_lower) for p in pats):
                present.add(uid)
        directory: List[str] = []
        used = 0
        for uid in sorted(present, key=lambda u: names.get(u, "").lower()):
            also = ", ".join(a[:40] for a in aliases.get(uid, ())[:3])
            line = f"- user_id={uid} name={names.get(uid, '')[:60]}" + (f" (also: {also})" if also else "")
            if used + len(line) + 1 > _DIRECTORY_RESERVE_CHARS:
                break
            directory.append(line)
            used += len(line) + 1
        return build_self_user_prompt(
            memory_render=render_for_prompt(
                current,
                SELF,
                relationship_user_ids=present,
                relationship_max_chars=_RELATIONSHIP_RESERVE_CHARS,
            ),
            empty_sections=[k for k in SELF.keys if k != "uncertain" and not current["sections"].get(k)],
            directory_lines=directory,
            message_lines=[u[2] for u in chunk],
        )

    first_live_save = [not sample and state is None]

    def _save(snapshot: Dict[str, Any], new_cursor: int) -> None:
        if first_live_save[0]:
            copied = _backup_v1_files(guild_id)
            if copied:
                logger.info("self profile: backed up %d SELF.MD file(s) for guild %s", copied, guild_id)
            first_live_save[0] = False
        payload = {
            "bot_user_id": bot_user_id,
            "cursor": new_cursor,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "doc": snapshot,
        }
        if not save_json_state(str(self_json_path(guild_id, sample=sample)), payload):
            raise RuntimeError(f"could not save Soupy's memory for guild {guild_id}")
        if not sample:
            write_views(guild_id, snapshot)

    _p(
        f"━━━ Soupy's memory · {mode} · {len(units)} exchange(s) to read ({units[0][1]} → {units[-1][1]}) · "
        f"{item_count(doc, SELF)} items ━━━"
    )
    run = await run_edit_passes(
        doc=doc,
        units=units,
        system_prompt=build_self_system_prompt(),
        user_prompt=_user_prompt,
        save=_save,
        kind=SELF,
        directory_names=names,
        lately_days=lately_days,
        pass_max_messages=pass_max_messages,
        busy_pass_messages=busy_pass_messages,
        what="Soupy's memory",
        reserve_chars=_RELATIONSHIP_RESERVE_CHARS + _DIRECTORY_RESERVE_CHARS,
        max_passes=sample_passes,
        aliases=aliases,
        progress=progress,
        should_stop=should_stop,
        heartbeat=heartbeat,
    )
    if not sample and run["passes"]:
        await index_self_items(guild_id, run["doc"], progress=_p)
    logger.info(
        "self profile %s guild=%s passes=%s processed=%s/%s skipped=%s stopped=%s",
        mode,
        guild_id,
        run["passes"],
        run["processed"],
        len(units),
        run["skipped"],
        run["stopped"],
    )
    return {
        "ok": True,
        "mode": mode,
        "passes": run["passes"],
        "messages_processed": run["processed"],
        "messages_remaining": len(units) - run["processed"],
        "messages_skipped": run["skipped"],
        "stopped": run["stopped"] and not sample,
        "items": item_count(run["doc"], SELF),
    }


# ---------------------------------------------------------------------------
# Refresh requests (/soupyself refresh, previews)
# ---------------------------------------------------------------------------


def request_self_refresh(guild_id: int, *, sample_passes: int = 0) -> None:
    """Ask the bot's profile worker to refresh (or preview) this guild's memory at its next opportunity."""
    requests = load_json_state(_requests_path())
    requests = requests if isinstance(requests, dict) else {}
    requests[str(guild_id)] = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "sample_passes": max(0, int(sample_passes)),
    }
    save_json_state(_requests_path(), requests)


def pop_self_refresh_requests() -> Dict[int, Dict[str, Any]]:
    requests = load_json_state(_requests_path())
    if not isinstance(requests, dict) or not requests:
        return {}
    save_json_state(_requests_path(), {})
    out: Dict[int, Dict[str, Any]] = {}
    for gid, req in requests.items():
        try:
            out[int(gid)] = req if isinstance(req, dict) else {}
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


async def index_self_items(guild_id: int, doc: Dict[str, Any], progress: Optional[Callable[[str], None]] = None) -> int:
    """Replace this guild's self_chunks with one embedded row per chat-visible memory item."""
    import aiohttp

    from .rag import embed_texts_lm_studio

    rows: List[Tuple[str, str, str]] = []
    for sec in SELF.sections:
        if sec.key in SELF.chat_excluded:
            continue
        for it in section_items(doc, sec.key):
            name = it.get("name") or ""
            text = f"{name}: {it['text']}" if name and name.lower() not in it["text"].lower() else it["text"]
            rows.append((f"item:{it['id']}", sec.key, f"[{sec.label}] {text}"))
    db_path = get_db_path(guild_id)
    if not rows or not os.path.exists(db_path):
        return 0
    try:
        async with aiohttp.ClientSession() as session:
            vectors = await embed_texts_lm_studio(session, [r[2] for r in rows])
    except Exception as exc:
        logger.warning("self profile: embedding failed for guild %s: %s", guild_id, exc)
        if progress:
            progress(f"    memory saved, but embedding its items failed ({exc}); chat falls back to word matching")
        return 0
    if not vectors or len(vectors) != len(rows):
        return 0
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_self_chunks_schema(conn)
        conn.execute("DELETE FROM self_chunks")
        dim = len(vectors[0])
        conn.executemany(
            "INSERT INTO self_chunks (source, section, chunk_text, embedding_dim, embedding) VALUES (?, ?, ?, ?, ?)",
            [(src, sec, text, dim, _pack_embedding(vec)) for (src, sec, text), vec in zip(rows, vectors, strict=True)],
        )
        conn.commit()
    finally:
        conn.close()
    if progress:
        progress(f"    embedded {len(rows)} memory item(s) for chat")
    return len(rows)


def _ranked_item_ids(
    conn: sqlite3.Connection, query_vec: Sequence[float], min_sim: float, limit: int = 24
) -> List[str]:
    try:
        ensure_self_chunks_schema(conn)
        rows = conn.execute(
            "SELECT source, embedding, embedding_dim FROM self_chunks WHERE source LIKE 'item:%'"
        ).fetchall()
    except sqlite3.Error:
        return []
    qdim = len(query_vec)
    scored: List[Tuple[float, str]] = []
    for source, blob, dim in rows:
        if int(dim) != qdim:
            continue
        score = _cosine(query_vec, _unpack_embedding(blob))
        if score >= min_sim:
            scored.append((score, str(source)[5:]))
    scored.sort(reverse=True)
    return [iid for _score, iid in scored[:limit]]


def self_block_for_chat(
    conn: sqlite3.Connection,
    guild_id: int,
    *,
    people: Mapping[int, str],
    query_vec: Optional[Sequence[float]],
    query_tokens: Sequence[str],
    max_chars: int,
    min_sim: float,
) -> Optional[str]:
    """Soupy's memories for this reply, or None when this guild has no memory yet (callers fall back to SELF.MD)."""
    doc = load_self_document(guild_id)
    if doc is None:
        return None
    header = "(Mon YYYY) is when it came up. Things may have changed since."
    ranked = _ranked_item_ids(conn, query_vec, min_sim) if query_vec is not None else []
    body = render_self_for_chat(
        doc,
        people=people,
        ranked_ids=ranked,
        query_tokens=query_tokens,
        max_chars=max(0, max_chars - len(header) - 1),
        today=date.today(),
        lately_days=_env_int("USER_PROFILE_LATELY_DAYS", 90, minimum=0),
    )
    return f"{header}\n{body}" if body else ""


def memory_seed_lines(guild_id: int) -> Optional[List[str]]:
    """Chat-visible memory items as plain text (for musing seeds), or None when this guild has no memory yet."""
    doc = load_self_document(guild_id)
    if doc is None:
        return None
    return [
        it["text"] for sec in SELF.sections if sec.key not in SELF.chat_excluded for it in section_items(doc, sec.key)
    ]


def memory_summary(guild_id: int) -> Optional[Dict[str, Any]]:
    """Counts for the dashboard and /soupyself: items per section, coverage, last update."""
    state = load_self_state(guild_id)
    if state is None:
        return None
    doc = state["doc"]
    return {
        "items": item_count(doc, SELF),
        "sections": {sec.label: len(section_items(doc, sec.key)) for sec in SELF.sections},
        "coverage": doc.get("coverage") or {},
        "updated_at": state.get("updated_at"),
    }


__all__ = [
    "fetch_self_units",
    "index_self_items",
    "load_self_document",
    "load_self_state",
    "member_aliases",
    "memory_seed_lines",
    "memory_summary",
    "pop_self_refresh_requests",
    "refresh_self_profile",
    "request_self_refresh",
    "reset_self_memory",
    "self_block_for_chat",
    "self_json_path",
    "v1_backup_dir",
    "self_refresh_due",
    "write_views",
]
