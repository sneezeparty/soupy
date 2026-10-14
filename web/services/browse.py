"""
Read-only queries behind the dashboard's Database and Media tabs.

Kept apart from the routes in ``web/app.py`` so they can be tested against a
scratch SQLite file: importing the app starts logging and mounts ``media/``.

Discord IDs leave here as strings. They are 64-bit snowflakes, and a browser
reads a JSON number as a double, so ``123456789012345678`` arrives as
``123456789012345680``. Sent back as a filter, that matches nothing.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from soupy_database.profile_document import (
    MEMBER,
    SELF,
    DocumentKind,
    is_v2,
    normalize_document,
    section_items,
)

MESSAGE_COLUMNS = (
    "message_id, date, time, username, nickname, user_id, channel_name, channel_id, "
    "message_content, image_description, url_summary"
)
_SEARCH_COLUMNS = ("message_content", "username", "nickname", "channel_name", "image_description", "url_summary")
_NAME_SQL = "COALESCE(NULLIF(TRIM(nickname), ''), NULLIF(TRIM(username), ''))"

PREVIEW_CHARS = 220


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a guild database for browsing. ``query_only`` keeps a stray write from ever landing."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def parse_snowflake(raw: Any) -> Optional[int]:
    """A Discord ID from a query string, or None when it isn't one."""
    text = str(raw or "").strip()
    return int(text) if text.isdigit() else None


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _message_row(r: sqlite3.Row) -> Dict[str, Any]:
    return {
        "message_id": str(r["message_id"]),
        "date": r["date"],
        "time": r["time"],
        "username": r["username"],
        "nickname": r["nickname"],
        "user_id": str(r["user_id"]),
        "channel_name": r["channel_name"],
        "channel_id": str(r["channel_id"]),
        "message_content": r["message_content"],
        "image_description": r["image_description"],
        "url_summary": r["url_summary"],
    }


def query_messages(
    conn: sqlite3.Connection,
    *,
    search: str = "",
    channel_id: Optional[int] = None,
    user_id: Optional[int] = None,
    has_images: bool = False,
    has_urls: bool = False,
    after: str = "",
    before: str = "",
    newest_first: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """One page of archived messages.

    ``after`` / ``before`` are UTC ``YYYY-MM-DD HH:MM:SS`` bounds, the format
    the archive stores. Rows come back in message order: snowflakes sort by
    send time, where ``created_at`` only records when a scan inserted the row.
    """
    where: List[str] = []
    params: List[Any] = []
    term = (search or "").strip()
    if term:
        like = f"%{_escape_like(term)}%"
        where.append("(" + " OR ".join(f"{col} LIKE ? ESCAPE '\\'" for col in _SEARCH_COLUMNS) + ")")
        params.extend([like] * len(_SEARCH_COLUMNS))
    if channel_id is not None:
        where.append("channel_id = ?")
        params.append(channel_id)
    if user_id is not None:
        where.append("user_id = ?")
        params.append(user_id)
    if has_images:
        where.append("image_description IS NOT NULL AND TRIM(image_description) != ''")
    if has_urls:
        where.append("url_summary IS NOT NULL AND TRIM(url_summary) != ''")
    if after:
        where.append("(date || ' ' || time) >= ?")
        params.append(after)
    if before:
        where.append("(date || ' ' || time) <= ?")
        params.append(before)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    total = int(conn.execute(f"SELECT COUNT(*) FROM messages {where_sql}", params).fetchone()[0])
    order = "DESC" if newest_first else "ASC"
    rows = conn.execute(
        f"SELECT {MESSAGE_COLUMNS} FROM messages {where_sql} ORDER BY message_id {order} LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return {"total": total, "limit": limit, "offset": offset, "rows": [_message_row(r) for r in rows]}


def message_context(
    conn: sqlite3.Connection, message_id: int, *, before: int = 8, after: int = 8
) -> Optional[Dict[str, Any]]:
    """The message plus its neighbours in the same channel, oldest first. None if it isn't archived."""
    row = conn.execute(f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE message_id = ?", (message_id,)).fetchone()
    if row is None:
        return None
    older = conn.execute(
        f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE channel_id = ? AND message_id < ? "
        "ORDER BY message_id DESC LIMIT ?",
        (row["channel_id"], message_id, max(0, min(int(before), 50))),
    ).fetchall()
    newer = conn.execute(
        f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE channel_id = ? AND message_id > ? "
        "ORDER BY message_id ASC LIMIT ?",
        (row["channel_id"], message_id, max(0, min(int(after), 50))),
    ).fetchall()
    return {
        "message_id": str(message_id),
        "rows": [_message_row(r) for r in (*reversed(older), row, *newer)],
    }


def list_members(conn: sqlite3.Connection, limit: int = 400) -> List[Dict[str, Any]]:
    """Posters by message count, each under the name on their latest message."""
    rows = conn.execute(
        f"""
        SELECT g.user_id, g.n, {_NAME_SQL} AS label
        FROM (SELECT user_id, COUNT(*) AS n, MAX(message_id) AS last_id FROM messages GROUP BY user_id) g
        JOIN messages ON messages.message_id = g.last_id
        ORDER BY g.n DESC
        LIMIT ?
        """,
        (max(1, min(int(limit), 800)),),
    ).fetchall()
    return [
        {
            "user_id": str(r["user_id"]),
            "label": (r["label"] or "").strip() or f"user {r['user_id']}",
            "message_count": int(r["n"]),
        }
        for r in rows
    ]


def list_channels(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Channels by message count."""
    rows = conn.execute("""
        SELECT c.channel_id, c.channel_name, COUNT(m.message_id) AS n
        FROM channels c
        LEFT JOIN messages m ON m.channel_id = c.channel_id
        GROUP BY c.channel_id
        ORDER BY n DESC
        """).fetchall()
    return [
        {"id": str(r["channel_id"]), "name": r["channel_name"] or str(r["channel_id"]), "messages": int(r["n"])}
        for r in rows
    ]


def recent_scans(conn: sqlite3.Connection, limit: int = 10) -> Dict[str, Any]:
    total = int(conn.execute("SELECT COUNT(*) FROM scan_metadata").fetchone()[0])
    rows = conn.execute(
        "SELECT scan_type, last_scan_time, messages_scanned, created_at FROM scan_metadata ORDER BY id DESC LIMIT ?",
        (max(1, min(int(limit), 200)),),
    ).fetchall()
    return {"total": total, "rows": [dict(r) for r in rows]}


def _latest_names(conn: sqlite3.Connection, user_ids: Iterable[int]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for uid in set(user_ids):
        row = conn.execute(
            f"SELECT {_NAME_SQL} FROM messages WHERE user_id = ? ORDER BY message_id DESC LIMIT 1", (uid,)
        ).fetchone()
        if row and row[0]:
            names[uid] = str(row[0]).strip()
    return names


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def _loads(raw: Any) -> Any:
    try:
        return json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _label_from_key(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _preview(text: str) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1].rstrip() + "…"


def document_view(
    doc: Mapping[str, Any], kind: DocumentKind, names: Optional[Mapping[int, str]] = None
) -> Dict[str, Any]:
    """A v2 profile document laid out for display: sections in order, each with its label and items."""
    doc = normalize_document(doc, kind)
    names = names or {}
    sections: List[Dict[str, Any]] = []
    total = 0
    for sec in kind.sections:
        items: List[Dict[str, Any]] = []
        for it in section_items(doc, sec.key):
            item: Dict[str, Any] = {"id": it["id"], "text": it["text"], "date": it.get("date")}
            if it.get("last_seen") and it["last_seen"] != it.get("date"):
                item["last_seen"] = it["last_seen"]
            uid = it.get("user_id") or 0
            name = it.get("name") or names.get(uid, "")
            if name:
                item["name"] = name
            if uid:
                item["user_id"] = str(uid)
            items.append(item)
        total += len(items)
        sections.append({"key": sec.key, "label": sec.label, "cap": sec.cap, "guide": sec.guide, "items": items})
    return {
        "version": 2,
        "overview": doc["overview"],
        "communication_style": doc["communication_style"],
        "coverage": doc["coverage"],
        "sections": sections,
        "item_count": total,
    }


def legacy_view(doc: Any) -> Dict[str, Any]:
    """A pre-v2 profile: top-level keys holding a string or a list of strings."""
    view: Dict[str, Any] = {
        "version": 1,
        "overview": "",
        "communication_style": "",
        "coverage": {},
        "sections": [],
        "item_count": 0,
    }
    if not isinstance(doc, dict):
        view["version"] = 0
        return view
    for key, value in doc.items():
        if key in ("overview", "communication_style"):
            view[key] = str(value or "").strip()
            continue
        texts = value if isinstance(value, list) else [value]
        items = [
            {"id": f"{key}{i}", "text": str(t).strip(), "date": None}
            for i, t in enumerate(texts)
            if isinstance(t, (str, int, float)) and str(t).strip()
        ]
        if items:
            view["sections"].append({"key": key, "label": _label_from_key(key), "cap": 0, "guide": "", "items": items})
            view["item_count"] += len(items)
    return view


def _relationship_ids(doc: Any) -> List[int]:
    if not is_v2(doc):
        return []
    out = []
    for it in section_items(doc, MEMBER.relationship_key):
        try:
            uid = int(it.get("user_id") or 0)
        except (TypeError, ValueError):
            uid = 0
        if uid:
            out.append(uid)
    return out


def _profile_view(conn: sqlite3.Connection, doc: Any) -> Dict[str, Any]:
    if is_v2(doc):
        return document_view(doc, MEMBER, _latest_names(conn, _relationship_ids(doc)))
    return legacy_view(doc)


def count_profiles(conn: sqlite3.Connection) -> int:
    if not _has_table(conn, "user_profile_summaries"):
        return 0
    return int(conn.execute("SELECT COUNT(*) FROM user_profile_summaries").fetchone()[0])


def list_profiles(conn: sqlite3.Connection, *, search: str = "") -> List[Dict[str, Any]]:
    """Stored member profiles, most-read member first, without the profile bodies.

    ``search`` matches a name, an exact user ID, or any text in the profile.
    """
    if not _has_table(conn, "user_profile_summaries"):
        return []
    where, params = "", []
    term = (search or "").strip()
    if term:
        like = f"%{_escape_like(term)}%"
        where = (
            "WHERE CAST(user_id AS TEXT) = ? OR COALESCE(nickname_hint, '') LIKE ? ESCAPE '\\' "
            "OR COALESCE(summary, '') LIKE ? ESCAPE '\\'"
        )
        params = [term, like, like]
    rows = conn.execute(
        f"""
        SELECT user_id, nickname_hint, summary, structured_json, source_message_count, updated_at, model_used
        FROM user_profile_summaries {where}
        ORDER BY source_message_count DESC, user_id
        """,
        params,
    ).fetchall()
    names = _latest_names(conn, (int(r["user_id"]) for r in rows if not (r["nickname_hint"] or "").strip()))
    out: List[Dict[str, Any]] = []
    for r in rows:
        uid = int(r["user_id"])
        doc = _loads(r["structured_json"])
        v2 = is_v2(doc)
        overview = str(doc.get("overview") or "") if isinstance(doc, dict) else ""
        out.append(
            {
                "user_id": str(uid),
                "kind": "member",
                "name": (r["nickname_hint"] or "").strip() or names.get(uid) or f"user {uid}",
                "message_count": int(r["source_message_count"] or 0),
                "updated_at": r["updated_at"],
                "model_used": r["model_used"],
                "version": 2 if v2 else (1 if isinstance(doc, dict) else 0),
                "item_count": (
                    sum(len(section_items(doc, key)) for key in MEMBER.keys) if v2 else legacy_view(doc)["item_count"]
                ),
                "preview": _preview(overview or r["summary"] or ""),
            }
        )
    return out


def profile_detail(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    """One member's stored profile laid out for display, or None if there isn't one."""
    if not _has_table(conn, "user_profile_summaries"):
        return None
    row = conn.execute(
        """
        SELECT user_id, nickname_hint, summary, structured_json, source_message_count,
               source_max_message_id, updated_at, model_used
        FROM user_profile_summaries WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    view = _profile_view(conn, _loads(row["structured_json"]))
    name = (row["nickname_hint"] or "").strip() or _latest_names(conn, [user_id]).get(user_id) or f"user {user_id}"
    view.update(
        {
            "user_id": str(user_id),
            "kind": "member",
            "name": name,
            "message_count": int(row["source_message_count"] or 0),
            "source_max_message_id": (
                str(row["source_max_message_id"]) if row["source_max_message_id"] is not None else None
            ),
            "updated_at": row["updated_at"],
            "model_used": row["model_used"],
            "summary": row["summary"] or "",
        }
    )
    return view


def self_memory_detail(state: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Soupy's memory of itself in the shape :func:`profile_detail` returns.

    ``state`` is what ``self_profile.load_self_state`` gives back.
    """
    if not state or not is_v2(state.get("doc")):
        return None
    view = document_view(state["doc"], SELF)
    view.update(
        {
            "user_id": "self",
            "kind": "self",
            "name": "Soupy",
            "message_count": int((view["coverage"] or {}).get("messages") or 0),
            "source_max_message_id": None,
            "updated_at": state.get("updated_at"),
            "model_used": state.get("model"),
            "summary": "",
        }
    )
    return view


def self_memory_row(state: Optional[Mapping[str, Any]], *, search: str = "") -> Optional[Dict[str, Any]]:
    """The list entry for Soupy's memory, matching the rows of :func:`list_profiles`.

    None when there is no memory yet, or when ``search`` is in neither its name nor its text.
    """
    detail = self_memory_detail(state)
    if detail is None:
        return None
    term = (search or "").strip().lower()
    if term:
        texts = [detail["name"], detail["overview"], detail["communication_style"]]
        for section in detail["sections"]:
            for item in section["items"]:
                texts.extend((item["text"], item.get("name", "")))
        if not any(term in text.lower() for text in texts):
            return None
    return {
        "user_id": "self",
        "kind": "self",
        "name": "Soupy",
        "message_count": detail["message_count"],
        "updated_at": detail["updated_at"],
        "model_used": detail["model_used"],
        "version": 2,
        "item_count": detail["item_count"],
        "preview": _preview(detail["overview"]),
    }


# ---------------------------------------------------------------------------
# Media archive
# ---------------------------------------------------------------------------

_VISION_PREFIX_RE = re.compile(r"^\s*(?:\[Vision\]|\W{0,4}Vision:)\s*", re.IGNORECASE)


class JsonlFile:
    """A JSONL file's rows, re-read only when the file changes on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._signature: Optional[tuple] = None
        self._rows: List[Dict[str, Any]] = []

    def signature(self) -> Optional[tuple]:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def rows(self) -> List[Dict[str, Any]]:
        sig = self.signature()
        if sig is None:
            return []
        if sig != self._signature:
            rows: List[Dict[str, Any]] = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue  # one malformed line doesn't lose the rest
                    if isinstance(obj, dict):
                        rows.append(obj)
            self._rows, self._signature = rows, sig
        return self._rows


def _id_text(value: Any) -> Optional[str]:
    return str(value) if value not in (None, "") else None


def _source_image(ref: Any) -> Optional[str]:
    """A source-image reference from the index, if it names a file in one of the two media folders."""
    folder, _, name = str(ref or "").partition("/")
    if folder in ("images", "flux_sources") and name and "/" not in name and name not in (".", ".."):
        return f"{folder}/{name}"
    return None


class MediaArchive:
    """Generated and analyzed images, newest first.

    ``media/images/index.jsonl`` has one row per image but cuts a vision
    description off at about 110 characters. The whole description is in
    ``media/messages.jsonl``, so the two are joined here on the filename.
    """

    def __init__(self, media_dir: Path) -> None:
        self.index = JsonlFile(media_dir / "images" / "index.jsonl")
        self.messages = JsonlFile(media_dir / "messages.jsonl")
        self._signature: Optional[tuple] = None
        self._items: List[Dict[str, Any]] = []

    def _all(self) -> List[Dict[str, Any]]:
        sig = (self.index.signature(), self.messages.signature())
        if sig == self._signature:
            return self._items
        full_text: Dict[str, str] = {}
        for msg in self.messages.rows():
            if msg.get("image_filename") and msg.get("event_type") == "vision":
                full_text[str(msg["image_filename"])] = str(msg.get("content") or "")
        items: List[Dict[str, Any]] = []
        for row in reversed(self.index.rows()):
            filename = str(row.get("filename") or "")
            if not filename:
                continue
            vision = row.get("event_type") == "vision"
            text = (full_text.get(filename) or row.get("prompt") or "") if vision else (row.get("prompt") or "")
            items.append(
                {
                    "filename": filename,
                    "ts": row.get("ts"),
                    "username": row.get("username"),
                    "user_id": _id_text(row.get("user_id")),
                    "guild_id": _id_text(row.get("guild_id")),
                    "channel_id": _id_text(row.get("channel_id")),
                    "width": row.get("width"),
                    "height": row.get("height"),
                    "seed": row.get("seed"),
                    "event_type": "vision" if vision else "generated",
                    "source": "vision" if vision else str(row.get("source") or "sd").lower(),
                    "text": _VISION_PREFIX_RE.sub("", str(text)) if vision else str(text),
                    "original_url": row.get("original_url"),
                    # How a Flux picture was made. Older rows have none of these.
                    "model": row.get("model"),
                    "steps": row.get("steps"),
                    "guidance": row.get("guidance"),
                    "strength": row.get("strength"),
                    "seconds": row.get("seconds"),
                    "action": row.get("action"),
                    "source_image": _source_image(row.get("source_image")),
                    "original_prompt": row.get("original_prompt"),
                }
            )
        self._items, self._signature = items, sig
        return items

    def images(self, *, kind: str = "all", search: str = "", limit: int = 60, offset: int = 0) -> Dict[str, Any]:
        items = self._all()
        counts = {"all": len(items), "vision": sum(1 for it in items if it["event_type"] == "vision")}
        counts["generated"] = counts["all"] - counts["vision"]
        kind = kind if kind in ("vision", "generated") else "all"
        if kind != "all":
            items = [it for it in items if it["event_type"] == kind]
        term = (search or "").strip().lower()
        if term:
            items = [
                it
                for it in items
                if term in it["text"].lower()
                or term in str(it["username"] or "").lower()
                or term in it["filename"].lower()
            ]
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        return {
            # Copies: the caller adds channel names, and the cached rows stay as read.
            "items": [dict(it) for it in items[offset : offset + limit]],
            "total": len(items),
            "limit": limit,
            "offset": offset,
            "kind": kind,
            "counts": counts,
        }

    def log(self, *, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """The text activity log (``messages.jsonl``), newest first."""
        rows = self.messages.rows()
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        end = len(rows) - offset
        page = rows[max(0, end - limit) : max(0, end)]
        return {"items": list(reversed(page)), "total": len(rows), "limit": limit, "offset": offset}


def attach_channel_names(items: Sequence[Dict[str, Any]], names_for_guild: Any) -> None:
    """Add ``channel_name`` to each image row. ``names_for_guild(guild_id)`` returns ``{channel_id: name}``."""
    cache: Dict[str, Mapping[str, str]] = {}
    for it in items:
        gid = it.get("guild_id")
        if not gid or not it.get("channel_id"):
            continue
        if gid not in cache:
            cache[gid] = names_for_guild(gid) or {}
        name = cache[gid].get(it["channel_id"])
        if name:
            it["channel_name"] = name
