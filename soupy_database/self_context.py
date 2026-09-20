"""
Soupy's self-knowledge files and the pieces of chat that read them.

Soupy's memory is built by :mod:`soupy_database.self_profile` (dated items from
its own messages, the same machinery as member profiles). That module renders
the memory into the files here, so everything that reads them is unchanged:

Files (per guild, in data/self_md/):
  guild_{id}.md          — the whole memory as markdown (/soupyself view, musing seeds)
  guild_{id}_core.md     — compact first-person summary (musings / bluesky personality context)
  guild_{id}_anchor.md   — the short identity line in every reply's system prompt
  guild_{id}_archive.md  — entries pruned by the old reflection; read-only now

RAG integration:
  The ``self_chunks`` table in each guild's SQLite DB holds one embedded row per
  memory item. Guilds whose memory hasn't been built yet still have chunks of the
  old SELF.MD there, searched by :func:`search_self_chunks`.

History: until 2026-09 a nightly "reflection" rewrote guild_{id}.md whole from
the last 60 replies. Its output hit the token cap and came back truncated daily,
so it is gone; the files it left are copied to data/self_md/v1_backup/ before a
guild's first memory build.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sqlite3
import struct
from pathlib import Path
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELF_MD_DIR = _PROJECT_ROOT / "data" / "self_md"


def _ensure_dir() -> None:
    SELF_MD_DIR.mkdir(parents=True, exist_ok=True)


def self_md_path(guild_id: int) -> Path:
    _ensure_dir()
    return SELF_MD_DIR / f"guild_{guild_id}.md"


def self_core_path(guild_id: int) -> Path:
    _ensure_dir()
    return SELF_MD_DIR / f"guild_{guild_id}_core.md"


def self_archive_path(guild_id: int) -> Path:
    _ensure_dir()
    return SELF_MD_DIR / f"guild_{guild_id}_archive.md"


def self_anchor_path(guild_id: int) -> Path:
    _ensure_dir()
    return SELF_MD_DIR / f"guild_{guild_id}_anchor.md"


# ---------------------------------------------------------------------------
# Feature toggle
# ---------------------------------------------------------------------------


def is_self_md_enabled() -> bool:
    return os.getenv("SELF_MD_ENABLED", "false").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Read / write helpers
# ---------------------------------------------------------------------------


def _read_file(p: Path) -> str:
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("self_context: failed to read %s: %s", p, exc)
    return ""


def _write_file(p: Path, content: str, backup: bool = True) -> None:
    if backup and p.exists():
        bak = p.with_suffix(p.suffix + ".bak")
        try:
            shutil.copy2(p, bak)
        except Exception as exc:
            logger.warning("self_context: backup failed for %s: %s", p, exc)
    p.write_text(content.strip() + "\n", encoding="utf-8")


def load_self_md(guild_id: int) -> str:
    return _read_file(self_md_path(guild_id))


def load_self_core(guild_id: int) -> str:
    return _read_file(self_core_path(guild_id))


def load_self_archive(guild_id: int) -> str:
    return _read_file(self_archive_path(guild_id))


def load_self_anchor(guild_id: int) -> str:
    return _read_file(self_anchor_path(guild_id))


def save_self_md(guild_id: int, content: str) -> None:
    _write_file(self_md_path(guild_id), content)
    logger.info("self_context: saved full doc %d chars for guild %s", len(content), guild_id)


def save_self_core(guild_id: int, content: str) -> None:
    _write_file(self_core_path(guild_id), content)
    logger.info("self_context: saved core %d chars for guild %s", len(content), guild_id)


def save_self_anchor(guild_id: int, content: str) -> None:
    _write_file(self_anchor_path(guild_id), content)
    logger.info("self_context: saved anchor %d chars for guild %s", len(content), guild_id)


# ---------------------------------------------------------------------------
# Self-knowledge DB schema (per-guild SQLite)
# ---------------------------------------------------------------------------


def ensure_self_chunks_schema(conn: sqlite3.Connection) -> None:
    """Create the self_chunks table for embedded self-knowledge."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS self_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,           -- 'item:<id>' (memory), or 'full' / 'archive' (old SELF.MD)
            section TEXT NOT NULL DEFAULT '',  -- memory section key, or an old SELF.MD header
            chunk_text TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_self_chunks_source
        ON self_chunks(source)
    """)
    conn.commit()


def _pack_embedding(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# Retrieval (called during RAG assembly)
# ---------------------------------------------------------------------------


def search_self_chunks(
    conn: sqlite3.Connection,
    query_embedding: Sequence[float],
    top_k: int = 3,
) -> List[Tuple[float, str, str]]:
    """Return the top-k self-knowledge chunks by cosine similarity.

    Returns: [(score, chunk_text, section), ...]
    """
    try:
        ensure_self_chunks_schema(conn)
    except Exception:
        return []

    cur = conn.cursor()
    try:
        cur.execute("SELECT chunk_text, section, embedding, embedding_dim FROM self_chunks")
    except sqlite3.OperationalError:
        return []
    rows = cur.fetchall()
    if not rows:
        return []

    qdim = len(query_embedding)
    scored: List[Tuple[float, str, str]] = []
    for r in rows:
        if int(r["embedding_dim"]) != qdim:
            continue
        vec = _unpack_embedding(r["embedding"])
        if len(vec) != qdim:
            continue
        s = _cosine(query_embedding, vec)
        scored.append((s, r["chunk_text"], r["section"]))

    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Injection helper (core only — always in system prompt)
# ---------------------------------------------------------------------------

_INJECTION_FRAME = (
    "\n\nSELF-KNOWLEDGE (your memories, opinions, relationships, and running jokes "
    "from past conversations — this is what you know and think based on experience; "
    "use it naturally as things you just know, never quote or refer to this as a document; "
    "if someone asks what you think about something and you have an opinion here, use it; "
    "your voice and format rules always take priority over how this is written; "
    "you may also have deeper self-knowledge that surfaces in retrieved context below):\n\n"
)


def get_self_md_for_injection(guild_id: int) -> str:
    """Return the always-on identity slug for the system prompt.

    Preference order:
      1. Anchor (`guild_<id>_anchor.md`) — the memory's first-person overview, cut to
         SELF_MD_ANCHOR_MAX_CHARS (600) by self_profile.write_views.
      2. Truncated core (paragraph-bounded, capped at SELF_MD_ANCHOR_FALLBACK_CHARS) —
         used when there is no anchor.
      3. Truncated full doc — final fallback.
      4. Empty string when nothing exists yet.

    Topical detail (specific people, jokes, opinions on specific things) is retrieved
    separately via the self-knowledge RAG (`self_chunks` table) when relevant — it does
    not need to live in the always-on system prompt.
    """
    if not is_self_md_enabled():
        return ""

    # 1. Anchor: real, LLM-distilled.
    content = load_self_anchor(guild_id)

    if not content:
        # 2/3. Fallback: truncate core, then full, at a paragraph boundary.
        max_fallback = int(os.getenv("SELF_MD_ANCHOR_FALLBACK_CHARS", "600"))
        source = load_self_core(guild_id) or load_self_md(guild_id)
        if not source:
            return ""
        if len(source) > max_fallback:
            cut = source[:max_fallback].rfind("\n")
            if cut > max_fallback // 2:
                content = source[:cut].rstrip()
            else:
                content = source[:max_fallback].rstrip()
        else:
            content = source

    return _INJECTION_FRAME + content + "\n"
