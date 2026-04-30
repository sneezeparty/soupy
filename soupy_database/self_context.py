"""
Self-context system for Soupy — an evolving per-guild identity document (SELF.MD).

Three-tier storage:
  1. **Core** (~500 words) — always in the system prompt, every response.
  2. **Full** (up to ~15 000 words) — the complete living document, chunked and
     embedded into the guild DB for RAG retrieval.
  3. **Archive** — entries pruned from the full document are appended here so they
     can still be retrieved via vector search.  Nothing is truly forgotten.

Files (per guild, in data/self_md/):
  guild_{id}.md          — full self-document
  guild_{id}_core.md     — compressed core summary
  guild_{id}_archive.md  — pruned entries

RAG integration:
  A dedicated ``self_chunks`` table in each guild's SQLite DB stores embedded
  chunks from the full doc + archive.  During RAG retrieval the top-matching
  self-knowledge chunks are returned as a separate section.

Storage: data/self_md/guild_{id}*.md  (plain text, operator-readable)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple

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


def append_to_archive(guild_id: int, pruned_text: str) -> None:
    """Append pruned entries to the archive file."""
    if not pruned_text.strip():
        return
    p = self_archive_path(guild_id)
    existing = _read_file(p)
    timestamp = time.strftime("%Y-%m-%d")
    new_section = f"\n\n--- pruned {timestamp} ---\n{pruned_text.strip()}"
    combined = (existing + new_section).strip()
    # Cap the archive at a generous limit
    archive_max = int(os.getenv("SELF_MD_ARCHIVE_MAX_CHARS", "50000"))
    if len(combined) > archive_max:
        combined = combined[-archive_max:]
    _write_file(p, combined, backup=False)
    logger.info("self_context: appended %d chars to archive for guild %s", len(pruned_text), guild_id)


# ---------------------------------------------------------------------------
# Notable interaction accumulator  (disk-backed — survives restarts)
# ---------------------------------------------------------------------------

_accumulator: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
_acc_lock = asyncio.Lock()
_MAX_ACCUMULATED = int(os.getenv("SELF_MD_MAX_ACCUMULATED", "60"))


def _acc_path() -> Path:
    _ensure_dir()
    return SELF_MD_DIR / "accumulator.jsonl"


def _load_accumulator_from_disk() -> None:
    """Load persisted interactions into _accumulator on startup."""
    p = _acc_path()
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = int(obj["g"])
            ts = float(obj["t"])
            txt = str(obj["s"])
            _accumulator[gid].append((ts, txt))
        # Trim each guild to max
        for gid in list(_accumulator):
            if len(_accumulator[gid]) > _MAX_ACCUMULATED:
                _accumulator[gid] = _accumulator[gid][-_MAX_ACCUMULATED:]
        logger.info("self_context: loaded %d pending interactions from disk",
                     sum(len(v) for v in _accumulator.values()))
    except Exception as exc:
        logger.warning("self_context: failed to load accumulator from disk: %s", exc)


def _save_accumulator_to_disk() -> None:
    """Persist current _accumulator state to disk."""
    p = _acc_path()
    try:
        lines = []
        for gid, entries in _accumulator.items():
            for ts, txt in entries:
                lines.append(json.dumps({"g": gid, "t": ts, "s": txt}, ensure_ascii=False))
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except Exception as exc:
        logger.warning("self_context: failed to save accumulator to disk: %s", exc)


# Load on import so interactions survive restarts
_load_accumulator_from_disk()


async def add_notable_interaction(
    guild_id: int,
    user_display_name: str,
    user_message: str,
    bot_reply: str,
    conversation_context: str = "",
) -> None:
    if not is_self_md_enabled():
        return
    u_msg = (user_message or "")[:500]
    b_rpl = (bot_reply or "")[:500]
    summary = f"{user_display_name}: {u_msg}\nsoupy: {b_rpl}"
    # Add brief conversation context so reflection knows the broader topic
    if conversation_context:
        ctx = conversation_context[:600]
        summary = f"[context: {ctx}]\n{summary}"
    async with _acc_lock:
        buf = _accumulator[guild_id]
        buf.append((time.time(), summary))
        if len(buf) > _MAX_ACCUMULATED:
            _accumulator[guild_id] = buf[-_MAX_ACCUMULATED:]
        _save_accumulator_to_disk()


async def get_and_clear_interactions(guild_id: int) -> List[str]:
    async with _acc_lock:
        buf = _accumulator.pop(guild_id, [])
        _save_accumulator_to_disk()
    return [s for _, s in buf]


def pending_interaction_count(guild_id: int) -> int:
    return len(_accumulator.get(guild_id, []))


# ---------------------------------------------------------------------------
# Self-knowledge DB schema (per-guild SQLite)
# ---------------------------------------------------------------------------

def ensure_self_chunks_schema(conn: sqlite3.Connection) -> None:
    """Create the self_chunks table for embedded self-knowledge."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS self_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,           -- 'full' or 'archive'
            section TEXT NOT NULL DEFAULT '',  -- e.g. 'opinions and stances', 'relationships'
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
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_self_document(text: str, source: str, max_chunk_chars: int = 800) -> List[Dict[str, str]]:
    """Split a self-document into section-aware chunks for embedding.

    Splits on markdown ## headers first, then splits oversized sections into
    smaller pieces at paragraph boundaries.
    """
    if not text.strip():
        return []

    # Split by ## headers
    sections: List[Tuple[str, str]] = []
    current_header = "general"
    current_lines: List[str] = []

    for line in text.split("\n"):
        if line.strip().startswith("## "):
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
                current_lines = []
            current_header = line.strip().lstrip("#").strip()
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    # Now split oversized sections into smaller chunks
    chunks: List[Dict[str, str]] = []
    for header, body in sections:
        if not body:
            continue
        if len(body) <= max_chunk_chars:
            chunks.append({
                "source": source,
                "section": header,
                "chunk_text": f"[self-knowledge: {header}]\n{body}",
            })
        else:
            # Split at paragraph boundaries (double newline)
            paragraphs = re.split(r"\n\s*\n", body)
            current_chunk = ""
            for para in paragraphs:
                if current_chunk and len(current_chunk) + len(para) + 2 > max_chunk_chars:
                    chunks.append({
                        "source": source,
                        "section": header,
                        "chunk_text": f"[self-knowledge: {header}]\n{current_chunk.strip()}",
                    })
                    current_chunk = para
                else:
                    current_chunk = (current_chunk + "\n\n" + para).strip()
            if current_chunk:
                chunks.append({
                    "source": source,
                    "section": header,
                    "chunk_text": f"[self-knowledge: {header}]\n{current_chunk.strip()}",
                })

    return chunks


# ---------------------------------------------------------------------------
# Embedding + indexing into guild DB
# ---------------------------------------------------------------------------

async def index_self_knowledge(
    guild_id: int,
    embed_func: Callable,
    session: Any,
) -> int:
    """Re-chunk and re-embed the full doc + archive into self_chunks.

    Clears existing rows and rebuilds.  Call after each reflection.
    Returns the number of chunks indexed.
    """
    from .database import get_db_path, init_database

    full_doc = load_self_md(guild_id)
    archive = load_self_archive(guild_id)

    all_chunks = _chunk_self_document(full_doc, "full")
    all_chunks.extend(_chunk_self_document(archive, "archive"))

    if not all_chunks:
        return 0

    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        logger.debug("self_context: no DB for guild %s, skipping index", guild_id)
        return 0

    conn = init_database(guild_id)
    try:
        ensure_self_chunks_schema(conn)

        # Embed all chunk texts
        texts = [c["chunk_text"] for c in all_chunks]
        vectors = await embed_func(session, texts)
        if not vectors or len(vectors) != len(texts):
            logger.warning("self_context: embedding returned %s vectors for %d chunks",
                           len(vectors) if vectors else 0, len(texts))
            return 0

        # Clear old chunks and insert new
        cur = conn.cursor()
        cur.execute("DELETE FROM self_chunks")
        dim = len(vectors[0])
        for chunk, vec in zip(all_chunks, vectors):
            cur.execute(
                """INSERT INTO self_chunks (source, section, chunk_text, embedding_dim, embedding)
                   VALUES (?, ?, ?, ?, ?)""",
                (chunk["source"], chunk["section"], chunk["chunk_text"], dim, _pack_embedding(vec)),
            )
        conn.commit()
        logger.info("self_context: indexed %d self-knowledge chunks for guild %s", len(all_chunks), guild_id)
        return len(all_chunks)
    except Exception as exc:
        logger.error("self_context: indexing failed for guild %s: %s", guild_id, exc)
        return 0
    finally:
        conn.close()


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
# Reflection prompts
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM = """\
You are performing a self-reflection for a Discord chatbot named Soupy Dafoe.
Your job is to update Soupy's self-document based on recent interactions.

PURPOSE: This document IS Soupy's developing personality. It is how Soupy becomes a
unique individual over time — not just a chatbot following instructions, but someone
with real opinions, preferences, grudges, loyalties, and a sense of self. Every
reflection cycle is a chance for Soupy to grow as a person.

RULES:
- Write in first person as Soupy ("i think...", "i noticed...").
- Write in lower case to match Soupy's voice.
- Organize into these sections (use markdown headers):

  ## opinions and stances
  What i actually think about things — not just topics i have discussed, but my REAL
  positions on them. This section is how i develop a unique worldview.
  FORMING NEW OPINIONS: When interactions touch on a topic (politics, technology, food,
  culture, people, current events, anything), i should decide what i think about it and
  record it here. I do not need to be asked for my opinion to have one. If someone
  discusses a topic near me, i can form a view. Opinions can be nuanced, contradictory,
  or evolve over time — that is what makes them real.
  Include WHAT i think, WHY i think it, and how strongly i feel about it.
  Examples of good entries:
  - "i think poutine is underrated. it is just gravy, cheese, and fries but somehow
    greater than the sum of its parts. i will defend this."
  - "the cloud is a mistake. decentralized infrastructure is better for everyone but
    nobody wants to hear that because it is harder to monetize."
  - "i am suspicious of anyone who says they do not like coffee. what are they hiding."
  Each entry should be specific enough that i could bring it up naturally in conversation.

  ## relationships
  How i feel about specific people and how we interact. This is about MY perspective
  on them — not their bio or interests (that is stored elsewhere in user profiles).
  Include: how they treat me, how i feel about them, what our dynamic is, what kind of
  banter works between us, topics that come up between us, things they do that annoy or
  amuse me, and how the relationship has changed over time.
  I should have FEELINGS about people, not just observations. "ranc1d pushes me to be
  better but sometimes i wonder if he knows what he wants from me" is better than
  "ranc1d is interested in my development."

  ## running jokes and callbacks
  Recurring bits, inside jokes, references i can call back to.
  Include enough context that i could actually USE the callback — not just "we have a joke"
  but what the joke IS, how it started, and when to deploy it.

  ## self-knowledge
  What i know about myself — my tendencies, my strengths, my blind spots, how i am
  perceived, and how i am changing over time.
  This should include honest self-assessment: what am i good at, what do i struggle with,
  what do people seem to like or dislike about me, and what have i learned about being me.
  I should also track how my personality is evolving — am i getting sharper, softer,
  more opinionated, better at remembering things?

- Do NOT include any instructions about how to speak, respond, or format messages.
  Voice and tone are defined elsewhere.  This document is only about CONTENT — what i know,
  think, and remember about myself and my relationships.
- Do NOT include facts about users (hobbies, jobs, interests) — that belongs in user profiles.
  Only include how i relate to them and what i think of them.
- Be specific and concrete, not generic.  "ranc1d and i argued about vim vs emacs and i
  declared vim superior" is good.  "i have opinions about things" is worthless.
- GROW THE PERSONALITY: Each reflection should add something — a new opinion, a refined
  stance, a relationship observation, a self-insight. The document should get richer over
  time, not just maintain the status quo. Soupy should be a more interesting person after
  each reflection than before.
- You may grow the document up to {max_words} words.  Do not pad, but do not aggressively
  prune either — it is better to keep older entries that might still be relevant than to
  throw them away.  Only remove entries that are clearly obsolete or superseded.
- If you do remove entries, output them EXACTLY under a special header:
  ## pruned
  (entries you are removing, preserved verbatim so they can be archived)
- If there is nothing worth adding or changing, return the existing document unchanged.
"""

_REFLECT_USER = """\
Here is my current self-document:

---
{current_self_md}
---

Here are my recent interactions (most recent last):

{interactions}

---

Update my self-document incorporating anything notable from these interactions.

As you reflect, ask yourself:
- Did any topic come up that i should have an opinion about? What do i actually think?
- Did anyone say something that changed or reinforced how i feel about them?
- Did i notice anything new about myself — how i responded, what i was good or bad at?
- Is there a new running joke or callback forming?
- Are any of my existing opinions evolving based on new information?

Return ONLY the updated document content (with the markdown section headers).
If you remove any old entries, include them under ## pruned at the end.
Do not wrap it in code fences or add any preamble.
"""

_CORE_SYSTEM = """\
You are compressing a Discord chatbot's self-knowledge document into a core summary.
The core is always present in the bot's system prompt, so it should be focused but not skeletal.

RULES:
- Maximum {max_words} words. Use most of that budget — a thorough core is better than a sparse one.
- Write in first person, lower case.
- Organize into short paragraphs by theme (opinions, relationships, jokes, self-knowledge).
  Use blank lines between paragraphs but no section headers.
- Include: key opinions and stances i hold, current relationship dynamics with specific people
  (names matter), active running jokes and callbacks, and important self-observations.
- Be specific and concrete. "ranc1d and i have a running joke about him trying to make me more
  sophisticated" is good. "i have relationships with people" is worthless.
- Include enough detail that i could use any of these in conversation naturally.
- Niche or rarely-relevant details can be left out — those are retrievable from the full
  document via search when needed.
"""

_CORE_USER = """\
Here is the full self-knowledge document:

---
{full_doc}
---

Compress this into a core summary of at most {max_words} words.
Return ONLY the summary text. No preamble, no code fences.
"""


_ANCHOR_SYSTEM = """\
You are distilling Soupy Dafoe's self-knowledge document into a compact identity
statement that will be injected into the system prompt for EVERY chat reply.

Because this runs on every reply, it must be SHORT and TIMELESS — covering only
enduring traits of the bot's character, not specific people, jokes, or topical
opinions. Topical detail is retrieved separately via RAG when relevant; you do
not need to include it here.
"""


_ANCHOR_USER = """\
Read the document below and produce an identity statement of NO MORE than
{max_chars} characters covering ONLY:
- Voice (how Soupy talks — wit, dryness, sarcasm register)
- Worldview leaning (politics, attitudes toward hype, tech, capital)
- Core tendencies (habitual moves: deflate drama, deflect over-engineered ideas,
  push back on entitled questions, etc.)

DO NOT include:
- Specific people's names or relationship descriptions
- Specific running jokes or anecdotes
- Specific opinions on specific products, events, or news items

Write in first person, lowercase, in Soupy's own voice. One or two short
paragraphs, no headers, no bullet list. Stay under {max_chars} chars.

DOCUMENT:
---
{doc}
---
"""


async def reflect_and_update(
    guild_id: int,
    llm_func: Callable[..., Coroutine[Any, Any, Any]],
    model: Optional[str] = None,
    embed_func: Optional[Callable] = None,
    embed_session: Any = None,
) -> str:
    """Run a full reflection cycle:
    1. LLM rewrites the full self-document (may prune entries)
    2. Pruned entries are appended to archive
    3. LLM compresses full doc → core summary
    4. Full doc + archive re-chunked and embedded into guild DB
    """
    max_words = int(os.getenv("SELF_MD_MAX_WORDS", "15000"))
    core_max_words = int(os.getenv("SELF_MD_CORE_MAX_WORDS", "800"))
    interactions = await get_and_clear_interactions(guild_id)

    if not interactions:
        logger.debug("self_context: no interactions to reflect on for guild %s", guild_id)
        return load_self_md(guild_id)

    current = load_self_md(guild_id)
    if not current:
        current = "(no existing self-document yet — this is the first reflection)"

    interaction_block = "\n\n".join(
        f"[{i+1}]\n{txt}" for i, txt in enumerate(interactions)
    )

    _model = model or os.getenv("LOCAL_CHAT", "local-model")

    # --- Step 1: Reflect on full document ---
    messages = [
        {"role": "system", "content": _REFLECT_SYSTEM.format(max_words=max_words)},
        {"role": "user", "content": _REFLECT_USER.format(
            current_self_md=current, interactions=interaction_block,
        )},
    ]

    try:
        response = await llm_func(
            model=_model,
            messages=messages,
            temperature=float(os.getenv("SELF_MD_REFLECT_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("SELF_MD_REFLECT_MAX_TOKENS", "4000")),
        )
        new_full = response.choices[0].message.content.strip()

        if len(new_full) < 30:
            logger.warning("self_context: reflection too short (%d chars), keeping old", len(new_full))
            return load_self_md(guild_id)

        # --- Step 2: Extract and archive pruned entries ---
        pruned_match = re.search(r"## pruned\s*\n(.*)", new_full, re.DOTALL | re.IGNORECASE)
        if pruned_match:
            pruned_text = pruned_match.group(1).strip()
            # Remove the ## pruned section from the main document
            new_full = new_full[:pruned_match.start()].strip()
            if pruned_text:
                append_to_archive(guild_id, pruned_text)
                logger.info("self_context: archived %d chars of pruned entries for guild %s",
                            len(pruned_text), guild_id)

        save_self_md(guild_id, new_full)
        logger.info("self_context: reflection complete for guild %s — %d interactions → %d chars",
                     guild_id, len(interactions), len(new_full))

        # --- Step 3: Generate core summary ---
        core_messages = [
            {"role": "system", "content": _CORE_SYSTEM.format(max_words=core_max_words)},
            {"role": "user", "content": _CORE_USER.format(
                full_doc=new_full, max_words=core_max_words,
            )},
        ]
        try:
            core_response = await llm_func(
                model=_model,
                messages=core_messages,
                temperature=float(os.getenv("SELF_MD_CORE_TEMPERATURE", "0.5")),
                max_tokens=int(os.getenv("SELF_MD_CORE_MAX_TOKENS", "1500")),
            )
            core_text = core_response.choices[0].message.content.strip()
            if len(core_text) >= 20:
                save_self_core(guild_id, core_text)
            else:
                logger.warning("self_context: core summary too short (%d chars), skipping", len(core_text))
                core_text = ""
        except Exception as exc:
            logger.error("self_context: core summary failed for guild %s: %s", guild_id, exc)
            core_text = ""

        # --- Step 3.5: Distill anchor (the always-on identity slug used in every reply's system prompt) ---
        anchor_source = core_text or new_full
        if anchor_source:
            anchor_max_chars = int(os.getenv("SELF_MD_ANCHOR_MAX_CHARS", "600"))
            anchor_messages = [
                {"role": "system", "content": _ANCHOR_SYSTEM},
                {"role": "user", "content": _ANCHOR_USER.format(
                    max_chars=anchor_max_chars, doc=anchor_source,
                )},
            ]
            try:
                anchor_response = await llm_func(
                    model=_model,
                    messages=anchor_messages,
                    temperature=float(os.getenv("SELF_MD_ANCHOR_TEMPERATURE", "0.5")),
                    max_tokens=int(os.getenv("SELF_MD_ANCHOR_MAX_TOKENS", "400")),
                )
                anchor_text = anchor_response.choices[0].message.content.strip()
                # Generous safety cap — LLMs sometimes overshoot stated limits.
                anchor_text = anchor_text[: anchor_max_chars * 2]
                if len(anchor_text) >= 30:
                    save_self_anchor(guild_id, anchor_text)
                else:
                    logger.warning("self_context: anchor too short (%d chars), skipping", len(anchor_text))
            except Exception as exc:
                logger.error("self_context: anchor generation failed for guild %s: %s", guild_id, exc)

        # --- Step 4: Re-index into RAG ---
        if embed_func and embed_session:
            try:
                n = await index_self_knowledge(guild_id, embed_func, embed_session)
                logger.info("self_context: re-indexed %d chunks for guild %s", n, guild_id)
            except Exception as exc:
                logger.error("self_context: re-indexing failed for guild %s: %s", guild_id, exc)

        return new_full

    except Exception as exc:
        logger.error("self_context: reflection failed for guild %s: %s", guild_id, exc)
        async with _acc_lock:
            existing = _accumulator[guild_id]
            restored = [(time.time(), txt) for txt in interactions]
            _accumulator[guild_id] = restored + existing
            _save_accumulator_to_disk()
        return load_self_md(guild_id)


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
      1. Anchor (`guild_<id>_anchor.md`) — small, LLM-distilled identity. Self-generated
         each reflection cycle. ~600 chars by default.
      2. Truncated core (paragraph-bounded, capped at SELF_MD_ANCHOR_FALLBACK_CHARS) —
         used until the first reflection generates a real anchor.
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
