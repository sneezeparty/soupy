"""
Local RAG over per-guild scanned messages: chunking, LM Studio embeddings, SQLite storage, similarity search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sqlite3
import struct
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

from .database import ensure_rag_schema, get_db_path, init_database

logger = logging.getLogger(__name__)

# Serialize embedding HTTP calls so archive reindex + live chat RAG do not stampede LM Studio.
_embed_sem: Optional[asyncio.Semaphore] = None


def _get_embed_sem() -> asyncio.Semaphore:
    global _embed_sem
    if _embed_sem is None:
        try:
            n = int(os.getenv("RAG_EMBED_MAX_CONCURRENT", "2"))
        except ValueError:
            n = 2
        _embed_sem = asyncio.Semaphore(max(1, n))
    return _embed_sem


_reindex_locks: Dict[int, asyncio.Lock] = {}


def _reindex_lock(guild_id: int) -> asyncio.Lock:
    if guild_id not in _reindex_locks:
        _reindex_locks[guild_id] = asyncio.Lock()
    return _reindex_locks[guild_id]


def _rag_log_verbose() -> bool:
    """Per-hit / long previews on INFO. Set RAG_LOG_VERBOSE=0 to keep only one-line summaries."""
    return os.getenv("RAG_LOG_VERBOSE", "1").strip() not in ("0", "false", "no", "off")


def _preview_for_log(text: str, max_len: int = 220) -> str:
    one = (text or "").replace("\n", " ").strip()
    if len(one) > max_len:
        return one[: max_len - 1] + "…"
    return one


def pack_embedding(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_embedding(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _parse_msg_dt(date_s: str, time_s: str) -> datetime:
    return datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d %H:%M:%S")


def _line_for_row(row: sqlite3.Row) -> str:
    name = row["nickname"] or row["username"]
    uid = row["user_id"]
    parts: List[str] = []
    if row["message_content"]:
        parts.append(str(row["message_content"]))
    if row["image_description"]:
        parts.append(f"[image: {row['image_description']}]")
    if row["url_summary"]:
        parts.append(f"[link: {row['url_summary']}]")
    body = " ".join(parts) if parts else ""
    max_c = int(os.getenv("RAG_MAX_CHARS_PER_LINE", "800"))
    if len(body) > max_c:
        body = body[: max_c - 1] + "…"
    return f"{name} (user_id={uid}): {body}"


def _iter_conversation_chunks(
    conn: sqlite3.Connection,
    messages_per_chunk: int = 6,
    gap_minutes: int = 30,
) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT message_id, date, time, username, nickname, user_id, message_content,
               channel_id, channel_name, image_description, url_summary
        FROM messages
        ORDER BY date ASC, time ASC, message_id ASC
        """
    )
    rows = cur.fetchall()
    if not rows:
        return []

    chunks: List[Dict[str, Any]] = []
    group: List[sqlite3.Row] = []
    last_dt: Optional[datetime] = None
    last_channel: Optional[int] = None

    def flush_group() -> None:
        nonlocal group
        if not group:
            return
        ch_id = group[0]["channel_id"]
        ch_name = group[0]["channel_name"]
        for i in range(0, len(group), messages_per_chunk):
            sub = group[i : i + messages_per_chunk]
            lines = [_line_for_row(r) for r in sub]
            header = f"[#{ch_name} channel_id={ch_id}]"
            text = header + "\n" + "\n".join(lines)
            chunks.append(
                {
                    "first_message_id": sub[0]["message_id"],
                    "last_message_id": sub[-1]["message_id"],
                    "channel_id": ch_id,
                    "channel_name": ch_name,
                    "chunk_text": text,
                }
            )
        group = []

    gap = timedelta(minutes=gap_minutes)
    for row in rows:
        dt = _parse_msg_dt(row["date"], row["time"])
        cid = row["channel_id"]
        if last_dt is not None and last_channel is not None:
            if cid != last_channel or dt - last_dt > gap:
                flush_group()
        group.append(row)
        last_dt = dt
        last_channel = cid
    flush_group()
    return chunks


def get_rag_chunk_count(guild_id: int) -> int:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_rag_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM rag_chunks")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


async def embed_texts_lm_studio(
    session: aiohttp.ClientSession,
    texts: List[str],
) -> List[List[float]]:
    if not texts:
        return []
    async with _get_embed_sem():
        return await _embed_texts_lm_studio_inner(session, texts)


async def _embed_texts_lm_studio_inner(
    session: aiohttp.ClientSession,
    texts: List[str],
) -> List[List[float]]:
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    endpoint = f"{base}/embeddings"
    model = os.getenv("RAG_EMBEDDING_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "RAG_EMBEDDING_MODEL is not set. In LM Studio load an embedding model and set this env to its exact id."
        )
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: Dict[str, Any] = {"model": model, "input": texts}
    try:
        batch_max = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
    except ValueError:
        batch_max = 8

    all_out: List[List[float]] = []
    for start in range(0, len(texts), batch_max):
        batch = texts[start : start + batch_max]
        payload["input"] = batch if len(batch) > 1 else batch[0]
        async with session.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as r:
            body = await r.text()
            if r.status != 200:
                raise RuntimeError(f"embeddings HTTP {r.status}: {body[:500]}")
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"embeddings invalid JSON: {e}") from e
        items_raw = data.get("data")
        if not isinstance(items_raw, list):
            raise RuntimeError("embeddings response missing data[]")
        if items_raw and all("index" not in it for it in items_raw):
            items = items_raw
        else:
            items = sorted(items_raw, key=lambda x: x.get("index", 0))
        if len(items) != len(batch):
            raise RuntimeError(f"expected {len(batch)} embeddings, got {len(items)}")
        for it in items:
            emb = it.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("embedding entry missing 'embedding' list")
            all_out.append([float(x) for x in emb])
        # Yield the event loop so Discord gateway / other tasks stay responsive during long reindexes.
        await asyncio.sleep(0)
    if len(all_out) != len(texts):
        raise RuntimeError(f"embedding total mismatch: {len(all_out)} vs {len(texts)}")
    return all_out


def _reindex_sync_clear_and_build_chunks(guild_id: int) -> List[Dict[str, Any]]:
    """SQLite-heavy work for RAG rebuild; safe to run in a thread pool."""
    conn = init_database(guild_id)
    try:
        ensure_rag_schema(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM rag_chunks")
        conn.commit()
        return _iter_conversation_chunks(conn)
    finally:
        conn.close()


def _reindex_sync_write_chunks(
    guild_id: int,
    chunk_dicts: List[Dict[str, Any]],
    vectors: List[List[float]],
) -> Dict[str, Any]:
    """Persist embeddings; runs in a thread pool."""
    conn = init_database(guild_id)
    try:
        ensure_rag_schema(conn)
        cur = conn.cursor()
        dim = len(vectors[0])
        for c, vec in zip(chunk_dicts, vectors):
            if len(vec) != dim:
                raise RuntimeError("inconsistent embedding dimensions")
            cur.execute(
                """
                INSERT INTO rag_chunks (
                    first_message_id, last_message_id, channel_id, channel_name,
                    chunk_text, embedding_dim, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c["first_message_id"],
                    c["last_message_id"],
                    c["channel_id"],
                    c["channel_name"],
                    c["chunk_text"],
                    dim,
                    pack_embedding(vec),
                ),
            )
        conn.commit()
        logger.info("RAG reindex guild=%s chunks=%s dim=%s", guild_id, len(chunk_dicts), dim)
        return {"ok": True, "chunks": len(chunk_dicts), "embedding_dim": dim}
    finally:
        conn.close()


async def reindex_guild_rag(guild_id: int) -> Dict[str, Any]:
    """Replace all rag_chunks for a guild using current messages table."""
    async with _reindex_lock(guild_id):
        loop = asyncio.get_running_loop()
        chunk_dicts = await loop.run_in_executor(
            None, _reindex_sync_clear_and_build_chunks, guild_id
        )
        if not chunk_dicts:
            return {"ok": True, "chunks": 0, "message": "No messages to index."}

        texts = [c["chunk_text"] for c in chunk_dicts]
        async with aiohttp.ClientSession() as session:
            vectors = await embed_texts_lm_studio(session, texts)
        if len(vectors) != len(chunk_dicts):
            raise RuntimeError("embedding count mismatch")

        return await loop.run_in_executor(
            None, _reindex_sync_write_chunks, guild_id, chunk_dicts, vectors
        )


async def index_message_immediate(
    guild_id: int,
    message_id: int,
    message_content: str,
    username: str,
    nickname: Optional[str],
    user_id: int,
    channel_id: int,
    channel_name: str,
    msg_date: str,
    msg_time: str,
) -> bool:
    """
    Store a single plain-text message to the guild SQLite DB and immediately embed + index it.
    Skips silently if the message already exists. Returns True if newly indexed.
    Image/URL enrichment is intentionally omitted — the scheduled scan handles that.
    """
    from .database import insert_message, message_exists

    loop = asyncio.get_running_loop()

    def _store() -> bool:
        conn = init_database(guild_id)
        try:
            if message_exists(conn, message_id):
                return False
            insert_message(
                conn, message_id, msg_date, msg_time,
                username, nickname, user_id,
                message_content, channel_id, channel_name,
            )
            return True
        finally:
            conn.close()

    stored = await loop.run_in_executor(None, _store)
    if not stored:
        return False

    # Build single-message chunk in the same format as _iter_conversation_chunks
    display_name = nickname or username
    body = message_content or ""
    max_c = int(os.getenv("RAG_MAX_CHARS_PER_LINE", "800"))
    if len(body) > max_c:
        body = body[: max_c - 1] + "…"
    chunk_text = f"[#{channel_name} channel_id={channel_id}]\n{display_name} (user_id={user_id}): {body}"

    try:
        async with aiohttp.ClientSession() as session:
            vectors = await embed_texts_lm_studio(session, [chunk_text])
        vec = vectors[0]
        dim = len(vec)
    except Exception as exc:
        logger.warning("index_message_immediate: embedding failed guild=%s msg=%s: %s", guild_id, message_id, exc)
        return False

    def _index() -> None:
        conn = init_database(guild_id)
        try:
            ensure_rag_schema(conn)
            conn.execute(
                """
                INSERT OR IGNORE INTO rag_chunks
                    (first_message_id, last_message_id, channel_id, channel_name,
                     chunk_text, embedding_dim, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, message_id, channel_id, channel_name,
                 chunk_text, dim, pack_embedding(vec)),
            )
            conn.commit()
        finally:
            conn.close()

    await loop.run_in_executor(None, _index)
    logger.debug("index_message_immediate: indexed msg=%s guild=%s #%s", message_id, guild_id, channel_name)
    return True


async def index_new_messages(guild_id: int) -> Dict[str, Any]:
    """
    Incrementally update the RAG index — the only routine that should run after scans
    and on the periodic consolidation timer.

    Handles two cases in one pass:
      1. Messages in the DB with no chunk coverage yet (added by scan, or channels
         that were never indexed).  Detected via per-channel watermark: the max
         last_message_id already in rag_chunks.
      2. Single-message chunks from real-time indexing that need to be merged into
         proper grouped conversation chunks.

    For each affected channel the function expands the time window by ±30 min,
    deletes only the chunks that overlap that window, then re-chunks and embeds
    just those messages.  Everything outside the window is untouched.
    """
    loop = asyncio.get_running_loop()
    gap = timedelta(minutes=30)

    def _find_work() -> Dict[int, Dict[str, Any]]:
        conn = init_database(guild_id)
        try:
            ensure_rag_schema(conn)
            cur = conn.cursor()

            # Per-channel watermark: highest message_id already in a chunk
            cur.execute(
                "SELECT channel_id, MAX(last_message_id) AS watermark FROM rag_chunks GROUP BY channel_id"
            )
            watermarks: Dict[int, int] = {r["channel_id"]: r["watermark"] for r in cur.fetchall()}

            # Messages not yet covered (newer than watermark, or channel has no chunks)
            cur.execute(
                """
                SELECT m.message_id, m.channel_id, m.channel_name, m.date, m.time
                FROM messages m
                LEFT JOIN (
                    SELECT channel_id, MAX(last_message_id) AS watermark
                    FROM rag_chunks GROUP BY channel_id
                ) w ON m.channel_id = w.channel_id
                WHERE w.watermark IS NULL OR m.message_id > w.watermark
                ORDER BY m.channel_id, m.date ASC, m.time ASC, m.message_id ASC
                """
            )
            uncovered: Dict[int, List[Dict]] = {}
            for r in cur.fetchall():
                uncovered.setdefault(r["channel_id"], []).append(dict(r))

            # Single-message chunks that need consolidation
            cur.execute(
                """
                SELECT rc.first_message_id, rc.channel_id, rc.channel_name, m.date, m.time
                FROM rag_chunks rc
                JOIN messages m ON m.message_id = rc.first_message_id
                WHERE rc.first_message_id = rc.last_message_id
                """
            )
            singles: Dict[int, List[Dict]] = {}
            for r in cur.fetchall():
                singles.setdefault(r["channel_id"], []).append(dict(r))

            all_channels = set(uncovered) | set(singles)
            work: Dict[int, Dict[str, Any]] = {}
            for cid in all_channels:
                unc = uncovered.get(cid, [])
                sng = singles.get(cid, [])
                if not unc and not sng:
                    continue
                ch_name = (unc or sng)[0]["channel_name"]
                work[cid] = {"channel_name": ch_name, "uncovered": unc, "singles": sng}
            return work
        finally:
            conn.close()

    work = await loop.run_in_executor(None, _find_work)
    if not work:
        return {"ok": True, "new_messages": 0, "consolidated": 0, "new_chunks": 0}

    total_new = sum(len(v["uncovered"]) for v in work.values())
    total_singles = sum(len(v["singles"]) for v in work.values())

    def _process_channel(channel_id: int, channel_name: str,
                         uncovered: List[Dict], singles: List[Dict]) -> List[Dict[str, Any]]:
        conn = init_database(guild_id)
        try:
            all_rows = uncovered + singles
            dts = [_parse_msg_dt(r["date"], r["time"]) for r in all_rows]
            min_dt = min(dts) - gap
            max_dt = max(dts) + gap

            cur = conn.cursor()
            cur.execute(
                """
                SELECT message_id, date, time, username, nickname, user_id,
                       message_content, channel_id, channel_name,
                       image_description, url_summary
                FROM messages
                WHERE channel_id = ?
                  AND date >= ? AND date <= ?
                ORDER BY date ASC, time ASC, message_id ASC
                """,
                (channel_id, min_dt.strftime("%Y-%m-%d"), max_dt.strftime("%Y-%m-%d")),
            )
            in_window = [r for r in cur.fetchall()
                         if min_dt <= _parse_msg_dt(r["date"], r["time"]) <= max_dt]
            if not in_window:
                return []

            # Delete only chunks that overlap this window
            window_ids = [r["message_id"] for r in in_window]
            ph = ",".join("?" * len(window_ids))
            conn.execute(
                f"""
                DELETE FROM rag_chunks
                WHERE channel_id = ?
                  AND (first_message_id IN ({ph}) OR last_message_id IN ({ph}))
                """,
                (channel_id, *window_ids, *window_ids),
            )
            conn.commit()

            # Re-chunk with conversation grouping
            ch_name = in_window[0]["channel_name"]
            new_chunks: List[Dict[str, Any]] = []
            group: List = []
            last_dt: Optional[datetime] = None

            def flush() -> None:
                nonlocal group
                if not group:
                    return
                for i in range(0, len(group), 6):
                    sub = group[i : i + 6]
                    lines = [_line_for_row(r) for r in sub]
                    new_chunks.append({
                        "first_message_id": sub[0]["message_id"],
                        "last_message_id": sub[-1]["message_id"],
                        "channel_id": channel_id,
                        "channel_name": ch_name,
                        "chunk_text": f"[#{ch_name} channel_id={channel_id}]\n" + "\n".join(lines),
                    })
                group = []

            for row in in_window:
                dt = _parse_msg_dt(row["date"], row["time"])
                if last_dt is not None and dt - last_dt > gap:
                    flush()
                group.append(row)
                last_dt = dt
            flush()
            return new_chunks
        finally:
            conn.close()

    all_new_chunks: List[Dict[str, Any]] = []
    for channel_id, channel_work in work.items():
        chunks = await loop.run_in_executor(
            None, _process_channel,
            channel_id, channel_work["channel_name"],
            channel_work["uncovered"], channel_work["singles"],
        )
        all_new_chunks.extend(chunks)

    if not all_new_chunks:
        return {"ok": True, "new_messages": total_new, "consolidated": total_singles, "new_chunks": 0}

    texts = [c["chunk_text"] for c in all_new_chunks]
    async with aiohttp.ClientSession() as session:
        vectors = await embed_texts_lm_studio(session, texts)

    def _write() -> None:
        conn = init_database(guild_id)
        try:
            ensure_rag_schema(conn)
            for c, vec in zip(all_new_chunks, vectors):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO rag_chunks
                        (first_message_id, last_message_id, channel_id, channel_name,
                         chunk_text, embedding_dim, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (c["first_message_id"], c["last_message_id"],
                     c["channel_id"], c["channel_name"],
                     c["chunk_text"], len(vec), pack_embedding(vec)),
                )
            conn.commit()
        finally:
            conn.close()

    await loop.run_in_executor(None, _write)
    logger.info(
        "index_new_messages guild=%s: +%d new msg(s), %d single(s) consolidated → %d chunk(s)",
        guild_id, total_new, total_singles, len(all_new_chunks),
    )
    return {"ok": True, "new_messages": total_new, "consolidated": total_singles, "new_chunks": len(all_new_chunks)}


def search_rag_chunks(
    conn: sqlite3.Connection,
    query_embedding: Sequence[float],
    top_k: int,
    prefer_user_id: Optional[int] = None,
    user_boost_multiplier: float = 1.0,
) -> List[Tuple[float, str, Tuple[int, int, str]]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_text, first_message_id, last_message_id, channel_name, embedding, embedding_dim
        FROM rag_chunks
        """
    )
    rows = cur.fetchall()
    qdim = len(query_embedding)
    scored: List[Tuple[float, str, Tuple[int, int, str]]] = []
    uid_s = str(prefer_user_id) if prefer_user_id is not None else None
    try:
        boost = float(os.getenv("RAG_USER_MATCH_BOOST", "0.05"))
    except ValueError:
        boost = 0.05
    for r in rows:
        if int(r["embedding_dim"]) != qdim:
            continue
        vec = unpack_embedding(r["embedding"])
        if len(vec) != qdim:
            continue
        s = _cosine(query_embedding, vec)
        if uid_s and uid_s in (r["chunk_text"] or ""):
            s += boost * float(user_boost_multiplier)
        scored.append((s, r["chunk_text"], (int(r["first_message_id"]), int(r["last_message_id"]), r["channel_name"])))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def _log_rag_bundle(prof_out: int, kw_out: int, vec_n: int, total: int, cap: int, truncated: bool, note: str = "") -> None:
    trunc_tag = " ✗ TRUNCATED" if truncated else " ✓"
    note_tag = f" ({note})" if note else ""
    logger.info(
        "  bundle   : profile=%d + keyword=%d + %d vector(s) → %d / %d chars%s%s",
        prof_out, kw_out, vec_n, total, cap, trunc_tag, note_tag,
    )


def _assemble_rag_bundle(
    profile_prefix: str,
    supplement: str,
    hits: List[Tuple[float, str, Tuple[int, int, str]]],
    max_chars: int,
    keyword_max: int,
    self_hits: Optional[List[Tuple[float, str, str]]] = None,
) -> Tuple[str, int, int, int, bool, int]:
    """
    Optional profile sketch (soft context), then self-knowledge, then keyword/SQL block,
    then vector chunks.
    Returns: (text, profile_chars_in_output, keyword_chars_in_output, vector_chunks_in_output,
              truncated, raw_len_if_unlimited).
    """
    chunks: List[str] = []
    prof = (profile_prefix or "").strip()
    profile_len = 0
    if prof:
        try:
            pmax = int(os.getenv("RAG_PROFILE_MAX_CHARS", "2500"))
        except ValueError:
            pmax = 2500
        if len(prof) > pmax:
            prof = prof[: pmax - 1] + "…"
        chunks.append(prof)
        profile_len = len(prof)
    # Self-knowledge section (from Soupy's own identity/memory)
    if self_hits:
        try:
            self_max = int(os.getenv("RAG_SELF_KNOWLEDGE_MAX_CHARS", "2500"))
        except ValueError:
            self_max = 1500
        self_pieces: List[str] = []
        self_total = 0
        for score, text, section in self_hits:
            if self_total + len(text) > self_max:
                break
            self_pieces.append(text)
            self_total += len(text)
        if self_pieces:
            self_block = (
                "--- your own memories and self-knowledge (use naturally, as things you remember) ---\n"
                + "\n\n".join(self_pieces)
            )
            chunks.append(self_block)
    sup = supplement or ""
    kw_piece = ""
    if sup:
        kw_piece = sup[: keyword_max - 1] + "…" if len(sup) > keyword_max else sup
        chunks.append(kw_piece)
    kw_out = len(kw_piece)
    for i, (score, text, (mid_lo, mid_hi, ch)) in enumerate(hits, 1):
        chunks.append(
            f"--- related conversation {i} (relevance={score:.3f}, channel #{ch}, messages {mid_lo}-{mid_hi}) ---\n{text}"
        )
    out = "\n\n".join(chunks)
    raw_len = len(out)
    truncated = False
    if len(out) > max_chars:
        out = out[: max_chars - 1] + "…"
        truncated = True
    vec_in = len(hits) if hits else 0
    return out, profile_len, kw_out, vec_in, truncated, raw_len


def strip_rag_gate_word(content: str) -> str:
    return re.sub(r"\bRAG\b", "", content or "").strip()


def strip_rag_query_invocations(query_text: str) -> str:
    """
    Remove leading wake words (default: soupy) so embeddings and keyword search target the
    actual question, not the bot name. Does not remove names that appear later
    (e.g. 'what did soupy say about tariffs').
    Comma-separated extra prefixes: RAG_STRIP_QUERY_PREFIXES (default: soupy).
    """
    s = (query_text or "").strip()
    if not s:
        return ""
    raw = os.getenv("RAG_STRIP_QUERY_PREFIXES", "soupy").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        parts = ["soupy"]
    names = "|".join(re.escape(p) for p in parts)
    pat = re.compile(rf"^\s*(?:{names})\b\s*[:,]?\s*", re.IGNORECASE)
    while True:
        ns = pat.sub("", s, count=1)
        if ns == s:
            break
        s = ns.strip()
    return s


def build_rag_retrieval_query(
    recent_llm_messages: Sequence[Dict[str, Any]],
    current_display_name: str,
    current_body: str,
) -> str:
    """
    Combine recent channel turns (oldest→newest) with the current message so embeddings
    and keyword search see follow-ups like \"what about that?\" in context.
    Env: RAG_QUERY_RECENT_TURNS (default 14), RAG_QUERY_MAX_CHARS (default 2400).
    """
    try:
        max_turns = int(os.getenv("RAG_QUERY_RECENT_TURNS", "14"))
    except ValueError:
        max_turns = 14
    if max_turns < 0:
        max_turns = 0
    try:
        max_chars = int(os.getenv("RAG_QUERY_MAX_CHARS", "2400"))
    except ValueError:
        max_chars = 2400
    max_chars = max(400, max_chars)

    cur_raw = (current_body or "").strip()
    cur = strip_rag_gate_word(current_body or "").strip()
    cur = strip_rag_query_invocations(cur).strip() if cur else ""
    if not cur:
        cur = cur_raw

    lines: List[str] = []
    if max_turns > 0 and recent_llm_messages:
        tail = list(recent_llm_messages)[-max_turns:]
        for m in tail:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            one = re.sub(r"\s+", " ", content.replace("\n", " ")).strip()
            if not one:
                continue
            if role == "assistant":
                lines.append(f"assistant: {one}")
            else:
                lines.append(f"user: {one}")

    parts: List[str] = []
    if lines:
        parts.append("Recent conversation (oldest to newest):")
        parts.extend(lines)
    parts.append("Current message:")
    parts.append(f"user ({current_display_name}): {cur}")

    out = "\n".join(parts)
    if len(out) > max_chars:
        out = "…" + out[-(max_chars - 1) :]
    return out


# Common typos in chat queries (fixes token/embedding mismatch)
_QUERY_TYPO_FIXES: Tuple[Tuple[str, str], ...] = (
    ("veliefs", "beliefs"),
    ("belifs", "beliefs"),
    ("politicial", "political"),
)


def normalize_rag_query_text(query_text: str) -> str:
    """Light normalization before tokenizing and embedding (includes stripping wake words)."""
    if not query_text:
        return ""
    out = strip_rag_query_invocations(query_text)
    for wrong, right in _QUERY_TYPO_FIXES:
        out = re.sub(r"(?i)\b" + re.escape(wrong) + r"\b", right, out)
    out = " ".join(out.split())
    return out


def is_first_person_archive_query(query_text: str) -> bool:
    """Questions about the asker's own views/history (use their user_id, not name tokens)."""
    ql = (query_text or "").lower()
    if re.search(r"\bmy\b", ql):
        return True
    if re.search(r"\b(what|how)\s+(have|do)\s+i\s+", ql):
        return True
    # Catch "have i mentioned", "do i ever", "have i said" even with words in between
    if re.search(r"\bhave\s+i\b", ql):
        return True
    if re.search(r"\bdo\s+i\b", ql):
        return True
    if re.search(r"\bi\s+(have|mentioned|said|talked|posted|shared|discussed|watched|liked|played)\b", ql):
        return True
    if re.search(r"\btell\s+me\s+about\s+my\b", ql):
        return True
    if re.search(r"\babout\s+my\b", ql):
        return True
    # "i think/said/believe X" → first-person, but NOT "i want you to..." or "i don't want to..."
    # which are commands to the bot, not self-referencing archive queries.
    if re.search(r"\bi\s+(think|thought|believe|feel|mean)\b", ql):
        # Exclude negated commands: "i don't think you should..." is ambiguous, keep it
        return True
    if re.search(r"\bi\s+(say|said|like)\b", ql):
        # "i said X" or "i like X" → asking about own history
        return True
    if re.search(r"\bme\s+(think|thought|say|said|believe|feel|like|mean)\b", ql):
        return True
    # "i want" is usually a command ("i want you to tell me about X"), not a self-query.
    # Only treat as first-person if clearly self-referencing: "i want to know what i said"
    if re.search(r"\bi\s+want\b", ql) and re.search(r"\bwhat\s+i\b|\babout\s+me\b|\babout\s+my\b", ql):
        return True
    return False


# Dropped from topic SQL OR-clauses (too broad or redundant with first-person)
_TOPIC_NOISE_WORDS = frozenset(
    {
        "general",
        "specifically",
        "tell",
        "soupy",
        "please",
        "just",
        "really",
    }
)


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "again",
        "further",
        "once",
        "here",
        "there",
        "user",
        "users",
        "you",
        "your",
        "yours",
        "them",
        "they",
        "their",
        "said",
        "say",
        "says",
        "ever",
        "much",
        "many",
        "really",
        "well",
        "also",
        "soupy",
        "get",
        "got",
        "give",
        "given",
        "know",
        "think",
        "thought",
        "like",
        "thing",
        "things",
        "something",
        "anything",
        "nothing",
        "someone",
        "anyone",
        "people",
        "person",
        "tell",
        "told",
        "ask",
        "asked",
        "use",
        "used",
        "using",
        "hasnt",
        "havent",
        "dont",
        "didnt",
        "wasnt",
        "werent",
    }
)


def _topic_tokens_for_sql(tokens: Sequence[str]) -> List[str]:
    """Strip ultra-generic words from SQL topic OR (keeps embeddings full query)."""
    return [t for t in tokens if t not in _TOPIC_NOISE_WORDS]


def _extract_query_tokens(query_text: str) -> List[str]:
    """Lowercase tokens (>=3 chars) for keyword / author matching; drop boilerplate words."""
    raw = [t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", query_text or "")]
    out: List[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _token_is_name_match(token: str, name: str) -> bool:
    """Check if a token plausibly refers to a username/nickname.

    Returns True for:
      - Exact match:        "ranc1d" == "ranc1d"                     (100%)
      - Leading-word match: "glenn" matches first word of "glenn dafoe"
      - High-ratio match:   "glenn" in "glenneroo"   (5/9 = 56%)

    Returns False for:
      - Low-ratio substring: "cool" in "snackpacksarecool"  (4/17 = 24%)
      - "rights" in "squirrelrights"                        (6/14 = 43%)

    The threshold is that the token must cover at least 50% of the matched
    name (or the first word of a multi-word name).
    """
    name_lower = name.lower().strip()
    if not name_lower:
        return False

    # Exact match on full name
    if token == name_lower:
        return True

    # Match on the first word of a multi-word name (e.g. "glenn" → "glenn dafoe")
    first_word = name_lower.split()[0] if " " in name_lower else name_lower
    if token == first_word:
        return True

    # Check if token is a substantial portion of any individual word in the name.
    # "ranc1d" in "ranc1d"                 → 6/6  = 100% ✓
    # "glenn"  in "glenneroo"              → 5/9  =  56% ✓
    # "cool"   in "snackpacksarecool"      → 4/17 =  24% ✗
    # "rights" in "squirrelrights"         → 6/14 =  43% ✗
    # "art"    in "The Artist"             → 3/7  =  43% ✗ (checked against "artist")
    if token in name_lower:
        # Find the best matching word in the name and check ratio against it
        words = name_lower.replace("_", " ").replace("-", " ").split()
        best_ratio = 0.0
        for word in words:
            if token in word and len(word) > 0:
                ratio = len(token) / len(word)
                best_ratio = max(best_ratio, ratio)
        return best_ratio > 0.50

    return False


def _resolve_subject_user_id(
    conn: sqlite3.Connection, tokens: Sequence[str]
) -> Tuple[Optional[int], Optional[str]]:
    """
    If a token strongly matches one Discord user's username/nickname, return that user_id.

    Uses intelligent name matching instead of simple substring search:
    - Exact matches always win
    - Leading-word matches (first word of multi-word nickname) always win
    - Substring matches require the token to cover ≥50% of the name
    This prevents "cool" from matching "snackpacksarecool" or "rights" from
    matching "squirrelrights" without needing a stopword list.
    """
    try:
        min_hits = int(os.getenv("RAG_KEYWORD_MIN_AUTHOR_HITS", "5"))
    except ValueError:
        min_hits = 5
    try:
        dominance = float(os.getenv("RAG_KEYWORD_AUTHOR_DOMINANCE", "1.25"))
    except ValueError:
        dominance = 1.25
    cur = conn.cursor()

    # Pre-fetch the name directory: user_id → (username, nickname, msg_count)
    cur.execute(
        """
        SELECT user_id,
               lower(MAX(username)) AS uname,
               lower(MAX(coalesce(nickname, ''))) AS nname,
               COUNT(*) AS cnt
        FROM messages
        GROUP BY user_id
        HAVING cnt >= ?
        """,
        (min_hits,),
    )
    name_dir = cur.fetchall()
    if not name_dir:
        return None, None

    for tok in tokens:
        # Skip very short tokens (1-2 chars) — too ambiguous
        if len(tok) < 3:
            continue
        # Also try without trailing 's' so "glenns" matches "glenn", etc.
        candidates = [tok]
        if tok.endswith("s") and len(tok) > 3:
            candidates.append(tok[:-1])

        matched_uid = None
        matched_tok = None
        matched_cnt = 0

        for t in candidates:
            for row in name_dir:
                uname = row["uname"] or ""
                nname = row["nname"] or ""
                cnt = int(row["cnt"])
                # Check if token plausibly refers to this user's name
                if _token_is_name_match(t, uname) or _token_is_name_match(t, nname):
                    if cnt > matched_cnt:
                        matched_uid = int(row["user_id"])
                        matched_tok = t
                        matched_cnt = cnt
            if matched_uid is not None:
                break  # found a match with this candidate, don't try the de-pluralized form

        if matched_uid is not None:
            # Check dominance: the top match should be clearly dominant
            second_cnt = 0
            for t in candidates:
                for row in name_dir:
                    uid = int(row["user_id"])
                    if uid == matched_uid:
                        continue
                    uname = row["uname"] or ""
                    nname = row["nname"] or ""
                    if _token_is_name_match(t, uname) or _token_is_name_match(t, nname):
                        second_cnt = max(second_cnt, int(row["cnt"]))
            if second_cnt > 0 and matched_cnt < second_cnt * dominance:
                continue  # ambiguous — multiple users match
            logger.debug(
                "RAG keyword: resolved subject token %r -> user_id %s (%s msgs)",
                matched_tok, matched_uid, matched_cnt,
            )
            return matched_uid, matched_tok
    return None, None


def _fetch_rows_author_and_topics(
    conn: sqlite3.Connection, user_id: int, topic_tokens: Sequence[str], limit: int
) -> List[sqlite3.Row]:
    cur = conn.cursor()
    if not topic_tokens:
        cur.execute(
            """
            SELECT message_id, date, time, username, nickname, user_id, message_content,
                   channel_id, channel_name, image_description, url_summary
            FROM messages
            WHERE user_id = ?
            ORDER BY date DESC, time DESC, message_id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return cur.fetchall()
    or_clause = " OR ".join(
        ["lower(coalesce(message_content, '')) LIKE '%' || ? || '%'"] * len(topic_tokens)
    )
    sql = f"""
        SELECT message_id, date, time, username, nickname, user_id, message_content,
               channel_id, channel_name, image_description, url_summary
        FROM messages
        WHERE user_id = ?
          AND ({or_clause})
        ORDER BY date DESC, time DESC, message_id DESC
        LIMIT ?
    """
    params: List[Any] = [user_id, *topic_tokens, limit]
    cur.execute(sql, params)
    return cur.fetchall()


def _fetch_rows_content_cooccurrence(
    conn: sqlite3.Connection, token_a: str, token_b: str, limit: int
) -> List[sqlite3.Row]:
    """Two keywords must appear in the same message body (mentions / quotes)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT message_id, date, time, username, nickname, user_id, message_content,
               channel_id, channel_name, image_description, url_summary
        FROM messages
        WHERE lower(coalesce(message_content, '')) LIKE '%' || ? || '%'
          AND lower(coalesce(message_content, '')) LIKE '%' || ? || '%'
        ORDER BY date DESC, time DESC, message_id DESC
        LIMIT ?
        """,
        (token_a, token_b, limit),
    )
    return cur.fetchall()


def _fetch_global_topic_rows(conn: sqlite3.Connection, token: str, limit: int) -> List[sqlite3.Row]:
    """Last resort: any message containing token (short queries like 'guns')."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT message_id, date, time, username, nickname, user_id, message_content,
               channel_id, channel_name, image_description, url_summary
        FROM messages
        WHERE lower(coalesce(message_content, '')) LIKE '%' || ? || '%'
        ORDER BY date DESC, time DESC, message_id DESC
        LIMIT ?
        """,
        (token, limit),
    )
    return cur.fetchall()


def _keyword_supplement_block(
    conn: sqlite3.Connection,
    query_text: str,
    asking_user_id: int,
    first_person_hint: Optional[str] = None,
) -> str:
    """
    SQL-backed retrieval for names + topics. Embeddings often miss \"what did X say about Y\".
    When the asker talks about *my* views, we lock subject to asking_user_id.
    first_person_hint: usually the current message only (avoids \"my\" in older turns flipping mode).
    """
    try:
        lim = int(os.getenv("RAG_KEYWORD_SUPPLEMENT_LIMIT", "28"))
    except ValueError:
        lim = 28
    fallback_recent = min(lim, 12)

    # Use only the current message for subject/token resolution — the conversation history
    # prefix contains structural words ("talk", "mention", etc.) that pollute token extraction
    # and match wrong usernames before the actual subject name is reached.
    # Also strip the "user (displayname):" speaker prefix so the asker's name isn't extracted.
    _cur_msg_only = query_text or ""
    if "Current message:\n" in _cur_msg_only:
        _cur_msg_only = _cur_msg_only.split("Current message:\n")[-1].strip()
    _cur_msg_only = re.sub(r"^(?:user|assistant)\s*\([^)]*\)\s*:\s*", "", _cur_msg_only, flags=re.IGNORECASE).strip()
    qnorm = normalize_rag_query_text(_cur_msg_only or query_text or "")
    tokens = _extract_query_tokens(qnorm)
    fp_src = (first_person_hint if first_person_hint is not None else _cur_msg_only) or ""
    first_person = is_first_person_archive_query(fp_src)
    topic_sql = _topic_tokens_for_sql(tokens)
    if not tokens:
        return ""

    author_uid: Optional[int] = None
    author_tok: Optional[str] = None
    rows: List[sqlite3.Row] = []

    if first_person:
        author_uid = asking_user_id
        author_tok = None
        topic_tokens = topic_sql if topic_sql else tokens
        rows = _fetch_rows_author_and_topics(conn, author_uid, topic_tokens, lim)
        if not rows and topic_tokens:
            logger.debug("  keyword  : no topic match for %s (first-person), falling back to recent lines", topic_tokens)
            rows = _fetch_rows_author_and_topics(conn, author_uid, [], fallback_recent)
    else:
        author_uid, author_tok = _resolve_subject_user_id(conn, tokens)
        topic_tokens = [t for t in tokens if t != author_tok]
        topic_tokens = _topic_tokens_for_sql(topic_tokens) or topic_tokens

    if not first_person and author_uid is not None:
        rows = _fetch_rows_author_and_topics(conn, author_uid, topic_tokens, lim)
        if not rows and topic_tokens:
            logger.debug("  keyword  : no topic match for subject %r (user %s), falling back to recent lines", author_tok, author_uid)
            rows = _fetch_rows_author_and_topics(conn, author_uid, [], fallback_recent)
    elif not first_person and len(tokens) >= 2:
        partner = max(tokens[1:], key=len)
        rows = _fetch_rows_content_cooccurrence(conn, tokens[0], partner, lim)
        if not rows and tokens[0] != partner:
            rows = _fetch_rows_content_cooccurrence(conn, partner, tokens[0], lim)

    if not rows and not first_person and len(tokens) == 1 and asking_user_id:
        rows = _fetch_rows_author_and_topics(conn, asking_user_id, tokens, lim)
    # Short topic-only query (e.g. "guns"): if asker never used that word, still prefer their
    # recent lines over random server-wide matches before global LIKE fallback.
    if not rows and not first_person and len(tokens) == 1 and asking_user_id:
        rows = _fetch_rows_author_and_topics(conn, asking_user_id, [], fallback_recent)
    if not rows and not first_person and len(tokens) == 1 and tokens[0]:
        skip_global = os.getenv("RAG_KEYWORD_GLOBAL_FALLBACK", "1").strip() in ("0", "false", "no")
        if not skip_global:
            rows = _fetch_global_topic_rows(conn, tokens[0], min(lim, 25))

    if not rows:
        return ""

    mode = "first_person" if first_person else ("subject" if author_tok else "keyword")
    lines: List[str] = []
    for r in rows:
        base = _line_for_row(r)
        lines.append(
            f"{base} [message_id={r['message_id']}, #{r['channel_name']}, {r['date']} {r['time']}]"
        )
    header = (
        "--- Earlier messages (matched by topic or who posted; use for questions about a person or subject) ---\n"
    )
    block = header + "\n".join(lines)
    if _rag_log_verbose() and lines:
        _kw_first = re.sub(r" \(user_id=\d+\)", "", lines[0])
        _kw_preview = f"\n             {_preview_for_log(_kw_first, 120)}"
    else:
        _kw_preview = ""
    logger.info(
        "  keyword  : mode=%-12s | %d msg(s) → %d chars%s",
        mode, len(rows), len(block), _kw_preview,
    )
    # Full content dump — enable with RAG_LOG_FULL_CONTENT=1 for deep debugging
    if os.getenv("RAG_LOG_FULL_CONTENT", "0").strip() in ("1", "true", "yes"):
        for i, line in enumerate(lines, 1):
            clean = re.sub(r" \(user_id=\d+\)", "", line)
            logger.debug("  kw #%-3d  %s", i, clean)
    return block


async def fetch_rag_context_for_query(
    session: aiohttp.ClientSession,
    guild_id: int,
    author_user_id: int,
    query_text: str,
    first_person_hint: Optional[str] = None,
    max_chars_override: Optional[int] = None,
) -> Optional[str]:
    if not query_text:
        return None
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        logger.warning("RAG: no database for guild %s", guild_id)
        return None

    try:
        top_k = int(os.getenv("RAG_TOP_K", "10"))
    except ValueError:
        top_k = 10
    if max_chars_override is not None:
        max_chars = max(500, max_chars_override)
    else:
        try:
            max_chars = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "10000"))
        except ValueError:
            max_chars = 10000
    try:
        _kw_env = os.getenv("RAG_KEYWORD_MAX_CHARS", "").strip()
        keyword_max = int(_kw_env) if _kw_env else max(2800, max_chars // 2)
    except ValueError:
        keyword_max = max(2800, max_chars // 2)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    profile_pfx = ""
    supplement = ""
    hits: List[Tuple[float, str, Tuple[int, int, str]]] = []
    _self_hits: List[Tuple[float, str, str]] = []
    qv = None
    try:
        ensure_rag_schema(conn)
        from . import user_profiles as _user_profiles

        # Resolve subject early so profile and vector boost target the right person.
        # Use only the current message (not full conversation history) to avoid noise tokens.
        # Also strip the "user (displayname):" prefix so the asker's own name isn't resolved
        # as the subject (e.g. "user (ranc1d): what are glenns opinions" → "what are glenns opinions").
        _cur_for_subject = query_text or ""
        if "Current message:\n" in _cur_for_subject:
            _cur_for_subject = _cur_for_subject.split("Current message:\n")[-1].strip()
        _cur_for_subject = re.sub(r"^(?:user|assistant)\s*\([^)]*\)\s*:\s*", "", _cur_for_subject, flags=re.IGNORECASE).strip()
        _fp_early = is_first_person_archive_query(first_person_hint or _cur_for_subject)
        _subject_uid: Optional[int] = None
        _subject_tok: Optional[str] = None
        _subject_tokens: List[str] = []
        if not _fp_early:
            _subject_tokens = _extract_query_tokens(normalize_rag_query_text(_cur_for_subject))
            _subject_uid, _subject_tok = _resolve_subject_user_id(conn, _subject_tokens)

            # Pronoun fallback: if current message uses "him/her/them/he/she/they" but no
            # subject was resolved, try the previous user message from conversation history.
            if _subject_uid is None and re.search(
                r"\b(him|her|them|they|he|she)\b", _cur_for_subject, re.IGNORECASE
            ):
                _prev_user_msg = ""
                if "Current message:\n" in (query_text or ""):
                    _history_part = (query_text or "").split("Current message:\n")[0]
                    # Find the last "user (...): ..." line in conversation history
                    _prev_lines = re.findall(
                        r"^(?:user)\s*\([^)]*\)\s*:\s*(.+)$",
                        _history_part, re.MULTILINE | re.IGNORECASE,
                    )
                    if _prev_lines:
                        _prev_user_msg = _prev_lines[-1].strip()
                if _prev_user_msg:
                    _prev_tokens = _extract_query_tokens(normalize_rag_query_text(_prev_user_msg))
                    _subject_uid, _subject_tok = _resolve_subject_user_id(conn, _prev_tokens)
                    if _subject_uid:
                        logger.debug(
                            "RAG pronoun fallback: resolved '%s' from previous message → user_id %s",
                            _subject_tok, _subject_uid,
                        )

        # Log the resolution decisions at DEBUG so problems are visible
        _mode_label = "first-person" if _fp_early else ("subject=%s (%s)" % (_subject_tok, _subject_uid) if _subject_uid else "third-person (no subject)")
        logger.debug(
            "RAG resolve: stripped_msg=%r | tokens=%s | mode=%s",
            _cur_for_subject[:120], _subject_tokens or _extract_query_tokens(normalize_rag_query_text(_cur_for_subject)), _mode_label,
        )

        # Build a name-mapping note when the subject's display names don't obviously
        # match the token the user typed (e.g. "agro" → username ".agro.", nickname "oɹƃɐ").
        _name_map_note = ""
        if _subject_uid and _subject_tok:
            cur = conn.cursor()
            cur.execute(
                """SELECT DISTINCT username, nickname FROM messages
                   WHERE user_id = ? AND (username IS NOT NULL OR nickname IS NOT NULL)
                   ORDER BY rowid DESC LIMIT 1""",
                (_subject_uid,),
            )
            _row = cur.fetchone()
            if _row:
                _uname = (_row[0] or "").strip()
                _nick = (_row[1] or "").strip()
                # Collect display names that differ from the search token
                _aliases = []
                if _uname and _uname.lower().replace(".", "").replace("_", "") != _subject_tok:
                    _aliases.append(f'username "{_uname}"')
                if _nick and _nick.lower() != _subject_tok and _nick != _uname:
                    _aliases.append(f'current nickname "{_nick}"')
                if _aliases:
                    _name_map_note = (
                        f'Note: "{_subject_tok}" refers to the member with {", ".join(_aliases)}.'
                        f' All messages below from this member use their display name.'
                        f' Refer to this member as "{_subject_tok}" in your response.\n\n'
                    )
                else:
                    # Names match but still clarify who the data is about
                    _display = _nick or _uname or _subject_tok
                    _name_map_note = (
                        f'Note: The excerpts below are about the member "{_display}".'
                        f' Refer to them as "{_subject_tok}" in your response.\n\n'
                    )
                logger.debug("RAG name map: %s", _name_map_note.strip())

        # Load profile for the subject (if third-person query) AND the asker.
        # subject_user_id tells the profile function who else to include beyond the asker.
        profile_pfx = _user_profiles.format_profile_prefix_for_rag(
            conn, author_user_id, query_text, first_person_hint,
            subject_user_id=_subject_uid,
        )
        # Prepend the name-mapping note so the LLM knows the connection
        if _name_map_note:
            profile_pfx = _name_map_note + profile_pfx
        qnorm = normalize_rag_query_text(query_text)

        # Human-readable RAG block header.
        # query_text is the full build_rag_retrieval_query output (includes conversation history);
        # extract just the current message line for a readable header.
        _display_query = query_text or ""
        if "Current message:\n" in _display_query:
            _display_query = _display_query.split("Current message:\n")[-1].strip()
        _prof_names = re.findall(r"^— (.+?) \(id \d+\):", profile_pfx or "", re.MULTILINE)
        _prof_summary = (
            f"{', '.join(_prof_names)} → {len(profile_pfx)} chars"
            if _prof_names
            else "none"
        )
        logger.info(
            "[RAG] user=%s | query: %s\n  profile  : %s",
            author_user_id,
            _preview_for_log(_display_query, 140),
            _prof_summary,
        )
        if os.getenv("RAG_LOG_FULL_CONTENT", "0").strip() in ("1", "true", "yes") and profile_pfx:
            logger.debug("  profile full:\n%s", profile_pfx)

        supplement = _keyword_supplement_block(
            conn, query_text, author_user_id, first_person_hint
        )

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM rag_chunks")
        chunk_count = int(cur.fetchone()[0])
        if chunk_count == 0:
            logger.warning("RAG: vector index empty for guild %s; run Rebuild RAG index for semantic excerpts.", guild_id)
            if supplement or profile_pfx:
                out, prof_out, kw_out, vec_n, truncated, raw_len = _assemble_rag_bundle(
                    profile_pfx, supplement, [], max_chars, keyword_max
                )
                _log_rag_bundle(prof_out, kw_out, vec_n, len(out), max_chars, truncated, "no vector index")
                return out
            return None

        fp = _fp_early
        toks = _extract_query_tokens(normalize_rag_query_text(_cur_for_subject))
        short = len(toks) <= 2
        # Always boost when we have a resolved subject or first-person query,
        # not just when the query is short. Otherwise subject's vectors get drowned out.
        base_ubm = 3.0 if (fp or short or _subject_uid is not None) else 1.0
        try:
            ubm = float(os.getenv("RAG_VECTOR_USER_BOOST_MULTIPLIER", str(base_ubm)))
        except ValueError:
            ubm = base_ubm

        # Boost the subject's chunks in vector search (or the asker's for first-person queries).
        _boost_uid = author_user_id if (fp or _subject_uid is None) else _subject_uid

        vecs = await embed_texts_lm_studio(session, [qnorm])
        if not vecs:
            if supplement or profile_pfx:
                out, prof_out, kw_out, vec_n, truncated, raw_len = _assemble_rag_bundle(
                    profile_pfx, supplement, [], max_chars, keyword_max
                )
                _log_rag_bundle(prof_out, kw_out, vec_n, len(out), max_chars, truncated, "embeddings failed")
                return out
            return None
        qv = vecs[0]
        hits = search_rag_chunks(
            conn,
            qv,
            top_k=top_k,
            prefer_user_id=_boost_uid,
            user_boost_multiplier=ubm,
        )
        _mode_tag = "first-person" if fp else ("subject" if _subject_uid else "third-person")
        logger.info(
            "  vectors  : %d hits | boost=%.1fx | mode=%s | tokens=%s",
            len(hits), ubm, _mode_tag, toks[:6],
        )
        if _rag_log_verbose() and hits:
            _detail_limit = int(os.getenv("RAG_LOG_HIT_DETAIL_COUNT", "4"))
            _full_content = os.getenv("RAG_LOG_FULL_CONTENT", "0").strip() in ("1", "true", "yes")
            for i, (score, text, (mid_lo, mid_hi, ch)) in enumerate(hits, 1):
                if not _full_content and i > _detail_limit:
                    logger.info("             … %d more hit(s)", len(hits) - _detail_limit)
                    break
                # Strip the [#channel channel_id=X] header line from the chunk for readability
                _chunk_body = re.sub(r"^\[#[^\]]*\]\s*", "", (text or "").strip())
                if _full_content:
                    logger.debug(
                        "  vec #%-2d  sim=%.3f  #%s  ids=%s-%s\n%s",
                        i, score, ch, mid_lo, mid_hi,
                        re.sub(r" \(user_id=\d+\)", "", _chunk_body),
                    )
                else:
                    logger.info(
                        "             #%-2d  sim=%.3f  #%-14s  ids=%-12s  %s",
                        i, score, (ch or "")[:14], f"{mid_lo}-{mid_hi}",
                        _preview_for_log(_chunk_body, 95),
                    )

        # --- Self-knowledge retrieval (reuse query embedding) ---
        _self_hits: List[Tuple[float, str, str]] = []
        if qv is not None:
            try:
                from .self_context import search_self_chunks, is_self_md_enabled
                if is_self_md_enabled():
                    _self_top_k = int(os.getenv("RAG_SELF_KNOWLEDGE_TOP_K", "5"))
                    _self_min_sim = float(os.getenv("RAG_SELF_KNOWLEDGE_MIN_SIM", "0.3"))
                    _self_hits_raw = search_self_chunks(conn, qv, top_k=_self_top_k)
                    _self_hits = [(s, t, sec) for s, t, sec in _self_hits_raw if s >= _self_min_sim]
                    if _self_hits:
                        logger.info(
                            "  self-knowledge: %d chunk(s) (top sim=%.3f)",
                            len(_self_hits), _self_hits[0][0],
                        )
                        for i, (sc, txt, sec) in enumerate(_self_hits, 1):
                            logger.debug("  self #%-2d sim=%.3f  [%s]  %s", i, sc, sec,
                                         _preview_for_log(txt, 80))
            except Exception as exc:
                logger.debug("self-knowledge retrieval skipped: %s", exc)
    finally:
        conn.close()

    if not supplement and not hits and not profile_pfx and not _self_hits:
        return None
    out, prof_out, kw_out, vec_n, truncated, raw_len = _assemble_rag_bundle(
        profile_pfx, supplement, hits, max_chars, keyword_max,
        self_hits=_self_hits,
    )
    _log_rag_bundle(prof_out, kw_out, vec_n, len(out), max_chars, truncated)
    return out
