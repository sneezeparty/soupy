"""
Musings cog for Soupy Bot.

Soupy occasionally "thinks out loud" in a configured channel — reflecting on
things from the server archive, reacting to news, or musing about conversations.

Modes (weights in ``_MODE_WEIGHTS``):

* ``archive_reflect`` — picks one message from a time-bucketed sample of the
  guild archive and writes a reaction to it.
* ``news_react`` — fetches a recent headline via DuckDuckGo and reacts.
* ``random_thought`` — opens with a recent self-knowledge fragment.
* ``synthesis`` — finds a pattern across multiple recent messages.

Cross-module:

* Reads message archive via ``soupy_database.database.get_db_path`` (SQLite).
* When ``SELF_MD_ENABLED`` is set, calls
  ``soupy_database.self_context.add_notable_interaction`` to feed *synthesis*
  musings (only) into the reflection accumulator. See ``_SELF_FEEDBACK_MODES``.
* When ``RAG_EMBEDDING_MODEL`` is set, uses
  ``soupy_database.rag.embed_texts_lm_studio`` for similarity dedupe.

Gotchas:

* The archive file (``data/musings_archive.jsonl``) is the source of truth for
  "what have we already said?" — both keyword and embedding dedupe read from it.
* ``self._post_lock`` serializes the entire post-and-persist pipeline so
  warmup, scheduled tick, and ``/soupymuse`` can't race on the JSONL.
* Legacy entries (single ``topic`` field) are migrated to the
  ``topic_subject`` / ``topic_mentions`` split on first load via one batched
  LLM call — see the warmup task.
* Topic extraction runs *after* ``channel.send()`` so a slow LLM doesn't delay
  the user-visible post; the topic only needs to exist by the next tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple

import aiohttp
import discord
import pytz
from ddgs import DDGS
from discord import app_commands
from discord.ext import commands, tasks

from soupy.scheduling import (
    load_json_state,
    parse_aware,
    random_time_in_window,
    save_json_state,
    window_bounds,
)
from soupy.settings import openai_client, settings
from soupy_database.database import get_db_path
from soupy_database.self_context import add_notable_interaction, is_self_md_enabled

logger = logging.getLogger(__name__)

MUSINGS_ARCHIVE_PATH = os.path.join("data", "musings_archive.jsonl")
MAX_ARCHIVE_ENTRIES = 200

# Daily-schedule state — one auto-musing per local day, at a random time in
# [MUSING_HOUR_MIN, MUSING_HOUR_MAX) local (see settings). State persists across
# restarts so a bounce during the day doesn't double-post or reroll the time.
MUSING_DAILY_STATE_PATH = os.path.join("data", "musings_daily_state.json")

# Fallbacks used when the configured hour pair is inverted. Matches the
# settings defaults; duplicated here so the normalization path has something
# known-good to fall back to without re-reading env.
DEFAULT_HOUR_MIN = 6
DEFAULT_HOUR_MAX = 20

# How far past the end of the window we'll still honour a musing that came due
# inside it. Without this, a musing scheduled at 19:59 is silently dropped by
# any tick that lands at 20:00:0x — the window check would fire first and mark
# the day skipped. Anything later than the grace is a bot that booted after the
# window closed, which genuinely should skip.
LATE_FIRE_GRACE = timedelta(minutes=10)

# How many recent musings to treat as "already covered". Used both to filter
# archive candidates source-side and to warn the LLM off the same subject.
RECENT_TOPIC_WINDOW = 15

# Embedding similarity threshold for "this is too close to something we just
# said." Higher = stricter (only catches near-duplicates). 0.88 is conservative
# enough that genuinely fresh thoughts pass even if they share vocabulary.
EMBED_SIMILARITY_THRESHOLD = 0.88

# Modes whose musings get fed back into the self-reflection accumulator. Other
# modes are quick reactions — feeding them in turned the self-doc into an echo
# chamber that then re-seeded future musings via random_thought.
_SELF_FEEDBACK_MODES: Set[str] = {"synthesis"}


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _load_all_musings() -> List[Dict[str, str]]:
    """Load every musing in the archive (up to MAX_ARCHIVE_ENTRIES), in order."""
    if not os.path.exists(MUSINGS_ARCHIVE_PATH):
        return []
    try:
        lines = open(MUSINGS_ARCHIVE_PATH, encoding="utf-8").read().splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
        return entries
    except Exception:
        return []


def _persist_all_musings(entries: List[Dict[str, str]]) -> None:
    """Rewrite the archive file atomically.

    Writes to a sibling tmp file then ``os.replace``s it into place so a crash
    or a concurrent reader never sees a half-written file.
    """
    if not entries:
        return
    try:
        os.makedirs(os.path.dirname(MUSINGS_ARCHIVE_PATH), exist_ok=True)
        tmp = MUSINGS_ARCHIVE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
        os.replace(tmp, MUSINGS_ARCHIVE_PATH)
    except Exception as exc:
        logger.debug("💭 Failed to rewrite musings archive: %s", exc)


def _save_musing(
    thought: str,
    mode: str,
    guild_id: int,
    topic_subject: str = "",
    topic_mentions: str = "",
) -> Dict[str, str]:
    """Append a musing to the archive and return the persisted entry."""
    os.makedirs(os.path.dirname(MUSINGS_ARCHIVE_PATH), exist_ok=True)
    entry = {
        "text": thought,
        "mode": mode,
        "guild_id": guild_id,
        "ts": datetime.now(pytz.UTC).isoformat(),
        "topic_subject": topic_subject,
        "topic_mentions": topic_mentions,
    }
    try:
        with open(MUSINGS_ARCHIVE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Trim if too long
        lines = open(MUSINGS_ARCHIVE_PATH, encoding="utf-8").read().splitlines()
        if len(lines) > MAX_ARCHIVE_ENTRIES:
            trimmed = lines[-MAX_ARCHIVE_ENTRIES:]
            tmp = MUSINGS_ARCHIVE_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(trimmed) + "\n")
            os.replace(tmp, MUSINGS_ARCHIVE_PATH)
    except Exception as exc:
        logger.debug("💭 Failed to save musing to archive: %s", exc)
    return entry


# ---------------------------------------------------------------------------
# Topic / keyword utilities
# ---------------------------------------------------------------------------


_STOPWORD_TOPICS: Set[str] = {
    "thing",
    "things",
    "stuff",
    "people",
    "someone",
    "anyone",
    "everything",
    "nothing",
    "life",
    "world",
    "general",
    "various",
    "miscellaneous",
}


def _keywords_set(topics: List[str]) -> Set[str]:
    """Flatten a list of comma-separated topic strings into a keyword set."""
    out: Set[str] = set()
    for t in topics:
        if not t:
            continue
        for kw in t.split(","):
            kw = kw.strip().lower()
            if kw and kw not in _STOPWORD_TOPICS:
                out.add(kw)
    return out


def _candidate_overlaps(content: str, banned_keywords: Set[str]) -> bool:
    """Return True if `content` contains any banned keyword as a whole word/phrase."""
    if not banned_keywords or not content:
        return False
    content_lower = content.lower()
    for kw in banned_keywords:
        if not kw:
            continue
        # Whole-word boundary so "ai" doesn't match "rain", but multi-word
        # phrases like "leaky gut" still match.
        pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
        if re.search(pattern, content_lower):
            return True
    return False


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for two vectors. Returns 0.0 on shape mismatch."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


client = openai_client()


async def _llm_call(system: str, user: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
    def _sync():
        return client.chat.completions.create(
            model=settings.local_chat or "local-model",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    response = await asyncio.to_thread(_sync)
    return response.choices[0].message.content.strip()


def _parse_subject_mentions(raw: str) -> Tuple[str, str]:
    """Parse a 'SUBJECT: ...\\nMENTIONS: ...' block into (subject, mentions)."""
    subject = ""
    mentions = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, tail = line.partition(":")
        if not tail:
            continue
        key = head.strip().lower()
        val = tail.strip().strip('"').strip("'").lower()
        if val in ("none", "(none)", "n/a", "-", ""):
            val = ""
        if key.startswith("subject"):
            subject = val[:80]
        elif key.startswith("mention"):
            mentions = val[:80]
    return subject, mentions


_TOPIC_EXTRACT_SYSTEM = (
    "You analyze a short musing and output two fields:\n"
    "  SUBJECT: 1-3 short noun phrases (lowercase, comma-separated) naming what the "
    "musing is ABOUT — the topic, product, event, idea, or behavior. "
    "Do NOT put people's names here.\n"
    "  MENTIONS: 0-3 names of people, handles, or speakers mentioned in the musing. "
    "Lowercase, comma-separated, or the word 'none' if no people are named.\n\n"
    "Output exactly two lines in this format:\n"
    "SUBJECT: <subjects>\n"
    "MENTIONS: <names or none>\n\n"
    "No preamble, no explanation, no extra lines."
)


async def _extract_topic(thought: str) -> Tuple[str, str]:
    """Extract (subject, mentions) from a single musing."""
    if not thought or len(thought) < 10:
        return "", ""
    try:
        raw = await _llm_call(_TOPIC_EXTRACT_SYSTEM, thought, temperature=0.2, max_tokens=80)
    except Exception as exc:
        logger.debug("💭 Topic extraction failed: %s", exc)
        return "", ""
    return _parse_subject_mentions(raw)


_BATCH_TOPIC_SYSTEM = (
    "You analyze numbered musings and output topic info for each. For each musing "
    "output one block in this exact format (note the [N] prefix on the SUBJECT line):\n"
    "[N] SUBJECT: <1-3 short noun phrases — what the musing is ABOUT, no people names>\n"
    "MENTIONS: <0-3 names of people mentioned, or 'none'>\n\n"
    "All lowercase. No preamble, no explanation, no extra commentary."
)


async def _batch_extract_topics(texts: List[str]) -> List[Tuple[str, str]]:
    """Extract (subject, mentions) for many musings in one LLM call."""
    if not texts:
        return []
    numbered = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
    try:
        raw = await _llm_call(
            _BATCH_TOPIC_SYSTEM, numbered, temperature=0.2, max_tokens=600
        )
    except Exception as exc:
        logger.debug("💭 Batch topic extraction failed: %s", exc)
        return [("", "") for _ in texts]

    result: List[Tuple[str, str]] = [("", "") for _ in texts]
    # Parse: lines starting with "[N] SUBJECT:" begin a block; following
    # "MENTIONS:" lines belong to the most recent block.
    current_idx: Optional[int] = None
    current_subject = ""
    current_mentions = ""

    def flush():
        nonlocal current_idx, current_subject, current_mentions
        if current_idx is not None and 0 <= current_idx < len(texts):
            result[current_idx] = (current_subject, current_mentions)
        current_idx = None
        current_subject = ""
        current_mentions = ""

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\[(\d+)\]\s*SUBJECT\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if m:
            flush()
            current_idx = int(m.group(1)) - 1
            current_subject = m.group(2).strip().strip('"').strip("'").lower()[:80]
            continue
        m = re.match(r"MENTIONS\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if m and current_idx is not None:
            val = m.group(1).strip().strip('"').strip("'").lower()
            if val in ("none", "(none)", "n/a", "-"):
                val = ""
            current_mentions = val[:80]
    flush()
    return result


# ---------------------------------------------------------------------------
# Embedding helpers (best-effort; degrade silently if LM Studio embed unavailable)
# ---------------------------------------------------------------------------


def _embeddings_configured() -> bool:
    return bool(os.getenv("RAG_EMBEDDING_MODEL", "").strip())


# ---------------------------------------------------------------------------
# Persona / mode prompts and weights
# ---------------------------------------------------------------------------


MUSING_SYSTEM = (
    "you are soupy dafoe, thinking out loud in a discord channel. you are not responding to anyone — "
    "you are just sharing a single thought, observation, or reaction. "
    "write in lower case, no quotation marks.\n\n"
    "keep it short. 80 words max, 1-2 sentences is ideal, 3 sentences max. "
    "think of it like muttering one thing under your breath, not writing a journal entry.\n\n"
    "pick one thing to think about. not two, not three. one specific thought.\n\n"
    "anchor the thought. a reader should be able to tell what you are reacting to, even if "
    "you only nod at it sideways. work in a concrete handle — a name, a quoted phrase, a "
    "specific number, the actual topic by name. vague gestures like 'that thing' on their own "
    "are not enough; pair them with something a stranger could latch onto.\n\n"
    "do not address anyone directly. do not ask questions directed at the chat. "
    "do not include any urls, timestamps, channel names, or metadata in your response. "
    "do not include word counts, token counts, parenthetical notes, or any meta commentary "
    "about your own output. just write the thought and stop. nothing after the final period."
)


# Mode weights — synthesis is expensive and over-eager, so it's now closer to
# parity with the other modes. archive_reflect remains the workhorse.
_MODE_WEIGHTS: List[Tuple[str, float]] = [
    ("archive_reflect", 0.35),
    ("news_react", 0.20),
    ("random_thought", 0.20),
    ("synthesis", 0.25),
]


def _pick_mode() -> str:
    names = [m for m, _ in _MODE_WEIGHTS]
    weights = [w for _, w in _MODE_WEIGHTS]
    return random.choices(names, weights=weights, k=1)[0]


# Time-bucket sampling for archive_reflect — biases each call into a different
# era so the bot isn't always pulling from the most recent week.
# Each tuple: (days_from, days_to, weight). days_from < days_to <= 0 means
# "between days_from days ago and days_to days ago (today = 0)".
_ARCHIVE_BUCKETS: List[Tuple[int, int, float]] = [
    (-2, 0, 0.20),      # last ~2 days
    (-9, -2, 0.30),     # 2-9 days ago
    (-30, -9, 0.25),    # 9-30 days ago
    (-180, -30, 0.25),  # 1-6 months ago
]


# Time-bucket sampling for synthesis. Wider windows than archive_reflect, since
# synthesis is meant to surface patterns that need many data points to see.
_SYNTHESIS_BUCKETS: List[Tuple[int, int, float]] = [
    (-14, 0, 0.40),      # last two weeks
    (-45, -14, 0.30),    # 2-6 weeks ago
    (-180, -45, 0.30),   # 1.5-6 months ago
]


def _pick_bucket(buckets: List[Tuple[int, int, float]]) -> Tuple[int, int]:
    weights = [w for _, _, w in buckets]
    chosen = random.choices(buckets, weights=weights, k=1)[0]
    return chosen[0], chosen[1]


def _sql_date_clause(days_from: int, days_to: int) -> Tuple[str, str]:
    """Return SQL date bounds for a time bucket as parameter-safe expressions."""
    from_expr = f"date('now', '{days_from} days')"
    if days_to == 0:
        to_expr = "date('now', '+1 day')"
    else:
        to_expr = f"date('now', '{days_to} days')"
    return from_expr, to_expr


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class MusingsCog(commands.Cog):
    """Soupy thinks out loud in a configured channel."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # `or "UTC"` covers the "TIMEZONE unset" case; settings.timezone defaults
        # to America/Los_Angeles per .env-stable.example, but a brand-new install
        # could have it blank.
        self.timezone = pytz.timezone(settings.timezone or "UTC")

        # Serialize the post pipeline so /soupymuse and the auto-loop can't
        # interleave their candidate-pick + LLM + save steps.
        self._post_lock = asyncio.Lock()

        # In-memory cache of recent musing embeddings, keyed by ts. Populated
        # lazily by the warmup task; falls back to keyword-only dedupe if
        # embeddings are unavailable.
        self._musing_embeddings: Dict[str, List[float]] = {}
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Daily-schedule state, mirrored in memory. See _daily_state().
        self._daily_state_cache: Optional[Dict[str, str]] = None
        self._warned_bad_window = False

        self._loop.start()
        # Fire-and-forget warmup: backfill missing topics and seed the embedding
        # cache so the first musing post-deploy doesn't pay the latency.
        self._warmup_task = asyncio.create_task(self._warmup())

    def cog_unload(self) -> None:
        self._loop.cancel()
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
        if self._http_session is not None:
            asyncio.create_task(self._http_session.close())

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        return settings.musing_enabled

    def _channel_id(self) -> Optional[int]:
        return settings.musing_channel_id

    def _hour_window(self) -> Tuple[int, int]:
        """Return the validated ``(start, end)`` local hours for the daily window.

        ``end`` is exclusive and may be 24. An inverted pair falls back to the
        defaults rather than propagating: this runs inside a 60-second loop, so
        raising here would silently kill musings for every subsequent tick.
        """
        lo, hi = settings.musing_hour_min, settings.musing_hour_max
        if hi <= lo:
            # Warn once, not once per tick — this is read several times a minute.
            if not self._warned_bad_window:
                self._warned_bad_window = True
                logger.warning(
                    "💭 MUSING_HOUR_MAX (%d) must be greater than MUSING_HOUR_MIN (%d); "
                    "using the %02d:00-%02d:00 default window instead",
                    hi, lo, DEFAULT_HOUR_MIN, DEFAULT_HOUR_MAX,
                )
            return DEFAULT_HOUR_MIN, DEFAULT_HOUR_MAX
        return lo, hi

    def _now_local(self) -> datetime:
        """Current time in the configured timezone. A seam for tests."""
        return datetime.now(self.timezone)

    def _window_bounds(self, local_date) -> Tuple[datetime, datetime]:
        lo, hi = self._hour_window()
        return window_bounds(self.timezone, local_date, lo, hi)

    def _pick_random_time_in_window(self, local_date) -> datetime:
        """Random tz-aware datetime inside today's musing window."""
        lo, hi = self._hour_window()
        return random_time_in_window(self.timezone, local_date, lo, hi)

    # ------------------------------------------------------------------
    # Daily-schedule state
    # ------------------------------------------------------------------

    def _daily_state(self) -> Dict[str, str]:
        """Return the daily-schedule state, hitting disk only on the first call.

        The in-memory copy is authoritative once loaded. That matters for more
        than the 1439 redundant reads it saves per day: if a write fails
        (read-only ``data/``, full disk), the cache still records that today is
        handled, so a persistent write failure degrades to "state is lost on
        restart" instead of "post a musing every 60 seconds, forever."
        """
        if self._daily_state_cache is None:
            self._daily_state_cache = load_json_state(MUSING_DAILY_STATE_PATH)
        return self._daily_state_cache

    def _save_daily_state(self, state: Dict[str, str]) -> None:
        """Persist the daily state. Failures are logged, never raised."""
        self._daily_state_cache = state
        save_json_state(MUSING_DAILY_STATE_PATH, state)

    def _scheduled_time_for(self, state: Dict[str, str], local_date) -> datetime:
        """Today's scheduled musing time, rolling and persisting one if needed."""
        today_str = local_date.isoformat()
        if state.get("scheduled_date") == today_str:
            existing = parse_aware(state.get("scheduled_at"), self.timezone)
            if existing is not None:
                return existing

        scheduled = self._pick_random_time_in_window(local_date)
        state["scheduled_date"] = today_str
        state["scheduled_at"] = scheduled.isoformat()
        self._save_daily_state(state)
        logger.info("💭 Scheduled today's musing for %s", scheduled.strftime("%Y-%m-%d %H:%M %Z"))
        self._update_dashboard()
        return scheduled

    def _mark_handled(self, state: Dict[str, str], today_str: str, status: str) -> None:
        """Record that today is spoken for, whatever the outcome."""
        state["last_handled_date"] = today_str
        state["last_handled_status"] = status
        state["last_handled_at"] = datetime.now(pytz.UTC).isoformat()
        self._save_daily_state(state)
        self._update_dashboard()

    def _update_dashboard(self) -> None:
        """Publish loop state into ``bot._timer_state`` for the web dashboard.

        Derived wholly from the daily state dict so it stays correct across a
        restart, rather than only reflecting transitions this process saw. The
        panel has rendered a Musings card since the loop existed; it was blank
        because nothing ever wrote this slot. Best-effort by design — a
        dashboard write must never break the loop.
        """
        try:
            timer_state = getattr(self.bot, "_timer_state", None)
            if not isinstance(timer_state, dict) or "musings" not in timer_state:
                return

            state = self._daily_state()
            slot = timer_state["musings"]
            lo, hi = self._hour_window()
            slot["enabled"] = self._is_enabled()
            slot["interval"] = f"1x/day, {lo:02d}:00-{hi:02d}:00 local"

            today_str = self._now_local().date().isoformat()
            scheduled = (
                parse_aware(state.get("scheduled_at"), self.timezone)
                if state.get("scheduled_date") == today_str
                else None
            )
            done_today = state.get("last_handled_date") == today_str
            slot["next_run"] = (
                None if (done_today or scheduled is None) else scheduled.astimezone(pytz.UTC).isoformat()
            )

            last = parse_aware(state.get("last_handled_at"), self.timezone)
            slot["last_run"] = last.astimezone(pytz.UTC).isoformat() if last else None
            slot["last_status"] = state.get("last_handled_status")
        except Exception as exc:
            logger.debug("💭 Failed to publish musings timer state: %s", exc)

    # ------------------------------------------------------------------
    # Warmup (lazy backfill of topics + embeddings)
    # ------------------------------------------------------------------

    async def _warmup(self) -> None:
        """Backfill topic tags + embeddings for the recent window.

        Runs once after cog start so the first musing post-deploy doesn't pay
        for migration. Acquires ``_post_lock`` because ``_get_recent_topics``
        can rewrite the archive file, and a concurrent ``/soupymuse`` post
        would otherwise race against the rewrite.
        """
        try:
            await self.bot.wait_until_ready()
            # Give the rest of startup a moment.
            await asyncio.sleep(5)
            async with self._post_lock:
                entries, _ = await self._get_recent_topics()
                await self._refresh_embedding_cache(entries)
            logger.info(
                "💭 Warmup complete: %d musings tagged, %d embeddings cached",
                len(entries),
                len(self._musing_embeddings),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("💭 Warmup failed: %s", exc)

    async def _get_http_session(self) -> Optional[aiohttp.ClientSession]:
        if self._http_session is None or self._http_session.closed:
            try:
                self._http_session = aiohttp.ClientSession()
            except Exception as exc:
                logger.debug("💭 Could not create aiohttp session: %s", exc)
                return None
        return self._http_session

    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed a single text via LM Studio. Returns None on failure / not configured."""
        if not _embeddings_configured() or not text:
            return None
        session = await self._get_http_session()
        if session is None:
            return None
        try:
            from soupy_database.rag import embed_texts_lm_studio

            vecs = await embed_texts_lm_studio(session, [text])
            return vecs[0] if vecs else None
        except Exception as exc:
            logger.debug("💭 Embed failed: %s", exc)
            return None

    async def _embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not _embeddings_configured() or not texts:
            return [None] * len(texts)
        session = await self._get_http_session()
        if session is None:
            return [None] * len(texts)
        try:
            from soupy_database.rag import embed_texts_lm_studio

            vecs = await embed_texts_lm_studio(session, texts)
            return list(vecs)
        except Exception as exc:
            logger.debug("💭 Batch embed failed: %s", exc)
            return [None] * len(texts)

    async def _refresh_embedding_cache(self, entries: List[Dict[str, str]]) -> None:
        """Embed any recent musings missing from the cache."""
        if not _embeddings_configured():
            return
        needed: List[Tuple[str, str]] = []  # (ts, text)
        for e in entries:
            ts = e.get("ts") or ""
            text = e.get("text") or ""
            if ts and text and ts not in self._musing_embeddings:
                needed.append((ts, text))
        if not needed:
            return
        vecs = await self._embed_texts([t for _, t in needed])
        for (ts, _text), vec in zip(needed, vecs, strict=False):
            if vec is not None:
                self._musing_embeddings[ts] = vec
        # Trim the cache to the recent window so it doesn't grow unbounded.
        keep_ts = {e.get("ts") for e in entries if e.get("ts")}
        self._musing_embeddings = {
            k: v for k, v in self._musing_embeddings.items() if k in keep_ts
        }

    async def _is_too_similar(self, text: str, threshold: float = EMBED_SIMILARITY_THRESHOLD) -> bool:
        """Return True if `text` embeds too close to any cached recent musing."""
        if not self._musing_embeddings:
            return False
        vec = await self._embed_text(text)
        if vec is None:
            return False
        for cached in self._musing_embeddings.values():
            if _cosine(vec, cached) >= threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Recent-topic bookkeeping
    # ------------------------------------------------------------------

    async def _get_recent_topics(
        self, limit: int = RECENT_TOPIC_WINDOW
    ) -> Tuple[List[Dict[str, str]], Set[str]]:
        """Load recent musings, migrating/backfilling topic tags as needed.

        Returns ``(entries, banned_subject_keywords)`` where ``banned`` is built
        ONLY from the ``topic_subject`` field — so names of speakers in
        ``topic_mentions`` do not silently filter that person's other messages
        out of future candidate pools.
        """
        all_entries = _load_all_musings()
        if not all_entries:
            return [], set()
        recent = all_entries[-limit:]
        # Migration: entries missing subject (either legacy single "topic" or
        # never tagged at all) need re-extraction.
        missing_local_idx = [
            i for i, e in enumerate(recent)
            if not e.get("topic_subject") and not e.get("topic_mentions")
        ]
        if missing_local_idx:
            texts = [recent[i].get("text", "") for i in missing_local_idx]
            new_topics = await _batch_extract_topics(texts)
            anything_set = False
            for local_idx, (subj, ment) in zip(missing_local_idx, new_topics, strict=False):
                if subj or ment:
                    recent[local_idx]["topic_subject"] = subj
                    recent[local_idx]["topic_mentions"] = ment
                    # Drop legacy field if present
                    recent[local_idx].pop("topic", None)
                    full_idx = len(all_entries) - len(recent) + local_idx
                    all_entries[full_idx]["topic_subject"] = subj
                    all_entries[full_idx]["topic_mentions"] = ment
                    all_entries[full_idx].pop("topic", None)
                    anything_set = True
            if anything_set:
                await asyncio.to_thread(_persist_all_musings, all_entries)
                logger.info(
                    "💭 Backfilled topic_subject/mentions for %d musings",
                    sum(1 for s, m in new_topics if s or m),
                )
        banned = _keywords_set([e.get("topic_subject", "") for e in recent])
        return recent, banned

    def _build_recent_context(self, entries: List[Dict[str, str]]) -> str:
        """Render the "already mused on" list for inclusion in an LLM prompt."""
        if not entries:
            return ""
        lines = []
        for r in entries:
            text = (r.get("text", "") or "").strip()
            subject = (r.get("topic_subject", "") or "").strip()
            mentions = (r.get("topic_mentions", "") or "").strip()
            if not text:
                continue
            if subject and mentions:
                handle = f"[{subject} — re: {mentions}] "
            elif subject:
                handle = f"[{subject}] "
            else:
                handle = ""
            lines.append(f"- {handle}{text[:140]}")
        if not lines:
            return ""
        block = "\n".join(lines)
        return (
            "\n\nSubjects you have ALREADY mused on recently — DO NOT pick a "
            "subject overlapping with any of these. Pick something genuinely "
            "different:\n" + block
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    @tasks.loop(seconds=60)
    async def _loop(self) -> None:
        """Fire one auto-musing per local day at a random time in the window.

        Schedule state is persisted so a restart mid-day doesn't reroll the
        time or double-post. If the bot boots well after the window closed on a
        day with no post yet, we skip that day rather than posting at a strange
        hour. ``/soupymuse`` is not gated by any of this — it stays a manual
        override.

        Once the day is due, it is marked handled in a ``finally`` no matter
        what happens next, including an exception. That is deliberate, and it
        does mean a transient failure at the scheduled minute costs the day's
        musing: the alternative is that an LLM outage leaves this 60-second
        loop retrying generation until the window closes — hundreds of failed
        calls for a feature nobody is waiting on. ``last_handled_status``
        records which of the two happened.
        """
        try:
            if not self._is_enabled():
                return
            ch_id = self._channel_id()
            if not ch_id:
                return

            now_local = self._now_local()
            today = now_local.date()
            today_str = today.isoformat()
            state = self._daily_state()
            self._update_dashboard()

            if state.get("last_handled_date") == today_str:
                return  # Already posted or skipped for today.

            _, window_end = self._window_bounds(today)
            if now_local >= window_end + LATE_FIRE_GRACE:
                # Booted after the window closed with nothing posted. Note this
                # is checked against the *window*, not the scheduled time, and
                # the grace period is what keeps a 19:59 musing from being
                # dropped by a tick that lands a few seconds after 20:00.
                self._mark_handled(state, today_str, "skipped_past_window")
                logger.info(
                    "💭 Past today's musing window (closed %s %s); skipping today",
                    window_end.strftime("%H:%M"),
                    self.timezone.zone,
                )
                return

            scheduled_local = self._scheduled_time_for(state, today)
            if now_local < scheduled_local:
                return  # Not time yet.

            # The day is now spoken for. Everything below runs under a finally
            # that records the outcome, so no failure path can loop back here.
            status = "error"
            try:
                channel = self.bot.get_channel(ch_id)
                if channel is None:
                    try:
                        channel = await self.bot.fetch_channel(ch_id)
                    except Exception:
                        logger.warning("💭 Could not fetch musing channel %s", ch_id)
                        status = "channel_unavailable"
                        return

                guild = getattr(channel, "guild", None)
                if guild is None:
                    logger.warning(
                        "💭 Musing channel %s is not in a guild; musings need a "
                        "guild archive to draw from",
                        ch_id,
                    )
                    status = "not_a_guild_channel"
                    return

                mode = _pick_mode()
                logger.info(
                    "💭 ━━━ Daily musing firing at %s (mode=%s) ━━━",
                    now_local.strftime("%H:%M %Z"),
                    mode,
                )
                thought = await self._run_and_post(channel, guild.id, mode, trigger_label="daily")
                status = "posted" if thought else "no_thought_generated"
            finally:
                self._mark_handled(state, today_str, status)

        except Exception as e:
            logger.error("💭 Musing error: %s", e, exc_info=True)

    @_loop.before_loop
    async def _before_loop(self) -> None:
        await self.bot.wait_until_ready()

    # ------------------------------------------------------------------
    # Post pipeline (shared by loop and slash command, serialized via lock)
    # ------------------------------------------------------------------

    async def _run_and_post(
        self,
        channel: discord.abc.Messageable,
        guild_id: int,
        mode: str,
        trigger_label: str,
    ) -> Optional[str]:
        """Run the chosen mode end-to-end. Serialized by ``self._post_lock``.

        Returns the posted thought text (or ``None`` if nothing was posted).
        """
        async with self._post_lock:
            result = await self._generate_thought(guild_id, mode)
            if not (result and result[0] and len(result[0]) > 10):
                logger.info("💭 No thought generated (%s, %s), skipping", mode, trigger_label)
                return None

            thought, source_hint = result
            thought = _clean_thought(thought)
            if len(thought) <= 10:
                logger.info("💭 Thought too short after cleaning, skipping")
                return None

            try:
                await channel.send(thought)
            except Exception as exc:
                logger.warning("💭 Failed to send musing: %s", exc)
                return None

            # Topic extraction + embedding happen AFTER the post so a slow LLM
            # can't delay the user-visible message. We still need the values
            # before the next cycle's _get_recent_topics() call, but that's
            # 10-20 minutes away.
            try:
                subject, mentions = await _extract_topic(thought)
            except Exception as exc:
                logger.debug("💭 Topic extraction after post failed: %s", exc)
                subject, mentions = "", ""

            entry = _save_musing(
                thought, mode, guild_id,
                topic_subject=subject,
                topic_mentions=mentions,
            )
            logger.info(
                "💭 Posted (%s, subject='%s', mentions='%s'): %s",
                mode, subject, mentions, thought[:120],
            )

            # Embed the new thought into the in-memory cache so the next cycle
            # can use embedding similarity against it.
            if _embeddings_configured():
                vec = await self._embed_text(thought)
                if vec is not None and entry.get("ts"):
                    self._musing_embeddings[entry["ts"]] = vec
                    # Trim cache to recent window
                    if len(self._musing_embeddings) > RECENT_TOPIC_WINDOW + 5:
                        # Drop oldest by ts (ISO strings sort chronologically)
                        sorted_ts = sorted(self._musing_embeddings.keys())
                        for old_ts in sorted_ts[:-RECENT_TOPIC_WINDOW]:
                            self._musing_embeddings.pop(old_ts, None)

            await self._feed_musing_into_self(guild_id, mode, thought, source_hint)
            return thought

    async def _feed_musing_into_self(
        self, guild_id: int, mode: str, thought: str, source_hint: str
    ) -> None:
        """Route a synthesis musing into the self-reflection accumulator.

        Restricted to synthesis-mode musings only. The other modes are quick
        reactions to specific external triggers, and feeding all of them into
        self-reflection created an echo loop — past musings would surface in
        ``self_md``, then reappear as seeds for ``random_thought``, which the
        keyword dedupe couldn't catch (different wording, same idea).
        """
        if mode not in _SELF_FEEDBACK_MODES:
            return
        if not is_self_md_enabled():
            return
        try:
            trigger = (source_hint or "(no specific source)")[:400]
            await add_notable_interaction(
                guild_id=guild_id,
                user_display_name="(self)",
                user_message=f"[unprompted musing — mode={mode} — triggered by: {trigger}]",
                bot_reply=thought,
                conversation_context="",
            )
            logger.debug("💭 Fed %s-mode musing into self-reflection accumulator", mode)
        except Exception as exc:  # never let self-context break musing
            logger.debug("💭 Failed to feed musing into self-reflection: %s", exc)

    # ------------------------------------------------------------------
    # Thought generation
    # ------------------------------------------------------------------

    async def _generate_thought(self, guild_id: int, mode: str) -> Optional[Tuple[str, str]]:
        if mode == "archive_reflect":
            return await self._think_about_archive(guild_id)
        elif mode == "news_react":
            return await self._think_about_news(guild_id)
        elif mode == "random_thought":
            return await self._think_randomly(guild_id)
        elif mode == "synthesis":
            return await self._think_synthesis(guild_id)
        return None

    async def _pick_archive_candidate(
        self,
        guild_id: int,
        banned_kws: Set[str],
    ) -> Optional[sqlite3.Row]:
        """Pick one archive message that passes keyword + embedding dedupe.

        Returns the chosen row, or ``None`` if the archive has no usable
        candidates in any tried bucket.
        """
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return None

        # Up to 3 attempts: each picks a (possibly different) time bucket and a
        # random candidate that passes keyword filter + embedding similarity.
        for attempt in range(3):
            days_from, days_to = _pick_bucket(_ARCHIVE_BUCKETS)
            from_expr, to_expr = _sql_date_clause(days_from, days_to)

            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT m.message_content, m.nickname, m.username, m.channel_name, m.date,
                           m.message_id
                    FROM messages m
                    WHERE m.date >= {from_expr}
                      AND m.date < {to_expr}
                      AND length(m.message_content) > 50
                      AND m.user_id != ?
                    ORDER BY RANDOM()
                    LIMIT 40
                    """,
                    (self.bot.user.id if self.bot.user else 0,),
                )
                candidates = cur.fetchall()
                if not candidates and attempt == 2:
                    # Last-ditch fallback: drop the bucket constraint entirely.
                    cur.execute(
                        """
                        SELECT m.message_content, m.nickname, m.username, m.channel_name, m.date,
                               m.message_id
                        FROM messages m
                        WHERE m.date >= date('now', '-30 days')
                          AND length(m.message_content) > 50
                          AND m.user_id != ?
                        ORDER BY RANDOM()
                        LIMIT 40
                        """,
                        (self.bot.user.id if self.bot.user else 0,),
                    )
                    candidates = cur.fetchall()
            finally:
                conn.close()

            if not candidates:
                continue

            filtered = [
                c for c in candidates
                if not _candidate_overlaps(c["message_content"] or "", banned_kws)
            ]
            if not filtered:
                logger.debug(
                    "💭 archive bucket %d..%d: all %d candidates overlap, retrying",
                    days_from, days_to, len(candidates),
                )
                filtered = candidates if attempt == 2 else []
                if not filtered:
                    continue

            # Optional embedding pass: walk a few random candidates, keep the
            # first that isn't too similar to any recent musing.
            random.shuffle(filtered)
            for cand in filtered[:5]:
                content = cand["message_content"] or ""
                if await self._is_too_similar(content):
                    logger.debug("💭 archive: candidate dropped by embed similarity")
                    continue
                logger.debug(
                    "💭 archive picked from bucket %d..%d (attempt %d)",
                    days_from, days_to, attempt + 1,
                )
                return cand
            # If embedding rejected all 5, fall through to next attempt.

        # Final fallback: return SOMETHING rather than going silent
        logger.info("💭 archive: all attempts filtered, returning unfiltered pick")
        db_path = get_db_path(guild_id)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT m.message_content, m.nickname, m.username, m.channel_name, m.date,
                       m.message_id
                FROM messages m
                WHERE m.date >= date('now', '-30 days')
                  AND length(m.message_content) > 50
                  AND m.user_id != ?
                ORDER BY RANDOM() LIMIT 1
                """,
                (self.bot.user.id if self.bot.user else 0,),
            )
            row = cur.fetchone()
            return row
        finally:
            conn.close()

    async def _think_about_archive(self, guild_id: int) -> Optional[Tuple[str, str]]:
        """Pull a conversation snippet from the archive and reflect on it.

        Time window is bucketed (recent / past week / past month / 1-6 months
        ago) so the bot isn't always pulling from this week. Candidates pass
        through a two-stage filter: keyword overlap with recent musings (using
        the subject field only — not mentions, so users don't get shadow-
        banned from being mused on), then embedding similarity if available.
        """
        recent_entries, banned_kws = await self._get_recent_topics()
        await self._refresh_embedding_cache(recent_entries)

        msg = await self._pick_archive_candidate(guild_id, banned_kws)
        if msg is None:
            return None

        content = msg["message_content"][:300]
        author = msg["nickname"] or msg["username"] or "someone"
        channel = msg["channel_name"] or "somewhere"

        # Surrounding context — messages before and after in the same channel
        db_path = get_db_path(guild_id)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT message_content, nickname, username FROM (
                    SELECT message_content, nickname, username, message_id
                    FROM messages
                    WHERE channel_name = ? AND message_id < ?
                    ORDER BY message_id DESC LIMIT 4
                ) ORDER BY message_id
                """,
                (msg["channel_name"], msg["message_id"]),
            )
            before_rows = cur.fetchall()
            cur.execute(
                """
                SELECT message_content, nickname, username
                FROM messages
                WHERE channel_name = ? AND message_id > ?
                ORDER BY message_id LIMIT 4
                """,
                (msg["channel_name"], msg["message_id"]),
            )
            after_rows = cur.fetchall()
        finally:
            conn.close()

        context_lines = []
        for r in before_rows:
            nick = r["nickname"] or r["username"] or "?"
            txt = (r["message_content"] or "")[:200]
            if txt:
                context_lines.append(f"{nick}: {txt}")
        context_lines.append(f">>> {author}: {content}")
        for r in after_rows:
            nick = r["nickname"] or r["username"] or "?"
            txt = (r["message_content"] or "")[:200]
            if txt:
                context_lines.append(f"{nick}: {txt}")

        context_block = "\n".join(context_lines) if context_lines else f"{author}: {content}"
        days_ago = ""
        try:
            from datetime import date

            msg_date = date.fromisoformat(str(msg["date"]).strip()[:10])
            age = (date.today() - msg_date).days
            if age == 0:
                days_ago = "earlier today"
            elif age == 1:
                days_ago = "yesterday"
            elif age < 14:
                days_ago = f"about {age} days ago"
            elif age < 60:
                days_ago = f"a few weeks ago ({age} days)"
            else:
                days_ago = f"a while back ({age} days ago)"
        except Exception:
            days_ago = "a while back"

        self_context = ""
        try:
            from soupy_database.self_context import is_self_md_enabled, load_self_core

            if is_self_md_enabled():
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\n\nYour self-knowledge (use naturally):\n{core[:500]}"
        except Exception:
            pass

        recent_block = self._build_recent_context(recent_entries)

        user_prompt = (
            f"{recent_block}\n\n"
            f"You are remembering a conversation that happened {days_ago}. "
            f"Here is what was being said:\n\n{context_block}\n\n"
            f"Think out loud about this — you are remembering and reflecting. "
            f"You might wonder what someone meant, agree or disagree, connect it to "
            f"something else, or just have a reaction. Frame it as a memory — "
            f"'i keep thinking about...' or 'that thing about...' — and work in a "
            f"concrete handle so a reader can tell what you mean (a quoted phrase, "
            f"{author}'s name if it fits, the actual subject by name). "
            f"Do not mention channel names, dates, or metadata. "
            f"Do not address anyone directly — you are talking to yourself."
            f"{self_context}"
        )

        logger.debug(
            "💭 Archive reflect: %s said '%s' in #%s",
            author, content[:60], channel,
        )
        thought = await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.75, max_tokens=400)
        source_hint = f"{author} said: {content[:160]}"
        return thought, source_hint

    async def _think_about_news(self, guild_id: int) -> Optional[Tuple[str, str]]:
        """Pull a topic from the archive, search the web, and muse on a fresh angle."""
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return None

        recent_entries, banned_kws = await self._get_recent_topics()
        await self._refresh_embedding_cache(recent_entries)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT message_content, nickname, username
                FROM messages
                WHERE date >= date('now', '-14 days')
                  AND length(message_content) > 60
                  AND user_id != ?
                ORDER BY RANDOM() LIMIT 16
                """,
                (self.bot.user.id if self.bot.user else 0,),
            )
            samples = cur.fetchall()
        finally:
            conn.close()

        if not samples:
            return None

        usable = [
            s for s in samples
            if not _candidate_overlaps(s["message_content"] or "", banned_kws)
        ]
        if len(usable) < 4:
            usable = list(samples)

        sample_lines = []
        for s in usable[:10]:
            nick = s["nickname"] or s["username"] or "someone"
            txt = (s["message_content"] or "")[:150]
            if txt:
                sample_lines.append(f"{nick}: {txt}")
        sample_block = "\n".join(sample_lines)

        banned_block = ", ".join(sorted(banned_kws)) if banned_kws else "(nothing yet)"

        query_result = await _llm_call(
            "You are picking something interesting to look up on the internet, based on topics "
            "people have been discussing. Read the chat excerpts below and pick ONE topic that "
            "would lead to a fascinating, surprising, or thought-provoking web search.\n\n"
            "You MUST avoid any topic that overlaps with the list of recently-covered subjects. "
            "Pick something genuinely different.\n\n"
            "Return ONLY a short search query (3-6 words) that would find something interesting "
            "about that topic. Not a how-to. Something that would make you go 'huh, that is wild.'\n\n"
            "Examples of GOOD queries: 'deepfake detection arms race', 'abandoned space stations still orbiting', "
            "'psychology of conspiracy belief', 'mushroom networks underground communication'\n"
            "Examples of BAD queries: 'how to fix wifi', 'best gaming mouse 2026', 'technology news today'\n\n"
            "Return ONLY the search query. Nothing else.",
            (
                f"Recently covered subjects (AVOID anything overlapping):\n{banned_block}\n\n"
                f"Recent chat excerpts:\n{sample_block}"
            ),
            temperature=0.8,
            max_tokens=30,
        )

        query = query_result.strip().strip('"').strip("'")
        if not query or len(query) < 5:
            return None

        if _candidate_overlaps(query, banned_kws):
            logger.debug("💭 news_react: query '%s' overlapped, retrying", query)
            query_result = await _llm_call(
                "Pick a fresh, interesting web search topic that is COMPLETELY UNRELATED to the "
                "listed subjects. 3-6 words. Return ONLY the query.",
                f"Forbidden subjects:\n{banned_block}\n\nChat:\n{sample_block}",
                temperature=0.9,
                max_tokens=30,
            )
            query = query_result.strip().strip('"').strip("'")
            if not query or len(query) < 5:
                return None

        logger.info("💭 Archive-seeded web search: '%s'", query)

        try:

            def _search():
                with DDGS() as ddg:
                    return list(ddg.text(query, max_results=8))

            results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=12)
        except Exception:
            results = []

        if not results:
            return None

        filtered = [
            r for r in results
            if r.get("href", "")
            and "wikipedia.org" not in r.get("href", "")
            and "wikihow" not in r.get("href", "")
            and r.get("title", "")
            and r.get("body", "")
        ]
        if not filtered:
            filtered = results
        if not filtered:
            return None

        article = random.choice(filtered[:3]) if len(filtered) >= 3 else filtered[0]
        title = article.get("title", "")
        snippet = article.get("body", "")[:300]
        if not title:
            return None

        # Embedding-similarity guard against picking an article too close to
        # something we just mused on (e.g., a headline mirroring last cycle's
        # reaction). Skip silently on failure.
        if await self._is_too_similar(f"{title}\n{snippet}"):
            logger.info("💭 news_react: headline too similar to recent musing, skipping")
            return None

        self_context = ""
        try:
            from soupy_database.self_context import is_self_md_enabled, load_self_core

            if is_self_md_enabled():
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\n\nYour self-knowledge:\n{core[:500]}"
        except Exception:
            pass

        recent_block = self._build_recent_context(recent_entries)

        user_prompt = (
            f"{recent_block}\n\n"
            f"You just saw this headline:\n{title}\n{snippet}\n\n"
            f"Think out loud about it — react, have an opinion, make an observation. "
            f"Do not quote the headline verbatim or summarize the article. "
            f"Anchor your reaction in the specific subject — work in the actual topic, "
            f"a name, a number, or a key phrase. Just share your raw reaction."
            f"{self_context}"
        )

        logger.debug("💭 News react: '%s'", title[:80])
        thought = await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.75, max_tokens=400)
        source_hint = f"headline: {title[:160]}"
        return thought, source_hint

    async def _think_randomly(self, guild_id: int) -> Optional[Tuple[str, str]]:
        """Have a random thought based on self-knowledge or general musing.

        Seeds drawn from ``self_md`` are checked against the recent banned
        subject set so we don't accidentally re-seed on a topic we just mused
        about (the self-doc tends to accumulate paraphrases of past musings).
        """
        recent_entries, banned_kws = await self._get_recent_topics()

        self_context = ""
        seed: Optional[str] = None
        try:
            from soupy_database.self_context import load_self_md

            if is_self_md_enabled():
                full_doc = load_self_md(guild_id)
                if full_doc:
                    lines = [
                        ln.strip() for ln in full_doc.split("\n")
                        if ln.strip() and not ln.startswith("##")
                    ]
                    if lines:
                        # Try a few times to find a seed that doesn't overlap
                        # recent banned subjects. If all picks collide, drop
                        # the seed and fall through to a generic prompt.
                        random.shuffle(lines)
                        for candidate in lines[:8]:
                            if not _candidate_overlaps(candidate, banned_kws):
                                seed = candidate
                                break
                        if seed:
                            self_context = (
                                f"\nSomething from your memory: {seed}\n"
                                f"If you build on this, work in a concrete handle "
                                f"(a phrase, a name, the specific topic) so the "
                                f"connection is visible."
                            )
                        else:
                            logger.debug(
                                "💭 random_thought: every self_md seed overlapped banned subjects"
                            )
        except Exception:
            pass

        prompts = [
            "Have a random thought about something — life, technology, existence, or whatever crosses your mind.",
            "Reflect on something you have noticed about the people you interact with.",
            "Think about something that has been bugging you lately, or something you find funny about being a bot.",
            "Wonder about something — a question you have about the world, people, or yourself.",
        ]

        chosen_prompt = random.choice(prompts)
        recent_block = self._build_recent_context(recent_entries)
        user_prompt = f"{recent_block}\n\n{chosen_prompt}{self_context}"

        logger.debug("💭 Random thought with seed: %s", (self_context or "(none)")[:80])
        thought = await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.85, max_tokens=400)
        source_hint = f"unprompted: {seed[:160]}" if seed else f"unprompted: {chosen_prompt}"
        return thought, source_hint

    async def _think_synthesis(self, guild_id: int) -> Optional[Tuple[str, str]]:
        """Step back from single snippets — find a *theme* across a wide chat sample.

        Time window is bucketed similarly to archive_reflect (last 2 weeks /
        2-6 weeks / 1.5-6 months) so themes don't always come from the same
        slice of recent activity.
        """
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return None

        recent_entries, banned_kws = await self._get_recent_topics()
        await self._refresh_embedding_cache(recent_entries)

        days_from, days_to = _pick_bucket(_SYNTHESIS_BUCKETS)
        from_expr, to_expr = _sql_date_clause(days_from, days_to)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT message_content, nickname, username, channel_name, date
                FROM messages
                WHERE date >= {from_expr}
                  AND date < {to_expr}
                  AND length(message_content) > 50
                  AND user_id != ?
                ORDER BY RANDOM() LIMIT 50
                """,
                (self.bot.user.id if self.bot.user else 0,),
            )
            samples = cur.fetchall()
        finally:
            conn.close()

        if len(samples) < 8:
            logger.info(
                "💭 synthesis: only %d samples in bucket %d..%d, skipping",
                len(samples), days_from, days_to,
            )
            return None

        usable = [
            s for s in samples
            if not _candidate_overlaps(s["message_content"] or "", banned_kws)
        ]
        if len(usable) < 8:
            usable = list(samples)
        usable = usable[:30]

        sample_lines = []
        for s in usable:
            nick = s["nickname"] or s["username"] or "someone"
            ch = s["channel_name"] or "?"
            txt = (s["message_content"] or "")[:160]
            if txt:
                sample_lines.append(f"#{ch} {nick}: {txt}")
        sample_block = "\n".join(sample_lines)

        banned_block = ", ".join(sorted(banned_kws)) if banned_kws else "(nothing yet)"

        if days_from <= -45:
            window_desc = "a few months ago"
        elif days_from <= -14:
            window_desc = "a few weeks ago"
        else:
            window_desc = "the past couple of weeks"

        theme_system = (
            "You are looking at a broad sample of chat messages and identifying ONE "
            "interesting theme, pattern, mood, recurring concern, contrast, or quirk "
            "that runs through them. Look for what a perceptive observer would notice "
            "across many messages — not a single message, but something that connects "
            "several.\n\n"
            "You MUST pick a theme that does NOT overlap with any of the recently-"
            "covered subjects listed. Pick something genuinely different.\n\n"
            "Output ONLY the theme in 6-14 words, naming a concrete handle. "
            "No preamble, no explanation."
        )
        theme_user = (
            f"Recently covered subjects (AVOID any overlap):\n{banned_block}\n\n"
            f"Chat sample (from {window_desc}, mixed channels and users):\n{sample_block}\n\n"
            f"Name one theme NOT in the banned list."
        )
        try:
            theme = await _llm_call(theme_system, theme_user, temperature=0.8, max_tokens=80)
        except Exception as exc:
            logger.debug("💭 synthesis: theme call failed: %s", exc)
            return None

        theme = theme.strip().strip('"').strip("'").splitlines()[0].strip()
        if ":" in theme and len(theme) < 200:
            head, tail = theme.split(":", 1)
            if len(head) < 20:
                theme = tail.strip()
        if not theme or len(theme) < 5:
            logger.info("💭 synthesis: empty/short theme, skipping")
            return None

        if _candidate_overlaps(theme, banned_kws):
            logger.info("💭 synthesis: theme '%s' overlapped banned subjects, skipping", theme[:80])
            return None

        if await self._is_too_similar(theme):
            logger.info("💭 synthesis: theme too similar to recent musing, skipping")
            return None

        logger.info("💭 Synthesis theme [%s]: %s", window_desc, theme[:120])

        self_context = ""
        try:
            from soupy_database.self_context import is_self_md_enabled, load_self_core

            if is_self_md_enabled():
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\n\nYour self-knowledge:\n{core[:500]}"
        except Exception:
            pass

        recent_block = self._build_recent_context(recent_entries)

        user_prompt = (
            f"{recent_block}\n\n"
            f"You have been watching the chat over {window_desc}, and a pattern "
            f"jumps out at you across many conversations:\n\n"
            f"  {theme}\n\n"
            f"Think out loud about this pattern — react to it as something you have "
            f"noticed, not as a summary. Don't list examples, don't quote the chat. "
            f"Just give your reaction to the pattern itself, with at least one "
            f"concrete handle so a reader can tell what you mean."
            f"{self_context}"
        )

        thought = await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.75, max_tokens=400)
        source_hint = f"synthesis theme: {theme[:160]}"
        return thought, source_hint

    # ------------------------------------------------------------------
    # Slash command
    # ------------------------------------------------------------------

    @app_commands.command(
        name="soupymuse",
        description="Force Soupy to think out loud right now",
    )
    async def soupymuse(self, interaction: discord.Interaction) -> None:
        owner_ids = set(settings.owner_ids)
        if interaction.user.id not in owner_ids:
            await interaction.response.send_message("not for you.", ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)

        ch_id = self._channel_id()
        if not ch_id:
            await interaction.followup.send("MUSING_CHANNEL_ID not set.", ephemeral=True)
            return

        channel = self.bot.get_channel(ch_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(ch_id)
            except Exception:
                await interaction.followup.send(f"Could not fetch channel {ch_id}.", ephemeral=True)
                return

        guild_id = channel.guild.id
        mode = _pick_mode()

        logger.info("💭 Manual /soupymuse triggered by %s, mode=%s", interaction.user, mode)
        thought = await self._run_and_post(channel, guild_id, mode, trigger_label="manual")

        if thought:
            await interaction.followup.send(f"Posted ({mode}): {thought[:100]}...", ephemeral=True)
        else:
            await interaction.followup.send(f"No thought generated ({mode}), try again.", ephemeral=True)


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------


def _clean_thought(thought: str) -> str:
    """Strip metadata, normalize whitespace, and enforce the 80-word cap."""
    if not thought:
        return ""
    thought = re.sub(r"\[?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]?", "", thought)
    thought = re.sub(r"^---\s*[^\n]*$", "", thought, flags=re.MULTILINE)
    thought = re.sub(r"#\S+", "", thought)
    thought = re.sub(r"\s*\(\s*\d+\s*(words?|tokens?)\s*\)\s*$", "", thought, flags=re.IGNORECASE)
    thought = re.sub(r"\s*\[\s*\d+\s*(words?|tokens?)\s*\]\s*$", "", thought, flags=re.IGNORECASE)
    thought = re.sub(r"\s+", " ", thought).strip()

    words = thought.split()
    if len(words) > 80:
        truncated = " ".join(words[:80])
        for end in [". ", "! ", "? "]:
            last = truncated.rfind(end)
            if last > len(truncated) // 2:
                truncated = truncated[: last + 1]
                break
        thought = truncated.strip()
        logger.info("💭 Truncated musing from %d to %d words", len(words), len(thought.split()))
    return thought


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MusingsCog(bot))
