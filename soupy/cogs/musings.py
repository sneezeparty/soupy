"""
Musings cog for Soupy Bot.

Soupy occasionally "thinks out loud" in a configured channel — reflecting on
things from the server archive, reacting to news, or musing about conversations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import discord
import pytz
from ddgs import DDGS
from discord import app_commands
from discord.ext import commands, tasks

from soupy.settings import openai_client, settings
from soupy_database.database import get_db_path
from soupy_database.self_context import add_notable_interaction, is_self_md_enabled

logger = logging.getLogger(__name__)

MUSINGS_ARCHIVE_PATH = os.path.join("data", "musings_archive.jsonl")
MAX_ARCHIVE_ENTRIES = 200

# How many recent musings to treat as "already covered". Used both to filter
# archive candidates source-side and to warn the LLM off the same subject.
RECENT_TOPIC_WINDOW = 15


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
    """Rewrite the archive file with the given entries (used after topic backfill)."""
    if not entries:
        return
    try:
        os.makedirs(os.path.dirname(MUSINGS_ARCHIVE_PATH), exist_ok=True)
        with open(MUSINGS_ARCHIVE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
    except Exception as exc:
        logger.debug("💭 Failed to rewrite musings archive: %s", exc)


def _save_musing(thought: str, mode: str, guild_id: int, topic: str = "") -> None:
    """Append a musing to the archive."""
    os.makedirs(os.path.dirname(MUSINGS_ARCHIVE_PATH), exist_ok=True)
    entry = {
        "text": thought,
        "mode": mode,
        "guild_id": guild_id,
        "ts": datetime.now(pytz.UTC).isoformat(),
        "topic": topic,
    }
    try:
        with open(MUSINGS_ARCHIVE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Trim if too long
        lines = open(MUSINGS_ARCHIVE_PATH, encoding="utf-8").read().splitlines()
        if len(lines) > MAX_ARCHIVE_ENTRIES:
            trimmed = lines[-MAX_ARCHIVE_ENTRIES:]
            open(MUSINGS_ARCHIVE_PATH, "w", encoding="utf-8").write("\n".join(trimmed) + "\n")
    except Exception as exc:
        logger.debug("💭 Failed to save musing to archive: %s", exc)


_STOPWORD_TOPICS = {
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


def _topic_keywords_set(topics: List[str]) -> Set[str]:
    """Flatten a list of comma-separated topic strings into a set of keywords."""
    out: Set[str] = set()
    for t in topics:
        if not t:
            continue
        for kw in t.split(","):
            kw = kw.strip().lower()
            # Skip empty and obvious filler. Names like "kat", "jdk" are kept.
            if kw and kw not in _STOPWORD_TOPICS:
                out.add(kw)
    return out


def _candidate_overlaps_topics(content: str, banned_keywords: Set[str]) -> bool:
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


async def _extract_topic(thought: str) -> str:
    """Single-musing topic extractor. Returns a short comma-separated keyword list."""
    if not thought or len(thought) < 10:
        return ""
    system = (
        "You extract topic keywords from a short musing. Output 1-3 concrete keyword "
        "phrases (1-2 words each), lowercase, comma-separated. Use specific nouns — names of "
        "people, products, places, the actual subject. Skip generic words like 'thing', "
        "'stuff', 'people', 'life'. Output ONLY the keywords, no explanation, no preamble."
    )
    try:
        raw = await _llm_call(system, thought, temperature=0.2, max_tokens=40)
    except Exception as exc:
        logger.debug("💭 Topic extraction failed: %s", exc)
        return ""
    raw = raw.strip().strip('"').strip("'").splitlines()[0].strip()
    # Sanity: if the model rambled, take only up to the first 80 chars and discard
    # anything that looks like an explanation.
    if len(raw) > 80:
        raw = raw[:80]
    if ":" in raw and len(raw) < 200:
        # "Topic: paradise, kat" -> "paradise, kat"
        raw = raw.split(":", 1)[1].strip()
    return raw.lower()


async def _batch_extract_topics(texts: List[str]) -> List[str]:
    """Extract topic keywords for many musings in a single LLM call (backfill path)."""
    if not texts:
        return []
    numbered = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
    system = (
        "You extract topic keywords from short musings. For each numbered musing, output "
        "one line in the form '[N] keyword1, keyword2' where keywords are 1-3 concrete "
        "lowercase phrases (1-2 words each) naming the specific subject — people, products, "
        "places, the actual topic. Skip generic filler like 'thing', 'people', 'stuff'. "
        "Output ONLY numbered lines, no preamble, no explanation."
    )
    try:
        raw = await _llm_call(system, numbered, temperature=0.2, max_tokens=400)
    except Exception as exc:
        logger.debug("💭 Batch topic extraction failed: %s", exc)
        return ["" for _ in texts]
    result = ["" for _ in texts]
    for line in raw.splitlines():
        m = re.match(r"\s*\[?(\d+)\]?[.:)\s]+(.+)", line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(texts):
            kw = m.group(2).strip().strip('"').strip("'").lower()
            if ":" in kw:
                kw = kw.split(":", 1)[1].strip()
            result[idx] = kw[:80]
    return result


MUSING_SYSTEM = (
    "you are soupy dafoe, thinking out loud in a discord channel. you are not responding to anyone — "
    "you are just sharing a single thought, observation, or reaction. "
    "write in lower case, no quotation marks.\n\n"
    "Keep it SHORT. 80 words maximum, 1-2 sentences is ideal, 3 sentences max. "
    "think of it like muttering one thing under your breath, not writing a journal entry.\n\n"
    "Pick ONE thing to think about. Not two, not three. ONE specific thought.\n\n"
    "ANCHOR THE THOUGHT. A reader should be able to tell what you are reacting to, even if "
    "you only nod at it sideways. work in a concrete handle — a name, a quoted phrase, a "
    "specific number, the actual topic by name. vague gestures like 'that thing' or 'that "
    "whole situation' on their own are not enough; pair them with something a stranger could "
    "latch onto. subtle is fine, opaque is not.\n\n"
    "do not address anyone directly. do not ask questions directed at the chat. "
    "do not include any URLs, timestamps, channel names, or metadata in your response. "
    "do NOT include word counts, token counts, parenthetical notes, or any meta commentary "
    "about your own output. just write the thought and stop. nothing after the final period."
)


# Mode weights — synthesis gets a healthy share so the bot frequently steps
# back from "one random snippet" and instead names a theme across many messages.
_MODE_WEIGHTS: List[Tuple[str, float]] = [
    ("archive_reflect", 0.30),
    ("news_react", 0.20),
    ("random_thought", 0.15),
    ("synthesis", 0.35),
]


def _pick_mode() -> str:
    names = [m for m, _ in _MODE_WEIGHTS]
    weights = [w for _, w in _MODE_WEIGHTS]
    return random.choices(names, weights=weights, k=1)[0]


# Time-bucket sampling for archive_reflect — biases each call into a different
# era so the bot isn't always pulling from the most recent week.
# Each tuple: (days_from, days_to, weight). days_from < days_to <= 0 means
# "between days_from days ago and days_to days ago (today = 0)".
_TIME_BUCKETS: List[Tuple[int, int, float]] = [
    (-2, 0, 0.20),      # last ~2 days
    (-9, -2, 0.30),     # 2-9 days ago
    (-30, -9, 0.25),    # 9-30 days ago
    (-180, -30, 0.25),  # 1-6 months ago
]


def _pick_time_bucket() -> Tuple[int, int]:
    weights = [w for _, _, w in _TIME_BUCKETS]
    bucket = random.choices(_TIME_BUCKETS, weights=weights, k=1)[0]
    return bucket[0], bucket[1]


class MusingsCog(commands.Cog):
    """Soupy thinks out loud in a configured channel."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # `or "UTC"` covers the "TIMEZONE unset" case; settings.timezone defaults
        # to America/Los_Angeles per .env-stable.example, but a brand-new install
        # could have it blank.
        self.timezone = pytz.timezone(settings.timezone or "UTC")
        self._loop.start()

    def cog_unload(self) -> None:
        self._loop.cancel()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        return settings.musing_enabled

    def _channel_id(self) -> Optional[int]:
        return settings.musing_channel_id

    def _poll_range(self) -> Tuple[int, int]:
        lo, hi = settings.musing_poll_minutes_min, settings.musing_poll_minutes_max
        return max(1, lo), max(lo + 1, hi)

    def _chance(self) -> float:
        return settings.musing_chance

    # ------------------------------------------------------------------
    # Recent-topic bookkeeping
    # ------------------------------------------------------------------

    async def _get_recent_topics(
        self, limit: int = RECENT_TOPIC_WINDOW
    ) -> Tuple[List[Dict[str, str]], Set[str]]:
        """Load recent musings, backfilling any missing topic tags lazily.

        Returns ``(entries, banned_keywords)`` where ``entries`` is the trailing
        slice (size ≤ limit) of the archive and ``banned_keywords`` is the
        flattened set of topic keywords across those entries.
        """
        all_entries = _load_all_musings()
        if not all_entries:
            return [], set()
        recent = all_entries[-limit:]
        missing_local_idx = [i for i, e in enumerate(recent) if not e.get("topic")]
        if missing_local_idx:
            texts = [recent[i].get("text", "") for i in missing_local_idx]
            new_topics = await _batch_extract_topics(texts)
            anything_set = False
            for local_idx, topic in zip(missing_local_idx, new_topics):
                if topic:
                    recent[local_idx]["topic"] = topic
                    # Mirror into the full list so we can persist
                    full_idx = len(all_entries) - len(recent) + local_idx
                    all_entries[full_idx]["topic"] = topic
                    anything_set = True
            if anything_set:
                await asyncio.to_thread(_persist_all_musings, all_entries)
                logger.info("💭 Backfilled %d musing topics", sum(1 for t in new_topics if t))
        banned = _topic_keywords_set([e.get("topic", "") for e in recent])
        return recent, banned

    def _build_recent_context(self, entries: List[Dict[str, str]]) -> str:
        """Render the "already mused on" list for inclusion in an LLM prompt."""
        if not entries:
            return ""
        lines = []
        for r in entries:
            text = (r.get("text", "") or "").strip()
            topic = (r.get("topic", "") or "").strip()
            if not text:
                continue
            handle = f"[{topic}] " if topic else ""
            lines.append(f"- {handle}{text[:140]}")
        if not lines:
            return ""
        block = "\n".join(lines)
        return (
            "\n\nThings you have ALREADY mused on recently — DO NOT repeat any of "
            "these subjects, people, or angles. Pick something genuinely different:\n"
            f"{block}"
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    @tasks.loop(seconds=60)
    async def _loop(self) -> None:
        try:
            if not self._is_enabled():
                return
            ch_id = self._channel_id()
            if not ch_id:
                return

            # Random interval between polls
            lo, hi = self._poll_range()
            wait_minutes = random.randint(lo, hi)
            await asyncio.sleep(wait_minutes * 60)

            # Roll the dice
            chance = self._chance()
            roll = random.random()
            if roll > chance:
                logger.debug("💭 Musing check: roll=%.3f > chance=%.2f, skipping", roll, chance)
                return

            logger.info("💭 ━━━ Musing triggered (roll=%.3f, chance=%.2f) ━━━", roll, chance)

            channel = self.bot.get_channel(ch_id)
            if channel is None:
                try:
                    channel = await self.bot.fetch_channel(ch_id)
                except Exception:
                    logger.warning("💭 Could not fetch musing channel %s", ch_id)
                    return

            guild_id = channel.guild.id
            mode = _pick_mode()
            logger.info("💭 Mode: %s", mode)
            await self._run_and_post(channel, guild_id, mode, trigger_label="auto")

        except Exception as e:
            logger.error("💭 Musing error: %s", e, exc_info=True)

    @_loop.before_loop
    async def _before_loop(self) -> None:
        await self.bot.wait_until_ready()

    # ------------------------------------------------------------------
    # Post pipeline (shared by loop and slash command)
    # ------------------------------------------------------------------

    async def _run_and_post(
        self,
        channel: discord.abc.Messageable,
        guild_id: int,
        mode: str,
        trigger_label: str,
    ) -> Optional[str]:
        """Run the chosen mode end-to-end: generate, clean, post, save, feed.

        Returns the posted thought text (or ``None`` if nothing was posted).
        """
        result = await self._generate_thought(guild_id, mode)
        if not (result and result[0] and len(result[0]) > 10):
            logger.info("💭 No thought generated (%s, %s), skipping", mode, trigger_label)
            return None

        thought, source_hint = result
        thought = _clean_thought(thought)
        if len(thought) <= 10:
            logger.info("💭 Thought too short after cleaning, skipping")
            return None

        # Extract a topic for source-side dedupe next cycle. Done BEFORE posting
        # so a save failure can't leave a posted thought with no topic.
        topic = await _extract_topic(thought)

        try:
            await channel.send(thought)
        except Exception as exc:
            logger.warning("💭 Failed to send musing: %s", exc)
            return None

        _save_musing(thought, mode, guild_id, topic=topic)
        logger.info("💭 Posted (%s, topic='%s'): %s", mode, topic, thought[:120])
        await self._feed_musing_into_self(guild_id, mode, thought, source_hint)
        return thought

    async def _feed_musing_into_self(
        self, guild_id: int, mode: str, thought: str, source_hint: str
    ) -> None:
        """Route a posted musing into the self-reflection accumulator.

        Musings are unprompted soupy-only utterances. We adapt them to the
        notable-interaction shape so the periodic reflection cycle treats them
        as material for opinion / self-knowledge growth.
        """
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
            logger.debug("💭 Fed musing into self-reflection accumulator (mode=%s)", mode)
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

    async def _think_about_archive(self, guild_id: int) -> Optional[Tuple[str, str]]:
        """Pull a conversation snippet from the archive and reflect on it.

        The time window is bucketed (recent / past week / past month / 1-6
        months ago) so the bot isn't always pulling from this week, and
        candidates are filtered against recent musing topics so the same
        ongoing conversation can't keep dominating.
        """
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return None

        recent_entries, banned_kws = await self._get_recent_topics()

        days_from, days_to = _pick_time_bucket()
        from_clause = f"date('now', '{days_from} days')"
        # days_to == 0 means "up to and including today"
        if days_to == 0:
            to_clause = "date('now', '+1 day')"
        else:
            to_clause = f"date('now', '{days_to} days')"

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT m.message_content, m.nickname, m.username, m.channel_name, m.date,
                       m.message_id
                FROM messages m
                WHERE m.date >= {from_clause}
                  AND m.date < {to_clause}
                  AND length(m.message_content) > 50
                  AND m.user_id != ?
                ORDER BY RANDOM()
                LIMIT 40
                """,
                (self.bot.user.id if self.bot.user else 0,),
            )
            candidates = cur.fetchall()
            # If the bucket is empty (e.g., quiet server, narrow window), fall
            # back to the last 30 days so we still produce something.
            if not candidates:
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
            if not candidates:
                return None

            # Filter out candidates whose content collides with recent topic
            # keywords. If everything collides, fall back to the raw pool — we
            # don't want to go silent just because the chat has been one-track.
            filtered = [
                c for c in candidates
                if not _candidate_overlaps_topics(c["message_content"] or "", banned_kws)
            ]
            if not filtered:
                logger.info(
                    "💭 archive_reflect: all %d candidates overlap recent topics, using raw pool",
                    len(candidates),
                )
                filtered = candidates
            else:
                logger.debug(
                    "💭 archive_reflect: %d/%d candidates passed topic filter",
                    len(filtered), len(candidates),
                )

            msg = random.choice(filtered)
            content = msg["message_content"][:300]
            author = msg["nickname"] or msg["username"] or "someone"
            channel = msg["channel_name"] or "somewhere"

            # Get surrounding context — messages before and after in the same channel
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

            context_lines = []
            for r in before_rows:
                nick = r["nickname"] or r["username"] or "?"
                txt = (r["message_content"] or "")[:200]
                if txt:
                    context_lines.append(f"{nick}: {txt}")
            context_lines.append(f">>> {author}: {content}")  # highlight the focus message
            for r in after_rows:
                nick = r["nickname"] or r["username"] or "?"
                txt = (r["message_content"] or "")[:200]
                if txt:
                    context_lines.append(f"{nick}: {txt}")
        finally:
            conn.close()

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

        # Self-knowledge for richer reflection
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
            f"You might wonder what someone meant, agree or disagree with what was said, "
            f"connect it to something else you know, or just have a reaction. "
            f"Frame it as a memory — like 'i keep thinking about...' or 'that thing about...' "
            f"BUT do not stop at the vague gesture: work in a concrete handle from the conversation "
            f"so a reader can tell what you are reflecting on — quote a short phrase someone "
            f"used, name {author} if it fits naturally, or reference the specific topic by name "
            f"(not 'that thing', but the actual subject). subtle is fine; cryptic is not. "
            f"Do not mention channel names, dates, or metadata. "
            f"Do not address anyone directly — you are talking to yourself."
            f"{self_context}"
        )

        logger.debug(
            "💭 Archive reflect [bucket=%d..%d days]: %s said '%s' in #%s",
            days_from, days_to, author, content[:60], channel,
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

        # Step 1: Grab random substantive messages from the archive as topic seeds
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

        # Prefer samples that don't overlap recent topics. Keep enough for variety.
        usable = [
            s for s in samples
            if not _candidate_overlaps_topics(s["message_content"] or "", banned_kws)
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

        # Step 2: Ask LLM to pick the most interesting topic and craft a search query
        query_result = await _llm_call(
            "You are picking something interesting to look up on the internet, based on topics "
            "people have been discussing. Read the chat excerpts below and pick ONE topic that "
            "would lead to a fascinating, surprising, or thought-provoking web search.\n\n"
            "You MUST avoid any topic that overlaps with the list of recently-covered subjects. "
            "Pick something genuinely different.\n\n"
            "Return ONLY a short search query (3-6 words) that would find something interesting "
            "about that topic. Not a how-to. Something that would make you go 'huh, that is wild.' "
            "Think deeper — not the obvious angle, but the weird, surprising, or lesser-known aspect.\n\n"
            "Examples of GOOD queries: 'deepfake detection arms race', 'abandoned space stations still orbiting', "
            "'psychology of conspiracy belief', 'mushroom networks underground communication'\n"
            "Examples of BAD queries: 'how to fix wifi', 'best gaming mouse 2026', 'technology news today'\n\n"
            "Return ONLY the search query. Nothing else.",
            (
                f"Recently covered subjects (AVOID anything overlapping these):\n{banned_block}\n\n"
                f"Recent chat excerpts:\n{sample_block}"
            ),
            temperature=0.8,
            max_tokens=30,
        )

        query = query_result.strip().strip('"').strip("'")
        if not query or len(query) < 5:
            return None

        # If the LLM picked something that still collides with banned keywords, retry once
        # with a stronger nudge. (Cheap insurance.)
        if _candidate_overlaps_topics(query, banned_kws):
            logger.debug("💭 news_react: first query '%s' overlapped recent topics; retrying", query)
            query_result = await _llm_call(
                "Pick a fresh, interesting web search topic that is COMPLETELY UNRELATED to the "
                "listed subjects. 3-6 words. Surprising or lesser-known angle preferred. "
                "Return ONLY the query.",
                f"Forbidden subjects:\n{banned_block}\n\nChat:\n{sample_block}",
                temperature=0.9,
                max_tokens=30,
            )
            query = query_result.strip().strip('"').strip("'")
            if not query or len(query) < 5:
                return None

        logger.info("💭 Archive-seeded web search: '%s'", query)

        # Step 3: Search
        try:

            def _search():
                with DDGS() as ddg:
                    return list(ddg.text(query, max_results=8))

            results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=12)
        except Exception:
            results = []

        if not results:
            return None

        # Filter junk
        filtered = [
            r
            for r in results
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

        # Load self-knowledge
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
            f"BUT do anchor your reaction in the specific subject — work in the actual topic, "
            f"a name, a number, or a key phrase from the headline or snippet so a reader can "
            f"tell what set you off. don't reduce it to 'that thing' with no handle attached. "
            f"Just share your raw reaction as a thought."
            f"{self_context}"
        )

        logger.debug("💭 News react: '%s'", title[:80])
        thought = await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.75, max_tokens=400)
        source_hint = f"headline: {title[:160]}"
        return thought, source_hint

    async def _think_randomly(self, guild_id: int) -> Optional[Tuple[str, str]]:
        """Have a random thought based on self-knowledge or general musing."""
        recent_entries, _ = await self._get_recent_topics()

        self_context = ""
        seed: Optional[str] = None
        try:
            from soupy_database.self_context import load_self_md

            if is_self_md_enabled():
                full_doc = load_self_md(guild_id)
                if full_doc:
                    # Pick a random section to think about
                    lines = [
                        ln.strip() for ln in full_doc.split("\n")
                        if ln.strip() and not ln.startswith("##")
                    ]
                    if lines:
                        seed = random.choice(lines)
                        self_context = (
                            f"\nSomething from your memory: {seed}\n"
                            f"If you build on this, work a concrete handle from it into the thought "
                            f"(a phrase, a name, the specific topic) so the connection is visible — "
                            f"don't just allude to it abstractly."
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

        Pulls ~40 messages from the past 14 days across many channels/users, asks
        the LLM to name a pattern or theme that does NOT overlap with recently-
        covered topics, then musings on that theme.
        """
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return None

        recent_entries, banned_kws = await self._get_recent_topics()

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT message_content, nickname, username, channel_name, date
                FROM messages
                WHERE date >= date('now', '-14 days')
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
            logger.info("💭 synthesis: only %d samples available, skipping", len(samples))
            return None

        # Bias the sample toward messages that don't overlap with recent topics,
        # so the LLM has fresh material to find a theme in.
        usable = [
            s for s in samples
            if not _candidate_overlaps_topics(s["message_content"] or "", banned_kws)
        ]
        if len(usable) < 8:
            usable = list(samples)
        # Cap at ~30 so we don't blow the LLM context.
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

        # Step 1: identify a theme
        theme_system = (
            "You are looking at a broad sample of recent chat messages and identifying ONE "
            "interesting theme, pattern, mood, recurring concern, contrast, or quirk that "
            "runs through them. Look for what a perceptive observer would notice across the "
            "whole sample — not a single message, but something that connects several.\n\n"
            "You MUST pick a theme that does NOT overlap with any of the recently-covered "
            "subjects listed. Pick something genuinely different.\n\n"
            "Output ONLY the theme in 6-14 words, naming a concrete handle (a person, a "
            "specific topic, a behavior). No preamble, no explanation."
        )
        theme_user = (
            f"Recently covered subjects (AVOID any overlap):\n{banned_block}\n\n"
            f"Chat sample (last 14 days, mixed channels and users):\n{sample_block}\n\n"
            f"Name one theme NOT in the banned list."
        )
        try:
            theme = await _llm_call(theme_system, theme_user, temperature=0.8, max_tokens=80)
        except Exception as exc:
            logger.debug("💭 synthesis: theme call failed: %s", exc)
            return None

        theme = theme.strip().strip('"').strip("'").splitlines()[0].strip()
        # Strip a leading label if the LLM added one ("Theme: ...")
        if ":" in theme and len(theme) < 200:
            head, tail = theme.split(":", 1)
            if len(head) < 20:
                theme = tail.strip()
        if not theme or len(theme) < 5:
            logger.info("💭 synthesis: empty/short theme, skipping")
            return None

        # Bail out if the LLM still picked something on the banned list.
        if _candidate_overlaps_topics(theme, banned_kws):
            logger.info("💭 synthesis: theme '%s' overlapped banned topics, skipping", theme[:80])
            return None

        logger.info("💭 Synthesis theme: %s", theme[:120])

        # Step 2: muse on the theme
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
            f"You have been watching the chat over the past couple of weeks, and a pattern "
            f"jumps out at you across many conversations:\n\n"
            f"  {theme}\n\n"
            f"Think out loud about this pattern — react to it as something you have noticed, "
            f"not as a summary. don't list examples, don't quote the chat. just give your "
            f"reaction to the pattern itself. work in at least one concrete handle (a person, "
            f"a specific topic, a phrase) so a reader can tell what you mean."
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
    thought = re.sub(r"#\S+", "", thought)  # channel references
    # Strip trailing word/token count annotations like "(52 words)" or "(token count: 48)"
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
