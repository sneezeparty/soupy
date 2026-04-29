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
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import discord
import pytz
from ddgs import DDGS
from discord import app_commands
from discord.ext import commands, tasks
from openai import OpenAI

from soupy_database.database import get_db_path

logger = logging.getLogger(__name__)

MUSINGS_ARCHIVE_PATH = os.path.join("data", "musings_archive.jsonl")
MAX_ARCHIVE_ENTRIES = 200


def _load_recent_musings(limit: int = 10) -> List[Dict[str, str]]:
    """Load the most recent musings from the archive."""
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
        return entries[-limit:]
    except Exception:
        return []


def _save_musing(thought: str, mode: str, guild_id: int) -> None:
    """Append a musing to the archive."""
    os.makedirs(os.path.dirname(MUSINGS_ARCHIVE_PATH), exist_ok=True)
    entry = {
        "text": thought,
        "mode": mode,
        "guild_id": guild_id,
        "ts": datetime.now(pytz.UTC).isoformat(),
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


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
)


async def _llm_call(
    system: str, user: str, temperature: float = 0.7, max_tokens: int = 200
) -> str:
    def _sync():
        return client.chat.completions.create(
            model=os.getenv("LOCAL_CHAT", "local-model"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    response = await asyncio.to_thread(_sync)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Thought modes
# ---------------------------------------------------------------------------

MUSING_SYSTEM = (
    "you are soupy dafoe, thinking out loud in a discord channel. you are not responding to anyone — "
    "you are just sharing a single thought, observation, or reaction. "
    "write in lower case, no quotation marks.\n\n"
    "Keep it SHORT. 80 words maximum, 1-2 sentences is ideal, 3 sentences max. "
    "think of it like muttering one thing under your breath, not writing a journal entry.\n\n"
    "Pick ONE thing to think about. Not two, not three. ONE specific thought.\n\n"
    "do not address anyone directly. do not ask questions directed at the chat. "
    "do not include any URLs, timestamps, channel names, or metadata in your response. "
    "do NOT include word counts, token counts, parenthetical notes, or any meta commentary "
    "about your own output. just write the thought and stop. nothing after the final period."
)


class MusingsCog(commands.Cog):
    """Soupy thinks out loud in a configured channel."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.timezone = pytz.timezone(os.getenv("TIMEZONE", "UTC"))
        self._loop.start()

    def cog_unload(self) -> None:
        self._loop.cancel()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        return os.getenv("MUSING_ENABLED", "false").lower() in ("true", "1", "yes")

    def _channel_id(self) -> Optional[int]:
        raw = os.getenv("MUSING_CHANNEL_ID", "").strip()
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    def _poll_range(self) -> Tuple[int, int]:
        try:
            lo = int(os.getenv("MUSING_POLL_MINUTES_MIN", "10"))
        except ValueError:
            lo = 10
        try:
            hi = int(os.getenv("MUSING_POLL_MINUTES_MAX", "20"))
        except ValueError:
            hi = 20
        return max(1, lo), max(lo + 1, hi)

    def _chance(self) -> float:
        try:
            return float(os.getenv("MUSING_CHANCE", "0.10"))
        except ValueError:
            return 0.10

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

            # Pick a random thought mode
            mode = random.choices(
                ["archive_reflect", "news_react", "random_thought"],
                weights=[0.45, 0.30, 0.25],
                k=1,
            )[0]

            logger.info("💭 Mode: %s", mode)
            thought = await self._generate_thought(guild_id, mode)

            if thought and len(thought) > 10:
                # Strip any accidentally included metadata
                import re
                thought = re.sub(r"\[?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]?", "", thought)
                thought = re.sub(r"^---\s*[^\n]*$", "", thought, flags=re.MULTILINE)
                thought = re.sub(r"#\S+", "", thought)  # channel references
                # Strip trailing word/token count annotations like "(52 words)" or "(token count: 48)"
                thought = re.sub(r"\s*\(\s*\d+\s*(words?|tokens?)\s*\)\s*$", "", thought, flags=re.IGNORECASE)
                thought = re.sub(r"\s*\[\s*\d+\s*(words?|tokens?)\s*\]\s*$", "", thought, flags=re.IGNORECASE)
                thought = re.sub(r"\s+", " ", thought).strip()

                # Hard word limit — truncate at sentence boundary if over 80 words
                words = thought.split()
                if len(words) > 80:
                    truncated = " ".join(words[:80])
                    for end in [". ", "! ", "? "]:
                        last = truncated.rfind(end)
                        if last > len(truncated) // 2:
                            truncated = truncated[:last + 1]
                            break
                    thought = truncated.strip()
                    logger.info("💭 Truncated musing from %d to %d words", len(words), len(thought.split()))

                if len(thought) > 10:
                    await channel.send(thought)
                    _save_musing(thought, mode, guild_id)
                    logger.info("💭 Posted: %s", thought[:120])
                else:
                    logger.info("💭 Thought too short after cleaning, skipping")
            else:
                logger.info("💭 No thought generated, skipping")

        except Exception as e:
            logger.error("💭 Musing error: %s", e, exc_info=True)

    @_loop.before_loop
    async def _before_loop(self) -> None:
        await self.bot.wait_until_ready()

    # ------------------------------------------------------------------
    # Thought generation
    # ------------------------------------------------------------------

    def _recent_musings_context(self, limit: int = 5) -> str:
        """Build a context block from recent musings so Soupy can build on them."""
        recent = _load_recent_musings(limit)
        if not recent:
            return ""
        lines = [r.get("text", "") for r in recent if r.get("text")]
        if not lines:
            return ""
        block = "\n".join(f"- {l}" for l in lines)
        return (
            f"\n\nThings you have thought about recently (do not repeat these — "
            f"build on them, go deeper, or think about something new):\n{block}"
        )

    async def _generate_thought(self, guild_id: int, mode: str) -> Optional[str]:
        if mode == "archive_reflect":
            return await self._think_about_archive(guild_id)
        elif mode == "news_react":
            return await self._think_about_news(guild_id)
        elif mode == "random_thought":
            return await self._think_randomly(guild_id)
        return None

    async def _think_about_archive(self, guild_id: int) -> Optional[str]:
        """Pull an interesting conversation snippet from the archive and reflect on it."""
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return None

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            # Get a random interesting conversation chunk from the last 30 days
            # Pick messages that are substantive (>50 chars) and not from the bot
            cur.execute(
                """
                SELECT m.message_content, m.nickname, m.username, m.channel_name, m.date,
                       m.message_id
                FROM messages m
                WHERE m.date >= date('now', '-30 days')
                  AND length(m.message_content) > 50
                  AND m.user_id != ?
                ORDER BY RANDOM()
                LIMIT 20
                """,
                (self.bot.user.id if self.bot.user else 0,),
            )
            candidates = cur.fetchall()
            if not candidates:
                return None

            # Pick one interesting message
            msg = random.choice(candidates)
            content = msg["message_content"][:300]
            author = msg["nickname"] or msg["username"] or "someone"
            channel = msg["channel_name"] or "somewhere"

            # Get surrounding context — messages before and after in the same channel
            # Use a subquery to find nearby messages by position, not by ID arithmetic
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
            else:
                days_ago = f"about {age} days ago"
        except Exception:
            days_ago = "a while back"

        # Load self-knowledge for richer reflection
        self_context = ""
        try:
            from soupy_database.self_context import load_self_core, is_self_md_enabled
            if is_self_md_enabled():
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\n\nYour self-knowledge (use naturally):\n{core[:500]}"
        except Exception:
            pass

        user_prompt = (
            f"You are remembering a conversation that happened {days_ago}. "
            f"Here is what was being said:\n\n{context_block}\n\n"
            f"Think out loud about this — you are remembering and reflecting. "
            f"You might wonder what someone meant, agree or disagree with what was said, "
            f"connect it to something else you know, or just have a reaction. "
            f"Frame it as a memory — like 'i keep thinking about...' or 'that thing about...' "
            f"Do not mention channel names, dates, or metadata. "
            f"Do not address anyone directly — you are talking to yourself."
            f"{self_context}"
            f"{self._recent_musings_context()}"
        )

        logger.debug("💭 Archive reflect: %s said '%s' in #%s", author, content[:60], channel)
        return await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.75, max_tokens=400)

    async def _think_about_news(self, guild_id: int) -> Optional[str]:
        """Pull a topic from the archive, search the web for something interesting about it, and muse."""
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return None

        # Step 1: Grab random substantive messages from the archive to find a topic seed
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
                ORDER BY RANDOM() LIMIT 8
                """,
                (self.bot.user.id if self.bot.user else 0,),
            )
            samples = cur.fetchall()
        finally:
            conn.close()

        if not samples:
            return None

        # Build a snippet of what people have been talking about
        sample_lines = []
        for s in samples:
            nick = s["nickname"] or s["username"] or "someone"
            txt = (s["message_content"] or "")[:150]
            if txt:
                sample_lines.append(f"{nick}: {txt}")

        sample_block = "\n".join(sample_lines)

        # Step 2: Ask LLM to pick the most interesting topic and craft a search query
        query_result = await _llm_call(
            "You are picking something interesting to look up on the internet, based on topics "
            "people have been discussing. Read the chat excerpts below and pick ONE topic that "
            "would lead to a fascinating, surprising, or thought-provoking web search.\n\n"
            "Return ONLY a short search query (3-6 words) that would find something interesting "
            "about that topic. Not a how-to. Something that would make you go 'huh, that is wild.' "
            "Think deeper — not the obvious angle, but the weird, surprising, or lesser-known aspect.\n\n"
            "Examples of GOOD queries: 'deepfake detection arms race', 'abandoned space stations still orbiting', "
            "'psychology of conspiracy belief', 'mushroom networks underground communication'\n"
            "Examples of BAD queries: 'how to fix wifi', 'best gaming mouse 2026', 'technology news today'\n\n"
            "Return ONLY the search query. Nothing else.",
            f"Recent chat excerpts:\n{sample_block}",
            temperature=0.8,
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

        # Load self-knowledge
        self_context = ""
        try:
            from soupy_database.self_context import load_self_core, is_self_md_enabled
            if is_self_md_enabled():
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\n\nYour self-knowledge:\n{core[:500]}"
        except Exception:
            pass

        user_prompt = (
            f"You just saw this headline:\n{title}\n{snippet}\n\n"
            f"Think out loud about it — react, have an opinion, make an observation. "
            f"Do not include the URL or headline in your response. Do not summarize the article. "
            f"Just share your raw reaction as a thought."
            f"{self_context}"
            f"{self._recent_musings_context()}"
        )

        logger.debug("💭 News react: '%s'", title[:80])
        return await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.75, max_tokens=400)

    async def _think_randomly(self, guild_id: int) -> Optional[str]:
        """Have a random thought based on self-knowledge or general musing."""
        self_context = ""
        try:
            from soupy_database.self_context import load_self_md, is_self_md_enabled
            if is_self_md_enabled():
                full_doc = load_self_md(guild_id)
                if full_doc:
                    # Pick a random section to think about
                    lines = [l.strip() for l in full_doc.split("\n") if l.strip() and not l.startswith("##")]
                    if lines:
                        seed = random.choice(lines)
                        self_context = f"\nSomething from your memory: {seed}"
        except Exception:
            pass

        prompts = [
            "Have a random thought about something — life, technology, existence, or whatever crosses your mind.",
            "Reflect on something you have noticed about the people you interact with.",
            "Think about something that has been bugging you lately, or something you find funny about being a bot.",
            "Wonder about something — a question you have about the world, people, or yourself.",
        ]

        user_prompt = random.choice(prompts) + self_context + self._recent_musings_context()

        logger.debug("💭 Random thought with seed: %s", (self_context or "(none)")[:80])
        return await _llm_call(MUSING_SYSTEM, user_prompt, temperature=0.85, max_tokens=400)


    # ------------------------------------------------------------------
    # Slash command
    # ------------------------------------------------------------------

    @app_commands.command(
        name="soupymuse",
        description="Force Soupy to think out loud right now",
    )
    async def soupymuse(self, interaction: discord.Interaction) -> None:
        owner_ids = set()
        try:
            raw = os.getenv("OWNER_IDS", "")
            owner_ids = {int(x.strip()) for x in raw.split(",") if x.strip()}
        except Exception:
            pass
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
        mode = random.choices(
            ["archive_reflect", "news_react", "random_thought"],
            weights=[0.50, 0.30, 0.20],
            k=1,
        )[0]

        logger.info("💭 Manual /soupymuse triggered by %s, mode=%s", interaction.user, mode)
        thought = await self._generate_thought(guild_id, mode)

        if thought and len(thought) > 10:
            import re
            thought = re.sub(r"\[?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]?", "", thought)
            thought = re.sub(r"^---\s*[^\n]*$", "", thought, flags=re.MULTILINE)
            thought = re.sub(r"#\S+", "", thought)
            thought = re.sub(r"\s+", " ", thought).strip()

        if thought and len(thought) > 10:
            await channel.send(thought)
            _save_musing(thought, mode, guild_id)
            await interaction.followup.send(f"Posted ({mode}): {thought[:100]}...", ephemeral=True)
        else:
            await interaction.followup.send(f"No thought generated ({mode}), try again.", ephemeral=True)


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MusingsCog(bot))
