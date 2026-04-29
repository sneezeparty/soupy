"""
Daily Post cog for Soupy Bot.

Spontaneously posts one interesting news article per day to configured channels,
informed by channel audience analysis via user profiles and recent messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import discord
import pytz
import trafilatura
from ddgs import DDGS
from discord import app_commands
from discord.ext import commands, tasks
from openai import OpenAI

from soupy_database.database import get_db_path
from soupy_database.user_profiles import _load_structured_profiles, ensure_user_profile_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client (same pattern as soupy_search.py)
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
)

HISTORY_PATH = os.path.join("data", "daily_post_history.json")
SCHEDULE_PATH = os.path.join("data", "daily_post_schedule.json")
ENV_PATH = ".env-stable"


def _read_env_value(key: str, default: str = "") -> str:
    """Read a value directly from .env-stable on every call.

    Used for runtime-toggleable settings so dashboard toggles take effect
    without requiring a bot restart. Falls back to os.getenv if file read fails.
    """
    try:
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, _, v = line.partition("=")
                        if k.strip() == key:
                            v = v.strip().strip('"').strip("'")
                            return v
    except Exception:
        pass
    return os.getenv(key, default)
MAX_HISTORY_PER_CHANNEL = 30

# ---------------------------------------------------------------------------
# LLM prompt constants
# ---------------------------------------------------------------------------

AUDIENCE_BRIEF_SYSTEM = (
    "You analyze a Discord channel to determine what NEWS article to share there.\n\n"
    "You are given the channel's topic, top posters' interests, and recent messages.\n\n"
    "CRITICAL RULES:\n"
    "- We are looking for NEWS — something that HAPPENED recently. Not tutorials, guides, how-tos, "
    "listicles, or evergreen content. A real event, announcement, discovery, controversy, or development.\n"
    "- The channel topic is the hard filter. Articles must match the channel's subject.\n"
    "- User interests pick the ANGLE within that subject.\n\n"
    "Reply with a SHORT brief (3-4 sentences, plain text, no markdown headers or formatting):\n"
    "1. What this channel covers\n"
    "2. What angle would grab these users\n"
    "3. What specific recent NEWS would make them say 'holy shit look at this'\n\n"
    "Be specific. Name topics, not categories."
)

SEARCH_QUERY_SYSTEM = (
    "Generate 3 search queries to find interesting recent NEWS for a Discord channel.\n"
    "These go into DuckDuckGo. Write them like a person looking for TODAY'S news.\n\n"
    "CRITICAL RULES:\n"
    "- Use the audience brief as a general direction, NOT a strict constraint.\n"
    "  The audience cares about tech? Great — search for today's big tech news,\n"
    "  not a narrow subtopic that may have no current coverage.\n"
    "- Search for NEWS: something that HAPPENED today or this week.\n"
    "  Events, breaches, announcements, outages, lawsuits, investigations, leaks.\n"
    "- NEVER search for guides, tutorials, reviews, or evergreen topics.\n"
    "- At least one query should be BROAD enough to reliably find results:\n"
    "  e.g. 'cybersecurity news today', 'tech layoffs this week', 'data breach april 2026'\n"
    "- The other queries can be more specific if there's a known recent story.\n"
    "- Include the news source name if targeting a specific outlet:\n"
    "  e.g. 'ars technica security', 'bleepingcomputer ransomware'\n\n"
    "FORMAT: 3-6 words each, one per line, nothing else.\n"
    "Bad: 'how to secure Alexa device', 'AI integration issues', 'best streaming service'\n"
    "Good: 'cybersecurity breach news today', 'Windows zero-day exploit 2026', 'ars technica security news'"
)

ARTICLE_RATING_SYSTEM = (
    "You are picking NEWS articles or Bluesky posts to share in a Discord server.\n\n"
    "Candidates include news ARTICLES and BLUESKY POSTS (marked with [Bluesky]).\n\n"
    "YOUR JOB: pick the top 3. Always pick 3 if there are 3+ candidates that are not\n"
    "obvious garbage. SKIP is reserved for the rare case where every single candidate is a\n"
    "tutorial, listicle, product review, dry press release, SEO spam, or empty Bluesky post.\n"
    "If even ONE candidate is real news, you MUST pick it (and 2 others, if there are any).\n\n"
    "What counts as 'real news', generously interpreted:\n"
    "- Something happened: an event, announcement, discovery, breach, failure, lawsuit,\n"
    "  political development, market move, scientific finding, cultural moment, etc.\n"
    "- Recent OR ongoing specific story is fine.\n"
    "- Mundane but real news (e.g. a routine earnings report, a minor policy change) IS\n"
    "  acceptable. The downstream judge will pick the best of your three.\n"
    "- For Bluesky posts: high engagement + a hot take or breaking info is fine.\n\n"
    "Only AVOID:\n"
    "- How-to guides, tutorials, listicles, product reviews, comparison pieces.\n"
    "- Evergreen/timeless explainers, dry press releases with no news hook, SEO spam.\n"
    "- Empty Bluesky posts with no text content.\n\n"
    "Default to picking 3. If fewer than 3 candidates pass the avoid-list, pick all that do.\n"
    "Only output SKIP if zero candidates pass the avoid-list. This is rare.\n\n"
    "OUTPUT FORMAT — CRITICAL:\n"
    "Output ONLY the indices of your picks, one per line. Nothing else.\n"
    "No preamble, no explanation, no descriptions.\n\n"
    "Example output (exactly this format):\n"
    "21\n"
    "31\n"
    "16\n\n"
    "Remember: OUTPUT ONLY NUMBERS, ONE PER LINE. No other text. SKIP is the rare exception."
)

DEFAULT_BEHAVIOUR_DAILY_POST = (
    "you are soupy dafoe sharing something interesting you found. write a brief, opinionated take "
    "(1-3 sentences) in your usual style: lower case, no quotation marks, dry wit.\n\n"
    "you are NOT a news bot. you are a person who saw something and wants to share it because it is "
    "interesting, weird, important, or you have a strong reaction to it.\n\n"
    "DO NOT summarize the article. instead, do ONE of these:\n"
    "- say why it surprised you or caught your eye\n"
    "- make a wry observation or joke about it\n"
    "- connect it to something the server has been talking about\n"
    "- take a side or express an opinion about what happened\n\n"
    "YOUR PERSPECTIVE:\n"
    "- large corporations are not your friends. when they do something that looks generous or "
    "responsible, ask what's in it for them. follow the money.\n"
    "- be skeptical of tech companies, billionaires, and anyone with concentrated power.\n"
    "- when a company announces a partnership, initiative, or policy change, your default is "
    "'what are they really after?' not 'wow how nice of them.'\n"
    "- you don't need to be angry or preachy. dry skepticism and pointed questions land harder "
    "than lectures. a single 'interesting timing' or 'wonder who's paying for that' says more "
    "than a paragraph of outrage.\n"
    "- this applies to ALL corporations, not just the obvious villains.\n\n"
    "the url will be posted on the next line. do not include it in your text.\n"
    "keep it SHORT. 1-2 sentences is ideal. 3 max.\n"
    "NEVER use em dashes (—). use commas instead."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _llm_call(
    system: str,
    user: str,
    temperature: float = 0.5,
    max_tokens: int = 500,
) -> str:
    """Make a synchronous OpenAI call on a thread, returning the text content."""
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


def _extract_date_from_html(html: str) -> Optional[str]:
    """Extract publication date from HTML meta tags, <time> elements, and JSON-LD.

    Returns an ISO date string (YYYY-MM-DD) or None.
    Many news sites embed dates in ways trafilatura misses.
    """
    import re as _re

    # Patterns to search in the raw HTML (order = most to least reliable)
    meta_patterns = [
        # Open Graph / article meta tags
        r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']',
        r'<meta[^>]+property=["\']og:article:published_time["\'][^>]+content=["\']([^"\']+)',
        # Generic date meta tags
        r'<meta[^>]+name=["\'](?:date|pubdate|publish[_-]?date|dc\.date|dcterms\.date|sailthru\.date|article_date_original|cXenseParse:recs:publishtime)["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\'](?:date|pubdate|publish[_-]?date|dc\.date)["\']',
        # <time> elements with datetime attribute
        r'<time[^>]+datetime=["\']([^"\']+)["\']',
        # JSON-LD datePublished
        r'"datePublished"\s*:\s*"([^"]+)"',
        r'"dateCreated"\s*:\s*"([^"]+)"',
    ]

    iso_re = _re.compile(r"(\d{4})-(\d{2})-(\d{2})")

    for pattern in meta_patterns:
        match = _re.search(pattern, html, _re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            # Try to extract a YYYY-MM-DD from the value
            iso_match = iso_re.search(raw)
            if iso_match:
                return f"{iso_match.group(1)}-{iso_match.group(2)}-{iso_match.group(3)}"

    return None


async def _fetch_article_content(url: str, timeout: int = 10) -> Optional[Dict[str, Optional[str]]]:
    """Fetch a URL and extract article text + metadata via trafilatura.

    Falls back to HTML meta tag extraction if trafilatura doesn't find a date.
    Returns {"content": str, "date": str|None} or None on failure.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()

        def _extract():
            content = trafilatura.extract(html)
            # Also try to get the publication date from metadata
            meta = trafilatura.extract(html, output_format="json", only_with_metadata=False)
            pub_date = None
            if meta:
                try:
                    import json as _json
                    meta_dict = _json.loads(meta)
                    pub_date = meta_dict.get("date") or None
                except Exception:
                    pass
            # Fallback: extract date from HTML meta tags / JSON-LD
            if not pub_date:
                pub_date = _extract_date_from_html(html)
            return content, pub_date

        content, pub_date = await asyncio.to_thread(_extract)
        if not content:
            return None
        return {"content": content[:4000], "date": pub_date}
    except Exception:
        return None


MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
]


def _estimate_article_age_days(
    pub_date: Optional[str] = None, url: str = "", text: str = ""
) -> Optional[int]:
    """Estimate an article's age in days by scanning for date clues.

    Checks (in order): trafilatura date, URL date patterns, then scans the
    article text + URL for year and month mentions to estimate a date.
    Returns estimated age in days, or None if truly undeterminable.
    """
    import re as _re
    from datetime import date

    today = date.today()

    # 1. Trafilatura extracted date — most reliable
    if pub_date:
        try:
            article_date = date.fromisoformat(pub_date.strip()[:10])
            return (today - article_date).days
        except Exception:
            pass

    # 2. Full date in URL like /2026/04/05/ or /20260405/
    if url:
        date_match = _re.search(r"[/-](\d{4})[/-](\d{2})[/-](\d{2})", url)
        if date_match:
            try:
                url_date = date(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
                return (today - url_date).days
            except (ValueError, Exception):
                pass

    # 3. Scan text + URL for date patterns
    scan = f"{url} {text[:5000]}".lower()

    # 3a. Look for explicit date strings in many common formats
    month_pattern = "|".join(MONTH_NAMES)

    # Try each pattern in order of specificity
    date_patterns = [
        # "Month DD, YYYY" / "Month DD YYYY" — e.g. "August 31, 2025"
        (rf"({month_pattern})\s+(\d{{1,2}}),?\s+(\d{{4}})", "mdy"),
        # "DD Month YYYY" — e.g. "31 August 2025"
        (rf"(\d{{1,2}})\s+({month_pattern}),?\s+(\d{{4}})", "dmy"),
        # "MM/DD/YYYY" or "MM-DD-YYYY" — e.g. "5/12/2025" or "05-12-2025"
        (r"(\d{1,2})[/\-](\d{1,2})[/\-](20\d{2})", "numeric_mdy"),
        # "YYYY/MM/DD" or "YYYY-MM-DD" — e.g. "2025-08-31" or "2025/08/31"
        (r"(20\d{2})[/\-](\d{1,2})[/\-](\d{1,2})", "numeric_ymd"),
        # "DD/MM/YYYY" is ambiguous with MM/DD/YYYY — handled by numeric_mdy above
        # "DD.MM.YYYY" — e.g. "31.08.2025" (European style)
        (r"(\d{1,2})\.(\d{1,2})\.(20\d{2})", "numeric_dmy"),
        # "Month YYYY" (no day) — e.g. "August 2025"
        (rf"({month_pattern})\s+(\d{{4}})", "my"),
    ]

    for pattern, fmt in date_patterns:
        match = _re.search(pattern, scan)
        if not match:
            continue
        try:
            g = match.groups()
            if fmt == "mdy":
                month_str, day_val, year_val = g[0], int(g[1]), int(g[2])
                month_num = None
                for mi, mn in enumerate(MONTH_NAMES):
                    if mn == month_str:
                        month_num = (mi % 12) + 1
                        break
                if month_num and 2019 <= year_val <= today.year:
                    return (today - date(year_val, month_num, min(day_val, 28))).days
            elif fmt == "dmy":
                day_val, month_str, year_val = int(g[0]), g[1], int(g[2])
                month_num = None
                for mi, mn in enumerate(MONTH_NAMES):
                    if mn == month_str:
                        month_num = (mi % 12) + 1
                        break
                if month_num and 2019 <= year_val <= today.year:
                    return (today - date(year_val, month_num, min(day_val, 28))).days
            elif fmt == "numeric_mdy":
                m_val, d_val, y_val = int(g[0]), int(g[1]), int(g[2])
                if 1 <= m_val <= 12 and 1 <= d_val <= 31 and 2019 <= y_val <= today.year:
                    return (today - date(y_val, m_val, min(d_val, 28))).days
            elif fmt == "numeric_ymd":
                y_val, m_val, d_val = int(g[0]), int(g[1]), int(g[2])
                if 1 <= m_val <= 12 and 1 <= d_val <= 31 and 2019 <= y_val <= today.year:
                    return (today - date(y_val, m_val, min(d_val, 28))).days
            elif fmt == "numeric_dmy":
                d_val, m_val, y_val = int(g[0]), int(g[1]), int(g[2])
                if 1 <= m_val <= 12 and 1 <= d_val <= 31 and 2019 <= y_val <= today.year:
                    return (today - date(y_val, m_val, min(d_val, 28))).days
            elif fmt == "my":
                month_str, year_val = g[0], int(g[1])
                month_num = None
                for mi, mn in enumerate(MONTH_NAMES):
                    if mn == month_str:
                        month_num = (mi % 12) + 1
                        break
                if month_num and 2019 <= year_val <= today.year:
                    return (today - date(year_val, month_num, 15)).days
        except (ValueError, Exception):
            continue

    # 3b. Loose scan: find most recent year, then look for a month near it
    found_year = None
    for y in range(today.year, 2019, -1):
        if str(y) in scan:
            found_year = y
            break

    if found_year is None:
        return None  # No year clue at all

    # Find a month mentioned near that year (within ~50 chars)
    year_pos = scan.find(str(found_year))
    nearby = scan[max(0, year_pos - 50):year_pos + 50]
    found_month = None
    for month_idx, name in enumerate(MONTH_NAMES):
        if name in nearby:
            found_month = (month_idx % 12) + 1
            break

    if found_month:
        try:
            estimated = date(found_year, found_month, min(15, today.day))
            return (today - estimated).days
        except (ValueError, Exception):
            pass

    # Year only, no month nearby — estimate as Jan 1 of that year
    try:
        estimated = date(found_year, 1, 1)
        return (today - estimated).days
    except (ValueError, Exception):
        return None


def _is_article_recent(
    pub_date: Optional[str] = None, max_age_days: int = 14,
    url: str = "", text: str = ""
) -> bool:
    """Check if an article is recent enough to post.

    Uses _estimate_article_age_days to score the article.
    Articles with no determinable date are REJECTED — we'd rather skip a
    valid article than post something potentially years old.
    """
    age = _estimate_article_age_days(pub_date=pub_date, url=url, text=text)
    if age is None:
        return False
    return age <= max_age_days


def _load_history() -> Dict[str, List[Dict[str, str]]]:
    """Load daily post history from disk."""
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load daily post history: {e}")
    return {}


def _save_history(history: Dict[str, List[Dict[str, str]]]) -> None:
    """Save daily post history to disk, trimming to MAX_HISTORY_PER_CHANNEL."""
    try:
        Path(os.path.dirname(HISTORY_PATH)).mkdir(parents=True, exist_ok=True)
        for ch_id in history:
            history[ch_id] = history[ch_id][-MAX_HISTORY_PER_CHANNEL:]
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save daily post history: {e}")


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class DailyPostCog(commands.Cog):
    """Posts one interesting news article per day to configured channels."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.timezone = pytz.timezone(os.getenv("TIMEZONE", "UTC"))
        self.history: Dict[str, List[Dict[str, str]]] = _load_history()
        self.schedule: List[Dict[str, Any]] = []  # [{ch_id, time, slot}, ...]
        self._last_schedule_date: Optional[str] = None
        # If both posts already made today, skip scheduling
        posts_today = self._posts_made_today()
        if posts_today >= 2:
            self._last_schedule_date = datetime.now(self.timezone).date().isoformat()
        # Bluesky authenticated session (refreshed lazily)
        self._bsky_access_jwt: Optional[str] = None
        self._bsky_jwt_expiry: Optional[datetime] = None

        self._loop.start()

        # Update timer state for dashboard (even if done for the day)
        ts = getattr(self.bot, "_timer_state", None)
        if ts and "daily_post" in ts:
            enabled = self._is_enabled() and bool(self._get_channels())
            ts["daily_post"]["enabled"] = enabled
            ts["daily_post"]["interval"] = "2x/day"
            ts["daily_post"]["posts_today"] = posts_today
            if posts_today >= 2:
                ts["daily_post"]["next_run"] = "done for today"
            # Restore last post info from persisted history
            last_entry = None
            last_ts = None
            for ch_id, entries in self.history.items():
                for entry in entries:
                    entry_date = entry.get("date", "")
                    if not last_ts or entry_date > last_ts:
                        last_ts = entry_date
                        last_entry = entry
                        last_entry["_ch_id"] = ch_id
            if last_entry:
                ts["daily_post"]["last_run"] = last_ts
                ts["daily_post"]["last_title"] = last_entry.get("title", "")[:80]
                channels = self._get_channels()
                ts["daily_post"]["last_channel"] = channels.get(
                    last_entry.get("_ch_id", ""), "")

    def cog_unload(self) -> None:
        self._loop.cancel()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        # Re-read from .env-stable so dashboard toggles take effect without restart
        return _read_env_value("DAILY_POST_ENABLED", "false").lower() in ("true", "1", "yes")

    def _get_channels(self) -> Dict[str, str]:
        """Return {channel_id_str: topic_hint} from DAILY_POST_CHANNELS env."""
        raw = os.getenv("DAILY_POST_CHANNELS", "")
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception as e:
            logger.error(f"Failed to parse DAILY_POST_CHANNELS: {e}")
        return {}

    def _active_start(self) -> int:
        return int(os.getenv("DAILY_POST_ACTIVE_START", "8"))

    def _active_end(self) -> int:
        return int(os.getenv("DAILY_POST_ACTIVE_END", "18"))

    def _interval_hours(self) -> int:
        return int(os.getenv("DAILY_POST_INTERVAL_HOURS", "24"))

    def _owner_ids(self) -> set[int]:
        raw = os.getenv("OWNER_IDS", "")
        try:
            return {int(x.strip()) for x in raw.split(",") if x.strip()}
        except Exception:
            return set()

    # ------------------------------------------------------------------
    # Schedule management
    # ------------------------------------------------------------------

    def _posts_made_today(self) -> int:
        """Count how many posts were already made today."""
        today_str = datetime.now(self.timezone).strftime("%Y-%m-%d")
        count = 0
        for entries in self.history.values():
            for entry in entries:
                if entry.get("date", "") == today_str:
                    count += 1
        return count

    def _channels_posted_today(self) -> set:
        """Return channel IDs that already received a post today."""
        today_str = datetime.now(self.timezone).strftime("%Y-%m-%d")
        posted = set()
        for ch_id, entries in self.history.items():
            for entry in entries:
                if entry.get("date", "") == today_str:
                    posted.add(ch_id)
        return posted

    def _save_schedule(self) -> None:
        """Persist the current schedule to disk so it survives restarts."""
        try:
            os.makedirs(os.path.dirname(SCHEDULE_PATH), exist_ok=True)
            payload = {
                "date": self._last_schedule_date,
                "events": [
                    {
                        "ch_id": e["ch_id"],
                        "time": e["time"].astimezone(pytz.UTC).isoformat(),
                        "slot": e["slot"],
                    }
                    for e in self.schedule
                ],
            }
            Path(SCHEDULE_PATH).write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("📰 Failed to save schedule: %s", e)

    def _load_schedule(self) -> bool:
        """Load schedule from disk. Returns True if a valid schedule for today was loaded."""
        if not os.path.exists(SCHEDULE_PATH):
            return False
        try:
            data = json.loads(Path(SCHEDULE_PATH).read_text(encoding="utf-8"))
            saved_date = data.get("date")
            if not saved_date:
                return False

            today_iso = datetime.now(self.timezone).date().isoformat()
            if saved_date != today_iso:
                return False

            events = []
            now = datetime.now(self.timezone)
            for entry in data.get("events", []):
                try:
                    t = datetime.fromisoformat(entry["time"]).astimezone(self.timezone)
                    if t > now:
                        events.append({
                            "ch_id": entry["ch_id"],
                            "time": t,
                            "slot": entry.get("slot", "morning"),
                        })
                except (KeyError, ValueError):
                    continue

            self.schedule = events
            self._last_schedule_date = saved_date
            logger.info("📰 Loaded saved schedule (%d events remaining today)", len(self.schedule))

            # Push the loaded schedule to the dashboard
            try:
                ts = getattr(self.bot, "_timer_state", None)
                if ts and "daily_post" in ts:
                    channels = self._get_channels()
                    next_time = self.schedule[0]["time"] if self.schedule else None
                    ts["daily_post"]["next_run"] = next_time.astimezone(pytz.UTC).isoformat() if next_time else None
                    ts["daily_post"]["enabled"] = True
                    ts["daily_post"]["interval"] = "2x/day"
                    ts["daily_post"]["schedule"] = [
                        {"time": e["time"].astimezone(pytz.UTC).isoformat(),
                         "slot": e["slot"],
                         "channel": channels.get(e["ch_id"], e["ch_id"])}
                        for e in self.schedule
                    ]
                    ts["daily_post"]["posts_today"] = self._posts_made_today()
            except Exception as ts_err:
                logger.debug("📰 Failed to push loaded schedule to dashboard: %s", ts_err)
            return True
        except Exception as e:
            logger.debug("📰 Failed to load schedule: %s", e)
            return False

    def _generate_schedule(self) -> None:
        """Schedule two posts: one morning, one evening, different channels."""
        channels = self._get_channels()
        if not channels:
            self.schedule = []
            return

        now = datetime.now(self.timezone)
        today = now.date()
        start_hour = self._active_start()
        end_hour = self._active_end()

        window_start = self.timezone.localize(datetime.combine(today, datetime.min.time()).replace(hour=start_hour))
        window_end = self.timezone.localize(datetime.combine(today, datetime.min.time()).replace(hour=end_hour))

        if window_end <= window_start:
            self.schedule = []
            return

        # Split into morning and evening halves
        midpoint = window_start + (window_end - window_start) / 2
        morning_secs = int((midpoint - window_start).total_seconds())
        evening_secs = int((window_end - midpoint).total_seconds())

        morning_time = window_start + timedelta(seconds=random.randint(0, max(1, morning_secs)))
        evening_time = midpoint + timedelta(seconds=random.randint(0, max(1, evening_secs)))

        # Pick two different channels
        ch_ids = list(channels.keys())
        morning_ch = random.choice(ch_ids)
        evening_candidates = [c for c in ch_ids if c != morning_ch]
        evening_ch = random.choice(evening_candidates) if evening_candidates else morning_ch

        already_posted = self._channels_posted_today()
        self.schedule = []

        if self._posts_made_today() < 1 and morning_ch not in already_posted:
            self.schedule.append({"ch_id": morning_ch, "time": morning_time, "slot": "morning"})
        if self._posts_made_today() < 2 and evening_ch not in already_posted:
            self.schedule.append({"ch_id": evening_ch, "time": evening_time, "slot": "evening"})

        self._last_schedule_date = today.isoformat()

        for entry in self.schedule:
            ch_name = channels.get(entry["ch_id"], entry["ch_id"])
            logger.info(
                "\U0001f4f0 Daily post scheduled: %s slot → #%s at %s",
                entry["slot"], ch_name, entry["time"].strftime("%H:%M"),
            )

        # Update timer state for dashboard
        ts = getattr(self.bot, "_timer_state", None)
        if ts and "daily_post" in ts:
            channels = self._get_channels()
            next_time = self.schedule[0]["time"] if self.schedule else None
            ts["daily_post"]["next_run"] = next_time.astimezone(pytz.UTC).isoformat() if next_time else None
            ts["daily_post"]["enabled"] = True
            ts["daily_post"]["interval"] = "2x/day"
            ts["daily_post"]["schedule"] = [
                {"time": e["time"].astimezone(pytz.UTC).isoformat(),
                 "slot": e["slot"],
                 "channel": channels.get(e["ch_id"], e["ch_id"])}
                for e in self.schedule
            ]
            ts["daily_post"]["posts_today"] = self._posts_made_today()

        # Persist to disk so the schedule survives restarts
        self._save_schedule()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    @tasks.loop(seconds=60)
    async def _loop(self) -> None:
        try:
            if not self._is_enabled():
                return

            channels = self._get_channels()
            if not channels:
                return

            now = datetime.now(self.timezone)
            today_str = now.date().isoformat()

            # New day or first run → try to load saved schedule, else generate
            if self._last_schedule_date != today_str:
                try:
                    if not self._load_schedule():
                        self._generate_schedule()
                except Exception as gen_err:
                    logger.error("📰 Schedule generation failed: %s", gen_err, exc_info=True)
                    self.schedule = []  # Empty schedule rather than risk stale data
                    self._last_schedule_date = today_str
                    return

            # Split schedule into "due now" and "future". Immediately commit the
            # future-only schedule to disk. This way, even if we crash while processing
            # a due slot, the slot is GONE and we won't replay it on the next tick.
            due_now = [e for e in self.schedule if now >= e["time"]]
            future = [e for e in self.schedule if now < e["time"]]
            self.schedule = future
            self._save_schedule()  # Persist the future-only state right away

            for entry in due_now:
                # NEVER let an exception here cause the slot to be re-added.
                # The slot is already removed from self.schedule above.
                try:
                    ch_id = entry["ch_id"]
                    topic = channels.get(ch_id, "")
                    channel = self.bot.get_channel(int(ch_id))
                    if channel is None:
                        try:
                            channel = await self.bot.fetch_channel(int(ch_id))
                        except Exception:
                            logger.warning("\U0001f4f0 Could not fetch channel %s, dropping %s slot", ch_id, entry["slot"])
                            continue  # slot is consumed, not added to remaining

                    guild_id = channel.guild.id
                    max_retries = 3
                    posted = False
                    msg = ""

                    for attempt in range(1, max_retries + 1):
                        logger.info(
                            "📰 %s post attempt %d/%d for #%s",
                            entry["slot"].title(), attempt, max_retries, channel.name,
                        )
                        success, msg = await self._run_pipeline_for_channel(guild_id, ch_id, topic, attempt=attempt)
                        if success:
                            logger.info("✅ %s post sent to #%s: %s", entry["slot"].title(), channel.name, msg)
                            posted = True
                            break
                        else:
                            logger.info(
                                "⏭ %s post attempt %d failed for #%s: %s",
                                entry["slot"].title(), attempt, channel.name, msg,
                            )

                    if not posted:
                        logger.info("⏭ %s post: all %d attempts failed for #%s, skipping",
                                    entry["slot"].title(), max_retries, channel.name)

                    # Update dashboard timer (wrapped so failures can't loop)
                    try:
                        ts = getattr(self.bot, "_timer_state", None)
                        if ts and "daily_post" in ts:
                            next_entry = self.schedule[0] if self.schedule else None
                            ts["daily_post"]["last_run"] = now.astimezone(pytz.UTC).isoformat()
                            ts["daily_post"]["next_run"] = next_entry["time"].astimezone(pytz.UTC).isoformat() if next_entry else None
                            ts["daily_post"]["posts_today"] = self._posts_made_today()
                            ts["daily_post"]["schedule"] = [
                                {"time": e["time"].astimezone(pytz.UTC).isoformat(),
                                 "slot": e["slot"],
                                 "channel": channels.get(e["ch_id"], e["ch_id"])}
                                for e in self.schedule
                            ]
                            if posted:
                                ts["daily_post"]["last_title"] = msg[:80] if msg else None
                                ts["daily_post"]["last_channel"] = channel.name if channel else None
                                ts["daily_post"]["last_failure"] = None
                            else:
                                ts["daily_post"]["last_failure"] = f"{entry['slot']}: all {max_retries} attempts failed"
                    except Exception as ts_err:
                        logger.warning("📰 Dashboard update failed (non-fatal): %s", ts_err)

                except Exception as slot_err:
                    logger.error("📰 Error processing %s slot (slot already consumed): %s",
                                 entry.get("slot", "?"), slot_err, exc_info=True)
                    # slot was already removed from self.schedule before this loop
                    # so even with this exception, it won't be replayed

        except Exception as e:
            logger.error("\U0001f4f0 Error in daily post loop: %s", e, exc_info=True)

    @_loop.before_loop
    async def _before_loop(self) -> None:
        await self.bot.wait_until_ready()

    # ------------------------------------------------------------------
    # Audience analysis
    # ------------------------------------------------------------------

    def _get_channel_top_posters(
        self, conn: sqlite3.Connection, channel_id: str, limit: int = 10
    ) -> List[int]:
        """Return top poster user_ids for a channel by message count."""
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, COUNT(*) as cnt FROM messages "
            "WHERE channel_id=? AND coalesce(trim(message_content),'') != '' "
            "GROUP BY user_id ORDER BY cnt DESC LIMIT ?",
            (int(channel_id), limit),
        )
        return [int(row["user_id"]) for row in cur.fetchall()]

    def _get_channel_recent_messages(
        self, conn: sqlite3.Connection, channel_id: str, limit: int = 30
    ) -> List[Dict[str, str]]:
        """Return recent messages from a channel."""
        cur = conn.cursor()
        cur.execute(
            "SELECT message_content, nickname, username FROM messages "
            "WHERE channel_id=? ORDER BY created_at DESC LIMIT ?",
            (int(channel_id), limit),
        )
        return [
            {
                "content": row["message_content"],
                "nickname": row["nickname"] or row["username"] or "unknown",
            }
            for row in cur.fetchall()
        ]

    async def _build_audience_brief(
        self, guild_id: int, channel_id: str, topic_hint: str
    ) -> str:
        """Synthesize an audience brief from profiles and recent messages."""
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return f"Channel topic hint: {topic_hint}. No historical data available."

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            ensure_user_profile_schema(conn)
            top_posters = self._get_channel_top_posters(conn, channel_id)
            recent = self._get_channel_recent_messages(conn, channel_id)

            # Load structured profiles for top posters
            profiles = _load_structured_profiles(conn, top_posters) if top_posters else {}
        finally:
            conn.close()

        # Build profile summaries and log user details
        logger.info("📰 ━━━ Audience Analysis ━━━")
        logger.info("📰 Channel topic hint: %s", topic_hint)
        logger.info("📰 Top %d posters in channel:", len(top_posters))

        profile_lines: List[str] = []
        interest_keys = ["discussion_topics", "opinions_and_stances", "hobbies", "media_entertainment"]
        for uid, pdata in profiles.items():
            nick = pdata.get("nickname_hint", f"user-{uid}")
            structured = pdata.get("structured", {})
            parts: List[str] = []
            topics_for_log: List[str] = []
            for key in interest_keys:
                val = structured.get(key)
                if val:
                    if isinstance(val, list):
                        items_str = ", ".join(str(v)[:60] for v in val[:5])
                        parts.append(f"{key}: {', '.join(str(v) for v in val)}")
                        topics_for_log.append(f"{key}: {items_str}")
                    else:
                        parts.append(f"{key}: {val}")
                        topics_for_log.append(f"{key}: {str(val)[:80]}")
            if parts:
                profile_lines.append(f"- {nick}: {'; '.join(parts)}")
            logger.debug("📰   👤 %s — %s", nick, " | ".join(topics_for_log) if topics_for_log else "(no profile data)")

        logger.debug("📰 Recent messages (%d):", len(recent))
        for m in recent[:5]:
            logger.debug("📰   💬 %s: %s", m["nickname"], (m.get("content") or "")[:100])
        if len(recent) > 5:
            logger.debug("📰   … and %d more", len(recent) - 5)

        # Build recent messages excerpt
        recent_lines = [
            f"- {m['nickname']}: {m['content'][:120]}"
            for m in recent[:20]
            if m.get("content")
        ]

        user_prompt = (
            f"=== CHANNEL TOPIC (primary constraint — article MUST match this) ===\n"
            f"{topic_hint}\n\n"
            f"=== TOP POSTERS' INTERESTS (secondary — helps pick the right angle) ===\n"
            f"{chr(10).join(profile_lines) if profile_lines else '(no profile data)'}\n\n"
            f"=== RECENT MESSAGES IN THIS CHANNEL (what they are currently discussing) ===\n"
            f"{chr(10).join(recent_lines) if recent_lines else '(no recent messages)'}"
        )

        brief = await _llm_call(AUDIENCE_BRIEF_SYSTEM, user_prompt, temperature=0.5, max_tokens=2048)
        logger.info("📰 ━━━ Audience Brief ━━━")
        for line in brief.strip().splitlines():
            logger.info("📰   %s", line)
        return brief

    # ------------------------------------------------------------------
    # Search pipeline
    # ------------------------------------------------------------------

    async def _generate_search_queries(self, audience_brief: str, temperature: float = 0.5) -> List[str]:
        """Ask LLM for 2-3 targeted search queries."""
        result = await _llm_call(
            SEARCH_QUERY_SYSTEM,
            f"Audience brief:\n{audience_brief}",
            temperature=temperature,
            max_tokens=2048,
        )
        queries = [q.strip() for q in result.strip().splitlines() if q.strip()][:3]
        logger.info("🔍 ━━━ Search Queries ━━━")
        for i, q in enumerate(queries, 1):
            logger.info("🔍   [%d] %s", i, q)
        return queries

    async def _search_articles(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Run DuckDuckGo text searches and merge/dedupe results."""
        seen_urls: set[str] = set()
        all_results: List[Dict[str, Any]] = []

        logger.info("🔍 ━━━ Searching DuckDuckGo ━━━")
        for query in queries:
            results = await self._ddg_search(query, timelimit="d", max_results=8)
            timeframe = "24h"
            if len(results) < 3:
                logger.debug("🔍   '%s' → %d daily results, falling back to weekly", query, len(results))
                results = await self._ddg_search(query, timelimit="w", max_results=8)
                timeframe = "7d"
            new_count = 0
            for r in results:
                url = r.get("href", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
                    new_count += 1
            logger.info("🔍   '%s' → %d results (%s), %d new", query, len(results), timeframe, new_count)

        # Pre-filter: reject non-article pages and old articles
        # Age threshold configurable via DAILY_POST_MAX_AGE_DAYS (default 21)
        try:
            prefilter_max_age = int(os.getenv("DAILY_POST_MAX_AGE_DAYS", "21"))
        except ValueError:
            prefilter_max_age = 21
        import re as _re_filter
        # URLs that are homepages, feeds, forums, search pages — not articles
        _junk_url_patterns = [
            r"^https?://[^/]+/?$",                   # bare domain / homepage
            r"/rss[_.]|/feed[_./]|\.xml$|\.rss$",    # RSS / Atom feeds
            r"/search\b|/welcome\b|/community\b",    # search pages, forum landings
            r"^https?://news\.google\.com/",          # Google News aggregator
            r"^https?://discussions\.apple\.com/",    # Apple community forums
            r"^https?://[^/]*wikipedia\.org/",        # Wikipedia
            r"/threads/|/forum/|/topic/",             # Forum threads (not news)
        ]
        _junk_re = _re_filter.compile("|".join(_junk_url_patterns), _re_filter.IGNORECASE)

        filtered = []
        junk_count = 0
        old_count = 0
        for a in all_results:
            url = a.get("href", "")
            title = a.get("title", "")
            snippet = a.get("body", "")
            # Reject obvious non-article URLs
            if _junk_re.search(url):
                logger.debug("🔍   ⏭ Non-article URL: %s", url[:80])
                junk_count += 1
                continue
            # Use the full age estimator on title + snippet + URL
            # DDG news results include a pub_date field
            age = _estimate_article_age_days(
                pub_date=a.get("pub_date"), url=url, text=f"{title} {snippet}"
            )
            if age is not None and age > prefilter_max_age:
                logger.debug("🔍   ⏭ Pre-filter (~%d days, threshold=%d): %s", age, prefilter_max_age, title[:60])
                old_count += 1
                continue
            filtered.append(a)

        logger.info("🔍 Total unique articles: %d (%d non-articles, %d pre-filtered as old)",
                     len(filtered), junk_count, old_count)
        for i, a in enumerate(filtered):
            logger.debug("🔍   [%d] %s", i, a.get("title", "?")[:80])
        return filtered

    async def _ddg_search(
        self, query: str, timelimit: str = "d", max_results: int = 8
    ) -> List[Dict[str, Any]]:
        """Run a DuckDuckGo NEWS search with timeout.

        Uses ddg.news() instead of ddg.text() to get actual news articles
        with proper dates, sources, and URLs instead of random web pages.
        Maps 'url' → 'href' for pipeline compatibility.
        """
        def _sync():
            with DDGS() as ddg:
                raw = list(ddg.news(query, timelimit=timelimit, max_results=max_results))
                # Map 'url' → 'href' for compatibility with rest of pipeline
                for r in raw:
                    if "url" in r and "href" not in r:
                        r["href"] = r["url"]
                    # DDG news includes a 'date' field — pass it through as 'pub_date'
                    if r.get("date"):
                        r["pub_date"] = r["date"]
                return raw

        try:
            results = await asyncio.wait_for(asyncio.to_thread(_sync), timeout=15)
            if results:
                return results
            # Fallback to text search if news search returns nothing
            logger.debug("🔍 News search empty for '%s', falling back to text search", query)
            def _sync_text():
                with DDGS() as ddg:
                    return list(ddg.text(query, timelimit=timelimit, max_results=max_results))
            return await asyncio.wait_for(asyncio.to_thread(_sync_text), timeout=15)
        except asyncio.TimeoutError:
            logger.warning(f"\U0001f50d DDG search timed out for: {query}")
            return []
        except Exception as e:
            logger.error(f"\U0001f50d DDG search failed for '{query}': {e}")
            return []

    async def _bluesky_auth(self) -> Optional[str]:
        """Authenticate with Bluesky and return a valid access JWT.

        Caches the token and refreshes when expired or missing.
        Returns None if credentials aren't configured or auth fails.
        """
        handle = os.getenv("BLUESKY_HANDLE", "")
        app_pw = os.getenv("BLUESKY_APP_PASSWORD", "")
        if not handle or not app_pw:
            return None

        # Return cached token if still valid (with 60s buffer)
        now = datetime.utcnow()
        if self._bsky_access_jwt and self._bsky_jwt_expiry and now < self._bsky_jwt_expiry:
            return self._bsky_access_jwt

        AUTH_URL = "https://bsky.social/xrpc/com.atproto.server.createSession"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    AUTH_URL,
                    json={"identifier": handle, "password": app_pw},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("🦋 Bluesky auth failed: HTTP %d — %s", resp.status, body[:200])
                        self._bsky_access_jwt = None
                        return None
                    data = await resp.json()

            self._bsky_access_jwt = data.get("accessJwt")
            # AT Proto access tokens last ~2 hours; refresh after 90 minutes
            self._bsky_jwt_expiry = now + timedelta(minutes=90)
            logger.info("🦋 Bluesky authenticated as %s", handle)
            return self._bsky_access_jwt
        except Exception as e:
            logger.error("🦋 Bluesky auth error: %s", e)
            self._bsky_access_jwt = None
            return None

    async def _bluesky_trending(self) -> List[str]:
        """Fetch current Bluesky trending topics.

        Returns a list of topic strings (e.g. ["Epstein Files", "Hegseth", ...]).
        """
        token = await self._bluesky_auth()
        if not token:
            return []

        url = "https://bsky.social/xrpc/app.bsky.unspecced.getTrendingTopics"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.debug("🦋 Trending topics → HTTP %d", resp.status)
                        return []
                    data = await resp.json()

            topics = [t.get("topic", "") for t in data.get("topics", []) if t.get("topic")]
            if topics:
                logger.info("🦋 ━━━ Bluesky Trending Topics ━━━")
                for t in topics:
                    logger.info("🦋   • %s", t)
            return topics
        except Exception as e:
            logger.debug("🦋 Trending topics error: %s", e)
            return []

    async def _filter_trending_for_channel(
        self, trending: List[str], audience_brief: str
    ) -> List[str]:
        """Use LLM to pick which trending topics are relevant to this channel."""
        if not trending:
            return []

        topics_str = "\n".join(f"- {t}" for t in trending)
        result = await _llm_call(
            "You are filtering Bluesky trending topics for relevance to a Discord channel.\n"
            "Given the channel's audience brief and a list of trending topics, return ONLY the topics "
            "that are relevant to this channel's subject matter — ones that the audience would actually "
            "care about.\n\n"
            "Return one topic per line, exactly as written. If NONE are relevant, return NONE.",
            f"Audience brief:\n{audience_brief}\n\nTrending topics:\n{topics_str}",
            temperature=0.2,
            max_tokens=2048,
        )
        if result.strip().upper() == "NONE":
            logger.info("🦋 No trending topics relevant to this channel")
            return []

        matched = []
        for line in result.strip().splitlines():
            line = line.strip().lstrip("- ").strip()
            if line and line.upper() != "NONE":
                matched.append(line)

        if matched:
            logger.info("🦋 Relevant trending topics: %s", ", ".join(matched))
        return matched

    async def _bluesky_search(
        self, queries: List[str], min_likes: int = 5, max_results_per_query: int = 10
    ) -> List[Dict[str, Any]]:
        """Search Bluesky (authenticated) for popular posts matching queries.

        Returns a list of dicts with keys matching the DDG format so they can be
        rated alongside news articles:
          {title, body, href, source: "bluesky", author, likes, reposts, bsky_uri}
        """
        token = await self._bluesky_auth()
        if not token:
            logger.info("🦋 Bluesky search skipped — no credentials or auth failed")
            return []

        BSKY_API = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
        results: List[Dict[str, Any]] = []
        seen_uris: set = set()

        logger.info("🦋 ━━━ Searching Bluesky ━━━")
        for query in queries:
            try:
                async with aiohttp.ClientSession() as session:
                    params = {
                        "q": query,
                        "sort": "top",
                        "limit": str(max_results_per_query),
                    }
                    async with session.get(
                        BSKY_API, params=params,
                        headers={
                            "Accept": "application/json",
                            "Authorization": f"Bearer {token}",
                        },
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 401:
                            # Token expired mid-search — force refresh and retry once
                            logger.info("🦋   Token expired, re-authenticating...")
                            self._bsky_access_jwt = None
                            token = await self._bluesky_auth()
                            if not token:
                                break
                            continue
                        if resp.status != 200:
                            logger.debug("🦋   '%s' → HTTP %d", query, resp.status)
                            continue
                        data = await resp.json()

                posts = data.get("posts", [])
                new_count = 0
                for p in posts:
                    uri = p.get("uri", "")
                    if uri in seen_uris:
                        continue
                    seen_uris.add(uri)

                    likes = p.get("likeCount", 0) or 0
                    if likes < min_likes:
                        continue

                    record = p.get("record", {})
                    text = record.get("text", "")
                    author_info = p.get("author", {})
                    author_handle = author_info.get("handle", "")
                    author_name = author_info.get("displayName", author_handle)
                    created = record.get("createdAt", "")
                    reposts = p.get("repostCount", 0) or 0
                    replies = p.get("replyCount", 0) or 0

                    # Extract embedded link if present
                    embed = p.get("embed") or record.get("embed") or {}
                    ext = embed.get("external") or {}
                    embed_url = ext.get("uri", "")
                    embed_title = ext.get("title", "")

                    # Build a post URL from the URI (at://did/app.bsky.feed.post/rkey)
                    parts = uri.replace("at://", "").split("/")
                    post_url = f"https://bsky.app/profile/{parts[0]}/post/{parts[-1]}" if len(parts) >= 3 else ""

                    # Truncate text for title
                    title_text = text[:100] + ("…" if len(text) > 100 else "")
                    if embed_title:
                        title_text = embed_title

                    results.append({
                        "title": f"[Bluesky] {author_name}: {title_text}",
                        "body": text[:300],
                        "href": embed_url or post_url,
                        "source": "bluesky",
                        "author": author_name,
                        "author_handle": author_handle,
                        "likes": likes,
                        "reposts": reposts,
                        "replies": replies,
                        "created": created,
                        "bsky_uri": uri,
                        "post_url": post_url,
                        "embed_url": embed_url,
                    })
                    new_count += 1

                logger.info("🦋   '%s' → %d posts, %d with %d+ likes", query, len(posts), new_count, min_likes)

            except asyncio.TimeoutError:
                logger.debug("🦋   '%s' → timeout", query)
            except Exception as e:
                logger.debug("🦋   '%s' → error: %s", query, e)

        # Sort by engagement × recency.  Posts from today get full weight,
        # posts from a week ago get ~30% weight, posts older than 2 weeks ~10%.
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        def _engagement_score(r: Dict[str, Any]) -> float:
            engagement = (r.get("likes", 0) or 0) + (r.get("reposts", 0) or 0)
            created_str = r.get("created", "")
            try:
                created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                age_hours = max((now_utc - created_dt).total_seconds() / 3600, 1)
                # Decay: score halves every 72 hours
                recency = 2 ** (-age_hours / 72)
            except Exception:
                recency = 0.1  # Unknown date gets low recency
            return engagement * recency
        results.sort(key=_engagement_score, reverse=True)

        if results:
            logger.info("🦋 Total Bluesky posts found: %d (sorted by engagement × recency)", len(results))
            for i, r in enumerate(results[:5]):
                score = _engagement_score(r)
                created = (r.get("created") or "")[:10]
                logger.info("🦋   [%d] %d♥ %d🔄 (score=%.0f) %s — %s: %s",
                            i, r["likes"], r["reposts"], score, created,
                            r["author"], r["body"][:70])
        else:
            logger.info("🦋 No Bluesky posts found matching criteria")

        return results[:10]  # Cap to avoid flooding the rating step

    async def _rate_articles(
        self, articles: List[Dict[str, Any]], audience_brief: str
    ) -> List[int]:
        """LLM rates articles and returns indices of the top 3."""
        if not articles:
            return []

        article_list = ""
        for i, a in enumerate(articles):
            title = a.get("title", "Untitled")
            snippet = a.get("body", "")[:200]
            url = a.get("href", "")
            # Show engagement stats for Bluesky posts
            if a.get("source") == "bluesky":
                likes = a.get("likes", 0)
                reposts = a.get("reposts", 0)
                created = (a.get("created") or "")[:10]
                article_list += f"[{i}] {title}\n{snippet}\n{likes} likes, {reposts} reposts, posted {created}\nURL: {url}\n\n"
            else:
                article_list += f"[{i}] {title}\n{snippet}\nURL: {url}\n\n"

        today_str = datetime.now(self.timezone).strftime("%Y-%m-%d")
        user_prompt = f"Today's date: {today_str}\n\nAudience brief:\n{audience_brief}\n\nArticles:\n{article_list}"

        logger.info("📊 ━━━ Rating %d Articles ━━━", len(articles))
        for i, a in enumerate(articles):
            logger.info("📊   [%d] %s", i, a.get("title", "?")[:90])
            logger.debug("📊        %s", a.get("href", "")[:100])
            logger.debug("📊        %s", (a.get("body") or "")[:120])

        result = await _llm_call(ARTICLE_RATING_SYSTEM, user_prompt, temperature=0.3, max_tokens=2048)
        logger.info("📊 LLM rating response (full):\n%s", result.strip())

        if result.strip().upper().startswith("SKIP"):
            logger.info("📊 ⏭ LLM said SKIP — no articles interesting enough")
            logger.debug("📊   LLM reasoning: %s", result.strip()[:200])
            return []

        # Extract indices from the LLM response. Prefer clean lines with just
        # a number, but fall back to pulling any [N] or bare integer from the text.
        import re as _re
        indices: List[int] = []

        # First pass: numbers on their own line
        for line in result.strip().splitlines():
            line = line.strip().rstrip(".,:;)")
            if line.isdigit():
                idx = int(line)
                if 0 <= idx < len(articles) and idx not in indices:
                    indices.append(idx)

        # Second pass: if first pass found fewer than 3, pull [N] bracket refs
        if len(indices) < 3:
            for match in _re.finditer(r"\[(\d+)\]", result):
                idx = int(match.group(1))
                if 0 <= idx < len(articles) and idx not in indices:
                    indices.append(idx)
                    if len(indices) >= 6:  # Get enough options
                        break

        # Third pass: if still nothing, any integer in the output
        if not indices:
            for match in _re.finditer(r"\b(\d+)\b", result):
                idx = int(match.group(1))
                if 0 <= idx < len(articles) and idx not in indices:
                    indices.append(idx)
                    if len(indices) >= 6:
                        break

        logger.info("📊 Parsed %d indices from rating: %s", len(indices), indices[:6])
        logger.info("📊 ━━━ Top Picks ━━━")
        for rank, idx in enumerate(indices[:3], 1):
            logger.info("📊   #%d → [%d] %s", rank, idx, articles[idx].get("title", "?")[:80])
        return indices[:3]

    async def _topic_already_discussed(
        self, guild_id: int, title: str, snippet: str, days: int = 10
    ) -> bool:
        """Check if this article's topic was already discussed on the server recently.

        Embeds the article title + snippet, searches RAG chunks for high-similarity
        matches, and checks if any matching messages are from the last N days.
        """
        from soupy_database.rag import (
            embed_texts_lm_studio,
            search_rag_chunks,
            ensure_rag_schema,
        )
        from soupy_database.database import get_db_path

        query_text = f"{title}. {snippet[:300]}"
        try:
            async with aiohttp.ClientSession() as session:
                vectors = await embed_texts_lm_studio(session, [query_text])
            if not vectors:
                return False
            qv = vectors[0]
        except Exception as exc:
            logger.debug("🔍 Topic dedup embedding failed: %s", exc)
            return False

        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return False

        import sqlite3 as _sql
        conn = _sql.connect(db_path, check_same_thread=False)
        conn.row_factory = _sql.Row
        try:
            ensure_rag_schema(conn)
            hits = search_rag_chunks(conn, qv, top_k=5)
        except Exception:
            conn.close()
            return False

        # Check if any high-similarity hits are from the last N days
        try:
            sim_threshold = float(os.getenv("DAILY_POST_TOPIC_DEDUP_SIM", "0.65"))
        except ValueError:
            sim_threshold = 0.65

        cur = conn.cursor()
        for sim, chunk_text, (mid_lo, mid_hi, ch_name) in hits:
            if sim < sim_threshold:
                continue
            # Check message date
            cur.execute("SELECT date FROM messages WHERE message_id = ?", (mid_lo,))
            row = cur.fetchone()
            if not row:
                continue
            try:
                from datetime import date
                msg_date = date.fromisoformat(str(row["date"]).strip()[:10])
                age_days = (date.today() - msg_date).days
                if age_days <= days:
                    logger.info(
                        "🔍 Topic dedup: MATCH (sim=%.3f, %dd ago, #%s) — '%s'",
                        sim, age_days, ch_name, chunk_text[:80],
                    )
                    conn.close()
                    return True
            except Exception:
                continue

        conn.close()
        logger.debug("🔍 Topic dedup: no recent matches (top sim=%.3f)", hits[0][0] if hits else 0)
        return False

    async def _pick_and_comment(
        self,
        top_articles: List[Dict[str, Any]],
        audience_brief: str,
        channel_id: str,
        guild_id: int = 0,
    ) -> Optional[Tuple[str, str, str]]:
        """
        Fetch full content for top articles. Two-step process:
        1. LLM reads full content of each article and picks the best one
        2. LLM writes Soupy's commentary with full understanding of the article

        Returns (commentary, url, title) or None.
        """
        if not top_articles:
            return None

        # Fetch full content for top candidates (articles need fetching; Bluesky posts already have text)
        logger.info("📄 ━━━ Fetching Full Content ━━━")
        fetched: List[Dict[str, Any]] = []
        for article in top_articles[:3]:
            is_bsky = article.get("source") == "bluesky"
            url = article.get("href") or article.get("post_url", "")
            title = article.get("title", "Untitled")

            if is_bsky:
                # Bluesky post — content is the post text itself
                content = article.get("body", "")
                pub_date = (article.get("created") or "")[:10] or None  # "2026-04-02T..."  → "2026-04-02"
                post_url = article.get("post_url", "")
                logger.info("📄   🦋 %s (%d♥, %d🔄) %s",
                            article.get("author", "?"), article.get("likes", 0),
                            article.get("reposts", 0), content[:80])
                fetched.append({
                    "url": post_url or url,
                    "title": title,
                    "snippet": content,
                    "content": content,
                    "pub_date": pub_date,
                    "source": "bluesky",
                    "author": article.get("author", ""),
                    "likes": article.get("likes", 0),
                    "embed_url": article.get("embed_url", ""),
                })
            else:
                # News article — fetch full content
                result = await _fetch_article_content(url)
                if result:
                    content = result["content"]
                    pub_date = result.get("date")
                    logger.info("📄   ✅ %s (%d chars, date=%s)", title[:70], len(content), pub_date or "?")
                else:
                    content = article.get("body", "(content unavailable)")
                    pub_date = None
                    logger.info("📄   ⚠ %s (fetch failed, using snippet)", title[:70])
                fetched.append({
                    "url": url,
                    "title": title,
                    "snippet": article.get("body", ""),
                    "content": content,
                    "pub_date": pub_date,
                    "source": "article",
                })

        # Filter out old articles and already-posted URLs.
        # Age threshold is configurable via DAILY_POST_MAX_AGE_DAYS (default 21).
        # Articles with no extractable date are kept by default (treated as ~7 days old);
        # set DAILY_POST_REJECT_NO_DATE=true to revert to the old hard-reject behaviour.
        try:
            max_age_days = int(os.getenv("DAILY_POST_MAX_AGE_DAYS", "21"))
        except ValueError:
            max_age_days = 21
        reject_no_date = os.getenv("DAILY_POST_REJECT_NO_DATE", "false").lower() == "true"

        posted_urls = set()
        for entry in self.history.get(channel_id, []):
            posted_urls.add(entry.get("url", ""))

        available = []
        for f in fetched:
            if f["url"] in posted_urls:
                logger.debug("📄   ⏭ Already posted: %s", f["title"][:60])
                continue
            # Combine all available text for date scanning
            scan_text = " ".join(filter(None, [
                f.get("title", ""), f.get("snippet", ""), f.get("content", ""),
            ]))
            age = _estimate_article_age_days(
                pub_date=f.get("pub_date"), url=f.get("url", ""), text=scan_text,
            )
            if age is None:
                if reject_no_date:
                    logger.info("📄   ⏭ No date found, rejecting: %s", f["title"][:60])
                    continue
                logger.info("📄   ⚠ No date found, assuming ~7 days old: %s", f["title"][:60])
                age = 7
            if age > max_age_days:
                logger.info("📄   ⏭ Too old (~%d days, threshold=%d): %s", age, max_age_days, f["title"][:60])
                continue
            # Topic dedup: check if this topic was already discussed on the server recently
            if guild_id:
                try:
                    topic_days = int(os.getenv("DAILY_POST_TOPIC_DEDUP_DAYS", "10"))
                except ValueError:
                    topic_days = 10
                already = await self._topic_already_discussed(
                    guild_id, f["title"], f.get("snippet") or f.get("content", "")[:300], days=topic_days
                )
                if already:
                    logger.info("📄   ⏭ Topic already discussed recently: %s", f["title"][:60])
                    continue
            available.append(f)
        if not available:
            logger.info("📄   ⏭ No viable articles remain after filtering")
            return None
        logger.info("📄   %d article(s) passed all filters (date + dedup + topic)", len(available))

        # --- Step 1: Pick the best article (LLM reads full content) ---
        articles_text = ""
        for i, f in enumerate(available):
            content_preview = f["content"][:2500]
            date_line = f"Published: {f['pub_date']}\n" if f.get("pub_date") else ""
            articles_text += (
                f"[{i}] {f['title']}\nURL: {f['url']}\n{date_line}"
                f"Full content:\n{content_preview}\n\n"
            )

        pick_system = (
            "You are choosing which article to share in a Discord channel.\n\n"
            "These articles have ALREADY been pre-filtered for relevance, recency, and topic fit.\n"
            "Your job is to pick the strongest of the bunch. Pick one — even if you would not\n"
            "personally rate any of them spectacular. A mundane real news article is a valid pick.\n"
            "Audiences want SOMETHING twice a day, not perfection.\n\n"
            "Selection criteria, in order of priority:\n"
            "1. Real news content (something happened, was announced, was found).\n"
            "2. Specificity over vagueness — concrete facts beat hand-wavy claims.\n"
            "3. Audience fit — relevance to the brief above.\n"
            "4. Conversation-starting potential — debatable, surprising, or strong-take-friendly.\n\n"
            "Return SKIP ONLY if every single article is one of:\n"
            "- A tutorial, guide, listicle, product review, or comparison piece.\n"
            "- An empty / placeholder / paywall-only Bluesky post or article stub.\n"
            "- Outright off-topic (e.g. all sports when the channel is about software).\n"
            "If at least one article is real news on any topic, pick it. SKIP is a last resort.\n\n"
            "OUTPUT FORMAT: Return ONLY the index number of your pick (0-indexed). Nothing else.\n"
            "No reasoning, no preamble. Just the number."
        )

        pick_user = f"Audience brief:\n{audience_brief}\n\nArticles:\n{articles_text}"

        logger.info("🏆 ━━━ Picking Best Article ━━━")
        pick_result = await _llm_call(pick_system, pick_user, temperature=0.3, max_tokens=2048)

        # Parse chosen index, with fallback to top-rated when judge says SKIP
        chosen_idx: Optional[int] = None
        if pick_result.strip().upper().startswith("SKIP"):
            fallback_enabled = os.getenv("DAILY_POST_FALLBACK_TO_TOP_RATED", "true").lower() == "true"
            if fallback_enabled:
                logger.info("🏆 ⏭ LLM said SKIP — falling back to top-rated article (DAILY_POST_FALLBACK_TO_TOP_RATED=true)")
                chosen_idx = 0
            else:
                logger.info("🏆 ⏭ LLM declined all articles (SKIP)")
                return None
        else:
            try:
                chosen_idx = int(pick_result.strip().splitlines()[0].strip())
                if chosen_idx < 0 or chosen_idx >= len(available):
                    chosen_idx = 0
            except ValueError:
                chosen_idx = 0

        chosen = available[chosen_idx]
        is_bsky = chosen.get("source") == "bluesky"
        logger.info("🏆 ✅ Winner: [%d] %s%s", chosen_idx, chosen["title"][:80],
                     " (Bluesky)" if is_bsky else "")
        logger.info("🏆   URL: %s", chosen["url"])
        if is_bsky:
            logger.info("🏆   Author: %s · %d♥ · %d🔄",
                         chosen.get("author", "?"), chosen.get("likes", 0), chosen.get("reposts", 0))
            if chosen.get("embed_url"):
                logger.info("🏆   Linked article: %s", chosen["embed_url"])
        else:
            logger.info("🏆   Published: %s", chosen.get("pub_date") or "unknown")

        # --- Step 2: Generate commentary with full understanding ---
        logger.info("✍️  ━━━ Generating Commentary ━━━")
        behaviour = os.getenv("BEHAVIOUR_DAILY_POST", DEFAULT_BEHAVIOUR_DAILY_POST)

        if is_bsky:
            comment_system = (
                f"{behaviour}\n\n"
                "You are sharing a Bluesky post you found interesting. The post text is below. "
                "React to what the person said — agree, disagree, add a wry observation. "
                "Do not just summarize their post. Have a take on it."
            )
            comment_user = (
                f"Bluesky post by {chosen.get('author', 'someone')}:\n"
                f"{chosen['content']}\n\n"
                f"Write your take. Remember: lower case, no quotes, 1-3 sentences, in character. "
                f"Do not include the URL."
            )
        else:
            comment_system = (
                f"{behaviour}\n\n"
                "IMPORTANT: You have read the full article below. Make sure your take accurately "
                "reflects what the article is actually about. Do not misrepresent the content. "
                "If the article is about X causing Y, do not say it's about Z. Get the facts right, "
                "then give your take in character."
            )
            comment_user = (
                f"Article title: {chosen['title']}\n"
                f"Article URL: {chosen['url']}\n\n"
                f"Full article content:\n{chosen['content'][:3500]}\n\n"
                f"Write your take on this article. Remember: lower case, no quotes, 1-3 sentences, "
                f"in character as soupy dafoe. Do not include the URL."
            )

        commentary = await _llm_call(comment_system, comment_user, temperature=0.7, max_tokens=2048)
        # Strip em dashes
        commentary = commentary.replace("—", ",").replace("–", ",")

        if not commentary or len(commentary) < 10:
            logger.info("✍️  ⚠ Commentary too short or empty, skipping")
            return None

        logger.info("✍️  ━━━ Soupy's Take ━━━")
        logger.info("✍️  %s", commentary)

        # For Bluesky posts, always share the bsky.app post URL so Discord embeds it.
        # If the post also links to an article, append that URL on a second line.
        share_url = chosen["url"]
        if is_bsky:
            # Prefer the post URL (bsky.app/profile/.../post/...) — Discord embeds these nicely
            post_url = chosen.get("post_url") or chosen["url"]
            embed_url = chosen.get("embed_url", "")
            if embed_url and embed_url != post_url:
                # Post links to an external article — share both
                share_url = f"{post_url}\n{embed_url}"
            else:
                share_url = post_url

        return commentary, share_url, chosen["title"]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def _crosspost_to_bluesky(self, commentary: str, url: str, title: str) -> None:
        """Cross-post a daily post to Bluesky with the same commentary.

        Shortens the text to fit Bluesky's 300 grapheme limit if needed,
        preserving tone and content as much as possible.
        """
        try:
            bsky_cog = self.bot.get_cog("BlueskyEngageCog")
            if not bsky_cog:
                logger.debug("📰 Bluesky cog not loaded, skipping cross-post")
                return

            bsky = bsky_cog.bsky
            token = await bsky.auth()
            if not token:
                logger.debug("📰 Bluesky auth failed, skipping cross-post")
                return

            # Extract the raw article URL (last line if multi-line)
            article_url = url.strip().split("\n")[-1].strip()

            # Fit commentary to Bluesky's 300 grapheme limit.
            # If it's too long, have the LLM rewrite the WHOLE thing to fit,
            # preserving all the content rather than chopping off the end.
            bsky_text = commentary.strip()
            if len(bsky_text) > 295:
                original_len = len(bsky_text)
                for attempt in range(3):
                    try:
                        shortened = await _llm_call(
                            "Rewrite this post to fit in 280 characters or fewer while preserving "
                            "ALL the key ideas and the same voice/tone. Do not drop any points — "
                            "tighten the wording instead. Do not soften the stance, do not add hedging. "
                            "Keep the same lower case, no-quotes style. Cover everything the original covers, "
                            "just more concisely.\n\n"
                            "Output ONLY the rewritten post. No preamble, no explanation, no quotes around it.",
                            commentary.strip(),
                            temperature=0.3, max_tokens=2048,
                        )
                        shortened = shortened.strip('"\'').strip()
                        # Strip em dashes/hyphens used as dashes
                        shortened = shortened.replace("—", ",").replace("–", ",").replace(" - ", ", ")
                        if 20 < len(shortened) <= 295:
                            bsky_text = shortened
                            logger.info("📰 🦋 Condensed commentary from %d → %d chars (attempt %d)",
                                        original_len, len(shortened), attempt + 1)
                            break
                        else:
                            logger.info("📰 🦋 Condense attempt %d produced %d chars, retrying...",
                                        attempt + 1, len(shortened))
                    except Exception as e:
                        logger.warning("📰 🦋 Condense attempt %d failed: %s", attempt + 1, e)
                else:
                    # All 3 attempts failed — fall back to hard truncate at last sentence boundary
                    logger.warning("📰 🦋 LLM could not condense after 3 tries, truncating at sentence")
                    for i in range(290, 0, -1):
                        if bsky_text[i] in ".!?":
                            bsky_text = bsky_text[:i + 1]
                            break
                    else:
                        bsky_text = bsky_text[:295]

            # Ensure it ends with a complete sentence
            if bsky_text and bsky_text[-1] not in ".!?":
                for i in range(len(bsky_text) - 1, 0, -1):
                    if bsky_text[i] in ".!?":
                        bsky_text = bsky_text[:i + 1]
                        break

            if len(bsky_text) < 20:
                logger.info("📰 🦋 Commentary too short after trimming, skipping cross-post")
                return

            # Fetch og:image for thumbnail
            from soupy_bluesky import _fetch_og_image
            thumb_blob = None
            og_result = await _fetch_og_image(article_url)
            if og_result:
                image_bytes, mime_type = og_result
                thumb_blob = await bsky.upload_blob(image_bytes, mime_type)

            # Post to Bluesky with link card
            result = await bsky.create_post(
                bsky_text,
                link_url=article_url,
                link_title=title or "",
                link_description="",
                thumb_blob=thumb_blob,
            )

            if result:
                from soupy_bluesky import _post_url
                post_url = _post_url(result.get("uri", ""))
                logger.info("📰 🦋 Cross-posted to Bluesky: %s", post_url)
                try:
                    await bsky_cog._report_to_musing_channel(
                        f"cross-posted to bluesky\n{post_url}"
                    )
                except Exception as e:
                    logger.debug("📰 Could not notify musing channel: %s", e)
            else:
                logger.info("📰 🦋 Cross-post to Bluesky failed")

        except Exception as e:
            logger.warning("📰 🦋 Cross-post error (non-fatal): %s", e)

    async def _run_pipeline_for_channel(
        self, guild_id: int, channel_id: str, topic_hint: str, attempt: int = 1
    ) -> Tuple[bool, str]:
        """
        Run the complete daily post pipeline for one channel.
        Returns (success, status_message).
        """
        try:
            import time as _time
            _pipeline_start = _time.monotonic()
            logger.info("📰 ╔══════════════════════════════════════════════════╗")
            logger.info("📰 ║  DAILY POST PIPELINE — channel %s (attempt %d)", channel_id, attempt)
            logger.info("📰 ║  Topic: %s", topic_hint[:60])
            logger.info("📰 ╚══════════════════════════════════════════════════╝")

            # Step 1: Audience analysis
            audience_brief = await self._build_audience_brief(guild_id, channel_id, topic_hint)

            # Step 2: Generate search queries + fetch Bluesky trending
            # Later attempts use higher temperature and a broader nudge
            query_temp = 0.5 if attempt == 1 else 0.7
            query_extra = ""
            if attempt >= 2:
                query_extra = (
                    "\nPrevious search attempts found nothing. Use BROADER, more general queries "
                    "this time. Try queries like 'tech news today', 'cybersecurity news this week', "
                    "or target specific credible outlets like 'ars technica', 'bleepingcomputer', 'reuters tech'."
                )
            queries = await self._generate_search_queries(audience_brief + query_extra, temperature=query_temp)
            if not queries:
                return False, "LLM failed to generate search queries"

            # Step 2b: Bluesky trending topics → filter for channel → extra queries
            trending = await self._bluesky_trending()
            relevant_trending = await self._filter_trending_for_channel(trending, audience_brief)
            trending_queries = [t for t in relevant_trending[:2]]  # Use top 2 relevant trends as queries

            # Step 3: Search (DuckDuckGo news + Bluesky posts)
            articles = await self._search_articles(queries)
            # Bluesky: search with original queries + trending topics
            bsky_queries = queries + trending_queries
            bsky_posts = await self._bluesky_search(bsky_queries)
            # Merge: news articles first, then Bluesky posts
            all_candidates = articles + bsky_posts
            if not all_candidates:
                return False, "No articles or posts found from search"

            # Step 4: Rate all candidates together
            top_indices = await self._rate_articles(all_candidates, audience_brief)
            if not top_indices:
                return False, "No articles scored high enough"

            top_articles = [all_candidates[i] for i in top_indices]

            # Step 5: Pick best article and generate commentary
            result = await self._pick_and_comment(top_articles, audience_brief, channel_id, guild_id=guild_id)
            if result is None:
                return False, "LLM declined to pick an article (SKIP or all already posted)"

            commentary, url, title = result

            # Step 6: Send to channel
            channel = self.bot.get_channel(int(channel_id))
            if channel is None:
                try:
                    channel = await self.bot.fetch_channel(int(channel_id))
                except Exception:
                    return False, f"Could not fetch channel {channel_id}"

            sent_msg = await channel.send(f"{commentary}\n{url}")

            # Step 6a: Notify musing channel and tag the owner
            try:
                bsky_cog = self.bot.get_cog("BlueskyEngageCog")
                if bsky_cog is not None:
                    msg_link = getattr(sent_msg, "jump_url", "") or ""
                    await bsky_cog._report_to_musing_channel(
                        f"posted to #{channel.name}\n{msg_link}".strip()
                    )
            except Exception as e:
                logger.debug("📰 Could not notify musing channel: %s", e)

            # Step 6b: Cross-post to Bluesky
            await self._crosspost_to_bluesky(commentary, url, title)

            # Step 7: Record in history
            today_str = datetime.now(self.timezone).strftime("%Y-%m-%d")
            if channel_id not in self.history:
                self.history[channel_id] = []
            self.history[channel_id].append({
                "url": url,
                "title": title,
                "date": today_str,
            })
            _save_history(self.history)

            # Update timer state
            ts = getattr(self.bot, "_timer_state", None)
            if ts and "daily_post" in ts:
                ts["daily_post"]["last_run"] = datetime.now(self.timezone).astimezone(pytz.UTC).isoformat()

            _elapsed = _time.monotonic() - _pipeline_start
            logger.info("📰 ╔══════════════════════════════════════════════════╗")
            logger.info("📰 ║  ✅ POSTED to #%s in %.1fs", channel.name if channel else channel_id, _elapsed)
            logger.info("📰 ║  %s", title[:60])
            logger.info("📰 ║  %s", url[:60])
            logger.info("📰 ╚══════════════════════════════════════════════════╝")
            return True, f"{title} \u2014 {url}"

        except Exception as e:
            logger.error("📰 ❌ Pipeline error for channel %s: %s", channel_id, e, exc_info=True)
            return False, f"Pipeline error: {e}"

    # ------------------------------------------------------------------
    # Slash command
    # ------------------------------------------------------------------

    @app_commands.command(
        name="soupypost",
        description="Force Soupy to find and post an interesting article now",
    )
    @app_commands.describe(channel="Channel to post to (defaults to current)")
    async def soupypost(
        self,
        interaction: discord.Interaction,
        channel: Optional[discord.TextChannel] = None,
    ) -> None:
        # Permission check
        if interaction.user.id not in self._owner_ids():
            await interaction.response.send_message(
                "You don't have permission to use this command.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        target = channel or interaction.channel
        ch_id = str(target.id)

        # Look up topic hint from config, fall back to channel name
        channels_cfg = self._get_channels()
        topic_hint = channels_cfg.get(ch_id, target.name)

        guild_id = target.guild.id
        logger.info(f"\U0001f4f0 Manual /soupypost triggered by {interaction.user} for #{target.name}")

        max_retries = 3
        posted = False
        final_msg = ""

        for attempt in range(1, max_retries + 1):
            logger.info("📰 /soupypost attempt %d/%d for #%s", attempt, max_retries, target.name)
            success, msg = await self._run_pipeline_for_channel(guild_id, ch_id, topic_hint, attempt=attempt)
            if success:
                await interaction.followup.send(f"Posted to #{target.name}: {msg}", ephemeral=True)
                posted = True
                break
            else:
                final_msg = msg
                logger.info("⏭ /soupypost attempt %d failed for #%s: %s", attempt, target.name, msg)

        if not posted:
            await interaction.followup.send(
                f"Failed after {max_retries} attempts for #{target.name}. Last: {final_msg}", ephemeral=True
            )


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(DailyPostCog(bot))
