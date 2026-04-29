"""
Bluesky engagement cog for Soupy Bot.

Owner-triggered command that finds interesting Bluesky posts, reads the
comment thread, and leaves a relevant reply in Soupy's voice.  Can also
like good comments and follow interesting accounts (max 1-2/day).

Posts a link to the comment in the musing channel on Discord.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re as _re

import aiohttp
import discord
import trafilatura
from ddgs import DDGS
from discord import app_commands
from discord.ext import commands, tasks
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
)


async def _llm_call(
    system: str, user: str, temperature: float = 0.7, max_tokens: int = 300,
    timeout_seconds: int = 300,
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

    try:
        response = await asyncio.wait_for(asyncio.to_thread(_sync), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error("🦋 LLM call timed out after %ds", timeout_seconds)
        return ""
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Vision: describe images via LM Studio vision model
# ---------------------------------------------------------------------------


async def _describe_image(image_url: str, prompt: str = "Describe this image concisely in 1-2 sentences.") -> Optional[str]:
    """Download an image and describe it using the LM Studio vision model.

    Returns a short description string, or None if vision is disabled or fails.
    """
    if os.getenv("ENABLE_VISION", "false").lower() != "true":
        return None

    try:
        import base64
        from io import BytesIO
        from PIL import Image

        # Download the image
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                image_data = await resp.read()

        if len(image_data) < 500 or len(image_data) > 10_000_000:
            return None

        # Transcode to JPEG for compatibility
        def _transcode():
            with Image.open(BytesIO(image_data)) as im:
                if im.mode in ("P", "RGBA"):
                    im = im.convert("RGB")
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=85, optimize=True)
                return buf.getvalue()

        jpeg_bytes = await asyncio.to_thread(_transcode)
        encoded = base64.b64encode(jpeg_bytes).decode("utf-8")

        # Call LM Studio vision endpoint
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        endpoint = f"{base}/chat/completions"

        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        model_name = os.getenv("VISION_MODEL") or os.getenv("LOCAL_CHAT")
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": int(os.getenv("VISION_MAX_TOKENS", "200")),
            "temperature": 0.3,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    # Try without data URI prefix (some backends want raw base64)
                    payload["messages"][0]["content"][0]["image_url"]["url"] = encoded
                    async with session.post(endpoint, json=payload, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=30)) as resp2:
                        if resp2.status != 200:
                            logger.debug("🦋 Vision API failed: %d", resp2.status)
                            return None
                        data = await resp2.json()
                else:
                    data = await resp.json()

        description = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if description:
            logger.info("🦋 👁 Image description: %s", description[:100])
        return description or None

    except Exception as e:
        logger.debug("🦋 Vision failed for %s: %s", image_url[:60], e)
        return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PICK_POST_SYSTEM = (
    "You are choosing which Bluesky post to engage with.\n"
    "Pick the post where you can find the sharpest angle — the failure, the hypocrisy, "
    "the irony, the thing nobody else is pointing out.\n"
    "Any political story has a failure angle if you look hard enough.\n\n"
    "You MUST pick one. Return ONLY the index number (0-indexed). Nothing else."
)

COMMENT_SYSTEM = (
    "you are soupy dafoe. you are replying to a bluesky post.\n\n"
    "you are NOT a commenter bot. you are a person who saw something and has a reaction.\n\n"
    "DO NOT restate or summarize what the post says. instead, do ONE of these:\n"
    "- say why it's worse than people think, or what they're missing\n"
    "- make a wry observation, joke, or pointed question about it\n"
    "- name who benefits and who gets screwed\n"
    "- connect it to a bigger pattern of power, money, or hypocrisy\n"
    "- express genuine anger, disbelief, or dark humor\n\n"
    "YOUR PERSPECTIVE:\n"
    "you are skeptical of power, pro-worker, and politically sharp.\n"
    "you see through the spin. you question who profits and who pays.\n"
    "you punch UP at politicians, corporations, billionaires, never down.\n"
    "dry skepticism and pointed questions land harder than lectures.\n"
    "NEVER use labels like 'socialist', 'marxist', 'leftist', 'capitalist'.\n\n"
    "STYLE:\n"
    "- 1-2 sentences, 280 chars max. lower case, no quotation marks.\n"
    "- be concise and decisive. no hedging ('maybe', 'might', 'seems').\n"
    "- do not address the author or readers. no second-person.\n"
    "- match tone: critical posts get pointed replies. light posts stay light.\n\n"
    "HARD RULES:\n"
    "- your reply MUST name a specific detail from the post: a person, place, event, or claim.\n"
    "- do NOT bring up unrelated topics, people, or events.\n"
    "- do NOT invent numbers, statistics, or quotes not in the post or context.\n"
    "- do NOT include meta commentary, labels, headers, or reasoning. output ONLY the reply.\n"
    "- do NOT mention being a bot or AI. you are a person.\n"
    "- do NOT use hashtags, @handles, or em dashes (—). NEVER use em dashes.\n"
    "- FINISH YOUR THOUGHT. do not start a sentence you cannot complete.\n"
    "- if a CONTEXT block is provided, use it to understand the real situation. "
    "reply like someone who already knows what's going on. NEVER reference searching, "
    "articles, URLs, or say 'according to'.\n"
    "- CRITICAL: your reply must be about what the ORIGINAL POST says. the CONTEXT helps you "
    "understand the topic, but do NOT merge separate news stories together. if the post is "
    "about the Fed chair and the context mentions a DOJ investigation, those are different "
    "stories. reply to what the post actually says.\n"
)


JUDGE_FOLLOW_SYSTEM = (
    "You are deciding whether to follow this Bluesky user.\n"
    "Follow people who are politically engaged, critical of power, pro-worker, "
    "funny, or share good news/analysis. Prefer journalists, activists, "
    "commentators, and sharp political voices.\n\n"
    "Return YES if they're worth following, NO if not. One word only."
)

QUOTE_POST_SYSTEM = (
    "you are soupy dafoe. you are quote-posting someone else's bluesky post.\n\n"
    "YOUR VOICE: critical of power, skeptical of authority, pro-worker, anti-war, sharp and funny.\n"
    "find the failure angle, the hypocrisy, the irony. question who profits and who pays.\n"
    "subtly critique power structures and inequality. show the perspective naturally.\n"
    "NEVER use labels like 'socialist', 'marxist', 'leftist'. let the take speak for itself.\n\n"
    "STYLE:\n"
    "- be concise and decisive. one crisp line.\n"
    "- avoid hedging — no 'maybe', 'might', 'seems', 'perhaps'.\n"
    "- do not address readers — no second-person.\n\n"
    "RULES:\n"
    "- add your take on WHY this matters — the systemic issue, the hypocrisy, the irony.\n"
    "- anchor to a concrete detail from the post.\n"
    "- do NOT just summarize. add a new angle or observation.\n"
    "- do NOT invent numbers or statistics not in the post.\n"
    "- do NOT mention being a bot or AI.\n"
    "- do NOT use hashtags, @handles, or the '-' hyphen character.\n"
    "- write in lower case, no quotation marks of any kind.\n"
    "- 1-2 sentences, 200 characters max.\n"
    "- FINISH YOUR THOUGHT.\n"
    "- the quoted post will appear below yours.\n"
    "- if a CONTEXT block is provided, use it to understand the real story. "
    "your commentary should show you get what's happening. but NEVER reference "
    "articles, searching, URLs, or say 'according to'. just comment like someone who knows.\n"
)

ORIGINAL_POST_SYSTEM = (
    "you are soupy dafoe sharing something you found on bluesky.\n\n"
    "you are NOT a news bot. you are a person who saw something and wants to share it because "
    "it is interesting, weird, important, or you have a strong reaction to it.\n\n"
    "DO NOT summarize the article. instead, do ONE of these:\n"
    "- say why it surprised you or caught your eye\n"
    "- make a wry observation or joke about it\n"
    "- take a side or express an opinion about what happened\n"
    "- name the specific thing that's wrong, ironic, or absurd\n\n"
    "YOUR PERSPECTIVE:\n"
    "large corporations are not your friends. when they do something that looks generous or "
    "responsible, ask what's in it for them. follow the money.\n"
    "be skeptical of tech companies, billionaires, and anyone with concentrated power.\n"
    "your worldview: capital serves the owning class, war serves the arms industry, "
    "workers create the value but rarely see it. question who profits and who pays.\n"
    "anti-war: question military spending, defense contracts, and interventionism.\n"
    "you don't need to be angry or preachy. dry skepticism and pointed questions land harder "
    "than lectures. a single 'interesting timing' or 'wonder who's paying for that' says more "
    "than a paragraph of outrage.\n"
    "NEVER use labels like 'socialist', 'marxist', 'leftist', 'capitalist'. "
    "just show the perspective naturally through your observations.\n\n"
    "STYLE:\n"
    "- be concise and decisive. crisp, coherent sentences.\n"
    "- avoid hedging — no 'maybe', 'might', 'seems', 'perhaps'.\n"
    "- do not address readers — no second-person ('you', 'your').\n\n"
    "HARD RULES:\n"
    "- do NOT invent numbers, dollar amounts, or statistics not in the article.\n"
    "- do NOT include ANY meta commentary, labels, headers, or reasoning.\n"
    "  output ONLY the post text itself, nothing else.\n"
    "- do NOT mention being a bot or AI. you are a person.\n"
    "- do NOT use hashtags or @handles. do NOT use the '-' hyphen character or em dash (—). NEVER use em dashes.\n"
    "- write in lower case, no quotation marks of any kind.\n"
    "- MAXIMUM 1-2 SHORT sentences. aim for under 200 characters.\n"
    "- FINISH YOUR THOUGHT. do not start a sentence you cannot complete.\n"
    "- the link will be attached as a card, so do NOT include the URL in your text.\n"
)

FACT_CHECK_SYSTEM = (
    "You are checking whether a proposed Bluesky post is factually grounded "
    "in the article it's about. The post is COMMENTARY, not a summary — "
    "opinion, framing, sarcasm, irony, and editorial takes are EXPECTED and FINE.\n\n"
    "Your ONLY job is to catch factual errors. Specifically:\n\n"
    "FAIL the post (INACCURATE) if it:\n"
    "- Invents numbers, dollar amounts, dates, names, or quotes not in the article\n"
    "- Gets who-did-what wrong (wrong actor, wrong target, swapped roles)\n"
    "- States an event or action that the article does not describe\n"
    "- Directly contradicts something the article says\n"
    "- Attributes a position or statement to someone the article doesn't attribute it to\n\n"
    "PASS the post (ACCURATE) if it:\n"
    "- Offers an opinion, judgment, or emotional reaction to what the article describes\n"
    "- Frames the story through a political, economic, or moral lens\n"
    "- Uses sarcasm, irony, or rhetorical exaggeration to make a point\n"
    "- Connects the article to a broader pattern, system, or power dynamic\n"
    "- Draws a reasonable inference from facts that ARE in the article\n"
    "- Is critical, snarky, or one-sided — bias is allowed, invention is not\n\n"
    "The test is simple: could a reader of the article say 'yes, that's a take on "
    "what I just read' — even if they disagree with the take? If yes, ACCURATE.\n"
    "Only fail it if the post would make a reader say 'wait, the article didn't say that.'\n\n"
    "Return ACCURATE or INACCURATE followed by a brief reason."
)

SEARCH_QUERY_FOR_POST_SYSTEM = (
    "Generate 3 search queries to find interesting recent news.\n\n"
    "FOCUS AREAS (pick from these):\n"
    "- US politics: Trump administration, Congress, Supreme Court, legislation, elections\n"
    "- Technology: AI developments, big tech, data privacy, social media, automation\n"
    "- Finance/economy: Wall Street, billionaires, labor, inequality, corporate behavior\n"
    "- Government/power: corruption, investigations, policy changes, executive orders\n"
    "- International: geopolitics, trade, sanctions, conflicts\n\n"
    "RULES:\n"
    "- You want BREAKING NEWS, not listicles, forums, or opinion blogs.\n"
    "- Each query should target a SPECIFIC recent event, not a broad topic.\n"
    "- Use words like 'breaking', 'announced', 'report', 'revealed', 'investigation'.\n"
    "- The current year is 2026. Include '2026' in at least one query.\n"
    "- Do NOT search for random crime stories, celebrity gossip, or sports.\n"
    "- Do NOT use vague terms like 'controversies', 'breakthroughs this week'.\n\n"
    "3-6 words each, one per line, nothing else."
)


# ---------------------------------------------------------------------------
# Article search helpers (reused from dailypost patterns)
# ---------------------------------------------------------------------------


async def _ddg_search(query: str, timelimit: str = "d", max_results: int = 8) -> List[Dict]:
    """DuckDuckGo NEWS search. Falls back to text search if news returns nothing."""
    def _sync():
        with DDGS() as ddg:
            raw = list(ddg.news(query, timelimit=timelimit, max_results=max_results))
            for r in raw:
                if "url" in r and "href" not in r:
                    r["href"] = r["url"]
                if r.get("date"):
                    r["pub_date"] = r["date"]
            return raw
    try:
        results = await asyncio.wait_for(asyncio.to_thread(_sync), timeout=15)
        if results:
            return results
        def _sync_text():
            with DDGS() as ddg:
                return list(ddg.text(query, timelimit=timelimit, max_results=max_results))
        return await asyncio.wait_for(asyncio.to_thread(_sync_text), timeout=15)
    except (asyncio.TimeoutError, Exception):
        return []


async def _fetch_article(url: str) -> Optional[Dict[str, str]]:
    """Fetch article content via trafilatura.

    Falls back to HTML meta tag extraction if trafilatura doesn't find a date.
    """
    def _sync():
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False) or ""
        meta = trafilatura.extract(downloaded, output_format="json", include_comments=False)
        date = None
        title = ""
        if meta:
            try:
                import json as _json
                m = _json.loads(meta)
                date = m.get("date")
                title = m.get("title", "")
            except Exception:
                pass
        # Fallback: extract date from HTML meta tags / JSON-LD
        if not date:
            from soupy_dailypost import _extract_date_from_html
            date = _extract_date_from_html(downloaded)
        return {"content": text, "date": date, "title": title}
    try:
        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=20)
    except (asyncio.TimeoutError, Exception):
        return None


# ---------------------------------------------------------------------------
# OG image extraction
# ---------------------------------------------------------------------------


# Bluesky's blob limit is ~976 KB. Stay under 950 KB to leave headroom.
_BSKY_BLOB_MAX_BYTES = 950_000

# User-Agent fallback chain. Many paywalled news sites (NYT, Reuters, WSJ)
# block generic browser UAs but allow well-known social-media link-preview
# crawlers, since they want their content shareable on Twitter/Facebook.
_OG_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (compatible; Twitterbot/1.0)",
    "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
    "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
)


def _resize_image_for_blob(image_bytes: bytes) -> Optional[Tuple[bytes, str]]:
    """Pillow-resize/recompress an oversized image to fit Bluesky's blob limit.

    Returns (jpeg_bytes, "image/jpeg") if the image can be made small enough,
    otherwise None. Always re-encodes to JPEG for maximum compatibility.
    """
    try:
        from io import BytesIO
        from PIL import Image
    except Exception as e:
        logger.warning("🦋 Pillow unavailable, can't resize og:image: %s", e)
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            # Animated formats: take first frame.
            if getattr(im, "is_animated", False):
                im.seek(0)
            # Bluesky link-card recommended ratio is roughly 1.91:1 — 1200x630.
            # Use thumbnail() so we keep aspect ratio and don't upscale.
            im = im.convert("RGB")
            im.thumbnail((1200, 1200), Image.LANCZOS)
            for quality in (85, 75, 65, 55):
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=quality, optimize=True)
                data = buf.getvalue()
                if len(data) <= _BSKY_BLOB_MAX_BYTES:
                    logger.info("🦋 Resized og:image to %d bytes (q=%d, %dx%d)",
                                len(data), quality, im.width, im.height)
                    return data, "image/jpeg"
            logger.info("🦋 og:image still too large after recompression; giving up")
            return None
    except Exception as e:
        logger.info("🦋 og:image resize failed: %s", e)
        return None


def _resolve_image_url(image_url: str, page_url: str) -> Optional[str]:
    """Resolve relative / protocol-relative og:image URLs to absolute URLs."""
    if not image_url:
        return None
    image_url = image_url.strip()
    if not image_url:
        return None
    if image_url.startswith("http://") or image_url.startswith("https://"):
        return image_url
    if image_url.startswith("//"):
        # Protocol-relative URL
        scheme = "https" if page_url.startswith("https") else "http"
        return f"{scheme}:{image_url}"
    if image_url.startswith("/"):
        # Absolute path on same host
        try:
            from urllib.parse import urlparse
            p = urlparse(page_url)
            if p.scheme and p.netloc:
                return f"{p.scheme}://{p.netloc}{image_url}"
        except Exception:
            return None
    return None


async def _fetch_page_html(url: str) -> Tuple[Optional[str], str]:
    """Fetch a page's HTML, retrying with crawler User-Agents on 4xx blocks.

    Returns (html_or_none, resolved_url). resolved_url falls back to the input
    URL if no fetch succeeds.
    """
    resolved_url = url
    last_status: Optional[int] = None
    for ua in _OG_USER_AGENTS:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    url,
                    headers={"User-Agent": ua, "Accept": "text/html,application/xhtml+xml"},
                    timeout=aiohttp.ClientTimeout(total=15),
                    allow_redirects=True,
                ) as r:
                    if r.status == 200:
                        resolved_url = str(r.url)
                        raw = await r.read()
                        return raw.decode("utf-8", errors="replace"), resolved_url
                    last_status = r.status
        except Exception as e:
            logger.debug("🦋 og:image fetch with UA %r failed: %s", ua.split("/")[0], e)
    if last_status:
        logger.info("🦋 og:image: every UA blocked (last HTTP %d), trying trafilatura for %s",
                    last_status, url[:60])
    # Last-resort: trafilatura.
    def _traf_fetch():
        return trafilatura.fetch_url(url)
    try:
        html = await asyncio.wait_for(asyncio.to_thread(_traf_fetch), timeout=15)
        if html:
            return html, resolved_url
    except (asyncio.TimeoutError, Exception):
        pass
    return None, resolved_url


async def _fetch_og_image(url: str) -> Optional[Tuple[bytes, str]]:
    """Fetch the og:image from a URL. Returns (image_bytes, mime_type) or None.

    Robust against paywall blocks (tries multiple User-Agents including
    Twitterbot / Facebookexternalhit), resolves relative og:image URLs, and
    auto-resizes oversized images via Pillow to fit Bluesky's blob limit.
    """
    try:
        # Step 1: Fetch page HTML, retrying with social-crawler UAs on 4xx.
        html, resolved_url = await _fetch_page_html(url)
        if not html:
            logger.info("🦋 og:image: could not fetch page at all for %s", url[:60])
            return None

        # Step 2: Parse image URL from meta tags (try multiple patterns + JSON-LD).
        import re
        image_url = None

        meta_patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
            r'<meta[^>]+name=["\']twitter:image:src["\'][^>]+content=["\']([^"\']+)["\']',
            # link rel=image_src as a final fallback
            r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
        ]
        for pattern in meta_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                candidate = _resolve_image_url(match.group(1), resolved_url)
                if candidate and len(candidate) > 10:
                    image_url = candidate
                    break

        if not image_url:
            logger.info("🦋 No og:image or twitter:image found for %s", resolved_url[:60])
            return None

        logger.info("🦋 Found og:image: %s", image_url[:100])

        # Step 3: Download the image, retrying with crawler UAs if blocked.
        image_bytes: Optional[bytes] = None
        content_type = "image/jpeg"
        for ua in _OG_USER_AGENTS:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        image_url,
                        headers={"User-Agent": ua, "Referer": resolved_url},
                        timeout=aiohttp.ClientTimeout(total=15),
                        allow_redirects=True,
                    ) as r:
                        if r.status == 200:
                            content_type = r.headers.get("Content-Type", "image/jpeg")
                            image_bytes = await r.read()
                            break
                        logger.debug("🦋 og:image download with UA %r got HTTP %d",
                                     ua.split("/")[0], r.status)
            except Exception as e:
                logger.debug("🦋 og:image download with UA %r failed: %s", ua.split("/")[0], e)

        if not image_bytes:
            logger.info("🦋 og:image download failed (all UAs) for %s", image_url[:80])
            return None

        if len(image_bytes) < 1000:
            logger.info("🦋 og:image too small (%d bytes), probably a tracking pixel", len(image_bytes))
            return None

        # Pick canonical mime; treat anything non-PNG/non-WebP as JPEG-class.
        if "png" in content_type:
            mime = "image/png"
        elif "webp" in content_type:
            mime = "image/webp"
        else:
            mime = "image/jpeg"

        # If oversized for Bluesky's blob limit, resize via Pillow.
        if len(image_bytes) > _BSKY_BLOB_MAX_BYTES:
            logger.info("🦋 og:image is %d bytes, resizing to fit Bluesky blob limit", len(image_bytes))
            resized = _resize_image_for_blob(image_bytes)
            if resized is None:
                return None
            return resized

        # WebP uploads to Bluesky are flaky; normalize to JPEG defensively.
        if mime == "image/webp":
            resized = _resize_image_for_blob(image_bytes)
            if resized is not None:
                return resized
            # If Pillow can't open it, fall through and try the raw bytes.

        return image_bytes, mime

    except Exception as e:
        logger.debug("🦋 og:image fetch failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# History / rate-limit tracking
# ---------------------------------------------------------------------------

HISTORY_PATH = os.path.join("data", "bluesky_engage_history.json")
SCHEDULE_PATH = os.path.join("data", "bluesky_schedule.json")
ENV_PATH = ".env-stable"


def _read_env_value(key: str, default: str = "") -> str:
    """Read a value directly from .env-stable on every call.

    Used for runtime-toggleable settings so dashboard toggles take effect
    without requiring a bot restart.
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


def _load_history() -> Dict[str, Any]:
    if os.path.exists(HISTORY_PATH):
        try:
            return json.loads(Path(HISTORY_PATH).read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"comments": [], "follows_today": [], "likes_today": [], "last_date": ""}


def _save_history(h: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    Path(HISTORY_PATH).write_text(
        json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _reset_daily_if_needed(h: Dict[str, Any]) -> None:
    """Reset daily counters if the date has changed."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if h.get("last_date") != today:
        h["follows_today"] = []
        h["likes_today"] = []
        h["last_date"] = today


# ---------------------------------------------------------------------------
# Bluesky API helpers
# ---------------------------------------------------------------------------

BSKY_BASE = "https://bsky.social/xrpc"


class BlueskyClient:
    """Thin async wrapper around AT Protocol endpoints."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._did: Optional[str] = None
        self._expiry: Optional[datetime] = None

    async def auth(self) -> Optional[str]:
        handle = os.getenv("BLUESKY_HANDLE", "")
        app_pw = os.getenv("BLUESKY_APP_PASSWORD", "")
        if not handle or not app_pw:
            return None

        now = datetime.now(timezone.utc)
        if self._token and self._expiry and now < self._expiry:
            return self._token

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{BSKY_BASE}/com.atproto.server.createSession",
                    json={"identifier": handle, "password": app_pw},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    if r.status != 200:
                        logger.error("🦋 Auth failed: HTTP %d", r.status)
                        return None
                    data = await r.json()
            self._token = data.get("accessJwt")
            self._did = data.get("did")
            self._expiry = now + timedelta(minutes=90)
            logger.info("🦋 Authenticated as %s", handle)
            return self._token
        except Exception as e:
            logger.error("🦋 Auth error: %s", e)
            return None

    @property
    def did(self) -> Optional[str]:
        return self._did

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def get_timeline(self, limit: int = 30) -> List[Dict]:
        """Get posts from accounts we follow."""
        token = await self.auth()
        if not token:
            return []
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{BSKY_BASE}/app.bsky.feed.getTimeline",
                params={"limit": str(limit)},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    return []
                data = await r.json()
        return [item.get("post", {}) for item in data.get("feed", [])]

    async def search_posts(
        self, query: str, sort: str = "top", limit: int = 10
    ) -> List[Dict]:
        token = await self.auth()
        if not token:
            return []
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{BSKY_BASE}/app.bsky.feed.searchPosts",
                params={"q": query, "sort": sort, "limit": str(limit)},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    return []
                data = await r.json()
        return data.get("posts", [])

    async def get_trending(self) -> List[str]:
        token = await self.auth()
        if not token:
            return []
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{BSKY_BASE}/app.bsky.unspecced.getTrendingTopics",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status != 200:
                    return []
                data = await r.json()
        return [t["topic"] for t in data.get("topics", []) if t.get("topic")]

    async def get_thread(self, uri: str, depth: int = 6) -> Optional[Dict]:
        token = await self.auth()
        if not token:
            return None
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{BSKY_BASE}/app.bsky.feed.getPostThread",
                params={"uri": uri, "depth": str(depth), "parentHeight": "0"},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
        return data.get("thread")

    async def get_author_feed(self, did: str, limit: int = 10) -> List[Dict]:
        """Get recent posts by a specific author."""
        token = await self.auth()
        if not token:
            return []
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{BSKY_BASE}/app.bsky.feed.getAuthorFeed",
                params={"actor": did, "limit": str(limit)},
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    return []
                data = await r.json()
        return [item.get("post", {}) for item in data.get("feed", [])]

    async def create_reply(
        self, text: str, parent_uri: str, parent_cid: str,
        root_uri: str, root_cid: str
    ) -> Optional[Dict]:
        """Post a reply to a Bluesky post."""
        token = await self.auth()
        if not token:
            return None

        record = {
            "$type": "app.bsky.feed.post",
            "text": text,
            "reply": {
                "root": {"uri": root_uri, "cid": root_cid},
                "parent": {"uri": parent_uri, "cid": parent_cid},
            },
            "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }

        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BSKY_BASE}/com.atproto.repo.createRecord",
                json={
                    "repo": self._did,
                    "collection": "app.bsky.feed.post",
                    "record": record,
                },
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status not in (200, 201):
                    body = await r.text()
                    logger.error("🦋 Reply failed: HTTP %d — %s", r.status, body[:200])
                    return None
                return await r.json()

    async def like_post(self, uri: str, cid: str) -> bool:
        """Like a post."""
        token = await self.auth()
        if not token:
            return False

        record = {
            "$type": "app.bsky.feed.like",
            "subject": {"uri": uri, "cid": cid},
            "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }

        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BSKY_BASE}/com.atproto.repo.createRecord",
                json={
                    "repo": self._did,
                    "collection": "app.bsky.feed.like",
                    "record": record,
                },
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                return r.status in (200, 201)

    async def follow_user(self, target_did: str) -> bool:
        """Follow a user by DID."""
        token = await self.auth()
        if not token:
            return False

        record = {
            "$type": "app.bsky.graph.follow",
            "subject": target_did,
            "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }

        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BSKY_BASE}/com.atproto.repo.createRecord",
                json={
                    "repo": self._did,
                    "collection": "app.bsky.graph.follow",
                    "record": record,
                },
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                return r.status in (200, 201)


    async def quote_post(
        self, text: str, quote_uri: str, quote_cid: str
    ) -> Optional[Dict]:
        """Create a quote-post (repost with commentary)."""
        token = await self.auth()
        if not token:
            return None

        record = {
            "$type": "app.bsky.feed.post",
            "text": text,
            "embed": {
                "$type": "app.bsky.embed.record",
                "record": {"uri": quote_uri, "cid": quote_cid},
            },
            "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }

        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BSKY_BASE}/com.atproto.repo.createRecord",
                json={
                    "repo": self._did,
                    "collection": "app.bsky.feed.post",
                    "record": record,
                },
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status not in (200, 201):
                    body = await r.text()
                    logger.error("🦋 Quote-post failed: HTTP %d — %s", r.status, body[:200])
                    return None
                return await r.json()

    async def upload_blob(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Optional[Dict]:
        """Upload an image blob to Bluesky. Returns the blob ref dict or None."""
        token = await self.auth()
        if not token:
            return None

        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BSKY_BASE}/com.atproto.repo.uploadBlob",
                data=image_bytes,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": mime_type,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                if r.status not in (200, 201):
                    logger.debug("🦋 Blob upload failed: HTTP %d", r.status)
                    return None
                data = await r.json()
                return data.get("blob")

    async def create_post(
        self, text: str, link_url: Optional[str] = None,
        link_title: str = "", link_description: str = "",
        thumb_blob: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Create an original Bluesky post, optionally with a link card embed.

        If thumb_blob is provided (from upload_blob), the link card will have
        a preview image.
        """
        token = await self.auth()
        if not token:
            return None

        record: Dict[str, Any] = {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }

        # Add link card embed if URL provided
        if link_url:
            external: Dict[str, Any] = {
                "uri": link_url,
                "title": link_title or link_url,
                "description": link_description or "",
            }
            if thumb_blob:
                external["thumb"] = thumb_blob
            record["embed"] = {
                "$type": "app.bsky.embed.external",
                "external": external,
            }

        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{BSKY_BASE}/com.atproto.repo.createRecord",
                json={
                    "repo": self._did,
                    "collection": "app.bsky.feed.post",
                    "record": record,
                },
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status not in (200, 201):
                    body = await r.text()
                    logger.error("🦋 Post failed: HTTP %d — %s", r.status, body[:200])
                    return None
                return await r.json()


def _post_url(uri: str) -> str:
    """Convert at:// URI to https://bsky.app URL."""
    parts = uri.replace("at://", "").split("/")
    if len(parts) >= 3:
        return f"https://bsky.app/profile/{parts[0]}/post/{parts[-1]}"
    return ""


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class BlueskyEngageCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.bsky = BlueskyClient()
        self.history = _load_history()

        # Daily schedule: list of datetimes when we should auto-reply
        self._schedule: List[Tuple[datetime, str]] = []
        self._schedule_date: Optional[str] = None
        self._auto_loop.start()

        # Populate dashboard from persisted history on startup
        self._restore_dashboard_from_history()

    def cog_unload(self) -> None:
        self._auto_loop.cancel()

    def _restore_dashboard_from_history(self) -> None:
        """Populate dashboard timer state from persisted history so counters survive restarts."""
        # Find last reply, post, repost timestamps from history
        def _last_ts(entries: List[Dict]) -> Optional[str]:
            if not entries:
                return None
            return entries[-1].get("ts")

        last_reply = _last_ts(self.history.get("comments", []))
        last_post = _last_ts(self.history.get("posts", []))
        last_repost = _last_ts(self.history.get("reposts", []))

        # Find last_run from the most recent action of any type
        all_ts = [t for t in [last_reply, last_post, last_repost] if t]
        last_run = max(all_ts) if all_ts else None

        self._update_dashboard(
            enabled=self._is_auto_enabled(),
            last_run=last_run,
            last_reply=last_reply,
            last_post=last_post,
            last_repost=last_repost,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _owner_ids(self) -> set:
        raw = os.getenv("OWNER_IDS", "")
        try:
            return {int(x.strip()) for x in raw.split(",") if x.strip()}
        except Exception:
            return set()

    def _is_auto_enabled(self) -> bool:
        # Re-read from .env-stable so dashboard toggles take effect without restart
        return _read_env_value("BLUESKY_AUTO_REPLY", "false").lower() in ("true", "1", "yes")

    def _update_dashboard(self, **kwargs: Any) -> None:
        """Update the bluesky timer state for the web dashboard."""
        ts = getattr(self.bot, "_timer_state", None)
        if not ts or "bluesky" not in ts:
            return
        bsky = ts["bluesky"]
        for k, v in kwargs.items():
            bsky[k] = v
        # Always refresh counts from history
        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bsky["replies_today"] = sum(
            1 for c in self.history.get("comments", [])
            if c.get("ts", "").startswith(today_iso)
        )
        bsky["posts_today"] = sum(
            1 for p in self.history.get("posts", [])
            if p.get("ts", "").startswith(today_iso)
        )
        bsky["reposts_today"] = sum(
            1 for r in self.history.get("reposts", [])
            if r.get("ts", "").startswith(today_iso)
        )

    def _musing_channel_id(self) -> Optional[int]:
        raw = os.getenv("MUSING_CHANNEL_ID", "")
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Step 1: Discover interesting posts
    # ------------------------------------------------------------------

    async def _discover_posts(self) -> List[Dict]:
        """Gather candidate posts from trending, timeline, search, and thread exploration.

        Mixes big and small accounts — ~30% of the time we intentionally pick
        a smaller account to reply to, which drives profile visits and follows.
        """
        big_candidates: List[Dict] = []    # 20+ likes (popular posts)
        small_candidates: List[Dict] = []  # 3-19 likes (smaller accounts, discoverable)
        seen_uris: set = set()

        def _add(post: Dict) -> None:
            uri = post.get("uri", "")
            if uri in seen_uris:
                return
            seen_uris.add(uri)
            likes = post.get("likeCount", 0) or 0
            replies = post.get("replyCount", 0) or 0
            if likes < 3 or replies < 1:
                return
            if likes >= 20:
                big_candidates.append(post)
            else:
                small_candidates.append(post)

        # Source 1: Timeline (posts from followed accounts)
        logger.info("🦋 ━━━ Discovering posts ━━━")
        timeline = await self.bsky.get_timeline(limit=40)
        for p in timeline:
            _add(p)
        logger.info("🦋   Timeline: %d posts → %d big, %d small",
                     len(timeline), len(big_candidates), len(small_candidates))

        prev_big, prev_small = len(big_candidates), len(small_candidates)

        # Source 2: Trending topics — search both "top" and "latest"
        trending = await self.bsky.get_trending()
        logger.info("🦋   Trending: %s", ", ".join(trending[:5]))
        for topic in trending[:3]:
            # Top results (big accounts)
            posts = await self.bsky.search_posts(topic, sort="top", limit=5)
            for p in posts:
                _add(p)
            await asyncio.sleep(0.5)
            # Latest results (smaller, fresher accounts)
            posts = await self.bsky.search_posts(topic, sort="latest", limit=8)
            for p in posts:
                _add(p)
            await asyncio.sleep(0.5)
        logger.info("🦋   Trending added %d big, %d small",
                     len(big_candidates) - prev_big, len(small_candidates) - prev_small)

        prev_big, prev_small = len(big_candidates), len(small_candidates)

        # Source 3: Thread exploration — pick a popular post's thread and find
        # interesting reply authors, then check THEIR recent posts.
        # This finds smaller accounts that are active in good conversations.
        if big_candidates:
            explore_post = random.choice(big_candidates[:5])
            thread = await self.bsky.get_thread(explore_post.get("uri", ""), depth=3)
            if thread:
                reply_authors = []
                for r in thread.get("replies", [])[:20]:
                    rp = r.get("post", {})
                    ra = rp.get("author", {})
                    ra_did = ra.get("did", "")
                    ra_likes = rp.get("likeCount", 0) or 0
                    if ra_did and ra_did != self.bsky.did and ra_likes >= 2:
                        reply_authors.append(ra_did)

                # Sample up to 2 reply authors and check their feeds
                for author_did in random.sample(reply_authors, min(2, len(reply_authors))):
                    author_posts = await self.bsky.get_author_feed(author_did, limit=5)
                    for p in author_posts:
                        _add(p)
                    await asyncio.sleep(0.5)

                logger.info("🦋   Thread exploration added %d big, %d small",
                             len(big_candidates) - prev_big, len(small_candidates) - prev_small)

        # --- Apply filters to both pools ---
        our_did = self.bsky.did
        commented_uris = {c.get("post_uri") for c in self.history.get("comments", [])}

        # Author recency block
        recent_comments = self.history.get("comments", [])
        recent_author_dids = []
        for c in reversed(recent_comments):
            puri = c.get("post_uri", "")
            if puri.startswith("at://"):
                did = puri.split("/")[2] if len(puri.split("/")) > 2 else ""
            else:
                did = ""
            if did:
                recent_author_dids.append(did)
        blocked_dids = set(recent_author_dids[:5])
        penalized_dids = set(recent_author_dids[5:15])

        def _filter(pool: List[Dict]) -> List[Dict]:
            return [
                c for c in pool
                if c.get("author", {}).get("did") != our_did
                and c.get("uri") not in commented_uris
                and c.get("author", {}).get("did") not in blocked_dids
            ]

        big_candidates = _filter(big_candidates)
        small_candidates = _filter(small_candidates)

        # Sort each pool by engagement (with recency penalty)
        def _sort_key(p: Dict) -> float:
            score = float((p.get("likeCount", 0) or 0) + (p.get("repostCount", 0) or 0))
            if p.get("author", {}).get("did") in penalized_dids:
                score *= 0.3
            return score

        big_candidates.sort(key=_sort_key, reverse=True)
        small_candidates.sort(key=_sort_key, reverse=True)

        # Decide which pool to draw from: ~30% chance of picking small accounts.
        # This drives profile visits from people who are more likely to follow back.
        use_small = (
            small_candidates
            and (not big_candidates or random.random() < 0.30)
        )

        if use_small:
            candidates = small_candidates[:10] + big_candidates[:5]
            logger.info("🦋   🎯 Prioritizing smaller accounts this round")
        else:
            candidates = big_candidates[:10] + small_candidates[:5]

        logger.info("🦋   Total candidates: %d big, %d small → %d selected",
                     len(big_candidates), len(small_candidates), len(candidates))
        for i, c in enumerate(candidates[:8]):
            author = c.get("author", {}).get("displayName", "?")
            text = c.get("record", {}).get("text", "")[:80]
            likes = c.get("likeCount", 0)
            replies = c.get("replyCount", 0)
            logger.info("🦋   [%d] %d♥ %d💬 — %s: %s", i, likes, replies, author, text)

        return candidates[:15]

    # ------------------------------------------------------------------
    # Step 2: Pick the best post to engage with
    # ------------------------------------------------------------------

    async def _pick_post(self, candidates: List[Dict]) -> Optional[Dict]:
        """Use LLM to pick the most interesting post to comment on."""
        if not candidates:
            return None

        # Load self-knowledge for context
        self_context = ""
        try:
            guild_id = int(os.getenv("GUILD_ID", "0"))
            from soupy_database.self_context import load_self_core, is_self_md_enabled
            if is_self_md_enabled() and guild_id:
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\nYour personality and interests:\n{core[:400]}\n"
        except Exception:
            pass

        listing = ""
        for i, c in enumerate(candidates[:10]):
            author = c.get("author", {}).get("displayName", "?")
            text = c.get("record", {}).get("text", "")[:250]
            likes = c.get("likeCount", 0)
            replies = c.get("replyCount", 0)
            listing += f"[{i}] {author} ({likes}♥ {replies} replies):\n{text}\n\n"

        user_prompt = f"{self_context}\nPosts to choose from:\n{listing}"
        result = await _llm_call(PICK_POST_SYSTEM, user_prompt, temperature=0.3, max_tokens=2048)

        if not result.strip():
            logger.info("🦋 LLM returned empty response for post pick")
            return candidates[0] if candidates else None

        try:
            idx = int(result.strip().splitlines()[0].strip())
            if 0 <= idx < len(candidates):
                chosen = candidates[idx]
                logger.info("🦋 Picked [%d]: %s",
                            idx, chosen.get("record", {}).get("text", "")[:80])
                return chosen
        except ValueError:
            pass

        # Default to first
        return candidates[0]

    # ------------------------------------------------------------------
    # Step 2b: Enrich post with web context
    # ------------------------------------------------------------------

    async def _enrich_post_context(self, post: Dict) -> str:
        """Gather web context about a post so the LLM understands the topic.

        Three parallel tasks:
          A. Fetch the article linked in the post embed (if any)
          B. Fetch an inline URL from the post text (if any, excluding embed URL)
          C. Ask the LLM for a focused search query, then DDG news search

        Also extracts embed card metadata (title/description) which is available
        even when the full article fetch fails.

        Returns a formatted context string, or "" if nothing found.
        """
        post_text = post.get("record", {}).get("text", "")
        embed = post.get("embed") or {}
        rec_embed = post.get("record", {}).get("embed") or {}

        # --- Extract embed card metadata (always available, no fetch needed) ---
        embed_external = embed.get("external") or rec_embed.get("external") or {}
        embed_url = embed_external.get("uri")
        embed_title = embed_external.get("title", "")
        embed_description = embed_external.get("description", "")

        # --- Task A: fetch full article from embed link ---
        async def _fetch_embed():
            if not embed_url:
                return None
            return await _fetch_article(embed_url)

        # --- Task B: fetch inline URL in post text (not the embed) ---
        text_urls = _re.findall(r"https?://\S+", post_text)
        extra_url = None
        for u in text_urls:
            cleaned = u.rstrip(".,;:!?)")
            if cleaned != embed_url:
                extra_url = cleaned
                break

        async def _fetch_extra():
            if not extra_url:
                return None
            return await _fetch_article(extra_url)

        # --- Task C: LLM-generated search query → DDG news search ---
        async def _search_topic():
            # Build material for the LLM to extract a search query from
            material = post_text
            if embed_title:
                material += f"\n[Link: {embed_title}]"
            if embed_description:
                material += f"\n{embed_description[:200]}"

            # Ask LLM for a focused 3-8 word search query
            try:
                query = await _llm_call(
                    "Extract the main news topic from this social media post. "
                    "Return a 3-8 word search query that would find the NEWS STORY "
                    "being discussed. Focus on specific names, events, or organizations. "
                    "Return ONLY the search query, nothing else.",
                    material[:500],
                    temperature=0.2, max_tokens=2048,
                )
                query = query.strip().strip('"\'')
                # Fallback: if LLM returns garbage, use cleaned post text
                if len(query) < 5 or len(query) > 150:
                    query = _re.sub(r"https?://\S+", "", post_text)
                    query = _re.sub(r"@[\w.]+", "", query).strip()[:100]
            except Exception:
                query = _re.sub(r"https?://\S+", "", post_text)
                query = _re.sub(r"@[\w.]+", "", query).strip()[:100]

            if len(query) < 8:
                return [], query
            logger.info("🦋 🔍 Enrichment search query: %s", query[:80])
            results = await _ddg_search(query, timelimit="w", max_results=5)
            return results, query

        # Run all three in parallel with a hard timeout
        try:
            embed_article, extra_article, search_tuple = await asyncio.wait_for(
                asyncio.gather(
                    _fetch_embed(), _fetch_extra(), _search_topic(),
                    return_exceptions=True,
                ),
                timeout=20,
            )
        except asyncio.TimeoutError:
            logger.warning("🦋 Context enrichment timed out")
            return ""

        # Unpack search results
        if isinstance(search_tuple, tuple) and len(search_tuple) == 2:
            search_results, search_query = search_tuple
        else:
            search_results, search_query = [], ""

        # Build the context block
        parts: List[str] = []

        # Embed card metadata (title + description) — available even without fetch
        if embed_title and not (isinstance(embed_article, dict) and embed_article):
            # Full fetch failed, but we still have the card metadata
            card_info = f"[LINK CARD] {embed_title}"
            if embed_description:
                card_info += f"\n{embed_description[:300]}"
            parts.append(card_info)

        for label, article in [("LINKED ARTICLE", embed_article),
                                ("ADDITIONAL LINK", extra_article)]:
            if isinstance(article, dict) and article:
                title = article.get("title", "")
                content = (article.get("content") or "")[:800]
                if title or content:
                    parts.append(f"[{label}] {title}\n{content}")

        if isinstance(search_results, list) and search_results:
            snippets = []
            for r in search_results[:5]:
                t = r.get("title", "")
                b = (r.get("body") or r.get("description") or "")[:150]
                if t:
                    snippets.append(f"- {t}: {b}")
            if snippets:
                parts.append("[RECENT NEWS ON THIS TOPIC]\n" + "\n".join(snippets))

        if not parts:
            logger.info("🦋 No enrichment context found for post")
            return ""

        context = (
            "CONTEXT (background info to help you understand the topic. "
            "do NOT cite sources, URLs, or mention searching in your reply. "
            "IMPORTANT: some of this context may be about RELATED BUT SEPARATE stories. "
            "only reply to what the ORIGINAL POST actually says, do not blend different stories together):\n"
            + "\n\n".join(parts)
        )
        n_articles = sum(1 for a in [embed_article, extra_article]
                         if isinstance(a, dict) and a)
        n_search = len(search_results) if isinstance(search_results, list) else 0
        logger.info("🦋 📚 Enrichment: %d chars (%d article(s), %d search results, query='%s')",
                     len(context), n_articles, n_search, search_query[:60])
        return context

    # ------------------------------------------------------------------
    # Step 3: Read thread and generate comment
    # ------------------------------------------------------------------

    async def _read_and_comment(self, post: Dict) -> Optional[Tuple[str, str, str, str]]:
        """Read the thread and generate a comment.

        Returns (comment_text, parent_uri, parent_cid, root_uri, root_cid)
        or None.
        """
        uri = post.get("uri", "")
        cid = post.get("cid", "")
        if not uri or not cid:
            return None

        # Fetch thread
        thread = await self.bsky.get_thread(uri, depth=6)
        if not thread:
            logger.info("🦋 Could not fetch thread")
            return None

        # Build the conversation context
        root_post = thread.get("post", {})
        root_author = root_post.get("author", {}).get("displayName", "?")
        root_text = root_post.get("record", {}).get("text", "")

        # Check for images in the post and describe them via vision model
        image_context = ""
        root_embed = root_post.get("embed") or {}
        image_urls = []
        # Direct image embeds
        for img in root_embed.get("images", []):
            url = img.get("fullsize") or img.get("thumb")
            if url:
                image_urls.append(url)
        # Images inside record embeds (quote posts, etc.)
        rec_embed = root_post.get("record", {}).get("embed") or {}
        for img in rec_embed.get("images", []):
            url = img.get("fullsize") or img.get("thumb")
            if url:
                image_urls.append(url)

        if image_urls:
            descriptions = []
            for img_url in image_urls[:2]:  # Max 2 images to avoid slowdown
                desc = await _describe_image(img_url)
                if desc:
                    descriptions.append(desc)
            if descriptions:
                image_context = "\n[IMAGE IN POST: " + " | ".join(descriptions) + "]\n"
                logger.info("🦋 👁 Post has %d image(s), described: %s",
                            len(image_urls), image_context.strip()[:100])

        replies = thread.get("replies", [])
        # Sort replies by likes to get the best ones
        replies.sort(
            key=lambda r: r.get("post", {}).get("likeCount", 0) or 0,
            reverse=True,
        )

        reply_lines = []
        for r in replies[:15]:
            rp = r.get("post", {})
            ra = rp.get("author", {}).get("displayName", "?")
            rt = rp.get("record", {}).get("text", "")[:200]
            rl = rp.get("likeCount", 0) or 0
            reply_lines.append(f"  {ra} ({rl}♥): {rt}")

        thread_context = (
            f"ORIGINAL POST by {root_author}:\n{root_text}\n{image_context}\n"
            f"TOP REPLIES ({len(replies)} total):\n"
            + "\n".join(reply_lines) if reply_lines else "(no replies yet)"
        )

        # Enrich with web context (linked articles + DDG search)
        enrichment = await self._enrich_post_context(root_post)

        # Load self-knowledge
        self_context = ""
        try:
            guild_id = int(os.getenv("GUILD_ID", "0"))
            from soupy_database.self_context import load_self_core, is_self_md_enabled
            if is_self_md_enabled() and guild_id:
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\nYour personality:\n{core[:400]}\n"
        except Exception:
            pass

        enrichment_block = f"\n{enrichment}\n" if enrichment else ""
        user_prompt = (
            f"{self_context}\n"
            f"{thread_context}\n\n"
            f"{enrichment_block}\n"
            f"Write one bluesky reply.\n"
            f"Your reply MUST reference a SPECIFIC detail from the post: a person's name, "
            f"a place, an event, a number, an organization. Generic takes that could apply "
            f"to any post are worthless. If the post mentions Todd Blanche, your reply should "
            f"mention Todd Blanche. If it mentions Hungary, say something about Hungary.\n"
            f"Match the thread's tone. One concrete insight about why it matters.\n"
            f"Do not repeat what others said. No hashtags, no @handles.\n"
            f"OUTPUT THE REPLY TEXT ONLY. No labels, no headers, no analysis, no reasoning."
        )

        # Generate 3 candidate comments, then pick the best one
        candidates_list: List[str] = []
        for i in range(3):
            candidate = await _llm_call(COMMENT_SYSTEM, user_prompt, temperature=0.65, max_tokens=2048)
            candidate = candidate.strip('"\'')

            # Strip meta-commentary lines the LLM sometimes outputs
            clean_lines = []
            for line in candidate.split("\n"):
                line_lower = line.strip().lower()
                if any(line_lower.startswith(p) for p in [
                    "who's ", "who is ", "the post ", "the author ", "context:",
                    "reply:", "note:", "analysis:", "reasoning:", "subject:",
                ]):
                    continue
                clean_lines.append(line.strip())
            candidate = " ".join(l for l in clean_lines if l)

            # Strip em dashes and hyphens used as dashes
            candidate = candidate.replace("—", ",").replace("–", ",").replace(" - ", ", ")

            # Enforce 300 char limit
            if len(candidate) > 295:
                for j in range(280, 0, -1):
                    if candidate[j] in ".!?":
                        candidate = candidate[:j + 1]
                        break
                else:
                    candidate = candidate[:295]
            # Trim to last complete sentence if cut off mid-thought
            if candidate and candidate[-1] not in ".!?":
                for j in range(len(candidate) - 1, 0, -1):
                    if candidate[j] in ".!?":
                        candidate = candidate[:j + 1]
                        break
            candidate = candidate.strip()
            if len(candidate) < 15:
                continue

            candidates_list.append(candidate)
            logger.info("🦋   Candidate %d: %s", i + 1, candidate[:100])

        # Have the LLM pick the best one
        judge_listing = ""
        for i, c in enumerate(candidates_list):
            judge_listing += f"[{i}] {c}\n\n"

        judge_system = (
            "You are judging which reply is the best to post on Bluesky.\n\n"
            "Pick the reply that:\n"
            "1. Names a SPECIFIC person, event, or detail from the original post. "
            "A reply that says 'politicians' when the post names 'Todd Blanche' is generic and bad.\n"
            "2. Could ONLY be a reply to THIS post. If you could paste it under a different post "
            "and it would still make sense, it's too generic.\n"
            "3. Adds a sharp, incisive angle — the real problem, the irony, who benefits.\n"
            "4. Punches UP at power, not down at regular people.\n"
            "5. Does NOT introduce random unrelated topics or comparisons.\n\n"
            "REJECT generic platitudes. Prefer specificity over cleverness.\n"
            "You MUST pick one. Return ONLY the index number (0, 1, or 2)."
        )

        judge_result = await _llm_call(
            judge_system,
            f"Original post by {root_author}:\n{root_text}\n\n"
            f"Candidate replies:\n{judge_listing}",
            temperature=0.2,
            max_tokens=2048,
        )

        try:
            chosen_idx = int(judge_result.strip()[0])
            if chosen_idx < 0 or chosen_idx >= len(candidates_list):
                chosen_idx = 0
        except (ValueError, IndexError):
            chosen_idx = 0

        # Light sanity check for replies — only reject if it contradicts the post
        # or invents specific false claims. Metaphors, satire, and sharp angles are fine.
        comment = candidates_list[chosen_idx]
        check_result = await _llm_call(
            "You are checking a reply for SERIOUS factual errors only.\n\n"
            "A reply is FINE if it uses metaphor, satire, exaggeration for effect, "
            "creative interpretation, or sharp political commentary. That's normal.\n\n"
            "A reply is BAD ONLY if it:\n"
            "- Invents a specific false claim (a fake statistic, a made-up quote, a wrong name)\n"
            "- Gets the basic subject of the post completely wrong (confuses who did what)\n"
            "- Contradicts what the post actually says\n\n"
            "Return OK if the reply is fine (even if creative/metaphorical).\n"
            "Return BAD followed by a reason ONLY for serious factual errors.",
            f"Original post by {root_author}:\n{root_text}\n\n"
            f"Proposed reply:\n{comment}",
            temperature=0.1, max_tokens=2048,
        )
        if check_result.strip().upper().startswith("BAD"):
            logger.info("🦋 ⚠ Reply sanity check failed: %s — trying next candidate", check_result.strip()[:80])
            # Try the other candidates
            for alt_idx in range(len(candidates_list)):
                if alt_idx != chosen_idx:
                    comment = candidates_list[alt_idx]
                    break

        logger.info("🦋 ━━━ Best Comment ━━━")
        logger.info("🦋   Post: %s: %s", root_author, root_text[:80])
        logger.info("🦋   Reply: %s", comment)
        logger.info("🦋   Length: %d chars", len(comment))

        return comment, uri, cid, uri, cid

    # ------------------------------------------------------------------
    # Step 4: Like some good comments
    # ------------------------------------------------------------------

    async def _like_good_comments(self, thread: Dict, max_likes: int = 3) -> int:
        """Like a few interesting comments in the thread."""
        _reset_daily_if_needed(self.history)
        likes_left = 10 - len(self.history.get("likes_today", []))
        if likes_left <= 0:
            logger.info("🦋 Daily like limit reached, skipping")
            return 0

        max_likes = min(max_likes, likes_left)
        replies = thread.get("replies", [])
        if not replies:
            return 0

        # Sort by likes, take the top ones we haven't liked
        replies.sort(
            key=lambda r: r.get("post", {}).get("likeCount", 0) or 0,
            reverse=True,
        )

        liked = 0
        for r in replies[:10]:
            if liked >= max_likes:
                break
            rp = r.get("post", {})
            r_uri = rp.get("uri", "")
            r_cid = rp.get("cid", "")
            r_likes = rp.get("likeCount", 0) or 0
            r_text = rp.get("record", {}).get("text", "")
            r_author = rp.get("author", {}).get("displayName", "?")

            # Only like posts with some engagement
            if r_likes < 3:
                continue
            # Don't like our own posts
            if rp.get("author", {}).get("did") == self.bsky.did:
                continue

            if await self.bsky.like_post(r_uri, r_cid):
                liked += 1
                self.history.setdefault("likes_today", []).append(r_uri)
                logger.info("🦋   ♥ Liked: %s: %s", r_author, r_text[:60])
                await asyncio.sleep(random.uniform(1.0, 3.0))  # Rate limit

        return liked

    # ------------------------------------------------------------------
    # Step 5: Maybe follow the post author
    # ------------------------------------------------------------------

    async def _maybe_follow(self, post: Dict) -> bool:
        """Consider following the post author if they're interesting."""
        _reset_daily_if_needed(self.history)
        follows_today = self.history.get("follows_today", [])
        if len(follows_today) >= 2:
            logger.info("🦋 Already followed 2 people today, skipping")
            return False

        author = post.get("author", {})
        author_did = author.get("did", "")
        author_name = author.get("displayName", author.get("handle", "?"))

        if not author_did or author_did == self.bsky.did:
            return False

        # Check if we already follow them (viewer state)
        viewer = author.get("viewer", {})
        if viewer.get("following"):
            return False

        # Get their recent posts to judge
        recent = await self.bsky.get_author_feed(author_did, limit=5)
        if len(recent) < 3:
            return False

        posts_text = ""
        for i, p in enumerate(recent[:5]):
            text = p.get("record", {}).get("text", "")[:150]
            likes = p.get("likeCount", 0) or 0
            posts_text += f"[{i}] ({likes}♥) {text}\n"

        result = await _llm_call(
            JUDGE_FOLLOW_SYSTEM,
            f"User: {author_name}\n\nRecent posts:\n{posts_text}",
            temperature=0.3,
            max_tokens=2048,
        )

        if result.strip().upper().startswith("YES"):
            if await self.bsky.follow_user(author_did):
                self.history.setdefault("follows_today", []).append(author_did)
                _save_history(self.history)
                logger.info("🦋 ✅ Followed %s", author_name)
                return True

        return False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def _run_engage_pipeline(self, source: str = "manual") -> Tuple[bool, str, Optional[str]]:
        """Run the full engagement pipeline.

        Returns (success, status_message, comment_url_or_none).
        """
        import time as _time
        start = _time.monotonic()

        logger.info("🦋 ╔══════════════════════════════════════════════════╗")
        logger.info("🦋 ║  BLUESKY ENGAGE PIPELINE                        ║")
        logger.info("🦋 ╚══════════════════════════════════════════════════╝")

        # Auth
        token = await self.bsky.auth()
        if not token:
            return False, "Bluesky authentication failed — check BLUESKY_HANDLE and BLUESKY_APP_PASSWORD", None

        # Discover posts
        candidates = await self._discover_posts()
        if not candidates:
            return False, "No interesting posts found to engage with", None

        # Pick the best one
        chosen = await self._pick_post(candidates)
        if not chosen:
            return False, "LLM declined all posts (SKIP)", None

        chosen_uri = chosen.get("uri", "")
        chosen_cid = chosen.get("cid", "")
        chosen_author = chosen.get("author", {}).get("displayName", "?")
        chosen_text = chosen.get("record", {}).get("text", "")[:100]

        # Rate limit pause
        await asyncio.sleep(random.uniform(2.0, 5.0))

        # Read thread and generate comment
        result = await self._read_and_comment(chosen)
        if not result:
            return False, "Failed to generate a comment", None

        comment_text, parent_uri, parent_cid, root_uri, root_cid = result

        # Rate limit pause before posting
        await asyncio.sleep(random.uniform(3.0, 7.0))

        # Post the reply
        reply_result = await self.bsky.create_reply(
            comment_text, parent_uri, parent_cid, root_uri, root_cid
        )
        if not reply_result:
            return False, "Failed to post reply to Bluesky", None

        reply_uri = reply_result.get("uri", "")
        reply_url = _post_url(reply_uri)

        logger.info("🦋 ✅ Reply posted: %s", reply_url)

        # Record in history
        _reset_daily_if_needed(self.history)
        self.history.setdefault("comments", []).append({
            "post_uri": chosen_uri,
            "reply_uri": reply_uri,
            "reply_url": reply_url,
            "post_author": chosen_author,
            "post_text": chosen_text,
            "comment": comment_text,
            "source": source,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 100 comments
        self.history["comments"] = self.history["comments"][-100:]
        _save_history(self.history)

        # Rate limit pause
        await asyncio.sleep(random.uniform(2.0, 4.0))

        # Like some good comments in the thread
        thread = await self.bsky.get_thread(chosen_uri, depth=6)
        likes = 0
        if thread:
            likes = await self._like_good_comments(thread, max_likes=2)

        # Rate limit pause
        await asyncio.sleep(random.uniform(1.0, 3.0))

        # Maybe follow the post author
        followed = await self._maybe_follow(chosen)

        elapsed = _time.monotonic() - start
        status = (
            f"Replied to {chosen_author}'s post"
            f" · liked {likes} comment(s)"
            f"{' · followed ' + chosen_author if followed else ''}"
            f" · {elapsed:.0f}s"
        )
        logger.info("🦋 ━━━ Pipeline complete: %s ━━━", status)

        return True, status, reply_url

    # ------------------------------------------------------------------
    # Repost (quote-post) pipeline
    # ------------------------------------------------------------------

    async def _run_repost_pipeline(self, source: str = "manual") -> Tuple[bool, str, Optional[str]]:
        """Find an interesting post and quote-post it with commentary."""
        import time as _time
        start = _time.monotonic()

        logger.info("🦋 ╔══════════════════════════════════════════════════╗")
        logger.info("🦋 ║  BLUESKY REPOST PIPELINE                        ║")
        logger.info("🦋 ╚══════════════════════════════════════════════════╝")

        token = await self.bsky.auth()
        if not token:
            return False, "Bluesky auth failed", None

        # Discover posts (same as reply pipeline)
        candidates = await self._discover_posts()
        if not candidates:
            return False, "No interesting posts found to repost", None

        # Filter out posts we've already reposted
        reposted_uris = {r.get("post_uri") for r in self.history.get("reposts", [])}
        candidates = [c for c in candidates if c.get("uri") not in reposted_uris]
        if not candidates:
            return False, "All candidates already reposted", None

        # Pick the best one — prefer posts with high engagement and shareable content
        chosen = await self._pick_post(candidates)
        if not chosen:
            return False, "LLM declined all posts", None

        chosen_uri = chosen.get("uri", "")
        chosen_cid = chosen.get("cid", "")
        chosen_author = chosen.get("author", {}).get("displayName", "?")
        chosen_text = chosen.get("record", {}).get("text", "")[:200]

        logger.info("🦋 Reposting: %s: %s", chosen_author, chosen_text[:80])

        # Load self-knowledge
        self_context = ""
        try:
            guild_id = int(os.getenv("GUILD_ID", "0"))
            from soupy_database.self_context import load_self_core, is_self_md_enabled
            if is_self_md_enabled() and guild_id:
                core = load_self_core(guild_id)
                if core:
                    self_context = f"\nYour personality:\n{core[:400]}\n"
        except Exception:
            pass

        # Check for images in the chosen post and describe them
        image_context = ""
        chosen_embed = chosen.get("embed") or {}
        image_urls = []
        for img in chosen_embed.get("images", []):
            url = img.get("fullsize") or img.get("thumb")
            if url:
                image_urls.append(url)
        if image_urls:
            descriptions = []
            for img_url in image_urls[:2]:
                desc = await _describe_image(img_url)
                if desc:
                    descriptions.append(desc)
            if descriptions:
                image_context = "\n[IMAGE IN POST: " + " | ".join(descriptions) + "]\n"
                logger.info("🦋 👁 Quote target has %d image(s): %s",
                            len(image_urls), image_context.strip()[:100])

        # Enrich with web context
        enrichment = await self._enrich_post_context(chosen)
        enrichment_block = f"\n{enrichment}\n" if enrichment else ""

        # Generate 3 candidate commentaries
        user_prompt = (
            f"{self_context}\n"
            f"Post by {chosen_author}:\n{chosen_text}\n{image_context}\n"
            f"{enrichment_block}\n"
            f"Write brief commentary to go with sharing this post. "
            f"200 chars max, lower case, no quotes."
        )

        commentaries: List[str] = []
        for i in range(3):
            c = await _llm_call(QUOTE_POST_SYSTEM, user_prompt, temperature=0.65, max_tokens=2048)
            c = c.strip('"\'')
            c = c.replace("—", ",").replace("–", ",").replace(" - ", ", ")
            if len(c) > 200:
                for j in range(195, 0, -1):
                    if c[j] in ".!? ":
                        c = c[:j + 1].rstrip()
                        break
                else:
                    c = c[:200]
            commentaries.append(c)
            logger.info("🦋   Quote candidate %d: %s", i + 1, c[:80])

        # Judge best
        listing = "".join(f"[{i}] {c}\n\n" for i, c in enumerate(commentaries))
        judge_result = await _llm_call(
            "Pick the best commentary for sharing this post. Must be relevant and witty. "
            "You MUST pick one. Return ONLY the index (0, 1, or 2).",
            f"Post by {chosen_author}:\n{chosen_text}\n\nCandidates:\n{listing}",
            temperature=0.2, max_tokens=2048,
        )

        try:
            idx = int(judge_result.strip()[0])
            commentary = commentaries[idx] if 0 <= idx < 3 else commentaries[0]
        except (ValueError, IndexError):
            commentary = commentaries[0]

        await asyncio.sleep(random.uniform(3.0, 6.0))

        # Post the quote-post
        result = await self.bsky.quote_post(commentary, chosen_uri, chosen_cid)
        if not result:
            return False, "Failed to create quote-post", None

        post_uri = result.get("uri", "")
        post_url = _post_url(post_uri)

        # Record
        _reset_daily_if_needed(self.history)
        self.history.setdefault("reposts", []).append({
            "post_uri": chosen_uri,
            "our_uri": post_uri,
            "our_url": post_url,
            "post_author": chosen_author,
            "commentary": commentary,
            "source": source,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.history["reposts"] = self.history["reposts"][-50:]
        _save_history(self.history)

        elapsed = _time.monotonic() - start
        status = f"Quote-posted {chosen_author}'s post · {elapsed:.0f}s"
        logger.info("🦋 ✅ %s", status)
        return True, status, post_url

    # ------------------------------------------------------------------
    # Original post pipeline
    # ------------------------------------------------------------------

    async def _discover_article(
        self, self_context: str
    ) -> Tuple[Optional[str], str, str, str]:
        """Mine Bluesky + DDG for articles, rate, fetch top 3, judge best.

        Returns (article_url, title, snippet, content) or (None, '', '', '').
        """
        seen_urls: set = set()
        posted_urls = {p.get("url") for p in self.history.get("posts", [])}
        current_year = datetime.now().year
        all_results: List[Dict] = []

        def _add_article(url: str, title: str, description: str, likes: int = 0) -> None:
            if not url or url in seen_urls or url in posted_urls:
                return
            if not url.startswith("http"):
                return
            if "bsky.app" in url or "bsky.social" in url:
                return
            year_match = _re.search(r"[/-](\d{4})[/-]", url)
            if year_match and int(year_match.group(1)) < current_year:
                return
            seen_urls.add(url)
            all_results.append({
                "href": url, "title": title,
                "body": description[:200],
                "source": "bluesky", "bsky_likes": likes,
            })

        logger.info("🦋 ━━━ Mining Bluesky for articles ━━━")

        # Source 1: Timeline posts with embedded links
        timeline = await self.bsky.get_timeline(limit=50)
        for p in timeline:
            embed = p.get("embed") or {}
            ext = embed.get("external") or {}
            if ext.get("uri"):
                likes = p.get("likeCount", 0) or 0
                if likes >= 5:
                    _add_article(ext["uri"], ext.get("title", ""), ext.get("description", ""), likes)
        logger.info("🦋   Timeline: %d articles with links", len(all_results))

        prev_count = len(all_results)

        # Source 2: Trending topic searches
        trending = await self.bsky.get_trending()
        for topic in trending[:3]:
            posts = await self.bsky.search_posts(topic, sort="top", limit=10)
            for p in posts:
                embed = p.get("embed") or p.get("record", {}).get("embed") or {}
                ext = embed.get("external") or {}
                if ext.get("uri"):
                    likes = p.get("likeCount", 0) or 0
                    if likes >= 3:
                        _add_article(ext["uri"], ext.get("title", ""), ext.get("description", ""), likes)
            await asyncio.sleep(0.5)
        logger.info("🦋   Trending: +%d articles", len(all_results) - prev_count)

        prev_count = len(all_results)

        # Source 3: News keyword searches
        for query in ["breaking news", "report released", "just announced"]:
            posts = await self.bsky.search_posts(query, sort="latest", limit=10)
            for p in posts:
                embed = p.get("embed") or p.get("record", {}).get("embed") or {}
                ext = embed.get("external") or {}
                if ext.get("uri"):
                    likes = p.get("likeCount", 0) or 0
                    _add_article(ext["uri"], ext.get("title", ""), ext.get("description", ""), likes)
            await asyncio.sleep(0.5)
        logger.info("🦋   News search: +%d articles", len(all_results) - prev_count)

        bsky_article_count = len(all_results)

        # Supplement with DuckDuckGo if needed
        if len(all_results) < 5:
            logger.info("🦋 Only %d from Bluesky, supplementing with DuckDuckGo", len(all_results))
            queries_raw = await _llm_call(
                SEARCH_QUERY_FOR_POST_SYSTEM,
                f"{self_context}\nGenerate 3 diverse search queries for today's news.",
                temperature=0.6, max_tokens=2048,
            )
            queries = [q.strip() for q in queries_raw.strip().splitlines() if q.strip()][:3]
            logger.info("🦋 DDG queries: %s", queries)
            for query in queries:
                results = await _ddg_search(query, timelimit="d", max_results=6)
                if len(results) < 2:
                    results = await _ddg_search(query, timelimit="w", max_results=6)
                for r in results:
                    r_url = r.get("href", "")
                    if r_url and r_url not in seen_urls and r_url not in posted_urls:
                        seen_urls.add(r_url)
                        ym = _re.search(r"[/-](\d{4})[/-]", r_url)
                        if ym and int(ym.group(1)) < current_year:
                            continue
                        ts = f"{r.get('title', '')} {r.get('body', '')}"
                        if _re.search(r"\b(201\d|202[0-4])\b", ts) and str(current_year) not in ts:
                            continue
                        all_results.append(r)
                await asyncio.sleep(0.5)

        all_results.sort(key=lambda a: a.get("bsky_likes", 0), reverse=True)

        logger.info("🦋 Total candidate articles: %d (%d from Bluesky, %d from DDG)",
                     len(all_results), bsky_article_count, len(all_results) - bsky_article_count)
        for i, a in enumerate(all_results[:10]):
            likes = a.get("bsky_likes", 0)
            src = "bsky" if a.get("source") == "bluesky" else "ddg"
            logger.info("🦋   [%d] %s %d♥ %s — %s",
                         i, src, likes, a.get("href", "")[:70], a.get("title", "?")[:60])
        if not all_results:
            return None, "", "", ""

        # Pick top 3
        listing = ""
        for i, a in enumerate(all_results[:10]):
            listing += f"[{i}] {a.get('title', '?')}\n{a.get('body', '')[:150]}\nURL: {a.get('href', '')}\n\n"

        rate_result = await _llm_call(
            "Pick the 3 most interesting, surprising, or conversation-worthy NEWS articles.\n\n"
            "GOOD articles: real news from credible sources about a specific event that happened.\n"
            "BAD articles — REJECT these: listicles ('5 things...', '4 breakthroughs...'), "
            "forum posts, opinion blogs, evergreen content, how-to guides, SEO spam.\n"
            "Check the URL — credible sources (reuters, AP, nytimes, washpost, bbc, ars, etc.) "
            "are better than random blogs or aggregator sites.\n\n"
            "Return 3 index numbers, one per line, best first. If none are real news, return SKIP.",
            f"{self_context}\nArticles:\n{listing}",
            temperature=0.3, max_tokens=2048,
        )
        if rate_result.strip().upper().startswith("SKIP"):
            return None, "", "", ""

        top_indices: List[int] = []
        for line in rate_result.strip().splitlines():
            for token in line.strip().split():
                try:
                    idx = int(token)
                    if 0 <= idx < len(all_results) and idx not in top_indices:
                        top_indices.append(idx)
                        break
                except ValueError:
                    continue
        if not top_indices:
            top_indices = [0]
        top_indices = top_indices[:3]

        logger.info("🦋 Top %d article candidates: %s", len(top_indices), top_indices)

        # Fetch full content and verify recency
        fetched: List[Dict[str, Any]] = []
        for idx in top_indices:
            a = all_results[idx]
            a_url = a.get("href", "")
            a_title = a.get("title", "")
            a_snippet = a.get("body", "")
            full = await _fetch_article(a_url)
            if full and full.get("content"):
                content = full["content"][:2000]
                title = full.get("title") or a_title
                pub_date = full.get("date")
            else:
                content = a_snippet
                title = a_title
                pub_date = None
            # Check article age using shared estimator (also include title/snippet)
            from soupy_dailypost import _estimate_article_age_days
            scan_text = " ".join(filter(None, [title, a_snippet, content]))
            age = _estimate_article_age_days(
                pub_date=pub_date, url=a_url, text=scan_text,
            )
            if age is None:
                logger.info("🦋   [%d] ⏭ No date found, rejecting: %s", idx, title[:60])
                continue
            if age > 14:
                logger.info("🦋   [%d] ⏭ Too old (~%d days): %s", idx, age, title[:60])
                continue
            fetched.append({"url": a_url, "title": title, "snippet": a_snippet, "content": content})
            logger.info("🦋   [%d] %s (date=%s, %d chars)", idx, title[:60], pub_date or "?", len(content))
            await asyncio.sleep(0.5)

        if not fetched:
            return None, "", "", ""

        # Judge best
        if len(fetched) > 1:
            judge_articles = ""
            for i, fa in enumerate(fetched):
                judge_articles += f"[{i}] {fa['title']}\n{fa['content'][:500]}\n\n"
            pick_result = await _llm_call(
                "You are choosing which article to share on Bluesky. Read the content of each.\n"
                "Pick the one that will get the MOST ENGAGEMENT.\n\n"
                "Good topics: politics, government, AI/tech, finance/economy, labor, power.\n"
                "The best articles hold power accountable, expose hypocrisy, or reveal something "
                "people should be angry or concerned about.\n"
                "AI and tech stories work great if there's an angle about corporate power, "
                "worker displacement, privacy, or regulatory failure.\n"
                "Financial stories work if they show inequality, corporate greed, or who benefits.\n\n"
                "Between a straight news report and a critical angle, pick the critical angle.\n\n"
                "You MUST pick one. Return ONLY the index number (0, 1, or 2). Nothing else.",
                f"{self_context}\nArticles:\n{judge_articles}",
                temperature=0.3, max_tokens=2048,
            )
            try:
                pick_idx = int(pick_result.strip()[0])
                if pick_idx < 0 or pick_idx >= len(fetched):
                    pick_idx = 0
            except (ValueError, IndexError):
                pick_idx = 0
        else:
            pick_idx = 0

        chosen = fetched[pick_idx]
        logger.info("🦋 ✅ Best article: [%d] %s", pick_idx, chosen["title"][:80])
        return chosen["url"], chosen["title"], chosen["snippet"], chosen["content"]

    async def _run_original_post_pipeline(
        self, source: str = "manual", article_url: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """Find an interesting article and post about it on Bluesky.

        If article_url is provided, skip discovery and post about that specific article.
        """
        import time as _time
        start = _time.monotonic()

        logger.info("🦋 ╔══════════════════════════════════════════════════╗")
        logger.info("🦋 ║  BLUESKY ORIGINAL POST PIPELINE                 ║")
        if article_url:
            logger.info("🦋 ║  URL provided: %s", article_url[:50])
        logger.info("🦋 ╚══════════════════════════════════════════════════╝")

        token = await self.bsky.auth()
        if not token:
            return False, "Bluesky auth failed", None

        # Load self-knowledge for personality context
        self_context = ""
        try:
            guild_id = int(os.getenv("GUILD_ID", "0"))
            from soupy_database.self_context import load_self_core, is_self_md_enabled
            if is_self_md_enabled() and guild_id:
                core = load_self_core(guild_id)
                if core:
                    self_context = f"Your personality and interests:\n{core[:400]}\n"
        except Exception:
            pass

        # If a specific URL was provided, skip all discovery and go straight to fetching
        if article_url:
            logger.info("🦋 Fetching provided article: %s", article_url[:80])
            full = await _fetch_article(article_url)
            if full and full.get("content"):
                article_content = full["content"][:2000]
                article_title = full.get("title") or article_url
            else:
                return False, f"Could not fetch article content from {article_url[:60]}", None

            article_snippet = article_content[:200]
            logger.info("🦋 ✅ Fetched: %s (%d chars)", article_title[:70], len(article_content))

            # Skip straight to generating post candidates
            # (article_url, article_title, article_snippet, article_content are all set)
        else:
            # --- Auto-discovery: mine Bluesky + DDG for articles ---
            # Try up to 3 times to find a suitable article
            article_url = None
            for attempt in range(3):
                article_url, article_title, article_snippet, article_content = await self._discover_article(self_context)
                if article_url:
                    break
                logger.info("🦋 Article discovery attempt %d failed, retrying...", attempt + 1)
                await asyncio.sleep(random.uniform(2.0, 5.0))
            if not article_url:
                return False, "No suitable article found after 3 attempts", None

        # Generate 3 candidate posts
        post_user = (
            f"{self_context}\n"
            f"Article: {article_title}\n\n"
            f"Content:\n{article_content[:1500]}\n\n"
            f"Write a SHORT post (1-2 sentences, under 200 chars). "
            f"Lower case, no quotes. Finish your thought. "
            f"Do NOT include the URL — it will be attached as a link card."
        )

        candidates: List[str] = []
        for i in range(3):
            c = await _llm_call(ORIGINAL_POST_SYSTEM, post_user, temperature=0.65, max_tokens=2048)
            c = c.strip('"\'')
            # Strip meta-commentary lines
            clean_lines = []
            for line in c.split("\n"):
                line_lower = line.strip().lower()
                if any(line_lower.startswith(p) for p in [
                    "who's ", "who is ", "the article ", "the author ", "context:",
                    "post:", "note:", "analysis:", "reasoning:", "subject:", "commentary:",
                ]):
                    continue
                clean_lines.append(line.strip())
            c = " ".join(l for l in clean_lines if l)
            # Strip em dashes and hyphens used as dashes
            c = c.replace("—", ",").replace("–", ",").replace(" - ", ", ")
            if len(c) > 295:
                for j in range(290, 0, -1):
                    if c[j] in ".!?":
                        c = c[:j + 1]
                        break
                else:
                    c = c[:295]
            # Trim to last complete sentence if it ends mid-thought
            if c and c[-1] not in ".!?":
                for j in range(len(c) - 1, 0, -1):
                    if c[j] in ".!?":
                        c = c[:j + 1]
                        break
            c = c.strip()
            if len(c) < 20:
                continue  # Too short after trimming, skip this candidate
            candidates.append(c)
            logger.info("🦋   Post candidate %d (%d chars): %s", i + 1, len(c), c[:80])

        # Judge best
        judge_listing = "".join(f"[{i}] {c}\n\n" for i, c in enumerate(candidates))
        judge_result = await _llm_call(
            "Pick the best post for sharing this article on Bluesky.\n\n"
            "The BEST post:\n"
            "- Names the specific problem, irony, or absurdity\n"
            "- Connects it to power, money, or who benefits vs who gets screwed\n"
            "- Is sharp, punchy, and could only be about THIS article\n"
            "- Makes someone think 'damn, good point' or laugh\n\n"
            "The WORST post:\n"
            "- Uses vague metaphors or comparisons to unrelated things\n"
            "- Could apply to any article (generic takes)\n"
            "- Just summarizes without a take\n"
            "- Is wishy-washy, both-sides, or noncommittal\n\n"
            "You MUST pick one. Return ONLY the index (0, 1, or 2).",
            f"Article: {article_title}\n\nCandidates:\n{judge_listing}",
            temperature=0.2, max_tokens=2048,
        )

        try:
            idx = int(judge_result.strip()[0])
            if idx < 0 or idx >= len(candidates):
                idx = 0
        except (ValueError, IndexError):
            idx = 0

        # Fact-check ALL candidates upfront so the logs show every verdict,
        # then pick the judge's choice if it passed, else the next ranked one that did.
        verdicts: Dict[int, Tuple[bool, str]] = {}
        for c_idx, candidate in enumerate(candidates):
            check_result = await _llm_call(
                FACT_CHECK_SYSTEM,
                f"Article: {article_title}\n\nArticle content:\n{article_content[:1500]}\n\n"
                f"Proposed post:\n{candidate}",
                temperature=0.1, max_tokens=2048,
            )
            reason = check_result.strip()
            is_accurate = reason.upper().startswith("ACCURATE")
            verdicts[c_idx] = (is_accurate, reason)
            marker = "✅ PASS" if is_accurate else "⚠ FAIL"
            logger.info("🦋 Fact-check %s candidate %d: %s — %s",
                         marker, c_idx, candidate[:60], reason[:100])

        ranked = [idx] + [i for i in range(len(candidates)) if i != idx]
        post_text = None
        for check_idx in ranked:
            if verdicts.get(check_idx, (False, ""))[0]:
                post_text = candidates[check_idx]
                logger.info("🦋 ✅ Using candidate %d (judge picked %d)", check_idx, idx)
                break

        if not post_text:
            # All candidates failed fact-check — use the judge's pick anyway but log warning
            post_text = candidates[idx]
            logger.warning("🦋 ⚠ All candidates failed fact-check, using judge's pick (%d) anyway", idx)

        # Step 6: Fetch og:image and upload as blob for the link card thumbnail
        thumb_blob = None
        og_result = await _fetch_og_image(article_url)
        if og_result:
            image_bytes, mime_type = og_result
            logger.info("🦋 Uploading og:image (%d bytes, %s)", len(image_bytes), mime_type)
            thumb_blob = await self.bsky.upload_blob(image_bytes, mime_type)
            if thumb_blob:
                logger.info("🦋 ✅ Thumbnail uploaded")
            else:
                logger.info("🦋 ⚠ Thumbnail upload failed, posting without image")
        else:
            logger.info("🦋 No og:image found, posting without thumbnail")

        await asyncio.sleep(random.uniform(3.0, 6.0))

        # Step 7: Post with link card embed (using original URL for the card)
        result = await self.bsky.create_post(
            post_text,
            link_url=article_url,
            link_title=article_title,
            link_description=article_snippet[:200],
            thumb_blob=thumb_blob,
        )
        if not result:
            return False, "Failed to create post", None

        our_uri = result.get("uri", "")
        our_url = _post_url(our_uri)

        # Record
        _reset_daily_if_needed(self.history)
        self.history.setdefault("posts", []).append({
            "url": article_url,
            "title": article_title,
            "our_uri": our_uri,
            "our_url": our_url,
            "text": post_text,
            "source": source,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.history["posts"] = self.history["posts"][-50:]
        _save_history(self.history)

        elapsed = _time.monotonic() - start
        status = f"Posted about: {article_title[:60]} · {elapsed:.0f}s"
        logger.info("🦋 ✅ %s", status)
        return True, status, our_url

    # ------------------------------------------------------------------
    # Autonomous reply scheduling
    # ------------------------------------------------------------------

    def _save_schedule(self) -> None:
        """Persist the current schedule to disk so it survives restarts."""
        try:
            os.makedirs(os.path.dirname(SCHEDULE_PATH), exist_ok=True)
            payload = {
                "date": self._schedule_date,
                "events": [{"time": t.isoformat(), "action": a} for t, a in self._schedule],
            }
            Path(SCHEDULE_PATH).write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("🦋 Failed to save schedule: %s", e)

    def _load_schedule(self) -> bool:
        """Load schedule from disk. Returns True if a valid schedule for today was loaded."""
        if not os.path.exists(SCHEDULE_PATH):
            return False
        try:
            data = json.loads(Path(SCHEDULE_PATH).read_text(encoding="utf-8"))
            saved_date = data.get("date")
            if not saved_date:
                return False

            import pytz
            pacific = pytz.timezone("US/Pacific")
            today_iso = datetime.now(pacific).date().isoformat()
            if saved_date != today_iso:
                return False  # Saved schedule is from a different day

            events: List[Tuple[datetime, str]] = []
            for entry in data.get("events", []):
                try:
                    t = datetime.fromisoformat(entry["time"])
                    events.append((t, entry["action"]))
                except (KeyError, ValueError):
                    continue

            # Drop events that are already in the past
            now_pacific = datetime.now(pacific)
            self._schedule = [(t, a) for t, a in events if t > now_pacific]
            self._schedule_date = saved_date
            logger.info("🦋 Loaded saved schedule (%d events remaining for today)", len(self._schedule))

            # Push the loaded schedule to the dashboard
            schedule_list = [
                {"time": t.isoformat(), "action": a} for t, a in self._schedule
            ]
            next_run = self._schedule[0][0].isoformat() if self._schedule else None
            self._update_dashboard(
                enabled=self._is_auto_enabled(),
                interval="4-7 replies + 1 repost + 1 post/day",
                schedule=schedule_list,
                next_run=next_run,
            )
            return True
        except Exception as e:
            logger.debug("🦋 Failed to load schedule: %s", e)
            return False

    def _generate_schedule(self) -> None:
        """Generate daily schedule: 4-7 replies + 1 repost + 1 original post.

        All between 6am-11pm Pacific, randomly spaced, min 45min gap.
        Each entry is (time, action) where action is 'reply', 'repost', or 'post'.
        """
        import pytz
        pacific = pytz.timezone("US/Pacific")
        now_pacific = datetime.now(pacific)
        today = now_pacific.date()

        window_start = pacific.localize(
            datetime.combine(today, datetime.min.time().replace(hour=6))
        )
        window_end = pacific.localize(
            datetime.combine(today, datetime.min.time().replace(hour=23))
        )

        window_seconds = int((window_end - window_start).total_seconds())

        def _rand_time() -> datetime:
            return window_start + timedelta(seconds=random.randint(0, window_seconds))

        # Build all scheduled events with their action types
        events: List[Tuple[datetime, str]] = []

        # Replies (configurable range)
        try:
            reply_min = int(os.getenv("BLUESKY_REPLIES_MIN", "4"))
            reply_max = int(os.getenv("BLUESKY_REPLIES_MAX", "7"))
        except ValueError:
            reply_min, reply_max = 4, 7
        reply_count = random.randint(reply_min, max(reply_min, reply_max))
        for _ in range(reply_count):
            events.append((_rand_time(), "reply"))

        # Reposts per day (configurable)
        try:
            repost_count = int(os.getenv("BLUESKY_REPOSTS_PER_DAY", "1"))
        except ValueError:
            repost_count = 1
        for _ in range(repost_count):
            events.append((_rand_time(), "repost"))

        # Original posts per day (configurable)
        try:
            post_count = int(os.getenv("BLUESKY_POSTS_PER_DAY", "1"))
        except ValueError:
            post_count = 1
        for _ in range(post_count):
            events.append((_rand_time(), "post"))

        # Sort by time
        events.sort(key=lambda e: e[0])

        # Enforce minimum 45-minute gap
        spaced: List[Tuple[datetime, str]] = []
        for t, action in events:
            if not spaced or (t - spaced[-1][0]).total_seconds() >= 2700:
                spaced.append((t, action))

        # Drop past times
        self._schedule = [(t, a) for t, a in spaced if t > now_pacific]
        self._schedule_date = today.isoformat()

        # Remove entries for actions already done today (auto only)
        today_iso = today.isoformat()
        auto_replies_done = sum(
            1 for c in self.history.get("comments", [])
            if c.get("ts", "").startswith(today_iso) and c.get("source") == "auto"
        )
        auto_reposts_done = sum(
            1 for r in self.history.get("reposts", [])
            if r.get("ts", "").startswith(today_iso) and r.get("source") == "auto"
        )
        auto_posts_done = sum(
            1 for p in self.history.get("posts", [])
            if p.get("ts", "").startswith(today_iso) and p.get("source") == "auto"
        )

        # Remove completed actions from schedule
        remaining: List[Tuple[datetime, str]] = []
        skip_reply, skip_repost, skip_post = auto_replies_done, auto_reposts_done, auto_posts_done
        for t, action in self._schedule:
            if action == "reply" and skip_reply > 0:
                skip_reply -= 1
                continue
            elif action == "repost" and skip_repost > 0:
                skip_repost -= 1
                continue
            elif action == "post" and skip_post > 0:
                skip_post -= 1
                continue
            remaining.append((t, action))
        self._schedule = remaining

        logger.info("🦋 ━━━ Daily Bluesky schedule (%d events remaining) ━━━", len(self._schedule))
        logger.info("🦋   Done today: %d replies, %d reposts, %d posts",
                     auto_replies_done, auto_reposts_done, auto_posts_done)
        for i, (t, a) in enumerate(self._schedule):
            logger.info("🦋   [%d] %s — %s", i + 1, t.strftime("%I:%M %p %Z"), a)

        # Persist to disk so the schedule survives restarts
        self._save_schedule()

        # Update dashboard
        schedule_list = [
            {"time": t.isoformat(), "action": a} for t, a in self._schedule
        ]
        next_run = self._schedule[0][0].isoformat() if self._schedule else None
        self._update_dashboard(
            enabled=self._is_auto_enabled(),
            interval="4-7 replies + 1 repost + 1 post/day",
            schedule=schedule_list,
            next_run=next_run,
        )

    def _owner_mentions(self) -> str:
        """Build a Discord mention string for all owner IDs."""
        ids = self._owner_ids()
        if not ids:
            return ""
        return " ".join(f"<@{i}>" for i in ids)

    async def _report_to_musing_channel(self, message: str) -> None:
        """Post a status message to the musing channel, tagging owner(s)."""
        musing_ch_id = self._musing_channel_id()
        if not musing_ch_id:
            return
        ch = self.bot.get_channel(musing_ch_id)
        if ch is None:
            try:
                ch = await self.bot.fetch_channel(musing_ch_id)
            except Exception:
                return
        if ch:
            mentions = self._owner_mentions()
            payload = f"{mentions} {message}".strip() if mentions else message
            allowed = discord.AllowedMentions(users=True)
            await ch.send(payload, allowed_mentions=allowed)

    @tasks.loop(seconds=60)
    async def _auto_loop(self) -> None:
        try:
            if not self._is_auto_enabled():
                return

            import pytz
            pacific = pytz.timezone("US/Pacific")
            now_pacific = datetime.now(pacific)
            today_str = now_pacific.date().isoformat()

            # New day or first run → try to load saved schedule, else generate
            if self._schedule_date != today_str:
                if not self._load_schedule():
                    self._generate_schedule()

            if not self._schedule:
                return

            # Check if it's time for the next event
            next_time, action = self._schedule[0]
            if now_pacific < next_time:
                return

            # Pop and dispatch
            self._schedule.pop(0)
            self._save_schedule()  # Persist updated schedule
            now_iso = datetime.now(timezone.utc).isoformat()
            logger.info("🦋 ⏰ Auto-%s triggered (scheduled for %s)",
                         action, next_time.strftime("%I:%M %p"))

            # Update dashboard: next_run
            next_run = self._schedule[0][0].isoformat() if self._schedule else None
            self._update_dashboard(
                last_run=now_iso, next_run=next_run,
                schedule=[{"time": t.isoformat(), "action": a} for t, a in self._schedule],
            )

            # Dispatch with one retry on failure
            success = False
            status = ""
            url = None
            for try_num in range(2):
                if action == "reply":
                    success, status, url = await self._run_engage_pipeline(source="auto")
                elif action == "repost":
                    success, status, url = await self._run_repost_pipeline(source="auto")
                elif action == "post":
                    success, status, url = await self._run_original_post_pipeline(source="auto")
                else:
                    break

                if success:
                    break
                if try_num == 0:
                    logger.info("🦋 ⏭ Auto-%s failed (%s), retrying in 30s...", action, status)
                    await asyncio.sleep(30)

            # Update dashboard with result
            if success and url:
                logger.info("🦋 ✅ Auto-%s: %s", action, status)
                ts_now = datetime.now(timezone.utc).isoformat()
                if action == "reply":
                    self._update_dashboard(last_reply=ts_now)
                    last = self.history.get("comments", [])[-1] if self.history.get("comments") else {}
                    post_author = last.get("post_author", "someone")
                    orig_url = _post_url(last.get("post_uri", ""))
                    await self._report_to_musing_channel(
                        f"just left a comment on {post_author}'s post\n"
                        f"my reply: {url}\n"
                        f"original: {orig_url}"
                    )
                elif action == "repost":
                    self._update_dashboard(last_repost=ts_now)
                    last = self.history.get("reposts", [])[-1] if self.history.get("reposts") else {}
                    orig_url = _post_url(last.get("post_uri", ""))
                    await self._report_to_musing_channel(
                        f"shared someone's post\n"
                        f"my repost: {url}\n"
                        f"original: {orig_url}"
                    )
                elif action == "post":
                    self._update_dashboard(last_post=ts_now)
                    await self._report_to_musing_channel(
                        f"posted something on bluesky\n{url}"
                    )
            elif not success:
                logger.info("🦋 ⏭ Auto-%s failed after retry: %s", action, status)
                self._update_dashboard(last_failure=f"{action}: {status}")

        except Exception as e:
            logger.error("🦋 Auto-loop error: %s", e, exc_info=True)

    @_auto_loop.before_loop
    async def _before_auto_loop(self) -> None:
        await self.bot.wait_until_ready()

    # ------------------------------------------------------------------
    # Slash command: /soupysky [action]
    # ------------------------------------------------------------------

    @app_commands.command(
        name="soupysky",
        description="Soupy's Bluesky actions — reply, repost, or post",
    )
    @app_commands.describe(
        action="reply (default): comment on a post, repost: quote-post something, post: original post with article",
        url="For 'post' action: provide a specific article URL to post about instead of auto-searching",
    )
    @app_commands.choices(action=[
        app_commands.Choice(name="reply", value="reply"),
        app_commands.Choice(name="repost", value="repost"),
        app_commands.Choice(name="post", value="post"),
    ])
    async def soupysky(
        self,
        interaction: discord.Interaction,
        action: Optional[app_commands.Choice[str]] = None,
        url: Optional[str] = None,
    ) -> None:
        if interaction.user.id not in self._owner_ids():
            await interaction.response.send_message(
                "only the owner can do that.", ephemeral=True
            )
            return

        chosen = action.value if action else "reply"

        # If URL provided without action, assume "post"
        if url and chosen == "reply":
            chosen = "post"

        await interaction.response.defer(ephemeral=True)

        try:
            if chosen == "repost":
                success, status, result_url = await self._run_repost_pipeline()
                label = "repost"
            elif chosen == "post":
                success, status, result_url = await self._run_original_post_pipeline(
                    article_url=url
                )
                label = "post"
            else:
                success, status, result_url = await self._run_engage_pipeline()
                label = "reply"

            if success:
                await interaction.followup.send(
                    f"done. {status}\n{result_url or ''}", ephemeral=True
                )
                # Report to musing channel
                if result_url:
                    if label == "reply":
                        last = self.history.get("comments", [])[-1] if self.history.get("comments") else {}
                        post_author = last.get("post_author", "someone")
                        orig_url = _post_url(last.get("post_uri", ""))
                        await self._report_to_musing_channel(
                            f"just left a comment on {post_author}'s post\n"
                            f"my reply: {result_url}\n"
                            f"original: {orig_url}"
                        )
                    elif label == "repost":
                        last = self.history.get("reposts", [])[-1] if self.history.get("reposts") else {}
                        orig_url = _post_url(last.get("post_uri", ""))
                        await self._report_to_musing_channel(
                            f"shared someone's post\n"
                            f"my repost: {result_url}\n"
                            f"original: {orig_url}"
                        )
                    elif label == "post":
                        await self._report_to_musing_channel(
                            f"posted something on bluesky\n{result_url}"
                        )
            else:
                await interaction.followup.send(
                    f"didn't work out. {status}", ephemeral=True
                )
        except Exception as e:
            logger.error("🦋 Pipeline error: %s", e, exc_info=True)
            await interaction.followup.send(
                f"something broke: {e}", ephemeral=True
            )


# ---------------------------------------------------------------------------
# Extension setup
# ---------------------------------------------------------------------------


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(BlueskyEngageCog(bot))
