"""
Shared async URL fetcher with TTL cache and structured extraction.

Used by ``soupy/cogs/search.py``. Keeps trafilatura-first extraction (with a
BeautifulSoup fallback for sites trafilatura can't parse) and honours the URL
processing settings in ``soupy/settings.py`` so search behaves consistently
with the rest of the bot's URL handling.

The cache is in-memory only (process-lifetime, lost on bot restart). That's
fine — it's a latency optimization, not durable state — and it matches the
shape of the ``url_cache`` dict already used in the main bot file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import aiohttp
import trafilatura
from bs4 import BeautifulSoup

from soupy.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Structured page extraction. Any field may be empty/None."""

    url: str
    content: str
    title: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[str] = None  # ISO-ish date string when known
    from_cache: bool = False


# {url: (FetchResult, fetched_at_epoch)}
_cache: Dict[str, tuple] = {}


def _cache_get(url: str) -> Optional[FetchResult]:
    entry = _cache.get(url)
    if not entry:
        return None
    result, fetched_at = entry
    if time.time() - fetched_at > settings.url_cache_ttl_seconds:
        _cache.pop(url, None)
        return None
    # Return a shallow copy with from_cache=True so the caller can log/diagnose.
    return FetchResult(
        url=result.url,
        content=result.content,
        title=result.title,
        source=result.source,
        published_at=result.published_at,
        from_cache=True,
    )


def _cache_put(url: str, result: FetchResult) -> None:
    _cache[url] = (result, time.time())


def _domain_source(url: str) -> Optional[str]:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _trafilatura_extract(html: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (content, title, published_at) using trafilatura with metadata."""
    content = trafilatura.extract(html)
    title = None
    published_at = None
    meta = trafilatura.extract(html, output_format="json", only_with_metadata=False)
    if meta:
        try:
            meta_dict = json.loads(meta)
            title = meta_dict.get("title") or None
            published_at = meta_dict.get("date") or None
        except Exception:
            pass
    return content, title, published_at


def _beautifulsoup_fallback(html: str) -> Optional[str]:
    """Best-effort body extraction when trafilatura returns nothing."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "header", "footer", "iframe"]):
        element.decompose()
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|article|post"))
    )
    if main_content:
        return main_content.get_text(strip=True, separator=" ")
    body = soup.find("body")
    if body:
        return body.get_text(strip=True, separator=" ")
    return None


async def fetch_url(session: aiohttp.ClientSession, url: str) -> Optional[FetchResult]:
    """Fetch + extract a single URL, honouring URL_* settings and the TTL cache.

    Returns ``None`` if the page can't be fetched or yields no usable content.
    The aiohttp session is injected so the caller owns lifecycle.
    """
    if not url:
        return None

    cached = _cache_get(url)
    if cached:
        logger.debug(f"url_fetch cache hit: {url}")
        return cached

    timeout_s = max(1, settings.url_fetch_timeout_ms // 1000)
    max_len = settings.url_max_content_length

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                logger.debug(f"url_fetch HTTP {resp.status} for {url}")
                return None
            html = await resp.text()
    except Exception as e:
        logger.warning(f"url_fetch failed for {url}: {e}")
        return None

    def _extract():
        content, title, published_at = _trafilatura_extract(html)
        if not content:
            content = _beautifulsoup_fallback(html)
        return content, title, published_at

    content, title, published_at = await asyncio.to_thread(_extract)
    if not content:
        logger.debug(f"url_fetch extracted no content from {url}")
        return None

    result = FetchResult(
        url=url,
        content=content.strip()[:max_len],
        title=title,
        source=_domain_source(url),
        published_at=published_at,
    )
    _cache_put(url, result)
    return result


def cache_stats() -> Dict[str, int]:
    """Tiny diagnostic helper — useful for ad-hoc logging / future telemetry."""
    return {"entries": len(_cache)}
