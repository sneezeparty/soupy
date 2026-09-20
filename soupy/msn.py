"""
Turn an msn.com link into the article it was syndicated from.

MSN serves a JavaScript shell to anything that isn't a browser. Fetched with
any of the bot's user agents (Facebook's and Slack's crawlers included) the
page comes back ~43 KB with ``<title>MSN</title>``, no ``og:`` tags and no
publish date, so a cross-post gets a link card with no thumbnail
("No og:image or twitter:image found", 2026-09-20) and the freshness check has
nothing to read.

The article underneath belongs to someone else, and MSN's own content endpoint
hands over their URL::

    https://assets.msn.com/content/view/v2/Detail/<locale>/<article id>
    → {"sourceHref": "https://www.kotatv.com/2026/09/20/...", "provider": {...}}

That original has the og:image, the real title and the publish time, and gives
trafilatura something to extract. So the cogs resolve MSN links at discovery
and carry the original from there on.

The endpoint is undocumented, so treat it as best-effort: every failure path
returns the URL that came in, and the caller posts the MSN link exactly as it
does today.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DETAIL_ENDPOINT = "https://assets.msn.com/content/view/v2/Detail/{locale}/{article_id}"
DEFAULT_LOCALE = "en-us"
_ARTICLE_ID_RE = re.compile(r"/a[ar]-([A-Za-z0-9]{6,})", re.I)
_LOCALE_RE = re.compile(r"^/([a-z]{2}-[a-z]{2})/", re.I)

JsonFetcher = Callable[[str, float], Awaitable[Optional[Dict[str, Any]]]]


def is_msn_url(url: str) -> bool:
    host = urlparse(url or "").netloc.lower()
    return host == "msn.com" or host.endswith(".msn.com")


def msn_article_id(url: str) -> Optional[str]:
    """The ``AA2cBYYq`` out of ``.../ar-AA2cBYYq``, or None if the link isn't shaped like an article."""
    if not is_msn_url(url):
        return None
    match = _ARTICLE_ID_RE.search(urlparse(url).path)
    return match.group(1) if match else None


def detail_endpoint(url: str) -> Optional[str]:
    article_id = msn_article_id(url)
    if not article_id:
        return None
    locale_match = _LOCALE_RE.match(urlparse(url).path)
    locale = locale_match.group(1).lower() if locale_match else DEFAULT_LOCALE
    return DETAIL_ENDPOINT.format(locale=locale, article_id=article_id)


async def _fetch_json(endpoint: str, timeout: float) -> Optional[Dict[str, Any]]:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(
            endpoint,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                logger.debug("msn: detail endpoint returned HTTP %d for %s", response.status, endpoint)
                return None
            payload = await response.json(content_type=None)
    return payload if isinstance(payload, dict) else None


async def resolve_article_url(
    url: str,
    *,
    timeout: float = 8.0,
    fetch: Optional[JsonFetcher] = None,
) -> str:
    """The original publisher's URL behind an MSN link, or ``url`` unchanged.

    Unchanged covers everything that isn't a resolvable MSN article: other
    hosts, MSN links without an article id, a failed or slow lookup, and a
    ``sourceHref`` that is missing, relative, or points back at MSN.
    """
    endpoint = detail_endpoint(url)
    if not endpoint:
        return url
    try:
        payload = await (fetch or _fetch_json)(endpoint, timeout)
    except Exception as exc:
        logger.debug("msn: could not resolve %s: %s", url[:80], exc)
        return url
    if not payload:
        return url
    source = str(payload.get("sourceHref") or "").strip()
    if not source.startswith("http") or is_msn_url(source):
        return url
    provider = payload.get("provider")
    name = provider.get("name") if isinstance(provider, dict) else None
    logger.info("📰 msn: resolved to %s%s", source[:90], f" ({name})" if name else "")
    return source


__all__ = ["detail_endpoint", "is_msn_url", "msn_article_id", "resolve_article_url"]
