"""
Search cog — ``/soupysearch``.

1. Runs a web search and a news search at once (``ddgs``). For a question
   about recent events the news results, which carry dates, are interleaved
   with the web ones; otherwise they only fill in.
2. Drops results that can't help: dictionary sites, pages with no article
   (TikTok, YouTube), results that share no word with the query, and
   syndicated copies of one story.
3. Asks the LLM to pick the best few, then fetches them in parallel via
   ``soupy.url_fetch``. A pick that fails, is a stub or never mentions the
   query is replaced by the next candidate. MSN links are resolved to the
   original publisher first.
4. Returns a Soupy-voiced summary with inline citations. Each source's age is
   worded in code ("about 5 weeks ago") in both prompts and the Sources list.

Cross-module:

* Reuses the LM Studio client via ``soupy.settings.openai_client``.
* Loads the search persona via ``soupy.prompts.load_prompt("behaviour_search")``.
* Shares URL fetching + the 1hr TTL cache with future cogs via ``soupy.url_fetch``.

Gotchas:

* Dictionary/glossary sites are hard-filtered (see ``_DEFAULT_BLOCKED_DOMAINS``).
  They pollute summaries; ``SEARCH_BLOCKED_DOMAINS`` extends the list.
* Per-user rate limit is 10 searches/min, tracked in-memory in the cog (lost
  on bot restart, which is fine — it's a soft anti-abuse measure, not security).
* ddgs asks the engines in ``SEARCH_BACKENDS`` two at a time, in that order,
  until it has enough results; the whole search gets
  ``SEARCH_BACKEND_TIMEOUT_SECONDS``. An engine that refuses (Brave answers
  429, DuckDuckGo 202) only costs its turn, so those go last.
* Articles are fetched with a browser User-Agent first, then link-preview
  crawlers (``soupy.url_fetch.USER_AGENTS``); LA Times, NYT and AP News refuse
  aiohttp's own.
* Context-length overflow falls back to truncated excerpts AND flips a flag
  so the embed shows a visible "based on excerpts" notice.
* Article fetches run in parallel; the slowest source bounds a round, and
  there are at most ``_FETCH_ROUNDS`` (the picks, then two of replacements).
* The log says what each step did: results kept and dropped (and why), each
  skipped source, and the sources used with their ages.
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import discord
from ddgs import DDGS
from discord import app_commands
from discord.ext import commands

from soupy import prompts as soupy_prompts
from soupy import timeaware
from soupy.llm_gate import llm_turn
from soupy.msn import is_msn_url, resolve_article_url
from soupy.settings import openai_client, settings
from soupy.url_fetch import FetchResult, fetch_url

logger = logging.getLogger(__name__)

# ddgs warns on every call when a backend name is unknown, and falls back to
# 'auto' anyway. We pass a curated comma list (settings.search_backends), so a
# stray typo in env config shouldn't produce per-query log spam. Real engine
# failures still raise + propagate via SearchBackendError below.
logging.getLogger("ddgs.ddgs").setLevel(logging.ERROR)

# Hosts we never want as sources in summarization-style search.
# Dictionaries and definition / glossary sites pollute the LLM summary
# (Soupy isn't a dictionary lookup tool — that's what direct chat is for).
# Users can extend this list via the SEARCH_BLOCKED_DOMAINS env var
# (comma-separated, additive). Match is on host or any parent domain.
_DEFAULT_BLOCKED_DOMAINS = (
    "merriam-webster.com",
    "learnersdictionary.com",
    "dictionary.com",
    "thesaurus.com",
    "wiktionary.org",
    "vocabulary.com",
    "wordnik.com",
    "lexico.com",
    "urbandictionary.com",
    "thefreedictionary.com",
    "collinsdictionary.com",
    "oxfordlearnersdictionaries.com",
    "dictionary.cambridge.org",
    "yourdictionary.com",
    "your-dictionary.com",
    "definitions.net",
    "wordreference.com",
    "etymonline.com",
    "ldoceonline.com",
    "macmillandictionary.com",
    "powerthesaurus.org",
)

# Triggers a "prefer fresh sources" instruction in the selection + summary
# prompts. Year tokens (2024-2030) are matched separately so the bot stays
# recency-aware on queries like "biden 2026".
_RECENCY_KEYWORDS = {
    "news",
    "latest",
    "recent",
    "today",
    "tonight",
    "now",
    "current",
    "currently",
    "breaking",
    "live",
    "yesterday",
    "this week",
    "this month",
    "this year",
}
_YEAR_RE = re.compile(r"\b20[2-3][0-9]\b")

# Matches markdown links — used to verify the LLM actually included citations.
_CITATION_RE = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")

# ddgs news engines, asked two at a time in this order. Bing News answers with
# dated results from named publishers (often as MSN links, resolved before
# fetching); DuckDuckGo's and Yahoo's news answered nothing in October 2026.
_NEWS_BACKENDS = "bing,yahoo,duckduckgo"

# Pages with no article to read. Their search results name the topic, then the
# fetch comes back empty (TikTok) or with a one-line description (YouTube).
_NO_ARTICLE_HOSTS = {
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "pinterest.com",
    "news.google.com",  # a story page is other outlets' headlines
}

# Words that say nothing about a query's topic. What is left are the words a
# result or an article has to mention to count as being about it.
_STOPWORDS = {
    "the", "and", "for", "with", "what", "when", "where", "which", "who", "why", "how",
    "are", "was", "were", "about", "from", "that", "this", "does", "did", "can", "will",
    "into", "over", "after", "before", "its", "they", "them", "their", "there", "have",
    "has", "had", "not", "but", "you", "your", "any", "all", "more", "most", "some",
    "than", "then", "also", "just", "like", "get", "got", "new", "best", "good",
}  # fmt: skip

# The model chooses from at most this many results (title, link and preview each).
_POOL_MAX = 12
_PREVIEW_CHARS = 300
# A fetched page shorter than this is a stub, a cookie wall or a video blurb.
_MIN_ARTICLE_CHARS = 300
# Fetch rounds: the picks, then replacements for the ones that failed, twice.
_FETCH_ROUNDS = 3

# Discord embed limits we rely on.
_EMBED_DESC_MAX = 4000  # actual limit is 4096; keep slack for the degraded prefix
_EMBED_FIELD_MAX = 1024


def _build_blocked_domains() -> set:
    """Default blocklist + any additions from settings.search_blocked_domains."""
    blocked = {d.lower() for d in _DEFAULT_BLOCKED_DOMAINS}
    for d in settings.search_blocked_domains:
        d = d.strip().lower().lstrip(".")
        if d:
            blocked.add(d)
    return blocked


def _is_blocked_url(url: str, blocked: set) -> bool:
    """True if URL's host (or any parent domain) is in the blocklist."""
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not host:
        return False
    if host.startswith("www."):
        host = host[4:]
    for blocked_domain in blocked:
        if host == blocked_domain or host.endswith("." + blocked_domain):
            return True
    return False


def _is_recency_query(query: str) -> bool:
    """True if the query looks like it wants fresh information."""
    if not query:
        return False
    q = query.lower()
    if _YEAR_RE.search(q):
        return True
    for kw in _RECENCY_KEYWORDS:
        # Word-boundary match so "newsletter" doesn't trigger on "news".
        if re.search(rf"\b{re.escape(kw)}\b", q):
            return True
    return False


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _query_terms(query: str) -> List[str]:
    """The words of a query that name its topic (no stopwords, no "latest news")."""
    terms = []
    for word in _words(query):
        if len(word) >= 3 and word not in _STOPWORDS and word not in _RECENCY_KEYWORDS and word not in terms:
            terms.append(word)
    return terms


def _mentions(text: str, terms: List[str], need: int = 1) -> bool:
    """True if ``text`` mentions ``need`` of ``terms``; with no terms, everything counts.

    A term longer than five letters matches on its first five, so "missile"
    finds "missiles" and "volcano" finds "volcanic". A shorter one matches
    itself or its plural ("tide", "tides").
    """
    if not terms:
        return True
    words = set(_words(text))
    found = 0
    for term in terms:
        if len(term) > 5:
            hit = any(word.startswith(term[:5]) for word in words)
        else:
            hit = term in words or term + "s" in words or (term.endswith("s") and term[:-1] in words)
        found += hit
        if found >= need:
            return True
    return False


def _article_need(terms: List[str]) -> int:
    """How many query words an article must mention: two once the query has three.

    One is enough for a search result's title and snippet, but an article is
    long, and one common word (the "port" of a town's name) turns up in stories
    about something else.
    """
    return 2 if len(terms) >= 3 else 1


def _news_query(query: str) -> str:
    """The query without its recency words; Bing News finds less with "latest news" in it."""
    stripped = query
    for kw in sorted(_RECENCY_KEYWORDS, key=len, reverse=True):
        stripped = re.sub(rf"\b{re.escape(kw)}\b", " ", stripped, flags=re.IGNORECASE)
    stripped = " ".join(stripped.split())
    return stripped or query


def _title_key(title: str) -> str:
    """Syndicated copies of one story (an AP piece in five papers) share this."""
    return " ".join(_words(title))[:50]


def _parse_day(value) -> Optional[date]:
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value or "").strip())
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _age(value, today: date) -> str:
    """``about 5 weeks ago`` for a date string, or "" when there is no date in it.

    Ages are worded here because the model is unreliable at date arithmetic.
    """
    day = _parse_day(value)
    if day is None:
        return ""
    return timeaware.describe_age_days(max(0, (today - day).days), day.year)


# Living pages: whatever date they carry (Wikipedia's is the article's creation)
# says nothing about how current they are.
_UNDATED_HOSTS = {"wikipedia.org"}


def _published(meta: Dict, res: Optional[FetchResult]) -> str:
    """A source's publish date as found on the page or in the search result, or ""."""
    if _is_blocked_url(meta.get("href", ""), _UNDATED_HOSTS):
        return ""
    return str((res.published_at if res else None) or meta.get("date") or "")


def _source_title(meta: Dict, res: FetchResult) -> str:
    """The search result's title, or the page's own where the result's is broken.

    A result's title can be glued from several results ("Pier reopens ...Harbor
    board ..."), a bare URL, or cut off ("after the storm ..."). The
    page's title is no better in general (one tides page calls itself
    "Winterizing Your Boat"), so it only replaces a cut-off title it continues.
    """
    page = " ".join((res.title or "").split())
    listed = " ".join((meta.get("title") or "").split())
    page_ok = len(page.split()) >= 3
    broken = not listed or " " not in listed or re.search(r"(\.\.\.|…)\S", listed)
    cut_off = listed.endswith(("...", "…"))
    if page_ok and (broken or (cut_off and page.lower().startswith(" ".join(listed.lower().split()[:3])))):
        title = page
    else:
        title = listed or page or "Untitled"
    return title[:150]


def _from_news(item: Dict) -> Dict:
    """A ddgs news result in the shape of a text result."""
    source = re.sub(r"\s+on MSN$", "", str(item.get("source") or "")).strip()
    return {
        "title": item.get("title") or "",
        "body": item.get("body") or "",
        "href": item.get("url") or "",
        "date": item.get("date") or "",
        "source": source,
    }


def _host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _channel_hint(interaction: discord.Interaction) -> Optional[str]:
    """One-line channel context for the selection prompt. Returns None for DMs."""
    ch = getattr(interaction, "channel", None)
    if ch is None:
        return None
    name = getattr(ch, "name", None)
    topic = getattr(ch, "topic", None)
    if not name and not topic:
        return None
    parts = []
    if name:
        parts.append(f"#{name}")
    if topic:
        parts.append(topic.strip()[:160])
    return " — ".join(parts) if parts else None


client = openai_client()


async def async_chat_completion(*args, **kwargs):
    """Wraps the OpenAI chat completion in an async context (one LM Studio call at a time; see soupy.llm_gate)."""
    async with llm_turn():
        return await asyncio.to_thread(client.chat.completions.create, *args, **kwargs)


class SearchBackendError(Exception):
    """Raised when every DuckDuckGo search backend failed or timed out.

    Distinct from an empty result list, which means the query genuinely had no
    hits. Lets ``/soupysearch`` show a network-error message instead of "no
    results found".
    """


class SearchCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.search_rate_limits = defaultdict(list)
        self.MAX_SEARCHES_PER_MINUTE = 10
        self.session = aiohttp.ClientSession()

    async def cog_unload(self):
        if hasattr(self, "session"):
            await self.session.close()

    async def is_rate_limited(self, user_id: int) -> bool:
        current_time = time.time()
        search_times = self.search_rate_limits.get(user_id, [])
        search_times = [t for t in search_times if current_time - t < 60]
        self.search_rate_limits[user_id] = search_times
        if len(search_times) >= self.MAX_SEARCHES_PER_MINUTE:
            return True
        self.search_rate_limits[user_id].append(current_time)
        return False

    async def perform_text_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Run a DDG text search via ``ddgs``, honouring ``SEARCH_BACKENDS``.

        ddgs 9.x does its own multi-engine fan-out internally; the cog passes a
        curated comma-list (``settings.search_backends``) so news-style queries
        skip the knowledge-base engines. Empty backend config drops the kwarg
        and lets ddgs use its ``auto`` mode.
        """
        start = time.time()
        timeout = settings.search_backend_timeout_seconds
        backends = settings.search_backends
        kwargs: Dict = {"query": query, "max_results": max_results}
        if backends:
            kwargs["backend"] = backends
        label = backends or "auto"

        def _run_text(call_kwargs: Dict) -> List[Dict]:
            with DDGS() as ddg:
                return list(ddg.text(**call_kwargs))

        try:
            results_list = await asyncio.wait_for(
                asyncio.to_thread(_run_text, kwargs),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            logger.warning(f"Search timed out after {timeout}s (backends={label})")
            raise SearchBackendError("DuckDuckGo search timed out") from e
        except Exception as e:
            logger.error(f"Search failed (backends={label}): {e}")
            raise SearchBackendError(f"DuckDuckGo search failed: {e}") from e

        logger.info(
            f"Search (backends={label}) returned {len(results_list)} results in "
            f"{round(time.time() - start, 2)}s"
        )
        return results_list

    async def perform_news_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """ddgs news results, in the shape of text results. Nothing found is ``[]``."""
        start = time.time()

        def _run_news() -> List[Dict]:
            with DDGS() as ddg:
                return list(ddg.news(query=query, max_results=max_results, backend=_NEWS_BACKENDS))

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_run_news), timeout=settings.search_backend_timeout_seconds
            )
        except Exception as e:  # ddgs raises when nothing is found, too
            logger.info(f"News search (backends={_NEWS_BACKENDS}) found nothing: {e}")
            return []
        logger.info(
            f"News search (backends={_NEWS_BACKENDS}) returned {len(results)} results in "
            f"{round(time.time() - start, 2)}s"
        )
        return [_from_news(r) for r in results]

    async def gather_candidates(self, query: str, recency_query: bool, pool_size: int) -> List[Dict]:
        """Web and news results the model can choose from, junk removed.

        News goes first for a question about recent events, interleaved with
        web results; otherwise it only fills in when the web search comes up
        short. Raises ``SearchBackendError`` only when the web search failed
        and the news search found nothing either.
        """
        text_outcome, news = await asyncio.gather(
            self.perform_text_search(query, max_results=pool_size),
            self.perform_news_search(_news_query(query), max_results=pool_size),
            return_exceptions=True,
        )
        if isinstance(news, BaseException):
            news = []
        if isinstance(text_outcome, BaseException):
            if not news:
                if isinstance(text_outcome, SearchBackendError):
                    raise text_outcome
                raise SearchBackendError(str(text_outcome)) from text_outcome
            text = []
        else:
            text = text_outcome

        if recency_query:
            ordered = []
            for i in range(max(len(news), len(text))):
                ordered += news[i : i + 1] + text[i : i + 1]
        else:
            ordered = text + news

        terms = _query_terms(query)
        blocked = _build_blocked_domains()
        kept: List[Dict] = []
        seen_urls, seen_titles = set(), set()
        dropped = {"blocked": 0, "no article": 0, "off topic": 0, "duplicate": 0}
        for r in ordered:
            url = r.get("href") or ""
            title_key = _title_key(r.get("title") or "")
            if not url:
                continue
            if _is_blocked_url(url, blocked):
                dropped["blocked"] += 1
            elif _is_blocked_url(url, _NO_ARTICLE_HOSTS):
                dropped["no article"] += 1
            elif not _mentions(f"{r.get('title', '')} {r.get('body', '')}", terms):
                dropped["off topic"] += 1
            elif url in seen_urls or (title_key and title_key in seen_titles):
                dropped["duplicate"] += 1
            else:
                seen_urls.add(url)
                seen_titles.add(title_key)
                kept.append(r)
        # Without a recency question, news only fills the places junk left.
        kept = kept[: _POOL_MAX if recency_query else min(_POOL_MAX, pool_size)]
        dropped_text = ", ".join(f"{n} {why}" for why, n in dropped.items() if n) or "none"
        logger.info(f"🔍 {len(text)} web + {len(news)} news results; kept {len(kept)} (dropped: {dropped_text})")
        return kept

    async def select_articles(
        self,
        search_results: List[Dict],
        target_count: int,
        recency_query: bool,
        channel_hint: Optional[str],
    ) -> List[Dict]:
        """Select the most relevant articles using the LLM, with recency + channel hints."""
        try:
            if len(search_results) <= target_count:
                return search_results

            today = timeaware.now_local().date()
            formatted_results: List[Dict] = []
            index_mapping: List[int] = []
            for idx, result in enumerate(search_results):
                if not all(key in result for key in ["title", "body", "href"]):
                    continue
                raw_date = result.get("date") or result.get("published") or ""
                age = _age(raw_date, today)
                formatted_results.append(
                    {
                        "title": result["title"],
                        "preview": result.get("body", "")[:_PREVIEW_CHARS],
                        "url": result["href"],
                        "date": f"{str(raw_date)[:10]} ({age})" if age else "",
                        "source": result.get("source") or "",
                    }
                )
                index_mapping.append(idx)

            if not formatted_results:
                return search_results[:target_count]

            criteria = [
                "Relevance to the topic",
                "Information quality and depth",
                "Source credibility",
                "Content uniqueness",
            ]
            if recency_query:
                criteria.append("Freshness — prefer dated, recent articles over undated or old ones")

            prompt_lines = [
                f"Select the {target_count} most informative and relevant articles from these search results.",
                "Consider:",
            ]
            for i, c in enumerate(criteria, 1):
                prompt_lines.append(f"{i}. {c}")
            if channel_hint:
                prompt_lines.append(f"\nContext — the user is in this channel: {channel_hint}")
            prompt_lines.append(
                f"\nRespond ONLY with the numbers (0-based) of the {target_count} best articles, "
                "separated by spaces.\n"
            )
            prompt = "\n".join(prompt_lines) + "\n"

            for i, result in enumerate(formatted_results):
                prompt += f"[{i}] {result['title']}\n"
                prompt += f"URL: {result['url']}\n"
                if result["source"]:
                    prompt += f"Publisher: {result['source']}\n"
                if result["date"]:
                    prompt += f"Date: {result['date']}\n"
                prompt += f"Preview: {result['preview']}\n\n"

            response = await async_chat_completion(
                model=settings.local_chat,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that selects the most relevant and informative "
                            "articles for a search summary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.search_select_temperature,
                max_tokens=64,
            )

            try:
                raw = response.choices[0].message.content.strip()
                mapped_indices: List[int] = []
                for tok in re.findall(r"\d+", raw):
                    i = int(tok)
                    if 0 <= i < len(formatted_results) and index_mapping[i] not in mapped_indices:
                        mapped_indices.append(index_mapping[i])
                # Short of picks: the rest come in search order.
                for i in range(len(search_results)):
                    if len(mapped_indices) >= target_count:
                        break
                    if i not in mapped_indices:
                        mapped_indices.append(i)
                return [search_results[i] for i in mapped_indices[:target_count]]
            except Exception:
                logger.warning("Failed to parse article selection response")

            return search_results[:target_count]

        except Exception as e:
            logger.error(f"Error selecting articles: {e}")
            return search_results[:target_count]

    async def _fetch_one(self, article: Dict) -> Tuple[Dict, Optional[FetchResult]]:
        """One result's page. The result comes back with its MSN link resolved."""
        url = article.get("href", "")
        if is_msn_url(url):
            # MSN serves a script shell; the original publisher has the article.
            original = await resolve_article_url(url)
            if original != url:
                # "MSN" isn't the publisher; the original's host names it instead.
                source = "" if (article.get("source") or "").lower() == "msn" else article.get("source")
                article = {**article, "href": original, "source": source}
                url = original
        try:
            return article, await fetch_url(self.session, url)
        except Exception as e:
            logger.warning(f"Article fetch raised: {e}")
            return article, None

    async def fetch_sources(
        self, ranked: List[Dict], target_count: int, terms: List[str]
    ) -> List[Tuple[Dict, FetchResult]]:
        """Fetch the best results until ``target_count`` have an article about the query.

        ``ranked`` is the model's picks followed by the other candidates, and
        each round fetches in parallel. A page that fails, is a stub, or never
        mentions the query (a paywall's terms of use) is replaced by the next
        candidate. Sources keep their ranked order, so citation numbers do too.
        """
        fetched: List[Tuple[Dict, FetchResult]] = []
        skipped: List[str] = []
        start = 0
        for _ in range(_FETCH_ROUNDS):
            need = target_count - len(fetched)
            if need <= 0 or start >= len(ranked):
                break
            batch = ranked[start : start + need]
            start += len(batch)
            for meta, res in await asyncio.gather(*(self._fetch_one(a) for a in batch)):
                host = _host(meta.get("href", ""))
                content = (res.content or "") if res else ""
                if res is None:
                    skipped.append(f"{host} (no page)")
                elif len(content) < _MIN_ARTICLE_CHARS:
                    skipped.append(f"{host} (stub)")
                # The text only: a paywall's terms-of-use page keeps the story's title.
                elif not _mentions(content, terms, _article_need(terms)):
                    skipped.append(f"{host} (off topic)")
                else:
                    fetched.append((meta, res))
        if skipped:
            logger.info(f"🔍 Skipped sources: {', '.join(skipped)}")
        return fetched

    def _build_system_message(self, today_iso: str, recency_query: bool) -> str:
        rules = (
            f"Today's date is {today_iso}.\n\n"
            "Citation rules:\n"
            "1. Every significant claim must include a [Source Name](URL) citation in Discord markdown.\n"
            "2. Naturally integrate citations into your prose — do not bolt them on at the end.\n"
            "3. If sources disagree, say so and cite each side.\n"
        )
        if recency_query:
            rules += (
                "4. This question is time-sensitive. Each article says how long ago it was published. "
                "Lead with the freshest, and say so when a source is more than a few months old.\n"
            )
        return f"{soupy_prompts.load_prompt('behaviour_search', fallback='')}\n\n{rules}"

    def _build_content_prompt(
        self,
        query: str,
        articles: List[Tuple[Dict, FetchResult]],
        per_article_limit: int,
        today: date,
    ) -> str:
        parts = [
            f"Search Query: {query}",
            "",
            "Summarize these articles in Soupy's voice. Cite every significant point.",
            "",
        ]
        for i, (meta, res) in enumerate(articles, 1):
            title = _source_title(meta, res)
            source = meta.get("source") or res.source or "Unknown Source"
            url = meta.get("href", "")
            raw_date = _published(meta, res)
            age = _age(raw_date, today)
            published = f"{str(raw_date)[:10]} ({age})" if age else "unknown"
            body = (res.content or "")[:per_article_limit]
            parts.append(f"Article {i}:")
            parts.append(f"Title: {title}")
            parts.append(f"Source: {source}")
            parts.append(f"URL: {url}")
            parts.append(f"Published: {published}")
            parts.append(f"Content: {body}")
            parts.append("")
        return "\n".join(parts)

    async def generate_final_response(
        self,
        query: str,
        articles: List[Tuple[Dict, FetchResult]],
        recency_query: bool,
    ) -> Tuple[str, bool, bool]:
        """Produce the Soupy-voiced summary.

        Returns ``(text, degraded, citations_missing)``:
          * ``degraded``  — True if we had to retry with shorter excerpts due to
            context-length errors.
          * ``citations_missing`` — True if the final text contains no markdown
            links despite a citation retry.
        """
        today = timeaware.now_local().date()
        system_message = self._build_system_message(today.isoformat(), recency_query)

        full_limit = settings.url_max_content_length
        prompt = self._build_content_prompt(query, articles, full_limit, today)
        degraded = False

        try:
            response = await async_chat_completion(
                model=settings.local_chat,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.search_summary_temperature,
                max_tokens=1500,
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            if "context length" in str(e).lower():
                logger.warning("Context length exceeded, retrying with shorter excerpts")
                degraded = True
                short_limit = max(200, full_limit // 4)
                prompt = self._build_content_prompt(query, articles, short_limit, today)
                response = await async_chat_completion(
                    model=settings.local_chat,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=settings.search_summary_temperature,
                    max_tokens=1000,
                )
                text = response.choices[0].message.content.strip()
            else:
                raise

        # Citation guard — one retry with an explicit nudge if the model
        # produced a citation-free wall of text.
        citations_missing = False
        if not _CITATION_RE.search(text):
            logger.info("Summary returned no citations — retrying with strict nudge")
            try:
                strict = (
                    prompt
                    + "\n\nIMPORTANT: Your previous answer had no citations. Rewrite it so that "
                    "every significant claim includes a [Source Name](URL) link in Discord markdown."
                )
                response = await async_chat_completion(
                    model=settings.local_chat,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": strict},
                    ],
                    temperature=settings.search_summary_temperature,
                    max_tokens=1500,
                )
                retry_text = response.choices[0].message.content.strip()
                if _CITATION_RE.search(retry_text):
                    text = retry_text
                else:
                    citations_missing = True
                    text = retry_text or text
            except Exception as e:
                logger.error(f"Citation retry failed: {e}")
                citations_missing = True

        return text, degraded, citations_missing

    def _build_embeds(
        self,
        query: str,
        summary: str,
        articles: List[Tuple[Dict, FetchResult]],
        elapsed: float,
        fetched_count: int,
        target_count: int,
        degraded: bool,
        citations_missing: bool,
        today: Optional[date] = None,
    ) -> List[discord.Embed]:
        """Render the response as one (or rarely multiple) embeds.

        Sources go in their own field rather than appended to the description
        so users can scan them at a glance.
        """
        prefix_parts = []
        if degraded:
            prefix_parts.append("*(based on excerpts — articles too long to fully analyze)*")
        if citations_missing:
            prefix_parts.append("*(no inline citations returned — see Sources below)*")
        prefix = "\n".join(prefix_parts)
        body = (prefix + "\n\n" + summary).strip() if prefix else summary

        # Chunk only if the description overflows.
        if len(body) <= _EMBED_DESC_MAX:
            chunks = [body]
        else:
            chunks = [body[i : i + _EMBED_DESC_MAX] for i in range(0, len(body), _EMBED_DESC_MAX)]

        # Build the Sources field value(s).
        source_lines = []
        for i, (meta, res) in enumerate(articles, 1):
            title = _source_title(meta, res).replace("[", "(").replace("]", ")")
            url = meta.get("href", "").strip()
            if not url:
                continue
            line = f"{i}. [{title}]({url})"
            age = _age(_published(meta, res), today) if today else ""
            if age:
                line += f" · {age}"
            source_lines.append(line)
        sources_text = "\n".join(source_lines)

        footer_bits = [f"{elapsed:.2f}s", f"sources: {fetched_count}/{target_count}"]
        if degraded:
            footer_bits.append("excerpts mode")
        footer_text = " · ".join(footer_bits)

        embeds: List[discord.Embed] = []
        for i, chunk in enumerate(chunks):
            title = f"🔍 {query}"
            if len(chunks) > 1:
                title += f" (Part {i+1}/{len(chunks)})"
            embed = discord.Embed(
                title=title[:256],
                description=chunk,
                color=discord.Color.green(),
            )
            if i == len(chunks) - 1 and sources_text:
                # Source list lives on the last embed. If sources overflow the
                # 1024-char field cap, paginate into multiple fields.
                if len(sources_text) <= _EMBED_FIELD_MAX:
                    embed.add_field(name="Sources", value=sources_text, inline=False)
                else:
                    buf, n = "", 1
                    for line in source_lines:
                        candidate = (buf + "\n" + line).strip()
                        if len(candidate) > _EMBED_FIELD_MAX:
                            embed.add_field(
                                name=f"Sources ({n})" if n > 1 else "Sources",
                                value=buf,
                                inline=False,
                            )
                            buf, n = line, n + 1
                        else:
                            buf = candidate
                    if buf:
                        embed.add_field(
                            name=f"Sources ({n})" if n > 1 else "Sources",
                            value=buf,
                            inline=False,
                        )
            embed.set_footer(text=footer_text)
            embeds.append(embed)
        return embeds

    @app_commands.command(
        name="soupysearch",
        description="Searches the web (and the news, for recent events) and answers with cited sources.",
    )
    @app_commands.describe(query="The search query.")
    async def search_command(self, interaction: discord.Interaction, query: str):
        start_time = time.time()
        logger.info(f"🔍 Search requested by {interaction.user}: '{query}'")

        if await self.is_rate_limited(interaction.user.id):
            await interaction.response.send_message(
                "⚠️ You are searching too quickly. Please wait a moment.", ephemeral=True
            )
            return

        await interaction.response.defer()

        try:
            target_count = settings.search_results_per_query
            recency_query = _is_recency_query(query)
            channel_hint = _channel_hint(interaction)
            if recency_query:
                logger.info(f"🔍 Recency-seeking query detected: '{query}'")

            # Fetch a wider pool so the LLM has options after the junk is filtered out.
            pool_size = max(target_count * 2, 10)
            try:
                candidates = await self.gather_candidates(query, recency_query, pool_size)
            except SearchBackendError:
                await interaction.followup.send(
                    "❌ Search failed: none of the search engines answered. Try again in a moment.",
                    ephemeral=True,
                )
                return

            if not candidates:
                await interaction.followup.send(
                    "❌ Nothing in the results was about that. Try other words.", ephemeral=True
                )
                return

            selected_results = await self.select_articles(
                candidates,
                target_count=target_count,
                recency_query=recency_query,
                channel_hint=channel_hint,
            )
            ranked = selected_results + [c for c in candidates if c not in selected_results]

            fetched = await self.fetch_sources(ranked, target_count, _query_terms(query))
            if not fetched:
                await interaction.followup.send(
                    "❌ Couldn't read any of the results. The sites may be blocking the bot or down.",
                    ephemeral=True,
                )
                return
            today = timeaware.now_local().date()
            logger.info(
                f"🔍 Sources {len(fetched)}/{target_count}: "
                + ", ".join(
                    f"{_host(meta.get('href', ''))} ({_age(_published(meta, res), today) or 'undated'})"
                    for meta, res in fetched
                )
            )

            summary, degraded, citations_missing = await self.generate_final_response(
                query, fetched, recency_query
            )

            elapsed = time.time() - start_time
            embeds = self._build_embeds(
                query=query,
                summary=summary,
                articles=fetched,
                elapsed=elapsed,
                fetched_count=len(fetched),
                target_count=target_count,
                degraded=degraded,
                citations_missing=citations_missing,
                today=today,
            )

            for i, embed in enumerate(embeds):
                await interaction.followup.send(embed=embed)
                if i < len(embeds) - 1:
                    await asyncio.sleep(1)

            logger.info(
                f"✅ Search completed for {interaction.user} in {elapsed:.2f}s "
                f"(degraded={degraded}, citations_missing={citations_missing})"
            )

        except Exception as e:
            logger.error(f"❌ Error in search command: {e}")
            await interaction.followup.send(f"❌ An error occurred: {str(e)}", ephemeral=True)


async def setup(bot):
    """Setup function for loading the cog."""
    await bot.add_cog(SearchCog(bot))
