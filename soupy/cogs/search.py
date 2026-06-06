"""
Search cog — ``/soupysearch``.

Runs a DuckDuckGo text query, drops dictionary/definition sites, asks the LLM
to pick the top results, fetches each article in parallel via the shared
``soupy.url_fetch`` helper (cached, settings-aware), and returns a Soupy-voiced
summary with inline citations.

Cross-module:

* Reuses the LM Studio client via ``soupy.settings.openai_client``.
* Loads the search persona via ``soupy.prompts.load_prompt("behaviour_search")``.
* Shares URL fetching + the 1hr TTL cache with future cogs via ``soupy.url_fetch``.

Gotchas:

* Dictionary/glossary sites are hard-filtered (see ``_DEFAULT_BLOCKED_DOMAINS``).
  They pollute summaries; ``SEARCH_BLOCKED_DOMAINS`` extends the list.
* Per-user rate limit is 10 searches/min, tracked in-memory in the cog (lost
  on bot restart, which is fine — it's a soft anti-abuse measure, not security).
* DuckDuckGo backend rotates (api → html → lite → default) with a per-attempt
  timeout from ``SEARCH_BACKEND_TIMEOUT_SECONDS`` — public backends vary.
* Context-length overflow falls back to truncated excerpts AND flips a flag
  so the embed shows a visible "based on excerpts" notice.
* Article fetches run in parallel; the slowest source bounds wall-clock, not
  the sum of all fetches.
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import discord
from ddgs import DDGS
from discord import app_commands
from discord.ext import commands

from soupy import prompts as soupy_prompts
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
    """Wraps the OpenAI chat completion in an async context."""
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
        curated comma-list (defaults to ``brave,duckduckgo,mojeek,yahoo,yandex``)
        so news-style queries skip the knowledge-base engines. Empty backend
        config drops the kwarg and lets ddgs use its ``auto`` mode.
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

            formatted_results: List[Dict] = []
            index_mapping: List[int] = []
            for idx, result in enumerate(search_results):
                if not all(key in result for key in ["title", "body", "href"]):
                    continue
                formatted_results.append(
                    {
                        "title": result["title"],
                        "preview": result.get("body", "")[:500],
                        "url": result["href"],
                        # DDG sometimes returns "date" or "published" — surface either.
                        "date": result.get("date") or result.get("published") or "",
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
                indices = [int(tok) for tok in raw.split() if tok.lstrip("-").isdigit()]
                valid_formatted = [i for i in indices if 0 <= i < len(formatted_results)]
                mapped_indices = [index_mapping[i] for i in valid_formatted]
                if len(mapped_indices) >= target_count:
                    return [search_results[i] for i in mapped_indices[:target_count]]
            except Exception:
                logger.warning("Failed to parse article selection response")

            return search_results[:target_count]

        except Exception as e:
            logger.error(f"Error selecting articles: {e}")
            return search_results[:target_count]

    async def fetch_articles_parallel(self, articles: List[Dict]) -> List[Tuple[Dict, FetchResult]]:
        """Fetch every selected article in parallel, dropping failures.

        Returns a list of ``(article_meta, FetchResult)`` pairs in the original
        order so citation numbering stays stable.
        """
        async def _one(article: Dict) -> Optional[Tuple[Dict, FetchResult]]:
            url = article.get("href", "")
            res = await fetch_url(self.session, url)
            if res is None:
                return None
            return (article, res)

        results = await asyncio.gather(
            *(_one(a) for a in articles),
            return_exceptions=True,
        )
        out: List[Tuple[Dict, FetchResult]] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Article fetch raised: {r}")
                continue
            if r is None:
                continue
            out.append(r)
        cache_hits = sum(1 for _, fr in out if fr.from_cache)
        if cache_hits:
            logger.info(f"🔍 URL cache: {cache_hits}/{len(out)} hits")
        return out

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
                "4. This question is time-sensitive. Prefer the freshest sources; if a source is older "
                "than a few months, say so explicitly.\n"
            )
        return f"{soupy_prompts.load_prompt('behaviour_search', fallback='')}\n\n{rules}"

    def _build_content_prompt(
        self,
        query: str,
        articles: List[Tuple[Dict, FetchResult]],
        per_article_limit: int,
    ) -> str:
        parts = [
            f"Search Query: {query}",
            "",
            "Summarize these articles in Soupy's voice. Cite every significant point.",
            "",
        ]
        for i, (meta, res) in enumerate(articles, 1):
            title = meta.get("title") or res.title or "Untitled"
            source = res.source or meta.get("source") or "Unknown Source"
            url = meta.get("href", "")
            date = res.published_at or meta.get("date") or "unknown"
            body = (res.content or "")[:per_article_limit]
            parts.append(f"Article {i}:")
            parts.append(f"Title: {title}")
            parts.append(f"Source: {source}")
            parts.append(f"URL: {url}")
            parts.append(f"Published: {date}")
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
        today_iso = datetime.now().strftime("%Y-%m-%d")
        system_message = self._build_system_message(today_iso, recency_query)

        full_limit = settings.url_max_content_length
        prompt = self._build_content_prompt(query, articles, full_limit)
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
                prompt = self._build_content_prompt(query, articles, short_limit)
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
            title = (meta.get("title") or res.title or "Untitled").strip()
            url = meta.get("href", "").strip()
            if not url:
                continue
            line = f"{i}. [{title}]({url})"
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
        description="Performs a DuckDuckGo search and returns a comprehensive answer based on the results.",
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

            # Fetch a wider pool so the LLM has options after blocklist filtering.
            pool_size = max(target_count * 2, 10)
            try:
                initial_results = await self.perform_text_search(query, max_results=pool_size)
            except SearchBackendError:
                await interaction.followup.send(
                    "❌ Search failed — couldn't reach DuckDuckGo. Try again in a moment.",
                    ephemeral=True,
                )
                return
            logger.info(f"Initial search results count: {len(initial_results)} for query='{query}'")

            if not initial_results:
                await interaction.followup.send("❌ No results found.", ephemeral=True)
                return

            blocked_domains = _build_blocked_domains()
            kept = []
            blocked_count = 0
            for r in initial_results:
                if _is_blocked_url(r.get("href", ""), blocked_domains):
                    blocked_count += 1
                    logger.debug(f"🔍 Blocked-domain filter dropped: {r.get('href', '')}")
                    continue
                kept.append(r)
            if blocked_count:
                logger.info(
                    f"🔍 Blocked-domain filter dropped {blocked_count} of {len(initial_results)} results"
                )
            initial_results = kept

            if not initial_results:
                await interaction.followup.send(
                    "❌ All results were dictionary/definition sites. Try a more specific query.",
                    ephemeral=True,
                )
                return

            selected_results = await self.select_articles(
                initial_results,
                target_count=target_count,
                recency_query=recency_query,
                channel_hint=channel_hint,
            )

            fetched = await self.fetch_articles_parallel(selected_results)
            if not fetched:
                await interaction.followup.send(
                    "❌ Couldn't extract content from any of the selected sources. They may be blocking "
                    "scrapers or returning errors.",
                    ephemeral=True,
                )
                return

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
                target_count=len(selected_results),
                degraded=degraded,
                citations_missing=citations_missing,
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
