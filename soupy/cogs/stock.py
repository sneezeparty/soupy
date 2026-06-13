"""
Stock cog — ``/soupystock``.

Looks up the current quote for a stock ticker or company name via Finnhub's
free REST API. Resolves company-name queries through ``/search``, then pulls
``/quote`` + ``/stock/profile2`` + ``/stock/metric`` in parallel and builds
an embed with price, day change, day range, 52-week range, and market cap.

If Finnhub's ``/stock/candle`` endpoint is reachable on the active tier,
a tiny intraday sparkline PNG is attached as well; on the free tier the
endpoint typically returns 403 and the sparkline is silently skipped.

Gotchas:

* The Finnhub webhook secret is NOT used by this cog — webhooks are for
  pushed events. The slash command only needs ``FINNHUB_API_KEY``.
* Per-user rate limit is 5 lookups/min; Finnhub free tier is 60/min total.
* ``profile2.marketCapitalization`` is in **millions** of USD.
* matplotlib is imported lazily inside ``_fetch_sparkline_png`` so the cog
  still loads even if the optional dep is missing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from soupy.settings import settings

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=10)

# Ticker shape: 1-8 chars, A-Z/0-9 plus '.' and '-' (covers BRK.B, BTC-USD).
_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,8}$")


class FinnhubError(Exception):
    """Raised when a Finnhub request fails (non-2xx or network error)."""

    def __init__(self, status: int, path: str, body: str = ""):
        self.status = status
        self.path = path
        self.body = body
        super().__init__(f"Finnhub {path} returned HTTP {status}: {body[:200]}")


def _humanize_market_cap_millions(cap_m: float) -> str:
    """Finnhub returns market cap in millions of USD. Render as $T/$B/$M."""
    if not cap_m or cap_m <= 0:
        return "—"
    if cap_m >= 1_000_000:  # >= 1 trillion (since input is in millions)
        return f"${cap_m / 1_000_000:.2f}T"
    if cap_m >= 1_000:  # >= 1 billion
        return f"${cap_m / 1_000:.2f}B"
    return f"${cap_m:.2f}M"


def _fmt_price(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


def _fmt_change(change: Optional[float], pct: Optional[float]) -> str:
    if change is None or pct is None:
        return "—"
    sign = "+" if change >= 0 else "-"
    return f"{sign}${abs(change):,.2f} ({sign}{abs(pct):.2f}%)"


def _build_minimal_embed(symbol: str, quote: Dict, profile: Dict) -> discord.Embed:
    """Compact embed: company name + ticker, price, day change. That's it."""
    price = quote.get("c")
    change = quote.get("d")
    pct = quote.get("dp")
    company_name = profile.get("name") or symbol
    logo_url = profile.get("logo") or ""

    color = discord.Color.green() if (change or 0) >= 0 else discord.Color.red()
    embed = discord.Embed(
        description=f"**{_fmt_price(price)}** · {_fmt_change(change, pct)}",
        color=color,
    )
    if logo_url:
        embed.set_author(name=f"{company_name} ({symbol})", icon_url=logo_url)
    else:
        embed.set_author(name=f"{company_name} ({symbol})")
    return embed


def _build_expanded_embed(symbol: str, quote: Dict, profile: Dict, metric: Dict) -> discord.Embed:
    """Full embed: price, change, day range, 52-week, market cap, exchange, thumbnail."""
    price = quote.get("c")
    change = quote.get("d")
    pct = quote.get("dp")
    high = quote.get("h")
    low = quote.get("l")
    prev_close = quote.get("pc")
    quote_ts = quote.get("t")

    company_name = profile.get("name") or symbol
    exchange = profile.get("exchange") or "—"
    logo_url = profile.get("logo") or ""
    market_cap = _humanize_market_cap_millions(profile.get("marketCapitalization") or 0.0)

    metric_block = (metric or {}).get("metric") or {}
    wk52_high = metric_block.get("52WeekHigh")
    wk52_low = metric_block.get("52WeekLow")

    color = discord.Color.green() if (change or 0) >= 0 else discord.Color.red()
    embed = discord.Embed(
        title=f"📈 {company_name} ({symbol})",
        description=f"**{_fmt_price(price)}**  {_fmt_change(change, pct)}",
        color=color,
    )
    embed.add_field(name="Day Range", value=f"{_fmt_price(low)} – {_fmt_price(high)}", inline=True)
    embed.add_field(name="Previous Close", value=_fmt_price(prev_close), inline=True)
    embed.add_field(name="Market Cap", value=market_cap, inline=True)
    if wk52_high is not None and wk52_low is not None:
        embed.add_field(
            name="52-Week Range",
            value=f"{_fmt_price(wk52_low)} – {_fmt_price(wk52_high)}",
            inline=True,
        )
    if exchange and exchange != "—":
        embed.add_field(name="Exchange", value=exchange, inline=True)
    if logo_url:
        embed.set_thumbnail(url=logo_url)

    if quote_ts:
        stamp = datetime.fromtimestamp(quote_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        embed.set_footer(text=f"Powered by Finnhub • Quote {stamp}")
    else:
        embed.set_footer(text="Powered by Finnhub")
    return embed


class StockMoreView(discord.ui.View):
    """A single "More" button that swaps the minimal embed for the full one.

    Holds the already-fetched quote + profile in memory so the button click
    only needs the remaining /stock/metric + (best-effort) /stock/candle
    round-trips. After expansion the view is removed (`view=None`).
    """

    def __init__(self, cog, symbol: str, quote: Dict, profile: Dict):
        super().__init__(timeout=600)  # button stays live for 10 min
        self.cog = cog
        self.symbol = symbol
        self.quote = quote
        self.profile = profile

    @discord.ui.button(label="More", style=discord.ButtonStyle.secondary, emoji="📊")
    async def more_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        try:
            metric = await self.cog._finnhub_get(
                "/stock/metric", {"symbol": self.symbol, "metric": "all"}
            )
        except FinnhubError as exc:
            logger.info(f"Finnhub /stock/metric failed for {self.symbol}: {exc}")
            metric = {}
        except Exception as exc:
            logger.warning(f"Unexpected /stock/metric error for {self.symbol}: {exc}")
            metric = {}

        sparkline_png = await self.cog._fetch_sparkline_png(self.symbol)

        embed = _build_expanded_embed(self.symbol, self.quote, self.profile, metric)
        attachments: List[discord.File] = []
        if sparkline_png:
            attachments.append(discord.File(BytesIO(sparkline_png), filename="sparkline.png"))
            embed.set_image(url="attachment://sparkline.png")

        try:
            await interaction.edit_original_response(
                embed=embed, attachments=attachments, view=None
            )
        except discord.HTTPException as exc:
            logger.warning(f"Failed to expand stock embed for {self.symbol}: {exc}")


class StockCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.session = aiohttp.ClientSession()
        self.stock_rate_limits: Dict[int, List[float]] = defaultdict(list)
        self.MAX_LOOKUPS_PER_MINUTE = 5

    async def cog_unload(self):
        if hasattr(self, "session"):
            await self.session.close()

    async def is_rate_limited(self, user_id: int) -> bool:
        current_time = time.time()
        lookups = self.stock_rate_limits.get(user_id, [])
        lookups = [t for t in lookups if current_time - t < 60]
        self.stock_rate_limits[user_id] = lookups
        if len(lookups) >= self.MAX_LOOKUPS_PER_MINUTE:
            return True
        self.stock_rate_limits[user_id].append(current_time)
        return False

    async def _finnhub_get(self, path: str, params: Dict) -> Dict:
        """GET an arbitrary Finnhub endpoint and return parsed JSON.

        Raises FinnhubError on non-2xx (caller decides whether to surface or
        swallow — e.g. sparkline fetch swallows 403, quote fetch surfaces it).
        """
        url = f"{FINNHUB_BASE}{path}"
        headers = {"X-Finnhub-Token": settings.finnhub_api_key}
        async with self.session.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise FinnhubError(resp.status, path, body)
            return await resp.json()

    async def _resolve_symbol(self, query: str) -> Optional[str]:
        """Return a Finnhub ticker for ``query``, or None if nothing matches.

        Strategy: if the query *looks* like a ticker and ``/quote`` returns a
        non-zero current price for it, accept it. Otherwise hit ``/search`` and
        take the top hit. Saves an extra round-trip in the common "AAPL" case.
        """
        candidate = query.strip().upper()
        if _TICKER_RE.match(candidate):
            try:
                quote = await self._finnhub_get("/quote", {"symbol": candidate})
            except FinnhubError:
                quote = {}
            if quote.get("c"):  # 0/None means Finnhub didn't recognise the symbol
                return candidate

        try:
            search = await self._finnhub_get("/search", {"q": query})
        except FinnhubError as exc:
            logger.warning(f"Finnhub /search failed for {query!r}: {exc}")
            return None

        for hit in search.get("result", []) or []:
            symbol = hit.get("symbol")
            if symbol:
                return symbol
        return None

    async def _fetch_sparkline_png(self, symbol: str) -> Optional[bytes]:
        """Best-effort intraday sparkline. Returns None on any failure.

        Free-tier Finnhub paywalls ``/stock/candle`` (HTTP 403), so this will
        usually fail in practice — that's expected and we ship the embed
        without an image rather than surfacing the error.
        """
        now = int(time.time())
        # ~36 hours back to cover overnight + the prior trading session.
        params = {
            "symbol": symbol,
            "resolution": "5",
            "from": now - 36 * 3600,
            "to": now,
        }
        try:
            data = await self._finnhub_get("/stock/candle", params)
        except FinnhubError as exc:
            logger.debug(f"Sparkline unavailable for {symbol}: HTTP {exc.status}")
            return None
        except Exception as exc:
            logger.debug(f"Sparkline fetch error for {symbol}: {exc}")
            return None

        if data.get("s") != "ok":
            return None
        closes = data.get("c") or []
        timestamps = data.get("t") or []
        if len(closes) < 2:
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping sparkline.")
            return None

        try:
            color = "#16a34a" if closes[-1] >= closes[0] else "#dc2626"
            fig, ax = plt.subplots(figsize=(4.0, 1.2), dpi=120)
            ax.plot(timestamps, closes, color=color, linewidth=1.8)
            ax.fill_between(timestamps, closes, min(closes), color=color, alpha=0.18)
            ax.axis("off")
            fig.patch.set_alpha(0)
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, transparent=True)
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
        except Exception as exc:
            logger.warning(f"Sparkline render failed for {symbol}: {exc}")
            return None

    async def _fetch_profile_and_metric(self, symbol: str) -> Tuple[Dict, Dict]:
        """Fetch profile + metric in parallel; either may end up empty."""

        async def _safe(coro):
            try:
                return await coro
            except FinnhubError as exc:
                logger.info(f"Finnhub side-call failed for {symbol}: {exc}")
                return {}

        profile, metric = await asyncio.gather(
            _safe(self._finnhub_get("/stock/profile2", {"symbol": symbol})),
            _safe(self._finnhub_get("/stock/metric", {"symbol": symbol, "metric": "all"})),
        )
        return profile or {}, metric or {}

    @app_commands.command(
        name="soupystock",
        description="Get the current stock price for a ticker or company name (powered by Finnhub).",
    )
    @app_commands.describe(query="Stock ticker (e.g., AAPL) or company name (e.g., apple)")
    async def stock_command(self, interaction: discord.Interaction, query: str):
        start_time = time.time()
        logger.info(f"📈 Stock lookup by {interaction.user}: '{query}'")

        if not settings.finnhub_api_key:
            await interaction.response.send_message(
                "⚠️ Finnhub API key not configured. Ask the owner to set "
                "`FINNHUB_API_KEY` in `.env-stable`.",
                ephemeral=True,
            )
            return

        if await self.is_rate_limited(interaction.user.id):
            await interaction.response.send_message(
                "⚠️ Slow down — you've hit the per-user stock lookup limit. Try again in a minute.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            symbol = await self._resolve_symbol(query)
            if not symbol:
                await interaction.followup.send(
                    f"❌ Couldn't find a stock matching `{query}`.",
                    ephemeral=True,
                )
                return

            try:
                quote = await self._finnhub_get("/quote", {"symbol": symbol})
            except FinnhubError as exc:
                logger.error(f"Finnhub /quote failed for {symbol}: {exc}")
                await interaction.followup.send(
                    f"❌ Finnhub returned HTTP {exc.status} for `{symbol}`. Try again shortly.",
                    ephemeral=True,
                )
                return

            current_price = quote.get("c")
            if not current_price:
                await interaction.followup.send(
                    f"❌ No live quote available for `{symbol}` right now.",
                    ephemeral=True,
                )
                return

            try:
                profile = await self._finnhub_get("/stock/profile2", {"symbol": symbol})
            except FinnhubError as exc:
                logger.info(f"Finnhub /stock/profile2 failed for {symbol}: {exc}")
                profile = {}

            embed = _build_minimal_embed(symbol, quote, profile)
            view = StockMoreView(self, symbol, quote, profile)
            await interaction.followup.send(embed=embed, view=view)
            logger.info(f"✅ Stock lookup complete for {interaction.user}: {symbol} (${current_price})")

        except Exception as exc:
            logger.exception(f"Unhandled error in /soupystock for {query!r}: {exc}")
            try:
                await interaction.followup.send(
                    "❌ Something went wrong looking up that stock. Try again shortly.",
                    ephemeral=True,
                )
            except discord.HTTPException:
                pass


async def setup(bot):
    try:
        await bot.add_cog(StockCog(bot))
        logger.info("✅ StockCog loaded successfully")
        for command in bot.tree.get_commands():
            if command.name == "soupystock":
                logger.info("✅ /soupystock command registered successfully")
                return
        logger.warning("⚠️ /soupystock command not found in command tree")
    except Exception as exc:
        logger.error(f"❌ Failed to load StockCog: {exc}")
