"""Link-fix cog: replies with a fixer link when a member's link is a video Discord can't play.

A member posts ``instagram.com/reel/X`` and Discord shows nothing, or an X
video that only opens in a browser. Soupy waits for Discord to build the
message's previews. For each link with no preview playing a video
(``soupy.linkfix`` decides), it reads the same post on a fixer site
(``oginstagram.com/reel/X``) the way Discord's crawler would. When that page
carries a video, Soupy replies with the fixer link, which Discord plays. A text
or photo post has nothing to fix and is left alone, preview or not. There is no
LLM call and no queue, so it never waits behind chat or images.

* It only acts on messages posted while the bot runs; nothing is backfilled.
* The reply pings nobody, and Soupy never touches the member's message (it
  needs no Manage Messages permission).
* When the member deletes their message, the reply goes too.
* A fixer page that can't be read means no reply (the fix wouldn't play), and
  a warning in the log; so does a fix that gets no preview of its own. Either
  way that fixer site may be down, and ``LINK_FIX_SITES`` is where to swap it.
* A fix reply is only links. Chat history and Soupy's memory of himself skip
  such messages (``soupy.linkfix.is_bare_links``), so he doesn't learn to post
  bare links or remember them as things he said.

Settings: ``LINK_FIX_ENABLED``, ``LINK_FIX_GUILD_IDS`` (empty: every server),
``LINK_FIX_SITES``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Set

import aiohttp
import discord
from discord.ext import commands

from soupy import linkfix
from soupy.settings import settings

logger = logging.getLogger(__name__)

# Discord builds a link's preview a second or two after the message arrives;
# Instagram's never comes. Looking this late keeps a slow but working preview
# from getting a needless fix.
PREVIEW_WAIT_SECONDS = 6.0
# The fixer site fetches the post before it answers Discord, so its preview takes longer.
FIX_CHECK_SECONDS = 20.0
FIXER_TIMEOUT_SECONDS = 8.0
# Fixers answer Discord's crawler with the page Discord turns into a preview, and
# send anyone else on to the original site, so Soupy asks the way Discord will.
DISCORD_CRAWLER = "Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)"
_PAGE_BYTES = 512 * 1024  # the meta tags are in the page's head
# Member message id -> Soupy's reply, so the reply can follow a deleted message.
_REMEMBER = 500


def preview_of(embed: discord.Embed) -> linkfix.Preview:
    return linkfix.Preview(
        url=embed.url or "",
        kind=str(embed.type or "rich"),
        video=bool(embed.video.url),
        video_file=bool(getattr(embed.video, "proxy_url", None)),
    )


async def fixer_has_video(url: str, fixers: Dict[str, str]) -> bool:
    """Whether the fixer page at ``url`` carries a video. False when it can't be read."""
    try:
        timeout = aiohttp.ClientTimeout(total=FIXER_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": DISCORD_CRAWLER}) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.info("🔗 Link fix: %s answered %s, so no fix", url, resp.status)
                    return False
                landed = linkfix.site_of(str(resp.url), fixers)
                if not landed or not landed[1]:
                    # Sent on to the original site: the fixer couldn't serve this post.
                    logger.info("🔗 Link fix: %s sent us on to %s, so no fix", url, resp.url.host)
                    return False
                page = bytearray()
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    page += chunk
                    if len(page) >= _PAGE_BYTES:
                        break
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning(
            "🔗 Link fix: couldn't read %s (%s), so no fix. If this keeps happening, that fixer site "
            "may be down: swap it in LINK_FIX_SITES.",
            url,
            exc.__class__.__name__,
        )
        return False
    return linkfix.is_video_page(page.decode("utf-8", "replace"))


class LinkFixCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._replies: "OrderedDict[int, int]" = OrderedDict()
        self._tasks: Set[asyncio.Task] = set()

    def _fixers_for(self, message: discord.Message) -> Optional[Dict[str, str]]:
        if message.author.bot or message.guild is None or not settings.link_fix_enabled:
            return None
        guilds = settings.link_fix_guild_ids
        if guilds and message.guild.id not in guilds:
            return None
        return linkfix.parse_sites(settings.link_fix_sites)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        fixers = self._fixers_for(message)
        if not fixers or not linkfix.find_links(message.content, fixers):
            return

        await asyncio.sleep(PREVIEW_WAIT_SECONDS)
        fresh = await self._reread(message)
        if fresh is None:
            return
        candidates = self._not_playing(fresh, fixers)
        if not candidates:
            return

        # Only a video is worth fixing; the fixer's page says which posts are videos.
        answers = await asyncio.gather(*(fixer_has_video(url, fixers) for url in candidates))
        videos = {url for url, is_video in zip(candidates, answers, strict=True) if is_video}
        if not videos:
            return
        # Asking took a moment, and a slow preview may have come in meanwhile.
        fresh = await self._reread(message)
        if fresh is None:
            return
        fixed = [url for url in self._not_playing(fresh, fixers) if url in videos]
        if not fixed:
            return
        try:
            reply = await fresh.reply(
                "\n".join(fixed), mention_author=False, allowed_mentions=discord.AllowedMentions.none()
            )
        except discord.HTTPException as exc:
            logger.warning("🔗 Link fix: couldn't reply in #%s: %s", getattr(message.channel, "name", "?"), exc)
            return

        logger.info("🔗 Link fix for %s: %s", message.author, " ".join(fixed))
        self._replies[message.id] = reply.id
        while len(self._replies) > _REMEMBER:
            self._replies.popitem(last=False)
        task = asyncio.create_task(self._check_fix(reply, fixed))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _reread(self, message: discord.Message) -> Optional[discord.Message]:
        """The message as it is now (edited, with its previews). None once it is deleted."""
        try:
            return await message.channel.fetch_message(message.id)
        except discord.NotFound:
            return None
        except discord.HTTPException as exc:
            logger.warning("🔗 Link fix: couldn't re-read message %s: %s", message.id, exc)
            return None

    @staticmethod
    def _not_playing(message: discord.Message, fixers: Dict[str, str]) -> List[str]:
        # The links are read again each time: the message may have been edited.
        links = linkfix.find_links(message.content, fixers)
        return linkfix.not_playing(links, [preview_of(e) for e in message.embeds], fixers)

    async def _check_fix(self, reply: discord.Message, fixed: List[str]) -> None:
        await asyncio.sleep(FIX_CHECK_SECONDS)
        try:
            fresh = await reply.channel.fetch_message(reply.id)
        except discord.HTTPException:
            return
        if not fresh.embeds:
            logger.warning(
                "🔗 Link fix: %s got no preview either. If this keeps happening, that fixer site "
                "may be down: swap it in LINK_FIX_SITES.",
                " ".join(fixed),
            )

    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        reply_id = self._replies.pop(payload.message_id, None)
        if reply_id is None:
            return
        try:
            await self.bot.get_partial_messageable(payload.channel_id).get_partial_message(reply_id).delete()
        except discord.HTTPException as exc:
            logger.debug("🔗 Link fix: couldn't delete reply %s: %s", reply_id, exc)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LinkFixCog(bot))
    if settings.link_fix_enabled:
        where = ", ".join(str(g) for g in settings.link_fix_guild_ids) or "every server"
        sites = ", ".join(f"{k}→{v}" for k, v in linkfix.parse_sites(settings.link_fix_sites).items())
        logger.info("🔗 Link fixes on in %s: %s", where, sites)
    else:
        logger.info("🔗 Link fixes off (LINK_FIX_ENABLED)")
