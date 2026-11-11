"""Link-fix cog: replies with a fixer link when a post's link got no working preview.

A member posts ``instagram.com/reel/X`` and Discord shows nothing. Soupy waits
for Discord to build the message's previews, and for each link to a post that
still has none (``soupy.linkfix`` decides), replies with the same post on a
fixer site (``oginstagram.com/reel/X``), which Discord can play. There is no
LLM call and no queue: it is a text rewrite, so it never waits behind chat or
images.

* It only acts on messages posted while the bot runs; nothing is backfilled.
* The reply pings nobody, and Soupy never touches the member's message (it
  needs no Manage Messages permission).
* When the member deletes their message, the reply goes too.
* A fix that gets no preview of its own is logged as a warning: that fixer
  site may be down, and ``LINK_FIX_SITES`` is where to swap it.
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
# Member message id -> Soupy's reply, so the reply can follow a deleted message.
_REMEMBER = 500


def preview_of(embed: discord.Embed) -> linkfix.Preview:
    return linkfix.Preview(
        url=embed.url or "",
        kind=str(embed.type or "rich"),
        image=bool(embed.image.url),
        thumbnail=bool(embed.thumbnail.url),
        video=bool(embed.video.url),
    )


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
        try:
            fresh = await message.channel.fetch_message(message.id)
        except discord.NotFound:
            return  # deleted while we waited
        except discord.HTTPException as exc:
            logger.warning("🔗 Link fix: couldn't re-read message %s: %s", message.id, exc)
            return

        # Read the links again: the message may have been edited while we waited.
        links = linkfix.find_links(fresh.content, fixers)
        fixed = linkfix.fixes(links, [preview_of(e) for e in fresh.embeds], fixers)
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
