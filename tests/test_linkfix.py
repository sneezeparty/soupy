"""Tests for link fixes: ``soupy.linkfix`` decides, ``soupy.cogs.linkfix`` replies.

Every account, post id and server id here is invented. pytest-asyncio isn't a
dependency, so coroutines run under ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from soupy import linkfix
from soupy.cogs import linkfix as cog_module
from soupy.linkfix import Preview, find_links, fixes, is_bare_links, parse_sites
from soupy.settings import settings

ALL = parse_sites("")


def fix(content, previews=(), fixers=ALL):
    return fixes(find_links(content, fixers), list(previews), fixers)


# ---------------------------------------------------------------------------
# Which links get a fix
# ---------------------------------------------------------------------------


def test_an_instagram_reel_without_a_preview_gets_the_fixer_link_without_tracking():
    assert fix("lol https://www.instagram.com/reel/AbC123xyz/?igsh=ZmFrZQ== look") == [
        "https://oginstagram.com/reel/AbC123xyz/"
    ]
    assert fix("https://instagram.com/p/QwErTy987/?img_index=3&igsh=ZmFrZQ==") == [
        "https://oginstagram.com/p/QwErTy987/?img_index=3"
    ], "img_index picks the picture in a carousel, so it stays"
    assert fix("https://www.instagram.com/some.creator/reel/AbC123xyz/") == [
        "https://oginstagram.com/some.creator/reel/AbC123xyz/"
    ]


def test_links_whose_preview_works_are_left_alone():
    # X builds a text-only card for a text post; that is a working preview.
    text_tweet = Preview("https://twitter.com/someone/status/1234567890123456789", kind="rich")
    assert fix("https://x.com/someone/status/1234567890123456789?s=20", [text_tweet]) == []
    tiktok_video = Preview("https://www.tiktok.com/@someone/video/7000000000000000001", kind="video", video=True)
    assert fix("https://www.tiktok.com/@someone/video/7000000000000000001", [tiktok_video]) == []
    reddit = Preview("https://www.reddit.com/r/somesub/comments/abc123/a_title/", kind="article", image=True)
    assert fix("https://www.reddit.com/r/somesub/comments/abc123/", [reddit]) == []


def test_login_walls_and_thumbnail_only_cards_count_as_broken():
    login_wall = Preview("https://www.facebook.com/share/r/1AbCdEfGh/", kind="link")
    assert fix("https://www.facebook.com/share/r/1AbCdEfGh/", [login_wall]) == [
        "https://facebed.com/share/r/1AbCdEfGh/"
    ]
    still = Preview("https://www.tiktok.com/@someone/video/7000000000000000001", kind="link", thumbnail=True)
    assert fix("https://www.tiktok.com/@someone/video/7000000000000000001", [still]) == [
        "https://tnktok.com/@someone/video/7000000000000000001"
    ], "a TikTok card with only a thumbnail doesn't play"


def test_a_dead_fixer_link_moves_to_the_current_fixer_and_a_live_one_is_left_alone():
    assert fix("https://kkinstagram.com/reel/AbC123xyz/") == ["https://oginstagram.com/reel/AbC123xyz/"]
    plays = Preview("https://vxtwitter.com/someone/status/1234567890123456789", kind="rich", video=True)
    assert fix("https://vxtwitter.com/someone/status/1234567890123456789", [plays]) == []
    # The fixer's own new layout carries no image fields, and still works.
    layout = Preview("https://oginstagram.com/reel/AbC123xyz/?srtk=x", kind="components")
    assert fix("https://www.oginstagram.com/reel/AbC123xyz/", [layout]) == []
    assert fix("https://www.oginstagram.com/reel/AbC123xyz/") == [], "posting the same fixer link again changes nothing"


def test_hidden_links_and_pages_that_are_not_posts_are_skipped():
    for content in (
        "<https://www.instagram.com/reel/AbC123xyz/>",
        "||https://www.instagram.com/reel/AbC123xyz/||",
        "`https://www.instagram.com/reel/AbC123xyz/`",
        "```\nhttps://www.instagram.com/reel/AbC123xyz/\n```",
        "https://www.instagram.com/some.creator/",
        "https://www.instagram.com/share/reel/BAbc123/",
        "https://www.youtube.com/watch?v=abcdefghijk",
        "no links at all",
    ):
        assert fix(content) == [], content
    assert fix("https://www.instagram.com/reel/AbC123xyz/", fixers=parse_sites("x=fxtwitter.com")) == [], (
        "a site left out of LINK_FIX_SITES is not fixed"
    )


def test_a_preview_for_another_link_does_not_count_and_repeats_are_fixed_once():
    youtube = Preview("https://www.youtube.com/watch?v=abcdefghijk", kind="video", video=True)
    assert fix(
        "https://www.youtube.com/watch?v=abcdefghijk and https://www.instagram.com/reel/AbC123xyz/ "
        "https://instagram.com/reel/AbC123xyz/?igsh=x",
        [youtube],
    ) == ["https://oginstagram.com/reel/AbC123xyz/"]


def test_short_links_keep_their_subdomain_and_match_a_preview_under_the_full_address():
    assert fix("https://vt.tiktok.com/ZSab12cd/") == ["https://vt.tnktok.com/ZSab12cd/"]
    resolved = Preview("https://www.tiktok.com/@someone/video/7000000000000000001", kind="video", video=True)
    assert fix("https://vt.tiktok.com/ZSab12cd/", [resolved]) == [], "one TikTok link, so the site's preview is its"
    assert fix("https://x.com/i/status/1234567890123456789") == ["https://fxtwitter.com/i/status/1234567890123456789"]
    assert fix("https://www.facebook.com/watch/?v=1234567890&ref=sharing") == [
        "https://facebed.com/watch/?v=1234567890"
    ]


def test_parse_sites_defaults_custom_and_bad_entries():
    assert parse_sites("") == {s.name: s.default_fixer for s in linkfix.SITES}
    assert parse_sites(" Instagram = https://newfixer.example/ , x=fixupx.com") == {
        "instagram": "newfixer.example",
        "x": "fixupx.com",
    }
    assert parse_sites("nosuchsite=fixer.example, tiktok=, reddit=vxreddit.com") == {"reddit": "vxreddit.com"}


def test_bare_links_are_recognized_with_or_without_the_archives_embed_notes():
    assert is_bare_links("https://oginstagram.com/reel/AbC123xyz/\nhttps://fxtwitter.com/a/status/1")
    assert is_bare_links("https://oginstagram.com/reel/AbC123xyz/ [Embed URL: https://oginstagram.com/reel/AbC123xyz/]")
    assert not is_bare_links("look at this https://oginstagram.com/reel/AbC123xyz/")
    assert not is_bare_links("")


# ---------------------------------------------------------------------------
# The cog
# ---------------------------------------------------------------------------

GUILD = 123456789012345678
OTHER_GUILD = 223456789012345678


@pytest.fixture
def link_settings(monkeypatch):
    monkeypatch.setenv("LINK_FIX_ENABLED", "true")
    monkeypatch.setenv("LINK_FIX_GUILD_IDS", str(GUILD))
    monkeypatch.delenv("LINK_FIX_SITES", raising=False)
    monkeypatch.setattr(cog_module, "PREVIEW_WAIT_SECONDS", 0)
    monkeypatch.setattr(cog_module, "FIX_CHECK_SECONDS", 0)
    settings.reload()
    yield
    monkeypatch.undo()
    settings.reload()


def _message(content, *, guild=GUILD, bot=False, embeds=(), fetch_error=None):
    reply = SimpleNamespace(id=900, channel=MagicMock())
    reply.channel.fetch_message = AsyncMock(return_value=SimpleNamespace(embeds=[discord.Embed(url="https://x")]))
    fresh = SimpleNamespace(content=content, embeds=list(embeds), reply=AsyncMock(return_value=reply))
    channel = MagicMock()
    channel.fetch_message = AsyncMock(side_effect=fetch_error, return_value=fresh)
    msg = SimpleNamespace(id=500, content=content, guild=SimpleNamespace(id=guild), channel=channel)
    msg.author = SimpleNamespace(bot=bot, name="member")
    return msg, fresh


def _run(cog, coro):
    async def go():
        await coro
        if cog._tasks:
            await asyncio.gather(*cog._tasks)

    asyncio.run(go())


def test_cog_replies_without_pinging_and_deletes_the_reply_with_the_message(link_settings):
    bot = MagicMock()
    cog = cog_module.LinkFixCog(bot)
    unrelated = discord.Embed.from_dict({"type": "video", "url": "https://www.youtube.com/watch?v=abcdefghijk"})
    msg, fresh = _message("https://www.instagram.com/reel/AbC123xyz/?igsh=x", embeds=[unrelated])

    _run(cog, cog.on_message(msg))

    fresh.reply.assert_awaited_once()
    args, kwargs = fresh.reply.call_args
    assert args == ("https://oginstagram.com/reel/AbC123xyz/",)
    assert kwargs["mention_author"] is False
    assert kwargs["allowed_mentions"].users is False and kwargs["allowed_mentions"].everyone is False

    partial = bot.get_partial_messageable.return_value.get_partial_message.return_value
    partial.delete = AsyncMock()
    asyncio.run(cog.on_raw_message_delete(SimpleNamespace(message_id=500, channel_id=42)))
    bot.get_partial_messageable.assert_called_once_with(42)
    bot.get_partial_messageable.return_value.get_partial_message.assert_called_once_with(900)
    partial.delete.assert_awaited_once()


def test_cog_leaves_working_previews_other_servers_bots_and_deleted_messages_alone(link_settings):
    cog = cog_module.LinkFixCog(MagicMock())
    works = discord.Embed.from_dict(
        {"type": "rich", "url": "https://twitter.com/someone/status/1234567890123456789", "description": "a post"}
    )
    cases = [
        _message("https://x.com/someone/status/1234567890123456789", embeds=[works]),
        _message("https://www.instagram.com/reel/AbC123xyz/", guild=OTHER_GUILD),
        _message("https://www.instagram.com/reel/AbC123xyz/", bot=True),
        _message(
            "https://www.instagram.com/reel/AbC123xyz/", fetch_error=discord.NotFound(MagicMock(status=404), "gone")
        ),
    ]
    for msg, fresh in cases:
        _run(cog, cog.on_message(msg))
        fresh.reply.assert_not_awaited()
    assert cases[1][0].channel.fetch_message.await_count == 0, "other servers aren't even re-read"


def test_cog_does_nothing_when_switched_off(link_settings, monkeypatch):
    monkeypatch.setenv("LINK_FIX_ENABLED", "false")
    settings.reload()
    cog = cog_module.LinkFixCog(MagicMock())
    msg, fresh = _message("https://www.instagram.com/reel/AbC123xyz/")
    _run(cog, cog.on_message(msg))
    assert msg.channel.fetch_message.await_count == 0
    fresh.reply.assert_not_awaited()


def test_preview_of_reads_media_from_discord_embeds():
    video = discord.Embed.from_dict(
        {"type": "video", "url": "https://a.example/v", "video": {"url": "https://a.example/v.mp4"}}
    )
    layout = discord.Embed.from_dict({"type": "components", "url": "https://a.example/c"})
    card = discord.Embed.from_dict({"type": "link", "url": "https://a.example/l", "title": "Log in"})
    assert cog_module.preview_of(video) == Preview("https://a.example/v", kind="video", video=True)
    assert cog_module.preview_of(layout) == Preview("https://a.example/c", kind="components")
    assert cog_module.preview_of(card) == Preview("https://a.example/l", kind="link")
