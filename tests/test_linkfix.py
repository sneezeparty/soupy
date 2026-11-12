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
from aiohttp import web

from soupy import linkfix
from soupy.cogs import linkfix as cog_module
from soupy.linkfix import Preview, find_links, is_bare_links, is_video_page, not_playing, parse_sites
from soupy.settings import settings

ALL = parse_sites("")


def fix(content, previews=(), fixers=ALL):
    """The fixer links Soupy would ask about: links with no preview playing a video."""
    return not_playing(find_links(content, fixers), list(previews), fixers)


# ---------------------------------------------------------------------------
# Which links are asked about
# ---------------------------------------------------------------------------


def test_an_instagram_reel_without_a_preview_is_asked_about_on_the_fixer_without_tracking():
    assert fix("lol https://www.instagram.com/reel/AbC123xyz/?igsh=ZmFrZQ== look") == [
        "https://oginstagram.com/reel/AbC123xyz/"
    ]
    assert fix("https://instagram.com/p/QwErTy987/?img_index=3&igsh=ZmFrZQ==") == [
        "https://oginstagram.com/p/QwErTy987/?img_index=3"
    ], "img_index picks the picture in a carousel, so it stays"
    assert fix("https://www.instagram.com/some.creator/reel/AbC123xyz/") == [
        "https://oginstagram.com/some.creator/reel/AbC123xyz/"
    ]


def test_a_preview_that_plays_a_video_is_left_alone():
    # Bluesky's own preview of a video post carries the file, which Discord copies.
    bluesky = Preview(
        "https://bsky.app/profile/someone.example/post/3abcdefghij2k", kind="rich", video=True, video_file=True
    )
    assert fix("https://bsky.app/profile/someone.example/post/3abcdefghij2k", [bluesky]) == []
    layout = Preview("https://oginstagram.com/reel/AbC123xyz/", kind="components")
    assert fix("https://kkinstagram.com/reel/AbC123xyz/", [layout]) == []


def test_previews_without_a_playing_video_are_asked_about():
    tweet = "https://x.com/someone/status/1234567890123456789?s=20"
    fixed = ["https://fxtwitter.com/someone/status/1234567890123456789"]
    # X's own preview of a video tweet names the tweet's page: a still that opens a browser.
    still = Preview("https://twitter.com/someone/status/1234567890123456789", kind="rich", video=True)
    assert fix(tweet, [still]) == fixed
    # A text or photo post lands here too; the fixer's page then says it is no video.
    assert fix(tweet, [Preview("https://twitter.com/someone/status/1234567890123456789")]) == fixed
    login_wall = Preview("https://www.facebook.com/share/r/1AbCdEfGh/", kind="link")
    assert fix("https://www.facebook.com/share/r/1AbCdEfGh/", [login_wall]) == [
        "https://facebed.com/share/r/1AbCdEfGh/"
    ]
    reddit = Preview("https://www.reddit.com/r/somesub/comments/abc123/a_title/", kind="article")
    assert fix("https://www.reddit.com/r/somesub/comments/abc123/", [reddit]) == [
        "https://vxreddit.com/r/somesub/comments/abc123/"
    ]
    # TikTok's own preview embeds TikTok's player, which doesn't play right in Discord.
    tiktok = Preview("https://www.tiktok.com/@someone/video/7000000000000000001", kind="video", video=True)
    assert fix("https://www.tiktok.com/@someone/video/7000000000000000001", [tiktok]) == [
        "https://tnktok.com/@someone/video/7000000000000000001"
    ]


def test_a_dead_fixer_link_moves_to_the_current_fixer_and_a_live_one_is_left_alone():
    assert fix("https://kkinstagram.com/reel/AbC123xyz/") == ["https://oginstagram.com/reel/AbC123xyz/"]
    plays = Preview(
        "https://vxtwitter.com/someone/status/1234567890123456789", kind="rich", video=True, video_file=True
    )
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
    resolved = Preview(
        "https://www.tiktok.com/@someone/video/7000000000000000001", kind="rich", video=True, video_file=True
    )
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


def test_is_video_page_reads_the_fixer_pages_meta_tags():
    for page in (
        '<meta property="og:video" content="https://cdn.fixer.example/v.mp4"/>',
        '<meta content="https://cdn.fixer.example/v.mp4" property="og:video:secure_url">',
        "<META NAME='twitter:player:stream' CONTENT='https://cdn.fixer.example/v.mp4'>",
        '<meta property="og:type" content="video.other">',
        '<meta property="twitter:card" content="player"/>',  # how fxtwitter marks a video tweet
    ):
        assert is_video_page(f"<html><head><title>a post</title>{page}</head></html>"), page
    for page in (
        '<meta property="og:type" content="article"><meta property="og:image" content="https://cdn.fixer.example/a.jpg">',
        '<meta property="twitter:card" content="summary_large_image">',
        '<meta property="og:video" content="">',
        "<p>og:video</p>",
        "",
    ):
        assert not is_video_page(page), page


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


class FakeFixers:
    """Stands in for reading fixer pages: ``videos`` are the fixer links whose page carries a video."""

    def __init__(self, *videos):
        self.videos = set(videos)
        self.asked = []

    async def __call__(self, url, fixers):
        self.asked.append(url)
        return url in self.videos


REEL_FIX = "https://oginstagram.com/reel/AbC123xyz/"


@pytest.fixture
def link_settings(monkeypatch):
    monkeypatch.setenv("LINK_FIX_ENABLED", "true")
    monkeypatch.setenv("LINK_FIX_GUILD_IDS", str(GUILD))
    monkeypatch.delenv("LINK_FIX_SITES", raising=False)
    monkeypatch.setattr(cog_module, "PREVIEW_WAIT_SECONDS", 0)
    monkeypatch.setattr(cog_module, "FIX_CHECK_SECONDS", 0)
    fixer_pages = FakeFixers(REEL_FIX)
    monkeypatch.setattr(cog_module, "fixer_has_video", fixer_pages)
    settings.reload()
    yield fixer_pages
    monkeypatch.undo()
    settings.reload()


def _message(content, *, guild=GUILD, bot=False, embeds=(), fetch_error=None, later_embeds=None):
    """A member's message; ``later_embeds`` are the previews it has by the second look."""
    reply = SimpleNamespace(id=900, channel=MagicMock())
    reply.channel.fetch_message = AsyncMock(return_value=SimpleNamespace(embeds=[discord.Embed(url="https://x")]))
    fresh = SimpleNamespace(content=content, embeds=list(embeds), reply=AsyncMock(return_value=reply))
    channel = MagicMock()
    if fetch_error is not None:
        channel.fetch_message = AsyncMock(side_effect=fetch_error)
    elif later_embeds is not None:
        later = SimpleNamespace(content=content, embeds=list(later_embeds), reply=fresh.reply)
        channel.fetch_message = AsyncMock(side_effect=[fresh, later])
    else:
        channel.fetch_message = AsyncMock(return_value=fresh)
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


def test_cog_leaves_a_post_that_is_no_video_alone_even_without_a_preview(link_settings):
    # A slow preview on a text post is nothing to fix: the fixer's page shows no video.
    cog = cog_module.LinkFixCog(MagicMock())
    msg, fresh = _message("https://bsky.app/profile/someone.example/post/3abcdefghij2k")
    _run(cog, cog.on_message(msg))
    assert link_settings.asked == ["https://bskx.app/profile/someone.example/post/3abcdefghij2k"]
    fresh.reply.assert_not_awaited()


def test_cog_leaves_a_video_whose_preview_arrives_while_it_asks(link_settings):
    late = discord.Embed.from_dict(
        {
            "type": "video",
            "url": "https://www.instagram.com/reel/AbC123xyz/",
            "video": {"url": "https://a.example/v.mp4", "proxy_url": "https://media.discord.example/v.mp4"},
        }
    )
    cog = cog_module.LinkFixCog(MagicMock())
    msg, fresh = _message("https://www.instagram.com/reel/AbC123xyz/", later_embeds=[late])
    _run(cog, cog.on_message(msg))
    assert link_settings.asked == [REEL_FIX]
    fresh.reply.assert_not_awaited()


def test_cog_leaves_playing_videos_other_servers_bots_and_deleted_messages_alone(link_settings):
    cog = cog_module.LinkFixCog(MagicMock())
    plays = discord.Embed.from_dict(
        {
            "type": "rich",
            "url": "https://bsky.app/profile/someone.example/post/3abcdefghij2k",
            "video": {"url": "https://cdn.bsky.example/v.mp4", "proxy_url": "https://media.discord.example/v.mp4"},
        }
    )
    cases = [
        _message("https://bsky.app/profile/someone.example/post/3abcdefghij2k", embeds=[plays]),
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
    assert link_settings.asked == [], "a video that plays, and everything else here, needs no fixer page"


def test_cog_does_nothing_when_switched_off(link_settings, monkeypatch):
    monkeypatch.setenv("LINK_FIX_ENABLED", "false")
    settings.reload()
    cog = cog_module.LinkFixCog(MagicMock())
    msg, fresh = _message("https://www.instagram.com/reel/AbC123xyz/")
    _run(cog, cog.on_message(msg))
    assert msg.channel.fetch_message.await_count == 0
    fresh.reply.assert_not_awaited()


def test_preview_of_reads_media_from_discord_embeds():
    copied = discord.Embed.from_dict(
        {
            "type": "rich",
            "url": "https://a.example/v",
            "video": {"url": "https://a.example/v.mp4", "proxy_url": "https://media.discord.example/v.mp4"},
        }
    )
    page_only = discord.Embed.from_dict(
        {"type": "rich", "url": "https://a.example/t", "video": {"url": "https://a.example/i/videos/1"}}
    )
    layout = discord.Embed.from_dict({"type": "components", "url": "https://a.example/c"})
    card = discord.Embed.from_dict({"type": "link", "url": "https://a.example/l", "title": "Log in"})
    assert cog_module.preview_of(copied) == Preview("https://a.example/v", video=True, video_file=True)
    assert cog_module.preview_of(page_only) == Preview("https://a.example/t", video=True)
    assert cog_module.preview_of(layout) == Preview("https://a.example/c", kind="components")
    assert cog_module.preview_of(card) == Preview("https://a.example/l", kind="link")


# ---------------------------------------------------------------------------
# Reading a fixer page, against a stand-in fixer on this machine
# ---------------------------------------------------------------------------

VIDEO_PAGE = '<html><head><meta property="og:video" content="https://cdn.fixer.example/v.mp4"></head></html>'
PHOTO_PAGE = '<html><head><meta property="og:image" content="https://cdn.fixer.example/p.jpg"></head></html>'


def _ask_fixer(paths):
    """``fixer_has_video`` for each path on a stand-in fixer, served from 127.0.0.1."""

    async def post(request):
        # Like a real fixer: the embed page for Discord's crawler, the original site for anyone else.
        if "Discordbot" not in request.headers.get("User-Agent", ""):
            raise web.HTTPFound("http://localhost/somewhere-else")
        return web.Response(text=VIDEO_PAGE if "reel" in request.path else PHOTO_PAGE, content_type="text/html")

    async def gone(request):
        raise web.HTTPNotFound()

    async def bounce(request):
        raise web.HTTPFound(f"http://localhost:{request.url.port}/reel/AbC123xyz/")

    async def go():
        app = web.Application()
        app.router.add_get("/reel/{id}/", post)
        app.router.add_get("/p/{id}/", post)
        app.router.add_get("/gone/{id}/", gone)
        app.router.add_get("/bounce/{id}/", bounce)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            fixers = {"instagram": "127.0.0.1"}
            return [await cog_module.fixer_has_video(f"http://127.0.0.1:{port}{p}", fixers) for p in paths]
        finally:
            await runner.cleanup()

    return asyncio.run(go())


def test_fixer_has_video_reads_the_page_discord_would_get():
    assert _ask_fixer(["/reel/AbC123xyz/", "/p/QwErTy987/"]) == [True, False]


def test_fixer_has_video_is_false_when_the_fixer_cant_serve_the_post(caplog):
    # A 404, and a fixer that sends the crawler on to another site.
    assert _ask_fixer(["/gone/AbC123xyz/", "/bounce/AbC123xyz/"]) == [False, False]
    unreachable = asyncio.run(
        cog_module.fixer_has_video("http://127.0.0.1:9/reel/AbC123xyz/", {"instagram": "127.0.0.1"})
    )
    assert unreachable is False
    assert "swap it in LINK_FIX_SITES" in caplog.text
