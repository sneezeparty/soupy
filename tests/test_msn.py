"""Tests for turning an msn.com link into the article it was syndicated from.

2026-09-20: Soupy cross-posted an MSN link to Bluesky and the card came out
with no thumbnail. MSN serves a JavaScript shell with no og:image and no date,
so there was nothing to attach. The URL below is that post's, and the source it
resolves to is the real one from MSN's content endpoint.

The lookup is faked here. Nothing in the suite touches the network.
"""

from __future__ import annotations

import asyncio

import pytest

from soupy import msn

MSN_URL = (
    "https://www.msn.com/en-us/news/other/growing-revenue-in-airport-assistance-fund-leads-"
    "south-dakota-commission-to-loosen-spending-policies/ar-AA2cBYYq"
)
ORIGINAL = (
    "https://www.kotatv.com/2026/09/20/growing-revenue-airport-assistance-fund-leads-"
    "south-dakota-commission-loosen-spending-policies/"
)
PAYLOAD = {
    "title": "Growing revenue in airport assistance fund leads South Dakota commission to loosen spending policies",
    "sourceHref": ORIGINAL,
    "provider": {"name": "Rapid City KOTA-TV"},
}


def _resolve(url, payload=PAYLOAD, **kwargs):
    calls = []

    async def fake_fetch(endpoint, timeout):
        calls.append((endpoint, timeout))
        if isinstance(payload, Exception):
            raise payload
        return payload

    resolved = asyncio.run(msn.resolve_article_url(url, fetch=fake_fetch, **kwargs))
    return resolved, calls


def test_an_msn_link_becomes_the_original_article():
    resolved, calls = _resolve(MSN_URL)
    assert resolved == ORIGINAL
    assert calls == [("https://assets.msn.com/content/view/v2/Detail/en-us/AA2cBYYq", 8.0)]


def test_the_locale_in_the_link_is_used():
    url = MSN_URL.replace("/en-us/", "/en-gb/")
    _resolved, calls = _resolve(url)
    assert calls[0][0].endswith("/en-gb/AA2cBYYq")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.kotatv.com/2026/09/20/some-story/",
        "https://www.msn.com/en-us/news/other/",  # MSN, but not an article link
        "",
    ],
)
def test_anything_that_isnt_an_msn_article_is_left_alone_without_a_lookup(url):
    resolved, calls = _resolve(url)
    assert resolved == url
    assert calls == []


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"sourceHref": ""},
        {"sourceHref": "/relative/path"},
        {"sourceHref": "https://www.msn.com/en-us/news/other/thing/ar-AA123456"},  # points back at MSN
        RuntimeError("endpoint gone"),
    ],
)
def test_a_failed_lookup_keeps_the_msn_link(payload):
    """The endpoint is undocumented. Every failure has to leave the caller posting what it had."""
    resolved, _calls = _resolve(MSN_URL, payload=payload)
    assert resolved == MSN_URL


def test_article_id_and_host_checks():
    assert msn.msn_article_id(MSN_URL) == "AA2cBYYq"
    assert msn.msn_article_id("https://www.kotatv.com/ar-AA2cBYYq") is None
    assert msn.is_msn_url("https://www.msn.com/en-us/news") and msn.is_msn_url("https://msn.com/x")
    assert not msn.is_msn_url("https://notmsn.com/en-us/news")
    assert msn.detail_endpoint("https://www.msn.com/en-us/news") is None


def test_both_cogs_resolve_before_fetching_an_article():
    """The seams: whatever an article pipeline fetches and posts is already the original."""
    import inspect

    from soupy.cogs import bluesky, dailypost

    picking = inspect.getsource(dailypost.DailyPostCog._pick_and_comment)
    assert "url = await resolve_article_url(url)" in picking

    discovery = inspect.getsource(bluesky.BlueskyEngageCog._discover_article)
    assert 'await resolve_article_url(a.get("href", ""))' in discovery

    provided = inspect.getsource(bluesky.BlueskyEngageCog._run_original_post_pipeline)
    assert "article_url = await resolve_article_url(article_url)" in provided
