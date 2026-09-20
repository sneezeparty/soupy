"""Tests for what Soupy engages with on Bluesky.

2026-09-20: it quote-posted Jacob Soboroff's post at 15:52 and replied to the
same post at 16:03. Reply discovery filtered on the comment history alone, so a
post it had quote-posted was still a candidate for a reply, and the author
recency block (which spreads engagement across accounts) didn't count
quote-posts either.
"""

from __future__ import annotations

from soupy.cogs import bluesky

SOBOROFF = "at://did:plc:dw2ayzj7ay2s3zo52rpzx4u5/app.bsky.feed.post/3mvvrgirq5s2c"
HISTORY = {
    "comments": [
        {"post_uri": "at://did:plc:aaa/app.bsky.feed.post/1", "ts": "2026-09-05T13:54:22+00:00"},
    ],
    "reposts": [
        {"post_uri": SOBOROFF, "ts": "2026-09-20T22:52:15+00:00"},
    ],
}


def test_a_quote_posted_post_is_not_offered_for_a_reply():
    assert SOBOROFF in bluesky._engaged_uris(HISTORY)
    assert "at://did:plc:aaa/app.bsky.feed.post/1" in bluesky._engaged_uris(HISTORY)


def test_engaged_uris_survives_junk_history():
    assert bluesky._engaged_uris({}) == set()
    assert bluesky._engaged_uris({"comments": ["junk", {}, {"post_uri": ""}]}) == set()


def test_recent_authors_count_quote_posts_and_come_back_newest_first():
    authors = bluesky._recent_engagement_authors(HISTORY)
    assert [did for did, _ts in authors] == ["did:plc:dw2ayzj7ay2s3zo52rpzx4u5", "did:plc:aaa"]
    assert bluesky._recent_engagement_authors({"reposts": [{"post_uri": "not-a-uri"}]}) == []


# ---------------------------------------------------------------------------
# Link-card thumbnails
# ---------------------------------------------------------------------------

# The og:image URL is written into the page with HTML entities. Fetched verbatim,
# "?auth=sig&amp;width=1200" is a different URL than the signature covers, and the CDN
# answers HTTP 400 (Gray TV, 2026-09-20), so the post went out with no thumbnail.
SIGNED = '<meta property="og:image" content="https://cdn.example.com/a.jpg?auth=sig&amp;width=1200"/>'


def test_the_share_image_url_is_entity_decoded():
    assert (
        bluesky._extract_image_url(SIGNED, "https://example.com/story")
        == "https://cdn.example.com/a.jpg?auth=sig&width=1200"
    )


def test_og_image_wins_over_twitter_image_and_relative_paths_resolve():
    page = '<meta name="twitter:image" content="/second.jpg">' '<meta property="og:image" content="/first.jpg">'
    assert bluesky._extract_image_url(page, "https://example.com/news/story") == "https://example.com/first.jpg"


def test_no_image_tags_means_no_thumbnail():
    assert bluesky._extract_image_url("<html><title>MSN</title></html>", "https://www.msn.com/x") is None
    assert bluesky._extract_image_url("", "https://example.com") is None
