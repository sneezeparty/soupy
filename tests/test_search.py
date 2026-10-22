"""/soupysearch: which results reach the model, which pages become sources, and how ages read."""

import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from soupy.cogs import search
from soupy.url_fetch import FetchResult

TODAY = date(2026, 10, 22)
QUERY = "harbor pier storm damage latest news"


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(search.timeaware, "now_local", lambda: datetime(2026, 10, 22, 9, 0))


def result(title, url, body="", **extra):
    return {"title": title, "href": url, "body": body, **extra}


def make_cog():
    """A cog whose aiohttp session is already closed: the tests fake every fetch."""

    async def build():
        cog = search.SearchCog(bot=None)
        await cog.session.close()
        return cog

    return asyncio.run(build())


def reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# ---- the words a result has to mention ------------------------------------------


def test_query_terms_leave_out_stopwords_and_recency_words():
    assert search._query_terms("What is the latest news about the harbor pier storm?") == ["harbor", "pier", "storm"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Storms batter the coast", True),  # plural of a short term
        ("New harbour rules", False),
        ("Two piers closed", True),
        ("Volcanic ash over the bay", True),  # "volcano" matches on its first five letters
        ("Erewhon locations and hours", False),
    ],
)
def test_mentions(text, expected):
    assert search._mentions(text, ["storm", "pier", "volcano"]) is expected


def test_with_no_topic_words_everything_counts():
    assert search._query_terms("what is it") == []
    assert search._mentions("anything at all", []) is True


def test_an_article_needs_two_query_words_once_the_query_has_three():
    terms = search._query_terms("port wendle harbor erosion")
    assert search._article_need(terms) == 2
    wildfire = "Crews near Port Alder held the fire line overnight as winds eased."
    assert search._mentions(wildfire, terms) is True  # fine for a search result
    assert search._mentions(wildfire, terms, search._article_need(terms)) is False
    assert search._article_need(["tides", "work"]) == 1


def test_the_news_query_leaves_out_the_recency_words():
    assert search._news_query("Harbor pier storm latest news") == "Harbor pier storm"
    assert search._news_query("what happened this week at the pier") == "what happened at the pier"
    assert search._news_query("latest news") == "latest news"


# ---- ages ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-09-17", "about 5 weeks ago"),
        ("2026-10-21T18:30:00+00:00", "yesterday"),
        ("2026-10-23", "today"),  # a UTC date a day ahead is not in the future
        ("2024-03-01", "about 3 years ago, in 2024"),
        ("Opinion2 d", ""),
        (None, ""),
    ],
)
def test_age_is_worded(value, expected):
    assert search._age(value, TODAY) == expected


def test_a_wikipedia_page_has_no_age():
    page = FetchResult(url="u", content="x", published_at="2002-05-14")
    assert search._published(result("Pier", "https://en.wikipedia.org/wiki/Pier"), page) == ""
    assert search._published(result("Pier", "https://news.example/pier"), page) == "2002-05-14"


@pytest.mark.parametrize(
    "listed, page, expected",
    [
        # glued from several results, or a bare URL: the page's title
        ("Pier reopens ...Pier closes ...Pier sold ...", "Pier reopens next summer", "Pier reopens next summer"),
        ("news.example/local/16980", "Harbor board meets on Tuesday", "Harbor board meets on Tuesday"),
        # cut off: the page's title when it continues it
        ("Harbor board votes to rebuild pier ...", "Harbor board votes to rebuild pier after storm", "Harbor board votes to rebuild pier after storm"),
        ("What are tides and how ...", "Winterizing your boat: a checklist", "What are tides and how ..."),
        # otherwise the result's
        ("Pier reopens next summer", "Marina news and events", "Pier reopens next summer"),
        ("Pier reopens next summer", None, "Pier reopens next summer"),
    ],
)
def test_a_source_keeps_its_result_title_unless_that_is_broken(listed, page, expected):
    assert search._source_title(result(listed, "u"), FetchResult(url="u", content="x", title=page)) == expected


def test_news_results_take_the_publisher_without_msn():
    item = {"title": "Pier closes", "body": "b", "url": "https://www.msn.com/en-us/news/aa-AB12CD3", "date": "2026-10-20T00:00:00+00:00", "source": "Coast Gazette on MSN"}
    assert search._from_news(item) == {
        "title": "Pier closes",
        "body": "b",
        "href": "https://www.msn.com/en-us/news/aa-AB12CD3",
        "date": "2026-10-20T00:00:00+00:00",
        "source": "Coast Gazette",
    }


# ---- the candidates ----------------------------------------------------------------


def gather(monkeypatch, text, news, recency=True, pool_size=10):
    cog = make_cog()

    async def fake_text(query, max_results=10):
        if isinstance(text, Exception):
            raise text
        return text

    async def fake_news(query, max_results=10):
        return news

    monkeypatch.setattr(cog, "perform_text_search", fake_text)
    monkeypatch.setattr(cog, "perform_news_search", fake_news)
    return asyncio.run(cog.gather_candidates(QUERY, recency, pool_size))


WEB = [
    result("Harbor pier damaged in storm", "https://web.example/a", "the pier lost two spans"),
    result("Locations and hours", "https://grocer.example/", "organic grocer and cafe"),
    result("Storm footage", "https://www.tiktok.com/discover/storm"),
    result("pier", "https://www.merriam-webster.com/dictionary/pier", "pier definition"),
    result("Harbor repairs begin after the storm", "https://web.example/b", "crews"),
]
NEWS = [
    result("Pier to reopen next summer, harbor board says", "https://news.example/1", "", date="2026-10-20"),
    result("Harbor pier damaged in storm", "https://other.example/copy", "the same wire story"),
]


def test_junk_is_dropped_and_news_leads_a_recency_question(monkeypatch):
    kept = gather(monkeypatch, WEB, NEWS)
    assert [r["href"] for r in kept] == [
        "https://news.example/1",
        "https://web.example/a",
        "https://web.example/b",
    ]


def test_without_a_recency_question_news_only_fills_in(monkeypatch):
    many = [result(f"Harbor story {i}", f"https://web.example/{i}", "pier") for i in range(10)]
    kept = gather(monkeypatch, many, NEWS, recency=False)
    assert [r["href"] for r in kept] == [r["href"] for r in many]
    few = gather(monkeypatch, WEB, NEWS, recency=False)
    assert [r["href"] for r in few][-1] == "https://news.example/1"


def test_a_failed_web_search_falls_back_to_the_news(monkeypatch):
    kept = gather(monkeypatch, search.SearchBackendError("down"), NEWS)
    assert [r["href"] for r in kept] == ["https://news.example/1", "https://other.example/copy"]


def test_no_engine_answering_is_an_error(monkeypatch):
    with pytest.raises(search.SearchBackendError):
        gather(monkeypatch, search.SearchBackendError("down"), [])


# ---- the model's picks --------------------------------------------------------------


def test_picks_are_deduplicated_and_topped_up(monkeypatch):
    seen = {}

    async def fake_llm(**kwargs):
        seen["prompt"] = kwargs["messages"][1]["content"]
        return reply("3 3 1")

    monkeypatch.setattr(search, "async_chat_completion", fake_llm)
    pool = [result(f"Story {i}", f"https://web.example/{i}", "pier", date="2026-09-17" if i == 3 else "") for i in range(6)]
    picked = asyncio.run(make_cog().select_articles(pool, 4, True, None))
    assert [r["href"][-1] for r in picked] == ["3", "1", "0", "2"]
    assert "Date: 2026-09-17 (about 5 weeks ago)" in seen["prompt"]


# ---- the sources ----------------------------------------------------------------------


ARTICLE = "The harbor board said the pier will reopen next summer after storm repairs. " * 6


def test_failed_and_useless_pages_are_replaced(monkeypatch):
    pages = {
        "https://a.example/": None,  # blocked
        "https://b.example/": FetchResult(url="https://b.example/", content=ARTICLE),
        # a paywall: the story's title over its terms of use
        "https://c.example/": FetchResult(url="https://c.example/", content="Subscribe to keep reading. " * 20, title="Harbor pier damaged"),
        "https://d.example/": FetchResult(url="https://d.example/", content="pier"),  # stub
        "https://e.example/": FetchResult(url="https://e.example/", content=ARTICLE),
        "https://f.example/": FetchResult(url="https://f.example/", content=ARTICLE),
        "https://g.example/": FetchResult(url="https://g.example/", content=ARTICLE),
    }
    asked = []

    async def fake_fetch(article):
        asked.append(article["href"])
        return article, pages[article["href"]]

    cog = make_cog()
    monkeypatch.setattr(cog, "_fetch_one", fake_fetch)
    ranked = [result(u, u) for u in pages]
    got = asyncio.run(cog.fetch_sources(ranked, 3, ["harbor", "pier"]))
    assert [meta["href"] for meta, _ in got] == ["https://b.example/", "https://e.example/", "https://f.example/"]
    assert "https://g.example/" not in asked  # three were enough


def test_an_msn_link_is_fetched_from_the_publisher(monkeypatch):
    async def resolve(url):
        return "https://gazette.example/pier"

    async def fetch(session, url):
        return FetchResult(url=url, content=ARTICLE)

    monkeypatch.setattr(search, "resolve_article_url", resolve)
    monkeypatch.setattr(search, "fetch_url", fetch)
    msn = result("Pier", "https://www.msn.com/en-us/news/other/pier/ar-AB12CD3", source="MSN")
    meta, res = asyncio.run(make_cog()._fetch_one(msn))
    assert meta["href"] == res.url == "https://gazette.example/pier"
    assert meta["source"] == ""  # the publisher's host names it, not "MSN"


# ---- what the model and the reader see -------------------------------------------------


def test_the_summary_prompt_and_sources_say_how_old_each_article_is():
    cog = make_cog()
    articles = [
        (result("Pier to reopen", "https://news.example/1", source="Coast Gazette"), FetchResult(url="u", content=ARTICLE, published_at="2026-10-20", source="news.example")),
        (result("Harbor history", "https://web.example/h"), FetchResult(url="u", content=ARTICLE)),
    ]
    prompt = cog._build_content_prompt(QUERY, articles, 2000, TODAY)
    assert "Source: Coast Gazette\nURL: https://news.example/1\nPublished: 2026-10-20 (2 days ago)" in prompt
    assert "Published: unknown" in prompt
    (embed,) = cog._build_embeds(QUERY, "summary", articles, 9.5, 2, 5, False, False, today=TODAY)
    assert embed.fields[0].value.split("\n")[0] == "1. [Pier to reopen](https://news.example/1) · 2 days ago"
    assert embed.footer.text == "9.50s · sources: 2/5"


def test_the_command_end_to_end(monkeypatch):
    cog = make_cog()
    sent = []

    class Followup:
        async def send(self, *args, **kwargs):
            sent.append((args, kwargs))

    class Response:
        async def defer(self):
            pass

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=1, __str__=lambda self: "someone"),
        channel=SimpleNamespace(name="general", topic=None),
        response=Response(),
        followup=Followup(),
    )

    async def fake_text(query, max_results=10):
        return WEB

    async def fake_news(query, max_results=10):
        return NEWS

    async def fake_fetch(article):
        return article, FetchResult(url=article["href"], content=ARTICLE, published_at="2026-10-20")

    calls = []

    async def fake_llm(**kwargs):
        calls.append(kwargs)
        return reply("[Coast Gazette](https://news.example/1) says the pier reopens next summer.")

    monkeypatch.setattr(cog, "perform_text_search", fake_text)
    monkeypatch.setattr(cog, "perform_news_search", fake_news)
    monkeypatch.setattr(cog, "_fetch_one", fake_fetch)
    monkeypatch.setattr(search, "async_chat_completion", fake_llm)
    asyncio.run(cog.search_command.callback(cog, interaction, QUERY))

    ((args, kwargs),) = sent
    embed = kwargs["embed"]
    assert embed.description.startswith("[Coast Gazette](https://news.example/1)")
    assert embed.footer.text.endswith("sources: 3/5")
    assert len(calls) == 1  # three candidates, so the model wasn't asked to choose
