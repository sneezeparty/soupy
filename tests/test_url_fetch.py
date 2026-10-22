"""soupy.url_fetch: the User-Agent chain and the article date."""

import asyncio

import pytest

from soupy import url_fetch

PAGE = """<html><head>
<title>Harbor town votes to rebuild its pier</title>
<meta property="article:published_time" content="2026-09-17T08:00:00Z">
</head><body><article>
<h1>Harbor town votes to rebuild its pier</h1>
<p>The council of an invented harbor town voted on Tuesday to rebuild the pier that winter
storms broke in two, after a long meeting in which residents argued about the cost, the
schedule and whether the new pier should be built further from the cliffs.</p>
<p>Work is set to start in the spring and take about eighteen months, the town's engineer
said, with the fishing dock reopening first so the boats can tie up again by summer.</p>
</article></body></html>"""


class FakeResponse:
    def __init__(self, status: int, body: str = ""):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._body


class FakeSession:
    """Answers 200 to a User-Agent containing ``accepts``, ``refusal`` to the rest."""

    def __init__(self, accepts: str = "", refusal: int = 403):
        self.accepts = accepts
        self.refusal = refusal
        self.agents = []

    def get(self, url, headers=None, timeout=None):
        agent = headers["User-Agent"]
        self.agents.append(agent)
        if self.accepts and self.accepts in agent:
            return FakeResponse(200, PAGE)
        return FakeResponse(self.refusal)


@pytest.fixture(autouse=True)
def empty_cache():
    url_fetch._cache.clear()
    yield
    url_fetch._cache.clear()


def fetch(session, url="https://news.example/pier"):
    return asyncio.run(url_fetch.fetch_url(session, url))


def test_a_browser_user_agent_goes_first():
    session = FakeSession(accepts="Chrome")
    result = fetch(session)
    assert result is not None and len(session.agents) == 1
    assert session.agents[0].startswith("Mozilla/5.0 (Windows")


def test_a_refusal_tries_the_link_preview_crawlers():
    # AP News answers 403 to everything but Slackbot.
    session = FakeSession(accepts="Slackbot")
    result = fetch(session)
    assert result is not None and "rebuild the pier" in result.content
    assert session.agents == list(url_fetch.USER_AGENTS)


def test_a_missing_page_is_not_retried():
    session = FakeSession(refusal=404)
    assert fetch(session) is None
    assert len(session.agents) == 1


def test_every_agent_refused():
    session = FakeSession(refusal=403)
    assert fetch(session) is None
    assert len(session.agents) == len(url_fetch.USER_AGENTS)


def test_the_article_date_and_title_are_read():
    result = fetch(FakeSession(accepts="Chrome"))
    assert result.published_at == "2026-09-17"
    assert result.title == "Harbor town votes to rebuild its pier"
    assert result.source == "news.example"
