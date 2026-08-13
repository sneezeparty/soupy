"""Tests for /soupystock symbol resolution.

The failure these guard against is the quiet one: Finnhub's ``/search`` returns
a plausible-looking hit for a query it has no real match for, and the bot ships
a confident quote card for the wrong company.

pytest-asyncio isn't a dependency, so the async paths are driven with
``asyncio.run`` (same convention as ``test_process_chat_message.py``).
"""

from __future__ import annotations

import asyncio

from soupy.cogs.stock import StockCog, _score_search_hit


def hit(symbol, description, type_="Common Stock"):
    return {"symbol": symbol, "description": description, "type": type_}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_coincidental_substring_match_is_rejected():
    # "spacex" is a substring of "metaspacex", but shares no whole word with it.
    assert _score_search_hit("spacex", hit("1796.HK", "Metaspacex Ltd")) <= 0


def test_unrelated_us_common_stock_is_rejected():
    # The regression that motivated this: a plain US common stock used to earn
    # a baseline of 4 with zero relevance, which cleared the threshold on its
    # own and returned the wrong company.
    score = _score_search_hit("tesla motors", hit("TSLX", "Sixth Street Specialty Lending"))
    assert score <= 0


def test_partial_company_name_still_matches():
    # One of two query words matches, which is how real company names behave.
    assert _score_search_hit("tesla motors", hit("TSLA", "Tesla Inc")) > 0


def test_exact_symbol_counts_as_full_relevance():
    # A ticker query shares no words with its own company name.
    assert _score_search_hit("MSFT", hit("MSFT", "Microsoft Corp")) > 0


def test_leading_name_match_outranks_a_same_word_competitor():
    primary = _score_search_hit("apple", hit("AAPL", "Apple Inc"))
    other = _score_search_hit("apple", hit("APLE", "Apple Hospitality REIT", "REIT"))
    assert primary > other > 0


def test_foreign_listing_ranks_below_its_us_equivalent():
    us = _score_search_hit("siemens", hit("SIEGY", "Siemens AG"))
    foreign = _score_search_hit("siemens", hit("SIE.DE", "Siemens AG"))
    assert us > foreign


def test_empty_query_is_rejected():
    assert _score_search_hit("   ", hit("AAPL", "Apple Inc")) <= 0
    assert _score_search_hit("!!!", hit("AAPL", "Apple Inc")) <= 0


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def make_cog(responses):
    """A StockCog that answers _finnhub_get from a {(path, symbol_or_q): payload} map."""
    cog = object.__new__(StockCog)

    async def fake_get(path, params):
        key = (path, params.get("symbol") or params.get("q"))
        return responses.get(key, {})

    cog._finnhub_get = fake_get
    return cog


def resolve(cog, query):
    return asyncio.run(cog._resolve_symbol(query))


def test_ticker_shortcuts_search():
    cog = make_cog({("/quote", "AAPL"): {"c": 231.4}})
    assert resolve(cog, "AAPL") == "AAPL"


def test_alias_is_used_when_it_has_a_live_quote():
    cog = make_cog({("/quote", "SPCX"): {"c": 41.2}})
    assert resolve(cog, "spacex") == "SPCX"


def test_stale_alias_falls_through_to_search():
    # A hardcoded alias that no longer quotes must not become a dead end: once
    # Finnhub's index catches up, /search should still be able to answer.
    cog = make_cog(
        {
            ("/quote", "SPCX"): {"c": 0},
            ("/quote", "SPACEX"): {"c": 0},
            ("/search", "spacex"): {"result": [hit("SPXX", "SpaceX Corp")]},
        }
    )
    assert resolve(cog, "spacex") == "SPXX"


def test_search_returning_only_junk_resolves_to_nothing():
    cog = make_cog(
        {
            ("/quote", "SPCX"): {"c": 0},
            ("/quote", "SPACEX"): {"c": 0},
            ("/search", "spacex"): {"result": [hit("1796.HK", "Metaspacex Ltd")]},
        }
    )
    assert resolve(cog, "spacex") is None


def test_best_scoring_hit_wins_regardless_of_order():
    cog = make_cog(
        {
            ("/search", "apple"): {
                "result": [
                    hit("APLE", "Apple Hospitality REIT", "REIT"),
                    hit("AAPL", "Apple Inc"),
                ]
            }
        }
    )
    assert resolve(cog, "apple") == "AAPL"
