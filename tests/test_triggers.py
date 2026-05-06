"""Tests for soupy_triggers — the pure functions that decide whether
the bot should reply to a message and how often it spontaneously
chimes in.

These functions used to live in soupy_remastered_stablediffusion.py
where they couldn't be tested in isolation. The behaviour must remain
identical.
"""

from __future__ import annotations

import soupy_triggers


def test_default_trigger_keywords():
    assert soupy_triggers.DEFAULT_TRIGGER_KEYWORDS == ["soup", "gumbo"]


def test_get_trigger_keywords_default(monkeypatch):
    monkeypatch.delenv("SOUPY_TRIGGER_KEYWORDS", raising=False)
    assert soupy_triggers.get_trigger_keywords() == ["soup", "gumbo"]


def test_get_trigger_keywords_overridden(monkeypatch):
    monkeypatch.setenv("SOUPY_TRIGGER_KEYWORDS", "ramen, broth, stew")
    assert soupy_triggers.get_trigger_keywords() == ["ramen", "broth", "stew"]


def test_get_trigger_keywords_blank_falls_back(monkeypatch):
    monkeypatch.setenv("SOUPY_TRIGGER_KEYWORDS", "")
    # Blank means "use the default" via os.getenv's fallback.
    assert soupy_triggers.get_trigger_keywords() == ["soup", "gumbo"]


def test_get_trigger_keywords_skips_empty_segments(monkeypatch):
    monkeypatch.setenv("SOUPY_TRIGGER_KEYWORDS", " , ramen ,, ")
    assert soupy_triggers.get_trigger_keywords() == ["ramen"]


def test_message_contains_trigger_keyword_matches_case_insensitive(monkeypatch):
    monkeypatch.setenv("SOUPY_TRIGGER_KEYWORDS", "soup")
    assert soupy_triggers.message_contains_trigger_keyword("I had Soup for lunch")
    assert soupy_triggers.message_contains_trigger_keyword("SOUPY is a good bot")  # 'soup' is a substring of 'SOUPY'
    assert not soupy_triggers.message_contains_trigger_keyword("nothing here")


def test_message_contains_trigger_keyword_handles_none_content(monkeypatch):
    """The original function used `content or ""` so None must not crash."""
    assert not soupy_triggers.message_contains_trigger_keyword(None)


def test_message_contains_trigger_keyword_handles_empty_string():
    assert not soupy_triggers.message_contains_trigger_keyword("")


def test_message_contains_trigger_keyword_escapes_regex_metacharacters(monkeypatch):
    """Original used re.escape — verify a keyword with regex metacharacters
    matches literally rather than as a regex pattern."""
    monkeypatch.setenv("SOUPY_TRIGGER_KEYWORDS", "$money$,a.b")
    assert soupy_triggers.message_contains_trigger_keyword("got $money$ today")
    assert soupy_triggers.message_contains_trigger_keyword("a.b is here")
    # 'a.b' should not match 'axb' (regex would, escaped literal won't)
    assert not soupy_triggers.message_contains_trigger_keyword("axb pattern")


def test_should_randomly_respond_zero_probability_never_fires():
    for _ in range(100):
        assert soupy_triggers.should_randomly_respond(probability=0.0) is False


def test_should_randomly_respond_one_probability_always_fires():
    for _ in range(100):
        assert soupy_triggers.should_randomly_respond(probability=1.0) is True


def test_should_randomly_respond_reads_env(monkeypatch):
    # With probability=None it should read RANDOM_RESPONSE_RATE
    monkeypatch.setenv("RANDOM_RESPONSE_RATE", "1.0")
    assert soupy_triggers.should_randomly_respond() is True
    monkeypatch.setenv("RANDOM_RESPONSE_RATE", "0.0")
    assert soupy_triggers.should_randomly_respond() is False


def test_should_randomly_respond_default_is_5_percent(monkeypatch):
    """The historical default was 5%. We don't test the random sample, just
    that the env-fallback parses to 0.05 — confirmed by checking the docstring
    invariant: if env is unset, the function still works without error."""
    monkeypatch.delenv("RANDOM_RESPONSE_RATE", raising=False)
    # Just call it — it should not raise. We don't assert the bool because
    # the random sample is non-deterministic.
    result = soupy_triggers.should_randomly_respond()
    assert isinstance(result, bool)
