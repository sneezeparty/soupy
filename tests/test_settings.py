"""Tests for soupy_settings.

Verify type coercion, defaults, list parsing, reload semantics. Each
test isolates its own env via monkeypatch so the suite stays
deterministic.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import soupy_settings


@pytest.fixture
def fresh_settings():
    """Each test gets a brand-new Settings object so cached_property lookups
    don't leak between tests."""
    return soupy_settings.Settings()


def test_string_env_with_default(monkeypatch, fresh_settings):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert fresh_settings.openai_base_url == "http://localhost:1234/v1"


def test_string_env_with_value(monkeypatch, fresh_settings):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://10.0.0.5:1234/v1")
    assert fresh_settings.openai_base_url == "http://10.0.0.5:1234/v1"


def test_int_with_default(monkeypatch, fresh_settings):
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    assert fresh_settings.max_tokens == 4096


def test_int_parses(monkeypatch, fresh_settings):
    monkeypatch.setenv("MAX_TOKENS", "8192")
    assert fresh_settings.max_tokens == 8192


def test_int_falls_back_on_garbage(monkeypatch, fresh_settings, caplog):
    monkeypatch.setenv("MAX_TOKENS", "not-a-number")
    with caplog.at_level("WARNING", logger="soupy_settings"):
        v = fresh_settings.max_tokens
    assert v == 4096
    assert any("not an int" in r.message for r in caplog.records)


def test_float_parses(monkeypatch, fresh_settings):
    monkeypatch.setenv("CHAT_TEMPERATURE", "0.9")
    assert fresh_settings.chat_temperature == pytest.approx(0.9)


def test_bool_truthy_values(monkeypatch, fresh_settings):
    for raw in ["1", "true", "TRUE", "yes", "Y", "on"]:
        s = soupy_settings.Settings()
        monkeypatch.setenv("ENABLE_VISION", raw)
        assert s.enable_vision is True, f"expected truthy for {raw!r}"


def test_bool_falsy_values(monkeypatch, fresh_settings):
    for raw in ["0", "false", "no", "off", ""]:
        s = soupy_settings.Settings()
        monkeypatch.setenv("ENABLE_VISION", raw)
        assert s.enable_vision is False, f"expected falsy for {raw!r}"


def test_int_list_parses(monkeypatch, fresh_settings):
    monkeypatch.setenv("CHANNEL_IDS", "111, 222 ,, 333")
    assert fresh_settings.channel_ids == [111, 222, 333]


def test_int_list_empty(monkeypatch, fresh_settings):
    monkeypatch.delenv("CHANNEL_IDS", raising=False)
    assert fresh_settings.channel_ids == []


def test_int_list_skips_garbage(monkeypatch, fresh_settings, caplog):
    monkeypatch.setenv("OWNER_IDS", "111, abc, 222")
    with caplog.at_level("WARNING", logger="soupy_settings"):
        v = fresh_settings.owner_ids
    assert v == [111, 222]


def test_str_list_with_default(monkeypatch, fresh_settings):
    monkeypatch.delenv("SOUPY_TRIGGER_KEYWORDS", raising=False)
    assert fresh_settings.trigger_keywords == ["soup", "gumbo"]


def test_str_list_overrides_default(monkeypatch, fresh_settings):
    monkeypatch.setenv("SOUPY_TRIGGER_KEYWORDS", "stew, broth, ramen")
    assert fresh_settings.trigger_keywords == ["stew", "broth", "ramen"]


def test_openai_api_key_falls_through_to_local_key(monkeypatch, fresh_settings):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_KEY", "lm-studio")
    assert fresh_settings.openai_api_key == "lm-studio"


def test_openai_api_key_takes_precedence(monkeypatch, fresh_settings):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
    monkeypatch.setenv("LOCAL_KEY", "ignored")
    assert fresh_settings.openai_api_key == "sk-real-key"


def test_guild_id_optional(monkeypatch, fresh_settings):
    monkeypatch.delenv("GUILD_ID", raising=False)
    assert fresh_settings.guild_id is None
    s = soupy_settings.Settings()
    monkeypatch.setenv("GUILD_ID", "987654321098765432")
    assert s.guild_id == 987654321098765432


def test_reload_invalidates_cache(monkeypatch, fresh_settings):
    monkeypatch.setenv("MAX_TOKENS", "1000")
    assert fresh_settings.max_tokens == 1000
    # Change the env var; without reload the cached value sticks.
    monkeypatch.setenv("MAX_TOKENS", "2000")
    assert fresh_settings.max_tokens == 1000  # still cached
    fresh_settings.reload()
    assert fresh_settings.max_tokens == 2000


def test_reload_drops_only_cached_properties(monkeypatch, fresh_settings):
    monkeypatch.setenv("MAX_TOKENS", "1000")
    _ = fresh_settings.max_tokens  # populate cache
    fresh_settings.some_random_attr = "should-survive"
    fresh_settings.reload()
    assert fresh_settings.some_random_attr == "should-survive"


def test_singleton_exists():
    assert isinstance(soupy_settings.settings, soupy_settings.Settings)


def test_singleton_supports_reload(monkeypatch):
    # The module-level singleton should be reloadable too.
    monkeypatch.setenv("MAX_TOKENS", "100")
    soupy_settings.settings.reload()
    assert soupy_settings.settings.max_tokens == 100
