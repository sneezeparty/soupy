"""Tests for installer.validators.

Network calls are mocked at the installer.http layer.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from installer import http, validators


def test_is_snowflake_accepts_valid_lengths():
    assert validators.is_snowflake("12345678901234567")  # 17
    assert validators.is_snowflake("123456789012345678")  # 18
    assert validators.is_snowflake("12345678901234567890")  # 20
    assert not validators.is_snowflake("1234")
    assert not validators.is_snowflake("123456789012345678901")  # 21
    assert not validators.is_snowflake("not-a-number")
    assert not validators.is_snowflake("12345678901234567a")


def test_parse_snowflake_list_strips_whitespace():
    out = validators.parse_snowflake_list(" 12345678901234567 , 123456789012345678 ")
    assert out == ["12345678901234567", "123456789012345678"]


def test_parse_snowflake_list_skips_empties():
    out = validators.parse_snowflake_list("12345678901234567,,, ,")
    assert out == ["12345678901234567"]


def test_parse_snowflake_list_raises_on_bad_id():
    with pytest.raises(ValueError):
        validators.parse_snowflake_list("12345678901234567,nope")


# ---------------------------------------------------------------------------
# Discord token
# ---------------------------------------------------------------------------

def test_discord_token_blank():
    result = validators.discord_token("")
    assert not result.ok


def test_discord_token_placeholder():
    result = validators.discord_token("your_discord_bot_token_here")
    assert not result.ok


def test_discord_token_success():
    with patch.object(
        http, "get_json", return_value=(200, {"id": "999", "username": "soupy"})
    ):
        result = validators.discord_token("some-token")
    assert result.ok
    assert "soupy" in result.message
    assert result.data["id"] == "999"


def test_discord_token_unauthorized():
    with patch.object(http, "get_json", return_value=(401, {"message": "401: Unauthorized"})):
        result = validators.discord_token("bad-token")
    assert not result.ok
    assert "401" in result.message


def test_discord_token_network_error():
    def boom(*_args, **_kwargs):
        raise http.HttpError(0, "connection refused", "https://discord.com/api/v10/users/@me")

    with patch.object(http, "get_json", side_effect=boom):
        result = validators.discord_token("token")
    assert not result.ok
    assert "network" in result.message.lower()


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

def test_lm_studio_models_success():
    body = {"data": [{"id": "chat-model"}, {"id": "embed-model"}]}
    with patch.object(http, "get_json", return_value=(200, body)):
        result = validators.lm_studio_models("http://localhost:1234/v1")
    assert result.ok
    assert result.data == ["chat-model", "embed-model"]


def test_lm_studio_models_empty():
    body = {"data": []}
    with patch.object(http, "get_json", return_value=(200, body)):
        result = validators.lm_studio_models("http://localhost:1234/v1")
    assert not result.ok
    assert "no models loaded" in result.message


def test_lm_studio_models_bad_status():
    with patch.object(http, "get_json", return_value=(500, "boom")):
        result = validators.lm_studio_models("http://localhost:1234/v1")
    assert not result.ok
    assert "500" in result.message


def test_lm_studio_models_unreachable():
    def boom(*_args, **_kwargs):
        raise http.HttpError(0, "connection refused", "http://localhost:1234/v1/models")

    with patch.object(http, "get_json", side_effect=boom):
        result = validators.lm_studio_models("http://localhost:1234/v1")
    assert not result.ok


def test_lm_studio_models_strips_trailing_slash():
    captured = {}

    def fake(url, **_kwargs):
        captured["url"] = url
        return 200, {"data": [{"id": "m"}]}

    with patch.object(http, "get_json", side_effect=fake):
        validators.lm_studio_models("http://localhost:1234/v1/")
    assert captured["url"] == "http://localhost:1234/v1/models"


# ---------------------------------------------------------------------------
# SD backend
# ---------------------------------------------------------------------------

def test_sd_backend_health_ok():
    with patch.object(http, "get_json", return_value=(200, {"status": "ok", "model": "sd35"})):
        result = validators.sd_backend("http://localhost:8000")
    assert result.ok
    assert "sd35" in result.message


def test_sd_backend_falls_back_to_root():
    calls = []

    def fake(url, **_kwargs):
        calls.append(url)
        if url.endswith("/health"):
            raise http.HttpError(0, "no route", url)
        return 200, {"message": "ok"}

    with patch.object(http, "get_json", side_effect=fake):
        result = validators.sd_backend("http://localhost:8000")
    assert result.ok
    assert any(c.endswith("/health") for c in calls)
    assert any(c.endswith("/") for c in calls if not c.endswith("/health"))


def test_sd_backend_unreachable():
    def boom(*_args, **_kwargs):
        raise http.HttpError(0, "connection refused", "http://nope")

    with patch.object(http, "get_json", side_effect=boom):
        result = validators.sd_backend("http://nope:8000")
    assert not result.ok


def test_url_looks_valid():
    assert validators.url_looks_valid("http://localhost")
    assert validators.url_looks_valid("https://example.com")
    assert not validators.url_looks_valid("localhost:8000")
    assert not validators.url_looks_valid("ftp://example.com")
