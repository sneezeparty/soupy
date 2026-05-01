"""Tests for installer.state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from installer import state


def test_secret_keys_are_detected():
    assert state.is_secret_key("DISCORD_TOKEN")
    assert state.is_secret_key("BLUESKY_APP_PASSWORD")
    assert state.is_secret_key("OPENAI_API_KEY")
    assert state.is_secret_key("MY_SECRET_THING")
    assert not state.is_secret_key("CHANNEL_IDS")
    assert not state.is_secret_key("OPENAI_BASE_URL")
    assert not state.is_secret_key("LOCAL_CHAT")


def test_save_scrubs_secrets(tmp_path: Path):
    p = tmp_path / "state.json"
    data = {
        "DISCORD_TOKEN": "actual-token-here",
        "BLUESKY_APP_PASSWORD": "xxxx-xxxx",
        "CHANNEL_IDS": "111,222",
        "image_gen_mode": "none",
    }
    state.save(p, data)
    on_disk = json.loads(p.read_text())
    assert on_disk["DISCORD_TOKEN"] == state.PLACEHOLDER
    assert on_disk["BLUESKY_APP_PASSWORD"] == state.PLACEHOLDER
    assert on_disk["CHANNEL_IDS"] == "111,222"
    assert on_disk["image_gen_mode"] == "none"


def test_save_does_not_mutate_input(tmp_path: Path):
    p = tmp_path / "state.json"
    data = {"DISCORD_TOKEN": "actual-token"}
    state.save(p, data)
    assert data["DISCORD_TOKEN"] == "actual-token"


def test_load_returns_empty_dict_for_missing_file(tmp_path: Path):
    p = tmp_path / "missing.json"
    assert state.load(p) == {}


def test_load_returns_empty_dict_for_corrupt_file(tmp_path: Path):
    p = tmp_path / "corrupt.json"
    p.write_text("not json {")
    assert state.load(p) == {}


def test_round_trip_with_secrets(tmp_path: Path):
    p = tmp_path / "state.json"
    original = {
        "DISCORD_TOKEN": "real-token",
        "OWNER_IDS": "12345678901234567",
        "image_gen_mode": "local_cuda",
    }
    state.save(p, original)
    loaded = state.load(p)
    # Secret should be the placeholder; non-secrets round-trip cleanly.
    assert loaded["DISCORD_TOKEN"] == state.PLACEHOLDER
    assert loaded["OWNER_IDS"] == "12345678901234567"
    assert loaded["image_gen_mode"] == "local_cuda"


def test_needs_reprompt_flags_placeholders():
    s = {
        "DISCORD_TOKEN": state.PLACEHOLDER,
        "OWNER_IDS": "12345",
        "GUILD_ID": "",
    }
    out = state.needs_reprompt(s, ["DISCORD_TOKEN", "OWNER_IDS", "GUILD_ID", "MISSING"])
    assert "DISCORD_TOKEN" in out
    assert "GUILD_ID" in out
    assert "MISSING" in out
    assert "OWNER_IDS" not in out


def test_clear_removes_state_file(tmp_path: Path):
    p = tmp_path / "state.json"
    p.write_text("{}")
    state.clear(p)
    assert not p.exists()
    # Idempotent.
    state.clear(p)


def test_scrub_handles_nested_structures(tmp_path: Path):
    p = tmp_path / "state.json"
    data = {
        "outer": {
            "DISCORD_TOKEN": "secret",
            "value": "kept",
        },
        "list": [{"BLUESKY_APP_PASSWORD": "secret"}, "ok"],
    }
    state.save(p, data)
    on_disk = json.loads(p.read_text())
    assert on_disk["outer"]["DISCORD_TOKEN"] == state.PLACEHOLDER
    assert on_disk["outer"]["value"] == "kept"
    assert on_disk["list"][0]["BLUESKY_APP_PASSWORD"] == state.PLACEHOLDER
    assert on_disk["list"][1] == "ok"
