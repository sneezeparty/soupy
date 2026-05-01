"""Tests for installer.env_writer.

Golden-file style: the fixture under tests/fixtures/ is the input; we
assert specific properties of the rendered output rather than diffing
against a second fixture (so the test is robust to whitespace tweaks).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from installer import env_writer


FIXTURE = Path(__file__).parent / "fixtures" / "env-stable.example.fixture"


def test_render_replaces_simple_keys():
    out = env_writer.render(
        FIXTURE,
        {
            "DISCORD_TOKEN": "real-token-value",
            "OWNER_IDS": "12345678901234567",
            "CHANNEL_IDS": "111,222",
        },
    )
    assert "DISCORD_TOKEN=real-token-value" in out
    assert "DISCORD_TOKEN=your_discord_bot_token_here" not in out
    assert "OWNER_IDS=12345678901234567" in out
    assert "CHANNEL_IDS=111,222" in out


def test_render_preserves_comments_and_blanks():
    out = env_writer.render(FIXTURE, {"DISCORD_TOKEN": "x"})
    assert "# DISCORD BOT CONFIGURATION" in out
    assert "# Comment that should survive untouched." in out
    # Blank lines from the fixture should be preserved.
    assert "\n\n" in out


def test_render_preserves_multiline_quoted_value():
    out = env_writer.render(FIXTURE, {"DISCORD_TOKEN": "x"})
    assert 'BEHAVIOUR="line one' in out
    assert "line two with a" in out
    assert "line three\"" in out


def test_render_does_not_touch_keys_not_in_updates():
    out = env_writer.render(FIXTURE, {})
    assert "DISCORD_TOKEN=your_discord_bot_token_here" in out
    assert "OPENAI_BASE_URL=http://localhost:1234/v1" in out
    assert "BLUESKY_AUTO_REPLY=false" in out


def test_render_quotes_values_with_spaces():
    out = env_writer.render(FIXTURE, {"LOCAL_CHAT": "google/gemma 3 27b"})
    assert 'LOCAL_CHAT="google/gemma 3 27b"' in out


def test_render_handles_json_blob():
    payload = '{"123456789012345678": "tech and gaming"}'
    out = env_writer.render(FIXTURE, {"DAILY_POST_CHANNELS": payload})
    # Quotes inside the value get escaped to keep the file valid.
    assert 'DAILY_POST_CHANNELS="{\\"123456789012345678\\": \\"tech and gaming\\"}"' in out


def test_render_appends_unknown_keys():
    out = env_writer.render(FIXTURE, {"BRAND_NEW_KEY": "value"})
    assert "BRAND_NEW_KEY=value" in out
    assert "# Added by installer" in out


def test_write_makes_a_backup(tmp_path: Path):
    target = tmp_path / ".env-stable"
    target.write_text("DISCORD_TOKEN=old\n")
    backup = env_writer.write(FIXTURE, target, {"DISCORD_TOKEN": "new"})
    assert backup is not None
    assert backup.exists()
    assert backup.read_text() == "DISCORD_TOKEN=old\n"
    assert "DISCORD_TOKEN=new" in target.read_text()


def test_write_no_backup_when_target_absent(tmp_path: Path):
    target = tmp_path / ".env-stable"
    backup = env_writer.write(FIXTURE, target, {"DISCORD_TOKEN": "x"})
    assert backup is None
    assert target.exists()


def test_parse_existing_returns_kv(tmp_path: Path):
    target = tmp_path / ".env-stable"
    target.write_text(
        "DISCORD_TOKEN=hello\n"
        "# comment\n"
        "OWNER_IDS=12345\n"
        'BEHAVIOUR="multi\nline"\n'
    )
    out = env_writer.parse_existing(target)
    assert out["DISCORD_TOKEN"] == "hello"
    assert out["OWNER_IDS"] == "12345"
    assert out["BEHAVIOUR"] == "multi\nline"


def test_diff_masks_secrets():
    rows = env_writer.diff(
        {"DISCORD_TOKEN": "old", "CHANNEL_IDS": "111"},
        {"DISCORD_TOKEN": "new", "CHANNEL_IDS": "111,222"},
    )
    by_key = {k: (old, new) for k, old, new in rows}
    assert by_key["DISCORD_TOKEN"] == ("****", "****")
    assert by_key["CHANNEL_IDS"] == ("111", "111,222")


def test_diff_skips_unchanged_values():
    rows = env_writer.diff(
        {"OWNER_IDS": "12345"},
        {"OWNER_IDS": "12345"},
    )
    assert rows == []


def test_render_missing_template_raises(tmp_path: Path):
    bogus = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        env_writer.render(bogus, {})
