"""Tests for soupy_prompts."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import soupy_prompts


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts with a fresh cache so legacy-warning state doesn't leak."""
    soupy_prompts.clear_cache()
    yield
    soupy_prompts.clear_cache()


def test_loads_default_file():
    """The shipped prompts/behaviour.default.txt must resolve."""
    text = soupy_prompts.load_prompt("behaviour", fallback="STUB")
    assert "soupy dafoe" in text.lower()
    assert text != "STUB"


def test_env_var_takes_precedence_over_default():
    with patch.dict(os.environ, {"BEHAVIOUR": "I am from env."}):
        text = soupy_prompts.load_prompt("behaviour")
    assert text == "I am from env."


def test_user_override_file_takes_precedence_over_default(tmp_path: Path, monkeypatch):
    """prompts/<name>.txt beats prompts/<name>.default.txt."""
    monkeypatch.setattr(soupy_prompts, "PROMPTS_DIR", tmp_path)
    (tmp_path / "behaviour.default.txt").write_text("DEFAULT", encoding="utf-8")
    (tmp_path / "behaviour.txt").write_text("USER OVERRIDE", encoding="utf-8")
    # No env var set
    monkeypatch.delenv("BEHAVIOUR", raising=False)
    assert soupy_prompts.load_prompt("behaviour") == "USER OVERRIDE"


def test_falls_back_to_caller_when_nothing_else_present(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(soupy_prompts, "PROMPTS_DIR", tmp_path)
    monkeypatch.delenv("BEHAVIOUR", raising=False)
    text = soupy_prompts.load_prompt("behaviour", fallback="LAST RESORT")
    assert text == "LAST RESORT"


def test_unknown_prompt_falls_back_cleanly(tmp_path: Path, monkeypatch):
    """A name with no file and no env mapping returns the caller fallback."""
    monkeypatch.setattr(soupy_prompts, "PROMPTS_DIR", tmp_path)
    text = soupy_prompts.load_prompt("brand_new_prompt", fallback="OK")
    assert text == "OK"


def test_cache_returns_same_value_on_second_call():
    first = soupy_prompts.load_prompt("nineball", fallback="")
    second = soupy_prompts.load_prompt("nineball", fallback="")
    assert first == second
    assert "8-ball" in first or "8 ball" in first or "magic" in first.lower()


def test_clear_cache_forgets_legacy_warning(monkeypatch, caplog):
    """Without clear_cache, the deprecation warning would fire only on the first
    lookup. clear_cache resets that so /reload_env can re-warn after re-edit."""
    monkeypatch.setenv("BEHAVIOUR", "via env")
    soupy_prompts.clear_cache()
    with caplog.at_level("INFO", logger="soupy_prompts"):
        soupy_prompts.load_prompt("behaviour")
    first_warnings = [r for r in caplog.records if "legacy env var" in r.message]
    assert len(first_warnings) == 1, f"expected one legacy warning, got {len(first_warnings)}"
