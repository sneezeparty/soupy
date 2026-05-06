"""Integration tests for the chat-flow helpers.

Imports `soupy_remastered_stablediffusion` and exercises the pure
helper functions that drive `process_chat_message`. The fully
end-to-end harness (mocked Discord + mocked LM Studio walking
on_message → process_chat_message → response_send) is a much bigger
project; this file is the safety net for the helpers in particular,
so future refactors that move them around have something to assert
against.

The bot is imported with `os.environ` pre-populated with harmless
defaults so the module-level `raise ValueError("No REMOVE_BG_API_URL ...")`
guard doesn't fire on a clean test run.
"""

from __future__ import annotations

import os
import sys

import pytest

# Pre-populate env so module-level guards in the bot don't abort import.
os.environ.setdefault("REMOVE_BG_API_URL", "http://localhost:8000/remove_background")
os.environ.setdefault("SD_SERVER_URL", "http://localhost:8000/")
os.environ.setdefault("SD_IMG2IMG_URL", "http://localhost:8000/sd_img2img")
os.environ.setdefault("SD_INPAINT_URL", "http://localhost:8000/sd_inpaint")

# The bot module logs a lot at import time. Suppress to keep test output clean.
import logging

logging.getLogger("soupy_prompts").setLevel(logging.ERROR)
logging.getLogger("soupy_remastered_stablediffusion").setLevel(logging.ERROR)

import soupy_remastered_stablediffusion as bot  # noqa: E402

# ---------------------------------------------------------------------------
# Token estimators
# ---------------------------------------------------------------------------


def test_estimate_tokens_floor_is_one():
    """Even an empty string returns at least 1 — protects token-budget math
    from div-by-zero / nothing-counted edge cases."""
    assert bot.estimate_tokens("") == 1
    assert bot.estimate_tokens(None) == 1


def test_estimate_tokens_grows_with_length():
    short = bot.estimate_tokens("hello")
    long = bot.estimate_tokens("hello " * 100)
    assert long > short


def test_estimate_messages_tokens_includes_overhead():
    """Each message contributes content-tokens + 4 (the per-message overhead)."""
    msgs = [{"role": "user", "content": ""}]
    # Empty content -> 1 token (estimate_tokens floor) + 4 overhead = 5
    assert bot.estimate_messages_tokens(msgs) == 5


def test_estimate_messages_tokens_handles_missing_content():
    """A message dict without `content` key shouldn't crash."""
    msgs = [{"role": "user"}]
    assert bot.estimate_messages_tokens(msgs) >= 4


# ---------------------------------------------------------------------------
# Trim-to-budget
# ---------------------------------------------------------------------------


def test_trim_keeps_messages_when_under_budget():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    out = bot.trim_messages_to_token_budget(msgs, 10000)
    assert out == msgs


def test_trim_drops_oldest_first():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    # Tiny budget — only the last one should fit
    out = bot.trim_messages_to_token_budget(msgs, 1)
    assert out == [msgs[-1]]


def test_trim_keeps_at_least_one():
    """Documented invariant: trim never returns an empty list."""
    msgs = [{"role": "user", "content": "x" * 10000}]
    out = bot.trim_messages_to_token_budget(msgs, 1)
    assert len(out) == 1


def test_trim_handles_empty():
    assert bot.trim_messages_to_token_budget([], 100) == []


# ---------------------------------------------------------------------------
# clean_response — strips wrapping quotes, em-dashes, RAG leakage
# ---------------------------------------------------------------------------


def test_clean_response_strips_outer_quotes():
    assert bot.clean_response('"hello there"') == "hello there"
    assert bot.clean_response("'hello there'") == "hello there"


def test_clean_response_strips_nested_quotes():
    assert bot.clean_response("'\"hello\"'") == "hello"


def test_clean_response_replaces_em_dashes_with_commas():
    """Em-dash policy: BEHAVIOUR forbids them. clean_response enforces."""
    assert bot.clean_response("yes — definitely") == "yes , definitely"
    assert bot.clean_response("yes – definitely") == "yes , definitely"


def test_clean_response_strips_rag_separator_lines():
    text = "actual reply text\n--- end sketches ---\n"
    assert "end sketches" not in bot.clean_response(text)


def test_clean_response_strips_iso_timestamps():
    assert "2026-03-28" not in bot.clean_response("user said hi at 2026-03-28 19:30:53")


def test_clean_response_strips_message_id_metadata():
    text = "real reply [message_id=12345]"
    assert "message_id" not in bot.clean_response(text)


def test_clean_response_strips_embed_metadata():
    text = "real reply [Embed Title: Some Article]"
    cleaned = bot.clean_response(text)
    assert "Embed" not in cleaned


def test_clean_response_strips_image_descriptions():
    text = "[image: a cat sitting on a chair] real reply"
    cleaned = bot.clean_response(text)
    assert "[image:" not in cleaned


def test_clean_response_collapses_multiple_blank_lines():
    text = "first\n\n\n\nsecond"
    cleaned = bot.clean_response(text)
    # No more than one blank line between paragraphs
    assert "\n\n\n" not in cleaned


def test_clean_response_truncates_runaway_long_output(monkeypatch):
    """When the LLM gets stuck in a repetition loop, we cap the response
    at CHAT_MAX_RESPONSE_WORDS and cut at the last sentence boundary."""
    monkeypatch.setenv("CHAT_MAX_RESPONSE_WORDS", "10")
    long = "this is a runaway sentence. " * 20
    cleaned = bot.clean_response(long)
    # Should be much shorter than the input
    assert len(cleaned.split()) <= 12, f"expected ≤12 words, got {len(cleaned.split())}"


# Note: should_bot_respond_to_message touches bot.bot.user, which discord.py
# exposes as a property that resists `setattr`. Until that function is
# refactored to take the bot user as an argument (a small chat-flow change
# that's outside this commit's behaviour-preserve constraint), the
# trigger-predicate logic is exercised in tests/test_triggers.py via
# `message_contains_trigger_keyword` and `should_randomly_respond` — the
# parts that don't depend on the bot instance.


# ---------------------------------------------------------------------------
# split_message — Discord 2000-char per-message limit handling
# ---------------------------------------------------------------------------


def test_split_message_passes_short_through():
    out = bot.split_message("short reply", max_len=1500)
    assert out == ["short reply"]


def test_split_message_splits_by_newline_first():
    """When the text fits across newline boundaries, prefer those over
    mid-line cuts."""
    text = "first paragraph\n\nsecond paragraph " + "x" * 1500
    parts = bot.split_message(text, max_len=200)
    # First chunk should end before the long second paragraph starts
    assert any("first paragraph" in p for p in parts)
    # No part exceeds the limit
    for p in parts:
        assert len(p) <= 200


def test_split_message_handles_very_long_single_line():
    text = "x" * 5000
    parts = bot.split_message(text, max_len=1500)
    assert sum(len(p) for p in parts) >= 5000
    for p in parts:
        assert len(p) <= 1500
