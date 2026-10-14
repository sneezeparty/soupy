"""End-to-end integration test for the chat reply pipeline.

This exercises ``process_chat_message`` — the ~540-line core that turns a Discord
message into a reply — with the OpenAI client, RAG, DB, and Discord objects all
mocked. It's the regression net for the most-edited, most-fragile surface in the
codebase: token budgeting, system-prompt assembly, RAG injection, candidate
generation/judging, response scrubbing, and message splitting all run for real;
only I/O is faked.

pytest-asyncio isn't a dependency, so each test drives the coroutine with
``asyncio.run`` rather than an async test function.

Note: ``process_chat_message`` wraps its whole body in ``try/except Exception``
that only logs, so a broken mock surfaces as "channel.send was never called"
rather than a raised error — the assertions on ``send`` are what catch setup
problems.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import soupy_remastered_stablediffusion as bot  # noqa: E402  (conftest sets env first)

NOW = datetime(2026, 10, 6, 6, 14, tzinfo=ZoneInfo("America/Los_Angeles"))  # a Tuesday


class _AsyncCM:
    """Minimal stand-in for ``channel.typing()`` (an async context manager).

    ``process_chat_message`` enters two of these (an outer and an inner), so a
    fresh instance is handed out per call.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_message(content: str = "hello soupy", *, guild_id: int = 222, channel_id: int = 111):
    """Build a discord.Message mock with the attributes the chat path touches."""
    msg = MagicMock(name="message")
    msg.id = 123456789
    msg.content = content
    msg.author = MagicMock(name="author")
    msg.author.id = 987654321
    msg.author.name = "test_user"
    msg.author.display_name = "TestUser"
    msg.author.bot = False
    msg.guild = MagicMock(name="guild")
    msg.guild.id = guild_id
    msg.channel = MagicMock(name="channel")
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock(name="send", return_value=MagicMock(id=999))
    msg.channel.typing = MagicMock(side_effect=lambda: _AsyncCM())
    return msg


def _base_patches(
    candidates, *, rag_enabled=False, self_md_enabled=False, rag_block="", time_aware=False, history=()
):
    """Patch every I/O seam in the bot module; return a contextlib-style list.

    Returns a dict of patchers already started — caller is responsible for
    stopping them (we use ``patch`` as a stack via ``contextlib.ExitStack``).
    """
    return {
        "get_guild_behaviour": patch.object(
            bot, "get_guild_behaviour", AsyncMock(return_value="you are soupy")
        ),
        "is_rag_enabled": patch.object(bot, "is_rag_enabled", MagicMock(return_value=rag_enabled)),
        "is_self_md_enabled": patch.object(
            bot, "is_self_md_enabled", MagicMock(return_value=self_md_enabled)
        ),
        "is_time_awareness_enabled": patch.object(
            bot, "is_time_awareness_enabled", MagicMock(return_value=time_aware)
        ),
        # The clock is frozen; it is only read when time awareness is on.
        "now_local": patch.object(bot.timeaware, "now_local", MagicMock(return_value=NOW)),
        "fetch_recent_messages": patch.object(
            bot, "fetch_recent_messages", AsyncMock(return_value=list(history))
        ),
        "generate_parallel_candidates": patch.object(
            bot, "generate_parallel_candidates", AsyncMock(return_value=candidates)
        ),
        "increment_user_stat": patch.object(bot, "increment_user_stat", AsyncMock()),
        "archive_sent_message": patch.object(bot, "archive_sent_message", MagicMock()),
        "build_rag_retrieval_query": patch.object(
            bot, "build_rag_retrieval_query", MagicMock(return_value="retrieval query")
        ),
        "fetch_rag_context_for_query": patch.object(
            bot, "fetch_rag_context_for_query", AsyncMock(return_value=rag_block)
        ),
    }


def _run(message, patches):
    """Start all patches, run the coroutine, stop the patches."""
    import contextlib

    with contextlib.ExitStack() as stack:
        started = {name: stack.enter_context(p) for name, p in patches.items()}
        asyncio.run(bot.process_chat_message(message, []))
        return started


# ---------------------------------------------------------------------------
# Basic happy path
# ---------------------------------------------------------------------------


def test_basic_flow_sends_reply():
    """A plain message produces exactly one channel.send with the model's reply."""
    msg = _make_message("hello soupy")
    patches = _base_patches(["hello back"])
    _run(msg, patches)

    msg.channel.send.assert_called_once()
    sent = msg.channel.send.call_args[0][0]
    assert sent == "hello back"


def test_single_candidate_skips_the_judge():
    """With one candidate, judge_best_of_candidates must not be called."""
    msg = _make_message()
    patches = _base_patches(["only one"])
    with patch.object(bot, "judge_best_of_candidates", AsyncMock()) as judge:
        _run(msg, patches)
    judge.assert_not_called()
    msg.channel.send.assert_called_once()


def test_multiple_candidates_invoke_the_judge():
    """With >1 candidate, the judge picks the index that gets sent."""
    msg = _make_message()
    patches = _base_patches(["candidate A", "candidate B"])
    # Judge picks index 1 -> "candidate B"
    with patch.object(bot, "judge_best_of_candidates", AsyncMock(return_value=1)) as judge:
        _run(msg, patches)
    judge.assert_called_once()
    sent = msg.channel.send.call_args[0][0]
    assert sent == "candidate B"


# ---------------------------------------------------------------------------
# RAG injection
# ---------------------------------------------------------------------------


def test_rag_enabled_fetches_and_injects_context():
    """When RAG is on and the message is non-empty, the retrieved block is fetched
    and threaded into the messages handed to the LLM."""
    msg = _make_message("do I own a cat?")
    patches = _base_patches(
        ["you mentioned a tabby once"], rag_enabled=True, rag_block="SNIPPET: user has a cat"
    )
    started = _run(msg, patches)

    started["fetch_rag_context_for_query"].assert_awaited()
    # The retrieved snippet must appear in the messages sent to the model.
    sent_messages = started["generate_parallel_candidates"].call_args.kwargs["messages"]
    blob = "\n".join(m.get("content", "") for m in sent_messages)
    assert "SNIPPET: user has a cat" in blob


def test_rag_disabled_does_not_fetch():
    """RAG off -> no retrieval call at all."""
    msg = _make_message("just chatting")
    patches = _base_patches(["sup"], rag_enabled=False)
    started = _run(msg, patches)
    started["fetch_rag_context_for_query"].assert_not_awaited()


# ---------------------------------------------------------------------------
# Response scrubbing & splitting (real clean_response / split_message run)
# ---------------------------------------------------------------------------


def test_response_metadata_is_scrubbed_end_to_end():
    """RAG leakage (timestamps, message_id) is stripped before the reply is sent."""
    polluted = "real reply here [message_id=12345] said at 2026-03-28 19:30:53"
    msg = _make_message()
    patches = _base_patches([polluted])
    _run(msg, patches)

    sent = " ".join(call.args[0] for call in msg.channel.send.call_args_list)
    assert "message_id" not in sent
    assert "2026-03-28" not in sent
    assert "real reply here" in sent


def test_long_reply_is_split_into_multiple_sends():
    """A reply far over the Discord per-message limit is split across sends."""
    long_reply = "word " * 1200  # ~6000 chars on one line after whitespace-collapse
    msg = _make_message()
    patches = _base_patches([long_reply])
    _run(msg, patches)

    assert msg.channel.send.call_count >= 2
    for call in msg.channel.send.call_args_list:
        assert len(call.args[0]) <= 1500


# ---------------------------------------------------------------------------
# Time awareness (soupy/timeaware.py)
# ---------------------------------------------------------------------------

_OLD_AND_NEW_HISTORY = (
    {"role": "user", "content": "brian: anyone watching the game", "ts": NOW - timedelta(hours=11, minutes=14)},
    {"role": "assistant", "content": "nah", "ts": NOW - timedelta(hours=11, minutes=13)},
    {"role": "user", "content": "jdk: morning", "ts": NOW - timedelta(minutes=2)},
)


def _llm_messages(started):
    return started["generate_parallel_candidates"].call_args.kwargs["messages"]


def test_time_awareness_ends_the_system_prompt_with_the_clock():
    msg = _make_message("good morning soupy")
    started = _run(msg, _base_patches(["morning"], time_aware=True))

    system = _llm_messages(started)[0]
    assert system["role"] == "system"
    assert "Right now it is tuesday, october 6, 2026, 6:14 am PDT (early morning)." in system["content"]
    assert system["content"].endswith("Never write a [chat from ...] marker yourself.\n"), "the clock block goes last"


def test_time_awareness_marks_old_history_and_keeps_markers_out_of_soupys_turns():
    msg = _make_message("soupy you there")
    started = _run(msg, _base_patches(["yep"], time_aware=True, history=_OLD_AND_NEW_HISTORY))

    sent = _llm_messages(started)
    assert all(set(m) == {"role", "content"} for m in sent), "history timestamps must never reach the LLM API"
    assert sent[1]["content"] == "[chat from yesterday evening]\nbrian: anyone watching the game"
    assert sent[2] == {"role": "assistant", "content": "nah"}
    assert sent[3]["content"].startswith("[chat from the last few minutes]\njdk: morning\n")
    assert "The channel had been quiet" not in sent[3]["content"], "the last message was two minutes ago"


def test_time_awareness_notes_when_the_message_broke_a_silence():
    msg = _make_message("soupy you there")
    msg.created_at = NOW
    started = _run(msg, _base_patches(["yep"], time_aware=True, history=_OLD_AND_NEW_HISTORY[:2]))

    task = _llm_messages(started)[-1]["content"]
    note = "This message is new. The channel had been quiet for about 11 hours before it.\nMESSAGE TO RESPOND TO:"
    assert note in task


def test_time_awareness_off_leaves_the_prompt_without_a_clock_or_markers():
    msg = _make_message("soupy you there")
    msg.created_at = NOW
    started = _run(msg, _base_patches(["yep"], history=_OLD_AND_NEW_HISTORY[:2]))

    sent = _llm_messages(started)
    assert all(set(m) == {"role", "content"} for m in sent)
    blob = "\n".join(m["content"] for m in sent)
    assert "Right now it is" not in blob and "[chat from" not in blob and "had been quiet" not in blob
    assert "brian: anyone watching the game" in blob


def test_retrieval_gets_the_same_clock_or_none_when_off():
    patches = _base_patches(["yes"], rag_enabled=True, rag_block="x", time_aware=True)
    on = _run(_make_message("do I own a cat?"), patches)
    assert on["fetch_rag_context_for_query"].call_args.kwargs["now"] == NOW
    off = _run(_make_message("do I own a cat?"), _base_patches(["yes"], rag_enabled=True, rag_block="x"))
    assert off["fetch_rag_context_for_query"].call_args.kwargs["now"] is None


def test_a_reply_that_opens_with_the_time_is_sent_whole():
    msg = _make_message("what time is it soupy")
    _run(msg, _base_patches(["its 6:14 am, go back to bed"], time_aware=True))
    assert msg.channel.send.call_args[0][0] == "its 6:14 am, go back to bed"


def test_a_history_marker_echoed_by_the_model_is_scrubbed():
    msg = _make_message("soupy you there")
    _run(msg, _base_patches(["[chat from yesterday evening] yeah i am here"], time_aware=True))
    assert msg.channel.send.call_args[0][0] == "yeah i am here"
