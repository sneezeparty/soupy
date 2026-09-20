"""Tests for the version-2 member profiles.

Covers the three layers separately:

* ``profile_document`` — the pure rules: dates, ids, dedup, caps, the
  "lately" age-out, the example-copy guard, and what chat gets to see.
* ``profile_llm`` — the context budget, which must never push LM Studio to
  its loaded window's ceiling.
* ``user_profiles`` — the pass loop against a throwaway SQLite archive with
  the LLM faked out: per-pass saves, resume, splitting on failure, the v1
  backup, and nightly candidate selection.

pytest-asyncio isn't a dependency, so coroutines run under ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import sqlite3
from datetime import date, datetime

import pytest
import pytz

from soupy import llm_gate
from soupy_database import profile_document as pdoc
from soupy_database import profile_llm, user_profiles

WINDOW = (date(2025, 3, 1), date(2025, 3, 20))


@pytest.fixture(autouse=True)
def quiet_chat(monkeypatch):
    """Start every test with no chat pending: other test modules drive ChatQueue, and the gate is module state."""
    monkeypatch.setattr(llm_gate, "_chat_pending", 0)
    monkeypatch.setattr(llm_gate, "_last_chat_activity", 0.0)


def _doc_with(edits, window=WINDOW, source="", **kw):
    doc, stats = pdoc.apply_edits(pdoc.new_document(), edits, window=window, source_text=source, **kw)
    return doc, stats


# ---------------------------------------------------------------------------
# Applying edits
# ---------------------------------------------------------------------------


def test_add_assigns_ids_and_keeps_message_dates():
    doc, stats = _doc_with(
        {
            "add": [
                {"section": "hobbies", "text": "restores vintage road bikes", "date": "2025-03-05"},
                {"section": "family_and_household", "text": "has a corgi named Waffles", "date": "2025-03-07"},
            ]
        }
    )
    hobbies = pdoc.section_items(doc, "hobbies")
    family = pdoc.section_items(doc, "family_and_household")
    assert hobbies[0]["id"].startswith("hob") and family[0]["id"].startswith("fam")
    assert hobbies[0]["id"] != family[0]["id"]
    assert hobbies[0]["date"] == "2025-03-05"
    assert stats.added == 2 and stats.dates_fixed == 0


def test_dates_outside_the_batch_are_replaced_with_the_batch_end():
    doc, stats = _doc_with(
        {
            "add": [
                {"section": "hobbies", "text": "plays chess online", "date": "2019-01-01"},
                {"section": "hobbies", "text": "bakes sourdough", "date": "not a date"},
            ]
        }
    )
    assert [it["date"] for it in pdoc.section_items(doc, "hobbies")] == ["2025-03-20", "2025-03-20"]
    assert stats.dates_fixed == 2


def test_update_confirms_with_last_seen_and_remove_deletes():
    doc, _ = _doc_with(
        {"add": [{"section": "work_and_education", "text": "works in hospital IT", "date": "2025-03-02"}]}
    )
    iid = pdoc.section_items(doc, "work_and_education")[0]["id"]

    later = (date(2025, 6, 1), date(2025, 6, 30))
    doc, stats = pdoc.apply_edits(
        doc,
        {"update": [{"id": iid, "date": "2025-06-10", "text": "works in hospital IT, now on the night shift"}]},
        window=later,
        source_text="",
    )
    item = pdoc.section_items(doc, "work_and_education")[0]
    assert item["date"] == "2025-03-02", "the original date is when it first came up"
    assert item["last_seen"] == "2025-06-10"
    assert "night shift" in item["text"]
    assert stats.updated == 1

    doc, stats = pdoc.apply_edits(doc, {"remove": [iid]}, window=later, source_text="")
    assert pdoc.section_items(doc, "work_and_education") == []
    assert stats.removed == 1


def test_near_duplicate_add_confirms_the_existing_item():
    doc, _ = _doc_with({"add": [{"section": "hobbies", "text": "restores vintage road bikes", "date": "2025-03-05"}]})
    doc, stats = pdoc.apply_edits(
        doc,
        {"add": [{"section": "hobbies", "text": "Restores vintage road bikes.", "date": "2025-03-18"}]},
        window=WINDOW,
        source_text="",
    )
    items = pdoc.section_items(doc, "hobbies")
    assert len(items) == 1
    assert items[0]["last_seen"] == "2025-03-18"
    assert stats.merged_duplicates == 1


def test_invalid_sections_and_unknown_relationship_ids_are_handled():
    doc, stats = _doc_with(
        {
            "add": [
                {"section": "made_up_section", "text": "x", "date": "2025-03-05"},
                {
                    "section": "relationships_with_others",
                    "text": "rivals at trivia",
                    "date": "2025-03-05",
                    "user_id": 999,
                    "name": "bob",
                },
                {
                    "section": "relationships_with_others",
                    "text": "plays co-op games together",
                    "date": "2025-03-05",
                    "user_id": 42,
                    "name": "amy",
                },
            ]
        },
        directory={42: "amy"},
    )
    rels = pdoc.section_items(doc, "relationships_with_others")
    assert stats.dropped_invalid == 1
    assert [r["user_id"] for r in rels] == [0, 42], "ids not in the member directory become 0"


def test_relationship_items_get_names_and_lose_raw_ids():
    directory = {109839305987338240: "ohsmitt", 555245126172278785: "big daddio nitro"}
    doc, _ = _doc_with(
        {
            "add": [
                # The two shapes gemma actually produced in the first sample build.
                {
                    "section": "relationships_with_others",
                    "text": "user 0 (?): user 109839305987338240; lives in Barrie",
                    "date": "2025-03-05",
                    "user_id": 0,
                },
                {
                    "section": "relationships_with_others",
                    "text": "met up with them in Yelapa",
                    "date": "2025-03-06",
                    "user_id": 555245126172278785,
                },
            ]
        },
        directory=directory,
    )
    rels = pdoc.section_items(doc, "relationships_with_others")
    assert [(r["user_id"], r["name"]) for r in rels] == [
        (109839305987338240, "ohsmitt"),
        (555245126172278785, "big daddio nitro"),
    ]
    assert rels[0]["text"] == "ohsmitt; lives in Barrie"
    assert "ohsmitt: ohsmitt" not in pdoc.render_summary(doc), "no name prefix when the text already names them"
    prompt = pdoc.render_for_prompt(doc)
    assert "{name=ohsmitt, user_id=109839305987338240}" in prompt

    iid = rels[0]["id"]
    doc, _ = pdoc.apply_edits(
        doc,
        {
            "update": [
                {
                    "id": iid,
                    "date": "2025-03-10",
                    "text": "fellow Canadian near Barrie {name=ohsmitt, user_id=109839305987338240}",
                }
            ]
        },
        window=WINDOW,
        source_text="",
        directory=directory,
    )
    assert pdoc.section_items(doc, "relationships_with_others")[0]["text"] == "fellow Canadian near Barrie"


def test_prompt_example_copies_are_dropped_unless_the_member_said_it():
    edit = {"add": [{"section": "family_and_household", "text": "has a greyhound named Pickles", "date": "2025-03-05"}]}

    doc, stats = _doc_with(edit, source="[2025-03-05 #general] my cat knocked over a plant")
    assert pdoc.section_items(doc, "family_and_household") == []
    assert stats.dropped_example_copies == 1

    doc, stats = _doc_with(edit, source="[2025-03-05 #general] pickles the greyhound ate my sandwich")
    assert len(pdoc.section_items(doc, "family_and_household")) == 1


def test_with_soupy_items_need_the_member_to_have_mentioned_soupy():
    edit = {"add": [{"section": "with_soupy", "text": "shares jokes with Soupy", "date": "2025-03-05"}]}
    doc, stats = _doc_with(edit, source="[2025-03-05 #general] new router day lol")
    assert pdoc.section_items(doc, "with_soupy") == [] and stats.dropped_ungrounded == 1
    doc, _ = _doc_with(edit, source="[2025-03-05 #general] soupy tell me a joke")
    assert len(pdoc.section_items(doc, "with_soupy")) == 1


def test_verbatim_copy_of_an_example_without_proper_nouns_is_dropped():
    example = pdoc.SECTION_BY_KEY["personality_traits"].example
    doc, stats = _doc_with({"add": [{"section": "personality_traits", "text": example, "date": "2025-03-05"}]})
    assert pdoc.section_items(doc, "personality_traits") == []
    assert stats.dropped_example_copies == 1


def test_section_caps_drop_the_least_recently_confirmed_items():
    cap = pdoc.SECTION_BY_KEY["location"].cap
    towns = ["Reno", "Boise", "Fresno", "Spokane", "Eugene", "Tacoma", "Ogden"]
    adds = [
        {"section": "location", "text": f"lived in {towns[i]}", "date": f"2025-03-{i + 1:02d}"} for i in range(cap + 2)
    ]
    doc, stats = _doc_with({"add": adds})
    kept = [it["date"] for it in pdoc.section_items(doc, "location")]
    assert len(kept) == cap
    assert "2025-03-01" not in kept and "2025-03-02" not in kept
    assert stats.dropped_over_cap == 2


def test_stale_lately_items_move_to_life_events_relative_to_the_batch():
    doc, _ = _doc_with(
        {"add": [{"section": "current_situation", "text": "training for the Portland marathon", "date": "2025-03-05"}]}
    )
    # Six months of history later (by message date, not wall clock) it ages out.
    doc, stats = pdoc.apply_edits(doc, {}, window=(date(2025, 9, 1), date(2025, 9, 10)), source_text="")
    assert pdoc.section_items(doc, "current_situation") == []
    moved = pdoc.section_items(doc, "life_events")
    assert moved[0]["text"] == "training for the Portland marathon" and moved[0]["date"] == "2025-03-05"
    assert stats.moved_lately == 1


def test_apply_edits_does_not_mutate_its_input():
    doc, _ = _doc_with({"add": [{"section": "hobbies", "text": "birdwatching", "date": "2025-03-05"}]})
    before = json.dumps(doc, sort_keys=True)
    pdoc.apply_edits(doc, {"remove": [pdoc.section_items(doc, "hobbies")[0]["id"]]}, window=WINDOW, source_text="")
    assert json.dumps(doc, sort_keys=True) == before


def test_section_texts_reads_v1_and_v2():
    v1 = {"hobbies": ["crochet", "  ", "hiking"]}
    assert pdoc.section_texts(v1, "hobbies") == ["crochet", "hiking"]
    doc, _ = _doc_with({"add": [{"section": "hobbies", "text": "crochet", "date": "2025-03-05"}]})
    assert pdoc.section_texts(doc, "hobbies") == ["crochet"]


def test_normalize_repairs_ids_and_drops_junk():
    raw = {
        "version": 2,
        "sections": {"hobbies": [{"id": "hob9", "text": "kayaking", "date": "2025-01-01"}, "junk", {"text": "no id"}]},
        "next_id": 2,
    }
    doc = pdoc.normalize_document(raw)
    assert [it["id"] for it in pdoc.section_items(doc, "hobbies")] == ["hob9"]
    assert doc["next_id"] == 10, "next_id never reuses an existing id"
    assert set(doc["sections"]) == set(pdoc.SECTION_KEYS)


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------


def _rich_doc():
    adds = [
        {"section": "family_and_household", "text": "has a corgi named Waffles", "date": "2025-03-02"},
        {"section": "politics", "text": "wants ranked-choice voting in state elections", "date": "2025-03-03"},
        {"section": "hobbies", "text": "restores vintage road bikes in the garage", "date": "2025-03-04"},
        {"section": "media_entertainment", "text": "rewatches The Expanse every winter", "date": "2025-03-05"},
        {"section": "uncertain", "text": "might be a teacher", "date": "2025-03-06"},
        {"section": "current_situation", "text": "shopping for a used camper van", "date": "2025-03-07"},
    ]
    doc, _ = _doc_with(
        {
            "add": adds,
            "overview": "Friendly regular who talks bikes and politics.",
            "communication_style": "Short, dry replies.",
        }
    )
    return doc


def test_chat_render_prefers_items_matching_the_question_under_a_tight_budget():
    doc = _rich_doc()
    # Room for the overview plus exactly one of the two items.
    budget = len("Overview: Friendly regular who talks bikes and politics.") + 75
    corgi = pdoc.render_for_chat(doc, ["corgi"], budget, today=date(2025, 3, 20))
    vote = pdoc.render_for_chat(doc, ["voting"], budget, today=date(2025, 3, 20))
    assert "Waffles" in corgi and "ranked-choice" not in corgi
    assert "ranked-choice" in vote and "Waffles" not in vote


def test_chat_render_fits_the_budget_with_whole_items_and_month_dates():
    doc = _rich_doc()
    for budget in (120, 200, 350, 5000):
        text = pdoc.render_for_chat(doc, ["tell", "about"], budget, today=date(2025, 3, 20))
        assert len(text) <= budget
        for line in text.splitlines()[1:]:
            if line.startswith("Style:"):
                continue
            assert line.endswith("(Mar 2025)"), f"item cut mid-way or missing its month: {line!r}"


def test_chat_render_hides_guesses_and_stale_lately_items():
    doc = _rich_doc()
    text = pdoc.render_for_chat(doc, ["teacher", "camper"], 5000, today=date(2026, 9, 14))
    assert "might be a teacher" not in text
    assert "camper van" not in text, "a lately item unconfirmed for 90+ days is no longer 'lately'"


def test_chat_render_ignores_pre_v2_documents():
    assert pdoc.render_for_chat({"overview": "old"}, ["x"], 1000, today=date(2025, 1, 1)) == ""


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------


@pytest.fixture
def budget_env(monkeypatch):
    for name in (
        "USER_PROFILE_LLM_N_CTX",
        "USER_PROFILE_CONTEXT_SAFETY",
        "USER_PROFILE_MAX_TOKENS",
        "USER_PROFILE_CHARS_PER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_CHAT", "test-model")
    profile_llm._ratio_samples.clear()
    yield monkeypatch
    profile_llm._ratio_samples.clear()


def _fake_probe(monkeypatch, value):
    async def probe(model):
        return value

    monkeypatch.setattr(profile_llm, "loaded_context_tokens", probe)


def test_budget_stays_under_the_loaded_window_with_margin(budget_env):
    _fake_probe(budget_env, 28000)
    b = asyncio.run(profile_llm.compute_budget())
    assert b.window_tokens == 28000
    assert b.prompt_tokens + b.max_output_tokens <= int(28000 * 0.85)


def test_budget_override_can_only_shrink_the_window(budget_env):
    _fake_probe(budget_env, 28000)
    budget_env.setenv("USER_PROFILE_LLM_N_CTX", "40000")
    assert asyncio.run(profile_llm.compute_budget()).window_tokens == 28000
    budget_env.setenv("USER_PROFILE_LLM_N_CTX", "20000")
    assert asyncio.run(profile_llm.compute_budget()).window_tokens == 20000


def test_budget_falls_back_to_chat_window_when_lm_studio_cannot_be_read(budget_env):
    _fake_probe(budget_env, None)
    budget_env.setenv("CONTEXT_WINDOW_TOKENS", "16000")
    b = asyncio.run(profile_llm.compute_budget())
    assert b.window_tokens == 16000 and "fallback" in b.window_source


def test_chars_per_token_uses_the_densest_observation(budget_env):
    assert profile_llm.chars_per_token() == 2.5
    profile_llm.observe_prompt_tokens(40000, 10000)  # 4.0
    profile_llm.observe_prompt_tokens(30000, 10000)  # 3.0
    assert profile_llm.chars_per_token() == pytest.approx(3.0 * 0.95)


def test_edit_schema_constrains_sections_to_known_keys():
    schema = profile_llm.edits_json_schema()
    enum = schema["properties"]["add"]["items"]["properties"]["section"]["enum"]
    assert enum == list(pdoc.SECTION_KEYS)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_llm_turn_is_reentrant_within_a_task():
    async def scenario():
        async with llm_gate.llm_turn():
            async with llm_gate.llm_turn():
                return llm_gate.gate_busy()

    assert asyncio.run(asyncio.wait_for(scenario(), 1)) is True


def test_llm_turn_serializes_tasks():
    active = 0
    peak = 0

    async def worker():
        nonlocal active, peak
        async with llm_gate.llm_turn():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    async def scenario():
        await asyncio.gather(*(worker() for _ in range(4)))

    asyncio.run(scenario())
    assert peak == 1


def test_wait_for_chat_returns_at_once_when_nothing_is_pending():
    waited = asyncio.run(llm_gate.wait_for_chat_to_clear(grace_seconds=60, max_wait_seconds=5, poll_seconds=0.01))
    assert waited < 0.5


def test_wait_for_chat_holds_until_the_reply_is_done_plus_grace():
    marks = {}

    async def scenario():
        llm_gate.note_chat_queued()
        waiter = asyncio.create_task(
            llm_gate.wait_for_chat_to_clear(grace_seconds=0.1, max_wait_seconds=5, poll_seconds=0.01)
        )
        await asyncio.sleep(0.15)
        marks["still_waiting"] = not waiter.done()
        llm_gate.note_chat_done()
        await asyncio.sleep(0.05)
        marks["waiting_in_grace"] = not waiter.done()
        return await waiter

    waited = asyncio.run(scenario())
    assert marks == {"still_waiting": True, "waiting_in_grace": True}
    assert 0.25 <= waited < 1.5
    assert llm_gate.chat_pending() == 0


def test_wait_for_chat_gives_up_on_a_stuck_counter_and_keeps_heartbeating():
    ticks = []
    llm_gate.note_chat_queued()
    waited = asyncio.run(
        llm_gate.wait_for_chat_to_clear(
            grace_seconds=0, max_wait_seconds=0.2, poll_seconds=0.01, tick=lambda: ticks.append(1), tick_seconds=0.05
        )
    )
    assert 0.2 <= waited < 1.0
    assert ticks, "a long wait must keep the job heartbeat fresh"


def test_chat_done_never_goes_negative():
    llm_gate.note_chat_done()
    assert llm_gate.chat_pending() == 0


# ---------------------------------------------------------------------------
# The pass loop, against a throwaway archive
# ---------------------------------------------------------------------------

_MESSAGES_SCHEMA = """
CREATE TABLE messages (
    message_id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    username TEXT NOT NULL,
    nickname TEXT,
    user_id INTEGER NOT NULL,
    message_content TEXT,
    channel_id INTEGER NOT NULL,
    channel_name TEXT NOT NULL,
    image_description TEXT,
    url_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


@pytest.fixture
def archive(tmp_path, monkeypatch):
    path = str(tmp_path / "guild.db")
    conn = sqlite3.connect(path)
    conn.execute(_MESSAGES_SCHEMA)
    rows = []
    mid = 1000
    for i in range(30):
        mid += 1
        rows.append((mid, f"2025-03-{i + 1:02d}", "12:00:00", "ann", "Ann", 1, f"ann message {i}", 5, "general"))
        mid += 1
        rows.append((mid, f"2025-03-{i + 1:02d}", "12:01:00", "bo", "Bo", 2, f"bo reply {i}", 5, "general"))
    conn.executemany(
        "INSERT INTO messages (message_id, date, time, username, nickname, user_id, message_content, channel_id, "
        "channel_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(user_profiles, "get_db_path", lambda gid: path)
    monkeypatch.setattr("soupy_database.profile_batch.get_db_path", lambda gid: path)
    monkeypatch.setenv("USER_PROFILE_PASS_MAX_MESSAGES", "10")
    monkeypatch.setenv("USER_PROFILE_MIN_MESSAGES", "8")

    async def budget():
        return profile_llm.Budget(28000, "test", 1000, 200000, 3.0)

    monkeypatch.setattr(user_profiles, "compute_budget", budget)
    return path


def _message_count(prompt: str) -> int:
    return int(re.search(r"NEW MESSAGES \((\d+),", prompt).group(1))


def _fake_llm(monkeypatch, behaviour):
    calls = []

    async def fake(system_prompt, user_prompt, *, max_tokens, progress=None, kind=None):
        n = _message_count(user_prompt)
        first_date = re.search(r"\n\[(\d{4}-\d{2}-\d{2}) #", user_prompt).group(1)
        calls.append(n)
        edits = behaviour(n, first_date, len(calls))
        if isinstance(edits, Exception):
            raise edits
        return profile_llm.LlmReply(edits, json.dumps(edits), "stop", 1234, 56, 0.1)

    monkeypatch.setattr(user_profiles, "request_edits", fake)
    return calls


_HOBBIES = ["archery", "beekeeping", "calligraphy", "dinghy sailing", "embroidery", "fencing"]


def _one_add(n, first_date, call_no):
    return {
        "add": [{"section": "hobbies", "text": _HOBBIES[(call_no - 1) % len(_HOBBIES)], "date": first_date}],
        "update": [],
        "remove": [],
        "overview": f"after pass {call_no}",
        "communication_style": "",
    }


def _stored(path, user_id):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT structured_json, source_max_message_id, source_message_count, summary "
            "FROM user_profile_summaries WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()


def test_full_build_saves_every_pass_with_dated_items(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, _one_add)
    result = asyncio.run(user_profiles.refresh_user_profile(1, 1))

    assert result["ok"] and result["mode"] == "rebuild" and result["passes"] == 3
    assert calls == [10, 10, 10]
    sj, cursor, count, summary = _stored(archive, 1)
    doc = json.loads(sj)
    assert pdoc.is_v2(doc)
    assert doc["coverage"]["messages"] == 30 and doc["coverage"]["passes"] == 3
    assert count == 30
    assert cursor == 1059, "cursor is the last message folded in"
    assert [it["date"] for it in pdoc.section_items(doc, "hobbies")] == ["2025-03-01", "2025-03-11", "2025-03-21"]
    assert "(2025-03-11)" in summary


def test_recent_chat_shrinks_passes(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, _one_add)
    monkeypatch.setenv("USER_PROFILE_BUSY_PASS_MESSAGES", "6")
    monkeypatch.setattr(user_profiles, "seconds_since_chat_activity", lambda: 30.0)
    asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert calls == [6, 6, 6, 6, 6]


def test_passes_wait_while_a_chat_reply_is_pending(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, _one_add)
    monkeypatch.setattr(user_profiles, "_CHAT_GRACE_SEC", 0.05)
    monkeypatch.setattr(
        user_profiles,
        "wait_for_chat_to_clear",
        functools.partial(llm_gate.wait_for_chat_to_clear, poll_seconds=0.01),
    )
    seen = {}

    async def scenario():
        llm_gate.note_chat_queued()
        build = asyncio.create_task(user_profiles.refresh_user_profile(1, 1))
        await asyncio.sleep(0.2)
        seen["calls_while_chat_pending"] = len(calls)
        llm_gate.note_chat_done()
        return await asyncio.wait_for(build, 5)

    result = asyncio.run(scenario())
    assert seen["calls_while_chat_pending"] == 0
    assert result["passes"] == 3 and calls == [10, 10, 10]


def test_stop_and_resume_continue_from_the_saved_pass(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, _one_add)
    checks = {"n": 0}

    def stop_after_first_pass():
        checks["n"] += 1
        return checks["n"] > 1

    first = asyncio.run(user_profiles.refresh_user_profile(1, 1, should_stop=stop_after_first_pass))
    assert first["stopped"] and first["messages_processed"] == 10 and first["messages_remaining"] == 20

    second = asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert second["mode"] == "update" and second["messages_processed"] == 20
    assert calls == [10, 10, 10]
    doc = json.loads(_stored(archive, 1)[0])
    assert doc["coverage"]["messages"] == 30

    third = asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert third["skipped"] and third["reason"] == "up_to_date"


def test_context_overflow_splits_the_pass_instead_of_growing_it(archive, monkeypatch):
    def behaviour(n, first_date, call_no):
        if n > 5:
            return profile_llm.ContextOverflowError("n_keep > n_ctx")
        return _one_add(n, first_date, call_no)

    calls = _fake_llm(monkeypatch, behaviour)
    result = asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert result["messages_processed"] == 30 and result["messages_skipped"] == 0
    assert max(c for c in calls if c <= 5) == 5
    assert all(c in (10, 5) for c in calls)


def test_repetitive_output_is_discarded_and_the_pass_split(archive, monkeypatch):
    def behaviour(n, first_date, call_no):
        if n > 5:
            looping = [{"section": "hobbies", "text": "archery at the range", "date": first_date}] * 20
            return {"add": looping, "update": [], "remove": [], "overview": "", "communication_style": ""}
        return _one_add(n, first_date, call_no)

    calls = _fake_llm(monkeypatch, behaviour)
    result = asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert result["messages_processed"] == 30 and result["messages_skipped"] == 0
    assert calls[:2] == [10, 5]
    doc = json.loads(_stored(archive, 1)[0])
    assert "archery at the range" not in json.dumps(doc), "the looping pass's edits were never applied"


def test_unparseable_output_is_skipped_after_splitting_down(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, lambda n, d, c: None)
    result = asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert result["messages_skipped"] == 30
    assert calls[:3] == [10, 5, 2]
    assert _stored(archive, 1)[1] == 1059, "skipped messages still advance the cursor"


def test_rebuilding_a_v1_profile_backs_it_up_first(archive, monkeypatch):
    conn = sqlite3.connect(archive)
    user_profiles.ensure_user_profile_schema(conn)
    conn.execute(
        "INSERT INTO user_profile_summaries (user_id, nickname_hint, summary, structured_json, source_message_count) "
        "VALUES (1, 'Ann', 'old summary', ?, 30)",
        (json.dumps({"overview": "old", "hobbies": ["knitting"]}),),
    )
    conn.commit()
    conn.close()

    _fake_llm(monkeypatch, _one_add)
    result = asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert result["mode"] == "rebuild"

    conn = sqlite3.connect(archive)
    backup = conn.execute(f"SELECT summary FROM {user_profiles.V1_BACKUP_TABLE} WHERE user_id = 1").fetchone()
    conn.close()
    assert backup == ("old summary",)


def test_too_few_messages_is_skipped_without_calling_the_llm(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, _one_add)
    monkeypatch.setenv("USER_PROFILE_MIN_MESSAGES", "100")
    result = asyncio.run(user_profiles.refresh_user_profile(1, 1))
    assert result["skipped"] and result["reason"] == "too_few_messages"
    assert calls == []


def test_nightly_candidates_put_updates_first_and_skip_quiet_members(archive, monkeypatch):
    _fake_llm(monkeypatch, _one_add)
    stop = {"n": 0}

    def stop_after_one():
        stop["n"] += 1
        return stop["n"] > 1

    # Ann has a v2 profile with 20 unread messages; Bo has no profile yet.
    asyncio.run(user_profiles.refresh_user_profile(1, 1, should_stop=stop_after_one))
    assert user_profiles.build_nightly_candidate_user_ids(1, min_new_messages=10) == [1, 2]
    assert user_profiles.build_nightly_candidate_user_ids(1, min_new_messages=25) == [2]


@pytest.mark.parametrize(
    "local, expected",
    [
        (datetime(2026, 9, 14, 23, 30), date(2026, 9, 14)),
        (datetime(2026, 9, 15, 1, 15), date(2026, 9, 14)),
        (datetime(2026, 9, 15, 2, 30), None),
        (datetime(2026, 9, 14, 22, 59), None),
    ],
)
def test_nightly_start_window_crosses_midnight(local, expected):
    tz = pytz.timezone("America/Los_Angeles")
    assert user_profiles._nightly_run_date(tz.localize(local), tz, 23) == expected


def test_manual_batch_only_queues_a_job(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, _one_add)
    result = user_profiles.start_profile_batch_job(1)
    assert result["ok"] and sorted(result["user_ids"]) == [1, 2]
    status = user_profiles.get_profile_batch_status(1)
    assert status["status"] == "running" and status["kind"] == "manual"
    assert status["waiting_for_bot"] is True, "no heartbeat until the bot's worker picks it up"
    assert calls == [], "the web process must never call the LLM"


def test_bot_worker_runs_a_queued_job_to_completion(archive, monkeypatch):
    _fake_llm(monkeypatch, _one_add)
    user_profiles.start_profile_batch_job(1)
    asyncio.run(user_profiles._run_profile_job(1))
    status = user_profiles.get_profile_batch_status(1)
    assert status["status"] == "completed"
    assert status["batch_stats"] == {"saved": 2, "skipped": 0, "failed": 0}
    assert status["heartbeat_at"]


def test_worker_never_profiles_soupy_itself(archive, monkeypatch):
    calls = _fake_llm(monkeypatch, _one_add)
    user_profiles.start_profile_batch_job(1)
    asyncio.run(user_profiles._run_profile_job(1, exclude_user_ids=[2]))
    status = user_profiles.get_profile_batch_status(1)
    assert status["batch_stats"] == {"saved": 1, "skipped": 1, "failed": 0}
    assert _stored(archive, 2) is None
    assert calls == [10, 10, 10]
    assert user_profiles.build_nightly_candidate_user_ids(1, 10, exclude_user_ids=[2]) == []


def test_paused_job_stops_the_worker_mid_member(archive, monkeypatch):
    def pause_after_first(n, first_date, call_no):
        if call_no == 1:
            user_profiles.pause_profile_batch_job(1)
        return _one_add(n, first_date, call_no)

    calls = _fake_llm(monkeypatch, pause_after_first)
    user_profiles.start_profile_batch_job(1)
    asyncio.run(user_profiles._run_profile_job(1))
    status = user_profiles.get_profile_batch_status(1)
    assert status["status"] == "paused" and status["next_index"] == 0
    assert calls == [10]

    user_profiles.resume_profile_batch_job(1)
    asyncio.run(user_profiles._run_profile_job(1))
    assert user_profiles.get_profile_batch_status(1)["status"] == "completed"
    assert json.loads(_stored(archive, 1)[0])["coverage"]["messages"] == 30
