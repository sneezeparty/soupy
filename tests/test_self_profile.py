"""Tests for Soupy's memory of itself (``soupy_database.self_profile``).

The memory reuses the member-profile machinery, so these cover what differs:

* the ``SELF`` document rules — relationship items need the person in the
  batch, a per-person cap, the "lately" age-out into moments;
* what chat and the files see — the people in the conversation first,
  nothing unrelated, habits and guesses never shown;
* reading exchanges out of the archive — member context, split replies,
  duplicates, slash-command output;
* the build against a throwaway archive with the LLM faked — per-pass saves
  and views, the one-time SELF.MD backup, updates, previews;
* the profile worker running it after a job, and the nightly refresh queueing
  a job just for it.

pytest-asyncio isn't a dependency, so coroutines run under ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import date

import pytest

from soupy import llm_gate
from soupy_database import profile_document as pdoc
from soupy_database import profile_llm, self_context, self_profile, user_profiles

# Realistic snowflake ids: mention parsing only matches 15-20 digits.
BOT = 900000000000000099
ANN = 100000000000000001
BO = 100000000000000002
WINDOW = (date(2025, 3, 1), date(2025, 3, 20))


@pytest.fixture(autouse=True)
def quiet_chat(monkeypatch):
    monkeypatch.setattr(llm_gate, "_chat_pending", 0)
    monkeypatch.setattr(llm_gate, "_last_chat_activity", 0.0)


def _edits(add=(), update=(), remove=(), overview=""):
    return {"add": list(add), "update": list(update), "remove": list(remove), "overview": overview}


def _self_doc(edits, source, window=WINDOW, doc=None, directory=None):
    return pdoc.apply_edits(
        doc or pdoc.new_document(pdoc.SELF),
        edits,
        window=window,
        source_text=source,
        directory=directory if directory is not None else {ANN: "ann", BO: "bo"},
        kind=pdoc.SELF,
    )


# ---------------------------------------------------------------------------
# Document rules
# ---------------------------------------------------------------------------


def test_relationship_items_need_the_person_in_the_batch():
    source = (
        "[2025-03-02 #general] ann: soupy you are the best at both\n[2025-03-02 #general] SOUPY: i know all about it"
    )
    doc, stats = _self_doc(
        _edits(
            add=[
                {
                    "section": "relationships",
                    "text": "i like how ann hypes me up",
                    "date": "2025-03-02",
                    "user_id": ANN,
                },
                # "bo" appears inside "about" and "both", never as a name: not grounded.
                {"section": "relationships", "text": "i think bo is a menace", "date": "2025-03-02", "user_id": BO},
                {"section": "relationships", "text": "i barely know this person", "date": "2025-03-02", "user_id": 0},
            ]
        ),
        source,
    )
    items = pdoc.section_items(doc, "relationships")
    assert [(it["user_id"], it["name"]) for it in items] == [(ANN, "ann")]
    assert stats.dropped_ungrounded == 2


def test_per_person_cap_keeps_each_persons_newest_items():
    source = "ann: hi\nbo: hi"
    adds = [
        {"section": "relationships", "text": f"ann and i {topic}", "date": f"2025-03-0{i + 1}", "user_id": ANN}
        for i, topic in enumerate(["argue about pizza", "trade movie picks", "share memes", "debate vim", "roast bo"])
    ]
    adds.append({"section": "relationships", "text": "bo keeps testing me", "date": "2025-03-01", "user_id": BO})
    doc, stats = _self_doc(_edits(add=adds), source)
    by_person = {}
    for it in pdoc.section_items(doc, "relationships"):
        by_person.setdefault(it["user_id"], []).append(it["text"])
    assert len(by_person[ANN]) == pdoc.SELF.per_person_cap
    assert "ann and i argue about pizza" not in by_person[ANN], "the least recent item goes first"
    assert by_person[BO] == ["bo keeps testing me"], "another person's items are untouched"
    assert stats.dropped_over_cap == 1


def test_same_words_about_different_people_are_not_duplicates():
    source = "ann: hey\nbo: hey"
    doc, stats = _self_doc(
        _edits(
            add=[
                {
                    "section": "relationships",
                    "text": "i trust them with code reviews",
                    "date": "2025-03-02",
                    "user_id": ANN,
                },
                {
                    "section": "relationships",
                    "text": "i trust them with code reviews",
                    "date": "2025-03-02",
                    "user_id": BO,
                },
            ]
        ),
        source,
    )
    assert len(pdoc.section_items(doc, "relationships")) == 2
    assert stats.merged_duplicates == 0


def test_stale_lately_items_become_moments():
    doc, _ = _self_doc(
        _edits(
            add=[
                {"section": "current_situation", "text": "people keep asking me about the outage", "date": "2025-01-02"}
            ]
        ),
        "SOUPY: ugh the outage",
        window=(date(2025, 1, 1), date(2025, 1, 3)),
    )
    doc, stats = _self_doc(_edits(), "SOUPY: hi", window=(date(2025, 6, 1), date(2025, 6, 2)), doc=doc)
    assert stats.moved_lately == 1
    assert [it["text"] for it in pdoc.section_items(doc, "memorable_moments")] == [
        "people keep asking me about the outage"
    ]


def test_self_example_copies_are_dropped():
    doc, stats = _self_doc(
        _edits(
            add=[
                {
                    "section": "relationships",
                    "text": "i trust Quillon's movie picks",
                    "date": "2025-03-02",
                    "user_id": ANN,
                }
            ]
        ),
        "ann: what should i watch",
    )
    assert pdoc.item_count(doc, pdoc.SELF) == 0
    assert stats.dropped_example_copies == 1


def test_self_schema_and_prompt_use_the_self_sections():
    enum = profile_llm.edits_json_schema(pdoc.SELF)["properties"]["add"]["items"]["properties"]["section"]["enum"]
    assert "relationships" in enum and "with_soupy" not in enum
    prompt = profile_llm.build_self_system_prompt()
    assert "- memorable_moments (max 16)" in prompt and "first person" in prompt


# ---------------------------------------------------------------------------
# What chat and the files see
# ---------------------------------------------------------------------------


def _rich_self_doc():
    doc = pdoc.new_document(pdoc.SELF)
    s = doc["sections"]
    s["relationships"] = [
        {
            "id": "rel1",
            "text": "i love arguing about pizza with ann",
            "date": "2025-02-01",
            "user_id": ANN,
            "name": "ann",
        },
        {"id": "rel2", "text": "bo tries to break me every week", "date": "2025-02-03", "user_id": BO, "name": "bo"},
    ]
    s["opinions_and_stances"] = [
        {"id": "op3", "text": "i think pineapple on pizza is a crime", "date": "2025-02-01"},
        {"id": "op4", "text": "i said the cloud is just someone else's computer", "date": "2025-02-05"},
    ]
    s["current_situation"] = [{"id": "now5", "text": "people keep asking me to rank sandwiches", "date": "2025-03-01"}]
    s["signature_habits"] = [{"id": "say6", "text": "honestly", "date": "2025-02-01"}]
    s["uncertain"] = [{"id": "unk7", "text": "maybe i secretly like jazz", "date": "2025-02-01"}]
    doc["overview"] = "i am soupy. i roast people i like. i have strong food opinions."
    return doc


def test_chat_memory_puts_the_people_talking_first_and_skips_unrelated_items():
    text = pdoc.render_self_for_chat(
        _rich_self_doc(),
        people={ANN: "ann"},
        query_tokens=["weather"],
        max_chars=2000,
        today=date(2025, 3, 5),
    )
    lines = text.split("\n")
    assert lines[0] == "About ann: i love arguing about pizza with ann (Feb 2025)"
    assert "bo tries" not in text, "relationships of people not in the conversation stay out"
    assert "cloud" not in text and "pineapple" not in text, "unrelated opinions don't fill the budget"
    assert "Lately: people keep asking me to rank sandwiches (Mar 2025)" in text
    assert "honestly" not in text and "jazz" not in text


def test_chat_memory_uses_ranked_and_word_matched_items_within_budget():
    doc = _rich_self_doc()
    text = pdoc.render_self_for_chat(
        doc, people={}, ranked_ids=["op4"], query_tokens=["pineapple"], max_chars=2000, today=date(2025, 3, 5)
    )
    assert "Opinions: i said the cloud is just someone else's computer (Feb 2025); i think pineapple" in text
    tight = pdoc.render_self_for_chat(
        doc, people={}, ranked_ids=["op4"], query_tokens=["pineapple"], max_chars=70, today=date(2025, 3, 5)
    )
    assert "cloud" in tight and "pineapple" not in tight and len(tight) <= 70


def test_prompt_render_only_shows_relationships_for_people_in_the_batch():
    doc = _rich_self_doc()
    text = pdoc.render_for_prompt(doc, pdoc.SELF, relationship_user_ids={BO}, relationship_max_chars=500)
    assert "bo tries to break me" in text and "arguing about pizza" not in text
    assert "[relationships] (2/60)" in text, "the count is the whole section's"
    none = pdoc.render_for_prompt(doc, pdoc.SELF, relationship_user_ids=set(), relationship_max_chars=500)
    assert "[relationships]" not in none


def test_views_markdown_core_and_anchor():
    doc = _rich_self_doc()
    md = pdoc.render_self_markdown(doc)
    assert md.startswith("## who i am\ni am soupy.")
    assert "- ann: i love arguing about pizza with ann" not in md, "names already in the text aren't repeated"
    assert "- i love arguing about pizza with ann (2025-02-01)" in md
    assert "## habits (never shown in chat)\n- honestly (2025-02-01)" in md
    assert pdoc.render_self_core(doc).startswith("i am soupy.")
    assert pdoc.self_anchor(doc, 30) == "i am soupy."
    assert pdoc.self_anchor(doc, 5) == "i am…", "no whole sentence fits, so cut"


# ---------------------------------------------------------------------------
# Reading exchanges out of the archive
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

_USERS = {ANN: ("ann", "Ann"), BO: ("bo", "Bo"), BOT: ("dafoe", None)}


def _insert(path, rows):
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO messages (message_id, date, time, username, nickname, user_id, message_content, channel_id, "
        "channel_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(mid, d, t, _USERS[u][0], _USERS[u][1], u, text, ch, f"chan{ch}") for mid, d, t, u, text, ch in rows],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    path = str(tmp_path / "guild.db")
    conn = sqlite3.connect(path)
    conn.execute(_MESSAGES_SCHEMA)
    conn.close()
    rows = []
    mid = 1000
    for i in range(30):
        mid += 1
        rows.append((mid, f"2025-04-{i + 1:02d}", "12:00:00", ANN, f"soupy what about topic{i}", 5))
        mid += 1
        rows.append((mid, f"2025-04-{i + 1:02d}", "12:00:30", BOT, f"my take on topic{i} is simple", 5))
    _insert(path, rows)

    self_dir = tmp_path / "self_md"
    monkeypatch.setattr(self_context, "SELF_MD_DIR", self_dir)
    monkeypatch.setattr(self_profile, "get_db_path", lambda gid: path)
    monkeypatch.setattr(user_profiles, "get_db_path", lambda gid: path)
    monkeypatch.setattr("soupy_database.profile_batch.get_db_path", lambda gid: path)
    monkeypatch.setenv("USER_PROFILE_PASS_MAX_MESSAGES", "10")
    monkeypatch.setenv("SELF_MD_ENABLED", "true")

    async def budget():
        return profile_llm.Budget(28000, "test", 1000, 200000, 3.0)

    async def no_embeddings(guild_id, doc, progress=None):
        return 0

    monkeypatch.setattr(user_profiles, "compute_budget", budget)
    monkeypatch.setattr(self_profile, "index_self_items", no_embeddings)
    return path


def test_exchanges_carry_member_context_and_skip_command_output(tmp_path):
    path = str(tmp_path / "g.db")
    conn = sqlite3.connect(path)
    conn.execute(_MESSAGES_SCHEMA)
    conn.close()
    _insert(
        path,
        [
            (1, "2025-03-01", "11:00:00", BO, "old news", 5),
            (2, "2025-03-01", "12:00:00", ANN, f"<@{BOT}> is pineapple ok on pizza", 5),
            (3, "2025-03-01", "12:00:30", BOT, "pineapple on pizza is a crime", 5),
            (4, "2025-03-01", "12:00:40", BOT, "and i will die on this hill", 5),
            (5, "2025-03-01", "12:00:41", BOT, "and i will die on this hill", 5),
            (6, "2025-03-01", "12:30:00", BO, f"tell <@{ANN}> she is wrong", 6),
            (7, "2025-03-01", "12:30:05", BOT, 'Question: "am i right" The 9-Ball says: "no"', 7),
            (8, "2025-03-01", "12:30:10", BOT, f"<@{ANN}> is wrong and <@{BO}> knows it", 6),
            (9, "2025-03-01", "12:31:00", BOT, "[Embed Title: 🔍 Search Results]", 6),
        ],
    )
    conn = sqlite3.connect(path)
    units, people, names = self_profile.fetch_self_units(conn, BOT, None)
    conn.close()

    assert [u[0] for u in units] == [5, 8], "the duplicate's id is the cursor, so it isn't re-read"
    assert units[0][2] == (
        "[2025-03-01 #chan5] Ann: @soupy is pineapple ok on pizza\n"
        "[2025-03-01 #chan5] SOUPY: pineapple on pizza is a crime\n"
        "[2025-03-01 #chan5] SOUPY: and i will die on this hill"
    ), "a split reply is one exchange, the duplicate is dropped, and hour-old context is left out"
    assert units[1][2] == (
        "[2025-03-01 #chan6] Bo: tell @Ann she is wrong\n[2025-03-01 #chan6] SOUPY: @Ann is wrong and @Bo knows it"
    )
    assert people == {5: frozenset({ANN}), 8: frozenset({ANN, BO})}
    assert names == {ANN: "Ann", BO: "Bo"}


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


_ANN_DYNAMICS = ["ann quizzes me about trivia nights", "ann and i trade movie picks", "ann roasts my spelling", "x"]


def _fake_self_llm(monkeypatch):
    calls = []

    async def fake(system_prompt, user_prompt, *, max_tokens, progress=None, kind=None):
        assert kind is pdoc.SELF
        n = int(re.search(r"NEW MESSAGES \((\d+) exchanges", user_prompt).group(1))
        topic = re.search(r"SOUPY: my take on (topic\d+)", user_prompt).group(1)
        day = re.search(r"\n\[(\d{4}-\d{2}-\d{2}) #", "\n" + user_prompt.split("NEW MESSAGES")[1]).group(1)
        calls.append((n, user_prompt))
        edits = _edits(
            add=[
                {"section": "opinions_and_stances", "text": f"i keep a simple take on {topic}", "date": day},
                {"section": "relationships", "text": _ANN_DYNAMICS[len(calls) - 1], "date": day, "user_id": ANN},
            ],
            overview=f"i am soupy and i have takes, most recently on {topic}.",
        )
        return profile_llm.LlmReply(edits, json.dumps(edits), "stop", 1234, 56, 0.1)

    monkeypatch.setattr(user_profiles, "request_edits", fake)
    return calls


def test_full_build_saves_every_pass_writes_views_and_backs_up_selfmd_once(archive, monkeypatch):
    calls = _fake_self_llm(monkeypatch)
    self_context.SELF_MD_DIR.mkdir(parents=True)
    self_context.self_md_path(7).write_text("## opinions and stances\nold truncated doc", encoding="utf-8")

    result = asyncio.run(self_profile.refresh_self_profile(7, BOT))

    assert result["ok"] and result["mode"] == "rebuild" and result["passes"] == 3
    assert [n for n, _ in calls] == [10, 10, 10]
    state = self_profile.load_self_state(7)
    assert state["cursor"] == 1060 and state["bot_user_id"] == BOT
    doc = state["doc"]
    assert doc["coverage"]["messages"] == 30
    assert [it["date"] for it in pdoc.section_items(doc, "opinions_and_stances")] == [
        "2025-04-01",
        "2025-04-11",
        "2025-04-21",
    ]
    rel = pdoc.section_items(doc, "relationships")
    assert {it["name"] for it in rel} == {"Ann"} and len(rel) == 3

    backup = self_profile.v1_backup_dir() / "guild_7.md"
    assert backup.read_text(encoding="utf-8").startswith("## opinions and stances\nold truncated doc")
    md = self_context.load_self_md(7)
    assert md.startswith("## who i am\ni am soupy") and "(2025-04-21)" in md
    assert self_context.load_self_anchor(7) == "i am soupy and i have takes, most recently on topic20."

    # The second pass's prompt shows ann's relationship item (she's in that batch) for updating.
    assert "ann quizzes me about trivia nights" in calls[1][1]


def test_update_reads_only_new_exchanges(archive, monkeypatch):
    calls = _fake_self_llm(monkeypatch)
    asyncio.run(self_profile.refresh_self_profile(7, BOT))

    _insert(
        archive,
        [
            (2001, "2025-05-01", "09:00:00", ANN, "soupy again", 5),
            (2002, "2025-05-01", "09:00:10", BOT, "my take on topic99 is new", 5),
        ],
    )
    calls.clear()
    assert self_profile.self_refresh_due(7, BOT, min_new_messages=1)
    result = asyncio.run(self_profile.refresh_self_profile(7, BOT))
    assert result["mode"] == "update" and [n for n, _ in calls] == [1]
    assert self_profile.load_self_state(7)["cursor"] == 2002
    assert not self_profile.self_refresh_due(7, BOT, min_new_messages=1)
    assert not self_profile.v1_backup_dir().exists(), "only a first build backs up SELF.MD"


def test_preview_writes_a_sample_and_leaves_the_live_memory_alone(archive, monkeypatch):
    calls = _fake_self_llm(monkeypatch)
    result = asyncio.run(self_profile.refresh_self_profile(7, BOT, sample_passes=2))
    assert result["passes"] == 2 and len(calls) == 2 and not result["stopped"]
    assert self_profile.load_self_state(7) is None
    assert self_profile.load_self_state(7, sample=True)["doc"]["coverage"]["messages"] == 20
    assert not self_context.self_md_path(7).exists()


def test_chat_block_is_none_until_built_then_renders_memory(archive, monkeypatch):
    conn = sqlite3.connect(archive)
    kwargs = dict(people={ANN: "Ann"}, query_vec=None, query_tokens=["topic29"], max_chars=1500, min_sim=0.3)
    assert self_profile.self_block_for_chat(conn, 7, **kwargs) is None
    _fake_self_llm(monkeypatch)
    asyncio.run(self_profile.refresh_self_profile(7, BOT))
    block = self_profile.self_block_for_chat(conn, 7, **kwargs)
    conn.close()
    assert block.startswith("(Mon YYYY) is when it came up.")
    assert "\nAbout Ann: ann roasts my spelling (Apr 2025); ann and i trade movie picks (Apr 2025)" in block


def test_refresh_requests_round_trip(archive):
    self_profile.request_self_refresh(7)
    self_profile.request_self_refresh(8, sample_passes=3)
    popped = self_profile.pop_self_refresh_requests()
    assert {gid: req["sample_passes"] for gid, req in popped.items()} == {7: 0, 8: 3}
    assert self_profile.pop_self_refresh_requests() == {}


# ---------------------------------------------------------------------------
# The profile worker
# ---------------------------------------------------------------------------


def _job_row(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return dict(conn.execute("SELECT * FROM profile_batch_jobs").fetchone())
    finally:
        conn.close()


def test_worker_builds_soupys_memory_after_the_members(archive, monkeypatch):
    member_calls = []
    self_calls = _fake_self_llm(monkeypatch)

    async def fake_member(guild_id, user_id, **kw):
        member_calls.append(user_id)
        assert not self_calls, "members come first"
        return {"ok": True, "mode": "rebuild", "passes": 1, "messages_processed": 30, "items": 1}

    monkeypatch.setattr(user_profiles, "refresh_user_profile", fake_member)
    user_profiles.start_profile_batch_job(7)
    asyncio.run(user_profiles._run_profile_job(7, exclude_user_ids=[BOT]))

    assert member_calls == [ANN]
    assert len(self_calls) == 3 and self_profile.load_self_state(7) is not None
    assert _job_row(archive)["status"] == "completed"


def test_worker_skips_the_memory_when_self_md_is_disabled(archive, monkeypatch):
    monkeypatch.setenv("SELF_MD_ENABLED", "false")
    self_calls = _fake_self_llm(monkeypatch)

    async def fake_member(guild_id, user_id, **kw):
        return {"ok": True, "mode": "update", "passes": 1, "messages_processed": 1, "items": 1}

    monkeypatch.setattr(user_profiles, "refresh_user_profile", fake_member)
    user_profiles.start_profile_batch_job(7)
    asyncio.run(user_profiles._run_profile_job(7, exclude_user_ids=[BOT]))
    assert self_calls == [] and _job_row(archive)["status"] == "completed"


def test_a_failing_memory_build_still_completes_the_job(archive, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("LM Studio is down")

    monkeypatch.setattr(self_profile, "refresh_self_profile", boom)

    async def fake_member(guild_id, user_id, **kw):
        return {"ok": True, "mode": "update", "passes": 1, "messages_processed": 1, "items": 1}

    monkeypatch.setattr(user_profiles, "refresh_user_profile", fake_member)
    user_profiles.start_profile_batch_job(7)
    asyncio.run(user_profiles._run_profile_job(7, exclude_user_ids=[BOT]))
    assert _job_row(archive)["status"] == "completed"


def test_nightly_queues_a_job_when_only_soupys_memory_is_due(archive, monkeypatch):
    # Ann's member profile is already current, so the only work is Soupy's memory.
    monkeypatch.setattr(user_profiles, "build_nightly_candidate_user_ids", lambda *a, **kw: [])
    monkeypatch.setattr(user_profiles, "_nightly_run_date", lambda *a, **kw: date(2025, 5, 1))
    monkeypatch.setattr(user_profiles, "NIGHTLY_STATE_PATH", str(self_context.SELF_MD_DIR / "nightly.json"))
    asyncio.run(user_profiles._maybe_schedule_nightly([7], exclude_user_ids=[BOT]))
    row = _job_row(archive)
    assert row["status"] == "running" and row["kind"] == "nightly" and json.loads(row["user_ids_json"]) == []

    self_calls = _fake_self_llm(monkeypatch)
    asyncio.run(user_profiles._run_profile_job(7, exclude_user_ids=[BOT]))
    assert len(self_calls) == 3 and _job_row(archive)["status"] == "completed"


def test_requested_preview_runs_between_members(archive, monkeypatch):
    self_calls = _fake_self_llm(monkeypatch)
    self_profile.request_self_refresh(7, sample_passes=1)
    asyncio.run(user_profiles._run_self_requests([7], [BOT]))
    assert len(self_calls) == 1
    assert self_profile.load_self_state(7, sample=True) is not None and self_profile.load_self_state(7) is None
    assert self_profile.pop_self_refresh_requests() == {}


# ---------------------------------------------------------------------------
# Members who go by several names
# ---------------------------------------------------------------------------


def test_people_are_matched_by_older_names_and_usernames():
    # The exchange shows sMURF0r's current nickname; Soupy called him by his username.
    source = "[2025-01-24 #714] SOUPY: wow, mr.keith7444 really knows how to make friends."
    edits = _edits(
        add=[
            {
                "section": "relationships",
                "text": "i roast him about his monetization ideas",
                "date": "2025-01-24",
                "name": "mr.keith7444",
                "user_id": 0,
            }
        ]
    )
    without, stats = _self_doc(edits, source, directory={BO: "sMURF0r"})
    assert pdoc.item_count(without, pdoc.SELF) == 0 and stats.dropped_ungrounded == 1
    with_aliases, _ = pdoc.apply_edits(
        pdoc.new_document(pdoc.SELF),
        edits,
        window=WINDOW,
        source_text=source,
        directory={BO: "sMURF0r"},
        aliases={BO: ["mr.keith7444"]},
        kind=pdoc.SELF,
    )
    [item] = pdoc.section_items(with_aliases, "relationships")
    assert item["user_id"] == BO


def test_rephrasings_of_the_same_dynamic_merge_within_one_person():
    source = "oɹƃɐ: are you ai"
    directory = {ANN: "oɹƃɐ"}
    doc, _ = _self_doc(
        _edits(
            add=[
                {
                    "section": "relationships",
                    "text": "i'm pretty dry with him; i told him he was just a guy with a camera when he asked if i was ai",
                    "date": "2025-01-07",
                    "user_id": ANN,
                }
            ]
        ),
        source,
        window=(date(2025, 1, 1), date(2025, 1, 8)),
        directory=directory,
    )
    doc, stats = _self_doc(
        _edits(
            add=[
                {
                    "section": "relationships",
                    "text": "i'm pretty dry with him; i tell him he's just a guy with a camera and mock his bravery",
                    "date": "2025-01-19",
                    "user_id": ANN,
                }
            ]
        ),
        source,
        window=(date(2025, 1, 10), date(2025, 1, 20)),
        doc=doc,
        directory=directory,
    )
    assert stats.merged_duplicates == 1 and len(pdoc.section_items(doc, "relationships")) == 1
    assert pdoc.section_items(doc, "relationships")[0]["last_seen"] == "2025-01-19"


def test_member_aliases_lists_other_names_newest_first(tmp_path):
    path = str(tmp_path / "g.db")
    conn = sqlite3.connect(path)
    conn.execute(_MESSAGES_SCHEMA)
    conn.executemany(
        "INSERT INTO messages (message_id, date, time, username, nickname, user_id, message_content, channel_id, "
        "channel_name) VALUES (?, ?, ?, ?, ?, ?, ?, 5, 'chan5')",
        [
            (1, "2025-01-01", "10:00:00", "diego4445", "aggressive squirrel baffle", ANN, "a"),
            (2, "2025-06-01", "10:00:00", "diego4445", "valued opinion haver", ANN, "b"),
            (3, "2026-09-01", "10:00:00", "diego4445", "i was fallout", ANN, "c"),
        ],
    )
    aliases = self_profile.member_aliases(conn, {ANN: "i was fallout"})
    conn.close()
    assert aliases == {ANN: ["diego4445", "valued opinion haver", "aggressive squirrel baffle"]}
