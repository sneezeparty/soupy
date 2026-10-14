"""Tests for ``web.services.browse``, the queries behind the dashboard's Database and Media tabs.

They run against a scratch archive built with the bot's own schema, and
against scratch ``index.jsonl`` / ``messages.jsonl`` files. ``web.app`` is never
imported: it starts logging and mounts ``media/`` at import.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from soupy_database import database
from soupy_database import profile_document as pdoc
from web.services import browse

# Real-sized snowflakes. Neither survives a trip through a JavaScript number.
RANC1D = 123456789012345678
GLENN = 234567890123456789
GENERAL = 345678901234567890
MEMES = 456789012345678901


def _add_message(conn, mid, user_id, text, *, channel=GENERAL, day="2026-10-13", time="12:00:00", **extra):
    conn.execute(
        """
        INSERT INTO messages (message_id, date, time, username, nickname, user_id, message_content,
                              channel_id, channel_name, image_description, url_summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mid,
            day,
            time,
            extra.get("username", "sneezeparty" if user_id == RANC1D else "glenn"),
            extra.get("nickname", "ranc1d" if user_id == RANC1D else None),
            user_id,
            text,
            channel,
            "general" if channel == GENERAL else "memes",
            extra.get("image"),
            extra.get("url"),
            extra.get("created_at", "2026-10-13 00:00:00"),
        ),
    )


@pytest.fixture()
def archive_path(tmp_path, monkeypatch):
    """A guild database built with the bot's real schema."""
    monkeypatch.setattr(database, "DB_DIR", str(tmp_path))
    setup = database.init_database(1)
    setup.executemany(
        "INSERT INTO channels (channel_id, channel_name) VALUES (?, ?)", [(GENERAL, "general"), (MEMES, "memes")]
    )
    setup.commit()
    setup.close()
    return database.get_db_path(1)


@pytest.fixture()
def write(archive_path):
    """Run ``fn(conn)`` against the archive on an ordinary connection: the browsing one is read-only."""

    def run(fn):
        conn = sqlite3.connect(archive_path)
        fn(conn)
        conn.commit()
        conn.close()

    return run


@pytest.fixture()
def archive(archive_path):
    """The archive opened the way the panel opens one."""
    conn = browse.connect(archive_path)
    yield conn
    conn.close()


@pytest.fixture()
def seeded(archive, write):
    def fill(conn):
        # Inserted newest-first, as a backfill scan does, so created_at runs against send order.
        _add_message(conn, 300, RANC1D, "third: 100% done", created_at="2026-10-13 00:00:01")
        _add_message(conn, 200, GLENN, "second, snake_case here", day="2026-10-12", created_at="2026-10-13 00:00:02")
        _add_message(conn, 100, RANC1D, "first", day="2026-10-11", channel=MEMES, created_at="2026-10-13 00:00:03")
        _add_message(conn, 150, GLENN, "", day="2026-10-11", channel=MEMES, image="a ginger cat asleep in a sink")
        # Says "1000" and "snake-case": near misses for the wildcard test below.
        _add_message(
            conn,
            250,
            GLENN,
            "read this snake-case 1000 times",
            day="2026-10-12",
            time="23:30:00",
            url="an article about llamas",
        )

    write(fill)
    return archive


def _ids(page):
    return [row["message_id"] for row in page["rows"]]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_ids_are_strings_and_filter_by_the_exact_id(seeded):
    # The old endpoint sent IDs as JSON numbers. The browser read 123456789012345678
    # as 123456789012345680, and picking that member in the filter matched no rows.
    page = browse.query_messages(seeded, user_id=RANC1D)

    assert _ids(page) == ["300", "100"]
    row = page["rows"][0]
    assert (row["user_id"], row["channel_id"]) == (str(RANC1D), str(GENERAL))
    assert json.loads(json.dumps(row))["user_id"] == "123456789012345678"


def test_messages_come_back_in_send_order_not_scan_order(seeded):
    # created_at is when a scan inserted the row; a backfill inserts old messages last.
    assert _ids(browse.query_messages(seeded)) == ["300", "250", "200", "150", "100"]
    assert _ids(browse.query_messages(seeded, newest_first=False)) == ["100", "150", "200", "250", "300"]


def test_paging_reports_the_whole_match_count(seeded):
    page = browse.query_messages(seeded, limit=2, offset=2)

    assert (page["total"], page["limit"], page["offset"]) == (5, 2, 2)
    assert _ids(page) == ["200", "150"]
    assert browse.query_messages(seeded, limit=100000)["limit"] == 200


def test_search_reads_descriptions_and_takes_wildcards_literally(seeded):
    assert _ids(browse.query_messages(seeded, search="ginger cat")) == ["150"]
    assert _ids(browse.query_messages(seeded, search="llamas")) == ["250"]
    assert _ids(browse.query_messages(seeded, search="ranc1d")) == ["300", "100"]
    # "%" and "_" are LIKE wildcards; unescaped, each of these also matches message 250.
    assert _ids(browse.query_messages(seeded, search="100%")) == ["300"]
    assert _ids(browse.query_messages(seeded, search="snake_case")) == ["200"]


def test_filters_combine(seeded):
    assert _ids(browse.query_messages(seeded, channel_id=MEMES)) == ["150", "100"]
    assert _ids(browse.query_messages(seeded, has_images=True)) == ["150"]
    assert _ids(browse.query_messages(seeded, has_urls=True)) == ["250"]
    assert _ids(browse.query_messages(seeded, channel_id=MEMES, user_id=GLENN)) == ["150"]
    assert browse.query_messages(seeded, channel_id=MEMES, has_urls=True)["total"] == 0


def test_time_bounds_are_utc_stamps(seeded):
    # The dashboard turns a local calendar day into these, so the bounds carry a time of day.
    page = browse.query_messages(seeded, after="2026-10-12 00:00:00", before="2026-10-12 23:00:00")
    assert _ids(page) == ["200"]
    assert _ids(browse.query_messages(seeded, after="2026-10-12 23:00:00")) == ["300", "250"]


def test_context_is_the_same_channel_oldest_first(seeded):
    context = browse.message_context(seeded, 250, before=1, after=5)

    # 150 is nearer in time but in another channel.
    assert [r["message_id"] for r in context["rows"]] == ["200", "250", "300"]
    assert context["message_id"] == "250"
    assert browse.message_context(seeded, 999) is None


def test_members_are_listed_under_their_latest_name(archive, write):
    def fill(conn):
        _add_message(conn, 1, RANC1D, "a", nickname="zzz old nickname")
        _add_message(conn, 2, RANC1D, "b", nickname="ranc1d")
        _add_message(conn, 3, GLENN, "c", nickname=None, username="glenn")
        _add_message(conn, 4, GLENN, "d", nickname="  ", username="glenn")

    write(fill)

    # MAX(name) would pick "zzz old nickname": the alphabetically last, not the current one.
    assert browse.list_members(archive) == [
        {"user_id": str(RANC1D), "label": "ranc1d", "message_count": 2},
        {"user_id": str(GLENN), "label": "glenn", "message_count": 2},
    ]


def test_channels_carry_string_ids_and_counts(seeded):
    assert browse.list_channels(seeded) == [
        {"id": str(GENERAL), "name": "general", "messages": 3},
        {"id": str(MEMES), "name": "memes", "messages": 2},
    ]


def test_browsing_connection_cannot_write(seeded):
    with pytest.raises(sqlite3.OperationalError):
        seeded.execute("DELETE FROM messages")


def test_snowflakes_from_the_query_string():
    assert browse.parse_snowflake(" 123456789012345678 ") == RANC1D
    for bad in ("", None, "abc", "12a", "-5", "1.5"):
        assert browse.parse_snowflake(bad) is None


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def _store_profile(conn, user_id, doc, *, nickname="", summary="", count=10):
    conn.execute(
        """
        INSERT INTO user_profile_summaries (user_id, nickname_hint, summary, structured_json,
                                            source_message_count, source_max_message_id, model_used)
        VALUES (?, ?, ?, ?, ?, ?, 'test-model')
        """,
        (user_id, nickname, summary, json.dumps(doc) if doc is not None else None, count, 567890123456789012),
    )


def _v2_doc():
    doc = pdoc.new_document()
    doc["overview"] = "Builds bots and argues about hardware."
    doc["communication_style"] = "dry"
    doc["sections"]["hobbies"] = [
        {"id": "hob1", "text": "restores vintage road bikes", "date": "2025-03-05", "last_seen": "2026-09-01"},
        {"id": "hob2", "text": "bakes sourdough", "date": "2025-04-01", "last_seen": "2025-04-01"},
    ]
    doc["sections"]["relationships_with_others"] = [
        {"id": "rel1", "text": "trades build advice", "date": "2026-01-02", "user_id": GLENN},
        {"id": "rel2", "text": "argues about keyboards", "date": "2026-01-03", "user_id": 0, "name": "someone"},
    ]
    doc["coverage"] = {"first": "2025-03-05", "last": "2026-09-01", "messages": 120, "passes": 3}
    return doc


@pytest.fixture()
def profiled(seeded, write):
    def fill(conn):
        _store_profile(conn, RANC1D, _v2_doc(), nickname="ranc1d", summary="Overview: builds bots", count=120)
        legacy = {"overview": "An older sketch.", "hobbies": ["chess", " "], "inferred_location": "somewhere cold"}
        _store_profile(conn, GLENN, legacy, summary="Overview: An older sketch. Mentions llamas.", count=40)

    write(fill)
    return seeded


def test_profile_detail_is_laid_out_in_sections(profiled):
    detail = browse.profile_detail(profiled, RANC1D)

    assert (detail["user_id"], detail["name"], detail["version"]) == (str(RANC1D), "ranc1d", 2)
    assert (detail["item_count"], detail["message_count"]) == (4, 120)
    assert detail["source_max_message_id"] == "567890123456789012"
    assert detail["coverage"]["passes"] == 3
    # Every section is listed, in the document's order, with the label a reader sees.
    assert [s["key"] for s in detail["sections"]] == list(pdoc.SECTION_KEYS)
    by_key = {s["key"]: s for s in detail["sections"]}
    assert (by_key["hobbies"]["label"], by_key["hobbies"]["cap"]) == ("Hobbies", 12)
    # last_seen is only worth showing when it differs from the first date.
    assert by_key["hobbies"]["items"] == [
        {"id": "hob1", "text": "restores vintage road bikes", "date": "2025-03-05", "last_seen": "2026-09-01"},
        {"id": "hob2", "text": "bakes sourdough", "date": "2025-04-01"},
    ]


def test_people_in_a_profile_are_named_from_the_archive(profiled):
    people = {s["key"]: s for s in browse.profile_detail(profiled, RANC1D)["sections"]}["relationships_with_others"]

    # rel1 has only an ID, so the name on that member's latest message fills it in.
    assert people["items"][0] == {
        "id": "rel1",
        "text": "trades build advice",
        "date": "2026-01-02",
        "name": "glenn",
        "user_id": str(GLENN),
    }
    assert people["items"][1]["name"] == "someone" and "user_id" not in people["items"][1]


def test_old_format_profiles_still_read(profiled):
    detail = browse.profile_detail(profiled, GLENN)

    assert (detail["version"], detail["overview"], detail["name"]) == (1, "An older sketch.", "glenn")
    assert [(s["label"], [i["text"] for i in s["items"]]) for s in detail["sections"]] == [
        ("Hobbies", ["chess"]),
        ("Inferred location", ["somewhere cold"]),
    ]
    assert detail["item_count"] == 2


def test_a_profile_with_no_document_falls_back_to_its_text(seeded, write):
    write(lambda conn: _store_profile(conn, RANC1D, None, summary="plain text only"))

    detail = browse.profile_detail(seeded, RANC1D)

    assert (detail["version"], detail["sections"], detail["summary"]) == (0, [], "plain text only")
    assert browse.profile_detail(seeded, GLENN) is None


def test_profile_list_leaves_the_bodies_out(profiled):
    rows = browse.list_profiles(profiled)

    assert [(r["user_id"], r["name"], r["version"], r["item_count"]) for r in rows] == [
        (str(RANC1D), "ranc1d", 2, 4),
        (str(GLENN), "glenn", 1, 2),
    ]
    assert rows[0]["preview"] == "Builds bots and argues about hardware."
    assert not {"summary", "structured_json", "sections"} & rows[0].keys()


def test_profile_search_matches_name_id_or_text(profiled):
    def found(term):
        return [r["user_id"] for r in browse.list_profiles(profiled, search=term)]

    assert found("ranc") == [str(RANC1D)]
    assert found(str(GLENN)) == [str(GLENN)]
    assert found("llamas") == [str(GLENN)]
    assert found("nobody says this") == []
    assert browse.count_profiles(profiled) == 2


def test_an_archive_without_a_profile_table_has_no_profiles(tmp_path):
    path = tmp_path / "bare.db"
    sqlite3.connect(path).close()
    conn = browse.connect(path)

    assert browse.list_profiles(conn) == []
    assert browse.count_profiles(conn) == 0
    assert browse.profile_detail(conn, RANC1D) is None


def _self_state():
    doc = pdoc.new_document(pdoc.SELF)
    doc["overview"] = "i roast people and then help anyway"
    doc["sections"]["relationships"] = [
        {"id": "rel1", "text": "i trust his movie picks", "date": "2026-05-01", "user_id": RANC1D, "name": "ranc1d"}
    ]
    doc["coverage"] = {"first": "2026-01-01", "last": "2026-05-01", "messages": 77, "passes": 2}
    return {"doc": doc, "updated_at": "2026-10-13T11:10:28+00:00", "model": "test-model"}


def test_soupys_memory_takes_the_same_shape_as_a_profile():
    detail = browse.self_memory_detail(_self_state())

    assert (detail["user_id"], detail["kind"], detail["name"]) == ("self", "self", "Soupy")
    assert (detail["message_count"], detail["model_used"], detail["item_count"]) == (77, "test-model", 1)
    assert [s["label"] for s in detail["sections"]] == [s.label for s in pdoc.SELF.sections]
    assert browse.self_memory_detail(None) is None
    assert browse.self_memory_detail({"doc": {"overview": "pre-v2"}}) is None


def test_soupys_memory_is_listed_only_when_the_search_fits_it():
    state = _self_state()

    assert browse.self_memory_row(state)["item_count"] == 1
    assert browse.self_memory_row(state, search="movie picks") is not None
    assert browse.self_memory_row(state, search="RANC1D") is not None
    assert browse.self_memory_row(state, search="soup") is not None
    assert browse.self_memory_row(state, search="llamas") is None


# ---------------------------------------------------------------------------
# Media archive
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def media(tmp_path):
    full = "A coastal scene with a sandy beach, turquoise water and buildings on the hillside above."
    _write_jsonl(
        tmp_path / "images" / "index.jsonl",
        [
            {"filename": "old_sd.png", "prompt": "a castle", "username": "glenn", "user_id": GLENN, "seed": 7},
            {
                "filename": "vision_1.png",
                "prompt": "[Vision] " + full[:40],
                "event_type": "vision",
                "username": "ranc1d",
                "user_id": RANC1D,
                "guild_id": 1,
                "channel_id": GENERAL,
            },
            {"filename": "flux_1.png", "prompt": "a theme park", "source": "flux-img2img", "username": "ranc1d"},
        ],
    )
    _write_jsonl(
        tmp_path / "messages.jsonl",
        [
            {"content": "hello", "event_type": "message", "username": "ranc1d"},
            {"content": "🖼️ Vision: " + full, "event_type": "vision", "image_filename": "vision_1.png"},
            {
                "content": "Generated flux image: a theme park",
                "event_type": "image_generation",
                "image_filename": "flux_1.png",
            },
        ],
    )
    return browse.MediaArchive(tmp_path), full


def test_images_are_newest_first_with_ids_as_strings(media):
    archive, _full = media
    page = archive.images()

    assert [it["filename"] for it in page["items"]] == ["flux_1.png", "vision_1.png", "old_sd.png"]
    assert page["counts"] == {"all": 3, "vision": 1, "generated": 2}
    assert [it["source"] for it in page["items"]] == ["flux-img2img", "vision", "sd"]
    vision = page["items"][1]
    assert (vision["user_id"], vision["channel_id"], vision["guild_id"]) == (str(RANC1D), str(GENERAL), "1")


def test_a_flux_row_says_how_the_picture_was_made(tmp_path):
    made = {
        "model": "klein", "steps": 4, "guidance": 1.0, "strength": 0.6, "seconds": 31.2, "action": "Img2Img",
        "source_image": "images/flux_0.png", "original_prompt": "a park",
    }
    _write_jsonl(
        tmp_path / "images" / "index.jsonl",
        [
            {"filename": "flux_0.png", "prompt": "a park", "source": "flux"},
            {"filename": "flux_1.png", "prompt": "a theme park", "source": "flux-img2img", **made},
            # Only a file in media/images or media/flux_sources is linked to.
            {"filename": "flux_2.png", "prompt": "x", "source": "flux-edit", "source_image": "../.env-stable"},
        ],
    )
    odd, new, old = browse.MediaArchive(tmp_path).images()["items"]
    assert {key: new[key] for key in made} == made
    assert odd["source_image"] is None
    # A picture from before these were recorded has them all, empty.
    assert [old[key] for key in made] == [None] * len(made)


def test_a_vision_row_carries_the_whole_description(media):
    archive, full = media
    by_name = {it["filename"]: it for it in archive.images()["items"]}

    # The index keeps the first ~110 characters; the rest is only in messages.jsonl.
    assert by_name["vision_1.png"]["text"] == full
    # A generated image's text is its prompt, not the "Generated flux image: ..." log line.
    assert by_name["flux_1.png"]["text"] == "a theme park"


def test_image_kind_and_search(media):
    archive, _full = media

    assert [it["filename"] for it in archive.images(kind="vision")["items"]] == ["vision_1.png"]
    assert archive.images(kind="generated")["total"] == 2
    assert archive.images(kind="nonsense")["kind"] == "all"
    # "hillside" is past the point where the index cuts the description off.
    assert [it["filename"] for it in archive.images(search="HILLSIDE")["items"]] == ["vision_1.png"]
    assert [it["filename"] for it in archive.images(search="glenn")["items"]] == ["old_sd.png"]
    assert archive.images(search="flux_1")["total"] == 1
    assert archive.images(limit=1, offset=1)["items"][0]["filename"] == "vision_1.png"


def test_new_images_show_up_without_a_restart(media, tmp_path):
    archive, _full = media
    assert archive.images()["total"] == 3

    with open(tmp_path / "images" / "index.jsonl", "a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write(json.dumps({"filename": "new.png", "prompt": "a lighthouse"}) + "\n")

    page = archive.images()
    assert (page["total"], page["items"][0]["filename"]) == (4, "new.png")


def test_channel_names_go_on_the_page_not_the_cache(media):
    archive, _full = media
    page = archive.images()
    asked = []

    def names(guild_id):
        asked.append(guild_id)
        return {str(GENERAL): "general"}

    browse.attach_channel_names(page["items"], names)

    assert [it.get("channel_name") for it in page["items"]] == [None, "general", None]
    assert asked == ["1"]
    assert all("channel_name" not in it for it in archive.images()["items"])


def test_activity_log_pages_newest_first(media):
    archive, _full = media

    assert [r["event_type"] for r in archive.log(limit=2)["items"]] == ["image_generation", "vision"]
    assert [r["event_type"] for r in archive.log(limit=2, offset=2)["items"]] == ["message"]
    assert archive.log()["total"] == 3
    assert archive.log(offset=50)["items"] == []


def test_a_missing_media_folder_is_an_empty_archive(tmp_path):
    archive = browse.MediaArchive(tmp_path / "nowhere")

    assert archive.images() == {
        "items": [],
        "total": 0,
        "limit": 60,
        "offset": 0,
        "kind": "all",
        "counts": {"all": 0, "vision": 0, "generated": 0},
    }
    assert archive.log()["total"] == 0
