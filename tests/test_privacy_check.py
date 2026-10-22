"""tools/privacy_check.py stops data from the bot's private archive reaching the public repository.

Everything "private" here is made up for the test: a scratch archive, profile,
memory file and ``.env-stable`` are built in a temporary folder, and a scratch
git repository stands in for this one.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess

import pytest

from tools import privacy_check as pc

# The scratch server's members. None of this exists anywhere else.
SERVER_ID = 555000000000000001
CHANNEL_ID = 555000000000000002
VEX_ID = 555000000000000010
MESSAGE_ID = 555000000000000101
TOKEN = "zq9-SCRATCH-TOKEN-4f7a1c2e"
REAL_LINE = "honestly the lighthouse keeper stole my bicycle again last tuesday"
MEMORY_LINE = "i told him his ferry schedule joke was the worst thing i heard all week"

# Stands in for /usr/share/dict/words, which CI machines don't have.
DICTIONARY = {
    "rain", "stupid", "lives", "plays", "jump", "format", "count", "down", "believe", "party", "ferry",
    "schedule", "lighthouse", "keeper", "worst", "thing", "heard", "about", "there", "their", "which",
    "would", "write", "tests", "these", "words", "fixture", "member", "invented", "harbor", "quartz",
    "bot", "captain", "think", "fine", "secret", "recipe", "master", "mention", "never", "anyone",
}  # fmt: skip


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A scratch project folder holding a private archive, profile, memory and config."""
    monkeypatch.delenv("SOUPY_DB_DIR", raising=False)
    root = tmp_path / "project"
    (root / "soupy_database" / "databases").mkdir(parents=True)
    con = sqlite3.connect(root / "soupy_database" / "databases" / f"guild_{SERVER_ID}.db")
    con.executescript("""
        create table guild_metadata (guild_id integer, guild_name text);
        create table channels (channel_id integer, channel_name text);
        create table messages (message_id integer, username text, nickname text, user_id integer,
                               message_content text, channel_id integer, image_description text);
        create table user_profile_summaries (user_id integer, nickname_hint text, summary text, structured_json text);
        create table self_chunks (chunk_text text);
        """)
    con.execute("insert into guild_metadata values (?, 'Quartz Harbor')", (SERVER_ID,))
    con.execute("insert into channels values (?, 'vexlounge')", (CHANNEL_ID,))
    con.executemany(
        "insert into messages values (?, ?, ?, ?, ?, ?, ?)",
        [
            (MESSAGE_ID, "zorblat99", "Captain Vex", VEX_ID, REAL_LINE, CHANNEL_ID, None),
            (
                MESSAGE_ID + 1,
                ".mirk.",
                "rain",
                VEX_ID + 1,
                "short one",
                CHANNEL_ID,
                "a ferret in a tiny raincoat on a pier",
            ),
            (MESSAGE_ID + 2, "plainuser", "stupid bot", VEX_ID + 2, "another short one", CHANNEL_ID, None),
        ],
    )
    profile = {"sections": {"location": [{"text": "lives in Quillhaven"}], "hobbies": [{"text": "plays Glimmerforge"}]}}
    con.execute(
        "insert into user_profile_summaries values (?, 'Captain Vex', ?, ?)",
        (VEX_ID, "Lives in Quillhaven.", json.dumps(profile)),
    )
    # Five more members who all play the same game: common talk, not a fact about one of them.
    for n in range(5):
        con.execute(
            "insert into user_profile_summaries values (?, ?, 'plays Starwhittle', '{}')",
            (VEX_ID + 20 + n, f"member{n}xyz"),
        )
    con.execute("insert into self_chunks values (?)", (MEMORY_LINE,))
    con.commit()
    con.close()

    (root / "data" / "self_md").mkdir(parents=True)
    (root / "data" / "self_md" / "guild.md").write_text("i think Captain Vex is fine\n", encoding="utf-8")
    (root / "data" / "bluesky_engage_history.json").write_text('{"post": "a public post about ferries in general"}')
    (root / "media" / "images").mkdir(parents=True)
    (root / "media" / "images" / "index.jsonl").write_text(
        json.dumps({"filename": "a.png", "user_id": VEX_ID + 3, "username": "galleryposter"}) + "\n", encoding="utf-8"
    )
    (root / "prompts").mkdir()
    (root / "prompts" / "behaviour.default.txt").write_text("You are a bot. Be brief and kind to everyone.\n")
    (root / "prompts" / "behaviour.txt").write_text(
        "You are a bot. Never mention the harbor master's secret recipe to anyone at all.\n"
    )
    (root / ".env-stable.example").write_text(
        "DISCORD_TOKEN=your_discord_bot_token_here\nOPENAI_BASE_URL=http://localhost:1234/v1\nLOCAL_CHAT=your-model\n"
    )
    (root / ".env-stable").write_text(
        f"DISCORD_TOKEN={TOKEN}\nOPENAI_BASE_URL=http://quartz-box.lan:5112/v1\nLOCAL_CHAT=acme/widget-model-7b\n"
        f"OWNER_IDS={VEX_ID + 9},12345\nMAX_TOKENS=4096\n"
    )
    (root / "tools").mkdir()
    (root / "tools" / "privacy_allow.txt").write_text("# allowed here\nacme/widget-model-7b\n", encoding="utf-8")
    return root


@pytest.fixture
def private(machine):
    return pc.load_private_data(machine)


def check(private, text, *, added=None, allowed=(), known=frozenset(), path="tests/test_x.py"):
    """Findings for a file whose content is ``text``. By default every line is new."""
    lines = text.splitlines()
    new = set(range(1, len(lines) + 1)) if added is None else set(added)
    checker = pc.Checker(
        private, set(allowed), dictionary=DICTIONARY, known_words=None if known is None else set(known)
    )
    return checker.check(pc.Change(path, lines, new))


def kinds(findings):
    return [(f.line, f.kind) for f in findings]


# ---- what is read from the machine --------------------------------------------------


def test_the_private_data_is_gathered_from_every_source(private):
    assert private.ids[str(SERVER_ID)] == "server ID" and private.ids[str(CHANNEL_ID)] == "channel ID"
    assert private.ids[str(VEX_ID)] == "user ID" and private.ids[str(MESSAGE_ID)] == "message ID"
    assert str(VEX_ID + 3) in private.ids  # only ever seen in the image archive
    assert private.names["captain vex"] == "member's nickname" and private.names["zorblat99"] == "member's username"
    assert private.names["quartz harbor"] == "server name" and "galleryposter" in private.names
    assert REAL_LINE in private.text and MEMORY_LINE in private.text and "quillhaven" in private.text
    # A setting left at the template's value, or too short to be an identifier, is not private.
    keys = {key for key, _ in private.secrets}
    assert keys == {"DISCORD_TOKEN", "OPENAI_BASE_URL", "LOCAL_CHAT", "OWNER_IDS"}
    assert ("OWNER_IDS", str(VEX_ID + 9)) in private.secrets and ("OWNER_IDS", "12345") not in private.secrets


def test_public_posts_and_the_shipped_prompt_are_not_private(private):
    assert "a public post about ferries" not in private.text
    assert "be brief and kind to everyone" not in private.text  # also in behaviour.default.txt
    assert "harbor master's secret recipe" in private.text  # only in the maintainer's own prompt


def test_a_machine_with_no_archive_has_nothing_to_compare(tmp_path, capsys):
    assert pc.load_private_data(tmp_path).empty()
    assert pc.run(["--all", "--root", str(tmp_path)]) == 0
    assert "nothing to compare with" in capsys.readouterr().out


def test_an_archive_that_cannot_be_read_stops_the_commit(machine, capsys):
    """Reading nothing must not pass for finding nothing."""
    (machine / "soupy_database" / "databases" / "guild_1.db").write_bytes(b"this is not a database at all" * 50)
    assert pc.run(["--staged", "--root", str(machine)]) == 2
    assert "could not read the archive" in capsys.readouterr().err


# ---- ids -------------------------------------------------------------------------------


def test_a_real_id_is_found_on_any_line_of_a_touched_file(private):
    text = f"BOB = 123456789012345678\nVEX = {VEX_ID}\nurl = 'https://x/{MESSAGE_ID}'\nhash = 'ab{VEX_ID}'\n"
    found = check(private, text, added=[1])  # the real ones are on lines the change did not add
    assert kinds(found) == [(2, "id"), (3, "id")]
    assert str(VEX_ID) in found[0].what and "user ID" in found[0].what


def test_a_real_id_cannot_be_allowed(private):
    assert kinds(check(private, f"x = {VEX_ID}", allowed={str(VEX_ID)})) == [(1, "id")]


# ---- names -----------------------------------------------------------------------------


def test_real_names_are_found_as_whole_words_in_any_case(private):
    text = (
        'directory = {1: "captain VEX"}\n'  # nickname, other case
        "# thanks zorblat99!\n"  # username
        "zorblat99x = 1; myzorblat99 = 2\n"  # inside longer words: not a match
        "see #vexlounge on Quartz Harbor\n"  # channel and server
        'name = ".mirk."\n'  # punctuation at both ends
    )
    found = check(private, text, added=[])
    assert kinds(found) == [(1, "name"), (2, "name"), (4, "name"), (4, "name"), (5, "name")]
    assert {f.what for f in found if f.line == 4} == {'channel name "vexlounge"', 'server name "Quartz Harbor"'}


def test_a_name_made_of_everyday_words_does_not_match_ordinary_prose(private):
    assert check(private, "the rain came down\nwhat a stupid bot that was\n") == []


def test_an_allowed_name_is_left_alone(private):
    assert check(private, "by zorblat99", allowed={"zorblat99"}) == []


# ---- config values -----------------------------------------------------------------------


def test_values_from_the_real_config_are_found_and_secrets_are_not_echoed(private):
    text = f'TOKEN = "{TOKEN}"\nbase = "http://quartz-box.lan:5112/v1"\nmodel = "acme/widget-model-7b"\nold = "your-model"\n'
    found = check(private, text, added=[])
    assert kinds(found) == [(1, "secret"), (2, "secret"), (3, "secret")]
    assert TOKEN not in found[0].what and "DISCORD_TOKEN" in found[0].what
    assert "quartz-box.lan" in found[1].what
    # A public model name is the usual thing to allow.
    assert kinds(check(private, text, added=[], allowed={"acme/widget-model-7b"})) == [(1, "secret"), (2, "secret")]


# ---- copied text ---------------------------------------------------------------------------


def test_text_copied_from_the_archive_is_found_on_added_lines(private):
    text = f'old = "{REAL_LINE}"\nnew = "{REAL_LINE.upper()}"\nown = "an invented line that nobody ever said at all"\n'
    found = check(private, text, added=[2, 3])
    assert kinds(found) == [(2, "text")]  # line 1 was already there; line 3 is invented


def test_memory_and_prose_without_quote_marks_are_covered(private):
    doc = f"Soupy once said: {MEMORY_LINE}.\nNothing else of note happened that day in the harbor town.\n"
    assert kinds(check(private, doc, path="CHANGELOG.md")) == [(1, "text")]


def test_short_and_stock_phrases_are_not_copied_text(private):
    assert check(private, '"another short one"\n') == []  # in the archive, but three words
    allowed = {"lighthouse keeper stole my bicycle"}
    assert check(private, f'"{REAL_LINE}"', allowed=allowed) == []


def test_many_candidates_take_the_hashed_path_and_find_the_same_line(private):
    filler = [
        f'x{n} = "an invented sentence number {n} about nothing in particular"' for n in range(pc.MANY_CANDIDATES + 50)
    ]
    text = "\n".join(filler[:100] + [f'real = "{REAL_LINE}"'] + filler[100:])
    assert kinds(check(private, text)) == [(101, "text")]


# ---- words only one or two profiles mention ---------------------------------------------------


def test_a_word_from_one_members_profile_is_found_when_it_is_new_to_the_repository(private):
    """The town someone lives in, attached to an invented name: no ID or name gives it away."""
    text = 'item = {"text": "lives in Quillhaven", "name": "marlowe"}\n'
    found = check(private, text)
    assert kinds(found) == [(1, "word")] and '"Quillhaven"' in found[0].what
    assert check(private, text, known={"quillhaven"}) == []  # already in the repository
    assert check(private, text, allowed={"quillhaven"}) == []
    assert check(private, text, added=[]) == []  # only on lines the change adds


def test_a_word_many_profiles_share_is_common_talk(private):
    assert private.profile_words["starwhittle"] == 5 and private.profile_words["quillhaven"] == 1
    assert check(private, "plays Starwhittle on weekends") == []


@pytest.mark.parametrize("word", ["jumped", "believed", "parties", "reformatted", "countdown", "lives", "ferries"])
def test_ordinary_words_and_their_forms_are_not_flagged(private, word):
    private.profile_words[word] = 1
    assert check(private, f"the {word} in question") == []


def test_a_written_out_line_break_does_not_glue_onto_the_next_word(private):
    """In source, "\\nquillhaven" is a line break and a word. Without care it reads as "nquillhaven"."""
    private.profile_words["nharbor"] = 1  # what a profile stored as raw JSON would yield the same way
    assert check(private, 'text = "first line\\nharbor"') == []
    found = check(private, 'text = "first line\\nQuillhaven"')
    assert kinds(found) == [(1, "word")] and '"Quillhaven"' in found[0].what


def test_the_word_check_needs_a_dictionary_and_an_earlier_version_to_compare_with(private):
    text = "lives in Quillhaven"
    assert check(private, text, known=None) == []  # --all: every word would be "new"
    no_dictionary = pc.Checker(private, set(), dictionary=set(), known_words=set())
    assert no_dictionary.check(pc.Change("t.py", [text], {1})) == []


# ---- through git, the way the hooks run it ------------------------------------------------------


def git(repo, *args):
    base = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    return subprocess.run([*base, *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(machine, monkeypatch, tmp_path):
    """The scratch project as a git repository with one clean commit. The private files are ignored."""
    words = tmp_path / "words"
    words.write_text("\n".join(sorted(DICTIONARY)) + "\n", encoding="utf-8")
    monkeypatch.setattr(pc, "DICTIONARIES", (str(words),))
    monkeypatch.delenv("PRE_COMMIT_FROM_REF", raising=False)
    monkeypatch.delenv("PRE_COMMIT_TO_REF", raising=False)
    monkeypatch.delenv("SOUPY_DB_DIR", raising=False)
    (machine / ".gitignore").write_text(
        ".env-stable\ndata/\nmedia/\nsoupy_database/databases/\nprompts/behaviour.txt\n"
    )
    (machine / "tests").mkdir()
    (machine / "tests" / "test_a.py").write_text('NAME = "marlowe"\nTOWN = "Springfield"\n', encoding="utf-8")
    git(machine, "init", "-q", "-b", "main")
    git(machine, "add", "-A")
    git(machine, "commit", "-q", "-m", "start")
    return machine


def test_a_clean_commit_passes(repo, capsys):
    (repo / "tests" / "test_a.py").write_text('NAME = "marlowe"\nTOWN = "Springfield"\nPET = "a ferret"\n')
    git(repo, "add", "-A")
    assert pc.run(["--root", str(repo)]) == 0
    assert "nothing private in this commit" in capsys.readouterr().out


def test_a_commit_with_private_data_is_stopped_and_says_where(repo, capsys):
    (repo / "tests" / "test_a.py").write_text(
        f'NAME = "Captain Vex"\nTOWN = "Quillhaven"\nUSER = {VEX_ID}\nSAID = "{REAL_LINE}"\n', encoding="utf-8"
    )
    (repo / "web" / "static" / "vendor").mkdir(parents=True)
    (repo / "web" / "static" / "vendor" / "lib.min.js").write_text("var zorblat99 = 1;")  # third-party code: skipped
    git(repo, "add", "-A")
    assert pc.run(["--staged", "--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "4 problem(s) in this commit" in out
    for expected in (
        "tests/test_a.py:1",
        'nickname "Captain Vex"',
        "tests/test_a.py:2",
        '"Quillhaven"',
        "tests/test_a.py:3",
        "tests/test_a.py:4",
    ):
        assert expected in out
    assert "lib.min.js" not in out and "tools/privacy_allow.txt" in out


def test_only_what_is_staged_is_judged(repo):
    (repo / "notes.txt").write_text("Captain Vex said hello")  # in the working tree, not staged
    assert pc.run(["--staged", "--root", str(repo), "-q"]) == 0


def test_a_push_is_checked_over_the_commits_it_would_publish(repo, monkeypatch, capsys):
    """Catches a commit made with --no-verify."""
    base = git(repo, "rev-parse", "HEAD")
    (repo / "docs.md").write_text("Thanks to zorblat99 for the idea.\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "second")
    monkeypatch.setenv("PRE_COMMIT_FROM_REF", base)
    monkeypatch.setenv("PRE_COMMIT_TO_REF", git(repo, "rev-parse", "HEAD"))
    assert pc.run(["--push", "--root", str(repo)]) == 1
    assert "docs.md:1" in capsys.readouterr().out
    # A push of commits that changed nothing private passes.
    monkeypatch.setenv("PRE_COMMIT_FROM_REF", git(repo, "rev-parse", "HEAD"))
    assert pc.run(["--push", "--root", str(repo), "-q"]) == 0
    # A brand-new branch has no earlier point on the remote: the whole tree is checked.
    monkeypatch.setenv("PRE_COMMIT_FROM_REF", "0" * 40)
    assert pc.run(["--push", "--root", str(repo), "-q"]) == 1


def test_every_tracked_file_can_be_checked_at_once(repo, capsys):
    assert pc.run(["--all", "--root", str(repo), "-q"]) == 0
    (repo / "docs.md").write_text(f"server {SERVER_ID}\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "second")
    assert pc.run(["--all", "--root", str(repo)]) == 1
    assert "docs.md:1" in capsys.readouterr().out


def test_a_commit_message_is_checked_too(repo, tmp_path, capsys):
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("fix: handle Captain Vex's edge case\n\n# Please enter the commit message. zorblat99\n")
    assert pc.run(["--message", str(message), "--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "the commit message:1" in out and "zorblat99" not in out  # git's own comment lines are not part of it
    message.write_text("fix: handle an edge case in name matching\n")
    assert pc.run(["--message", str(message), "--root", str(repo), "-q"]) == 0


# ---- this repository's own wiring ---------------------------------------------------------------


def test_the_hooks_run_the_check_at_commit_at_push_and_on_the_message():
    config = (pc.REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "default_install_hook_types: [pre-commit, pre-push, commit-msg]" in config
    for entry, stage in (
        ("python3 tools/privacy_check.py --staged", "stages: [pre-commit]"),
        ("python3 tools/privacy_check.py --push", "stages: [pre-push]"),
        ("python3 tools/privacy_check.py -q --message", "stages: [commit-msg]"),
    ):
        assert entry in config and stage in config, entry


def test_the_allow_list_holds_no_ids():
    allowed = pc.load_allowed(pc.REPO_ROOT / "tools" / "privacy_allow.txt")
    assert allowed and not [entry for entry in allowed if pc.ID_RE.search(entry)]
