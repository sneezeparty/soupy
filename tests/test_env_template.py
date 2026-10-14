"""Tests for `.env-stable.example` as the bot will actually receive it.

Two parsers read this file. The bot calls python-dotenv, but under
`run_all.py` the web panel's `parse_env` reads it first and puts its values
in the bot's environment, where they win. A line the two read differently
gives a fresh install a setting nobody wrote, with no error anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import dotenv_values

from installer.steps.s06_write_config import WIZARD_KEYS
from web.services.env_store import parse_env

TEMPLATE = Path(__file__).resolve().parent.parent / ".env-stable.example"


def test_template_reads_the_same_in_the_bot_and_the_panel():
    # Three RAG_* lines carried inline comments, which the panel kept as part
    # of the value ("6    # vector chunks ..."), so int() failed and the bot
    # fell back to its code defaults. The persona's em dash arrived as mojibake.
    _lines, panel = parse_env(TEMPLATE)
    bot = {key: value or "" for key, value in dotenv_values(TEMPLATE).items()}

    assert panel.keys() == bot.keys()
    differing = sorted(key for key in bot if panel[key] != bot[key])
    assert differing == []


def test_panel_parser_keeps_non_ascii_characters(tmp_path: Path):
    env = tmp_path / ".env-stable"
    env.write_text('ONE_LINE="dry — knowing, café \\"quoted\\""\nMANY="first — line\nsecond 😀"\n', encoding="utf-8")

    _lines, values = parse_env(env)

    assert values["ONE_LINE"] == 'dry — knowing, café "quoted"'
    assert values["MANY"] == "first — line\nsecond 😀"


def test_every_wizard_key_has_a_line_in_the_template():
    # A key with no line to replace is appended under "# Added by installer",
    # away from the comment that documents it.
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = [key for key in WIZARD_KEYS if not re.search(rf"(?m)^{re.escape(key)}=", text)]
    assert missing == []
