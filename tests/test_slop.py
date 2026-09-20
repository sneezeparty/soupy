"""Tests for the anti-slop gate on Soupy's Bluesky writing.

Two things are being pinned here:

* ``soupy.slop`` catches the shapes the local model actually produces in a
  one- or two-sentence post, and leaves Soupy's real lines alone. The "clean"
  cases below are posts Soupy already published; the sloppy ones are the same
  thoughts written the way an AI reaches for.
* the cog keeps its politics. The rules are about shape, so a change that
  quietly drops the pro-worker / anti-war / punch-up framing is a regression,
  not a style fix.
"""

from __future__ import annotations

import pytest

from soupy import slop

CLEAN = [
    "seven minutes to charge an ev battery feels less like innovation and more like solving a problem they created",
    "the $200k payout keeps the career alive while the ethics complaint sits in a drawer",
    "they only noticed her credentials after she started working for someone who ignores them",
    "boeing got a waiver two weeks after the donation cleared",
]

SLOPPY = [
    ("here's the thing: the payout was never about ethics.", "throat-clearing opener"),
    ("it's not a scandal, it's a business model.", "contrast setup"),
    ("the problem isn't the donation. the problem is who took it.", "contrast setup"),
    ("what if the waiver was the point all along?", "rhetorical setup"),
    ("the donation cleared, the waiver followed, the satisfying kind of coincidence.", "two commas in one sentence"),
    ("they took the money and wrote the rules. that's the whole game.", "slogan closer"),
    ("this is just another example of how capital works.", "adverb crutch"),
    ("a nuanced take on a pivotal moment for labor rights.", "deck word"),
    ("the payout speaks volumes about who sets the price of silence.", "vague significance"),
    ("the market rewards the people who write the rules.", "things acting on their own"),
    ("before the merger, they promised jobs.", "before/after arc"),
    ("the implications are significant for every worker in the state.", "vague significance"),
    ("the deal serves as a reminder of who pays.", "copula dodge"),
    ("a 30 percent raise — after a decade of wage freezes.", "em dash"),
]


@pytest.mark.parametrize("text", CLEAN)
def test_soupys_own_posts_come_back_clean(text):
    assert slop.slop_tells(text) == []


@pytest.mark.parametrize("text,tell", SLOPPY)
def test_each_tell_is_caught_and_named(text, tell):
    assert tell in slop.slop_tells(text)


def test_politics_and_anger_are_not_tells():
    """The gate reads shape. Nothing about a target, a swear, or a stance may trip it."""
    for text in [
        "the contractors bill the state twice and the workers eat the difference",
        "billionaires bought the exemption and the senate delivered it in a week",
        "every one of these defense contracts is a transfer from schools to shareholders",
        "he gutted the union and called it a modernization plan, which is a lie",
    ]:
        assert slop.slop_tells(text) == [], text


def test_one_comma_is_fine_two_is_a_tell():
    assert slop.slop_tells("the waiver landed friday, a week after the cheque") == []
    assert "two commas in one sentence" in slop.slop_tells(
        "the waiver landed friday, a week after the cheque, the usual sequence"
    )


def test_a_tell_is_named_once_however_often_it_fires():
    assert slop.slop_tells("this is just really just noise") == ["adverb crutch"]


def test_repeated_openers_are_flagged_against_recent_posts():
    recent = ["it is funny how they only notice her credentials now", "another layer of circular financing"]
    assert slop.repeats_opener("it is funny how the ethics complaint vanished", recent)
    assert not slop.repeats_opener("the ethics complaint vanished the same week", recent)


def test_ranking_puts_the_cleanest_draft_first_and_keeps_ties_in_order():
    drafts = [
        "this is just another example of how capital works.",
        "boeing got a waiver two weeks after the donation cleared",
        "they only noticed her credentials after she started working for someone who ignores them",
    ]
    ranked = slop.rank_candidates(drafts)
    assert [text for text, _ in ranked] == [drafts[1], drafts[2], drafts[0]]
    assert ranked[-1][1] == ["adverb crutch"]


def test_a_familiar_opener_loses_to_an_equally_clean_one():
    drafts = ["boeing got a waiver two weeks later", "the waiver landed two weeks later"]
    ranked = slop.rank_candidates(drafts, recent=["boeing got a pass on the same rule"])
    assert ranked[0][0] == drafts[1]
    assert ranked[1][1] == ["repeats a recent opener"]


# ---------------------------------------------------------------------------
# The cog
# ---------------------------------------------------------------------------


def test_every_bluesky_writing_prompt_carries_the_rules():
    from soupy.cogs import bluesky

    for prompt in (bluesky.COMMENT_SYSTEM, bluesky.QUOTE_POST_SYSTEM, bluesky.ORIGINAL_POST_SYSTEM):
        assert "NO SLOP" in prompt
        assert "not about what you think" in prompt, "the rules must say they don't touch the stance"


def test_the_prompts_keep_soupys_politics():
    from soupy.cogs import bluesky

    assert "pro-worker" in bluesky.COMMENT_SYSTEM
    assert "punch UP" in bluesky.COMMENT_SYSTEM
    assert "pro-worker" in bluesky.QUOTE_POST_SYSTEM
    assert "capital serves the owning class" in bluesky.ORIGINAL_POST_SYSTEM
    assert "anti-war" in bluesky.ORIGINAL_POST_SYSTEM


def test_the_gate_keeps_the_cleanest_drafts_and_never_returns_nothing():
    from soupy.cogs import bluesky

    drafts = ["this is just noise.", "the waiver cleared in a week"]
    assert bluesky._drop_slop(drafts, [], "Post") == ["the waiver cleared in a week"]

    # Nothing clean: the drafts tied on one tell each both go through to the judge.
    all_sloppy = ["this is just noise.", "here's the thing: it is what it is."]
    assert bluesky._drop_slop(all_sloppy, [], "Post") == all_sloppy
    # A draft with more tells than another is dropped even when nothing is clean.
    assert bluesky._drop_slop(
        ["what if the market rewards it, again, the usual way?", "this is just noise."], [], "Post"
    ) == ["this is just noise."]
    assert bluesky._drop_slop([], [], "Post") == []


def test_recent_texts_reads_each_history_bucket():
    from soupy.cogs import bluesky

    history = {
        "posts": [{"text": "a"}, {"text": "b"}],
        "comments": [{"comment": "c"}],
        "reposts": [{"commentary": "d"}, "junk"],
    }
    assert bluesky._recent_texts(history, "posts", "text") == ["a", "b"]
    assert bluesky._recent_texts(history, "comments", "comment") == ["c"]
    assert bluesky._recent_texts(history, "reposts", "commentary") == ["d"]
    assert bluesky._recent_texts({}, "posts", "text") == []
