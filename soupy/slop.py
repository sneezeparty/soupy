"""
The anti-slop gate for Soupy's Bluesky writing: the rules in the prompt and the
check that reads the drafts back.

Soupy writes its posts, replies and quote-post commentary with a local model,
which reaches for the shapes that out a draft as AI: "not X, it's Y", a main
clause with two trailing comma-fragments, a slogan closer, deck vocabulary
("nuanced", "underscores", "a testament to"), adverb crutches. They survive a
persona prompt, because they're a shape rather than a word.

Two halves, kept together because they encode the same rules:

* :data:`SLOP_RULES` goes into the system prompt for every Bluesky draft.
* :func:`slop_tells` reads a draft back and names the tells it still has.
  Each pipeline writes three candidates, so :func:`rank_candidates` just
  prefers the cleanest instead of spending another LLM call on a rewrite.

The checks are tuned for one- or two-sentence posts, where two commas really is
an appositive triple and "what if" really is a rhetorical setup. Don't reuse
them on prose without re-tuning: a paragraph is allowed more.

Repetition is the tell this model falls into hardest, so
:func:`repeats_opener` compares a draft's first few words against what Soupy
already posted ("it is funny how ..." opened three of its last six posts).

Adapted from the stop-slop skill by Hardik Pandya (MIT), trimmed to the tells
that fit in 300 characters.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

# Goes into the system prompt of every Bluesky draft. Short on purpose: a local
# model follows five sharp rules better than twenty soft ones, and every line
# here is a shape it actually produces.
SLOP_RULES = (
    "NO SLOP. these shapes read as bot writing. none of them:\n"
    "- no contrast setups: 'not x, it's y', 'isn't about x, it's about y', 'the problem isn't x'. "
    "say the thing you mean, once.\n"
    "- no throat-clearing openers: 'here's the thing', 'turns out', 'make no mistake', 'the truth is'. "
    "start at the point.\n"
    "- no rhetorical setups: 'what if', 'think about it', 'ask yourself'.\n"
    "- no slogan or restatement after the point lands. the last sentence is a real sentence, not a tag "
    "like 'that's the whole game' or 'the quiet part out loud'. stop when you're done.\n"
    "- one comma at most. never a main clause plus two trailing fragments.\n"
    "- no adverb crutches: really, just, literally, actually, simply, genuinely, honestly, truly.\n"
    "- no deck words: nuanced, pivotal, underscores, testament, myriad, leverage, landscape, seamless, "
    "delve, tapestry, realm, showcase.\n"
    "- no vague significance: 'the implications are', 'the stakes are high', 'speaks volumes', "
    "'a stark reminder'.\n"
    "- things don't act. markets don't reward, data doesn't tell, the culture doesn't shift. "
    "name who did it.\n"
    "- no before/after arcs ('before x, they y') and no chains ('x becomes y, y becomes z').\n"
    "- anchor it: a name, a number, the actual thing that happened. a true but generic line is still slop.\n"
    "- don't open the way you opened your last post. vary the shape and the length.\n"
    "the test: would a tired person type this to a friend, or does it sound like it's being presented?\n"
    "these are rules about shape, not about what you think. your politics, your targets and your anger "
    "stay exactly as they are. say the same thing, in fewer and plainer words.\n"
)

_WORD = r"(?<!\w){}(?!\w)"


def _any(*alternatives: str) -> str:
    return r"(?:" + "|".join(alternatives) + r")"


# (name, pattern). Names are what shows up in the log, so they say what's wrong.
_TELLS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "contrast setup",
        re.compile(
            r"\b(?:is|are|was|were|it'?s|that'?s)?\s*(?:not|isn'?t|aren'?t|wasn'?t)\s+(?:about\s+)?[^.,;]{2,45}[,;]\s*"
            r"(?:it'?s|it is|they'?re|but|that'?s)\b|\bthe\s+(?:question|problem|issue|answer|point)\s+"
            r"(?:isn'?t|is not)\b",
            re.I,
        ),
    ),
    (
        "throat-clearing opener",
        re.compile(
            r"^\W*(?:here'?s\s+(?:the\s+thing|what|why|how)|(?:it\s+)?turns\s+out|the\s+truth\s+is|"
            r"let\s+me\s+be\s+clear|make\s+no\s+mistake|can\s+we\s+talk\s+about)\b",
            re.I,
        ),
    ),
    (
        "emphasis crutch",
        re.compile(
            r"\b(?:let\s+that\s+sink\s+in|full\s+stop|this\s+matters\s+because|and\s+that'?s\s+okay|"
            r"here'?s\s+why\s+that\s+matters)\b",
            re.I,
        ),
    ),
    ("rhetorical setup", re.compile(r"\b(?:what\s+if|think\s+about\s+it|ask\s+yourself|imagine\s+if)\b", re.I)),
    (
        "slogan closer",
        re.compile(
            r"(?:\.|^)\s*(?:that'?s\s+it|that'?s\s+the\s+(?:whole\s+)?\w+|welcome\s+to\s+\w+|"
            r"the\s+quiet\s+part\s+out\s+loud)\s*\.?\s*$|,\s*the\s+\w+\s+(?:thing|part)\s*\.?\s*$",
            re.I,
        ),
    ),
    (
        "adverb crutch",
        re.compile(
            _WORD.format(
                _any(
                    "really",
                    "just",
                    "literally",
                    "actually",
                    "simply",
                    "genuinely",
                    "honestly",
                    "truly",
                    "deeply",
                    "fundamentally",
                    "inevitably",
                    "interestingly",
                    "importantly",
                    "crucially",
                    "basically",
                    "essentially",
                )
            ),
            re.I,
        ),
    ),
    (
        "deck word",
        re.compile(
            _WORD.format(
                _any(
                    "delve",
                    "tapestry",
                    "multifaceted",
                    "nuanced",
                    "foster",
                    "realm",
                    "leverages?",
                    "leveraging",
                    "interplay",
                    "landscape",
                    "intricacies",
                    "intricate",
                    "pivotal",
                    "underscores?",
                    "garners?",
                    "showcases?",
                    "vibrant",
                    "testament",
                    "myriad",
                    "plethora",
                    "facilitates?",
                    "utilizes?",
                    "seamless",
                    "cutting-edge",
                    "groundbreaking",
                )
            ),
            re.I,
        ),
    ),
    (
        "jargon",
        re.compile(
            r"\b(?:unpack(?:ing)?\s+the|lean(?:ing)?\s+into|game-changer|double\s+down|deep\s+dive|"
            r"circle\s+back|moving\s+forward|at\s+the\s+end\s+of\s+the\s+day|in\s+today'?s\s+\w+)\b",
            re.I,
        ),
    ),
    (
        "copula dodge",
        re.compile(r"\b(?:serves\s+as|stands\s+as|functions\s+as|boasts\s+a|represents\s+a|marks\s+a)\b", re.I),
    ),
    (
        "vague significance",
        re.compile(
            r"\b(?:the\s+implications\s+are|the\s+stakes\s+are|the\s+consequences\s+are|"
            r"the\s+reasons\s+are\s+structural|speaks\s+volumes|a\s+stark\s+reminder|"
            r"a\s+testament\s+to|setting\s+the\s+stage|pivotal\s+moment|reflects\s+broader|"
            r"shaping\s+the\s+future)\b",
            re.I,
        ),
    ),
    (
        "things acting on their own",
        re.compile(
            r"\bthe\s+(?:market|markets|data|numbers|culture|conversation|algorithm|system|decision)\s+"
            r"(?:rewards?|punishes?|tells?|shifts?|moves?|decides?|emerges?|demands?|knows?)\b",
            re.I,
        ),
    ),
    ("before/after arc", re.compile(r"\bbefore\s+\w+[\w\s]{0,20},\s*(?:it|we|they|you|everyone)\b", re.I)),
    ("transformation chain", re.compile(r"\b(\w+)\s+becomes?\s+\w+[.,;]\s*\w*\s*becomes?\b", re.I)),
    (
        "chatbot artifact",
        re.compile(
            r"\b(?:great\s+question|i\s+hope\s+this\s+helps|as\s+an\s+ai|as\s+a\s+language\s+model|"
            r"let\s+me\s+know\s+if)\b",
            re.I,
        ),
    ),
    ("hedged urgency", re.compile(r"\b(?:may\s+be\s+one\s+of\s+the\s+most|potentially\s+transformative)\b", re.I)),
    ("em dash", re.compile(r"[—–]")),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def slop_tells(text: str) -> List[str]:
    """Name the AI tells in one draft, in the order they're listed above.

    Returns an empty list for a clean draft. A tell is named once however many
    times it fires, since the caller is counting drafts against each other, not
    occurrences.
    """
    draft = (text or "").strip()
    if not draft:
        return []
    found = [name for name, pattern in _TELLS if pattern.search(draft)]
    # Appositive triple: a claim with two trailing fragments. In a post-length
    # sentence a second comma is the trigger; in prose it wouldn't be.
    if any(sentence.count(",") >= 2 for sentence in _SENTENCE_SPLIT.split(draft)):
        found.append("two commas in one sentence")
    return found


def opener(text: str, words: int = 3) -> str:
    """The first few words, lowercased, for comparing one draft against earlier posts."""
    tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    return " ".join(tokens[:words])


def repeats_opener(text: str, recent: Iterable[str], words: int = 3) -> bool:
    """True when this draft starts the same way as something Soupy already posted."""
    head = opener(text, words)
    if not head:
        return False
    return any(opener(previous, words) == head for previous in recent if previous)


def rank_candidates(candidates: Sequence[str], recent: Iterable[str] = ()) -> List[Tuple[str, List[str]]]:
    """Sort drafts by how much slop they carry, cleanest first, keeping the original order within a tie.

    A repeated opener counts as one more tell, so a clean-but-familiar line
    loses to an equally clean one that starts differently.
    """
    previous = [p for p in recent if p]
    scored: List[Tuple[int, int, str, List[str]]] = []
    for position, candidate in enumerate(candidates):
        tells = slop_tells(candidate)
        if repeats_opener(candidate, previous):
            tells = tells + ["repeats a recent opener"]
        scored.append((len(tells), position, candidate, tells))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [(candidate, tells) for _count, _position, candidate, tells in scored]


__all__ = [
    "SLOP_RULES",
    "opener",
    "rank_candidates",
    "repeats_opener",
    "slop_tells",
]
