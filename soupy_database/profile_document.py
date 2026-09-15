"""
The member-profile document (version 2) and everything that reads or edits it.

Pure functions only — no SQLite, no HTTP — so the rules that decide what a
profile contains are unit-testable without LM Studio.

Shape::

    {
      "version": 2,
      "overview": "4-8 sentence portrait",
      "communication_style": "2-4 sentences",
      "sections": {
        "personality_traits": [
          {"id": "pt3", "text": "...", "date": "2024-03-02", "last_seen": "2025-06-01"},
          ...
        ],
        "relationships_with_others": [{"id": "rel1", "text": "...", "date": "...", "user_id": 123, "name": "elektron"}],
        "channels": [{"id": "chan1", "text": "...", "date": "...", "name": "#general"}],
        ...
      },
      "next_id": 57,
      "coverage": {"first": "2021-03-01", "last": "2026-09-14", "messages": 4178, "passes": 14}
    }

Every item's ``date`` is the date of the chat message it came from, not the
date the profile was built — a rebuild over years of history keeps real dates.
``last_seen`` only appears once a later message confirms the item.

Why edits instead of rewrites: the LLM returns a short list of add / update /
remove operations and :func:`apply_edits` applies them. Items keep their ids
and dates, section caps are enforced here rather than trusted to the model, and
the model's output stays small however large the profile grows. The old
builder had the model re-emit the whole profile on every merge, which silently
dropped items and could not fit a large profile into the context window.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

PROFILE_VERSION = 2

ITEM_TEXT_MAX = 300
OVERVIEW_MAX = 1500
STYLE_MAX = 800
NAME_MAX = 80


@dataclass(frozen=True)
class Section:
    key: str
    label: str
    prefix: str
    cap: int
    guide: str
    # Fictional example for the prompt. ``markers`` are its invented proper
    # nouns: an added item containing one that never appears in the source
    # messages is a copy of the example, not a fact (see apply_edits).
    example: str = ""
    markers: Tuple[str, ...] = ()


SECTIONS: Tuple[Section, ...] = (
    Section(
        "personality_traits",
        "Personality",
        "pt",
        12,
        "Lasting traits and temperament, each with the behavior that shows it.",
        "patient walking people through PC problems, but goes quiet when someone ignores the advice",
    ),
    Section(
        "current_situation",
        "Lately",
        "now",
        10,
        "What is going on in their life right now: ongoing projects, upcoming plans, current struggles, "
        "things they just started. Remove or update items once they are over.",
        "rebuilding the engine in the '71 Bramblewood truck, waiting on parts",
        ("bramblewood",),
    ),
    Section(
        "with_soupy",
        "With Soupy",
        "soup",
        8,
        "How they treat and use Soupy, the Discord bot: teasing, affection, arguing with it, testing it, "
        "what they ask it for, running bits with it.",
        "tries to get Soupy to admit it is sentient; calls it 'Brothbot' when it gets something wrong",
        ("brothbot",),
    ),
    Section(
        "family_and_household",
        "Family & home",
        "fam",
        18,
        "Partner, kids, parents, siblings, pets (names, species, ages), who they live with, and notable "
        "facts about those people.",
        "has a greyhound named Pickles, adopted from a racing rescue",
        ("pickles",),
    ),
    Section(
        "life_events",
        "Life events",
        "life",
        20,
        "Things that happened to them: moves, job changes, health events, losses, big trips, milestones. "
        "Use past tense.",
        "moved from Duluth to Tucson for the dry air",
        ("duluth",),
    ),
    Section(
        "work_and_education",
        "Work & school",
        "work",
        8,
        "Jobs, employer type, role, industry, schooling, skills, career changes.",
    ),
    Section(
        "location",
        "Location",
        "loc",
        3,
        "Where they live or have lived (region or city) and time zone hints.",
    ),
    Section(
        "possessions_and_setup",
        "Owns",
        "own",
        12,
        "Vehicles, computers, gear, home setup, collections — with specific models and brands.",
    ),
    Section(
        "opinions_and_stances",
        "Opinions",
        "op",
        35,
        "Their takes, likes, dislikes and pet peeves — WHAT they think and WHY, not just the topic.",
        "thinks Kestrel Mart's self-checkout is unpaid labor and refuses to use it",
        ("kestrel",),
    ),
    Section(
        "politics",
        "Politics",
        "pol",
        8,
        "Political views and specific policy positions only. Workplace or tech policy goes in opinions.",
    ),
    Section(
        "hobbies",
        "Hobbies",
        "hob",
        12,
        "Things they do: sports, crafts, making, outdoors, fitness, side projects — with specifics.",
    ),
    Section(
        "media_entertainment",
        "Media",
        "med",
        20,
        "Games, shows, films, music, books, podcasts, creators — the title plus their take on it.",
        "loved Starfall Harbor season 1, quit season 2 over the time-skip",
        ("starfall",),
    ),
    Section(
        "relationships_with_others",
        "People here",
        "rel",
        15,
        "Specific dynamics with other members: friendship, rivalry, in-jokes, shared history, real-life "
        "connections. Set user_id from the member directory (0 if unsure) and name.",
    ),
    Section(
        "running_jokes_and_nicknames",
        "Running jokes",
        "joke",
        10,
        "Nicknames they have or use, recurring bits, memes and catchphrases the server associates with them.",
    ),
    Section(
        "signature_phrases",
        "Phrases",
        "say",
        8,
        "Short phrases or verbal tics they actually use, copied exactly.",
    ),
    Section(
        "discussion_topics",
        "Talks about",
        "top",
        10,
        "Recurring themes they bring up here, each with their usual angle.",
    ),
    Section(
        "channels",
        "Channels",
        "chan",
        6,
        "Channels they are active in and what they do there. Set name to the channel.",
    ),
    Section(
        "uncertain",
        "Unsure",
        "unk",
        6,
        "Low-confidence guesses or open questions worth checking later.",
    ),
)

SECTION_BY_KEY: Dict[str, Section] = {s.key: s for s in SECTIONS}
SECTION_KEYS: Tuple[str, ...] = tuple(s.key for s in SECTIONS)
# Examples the prompt's general rules use, guarded the same way as section examples.
RULE_EXAMPLES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("plays Hollowmere co-op with their brother every Sunday", ("hollowmere",)),
)
EXAMPLE_MARKERS: Tuple[str, ...] = tuple(m for s in SECTIONS for m in s.markers) + tuple(
    m for _text, markers in RULE_EXAMPLES for m in markers
)

# "Lately" items older than this (by last confirmation) move to life events.
LATELY_DEFAULT_DAYS = 90

# Sections never shown in chat: guesses would read as facts.
_CHAT_EXCLUDED = frozenset({"uncertain"})

# Round-robin order for filling the chat budget once query-relevant items are in.
_CHAT_FILL_ORDER: Tuple[str, ...] = (
    "personality_traits",
    "current_situation",
    "with_soupy",
    "family_and_household",
    "work_and_education",
    "location",
    "hobbies",
    "opinions_and_stances",
    "relationships_with_others",
    "running_jokes_and_nicknames",
    "life_events",
    "media_entertainment",
    "possessions_and_setup",
    "politics",
    "discussion_topics",
    "signature_phrases",
    "channels",
)

# Query words that point at a section; each hit boosts every item in it.
_SECTION_QUERY_WORDS: Dict[str, Tuple[str, ...]] = {
    "personality_traits": ("personality", "person", "like", "vibe", "character", "psychological", "psych", "kind"),
    "current_situation": ("lately", "recently", "now", "doing", "going", "current", "currently", "up", "plan", "plans"),
    "with_soupy": ("soupy", "bot", "you", "treat", "me"),
    "family_and_household": (
        "family",
        "kid",
        "kids",
        "child",
        "children",
        "wife",
        "husband",
        "partner",
        "married",
        "mom",
        "dad",
        "mother",
        "father",
        "brother",
        "sister",
        "son",
        "daughter",
        "pet",
        "pets",
        "cat",
        "dog",
        "house",
        "home",
    ),
    "life_events": ("happened", "move", "moved", "event", "life", "history", "died", "surgery", "trip", "born"),
    "work_and_education": (
        "work",
        "job",
        "career",
        "school",
        "study",
        "degree",
        "college",
        "university",
        "major",
        "company",
        "boss",
    ),
    "location": ("live", "lives", "where", "location", "city", "state", "country", "from", "timezone", "based"),
    "possessions_and_setup": ("own", "owns", "car", "truck", "drive", "setup", "pc", "computer", "phone", "bought"),
    "opinions_and_stances": (
        "think",
        "thinks",
        "opinion",
        "believe",
        "feel",
        "view",
        "hate",
        "love",
        "like",
        "dislike",
        "why",
        "stance",
    ),
    "politics": (
        "politics",
        "political",
        "vote",
        "liberal",
        "conservative",
        "left",
        "right",
        "party",
        "democrat",
        "republican",
        "trump",
        "election",
    ),
    "hobbies": ("hobby", "hobbies", "sport", "sports", "craft", "outdoors", "fitness", "cook", "build", "project"),
    "media_entertainment": (
        "game",
        "games",
        "gaming",
        "play",
        "watch",
        "movie",
        "film",
        "show",
        "anime",
        "music",
        "book",
        "books",
        "read",
        "series",
        "tv",
        "podcast",
        "band",
    ),
    "relationships_with_others": ("friend", "friends", "relationship", "together", "get", "along", "enemy"),
    "running_jokes_and_nicknames": ("joke", "jokes", "nickname", "meme", "bit", "called", "name"),
    "signature_phrases": ("say", "says", "phrase", "talk", "talks"),
    "discussion_topics": ("talk", "talks", "discuss", "topic", "topics", "bring", "rant"),
    "channels": ("channel", "channels"),
}

# Queries asking for the whole person get a balanced sample across sections.
_BROAD_QUERY_WORDS = frozenset(
    {"about", "profile", "personality", "psychological", "psych", "describe", "summary", "summarize", "who", "vibe"}
)
_STYLE_QUERY_WORDS = frozenset({"style", "tone", "talk", "talks", "humor", "funny", "sarcastic", "write", "writes"})

_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with from by is are was were be been it its this that these those "
    "what which who whom how when where why do does did has have had not no yes they them their he she his her "
    "you your i me my we our us about into over than then so just very really can could would should will".split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_TRAILING_DATE_RE = re.compile(r"\s*[\(\[]\s*(?:\d{4}-\d{2}-\d{2}|[A-Z][a-z]{2} \d{4})\s*[\)\]]\s*$")
# The prompt shows item metadata as a trailing {...}; models sometimes echo it back into the text.
_TRAILING_META_RE = re.compile(r"\s*\{(?:name|channel|user_id)=[^}]*\}\s*$")
# ...or echo an older "user 123 (name):" prefix, or refer to members by raw Discord id.
_MEMBER_PREFIX_RE = re.compile(r"^(?:with\s+)?user(?:_id)?\s*[:=]?\s*\d+\s*(?:\([^)]*\))?\s*:\s*", re.IGNORECASE)
_RAW_MEMBER_ID_RE = re.compile(r"(?:\buser(?:_id)?\s*[:=]?\s*)?\b(\d{15,20})\b", re.IGNORECASE)
_ID_RE = re.compile(r"^([a-z]+)(\d+)$")


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def new_document() -> Dict[str, Any]:
    return {
        "version": PROFILE_VERSION,
        "overview": "",
        "communication_style": "",
        "sections": {k: [] for k in SECTION_KEYS},
        "next_id": 1,
        "coverage": {"first": None, "last": None, "messages": 0, "passes": 0},
    }


def is_v2(doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("version") == PROFILE_VERSION and isinstance(doc.get("sections"), dict)


def parse_date(raw: Any) -> Optional[date]:
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def format_month(d: Optional[date]) -> str:
    return d.strftime("%b %Y") if d else ""


def normalize_document(doc: Any) -> Dict[str, Any]:
    """Return a well-formed v2 document, repairing missing keys and dropping malformed items."""
    out = new_document()
    if not is_v2(doc):
        return out
    out["overview"] = str(doc.get("overview") or "")[:OVERVIEW_MAX]
    out["communication_style"] = str(doc.get("communication_style") or "")[:STYLE_MAX]
    max_seen = 0
    for key in SECTION_KEYS:
        clean: List[Dict[str, Any]] = []
        for raw in (doc.get("sections") or {}).get(key) or []:
            if not isinstance(raw, dict):
                continue
            text = _clean_text(raw.get("text"))
            iid = str(raw.get("id") or "")
            if not text or not iid:
                continue
            parsed = parse_date(raw.get("date"))
            item: Dict[str, Any] = {"id": iid, "text": text, "date": parsed.isoformat() if parsed else None}
            if parse_date(raw.get("last_seen")):
                item["last_seen"] = raw["last_seen"]
            if "user_id" in raw:
                item["user_id"] = _as_int(raw.get("user_id"))
            if raw.get("name"):
                item["name"] = str(raw["name"])[:NAME_MAX]
            clean.append(item)
            m = _ID_RE.match(iid)
            if m:
                max_seen = max(max_seen, int(m.group(2)))
        out["sections"][key] = clean
    out["next_id"] = max(_as_int(doc.get("next_id")) or 1, max_seen + 1)
    cov = doc.get("coverage") if isinstance(doc.get("coverage"), dict) else {}
    out["coverage"] = {
        "first": cov.get("first"),
        "last": cov.get("last"),
        "messages": _as_int(cov.get("messages")) or 0,
        "passes": _as_int(cov.get("passes")) or 0,
    }
    return out


def section_items(doc: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return list(((doc or {}).get("sections") or {}).get(key) or [])


def section_texts(doc: Dict[str, Any], key: str) -> List[str]:
    """Plain item texts for a section. Also reads pre-v2 profiles, where sections were top-level string lists."""
    if is_v2(doc):
        return [it["text"] for it in section_items(doc, key) if it.get("text")]
    raw = (doc or {}).get(key)
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if x and str(x).strip()]
    return []


def item_count(doc: Dict[str, Any]) -> int:
    return sum(len(section_items(doc, k)) for k in SECTION_KEYS)


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _clean_text(raw: Any, limit: int = ITEM_TEXT_MAX) -> str:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    text = _TRAILING_META_RE.sub("", text).strip()
    text = _TRAILING_DATE_RE.sub("", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _stem(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> Set[str]:
    return {_stem(w) for w in _WORD_RE.findall((text or "").lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _item_recency(item: Dict[str, Any]) -> Optional[date]:
    return parse_date(item.get("last_seen")) or parse_date(item.get("date"))


# ---------------------------------------------------------------------------
# Applying LLM edits
# ---------------------------------------------------------------------------


@dataclass
class EditStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    merged_duplicates: int = 0
    dates_fixed: int = 0
    dropped_example_copies: int = 0
    dropped_ungrounded: int = 0
    dropped_invalid: int = 0
    moved_lately: int = 0
    dropped_over_cap: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"+{self.added}", f"~{self.updated}", f"-{self.removed}"]
        for label, n in (
            ("dupes merged", self.merged_duplicates),
            ("dates fixed", self.dates_fixed),
            ("example copies dropped", self.dropped_example_copies),
            ("ungrounded dropped", self.dropped_ungrounded),
            ("invalid dropped", self.dropped_invalid),
            ("lately→life events", self.moved_lately),
            ("over cap dropped", self.dropped_over_cap),
        ):
            if n:
                parts.append(f"{label}={n}")
        return " ".join(parts)


def _clamp_date(raw: Any, window: Tuple[date, date], stats: EditStats) -> str:
    """Accept a date inside the batch's message range (±1 day); otherwise use the batch's last date."""
    lo, hi = window
    d = parse_date(raw)
    if d is None or d < lo - timedelta(days=1) or d > hi + timedelta(days=1):
        stats.dates_fixed += 1
        d = hi
    return d.isoformat()


def _find_item(doc: Dict[str, Any], iid: str) -> Optional[Tuple[str, int]]:
    for key in SECTION_KEYS:
        for idx, it in enumerate(doc["sections"][key]):
            if it.get("id") == iid:
                return key, idx
    return None


def _duplicate_of(items: Sequence[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
    new_tokens = _tokens(text)
    if not new_tokens:
        return None
    norm = " ".join(sorted(new_tokens))
    for it in items:
        old_tokens = _tokens(it.get("text") or "")
        if not old_tokens:
            continue
        if " ".join(sorted(old_tokens)) == norm:
            return it
        overlap = len(new_tokens & old_tokens) / len(new_tokens | old_tokens)
        if overlap >= 0.75:
            return it
    return None


def _bump_last_seen(item: Dict[str, Any], seen: str) -> None:
    current = _item_recency(item)
    new = parse_date(seen)
    if new and (current is None or new > current) and new != parse_date(item.get("date")):
        item["last_seen"] = new.isoformat()


def _new_id(doc: Dict[str, Any], key: str) -> str:
    n = int(doc.get("next_id") or 1)
    doc["next_id"] = n + 1
    return f"{SECTION_BY_KEY[key].prefix}{n}"


_EXAMPLE_TOKENS: Tuple[Set[str], ...] = tuple(_tokens(s.example) for s in SECTIONS if s.example) + tuple(
    _tokens(text) for text, _markers in RULE_EXAMPLES
)


def _copies_example(text: str, source_text_lower: str) -> bool:
    """True when an item is the prompt's fictional example rather than something the member said."""
    words = set(_WORD_RE.findall(text.lower()))
    if any(m in words and m not in source_text_lower for m in EXAMPLE_MARKERS):
        return True
    tokens = _tokens(text)
    for example_tokens in _EXAMPLE_TOKENS:
        if tokens and len(tokens & example_tokens) / len(tokens | example_tokens) >= 0.75:
            return True
    return False


def _resolve_member(raw: Mapping[str, Any], text: str, directory: Mapping[int, str]) -> Tuple[int, str, str]:
    """(user_id, name, cleaned text) for a relationship item.

    The model often gets the id wrong, leaves the name out, or writes raw
    Discord ids into the text. Resolve against the member directory: the given
    id if it's known, else a known id or name mentioned in the item. Raw ids in
    the text become names, since "user 1098..." is useless in chat.
    """
    text = _MEMBER_PREFIX_RE.sub("", text).strip()
    name = _clean_text(raw.get("name"), NAME_MAX)
    uid = _as_int(raw.get("user_id")) or 0
    if uid not in directory:
        uid = 0
        for m in _RAW_MEMBER_ID_RE.finditer(f"{text} {name}"):
            if int(m.group(1)) in directory:
                uid = int(m.group(1))
                break
    if not uid and name:
        uid = next((k for k, label in directory.items() if label.lower() == name.lower()), 0)
    if uid and (not name or name.isdigit() or name.strip("?() ") == ""):
        name = directory[uid]

    def _named(m: "re.Match[str]") -> str:
        known = directory.get(int(m.group(1)))
        return known if known else m.group(0)

    text = _RAW_MEMBER_ID_RE.sub(_named, text)
    text = re.sub(r"\s*\(\?\)", "", text).strip()
    if name:
        # "knows ohsmitt (ohsmitt)" → "knows ohsmitt"
        text = re.sub(rf"({re.escape(name)})\s*\({re.escape(name)}\)", r"\1", text, flags=re.IGNORECASE)
    return uid, name, text


def _labelled(item: Mapping[str, Any]) -> str:
    """Relationship/channel item text with its name in front, unless the text already says it."""
    text, name = item.get("text") or "", item.get("name") or ""
    if not name or name.lower() in text.lower():
        return text
    return f"{name}: {text}"


# with_soupy needs the member to have actually addressed the bot in this batch;
# asked to fill empty sections, a small model will otherwise invent it.
_SOUPY_MENTION = "soup"


def apply_edits(
    doc: Dict[str, Any],
    edits: Dict[str, Any],
    *,
    window: Tuple[date, date],
    source_text: str,
    directory: Optional[Mapping[int, str]] = None,
    lately_days: int = LATELY_DEFAULT_DAYS,
) -> Tuple[Dict[str, Any], EditStats]:
    """Apply one pass's edit list. Returns a new document; ``doc`` is left untouched.

    ``window`` is the (first, last) message date of the batch the edits came
    from, used to validate item dates. ``source_text`` is that batch's raw text,
    used to reject copies of the prompt's fictional examples. ``directory``
    maps known member ids to names, for relationship items.
    """
    out = normalize_document(copy.deepcopy(doc))
    stats = EditStats()
    edits = edits if isinstance(edits, dict) else {}
    directory = {int(k): str(v) for k, v in (directory or {}).items()}
    source_lower = (source_text or "").lower()

    for rem in edits.get("remove") or []:
        iid = rem.get("id") if isinstance(rem, dict) else rem
        loc = _find_item(out, str(iid or ""))
        if loc:
            key, idx = loc
            del out["sections"][key][idx]
            stats.removed += 1

    for upd in edits.get("update") or []:
        if not isinstance(upd, dict):
            stats.dropped_invalid += 1
            continue
        loc = _find_item(out, str(upd.get("id") or ""))
        if not loc:
            stats.dropped_invalid += 1
            continue
        key, idx = loc
        item = out["sections"][key][idx]
        text = _clean_text(upd.get("text"))
        if text and key == "relationships_with_others":
            uid, name, text = _resolve_member({"user_id": item.get("user_id"), "name": item.get("name")}, text, directory)
            item["user_id"], item["name"] = uid, name or item.get("name") or ""
        grounded = key != "with_soupy" or _SOUPY_MENTION in source_lower
        if text and text != item["text"] and grounded and not _copies_example(text, source_lower):
            item["text"] = text
        _bump_last_seen(item, _clamp_date(upd.get("date"), window, stats))
        stats.updated += 1

    for add in edits.get("add") or []:
        if not isinstance(add, dict):
            stats.dropped_invalid += 1
            continue
        key = str(add.get("section") or "")
        text = _clean_text(add.get("text"))
        member: Optional[Tuple[int, str]] = None
        if key == "relationships_with_others" and text:
            uid, name, text = _resolve_member(add, text, directory)
            member = (uid, name)
        if key not in SECTION_BY_KEY or not text:
            stats.dropped_invalid += 1
            continue
        if _copies_example(text, source_lower):
            stats.dropped_example_copies += 1
            stats.notes.append(f"dropped example copy: {text[:80]}")
            continue
        if key == "with_soupy" and _SOUPY_MENTION not in source_lower:
            stats.dropped_ungrounded += 1
            stats.notes.append(f"dropped with_soupy item with no Soupy mention in the batch: {text[:80]}")
            continue
        when = _clamp_date(add.get("date"), window, stats)
        existing = _duplicate_of(out["sections"][key], text)
        if existing is not None:
            if len(text) > len(existing["text"]) * 1.2:
                existing["text"] = text
            _bump_last_seen(existing, when)
            stats.merged_duplicates += 1
            continue
        item: Dict[str, Any] = {"id": _new_id(out, key), "text": text, "date": when}
        if member is not None:
            item["user_id"] = member[0]
            if member[1]:
                item["name"] = member[1]
        elif key == "channels" and add.get("name"):
            item["name"] = _clean_text(add.get("name"), NAME_MAX)
        out["sections"][key].append(item)
        stats.added += 1

    overview = _clean_text(edits.get("overview"), OVERVIEW_MAX)
    if overview:
        out["overview"] = overview
    style = _clean_text(edits.get("communication_style"), STYLE_MAX)
    if style:
        out["communication_style"] = style

    stats.moved_lately = age_out_lately(out, window[1], lately_days)
    stats.dropped_over_cap = enforce_caps(out)
    return out, stats


def age_out_lately(doc: Dict[str, Any], as_of: date, days: int = LATELY_DEFAULT_DAYS) -> int:
    """Move "lately" items not confirmed within ``days`` of ``as_of`` into life events.

    ``as_of`` is the newest message date processed, not the wall clock, so a
    rebuild walking through old history ages items relative to that history.
    """
    if days <= 0:
        return 0
    cutoff = as_of - timedelta(days=days)
    keep: List[Dict[str, Any]] = []
    moved = 0
    for it in doc["sections"]["current_situation"]:
        seen = _item_recency(it)
        if seen is not None and seen < cutoff:
            if _duplicate_of(doc["sections"]["life_events"], it["text"]) is None:
                moved_item = {"id": _new_id(doc, "life_events"), "text": it["text"], "date": it.get("date")}
                if it.get("last_seen"):
                    moved_item["last_seen"] = it["last_seen"]
                doc["sections"]["life_events"].append(moved_item)
            moved += 1
        else:
            keep.append(it)
    doc["sections"]["current_situation"] = keep
    return moved


def enforce_caps(doc: Dict[str, Any]) -> int:
    """Trim each section to its cap, dropping the items confirmed least recently."""
    dropped = 0
    for sec in SECTIONS:
        items = doc["sections"][sec.key]
        if len(items) <= sec.cap:
            continue
        ranked = sorted(
            range(len(items)),
            key=lambda i: (_item_recency(items[i]) or date.min, i),
        )
        drop = set(ranked[: len(items) - sec.cap])
        doc["sections"][sec.key] = [it for i, it in enumerate(items) if i not in drop]
        dropped += len(drop)
    return dropped


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_for_prompt(doc: Dict[str, Any]) -> str:
    """The current profile as compact id-tagged lines for the build prompt."""
    lines: List[str] = []
    if doc.get("overview"):
        lines.append(f"OVERVIEW: {doc['overview']}")
    if doc.get("communication_style"):
        lines.append(f"COMMUNICATION STYLE: {doc['communication_style']}")
    for sec in SECTIONS:
        items = section_items(doc, sec.key)
        if not items:
            continue
        lines.append(f"[{sec.key}] ({len(items)}/{sec.cap})")
        for it in items:
            when = it.get("date") or "?"
            if it.get("last_seen"):
                when = f"{when}..{it['last_seen']}"
            meta = ""
            if sec.key == "relationships_with_others":
                meta = f" {{name={it.get('name') or '?'}, user_id={it.get('user_id') or 0}}}"
            elif sec.key == "channels" and it.get("name"):
                meta = f" {{channel={it['name']}}}"
            lines.append(f"{it['id']} | {when} | {it['text']}{meta}")
    return "\n".join(lines) if lines else "(empty — nothing is known about this member yet)"


def render_summary(doc: Dict[str, Any], max_chars: int = 20000) -> str:
    """Readable plain-text profile with full dates (the ``summary`` column; shown in the dashboard)."""
    parts: List[str] = []
    if doc.get("overview"):
        parts.append("Overview:\n" + doc["overview"])
    if doc.get("communication_style"):
        parts.append("Communication style:\n" + doc["communication_style"])
    for sec in SECTIONS:
        items = section_items(doc, sec.key)
        if not items:
            continue
        rows = []
        for it in items:
            label = _labelled(it) if sec.key in ("relationships_with_others", "channels") else it["text"]
            when = it.get("date") or "undated"
            if it.get("last_seen"):
                when = f"{when}, last seen {it['last_seen']}"
            rows.append(f"• {label} ({when})")
        parts.append(f"{sec.label}:\n" + "\n".join(rows))
    out = "\n\n".join(parts).strip() or "(empty profile)"
    if len(out) > max_chars:
        out = out[: max_chars - 1] + "…"
    return out


@dataclass
class _Candidate:
    section: Section
    item: Dict[str, Any]
    relevance: float
    recency: date


def _cut_sentences(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end > limit // 2:
        return cut[: end + 1]
    return cut.rstrip() + "…"


def render_for_chat(
    doc: Dict[str, Any],
    query_tokens: Sequence[str],
    max_chars: int,
    *,
    today: date,
    lately_days: int = LATELY_DEFAULT_DAYS,
) -> str:
    """Pick the profile items most useful for this message and render them within ``max_chars``.

    Items matching the query come first, then a round-robin across sections
    (newest first) fills what's left. Whole items only — nothing is cut
    mid-sentence. Each item shows the month it came up in chat.
    """
    if not is_v2(doc):
        return ""
    qset = {_stem(t.lower()) for t in query_tokens if t and len(t) >= 2}
    broad = bool(qset & {_stem(w) for w in _BROAD_QUERY_WORDS})
    wants_style = broad or bool(qset & {_stem(w) for w in _STYLE_QUERY_WORDS})

    section_boost = {key: len(qset & {_stem(w) for w in words}) for key, words in _SECTION_QUERY_WORDS.items()}

    pool: List[_Candidate] = []
    for sec in SECTIONS:
        if sec.key in _CHAT_EXCLUDED:
            continue
        for it in section_items(doc, sec.key):
            seen = _item_recency(it) or date.min
            if sec.key == "current_situation" and lately_days > 0 and seen < today - timedelta(days=lately_days):
                continue
            overlap = len(qset & (_tokens(it.get("text") or "") | _tokens(it.get("name") or "")))
            relevance = overlap * 2 + section_boost.get(sec.key, 0)
            pool.append(_Candidate(sec, it, float(relevance), seen))

    ordered: List[_Candidate] = []
    seen_ids: Set[str] = set()
    if not broad:
        for c in sorted((c for c in pool if c.relevance > 0), key=lambda c: (-c.relevance, -c.recency.toordinal())):
            ordered.append(c)
            seen_ids.add(c.item["id"])
    by_section: Dict[str, List[_Candidate]] = {}
    for c in pool:
        if c.item["id"] not in seen_ids:
            by_section.setdefault(c.section.key, []).append(c)
    for lst in by_section.values():
        lst.sort(key=lambda c: -c.recency.toordinal())
    while any(by_section.values()):
        for key in _CHAT_FILL_ORDER:
            lst = by_section.get(key)
            if lst:
                ordered.append(lst.pop(0))

    head: List[str] = []
    overview = (doc.get("overview") or "").strip()
    if overview:
        head.append("Overview: " + _cut_sentences(overview, max(200, max_chars // 4)))
    style = (doc.get("communication_style") or "").strip()
    if style and wants_style:
        head.append("Style: " + _cut_sentences(style, 400))

    def _render(chosen: List[_Candidate], tail_style: bool) -> str:
        grouped: Dict[str, List[str]] = {}
        for c in chosen:
            label = _labelled(c.item) if c.section.key == "relationships_with_others" else c.item["text"]
            month = format_month(parse_date(c.item.get("date")))
            grouped.setdefault(c.section.key, []).append(f"{label} ({month})" if month else label)
        lines = list(head)
        for sec in SECTIONS:
            if sec.key in grouped:
                lines.append(f"{sec.label}: " + "; ".join(grouped[sec.key]))
        if tail_style:
            lines.append("Style: " + _cut_sentences(style, 400))
        return "\n".join(lines)

    chosen: List[_Candidate] = []
    for c in ordered:
        chosen.append(c)
        if len(_render(chosen, False)) > max_chars:
            chosen.pop()
    text = _render(chosen, False)
    if style and not wants_style and len(_render(chosen, True)) <= max_chars:
        text = _render(chosen, True)
    if len(text) > max_chars:
        # Only reachable when the overview alone overflows a tiny budget.
        text = text[: max_chars - 1] + "…"
    return text
