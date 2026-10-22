#!/usr/bin/env python3
"""Keep private data from this machine out of the repository.

The repository is public. The machine that runs the bot holds what must never
be in it: the Discord archive (members' IDs, names and messages), the profiles
built from it, Soupy's memory of its members, and the real ``.env-stable``.
A test fixture or a comment example written from a real row publishes that
person's data, so fixtures and examples use invented members, and this check
compares what is about to be published with the real data to make sure.

It runs from git hooks (see ``.pre-commit-config.yaml``) and by hand:

    python tools/privacy_check.py              # what `git commit` would record
    python tools/privacy_check.py --all        # every tracked file at HEAD
    python tools/privacy_check.py --push       # what `git push` would publish
    python tools/privacy_check.py --message F  # a commit message

What it looks for, in the files being changed:

  id        a 16 to 20 digit number that is a real Discord user, channel,
            server or message ID
  name      a real member's username or nickname, or a real server or channel
            name, as a whole word
  secret    a value from the real ``.env-stable`` that differs from the template
  text      a quoted string or a sentence of seven or more words found word
            for word in the archive, a profile or Soupy's memory (added lines
            only)
  word      a word of 5+ letters that only a few members' profiles mention
            (three at most), is not an ordinary dictionary word, and is new to
            the repository: the town someone lives in, the game only they
            play (added lines only)

Everything is read from local files that git ignores. On a machine with none
of them (CI, a fresh clone) there is nothing to compare with and the check
passes. A match that is not private goes in ``tools/privacy_allow.txt``; an
ID can't be allowed that way.

What it cannot see: a true fact about a member retold in ordinary words with
an invented name. Only the rule covers that: never write a fixture or an
example from a real row.

Standard library only, so the hook needs no virtualenv.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

ID_RE = re.compile(r"(?<![0-9A-Za-z])\d{16,20}(?!\d)")
WORD_RE = re.compile(r"[A-Za-z]{5,}")
QUOTED_RE = re.compile(r"\"([^\"\n]{24,400})\"|'([^'\n]{24,400})'|`([^`\n]{24,400})`")
SENTENCE_RE = re.compile(r"[^.!?\n]{40,400}[.!?]")
SECRET_KEY_RE = re.compile(r"TOKEN|KEY|PASSWORD|SECRET", re.I)

# Shortest value from .env-stable worth looking for; shorter ones are ports and flags.
MIN_SECRET = 8
MIN_NAME = 3
# A copied line is a sentence. Shorter stretches ("i don't think you should") turn up in
# any large chat archive by chance.
MIN_TEXT = 32
MIN_TEXT_WORDS = 7
# A word mentioned in more profiles than this is common talk, not a fact about someone.
MAX_PROFILES_FOR_A_WORD = 3
# Above this many text candidates, hash the private text once instead of searching it each time.
MANY_CANDIDATES = 150
# Files that are someone else's code or carry no prose.
SKIP_PATHS = ("web/static/vendor/",)
SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".lock", ".woff", ".woff2", ".webmanifest")
MAX_FILE_BYTES = 2_000_000
DICTIONARIES = ("/usr/share/dict/words", "/usr/share/dict/propernames")
# Files under data/ that hold public posts or plain settings, not anything about members.
PUBLIC_DATA = ("bluesky_", "daily_post_", "bot_dashboard", "runtime_flags", "musings_daily_state", "profile_nightly")
PREFIXES = ("anti", "co", "de", "dis", "mis", "non", "out", "over", "pre", "re", "sub", "un", "under")
INFLECTIONS = (
    ("ies", ("y",)),
    ("ied", ("y",)),
    ("ing", ("", "e")),
    ("est", ("", "e")),
    ("ed", ("", "e")),
    ("es", ("", "e")),
    ("er", ("", "e")),
    ("ly", ("",)),
    ("s", ("",)),
    ("d", ("",)),
)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _words(text: str) -> List[str]:
    """Words of 5+ letters. A written-out "\\n" is a line break, not the start of a word."""
    return WORD_RE.findall(re.sub(r"\\[nrtfv]", " ", text))


# --- the private data on this machine -------------------------------------------
@dataclass
class PrivateData:
    """What must not be published, as found on this machine."""

    ids: Dict[str, str] = field(default_factory=dict)  # number -> what it is
    names: Dict[str, str] = field(default_factory=dict)  # lower-case name -> what it is
    secrets: List[Tuple[str, str]] = field(default_factory=list)  # (.env-stable key, value)
    text: str = ""  # archive, profiles and memory, squashed
    # word -> how many people's profiles (and memory folders) mention it
    profile_words: Dict[str, int] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    _shingles: Optional[Set[int]] = None

    def empty(self) -> bool:
        return not (self.ids or self.names or self.secrets or self.text)

    def shingles(self) -> Set[int]:
        """Every run of four words in the private text, hashed. Built on first use."""
        if self._shingles is None:
            words = self.text.split(" ")
            self._shingles = {hash(tuple(words[i : i + 4])) for i in range(len(words) - 3)}
        return self._shingles


def _add_id(data: PrivateData, value, kind: str) -> None:
    text = str(value)
    if re.fullmatch(r"\d{16,20}", text):
        data.ids.setdefault(text, kind)


def _add_name(data: PrivateData, value, kind: str) -> None:
    name = _squash(str(value or ""))
    if len(name) >= MIN_NAME:
        data.names.setdefault(name, kind)


def _json_strings(value) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)


def _load_database(path: str, data: PrivateData, text: List[str], people: Dict[str, List[str]]) -> None:
    match = re.search(r"guild_(\d+)\.db$", path)
    if match:
        _add_id(data, match.group(1), "server ID")
    # Read-only, and never waiting on the bot for long: the bot writes to this file.
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)

    def rows(sql: str) -> list:
        try:
            return con.execute(sql).fetchall()
        except sqlite3.OperationalError as error:
            # An older schema without this table or column has nothing to add.
            # Anything else (a locked or damaged file) must not pass as "no data".
            if "no such table" in str(error) or "no such column" in str(error):
                return []
            raise

    try:
        for guild_id, name in rows("select guild_id, guild_name from guild_metadata"):
            _add_id(data, guild_id, "server ID")
            _add_name(data, name, "server name")
        for channel_id, name in rows("select channel_id, channel_name from channels"):
            _add_id(data, channel_id, "channel ID")
            _add_name(data, name, "channel name")
        for user_id, username, nickname in rows("select distinct user_id, username, nickname from messages"):
            _add_id(data, user_id, "user ID")
            _add_name(data, username, "member's username")
            _add_name(data, nickname, "member's nickname")
        for (message_id,) in rows("select message_id from messages"):
            _add_id(data, message_id, "message ID")
        for (channel_id,) in rows("select distinct channel_id from messages"):
            _add_id(data, channel_id, "channel ID")
        # Web page text (url_summary) is left out: it is public, and full of cookie banners.
        for column in ("message_content", "image_description"):
            text.extend(str(v) for (v,) in rows(f"select {column} from messages where {column} is not null"))
        for table in ("user_profile_summaries", "user_profile_summaries_v1"):
            for user_id, hint, summary, structured in rows(
                f"select user_id, nickname_hint, summary, structured_json from {table}"
            ):
                _add_id(data, user_id, "user ID")
                _add_name(data, hint, "member's nickname")
                about = people.setdefault(f"profile of {user_id}", [])
                about.append(str(summary or ""))
                try:
                    about.extend(_json_strings(json.loads(structured or "null")))
                except ValueError:
                    about.append(str(structured or ""))
        memory = people.setdefault(f"Soupy's memory in {os.path.basename(path)}", [])
        memory.extend(str(v) for (v,) in rows("select chunk_text from self_chunks where chunk_text is not null"))
    finally:
        con.close()


def _read(path: Path) -> str:
    try:
        if path.stat().st_size > 30_000_000:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in _read(path).splitlines():
        match = re.match(r"\s*([A-Za-z0-9_]+)\s*=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip("\"'")
    return values


def load_private_data(root: Path, db_dir: Optional[Path] = None) -> PrivateData:
    """Read the archive, profiles, memory and config under ``root``. Missing pieces are skipped."""
    data = PrivateData()
    text: List[str] = []  # the archive
    # What is known about people: one entry per person's profile and per memory folder.
    people: Dict[str, List[str]] = {}

    db_dir = db_dir or Path(os.environ.get("SOUPY_DB_DIR") or root / "soupy_database" / "databases")
    for db in sorted(glob.glob(str(db_dir / "guild_*.db"))):
        _load_database(db, data, text, people)
        data.sources.append(os.path.relpath(db, root))

    for name in ("media/images/index.jsonl", "user_stats.json"):
        raw = _read(root / name)
        if raw:
            data.sources.append(name)
            for number in ID_RE.findall(raw):
                _add_id(data, number, "ID in " + name)
    for line in _read(root / "media" / "images" / "index.jsonl").splitlines():
        try:
            _add_name(data, json.loads(line).get("username"), "member's username")
        except (ValueError, AttributeError):
            pass

    # Soupy's memory, and anything else kept under data/ about people.
    data_dir = root / "data"
    if data_dir.is_dir():
        for path in sorted(data_dir.rglob("*")):
            # Not .html: a rendered page repeats the JSON beside it and adds markup and script.
            if not path.is_file() or path.suffix not in (".json", ".jsonl", ".md", ".txt", ".bak"):
                continue
            if path.name.startswith(PUBLIC_DATA):
                continue
            raw = _read(path)
            if raw:
                # The files in one folder are copies and views of each other: one entry.
                folder = path.relative_to(data_dir).parts[0]
                people.setdefault(f"data/{folder}", []).append(raw)
                for number in ID_RE.findall(raw):
                    _add_id(data, number, "ID in data/")
        data.sources.append("data/")

    # The maintainer's own prompts: only what differs from the shipped *.default.txt.
    for path in sorted((root / "prompts").glob("*.txt")):
        if path.name.endswith(".default.txt"):
            continue
        shipped_prompt = _squash(_read(path.with_name(path.stem + ".default.txt")))
        own = [part for part in re.split(r"(?<=[.!?])\s+", _read(path)) if _squash(part) not in shipped_prompt]
        if own:
            text.extend(own)
            data.sources.append(f"prompts/{path.name}")

    real, template = parse_env(root / ".env-stable"), parse_env(root / ".env-stable.example")
    shipped = " ".join(template.values())
    for key, value in sorted(real.items()):
        if not value or value == template.get(key):
            continue
        for piece in {value, *re.findall(r"[A-Za-z0-9_\-./:@+=]{%d,}" % MIN_SECRET, value)}:
            if len(piece) < MIN_SECRET or piece in shipped:
                continue
            if re.fullmatch(r"[0-9.]+", piece) and len(piece) < 12:  # a number, not an identifier
                continue
            data.secrets.append((key, piece))
    if real:
        data.sources.append(".env-stable")

    data.text = _squash("\n".join(text + [part for parts in people.values() for part in parts]))
    for parts in people.values():
        for word in {w.lower() for w in _words("\n".join(parts))}:
            data.profile_words[word] = data.profile_words.get(word, 0) + 1
    return data


# --- what is being published ------------------------------------------------------
@dataclass
class Change:
    """One file as it would be published: all of its lines, and which of them are new."""

    path: str
    lines: List[str]
    added: Set[int]  # 1-based numbers of the lines this change adds


def _git(root: Path, *args: str, check: bool = False) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or f"git {' '.join(args)} failed")
    return result.stdout.decode("utf-8", "replace")


def _skipped(path: str) -> bool:
    return path.startswith(SKIP_PATHS) or path.lower().endswith(SKIP_SUFFIXES)


def _blob(root: Path, spec: str) -> Optional[str]:
    result = subprocess.run(["git", "show", spec], cwd=root, capture_output=True)
    if result.returncode != 0 or len(result.stdout) > MAX_FILE_BYTES or b"\0" in result.stdout[:8000]:
        return None
    return result.stdout.decode("utf-8", "replace")


def _added_lines(diff: str) -> Set[int]:
    added: Set[int] = set()
    line_no = 0
    for line in diff.splitlines():
        hunk = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk:
            line_no = int(hunk.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            added.add(line_no)
            line_no += 1
        elif line.startswith(" "):
            line_no += 1
    return added


def staged_changes(root: Path) -> List[Change]:
    names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split("\0")
    changes = []
    for path in filter(None, names):
        if _skipped(path):
            continue
        text = _blob(root, f":{path}")
        if text is not None:
            diff = _git(root, "diff", "--cached", "-U0", "--no-color", "--", path)
            changes.append(Change(path, text.splitlines(), _added_lines(diff)))
    return changes


def range_changes(root: Path, base: str, tip: str) -> List[Change]:
    names = _git(root, "diff", "--name-only", "--diff-filter=ACMR", "-z", base, tip).split("\0")
    changes = []
    for path in filter(None, names):
        if _skipped(path):
            continue
        text = _blob(root, f"{tip}:{path}")
        if text is not None:
            diff = _git(root, "diff", "-U0", "--no-color", base, tip, "--", path)
            changes.append(Change(path, text.splitlines(), _added_lines(diff)))
    return changes


def tree_changes(root: Path, ref: str) -> List[Change]:
    """Every file at ``ref``, with every line treated as new."""
    changes = []
    for path in filter(None, _git(root, "ls-tree", "-r", "--name-only", "-z", ref, check=True).split("\0")):
        if _skipped(path):
            continue
        text = _blob(root, f"{ref}:{path}")
        if text is not None:
            lines = text.splitlines()
            changes.append(Change(path, lines, set(range(1, len(lines) + 1))))
    return changes


def repo_words(root: Path, ref: str) -> Set[str]:
    """Every word of 5+ letters already in the repository at ``ref``."""
    out = _git(root, "grep", "-I", "-h", "-o", "-i", "-E", "[a-z]{5,}", ref, "--", ".")
    return {w.lower() for w in out.split()}


def dictionary_words() -> Set[str]:
    words: Set[str] = set()
    for path in DICTIONARIES:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                words.update(line.strip().lower() for line in f)
        except OSError:
            pass
    return words


def load_allowed(path: Path) -> Set[str]:
    return {_squash(line) for line in _read(path).splitlines() if line.strip() and not line.lstrip().startswith("#")}


# --- the comparison -----------------------------------------------------------------
@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    what: str


class Checker:
    def __init__(
        self,
        private: PrivateData,
        allowed: Set[str],
        *,
        dictionary: Optional[Set[str]] = None,
        known_words: Optional[Set[str]] = None,
    ) -> None:
        self.private = private
        self.allowed = allowed
        self.dictionary = dictionary or set()
        # None switches the "word" check off: with no earlier version of the
        # repository to compare with, every word would count as new.
        self.known_words = known_words
        self.names = self._names_to_look_for()
        # One pattern for every name. "\b" fails on names that start or end
        # with punctuation, so the edges are spelled out.
        alternatives = "|".join(re.escape(name) for name in sorted(self.names, key=len, reverse=True))
        self.name_re = (
            re.compile(r"(?<![A-Za-z0-9_.])(?:" + alternatives + r")(?![A-Za-z0-9_])", re.I) if self.names else None
        )
        self.secrets = [(key, value) for key, value in private.secrets if value.lower() not in allowed]

    def _plain(self, word: str) -> bool:
        """A dictionary word or a plain inflection of one ("jumped", "putting", "parties")."""
        words = self.dictionary
        if word in words:
            return True
        for suffix, endings in INFLECTIONS:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                stem = word[: -len(suffix)]
                if any(stem + ending in words for ending in endings):
                    return True
                if len(stem) > 3 and stem[-1] == stem[-2] and stem[:-1] in words:  # putt-ing
                    return True
        return False

    def everyday(self, word: str) -> bool:
        """An ordinary word: in the dictionary, inflected, prefixed ("reformatted") or two words joined ("countdown")."""
        if self._plain(word):
            return True
        if any(word.startswith(p) and len(word) - len(p) >= 4 and self._plain(word[len(p) :]) for p in PREFIXES):
            return True
        # Halves of four letters or more: the dictionary is full of obscure
        # three-letter words, and "bar" + "rie" must not excuse a place name.
        return any(self._plain(word[:cut]) and self._plain(word[cut:]) for cut in range(4, len(word) - 3))

    def _names_to_look_for(self) -> Dict[str, str]:
        names = {}
        for name, kind in self.private.names.items():
            if name in self.allowed:
                continue
            words = name.split()
            # A name made of one or two everyday words would match ordinary
            # prose. A server's name is looked for regardless.
            if kind != "server name" and self.dictionary and len(words) <= 2 and all(self.everyday(w) for w in words):
                continue
            names[name] = kind
        return names

    def _allowed_text(self, text: str) -> bool:
        return text in self.allowed or any(len(entry) >= 12 and entry in text for entry in self.allowed)

    def _in_private_text(self, squashed: str, many: bool) -> bool:
        if many:
            # The first and last words may be cut short by the quote marks, so
            # test four from the middle before paying for a full search.
            words = squashed.split(" ")
            if len(words) >= 6 and hash(tuple(words[1:5])) not in self.private.shingles():
                return False
        return squashed in self.private.text

    def check(self, change: Change) -> List[Finding]:
        findings: List[Finding] = []
        seen: Set[Tuple[int, str]] = set()

        def report(line_no: int, kind: str, what: str) -> None:
            if (line_no, what) not in seen:
                seen.add((line_no, what))
                findings.append(Finding(change.path, line_no, kind, what))

        whole = "\n".join(change.lines)
        starts = [0]
        for line in change.lines:
            starts.append(starts[-1] + len(line) + 1)

        def line_of(offset: int) -> int:
            low, high = 0, len(starts) - 1
            while low < high:
                mid = (low + high + 1) // 2
                if starts[mid] <= offset:
                    low = mid
                else:
                    high = mid - 1
            return low + 1

        # On every line of a touched file, new or not.
        for match in ID_RE.finditer(whole):
            if match.group() in self.private.ids:
                kind = self.private.ids[match.group()]
                report(line_of(match.start()), "id", f"real Discord {kind} {match.group()}")
        if self.name_re:
            for match in self.name_re.finditer(whole):
                kind = self.names.get(_squash(match.group()), "name")
                report(line_of(match.start()), "name", f'{kind} "{match.group()}"')
        for key, value in self.secrets:
            offset = whole.find(value)
            while offset != -1:
                shown = "its value" if SECRET_KEY_RE.search(key) else f'"{value[:60]}"'
                report(line_of(offset), "secret", f"{key} from the real .env-stable: {shown}")
                offset = whole.find(value, offset + 1)

        # On the lines this change adds.
        added = [(n, change.lines[n - 1]) for n in sorted(change.added) if 0 < n <= len(change.lines)]
        if self.private.text:
            candidates = [(n, candidate) for n, line in added for candidate in self._candidates(line)]
            many = len(candidates) > MANY_CANDIDATES
            for line_no, candidate in candidates:
                squashed = _squash(candidate)
                if (
                    len(squashed) < MIN_TEXT
                    or len(squashed.split(" ")) < MIN_TEXT_WORDS
                    or self._allowed_text(squashed)
                ):
                    continue
                if self._in_private_text(squashed, many):
                    shown = candidate.strip()[:70]
                    report(line_no, "text", f'copied from the archive, a profile or Soupy\'s memory: "{shown}"')
        if self.known_words is not None and self.dictionary:
            for line_no, line in added:
                for word in _words(line):
                    low = word.lower()
                    if (
                        0 < self.private.profile_words.get(low, 0) <= MAX_PROFILES_FOR_A_WORD
                        and low not in self.known_words
                        and low not in self.allowed
                        and not self.everyday(low)
                    ):
                        report(
                            line_no, "word", f'"{word}" comes up in a member\'s profile and nowhere in the repository'
                        )
        findings.sort(key=lambda finding: finding.line)
        return findings

    @staticmethod
    def _candidates(line: str) -> Iterable[str]:
        """Stretches of a line that could be a quotation, without the punctuation around them."""
        found = [next(group for group in match.groups() if group) for match in QUOTED_RE.finditer(line)]
        # Prose quotes people without quote marks: whole sentences, and what
        # follows a colon or sits between dashes ("Soupy once said: ...").
        if len(line.split()) >= 6:
            for sentence in SENTENCE_RE.findall(line):
                found.append(sentence)
                found.extend(re.split(r":\s+|;\s+|\s[—–-]\s", sentence))
        for candidate in found:
            yield candidate.strip(" \t.,;:!?\"'`()[]{}*_>#-—–")


# --- command line -------------------------------------------------------------------
def _default_base(root: Path, tip: str) -> Optional[str]:
    """Where a brand-new branch left the published history."""
    for ref in ("origin/HEAD", "origin/main", "origin/master"):
        base = _git(root, "merge-base", ref, tip).strip()
        if base:
            return base
    return None


def _push_range(root: Path) -> Tuple[Optional[str], str]:
    tip = os.environ.get("PRE_COMMIT_TO_REF") or "HEAD"
    base = os.environ.get("PRE_COMMIT_FROM_REF") or ""
    if not base or set(base) == {"0"}:
        base = _default_base(root, tip) or ""
    return (base or None), tip


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Keep private data from this machine out of the repository.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="what `git commit` would record (the default)")
    mode.add_argument("--all", nargs="?", const="HEAD", metavar="REF", help="every tracked file at REF (default HEAD)")
    mode.add_argument("--push", action="store_true", help="what `git push` would publish (for the pre-push hook)")
    mode.add_argument("--message", metavar="FILE", help="a commit message file")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--db-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("-q", "--quiet", action="store_true", help="say nothing when there is nothing to report")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        private = load_private_data(root, args.db_dir)
    except sqlite3.Error as error:
        # Refuse rather than wave through a change that could not be compared.
        print(f"privacy check: could not read the archive ({error}). Nothing was checked; try again.", file=sys.stderr)
        return 2
    if private.empty():
        if not args.quiet:
            print("privacy check: no archive or .env-stable on this machine, so nothing to compare with.")
        return 0

    allowed = load_allowed(args.allow or root / "tools" / "privacy_allow.txt")
    known: Optional[Set[str]] = None
    try:
        if args.message:
            lines = [line for line in _read(Path(args.message)).splitlines() if not line.startswith("#")]
            changes = [Change("the commit message", lines, set(range(1, len(lines) + 1)))]
            what = "the commit message"
        elif args.all:
            changes, what = tree_changes(root, args.all), f"every file at {args.all}"
        elif args.push:
            base, tip = _push_range(root)
            if base:
                changes, known, what = range_changes(root, base, tip), repo_words(root, base), "this push"
            else:
                changes, what = tree_changes(root, tip), "this push"
        else:
            changes, what = staged_changes(root), "this commit"
            known = repo_words(root, "HEAD")
    except RuntimeError as error:
        print(f"privacy check: {error}", file=sys.stderr)
        return 2

    checker = Checker(private, allowed, dictionary=dictionary_words(), known_words=known)
    findings = [finding for change in changes for finding in checker.check(change)]
    if not findings:
        if not args.quiet:
            print(
                f"privacy check: nothing private in {what} ({len(changes)} file(s) compared with this machine's data)."
            )
        return 0

    print(f"privacy check: {len(findings)} problem(s) in {what}\n")
    width = max(len(f"{f.path}:{f.line}") for f in findings)
    for f in findings:
        print(f"  {f.path + ':' + str(f.line):<{width}}  {f.what}")
    print(
        "\nThe repository is public. Use invented names, IDs, places and text in fixtures, examples and docs.\n"
        "If a match is not private (a public name, a product, a stock phrase), add it on its own line to\n"
        "tools/privacy_allow.txt. A real ID cannot be allowed."
    )
    return 1


if __name__ == "__main__":
    sys.exit(run())
