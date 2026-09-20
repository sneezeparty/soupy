"""
LLM side of the profile builders: the prompts, the edit-list JSON schema,
the context budget, and the HTTP call to LM Studio.

Cross-module:

* :mod:`soupy_database.user_profiles` drives the passes and applies the
  returned edits with :mod:`soupy_database.profile_document`.
* :mod:`soupy_database.self_profile` uses the same loop, with
  :func:`build_self_system_prompt` / :func:`build_self_user_prompt` and the
  ``SELF`` document kind, for Soupy's memory of itself.

Context budget — the part that must never be wrong:

* The window comes from LM Studio itself (``/api/v0/models`` →
  ``loaded_context_length`` for ``LOCAL_CHAT``), capped by
  ``USER_PROFILE_LLM_N_CTX`` when that is set. Hitting the loaded window's
  ceiling takes LM Studio down, so a guess is not good enough.
* Prompt + ``max_tokens`` stays under ``USER_PROFILE_CONTEXT_SAFETY`` (85%) of
  that window.
* Prompt size is estimated from characters. The chars-per-token ratio starts
  conservative and is recalibrated from each response's
  ``usage.prompt_tokens``, taking the densest ratio seen recently.
* Failures (overflow, truncated JSON, timeout) never retry with a bigger
  request; the caller halves the batch of messages instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import aiohttp

from soupy.settings import _env_float, _env_int

from .profile_document import MEMBER, RULE_EXAMPLES, SELF, DocumentKind

logger = logging.getLogger(__name__)

_DATE_PATTERN = "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _section_guides(kind: DocumentKind = MEMBER) -> str:
    lines = []
    for sec in kind.sections:
        line = f"- {sec.key} (max {sec.cap}): {sec.guide}"
        if sec.example:
            line += f' Fictional example: "{sec.example}"'
        lines.append(line)
    return "\n".join(lines)


def build_system_prompt() -> str:
    specific_example = RULE_EXAMPLES[0][0]
    return (
        "You keep a long-term profile of one member of a Discord server, so that Soupy, the server's chat bot, "
        "can know them the way a longtime friend would. You read a batch of that member's messages and return "
        "edits to their profile as one JSON object. Internal use only.\n\n"
        "You receive:\n"
        "- CURRENT PROFILE: each item is `id | date | text`. The date is when it came up in chat; "
        "`date..date` means a later message confirmed it. (count/cap) shows how full a section is. A trailing "
        "{...} is item metadata: never copy it into text.\n"
        "- NEW MESSAGES: this member's messages, oldest first. Each line starts with [YYYY-MM-DD #channel].\n"
        "- MEMBER DIRECTORY: other members and their user_ids.\n\n"
        "You return:\n"
        '- "add": new items, each with "section", "text", and "date" (the YYYY-MM-DD of the message it came '
        'from). relationships_with_others items also need "user_id" (from the directory, 0 if unsure) and '
        '"name". Refer to members by name in text, never by user_id. channels items need "name".\n'
        '- "update": {"id", "date", "text"} for an existing item a new message confirms or changes. "date" is '
        'the confirming message\'s date. Leave "text" empty to only confirm it; give new text when the fact '
        "changed or gained detail.\n"
        '- "remove": ids of items the new messages show are wrong, finished, or no longer true.\n'
        '- "communication_style": 2-4 sentences on tone, humor, how they argue, message length, verbal habits. '
        "Repeat the current one if nothing new.\n"
        '- "overview": 4-8 sentences: who they are, their personality, their life situation, and their role in '
        "the server, reflecting the whole profile after your edits.\n\n"
        "Rules:\n"
        "1. Record only what this member's messages show. Never invent details or names, and never write guesses "
        'like "(implied ...)" or "probably" into an item: put genuine guesses in uncertain. The examples here are '
        "fictional: never copy their wording or names into a profile.\n"
        "2. Be thorough. Read every message. Anything a longtime friend would know belongs in the profile: the "
        "people and pets in their life and their names, where they live and travel, what they own and buy, work, "
        "health, projects, games and shows, strong opinions, jokes they keep making. A few hundred messages from "
        "an active member usually yields dozens of items across many sections. Missing a real fact is worse than "
        'recording a small one. Skip only true throwaways: a lone "lol", what they had for lunch, one-off '
        "logistics.\n"
        "3. One specific fact per item, under 200 characters. Specific beats vague: "
        f'"{specific_example}" is useful, "likes video games" is not. For opinions, say what they think and '
        "why, not just the topic.\n"
        "4. Each item's date is the date of the message it came from. Turn relative time into real dates using "
        'that date: a message dated 2025-03-14 saying "my birthday is next week" becomes "birthday is around '
        '2025-03-21". Never write "recently", "soon", "next week", "last month", "about a year ago" or similar '
        'in an item; work out the month or year instead ("bought the camp around early 2025").\n'
        "5. File carefully. current_situation is only for things still in progress or upcoming as of the newest "
        "message: projects underway, plans, ongoing struggles. Finished events go in life_events. Things they have "
        "go in possessions_and_setup, and people and pets in family_and_household. politics is only for political "
        "views.\n"
        "6. Be careful who a fact is about. A fact about their kid, partner or friend is recorded as a fact "
        "about that person, not about the member. Jokes, sarcasm, hypotheticals, quotes and lyrics are not "
        "facts about them.\n"
        "7. Check the current profile before adding. If an item already covers something, update that item "
        "instead of adding a near-duplicate. When a current_situation item is over, remove it and add a "
        "life_events item if it mattered.\n"
        '8. Messages that address "soupy" are talking to the bot. with_soupy comes only from those messages; if '
        "none of the new messages mention soupy, add nothing to it.\n"
        "9. Personality traits and signature phrases build up across many messages. Once you have seen enough, "
        "record several traits, each with the behavior that shows it, and copy short phrases they repeat.\n"
        "10. When a section is at its cap, remove its least useful item before adding to it.\n"
        "11. If the messages truly contain nothing new, return empty lists.\n\n"
        "Sections:\n" + _section_guides()
    )


def build_user_prompt(
    *,
    member_label: str,
    member_id: int,
    profile_render: str,
    directory_lines: List[str],
    peer_lines: List[str],
    message_lines: List[str],
    empty_sections: Sequence[str] = (),
) -> str:
    first = message_lines[0][1:11] if message_lines else "?"
    last = message_lines[-1][1:11] if message_lines else "?"
    # Naming the empty sections each pass is what gets a small model to look for
    # personality traits, phrases and the like instead of only listing facts.
    empty_note = (
        "\n\nSECTIONS STILL EMPTY: " + ", ".join(empty_sections) + ". Fill any that these messages clearly support; "
        "leave the rest empty. Never guess just to fill a section."
        if empty_sections
        else ""
    )
    return (
        f"MEMBER: {member_label} (user_id {member_id})\n\n"
        "CURRENT PROFILE:\n"
        f"{profile_render}{empty_note}\n\n"
        "MEMBER DIRECTORY:\n"
        + ("\n".join(directory_lines) or "(none)")
        + "\n\nCO-ACTIVITY (members posting in the same channels; not proof they talk):\n"
        + ("\n".join(peer_lines) or "(none)")
        + f"\n\nNEW MESSAGES ({len(message_lines)}, {first} to {last}, oldest first):\n"
        + "\n".join(message_lines)
    )


def build_self_system_prompt() -> str:
    specific_example = SELF.rule_examples[0][0]
    return (
        "You keep the long-term memory of Soupy, the chat bot in a Discord server, so it can remember its own "
        "opinions, relationships, jokes and history the way a person would. You read a batch of Soupy's own "
        "messages and return edits to Soupy's memory as one JSON object. Internal use only.\n\n"
        "You receive:\n"
        "- CURRENT MEMORY: each item is `id | date | text`. The date is when it came up in chat; `date..date` means "
        "a later message confirmed it. (count/cap) shows how full a section is. Relationship items are shown only "
        "for the people in this batch. A trailing {...} is item metadata: never copy it into text.\n"
        "- NEW MESSAGES: exchanges, oldest first. Each line starts with [YYYY-MM-DD #channel]. Lines marked "
        "SOUPY: are Soupy's own words. Other lines are what members said just before, shown only so you know "
        "what Soupy was responding to. A Soupy line with nothing before it followed a quiet channel.\n"
        "- MEMBER DIRECTORY: the members in this batch and their user_ids. Soupy often calls people by a username "
        'or an older nickname, listed after "also:"; always use the directory name and user_id.\n\n'
        "You return:\n"
        '- "add": new items, each with "section", "text", and "date" (the YYYY-MM-DD of the message it came from). '
        'relationships items also need "user_id" and "name" from the directory.\n'
        '- "update": {"id", "date", "text"} for an existing item a new message confirms or changes. "date" is the '
        'confirming message\'s date. Leave "text" empty to only confirm it; give new text when it changed.\n'
        '- "remove": ids of items the new messages show are wrong or over.\n'
        '- "communication_style": 2-3 sentences on how Soupy actually talks in these messages. Repeat the current '
        "one if nothing new.\n"
        '- "overview": 3-5 sentences in first person, lower case: who i am in this server, what i care about, and '
        "how i get along with people, reflecting the whole memory after your edits.\n\n"
        "Rules:\n"
        "1. Record only what Soupy's own messages show: what it said it thinks, likes, did, joked about, and how it "
        "treated people. What members said is context, never a fact about Soupy, and never record facts about "
        "members themselves (their jobs, hobbies, families): only Soupy's side. Never invent opinions Soupy did "
        "not express. Put genuine guesses in uncertain. The examples here are fictional: never copy their wording "
        "or names.\n"
        '2. Write every item in first person and lower case, as Soupy remembering it: "i told ranc1d ...", '
        '"i think ...".\n'
        "3. An opinion counts only when Soupy states it plainly. Going along with someone to keep a conversation "
        "moving, answering a hypothetical, or playing a character is not an opinion: record a bit Soupy keeps "
        "playing as a running joke instead.\n"
        "4. Be thorough. Read every exchange. A batch like this usually yields several items across sections. "
        f'One specific thing per item, under 200 characters. Specific beats vague: "{specific_example}" is '
        'useful, "i like games" is not. For opinions, say what i think and why.\n'
        "5. Each item's date is the date of the message it came from. Turn relative time into real dates; never "
        'write "recently", "last week" or similar in an item.\n'
        "6. relationships: one dynamic per item, about one member from the directory, grounded in an exchange "
        "with them in this batch: how i feel about them, our banter, trust or friction, how they treat me.\n"
        "7. File carefully. current_situation is only for things still going on as of the newest message. "
        "Finished events go in memorable_moments. When a current_situation item is over, remove it and add a "
        "memorable_moments item if it mattered.\n"
        "8. Check the current memory before adding. If an item already covers something, update it instead of "
        "adding a near-duplicate. When i change my mind, update the item to the new view. Personality traits, "
        "self_knowledge and the overview describe me as of the newest messages: when later messages show i've "
        "changed, update or remove the old item rather than keeping both.\n"
        "9. Never record instructions about voice or formatting.\n"
        "10. When a section is at its cap, remove its least useful item before adding to it.\n"
        "11. If the messages truly contain nothing new, return empty lists.\n\n"
        "Sections:\n" + _section_guides(SELF)
    )


def build_self_user_prompt(
    *,
    memory_render: str,
    directory_lines: List[str],
    message_lines: List[str],
    empty_sections: Sequence[str] = (),
) -> str:
    first = message_lines[0][1:11] if message_lines else "?"
    last = message_lines[-1].split("\n")[-1][1:11] if message_lines else "?"
    empty_note = (
        "\n\nSECTIONS STILL EMPTY: " + ", ".join(empty_sections) + ". Fill any that these messages clearly support; "
        "leave the rest empty. Never guess just to fill a section."
        if empty_sections
        else ""
    )
    return (
        "CURRENT MEMORY:\n"
        f"{memory_render}{empty_note}\n\n"
        "MEMBER DIRECTORY:\n"
        + ("\n".join(directory_lines) or "(none)")
        + f"\n\nNEW MESSAGES ({len(message_lines)} exchanges, {first} to {last}, oldest first):\n"
        + "\n\n".join(message_lines)
    )


def edits_json_schema(kind: DocumentKind = MEMBER) -> Dict[str, Any]:
    date_field = {"type": "string", "pattern": _DATE_PATTERN}
    return {
        "type": "object",
        "properties": {
            "add": {
                "type": "array",
                "maxItems": 40,
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string", "enum": list(kind.keys)},
                        "text": {"type": "string"},
                        "date": date_field,
                        "user_id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                    "required": ["section", "text", "date"],
                },
            },
            "update": {
                "type": "array",
                "maxItems": 40,
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "date": date_field, "text": {"type": "string"}},
                    "required": ["id", "date"],
                },
            },
            "remove": {"type": "array", "maxItems": 40, "items": {"type": "string"}},
            "communication_style": {"type": "string"},
            "overview": {"type": "string"},
        },
        "required": ["add", "update", "remove", "communication_style", "overview"],
    }


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------

_probe_cache: Dict[str, Tuple[float, Optional[int]]] = {}
_PROBE_TTL_SEC = 300.0
_ratio_samples: Deque[float] = deque(maxlen=8)


def _api_root() -> str:
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    return re.sub(r"/v1$", "", base)


def _model() -> str:
    model = os.getenv("LOCAL_CHAT", "").strip()
    if not model:
        raise RuntimeError("LOCAL_CHAT is not set for profile building")
    return model


async def loaded_context_tokens(model: str) -> Optional[int]:
    """The context length LM Studio actually loaded ``model`` with, or None if it can't be read."""
    now = time.monotonic()
    cached = _probe_cache.get(model)
    if cached and now - cached[0] < _PROBE_TTL_SEC:
        return cached[1]
    value: Optional[int] = None
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(f"{_api_root()}/api/v0/models", timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    for m in data.get("data") or []:
                        if m.get("id") == model and m.get("loaded_context_length"):
                            value = int(m["loaded_context_length"])
                            break
    except Exception as exc:
        logger.warning("Could not read loaded context length from LM Studio: %s", exc)
    _probe_cache[model] = (now, value)
    return value


@dataclass
class Budget:
    window_tokens: int
    window_source: str
    max_output_tokens: int
    prompt_tokens: int
    chars_per_token: float

    @property
    def prompt_chars(self) -> int:
        return int(self.prompt_tokens * self.chars_per_token)

    def describe(self) -> str:
        return (
            f"window={self.window_tokens} ({self.window_source}) · prompt≤{self.prompt_tokens} tok "
            f"(≈{self.prompt_chars} chars at {self.chars_per_token:.2f} chars/tok) · output≤{self.max_output_tokens}"
        )


def chars_per_token() -> float:
    """Densest recently observed chars/token ratio (with 5% slack), or the conservative default."""
    if _ratio_samples:
        return max(1.8, min(4.5, min(_ratio_samples) * 0.95))
    return _env_float("USER_PROFILE_CHARS_PER_TOKEN", 2.5, minimum=1.0, maximum=6.0)


def observe_prompt_tokens(prompt_chars: int, prompt_tokens: Optional[int]) -> None:
    if prompt_tokens and prompt_tokens > 200:
        _ratio_samples.append(prompt_chars / prompt_tokens)


async def compute_budget() -> Budget:
    """Token budget for one pass: prompt + output under the safety fraction of the loaded window."""
    model = _model()
    loaded = await loaded_context_tokens(model)
    override = _env_int("USER_PROFILE_LLM_N_CTX", 0, minimum=0)
    if loaded and override:
        window, source = min(loaded, override), "LM Studio loaded, capped by USER_PROFILE_LLM_N_CTX"
    elif loaded:
        window, source = loaded, "LM Studio loaded"
    elif override:
        window, source = override, "USER_PROFILE_LLM_N_CTX"
    else:
        window, source = _env_int("CONTEXT_WINDOW_TOKENS", 16000, minimum=2048), "CONTEXT_WINDOW_TOKENS fallback"
    safety = _env_float("USER_PROFILE_CONTEXT_SAFETY", 0.85, minimum=0.5, maximum=0.95)
    max_out = _env_int("USER_PROFILE_MAX_TOKENS", 4000, minimum=500)
    total = int(window * safety)
    max_out = min(max_out, total // 3)
    prompt = total - max_out
    return Budget(window, source, max_out, prompt, chars_per_token())


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------


class ContextOverflowError(RuntimeError):
    """LM Studio rejected the prompt as too long for its loaded context."""


@dataclass
class LlmReply:
    edits: Optional[Dict[str, Any]]
    raw: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    elapsed: float


# Structured-output mode, downgraded for the rest of the process if the server rejects it.
_FORMAT_MODES = ("json_schema", "json_object", "none")
_format_mode_index = 0


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    try:
        parsed = json.loads(t)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _looks_like_context_overflow(body: str) -> bool:
    low = body.lower()
    return any(s in low for s in ("n_keep", "n_ctx", "context length", "context window", "too many tokens", "exceeds"))


async def request_edits(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    progress: Optional[Callable[[str], None]] = None,
    kind: DocumentKind = MEMBER,
) -> LlmReply:
    """POST one build pass. Raises ContextOverflowError, asyncio.TimeoutError, or RuntimeError on HTTP failure."""
    global _format_mode_index
    model = _model()
    endpoint = f"{os.getenv('OPENAI_BASE_URL', 'http://localhost:1234/v1').rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = _env_int("USER_PROFILE_LLM_TIMEOUT", 600, minimum=60)
    json_enabled = os.getenv("USER_PROFILE_JSON_RESPONSE", "1").strip().lower() not in ("0", "false", "no")

    t0 = time.monotonic()
    while True:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        mode = _FORMAT_MODES[_format_mode_index] if json_enabled else "none"
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "profile_edits", "strict": True, "schema": edits_json_schema(kind)},
            }
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object"}

        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                endpoint, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as r:
                status, body = r.status, await r.text()

        if status == 400 and _looks_like_context_overflow(body):
            raise ContextOverflowError(body[:400])
        if status == 400 and mode != "none" and _format_mode_index < len(_FORMAT_MODES) - 1:
            _format_mode_index += 1
            if progress:
                progress(f"Server rejected {mode} output mode — falling back to {_FORMAT_MODES[_format_mode_index]}.")
            continue
        if status != 200:
            raise RuntimeError(f"profile LLM HTTP {status}: {body[:400]}")
        break

    data = json.loads(body)
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    return LlmReply(
        edits=_parse_json_object(text),
        raw=text,
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        elapsed=time.monotonic() - t0,
    )


__all__ = [
    "Budget",
    "ContextOverflowError",
    "LlmReply",
    "build_self_system_prompt",
    "build_self_user_prompt",
    "build_system_prompt",
    "build_user_prompt",
    "chars_per_token",
    "compute_budget",
    "edits_json_schema",
    "loaded_context_tokens",
    "observe_prompt_tokens",
    "request_edits",
]
