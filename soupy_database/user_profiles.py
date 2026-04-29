"""
Server-local member sketches from archived messages (SQLite).

Structured JSON + plain summary for RAG. Message excerpts remain authoritative for quotes and facts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import aiohttp

from .database import get_db_path
from .profile_batch import (
    cancel_profile_task,
    ensure_profile_job_schema,
    profile_job_log_append,
    profile_job_log_clear,
    spawn_profile_worker,
    upsert_job,
    delete_job,
    get_job_row,
)

logger = logging.getLogger(__name__)


def _migrate_profile_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(user_profile_summaries)")
    cols = {row[1] for row in cur.fetchall()}
    if "structured_json" not in cols:
        cur.execute("ALTER TABLE user_profile_summaries ADD COLUMN structured_json TEXT")
    conn.commit()


def ensure_user_profile_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile_summaries (
            user_id INTEGER PRIMARY KEY,
            nickname_hint TEXT,
            summary TEXT NOT NULL,
            source_message_count INTEGER NOT NULL DEFAULT 0,
            source_max_message_id INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_used TEXT,
            structured_json TEXT
        )
        """
    )
    conn.commit()
    _migrate_profile_columns(conn)
    ensure_profile_job_schema(conn)


def _sample_messages_for_user(
    conn: sqlite3.Connection,
    user_id: int,
    limit: int,
    *,
    after_message_id: Optional[int] = None,
) -> Tuple[List[str], int, Optional[str]]:
    """Return (lines newest-first for prompt), count of rows used, latest nickname."""
    cur = conn.cursor()
    if after_message_id is not None:
        cur.execute(
            """
            SELECT message_content, channel_name, date, time, nickname
            FROM messages
            WHERE user_id = ?
              AND message_id > ?
              AND coalesce(trim(message_content), '') != ''
            ORDER BY date DESC, time DESC, message_id DESC
            LIMIT ?
            """,
            (user_id, int(after_message_id), limit),
        )
    else:
        cur.execute(
            """
            SELECT message_content, channel_name, date, time, nickname
            FROM messages
            WHERE user_id = ?
              AND coalesce(trim(message_content), '') != ''
            ORDER BY date DESC, time DESC, message_id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
    rows = cur.fetchall()
    if not rows:
        return [], 0, None
    lines: List[str] = []
    nick = None
    for r in rows:
        body = (r["message_content"] or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:399] + "…"
        ch = r["channel_name"] or "?"
        when = f"{r['date']} {r['time']}"
        nl = (r["nickname"] or "").strip()
        if nl and not nick:
            nick = nl
        lines.append(f"[#{ch} {when}] {body}")
    return lines, len(rows), nick


def _rows_to_lines_newest_first(rows: List[sqlite3.Row]) -> Tuple[List[str], Optional[str]]:
    """Build prompt lines (newest first) and a nickname hint from message rows."""
    lines_chrono: List[str] = []
    nick = None
    for r in rows:
        body = (r["message_content"] or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:399] + "…"
        ch = r["channel_name"] or "?"
        when = f"{r['date']} {r['time']}"
        nl = (r["nickname"] or "").strip()
        if nl and not nick:
            nick = nl
        lines_chrono.append(f"[#{ch} {when}] {body}")
    lines_newest_first = list(reversed(lines_chrono))
    return lines_newest_first, nick


def _stratify_row_numbers(
    n_total: int,
    budget: int,
) -> List[int]:
    """1-based row indices (chronological order) covering oldest, newest, and spread middle."""
    if n_total <= 0 or budget <= 0:
        return []
    if n_total <= budget:
        return list(range(1, n_total + 1))
    try:
        r_old = float(os.getenv("USER_PROFILE_STRATIFY_OLDEST_RATIO", "0.18"))
    except ValueError:
        r_old = 0.18
    try:
        r_new = float(os.getenv("USER_PROFILE_STRATIFY_NEWEST_RATIO", "0.38"))
    except ValueError:
        r_new = 0.38
    n_old = max(1, int(budget * r_old))
    n_new = max(1, int(budget * r_new))
    n_mid = max(0, budget - n_old - n_new)

    rn_set: Set[int] = set()
    for i in range(1, min(n_old, n_total) + 1):
        rn_set.add(i)
    for i in range(max(1, n_total - n_new + 1), n_total + 1):
        rn_set.add(i)
    lo = n_old + 1
    hi = n_total - n_new
    if n_mid > 0 and lo <= hi:
        span = hi - lo + 1
        if span <= n_mid:
            for i in range(lo, hi + 1):
                rn_set.add(i)
        else:
            for j in range(n_mid):
                if n_mid == 1:
                    rn_set.add(lo + span // 2)
                else:
                    rn_set.add(lo + int(j * (span - 1) / max(1, n_mid - 1)))

    out = sorted(rn_set)
    if len(out) <= budget:
        return out
    if budget == 1:
        return [out[len(out) // 2]]
    n = len(out)
    step = (n - 1) / float(budget - 1)
    raw = [out[min(n - 1, int(round(i * step)))] for i in range(budget)]
    uniq = list(dict.fromkeys(raw))
    if len(uniq) >= budget:
        return sorted(uniq[:budget])
    seen = set(uniq)
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
            if len(uniq) >= budget:
                break
    return sorted(uniq[:budget])


def _sample_messages_stratified(
    conn: sqlite3.Connection,
    user_id: int,
    budget: int,
) -> Tuple[List[str], int, Optional[str]]:
    """
    Sample up to `budget` messages across the user's full history (oldest + newest + spread).
    Returns (lines newest-first), count, nickname hint — same as _sample_messages_for_user.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM messages
        WHERE user_id = ? AND coalesce(trim(message_content), '') != ''
        """,
        (user_id,),
    )
    n_total = int(cur.fetchone()[0])
    if n_total == 0:
        return [], 0, None
    if n_total <= budget:
        return _sample_messages_for_user(conn, user_id, budget)

    rns = _stratify_row_numbers(n_total, budget)
    if not rns:
        return [], 0, None
    placeholders = ",".join("?" * len(rns))
    cur.execute(
        f"""
        WITH ordered AS (
            SELECT message_content, channel_name, date, time, nickname,
                   ROW_NUMBER() OVER (ORDER BY message_id) AS rn
            FROM messages
            WHERE user_id = ? AND coalesce(trim(message_content), '') != ''
        )
        SELECT message_content, channel_name, date, time, nickname
        FROM ordered
        WHERE rn IN ({placeholders})
        ORDER BY rn ASC
        """,
        (user_id, *rns),
    )
    rows = cur.fetchall()
    lines_nf, nick = _rows_to_lines_newest_first(rows)
    return lines_nf, len(rows), nick


def _top_member_directory(conn: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id,
               COUNT(*) AS c,
               MAX(COALESCE(NULLIF(TRIM(nickname), ''), NULLIF(TRIM(username), ''))) AS label
        FROM messages
        GROUP BY user_id
        ORDER BY c DESC
        LIMIT ?
        """,
        (limit,),
    )
    out: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        uid = int(r["user_id"])
        lbl = (r["label"] or "").strip() or f"user_{uid}"
        out.append({"user_id": uid, "label": lbl, "message_count": int(r["c"])})
    return out


def _peer_interaction_hints(
    conn: sqlite3.Connection, user_id: int, limit: int = 12
) -> List[str]:
    """Short lines: who shares channels most with this user (proxy for interaction)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m2.user_id AS oid,
               MAX(COALESCE(NULLIF(TRIM(m2.nickname), ''), NULLIF(TRIM(m2.username), ''))) AS olab,
               COUNT(*) AS c
        FROM messages m1
        JOIN messages m2 ON m1.channel_id = m2.channel_id
            AND m1.user_id = ? AND m2.user_id != ?
        GROUP BY m2.user_id
        ORDER BY c DESC
        LIMIT ?
        """,
        (user_id, user_id, limit),
    )
    lines: List[str] = []
    for r in cur.fetchall():
        oid = int(r["oid"])
        lab = (r["olab"] or "").strip() or str(oid)
        c = int(r["c"])
        lines.append(f"Shares channels with user_id={oid} ({lab}) — ~{c} messages in same channels")
    return lines


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _flatten_list_section(title: str, items: Any, max_items: int) -> Optional[str]:
    if not isinstance(items, list) or not items:
        return None
    lines = [str(x).strip() for x in items[:max_items] if x and str(x).strip()]
    if not lines:
        return None
    return title + ":\n" + "\n".join(f"• {t}" for t in lines)


def _flatten_structured_for_summary(data: Dict[str, Any]) -> str:
    """Plain-text block for DB summary column + RAG fallback (bounded length)."""
    try:
        max_chars = int(os.getenv("USER_PROFILE_SUMMARY_MAX_CHARS", "12000"))
    except ValueError:
        max_chars = 12000
    parts: List[str] = []
    ov = (data.get("overview") or "").strip()
    if ov:
        parts.append("Overview:\n" + ov)

    for title, key, cap in (
        ("Personal & biographical", "personal_and_biographical", 25),
        ("Opinions & stances", "opinions_and_stances", 20),
        ("Hobbies & activities", "hobbies", 18),
        ("Media & entertainment (games, film, TV, music, books, etc.)", "media_entertainment", 20),
        ("Discussion topics (recurring themes in this server)", "discussion_topics", 16),
    ):
        block = _flatten_list_section(title, data.get(key), cap)
        if block:
            parts.append(block)

    legacy = data.get("topics_interests") or []
    if isinstance(legacy, list) and legacy:
        lb = _flatten_list_section("Topics (legacy field)", legacy, 14)
        if lb:
            parts.append(lb)

    tone = (data.get("communication_style") or "").strip()
    if tone:
        parts.append("Communication:\n" + tone)
    chans = data.get("channels") or []
    if isinstance(chans, list) and chans:
        lines = []
        for c in chans[:12]:
            if isinstance(c, dict):
                nm = c.get("name") or ""
                note = c.get("note") or ""
                lines.append(f"• {nm}: {note}".strip())
            elif isinstance(c, str):
                lines.append(f"• {c}")
        if lines:
            parts.append("Channels:\n" + "\n".join(lines))
    rels = data.get("relationships_with_others") or []
    if isinstance(rels, list) and rels:
        lines = []
        for r in rels[:12]:
            if not isinstance(r, dict):
                continue
            uid = r.get("user_id")
            ref = (r.get("display_ref") or "").strip() or str(uid or "")
            rel = (r.get("relationship") or "").strip()
            if uid and uid != 0:
                lines.append(f"• With {ref} (id {uid}): {rel}")
            else:
                lines.append(f"• With {ref}: {rel}")
        if lines:
            parts.append("Relationships (approximate, from archive patterns):\n" + "\n".join(lines))
    thin = data.get("uncertain_or_thin_data") or []
    if isinstance(thin, list) and thin:
        parts.append("Uncertain / thin:\n" + "\n".join(f"• {t}" for t in thin[:8] if t))
    pol = (data.get("inferred_politics") or "").strip()
    if pol:
        parts.append(f"Inferred politics / ideology (low confidence): {pol}")
    loc = (data.get("inferred_location") or "").strip()
    if loc:
        parts.append(f"Inferred location / region (hints only): {loc}")
    work = (data.get("inferred_work_or_education") or "").strip()
    if work:
        parts.append(f"Work / school / field (inferred): {work}")
    out = "\n\n".join(parts).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1] + "…"
    return out or "(empty profile)"


def _profile_json_schema_instructions() -> str:
    """Shared LLM instructions for structured profile shape (create + merge)."""
    return (
        "Specificity: prefer concrete nouns (game titles, film names, tools, sports, bands) over vague phrases. "
        'Bad: "likes video games". Good: "Elden Ring, Factorio, fighting games". Empty arrays are fine if unknown.\n'
        "IMPORTANT: Do not be generic. Capture what makes this person THEM. "
        "Include their actual opinions, not just topic labels. "
        '"Discusses California politics" is useless. "Thinks California is over-regulated, frustrated with squatter laws '
        'and eviction timelines, has dealt with it as a landlord" is useful.\n\n'
        "CRITICAL STRUCTURE RULE: The overview is a SHORT summary paragraph. "
        "All specific facts, details, and examples MUST go into the appropriate array fields below, NOT the overview. "
        "If you dump everything into the overview and leave the arrays empty or sparse, the output is WRONG. "
        "The overview is a 4-8 sentence paragraph that gives the general picture. "
        "The detailed facts go into personal_and_biographical, opinions_and_stances, hobbies, media_entertainment, etc.\n\n"
        '- "overview": string, 4–8 sentences MAXIMUM. A concise portrait: who they are, their vibe/personality, '
        "general life situation, and role in the server. Mention key themes (e.g. 'busy mom with cats and a complicated "
        "marriage' or 'tech worker who debates politics and plays Valheim') but do NOT list every fact here. "
        "Specific details belong in the arrays below. If your overview exceeds 8 sentences, you are doing it wrong.\n"
        '- "personal_and_biographical": array of strings (up to 30): THIS IS WHERE SPECIFIC LIFE FACTS GO. '
        "Concrete personal facts mentioned in chat—"
        "pets and their names/species, family members (kids, spouse, parents) and details about them, "
        "life events (moves, divorces, health issues, job changes), "
        "possessions, vehicles, living situation, health mentions, DIY projects, things they own or have done. "
        "Each item should be a specific fact with enough detail to be useful, one fact per string. "
        'Examples: "has two cats named Chloe and Cyrus", '
        '"husband Rich started insulin due to health issues", '
        '"son is majoring in Technical Theatre and is already married for medical rights", '
        '"teaching herself to crochet, making paw print coasters for a rescue fundraiser", '
        '"struggling with potential divorce", '
        '"recently found a salon that spent 6 hours detangling her long red hair". '
        "This is the most important field for answering personal questions about the user. "
        "Be thorough — capture EVERY personal fact you can find in the messages.\n"
        '- "opinions_and_stances": array of strings (up to 25): their actual positions, opinions, and takes on topics '
        "they feel strongly about. Not just the topic name—capture WHAT they think and WHY. "
        'Examples: "thinks AI companies are moving too fast without safety guardrails", '
        '"frustrated with mother-in-law\'s gift-giving pressure", '
        '"believes Rich\'s behavior is a mix of bipolar, alcoholism, and narcissism", '
        '"critical of how insurance/Medi-Cal handles paperwork". '
        "Only include opinions clearly expressed in their messages, not inferences.\n"
        '- "hobbies": array of strings (up to 18): things they do or practice—sports, crafts, collecting, '
        "making, outdoors, fitness, coding side projects, etc. Be specific and include detail when available "
        '(not just "crocheting" but "crocheting complex patterns like doily snowflakes and paw print coasters for rescue fundraisers").\n'
        '- "media_entertainment": array of strings (up to 20): video games, films, series, anime, music, books, '
        "podcasts, streamers, creators they mention, recommend, or argue about. Include titles and their opinion of it "
        'when stated (not just "watches Disney+" but "watched Agatha All Along on Disney+, also 3 Body Problems").\n'
        '- "discussion_topics": array of strings (up to 16): recurring themes they engage with in this server. '
        "For each topic, briefly note their angle or stance if evident. "
        'Example: "marriage/relationship struggles — vents about Rich\'s behavior, seeks advice" '
        'rather than just "relationships".\n'
        '- "topics_interests": optional legacy array; may be empty. If older data used only this field, split items into '
        "hobbies, media_entertainment, and discussion_topics when merging.\n"
        '- "communication_style": string, 2-4 sentences: tone, humor, argument style, length habits. '
        "Include concrete examples of their phrasing and any verbal tics or patterns.\n"
        '- "inferred_politics": string, 1-3 sentences: political leaning, specific positions, or ideology '
        "if clearly inferable from chat. Include specific policy positions when stated, not just a label. "
        'Use "unknown" or "not stated" when absent. Never treat as verified fact.\n'
        '- "inferred_location": string: rough region, country, city, or timezone hints if stated; else "unknown".\n'
        '- "inferred_work_or_education": string, 1-2 sentences: job, studies, field, employer type, side projects '
        'if inferable. Include specifics when available; else "unknown".\n'
        '- "channels": array of { "name": "#channel or label", "note": "what they do there and their typical angle" } (up to 10).\n'
        '- "relationships_with_others": array of { "user_id": number or 0 if unknown, '
        '"display_ref": "nickname or label", "relationship": "specific dynamics—how they interact, '
        'inside jokes, frequent banter topics" } '
        "(up to 12). Use ONLY user_ids from the member directory when you infer a specific person; use 0 if unsure.\n"
        '- "uncertain_or_thin_data": array of short strings (gaps, maybe-wrong guesses, low-confidence items).\n'
        "English only."
    )


def _profile_json_response_format_enabled() -> bool:
    """OpenAI-style json_object mode (LM Studio / many OpenAI-compatible servers)."""
    return os.getenv("USER_PROFILE_JSON_RESPONSE", "1").strip() not in ("0", "false", "no")


def _profile_completion_retry_cap() -> int:
    try:
        return int(os.getenv("USER_PROFILE_RETRY_MAX_TOKENS", "10000"))
    except ValueError:
        return 10000


def _profile_prompt_char_budget(reserved_output_tokens: int) -> int:
    """
    Rough max combined chars for system + user messages so prompt+completion fits n_ctx.
    LM Studio / llama.cpp rejects when n_keep (prompt) + max_tokens exceeds context.
    """
    if os.getenv("USER_PROFILE_SKIP_CONTEXT_CAP", "0").strip() in ("1", "true", "yes"):
        return 10**9
    try:
        n_ctx = int(os.getenv("USER_PROFILE_LLM_N_CTX", "22000"))
    except ValueError:
        n_ctx = 22000
    try:
        tok_buf = int(os.getenv("USER_PROFILE_LLM_CONTEXT_BUFFER_TOKENS", "384"))
    except ValueError:
        tok_buf = 384
    try:
        cpt = float(os.getenv("USER_PROFILE_CHARS_PER_TOKEN", "2.0"))
    except ValueError:
        cpt = 2.0
    prompt_tok = max(256, n_ctx - reserved_output_tokens - tok_buf)
    base = max(4000, int(prompt_tok * cpt))
    try:
        frac = float(os.getenv("USER_PROFILE_CONTEXT_BUDGET_FRACTION", "0.92"))
    except ValueError:
        frac = 0.92
    return int(base * max(0.5, min(1.0, frac)))


def _fallback_profile_from_raw(raw: str) -> Dict[str, Any]:
    return {
        "overview": raw[:2000],
        "personal_and_biographical": [],
        "opinions_and_stances": [],
        "hobbies": [],
        "media_entertainment": [],
        "discussion_topics": [],
        "topics_interests": [],
        "communication_style": "",
        "inferred_politics": "unknown",
        "inferred_location": "unknown",
        "inferred_work_or_education": "unknown",
        "channels": [],
        "relationships_with_others": [],
        "uncertain_or_thin_data": ["Model did not return valid JSON; stored raw overview only."],
    }


class _ContextOverflowError(RuntimeError):
    """Prompt exceeded LLM context window (n_keep >= n_ctx)."""


async def _llm_profile_build_or_merge(
    *,
    mode: str,
    sample_lines_chrono: List[str],
    member_directory: List[Dict[str, Any]],
    peer_hints: List[str],
    existing_structured: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], str]:
    """mode: 'create' | 'merge'. Returns (structured dict, raw model text)."""
    model = os.getenv("LOCAL_CHAT", "").strip()
    if not model:
        raise RuntimeError("LOCAL_CHAT is not set for profile summarization")
    base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    endpoint = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        max_lines = int(os.getenv("USER_PROFILE_SAMPLE_MESSAGES", "90"))
    except ValueError:
        max_lines = 90
    if mode == "create":
        try:
            max_lines = int(os.getenv("USER_PROFILE_CREATE_MAX_MESSAGES", "800"))
        except ValueError:
            max_lines = 800
    elif mode == "merge":
        try:
            max_lines = int(os.getenv("USER_PROFILE_MERGE_SAMPLE_MESSAGES", str(max_lines)))
        except ValueError:
            pass
    line_window = list(sample_lines_chrono[-max_lines:])

    try:
        dir_n = int(os.getenv("USER_PROFILE_MEMBER_DIRECTORY", "45"))
    except ValueError:
        dir_n = 45
    dir_lines = [
        f"- user_id={m['user_id']} display={m['label'][:80]} (msgs≈{m['message_count']})"
        for m in member_directory[:dir_n]
    ]
    peer_block = "\n".join(peer_hints[:20]) if peer_hints else "(no co-channel stats)"
    schema = _profile_json_schema_instructions()

    try:
        max_tok = int(os.getenv("USER_PROFILE_MAX_TOKENS", "5000"))
    except ValueError:
        max_tok = 5000
    if mode == "merge":
        try:
            merge_tok = int(os.getenv("USER_PROFILE_MERGE_MAX_TOKENS", "5000"))
        except ValueError:
            merge_tok = 5000
        max_tok = max(max_tok, merge_tok)

    prompt_char_budget = _profile_prompt_char_budget(max_tok)

    suffix_common = (
        "\n\n--- Member directory (for relationship user_ids) ---\n"
        + "\n".join(dir_lines)
        + "\n\n--- Co-activity hints (same channels; not proof of DMs) ---\n"
        + peer_block
    )

    if mode == "merge" and existing_structured:
        sys_prompt = (
            "You update a stored member profile (JSON) using NEW Discord messages. Internal use only. "
            "Reply with ONE JSON object (no markdown fences, no commentary). "
            "Merge intelligently with the SAME keys as below.\n\n"
            "Rules:\n"
            "1) Output a COMPLETE profile (all keys), not a patch.\n"
            "2) Keep prior conclusions when still supported by the archive.\n"
            "3) Update fields when new messages add detail or clearly contradict old inferences.\n"
            "4) Remove or replace items only when they are clearly wrong or obsolete—do not wipe a field "
            "just because new messages are quiet about it.\n"
            "5) Demographics (politics, location, work) are soft inferences—use \"unknown\" when unsupported.\n"
            "6) If the stored profile still has only \"topics_interests\", migrate entries into hobbies, "
            "media_entertainment, and discussion_topics where they fit.\n"
            "7) Pay special attention to new personal details: pets, family, possessions, life events, "
            "projects. Add them to \"personal_and_biographical\" even if they seem minor—these facts "
            "are how the bot answers personal questions about the user.\n"
            "8) Capture actual OPINIONS and STANCES in the \"opinions_and_stances\" field. Not just "
            "\"discusses politics\" but what they actually think and why.\n"
            "9) Be thorough—a rich, detailed profile is better than a short generic one.\n"
            "10) CRITICAL: The overview MUST stay short (4-8 sentences). It is a SUMMARY paragraph only. "
            "All specific facts, life events, and details MUST go into the array fields "
            "(personal_and_biographical, opinions_and_stances, hobbies, media_entertainment, etc.). "
            "Do NOT dump everything into the overview.\n\n"
            "Keys:\n"
            + schema
        )
    else:
        sys_prompt = (
            "You analyze one Discord member's archived messages for internal software use only. "
            "Reply with a single JSON object (no markdown fences, no commentary). "
            "Be DEEPLY SPECIFIC: capture their actual opinions, personal details, life facts, and personality—"
            "not just topic labels. "
            '"Discusses politics" is useless. "Thinks X about Y because Z" is useful. '
            "CRITICAL: Keep the overview SHORT (4-8 sentences). Put all specific facts, details, and life events "
            "into the array fields (personal_and_biographical, opinions_and_stances, hobbies, media_entertainment, etc.). "
            "A good profile has a concise overview and RICH, FULL arrays. A bad profile dumps everything into the overview "
            "and leaves the arrays empty. Keys:\n"
            + schema
        )

    def _trim_lines_to_budget(raw_lines: List[str]) -> List[str]:
        """Drop oldest lines until system + user prompt fits within the char budget."""
        trimmed = list(raw_lines)
        while len(trimmed) > 1:
            ub = _build_user_block(trimmed)
            if len(sys_prompt) + len(ub) <= prompt_char_budget:
                break
            trimmed = trimmed[1:]
        return trimmed

    def _build_user_block(cur_lines: List[str]) -> str:
        joined = "\n".join(cur_lines)
        if mode == "merge" and existing_structured:
            # Use compact JSON (no indent) to save context space for more message lines.
            # The LLM reads compact JSON fine and this can save 3000-5000 chars of whitespace.
            existing_blob = json.dumps(existing_structured, ensure_ascii=False, separators=(",", ":"))
            if len(existing_blob) > 14000:
                existing_blob = existing_blob[:13999] + "…"
            return (
                "--- CURRENT STORED PROFILE (merge with new evidence below) ---\n"
                + existing_blob
                + "\n\n--- NEW MESSAGES from this member since last profile (oldest first) ---\n"
                + joined
                + suffix_common
            )
        return (
            "Archived lines from THIS member only (oldest first):\n\n"
            + joined
            + "\n\n---\nMember directory (for interpreting names and relationship user_ids):\n"
            + "\n".join(dir_lines)
            + "\n\n---\nCo-activity hints (same channels as this member; not proof of DMs):\n"
            + peer_block
        )

    lines = _trim_lines_to_budget(list(line_window))
    if progress and len(lines) < len(line_window):
        progress(
            f"Profile prompt trimmed for LM context (budget≈{prompt_char_budget} chars, "
            f"n_ctx env): {len(lines)}/{len(line_window)} message lines kept (dropped oldest first)."
        )

    try:
        llm_timeout = int(os.getenv("USER_PROFILE_LLM_TIMEOUT", "600"))
    except ValueError:
        llm_timeout = 600
    llm_timeout = max(60, llm_timeout)

    host = urlparse(base).netloc or base.replace("https://", "").replace("http://", "").split("/")[0]
    retry_cap = _profile_completion_retry_cap()
    json_mode_failed = False
    max_timeout_retries = int(os.getenv("USER_PROFILE_TIMEOUT_RETRIES", "2"))

    async def _attempt_llm_call(
        cur_lines: List[str],
        cur_max_tok: int,
        cur_timeout: int,
        attempt_label: str,
    ) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
        """One full LLM round (including response_format fallback). Returns (parsed, raw, finish_reason)."""
        nonlocal json_mode_failed

        ub = _build_user_block(cur_lines)
        cur_payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": ub},
            ],
            "temperature": 0.25,
        }
        if _profile_json_response_format_enabled() and not json_mode_failed:
            cur_payload["response_format"] = {"type": "json_object"}

        sys_chars = len(sys_prompt)
        user_chars = len(ub)
        if progress:
            progress(
                f"LLM {mode} ({attempt_label}): model={model!r} · max_tokens={cur_max_tok} · "
                f"timeout={cur_timeout}s · host={host} · system≈{sys_chars} chars · "
                f"user≈{user_chars} chars (lines: {len(cur_lines)})"
            )

        async def _post(pay: Dict[str, Any], mt: int) -> Tuple[int, str]:
            req = dict(pay)
            req["max_tokens"] = min(mt, retry_cap)
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    endpoint,
                    headers=headers,
                    json=req,
                    timeout=aiohttp.ClientTimeout(total=cur_timeout),
                ) as r:
                    return r.status, await r.text()

        t0 = time.monotonic()
        http_status, body = await _post(cur_payload, cur_max_tok)
        if http_status == 400 and cur_payload.get("response_format"):
            del cur_payload["response_format"]
            json_mode_failed = True
            if progress:
                progress("Server rejected JSON response mode — retrying without response_format…")
            http_status, body = await _post(cur_payload, cur_max_tok)
        elapsed = time.monotonic() - t0
        if progress:
            progress(
                f"LLM HTTP {http_status} in {elapsed:.1f}s · body≈{len(body)} chars "
                f"(max_tokens={min(cur_max_tok, retry_cap)})"
            )
        if http_status == 400 and "n_keep" in body and "n_ctx" in body:
            raise _ContextOverflowError(f"profile LLM context overflow: {body[:400]}")
        if http_status != 200:
            raise RuntimeError(f"profile LLM HTTP {http_status}: {body[:400]}")
        data = json.loads(body)
        choice = (data.get("choices") or [{}])[0]
        fr = choice.get("finish_reason")
        text = (choice.get("message") or {}).get("content") or ""
        return _try_parse_json_object(text.strip()), text.strip(), fr

    attempt_max = max_tok
    parsed: Optional[Dict[str, Any]] = None
    raw = ""
    finish_reason: Optional[str] = None
    cur_lines = list(lines)

    for overall_attempt in range(1 + max_timeout_retries):
        label = f"attempt {overall_attempt + 1}"
        try:
            parsed, raw, finish_reason = await _attempt_llm_call(
                cur_lines, attempt_max, llm_timeout, label,
            )
        except _ContextOverflowError as ce:
            if overall_attempt < max_timeout_retries and len(cur_lines) > 8:
                new_count = max(8, len(cur_lines) * 2 // 3)
                if progress:
                    progress(
                        f"Context overflow with {len(cur_lines)} lines — "
                        f"retrying with {new_count} lines (reduced prompt)…"
                    )
                cur_lines = cur_lines[-new_count:]
                continue
            raise RuntimeError(
                f"Profile LLM context overflow even after reducing to "
                f"{len(cur_lines)} message lines: {ce}"
            ) from ce
        except (asyncio.TimeoutError, TimeoutError) as te:
            if overall_attempt < max_timeout_retries and len(cur_lines) > 8:
                new_count = max(8, len(cur_lines) // 2)
                if progress:
                    progress(
                        f"Timeout after {llm_timeout}s with {len(cur_lines)} lines — "
                        f"retrying with {new_count} lines (halved prompt)…"
                    )
                cur_lines = cur_lines[-new_count:]
                continue
            raise RuntimeError(
                f"Profile LLM timed out ({llm_timeout}s) even after reducing to "
                f"{len(cur_lines)} message lines: {te!r}"
            ) from te

        if parsed is not None:
            break
        looks_truncated = finish_reason == "length" or (
            len(raw) >= 100 and raw.lstrip().startswith("{")
        )
        if overall_attempt == 0 and looks_truncated:
            attempt_max = min(max(attempt_max * 2, max_tok + 800), retry_cap)
            if progress:
                progress(
                    f"Profile JSON truncated (finish={finish_reason!r}); "
                    f"retrying with max_tokens={min(attempt_max, retry_cap)}…"
                )
            continue
        break

    if progress:
        if parsed is None:
            progress(
                "Model output is not valid JSON — storing raw text as overview only (fallback shape)."
            )
        else:
            keys = [k for k in parsed.keys() if isinstance(k, str)]
            preview = ", ".join(keys[:8])
            extra = " …" if len(keys) > 8 else ""
            progress(f"Parsed structured profile JSON ({len(keys)} keys: {preview}{extra}).")
    if parsed is None:
        return _fallback_profile_from_raw(raw), raw
    return parsed, raw


def _fetch_all_user_messages_chrono(
    conn: sqlite3.Connection,
    user_id: int,
) -> Tuple[List[str], int, Optional[str]]:
    """Fetch ALL messages for a user in chronological order. Returns (lines, count, nickname)."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT message_content, channel_name, date, time, nickname
        FROM messages
        WHERE user_id = ?
          AND coalesce(trim(message_content), '') != ''
        ORDER BY message_id ASC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return [], 0, None
    lines: List[str] = []
    nick = None
    for r in rows:
        body = (r["message_content"] or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:399] + "…"
        ch = r["channel_name"] or "?"
        when = f"{r['date']} {r['time']}"
        nl = (r["nickname"] or "").strip()
        if nl:
            nick = nl
        lines.append(f"[#{ch} {when}] {body}")
    return lines, len(lines), nick


def _estimate_lines_per_chunk(
    sys_prompt_chars: int,
    existing_profile_chars: int,
    suffix_chars: int,
    budget_chars: int,
) -> int:
    """Estimate how many message lines fit in one LLM chunk after overhead."""
    overhead = sys_prompt_chars + existing_profile_chars + suffix_chars + 500  # padding
    available = max(2000, budget_chars - overhead)
    # Average message line is ~120 chars
    return max(20, available // 120)


async def _chunked_profile_build(
    *,
    all_lines_chrono: List[str],
    member_directory: List[Dict[str, Any]],
    peer_hints: List[str],
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Multi-pass profile build that processes the ENTIRE message archive.
    First chunk → CREATE, subsequent chunks → MERGE with accumulated profile.
    Chunk sizes are calculated dynamically based on the actual accumulated
    profile size, so messages aren't silently dropped by _trim_lines_to_budget.
    """
    try:
        max_tok = int(os.getenv("USER_PROFILE_MAX_TOKENS", "5000"))
    except ValueError:
        max_tok = 5000
    prompt_budget = _profile_prompt_char_budget(max_tok)
    schema = _profile_json_schema_instructions()

    # Fixed overhead: system prompt + member directory + peer hints
    sys_chars = len(schema) + 600  # approximate system prompt size
    suffix_chars = sum(len(f"- user_id={m['user_id']} display={m['label'][:80]} (msgs≈{m['message_count']})")
                       for m in member_directory[:45]) + 200
    peer_chars = sum(len(p) for p in peer_hints[:20]) + 100

    total = len(all_lines_chrono)
    accumulated: Optional[Dict[str, Any]] = None
    raw = ""
    pos = 0
    pass_num = 0
    t0 = time.monotonic()

    # Estimate total passes for progress display (will refine as we go)
    first_chunk = _estimate_lines_per_chunk(sys_chars, 0, suffix_chars + peer_chars, prompt_budget)
    # Use a conservative initial estimate for merge overhead
    est_merge_chunk = _estimate_lines_per_chunk(sys_chars, 12000, suffix_chars + peer_chars, prompt_budget)
    est_total_passes = 1 + max(1, (total - first_chunk) // max(est_merge_chunk, 1)) if total > first_chunk else 1

    if progress:
        progress(
            f"Chunked profile build: {total} messages · budget≈{prompt_budget} chars · "
            f"~{est_total_passes} estimated pass(es)"
        )

    while pos < total:
        pass_num += 1
        mode = "create" if pass_num == 1 else "merge"

        # Calculate chunk size dynamically based on actual accumulated profile size
        if accumulated is not None:
            profile_json_chars = len(json.dumps(accumulated, ensure_ascii=False, indent=2))
            # The profile JSON is capped at 12000 in _build_user_block, so respect that
            profile_overhead = min(profile_json_chars, 12000) + 200  # +200 for headers
        else:
            profile_overhead = 0

        chunk_size = _estimate_lines_per_chunk(
            sys_chars, profile_overhead, suffix_chars + peer_chars, prompt_budget
        )
        # Ensure minimum progress per chunk
        chunk_size = max(chunk_size, 15)

        end = min(pos + chunk_size, total)
        chunk_lines = all_lines_chrono[pos:end]
        msgs_pct = int((end / total) * 100) if total else 0

        if progress:
            progress(
                f"  ▸ Pass {pass_num} [{msgs_pct}% of archive]: {mode} · "
                f"messages {pos + 1}–{end} of {total} ({len(chunk_lines)} in this chunk"
                f"{f', profile JSON={profile_overhead} chars' if profile_overhead else ''})"
            )

        chunk_t0 = time.monotonic()
        accumulated, raw = await _llm_profile_build_or_merge(
            mode=mode,
            sample_lines_chrono=chunk_lines,
            member_directory=member_directory,
            peer_hints=peer_hints,
            existing_structured=accumulated,
            progress=progress,
        )
        chunk_elapsed = time.monotonic() - chunk_t0

        if progress:
            progress(
                f"  ◂ Pass {pass_num} done in {chunk_elapsed:.1f}s · "
                f"{end}/{total} messages processed [{msgs_pct}%]"
            )

        pos = end

    total_elapsed = time.monotonic() - t0
    if progress:
        progress(
            f"  Chunked build complete: {pass_num} pass(es) over {total} messages "
            f"in {total_elapsed:.1f}s ({total_elapsed / max(pass_num, 1):.1f}s/pass avg)"
        )

    if accumulated is None:
        accumulated = _fallback_profile_from_raw(raw)
    return accumulated, raw


def _upsert_profile(
    conn: sqlite3.Connection,
    user_id: int,
    nickname_hint: Optional[str],
    summary: str,
    structured_json: Optional[str],
    source_count: int,
    source_max_mid: Optional[int],
    model_used: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_profile_summaries (
            user_id, nickname_hint, summary, structured_json, source_message_count,
            source_max_message_id, updated_at, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            nickname_hint = excluded.nickname_hint,
            summary = excluded.summary,
            structured_json = excluded.structured_json,
            source_message_count = excluded.source_message_count,
            source_max_message_id = excluded.source_max_message_id,
            updated_at = CURRENT_TIMESTAMP,
            model_used = excluded.model_used
        """,
        (
            user_id,
            nickname_hint or "",
            summary,
            structured_json,
            source_count,
            source_max_mid,
            model_used,
        ),
    )
    conn.commit()


async def refresh_user_profile(
    guild_id: int,
    user_id: int,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Create or intelligently merge a profile from the messages table (LOCAL_CHAT).
    First run: by default a stratified sample across the full archive (see env).
    Later runs: merge stored JSON with new messages since source_max_message_id
    when enough new lines exist.
    """
    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "error": "no database"}

    try:
        sample_limit = int(os.getenv("USER_PROFILE_SAMPLE_MESSAGES", "90"))
    except ValueError:
        sample_limit = 90
    try:
        merge_sample_limit = int(os.getenv("USER_PROFILE_MERGE_SAMPLE_MESSAGES", "90"))
    except ValueError:
        merge_sample_limit = 90
    try:
        min_new_merge = int(os.getenv("USER_PROFILE_MIN_NEW_MESSAGES_FOR_MERGE", "5"))
    except ValueError:
        min_new_merge = 5
    try:
        create_budget = int(os.getenv("USER_PROFILE_CREATE_MAX_MESSAGES", "800"))
    except ValueError:
        create_budget = 800

    def _prepare() -> Dict[str, Any]:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            ensure_user_profile_schema(conn)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT structured_json, source_max_message_id, updated_at
                FROM user_profile_summaries
                WHERE user_id = ?
                """,
                (user_id,),
            )
            prow = cur.fetchone()
            existing_structured: Optional[Dict[str, Any]] = None
            last_max_mid: Optional[int] = None
            profile_age_days: Optional[float] = None
            if prow:
                try:
                    sj = prow["structured_json"] or ""
                    if sj.strip():
                        parsed = json.loads(sj)
                        if isinstance(parsed, dict) and parsed:
                            existing_structured = parsed
                except Exception:
                    existing_structured = None
                if prow["source_max_message_id"] is not None:
                    last_max_mid = int(prow["source_max_message_id"])
                # Compute profile age so we can force a merge when it's stale.
                try:
                    updated_raw = prow["updated_at"] or ""
                    if updated_raw:
                        updated_dt = datetime.fromisoformat(str(updated_raw).replace(" ", "T"))
                        profile_age_days = (datetime.utcnow() - updated_dt.replace(tzinfo=None)).total_seconds() / 86400
                except Exception:
                    profile_age_days = None

            cur.execute(
                """
                SELECT COUNT(*) AS c FROM messages
                WHERE user_id = ? AND coalesce(trim(message_content), '') != ''
                """,
                (user_id,),
            )
            total_msgs = int(cur.fetchone()["c"])
            min_m = int(os.getenv("USER_PROFILE_MIN_MESSAGES", "8"))
            # Never sample fewer lines than min_m (avoids spurious skips when env budgets are tiny).
            create_budget_eff = max(create_budget, min_m, 1)
            sample_limit_eff = max(sample_limit, min_m, 1)
            merge_sample_limit_eff = max(merge_sample_limit, min_m, 1)

            cur.execute(
                "SELECT MAX(message_id) AS mx FROM messages WHERE user_id = ?",
                (user_id,),
            )
            mx = cur.fetchone()
            max_mid = int(mx["mx"]) if mx and mx["mx"] is not None else None
            try:
                md_n = int(os.getenv("USER_PROFILE_MEMBER_DIRECTORY", "45"))
            except ValueError:
                md_n = 45
            directory = _top_member_directory(conn, md_n)
            peers = _peer_interaction_hints(conn, user_id)

            if existing_structured is not None:
                try:
                    max_age_days = float(os.getenv("USER_PROFILE_MAX_AGE_DAYS", "14"))
                except ValueError:
                    max_age_days = 14
                profile_is_stale = (
                    profile_age_days is not None and max_age_days > 0
                    and profile_age_days >= max_age_days
                )
                if last_max_mid is not None:
                    lines_nf, n_new, nick = _sample_messages_for_user(
                        conn,
                        user_id,
                        merge_sample_limit_eff,
                        after_message_id=last_max_mid,
                    )
                    # Skip only if: not enough new messages AND profile isn't stale.
                    # If the profile is stale (older than max_age_days), force a merge
                    # using recent messages even if the count is below min_new_merge.
                    if n_new < min_new_merge and not profile_is_stale:
                        return {
                            "kind": "skip",
                            "reason": "no_new_messages_for_merge",
                            "n_new": n_new,
                            "min_new": min_new_merge,
                            "total_msgs": total_msgs,
                        }
                    if n_new < min_new_merge and profile_is_stale:
                        # Stale profile: fall back to a recent window instead of just new messages
                        lines_nf, n_new, nick = _sample_messages_for_user(
                            conn, user_id, merge_sample_limit_eff
                        )
                    chrono = list(reversed(lines_nf))
                    stale_note = f" (profile was {profile_age_days:.0f}d old)" if profile_is_stale else ""
                    return {
                        "kind": "run",
                        "mode": "merge",
                        "chrono": chrono,
                        "sample_n": n_new,
                        "nick": nick,
                        "max_mid": max_mid,
                        "directory": directory,
                        "peers": peers,
                        "existing_structured": existing_structured,
                        "total_msgs": total_msgs,
                        "intro": (
                            f"Merge: {n_new} message(s) after message_id {last_max_mid}"
                            f"{stale_note} (archive total {total_msgs})."
                        ),
                    }
                merge_fetch = min(
                    total_msgs,
                    max(min_m, sample_limit, merge_sample_limit),
                )
                lines_nf, n_full, nick = _sample_messages_for_user(conn, user_id, merge_fetch)
                if n_full < min_m and total_msgs >= min_m:
                    lines_nf, n_full, nick = _sample_messages_for_user(
                        conn,
                        user_id,
                        min(total_msgs, max(merge_fetch, merge_sample_limit_eff, sample_limit_eff)),
                    )
                if n_full < min_m:
                    return {
                        "kind": "skip",
                        "reason": "too_few_messages",
                        "n": n_full,
                        "min_m": min_m,
                        "total_msgs": total_msgs,
                    }
                chrono = list(reversed(lines_nf))
                return {
                    "kind": "run",
                    "mode": "merge",
                    "chrono": chrono,
                    "sample_n": n_full,
                    "nick": nick,
                    "max_mid": max_mid,
                    "directory": directory,
                    "peers": peers,
                    "existing_structured": existing_structured,
                    "total_msgs": total_msgs,
                    "intro": (
                        f"Merge (no prior message cutoff): full recent window ({n_full} messages) "
                        "vs stored profile JSON."
                    ),
                }

            create_strat = os.getenv("USER_PROFILE_CREATE_STRATIFIED", "1").strip() not in (
                "0",
                "false",
                "no",
            )
            if create_strat:
                lines_nf, n, nick = _sample_messages_stratified(conn, user_id, create_budget_eff)
                intro_detail = (
                    f"stratified sample (budget {create_budget_eff} lines: oldest + newest + spread middle)"
                )
            else:
                lines_nf, n, nick = _sample_messages_for_user(conn, user_id, sample_limit_eff)
                intro_detail = f"recent-only window ({sample_limit_eff} newest messages)"
            if n < min_m and total_msgs >= min_m:
                # Archive qualifies but the sampler under-delivered (misconfigured budget, stratify edge case).
                fb = max(min_m, min(sample_limit_eff, total_msgs))
                lines_nf, n, nick = _sample_messages_for_user(conn, user_id, fb)
            if n < min_m:
                return {
                    "kind": "skip",
                    "reason": "too_few_messages",
                    "n": n,
                    "min_m": min_m,
                    "total_msgs": total_msgs,
                }
            chrono = list(reversed(lines_nf))
            return {
                "kind": "run",
                "mode": "create",
                "chrono": chrono,
                "sample_n": n,
                "nick": nick,
                "max_mid": max_mid,
                "directory": directory,
                "peers": peers,
                "existing_structured": None,
                "total_msgs": total_msgs,
                "intro": (
                    f"Create: {n} message line(s) in prompt — {intro_detail} "
                    f"(archive total {total_msgs} usable messages)."
                ),
            }
        finally:
            conn.close()

    loop = asyncio.get_running_loop()
    plan = await loop.run_in_executor(None, _prepare)

    if plan["kind"] == "skip":
        reason = plan["reason"]
        if reason == "no_new_messages_for_merge":
            _p(
                f"Skip merge: only {plan['n_new']} new message(s) since last profile "
                f"(need ≥{plan['min_new']})."
            )
            return {
                "ok": True,
                "skipped": True,
                "reason": reason,
                "n_new": plan["n_new"],
                "total_msgs": plan.get("total_msgs"),
            }
        _p(
            f"Skip profile: only {plan['n']} usable messages in archive "
            f"(minimum {plan['min_m']})."
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": reason,
            "count": plan["n"],
        }

    mode = plan["mode"]
    chrono = plan["chrono"]
    sample_n = plan["sample_n"]
    nick = plan["nick"]
    max_mid = plan["max_mid"]
    directory = plan["directory"]
    peers = plan["peers"]
    existing_structured = plan["existing_structured"]
    total_msgs = plan["total_msgs"]
    intro = plan.get("intro") or ""

    nick_s = (nick or "").strip() or "(none)"
    _p(
        f"user_id={user_id} ({nick_s}) · {intro} · "
        f"member_directory={len(directory)} · peer_hints={len(peers)}"
    )
    profile_t0 = time.monotonic()

    # Use chunked multi-pass build for CREATE when the archive has significantly
    # more messages than fit in a single LLM context window. This ensures the
    # entire archive contributes to the profile, not just a sample.
    chunked_enabled = os.getenv("USER_PROFILE_CHUNKED_BUILD", "1").strip() not in ("0", "false", "no")
    try:
        chunked_threshold = int(os.getenv("USER_PROFILE_CHUNKED_THRESHOLD", "400"))
    except ValueError:
        chunked_threshold = 400

    if mode == "create" and chunked_enabled and total_msgs > chunked_threshold:
        _p(
            f"Archive has {total_msgs} messages (threshold {chunked_threshold}) — "
            "using multi-pass chunked build to cover the full archive."
        )

        def _fetch_all() -> Tuple[List[str], Optional[str]]:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                all_lines, _count, _nick = _fetch_all_user_messages_chrono(conn, user_id)
                return all_lines, _nick
            finally:
                conn.close()

        all_lines, nick_from_all = await loop.run_in_executor(None, _fetch_all)
        if nick_from_all:
            nick = nick_from_all

        structured, _raw = await _chunked_profile_build(
            all_lines_chrono=all_lines,
            member_directory=directory,
            peer_hints=peers,
            progress=_p,
        )
    else:
        structured, _raw = await _llm_profile_build_or_merge(
            mode=mode,
            sample_lines_chrono=chrono,
            member_directory=directory,
            peer_hints=peers,
            existing_structured=existing_structured,
            progress=_p,
        )
    profile_elapsed = time.monotonic() - profile_t0
    summary = _flatten_structured_for_summary(structured)
    sjson = json.dumps(structured, ensure_ascii=False)
    model = os.getenv("LOCAL_CHAT", "")

    # Count fields in the structured profile for a quality summary
    n_personal = len(structured.get("personal_and_biographical") or [])
    n_opinions = len(structured.get("opinions_and_stances") or [])
    n_hobbies = len(structured.get("hobbies") or [])
    n_media = len(structured.get("media_entertainment") or [])
    overview_len = len(structured.get("overview") or "")

    _p(
        f"Saving profile: summary≈{len(summary)} chars · json≈{len(sjson)} chars · "
        f"overview={overview_len} chars · personal={n_personal} · opinions={n_opinions} · "
        f"hobbies={n_hobbies} · media={n_media} · {profile_elapsed:.1f}s elapsed"
    )

    def _save() -> None:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            ensure_user_profile_schema(conn)
            _upsert_profile(
                conn,
                user_id,
                nick,
                summary,
                sjson,
                total_msgs,
                max_mid,
                model,
            )
        finally:
            conn.close()

    await loop.run_in_executor(None, _save)
    _p(
        f"✅ Profile saved: user_id={user_id} ({nick_s}) · mode={mode} · "
        f"archive={total_msgs} msgs · {profile_elapsed:.1f}s total"
    )
    logger.info(
        "user profile refreshed guild=%s user_id=%s mode=%s sample_n=%s total_msgs=%s elapsed=%.1fs",
        guild_id,
        user_id,
        mode,
        sample_n,
        total_msgs,
        profile_elapsed,
    )
    return {
        "ok": True,
        "user_id": user_id,
        "mode": mode,
        "messages_in_prompt": sample_n,
        "messages_used": total_msgs,
    }


async def refresh_profiles_for_guild_limited(guild_id: int) -> Dict[str, Any]:
    """
    Refresh profiles for users with enough messages, capped per run (post-scan / manual).
    """
    try:
        max_users = int(os.getenv("USER_PROFILES_REFRESH_MAX_USERS", "35"))
    except ValueError:
        max_users = 35
    try:
        min_msgs = int(os.getenv("USER_PROFILE_MIN_MESSAGES", "8"))
    except ValueError:
        min_msgs = 8

    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "refreshed": 0}

    def _candidates() -> List[int]:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            ensure_user_profile_schema(conn)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT user_id, COUNT(*) AS c FROM messages
                GROUP BY user_id
                HAVING c >= ?
                ORDER BY c DESC
                LIMIT ?
                """,
                (min_msgs, max_users),
            )
            return [int(r["user_id"]) for r in cur.fetchall()]
        finally:
            conn.close()

    loop = asyncio.get_running_loop()
    uids = await loop.run_in_executor(None, _candidates)
    written = 0
    skipped_n = 0
    for uid in uids:
        try:
            r = await refresh_user_profile(guild_id, uid)
            if r.get("ok") and r.get("skipped"):
                skipped_n += 1
            elif r.get("ok"):
                written += 1
        except Exception as e:
            logger.warning("profile refresh failed guild=%s user=%s: %s", guild_id, uid, e)
        await asyncio.sleep(float(os.getenv("USER_PROFILE_REFRESH_GAP_SEC", "0.35")))
    logger.info(
        "user profiles batch guild=%s written=%s skipped=%s candidates=%s",
        guild_id,
        written,
        skipped_n,
        len(uids),
    )
    return {
        "ok": True,
        "refreshed": written,
        "skipped": skipped_n,
        "candidates": len(uids),
    }


def build_candidate_user_ids(guild_id: int) -> List[int]:
    try:
        max_users = int(os.getenv("USER_PROFILES_BATCH_MAX_USERS", "80"))
    except ValueError:
        max_users = 80
    try:
        min_msgs = int(os.getenv("USER_PROFILE_MIN_MESSAGES", "8"))
    except ValueError:
        min_msgs = 8
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT user_id, COUNT(*) AS c FROM messages
            GROUP BY user_id
            HAVING c >= ?
            ORDER BY c DESC
            LIMIT ?
            """,
            (min_msgs, max_users),
        )
        return [int(r["user_id"]) for r in cur.fetchall()]
    finally:
        conn.close()


def clear_stored_profiles(guild_id: int) -> Dict[str, Any]:
    """Delete all rows in user_profile_summaries and the batch job row."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM user_profile_summaries")
        conn.commit()
        ensure_profile_job_schema(conn)
        delete_job(conn, guild_id)
    finally:
        conn.close()
    profile_job_log_clear(guild_id)
    return {"ok": True}


def _parse_batch_stats_json(raw: Optional[str]) -> Dict[str, int]:
    if not raw or not str(raw).strip():
        return {"saved": 0, "skipped": 0, "failed": 0}
    try:
        d = json.loads(raw)
        if not isinstance(d, dict):
            return {"saved": 0, "skipped": 0, "failed": 0}
        return {
            "saved": int(d.get("saved", 0) or 0),
            "skipped": int(d.get("skipped", 0) or 0),
            "failed": int(d.get("failed", 0) or 0),
        }
    except Exception:
        return {"saved": 0, "skipped": 0, "failed": 0}


async def _run_profile_batch_inner(guild_id: int) -> None:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        profile_job_log_append(guild_id, "No database for this guild.")
        return
    loop = asyncio.get_running_loop()

    while True:

        def _read_job() -> Optional[sqlite3.Row]:
            cx = sqlite3.connect(db_path, check_same_thread=False)
            cx.row_factory = sqlite3.Row
            try:
                ensure_user_profile_schema(cx)
                ensure_profile_job_schema(cx)
                return get_job_row(cx, guild_id)
            finally:
                cx.close()

        row = await loop.run_in_executor(None, _read_job)
        if row is None:
            break
        st = str(row["status"] or "")
        if st in ("idle", "completed"):
            break
        if st == "cancelled":
            profile_job_log_append(guild_id, "Job status is cancelled; exiting.")
            break
        if st == "paused":
            await asyncio.sleep(0.35)
            continue

        uids: List[int] = json.loads(row["user_ids_json"] or "[]")
        i = int(row["next_index"] or 0)
        if not uids:
            profile_job_log_append(guild_id, "No user ids in job; exiting.")
            break
        if i >= len(uids):

            def _done() -> None:
                cx = sqlite3.connect(db_path, check_same_thread=False)
                try:
                    ensure_profile_job_schema(cx)
                    upsert_job(cx, guild_id, status="completed", next_index=len(uids))
                finally:
                    cx.close()

            await loop.run_in_executor(None, _done)
            # Read final stats for the completion message
            final_stats = _parse_batch_stats_json(row["stats_json"] if row else None)
            profile_job_log_append(
                guild_id,
                f"━━━ Batch complete: {len(uids)} users · "
                f"saved={final_stats.get('saved', 0)} · "
                f"skipped={final_stats.get('skipped', 0)} · "
                f"failed={final_stats.get('failed', 0)} · "
                f"guild={guild_id} ━━━",
            )
            break

        uid = uids[i]
        stats = _parse_batch_stats_json(row["stats_json"] if row else None)

        # Look up nickname for better log readability
        def _lookup_nick() -> str:
            cx = sqlite3.connect(db_path, check_same_thread=False)
            cx.row_factory = sqlite3.Row
            try:
                c = cx.cursor()
                c.execute(
                    """
                    SELECT COALESCE(NULLIF(TRIM(nickname), ''), NULLIF(TRIM(username), '')) AS nick
                    FROM messages WHERE user_id = ?
                    ORDER BY message_id DESC LIMIT 1
                    """,
                    (uid,),
                )
                r = c.fetchone()
                return (r["nick"] or "").strip() if r else ""
            finally:
                cx.close()

        nick = await loop.run_in_executor(None, _lookup_nick)
        nick_label = f" ({nick})" if nick else ""
        batch_pct = int((i / len(uids)) * 100) if uids else 0

        profile_job_log_append(
            guild_id,
            f"━━━ User {i + 1}/{len(uids)} [{batch_pct}%]: user_id={uid}{nick_label} · "
            f"guild={guild_id} · saved={stats.get('saved', 0)} skipped={stats.get('skipped', 0)} "
            f"failed={stats.get('failed', 0)} ━━━",
        )
        try:
            result = await refresh_user_profile(
                guild_id,
                uid,
                progress=lambda m: profile_job_log_append(guild_id, m),
            )
            if isinstance(result, dict) and result.get("skipped"):
                stats["skipped"] = stats.get("skipped", 0) + 1
                reason = result.get("reason") or "?"
                profile_job_log_append(
                    guild_id,
                    f"⏭ Skipped user_id={uid}{nick_label} ({reason})",
                )
            elif isinstance(result, dict) and result.get("ok"):
                stats["saved"] = stats.get("saved", 0) + 1
                msgs_used = result.get("messages_in_prompt", "?")
                mode = result.get("mode", "?")
                profile_job_log_append(
                    guild_id,
                    f"✅ Saved profile for user_id={uid}{nick_label} "
                    f"(mode={mode}, messages_in_prompt={msgs_used})",
                )
            else:
                stats["failed"] = stats.get("failed", 0) + 1
                profile_job_log_append(
                    guild_id,
                    f"⚠ Unexpected result for user_id={uid}{nick_label}: {result!r}",
                )
        except Exception as e:
            stats["failed"] = stats.get("failed", 0) + 1
            err_text = str(e).strip()
            if not err_text:
                err_text = f"{type(e).__name__} {repr(e)}".strip()
            profile_job_log_append(guild_id, f"❌ ERROR user_id={uid}{nick_label}: {err_text}")
            logger.exception(
                "profile batch refresh failed guild=%s user_id=%s",
                guild_id,
                uid,
            )

        nxt = i + 1
        stats_json = json.dumps(stats)

        def _advance() -> None:
            cx = sqlite3.connect(db_path, check_same_thread=False)
            try:
                ensure_profile_job_schema(cx)
                upsert_job(cx, guild_id, next_index=nxt, stats_json=stats_json)
            finally:
                cx.close()

        await loop.run_in_executor(None, _advance)
        await asyncio.sleep(float(os.getenv("USER_PROFILE_REFRESH_GAP_SEC", "0.35")))


async def run_profile_batch_worker(guild_id: int) -> None:
    from .profile_batch import forget_profile_task

    try:
        await _run_profile_batch_inner(guild_id)
    except asyncio.CancelledError:
        profile_job_log_append(guild_id, "Worker cancelled.")
        raise
    finally:
        forget_profile_task(guild_id)


def start_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    """Initialize job table and spawn async worker (web process)."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    uids = build_candidate_user_ids(guild_id)
    if not uids:
        return {"ok": False, "message": "No users meet USER_PROFILE_MIN_MESSAGES threshold"}

    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_profile_job_schema(conn)
        upsert_job(
            conn,
            guild_id,
            status="running",
            user_ids_json=json.dumps(uids),
            next_index=0,
            total=len(uids),
            stats_json=json.dumps({"saved": 0, "skipped": 0, "failed": 0}),
        )
    finally:
        conn.close()

    profile_job_log_clear(guild_id)
    chunked = os.getenv("USER_PROFILE_CHUNKED_BUILD", "1").strip() not in ("0", "false", "no")
    chunked_thresh = int(os.getenv("USER_PROFILE_CHUNKED_THRESHOLD", "400")) if chunked else 0
    profile_job_log_append(
        guild_id,
        f"━━━ Batch started: guild={guild_id} · {len(uids)} candidate user(s) · "
        f"chunked={'on' if chunked else 'off'} (threshold={chunked_thresh}) · "
        f"Pause/resume via dashboard ━━━",
    )
    spawn_profile_worker(guild_id, run_profile_batch_worker)
    return {"ok": True, "total": len(uids), "user_ids": uids}


def pause_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_profile_job_schema(conn)
        if get_job_row(conn, guild_id) is None:
            return {"ok": False, "message": "no job"}
        upsert_job(conn, guild_id, status="paused")
    finally:
        conn.close()
    profile_job_log_append(guild_id, "Pause requested — worker idles before the next user.")
    return {"ok": True, "status": "paused"}


def resume_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        ensure_profile_job_schema(conn)
        row = get_job_row(conn, guild_id)
        if row is None:
            return {"ok": False, "message": "no job"}
        upsert_job(conn, guild_id, status="running")
    finally:
        conn.close()
    profile_job_log_append(guild_id, "Resumed.")
    spawn_profile_worker(guild_id, run_profile_batch_worker)
    return {"ok": True, "status": "running"}


async def cancel_profile_batch_job(guild_id: int) -> Dict[str, Any]:
    db_path = get_db_path(guild_id)
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            ensure_profile_job_schema(conn)
            if get_job_row(conn, guild_id) is not None:
                upsert_job(conn, guild_id, status="cancelled")
        finally:
            conn.close()
    profile_job_log_append(guild_id, "Cancel requested.")
    await cancel_profile_task(guild_id)
    return {"ok": True, "status": "cancelled"}


def get_profile_batch_status(guild_id: int) -> Dict[str, Any]:
    from .profile_batch import profile_task_running

    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "message": "database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM user_profile_summaries")
        profile_row_count = int(cur.fetchone()["c"])
        ensure_profile_job_schema(conn)
        row = get_job_row(conn, guild_id)
        if row is None:
            return {
                "ok": True,
                "guild_id": str(guild_id),
                "status": "idle",
                "next_index": 0,
                "total": 0,
                "profile_row_count": profile_row_count,
                "batch_stats": None,
                "task_running": profile_task_running(guild_id),
            }
        d = {k: row[k] for k in row.keys()}
        raw_stats = d.get("stats_json")
        return {
            "ok": True,
            "guild_id": str(guild_id),
            "status": d.get("status"),
            "next_index": int(d.get("next_index") or 0),
            "total": int(d.get("total") or 0),
            "user_ids_json": d.get("user_ids_json"),
            "profile_row_count": profile_row_count,
            "batch_stats": _parse_batch_stats_json(raw_stats) if raw_stats else None,
            "task_running": profile_task_running(guild_id),
        }
    finally:
        conn.close()


def get_user_profile_stats(guild_id: int) -> Dict[str, Any]:
    """Row count and latest refresh time for stored summaries (creates table if missing)."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {"ok": False, "exists": False, "message": "Database not found"}
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_user_profile_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS c FROM user_profile_summaries")
        c = int(cur.fetchone()[0])
        cur.execute("SELECT MAX(updated_at) AS mx FROM user_profile_summaries")
        mx = cur.fetchone()["mx"]
        return {
            "ok": True,
            "exists": True,
            "guild_id": str(guild_id),
            "profile_count": c,
            "latest_updated_at": mx if mx else None,
        }
    finally:
        conn.close()


def _load_summaries(
    conn: sqlite3.Connection, user_ids: Sequence[int]
) -> Dict[int, Tuple[str, str]]:
    """user_id -> (nickname_hint, summary)"""
    ids = [int(u) for u in user_ids if u]
    if not ids:
        return {}
    cur = conn.cursor()
    qmarks = ",".join("?" * len(ids))
    cur.execute(
        f"SELECT user_id, nickname_hint, summary FROM user_profile_summaries WHERE user_id IN ({qmarks})",
        ids,
    )
    out: Dict[int, Tuple[str, str]] = {}
    for r in cur.fetchall():
        out[int(r["user_id"])] = (r["nickname_hint"] or "", r["summary"] or "")
    return out


def _load_structured_profiles(
    conn: sqlite3.Connection, user_ids: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    """user_id -> {nickname_hint, structured, age_days}"""
    ids = [int(u) for u in user_ids if u]
    if not ids:
        return {}
    cur = conn.cursor()
    qmarks = ",".join("?" * len(ids))
    cur.execute(
        f"SELECT user_id, nickname_hint, structured_json, updated_at "
        f"FROM user_profile_summaries WHERE user_id IN ({qmarks})",
        ids,
    )
    out: Dict[int, Dict[str, Any]] = {}
    for r in cur.fetchall():
        uid = int(r["user_id"])
        structured: Dict[str, Any] = {}
        try:
            sj = r["structured_json"] or ""
            if sj.strip():
                parsed = json.loads(sj)
                if isinstance(parsed, dict):
                    structured = parsed
        except Exception:
            pass
        age_days: Optional[float] = None
        try:
            raw = r["updated_at"] or ""
            if raw:
                dt = datetime.fromisoformat(str(raw).replace(" ", "T"))
                age_days = (datetime.utcnow() - dt.replace(tzinfo=None)).total_seconds() / 86400
        except Exception:
            pass
        out[uid] = {
            "nickname_hint": r["nickname_hint"] or "",
            "structured": structured,
            "age_days": age_days,
        }
    return out


# Keywords that suggest a query is asking about a particular profile section.
# Checked against query tokens; the highest-scoring sections are included first.
_SECTION_QUERY_KEYWORDS: Dict[str, List[str]] = {
    "personal_and_biographical": [
        "own", "have", "has", "got", "pet", "pets", "cat", "dog", "animal",
        "family", "kid", "kids", "child", "children", "wife", "husband",
        "partner", "house", "home", "car", "drive", "buy", "bought",
        "married", "move", "moved", "life", "personal", "my", "your",
    ],
    "opinions_and_stances": [
        "think", "thinks", "opinion", "believe", "believes", "feel", "feels",
        "stance", "position", "view", "agree", "disagree", "support",
        "oppose", "hate", "love", "like", "dislike", "prefer", "wrong",
        "right", "should", "shouldnt", "why", "because", "against", "for",
    ],
    "media_entertainment": [
        "game", "games", "gaming", "play", "playing", "watch", "movie", "film",
        "show", "anime", "music", "book", "books", "read", "stream", "streaming",
        "youtube", "twitch", "series", "tv", "netflix", "podcast", "manga",
    ],
    "hobbies": [
        "hobby", "hobbies", "sport", "sports", "craft", "outdoor", "outdoors",
        "fitness", "cook", "cooking", "build", "building", "code", "coding",
        "project", "gym", "run", "running", "drawing", "art", "photography",
    ],
    "discussion_topics": [
        "think", "opinion", "believe", "politics", "view", "stance", "feel",
        "discuss", "debate", "topic", "argue", "argument", "say", "said",
        "rant", "issue", "news", "current",
    ],
    "communication_style": [
        "style", "tone", "communicate", "talk", "argue", "funny", "humor",
        "sarcastic", "sarcasm", "serious", "joke", "how",
    ],
    "inferred_location": [
        "live", "where", "location", "country", "city", "region", "timezone",
        "from", "based", "local",
    ],
    "inferred_politics": [
        "politics", "political", "vote", "voting", "liberal", "conservative",
        "left", "right", "party", "ideology", "democrat", "republican",
    ],
    "inferred_work_or_education": [
        "work", "job", "career", "school", "study", "studying", "degree",
        "field", "profession", "college", "university", "major",
    ],
    "relationships_with_others": [
        "friend", "friends", "know", "knows", "relationship", "together",
        "interact", "crew", "group", "people",
    ],
}


def _score_sections_for_query(query_tokens: List[str]) -> List[Tuple[int, str]]:
    """Return (score, section_name) pairs sorted by descending score, ties broken by order."""
    token_set = set(query_tokens)
    scored: List[Tuple[int, str]] = []
    for section, keywords in _SECTION_QUERY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in token_set)
        scored.append((score, section))
    scored.sort(key=lambda x: -x[0])
    return scored


def _render_profile_sections(
    structured: Dict[str, Any],
    query_tokens: List[str],
    max_chars: int,
) -> str:
    """
    Build a concise profile block from structured JSON, prioritising sections
    most relevant to the query. Always leads with overview.
    """
    if not structured:
        return ""

    parts: List[str] = []
    used = 0

    # Overview is always first.
    overview = (structured.get("overview") or "").strip()
    if overview:
        # Allow overview up to 50% of the per-user budget.
        ov_cap = max(120, int(max_chars * 0.50))
        if len(overview) > ov_cap:
            overview = overview[: ov_cap - 1] + "…"
        parts.append(f"Overview: {overview}")
        used += len(parts[-1])

    # Score remaining sections by relevance to the query.
    scored = _score_sections_for_query(query_tokens)
    remaining = max_chars - used - len(parts) * 2  # rough separator cost

    for _score, section in scored:
        if remaining <= 40:
            break
        val = structured.get(section)
        if not val:
            continue

        if isinstance(val, list):
            items = [str(x).strip() for x in val if x and str(x).strip()]
            if not items:
                continue
            label = section.replace("_", " ").title()
            line = f"{label}: {', '.join(items)}"
        elif isinstance(val, str):
            val = val.strip()
            if not val or val.lower() in ("unknown", "not stated", ""):
                continue
            label = section.replace("_", " ").title()
            line = f"{label}: {val}"
        else:
            continue

        if len(line) > remaining:
            line = line[: remaining - 1] + "…"
        parts.append(line)
        used += len(line) + 2
        remaining = max_chars - used

    return "\n".join(parts)


def format_profile_prefix_for_rag(
    conn: sqlite3.Connection,
    author_user_id: int,
    query_text: str,
    first_person_hint: Optional[str],
    subject_user_id: Optional[int] = None,
) -> str:
    """
    Build a query-aware profile prefix for the RAG bundle.
    Reads structured JSON and picks the sections most relevant to the query,
    rather than always dumping the same flat summary.
    If subject_user_id is provided, load that profile instead of re-resolving from tokens.
    """
    if os.getenv("USER_PROFILES_IN_RAG", "1").strip() in ("0", "false", "no"):
        return ""

    from .rag import (  # local import: rag fully loaded
        _extract_query_tokens,
        is_first_person_archive_query,
        normalize_rag_query_text,
    )

    ensure_user_profile_schema(conn)
    fp_src = (first_person_hint if first_person_hint is not None else query_text) or ""
    fp = is_first_person_archive_query(fp_src)
    # Use current message only for token extraction (avoid conversation history noise)
    _cur_msg = query_text or ""
    if "Current message:\n" in _cur_msg:
        _cur_msg = _cur_msg.split("Current message:\n")[-1].strip()
    import re as _re
    _cur_msg = _re.sub(r"^(?:user|assistant)\s*\([^)]*\)\s*:\s*", "", _cur_msg, flags=_re.IGNORECASE).strip()
    qnorm = normalize_rag_query_text(_cur_msg or query_text or "")
    query_tokens = _extract_query_tokens(qnorm)
    # When asking about a specific other user, only include THAT user's profile
    # to prevent cross-contamination (LLM confuses asker's media list with subject's).
    if not fp and subject_user_id is not None and int(subject_user_id) != int(author_user_id):
        uids: Set[int] = {int(subject_user_id)}
    else:
        uids: Set[int] = {int(author_user_id)}

    profiles = _load_structured_profiles(conn, list(uids))
    if not profiles:
        return ""

    try:
        max_each = int(os.getenv("RAG_PROFILE_MAX_CHARS_PER_USER", "2000"))
    except ValueError:
        max_each = 2000
    try:
        max_total = int(os.getenv("RAG_PROFILE_MAX_CHARS", "3500"))
    except ValueError:
        max_total = 3500

    sketch_parts: List[str] = []
    for uid in sorted(uids):
        row = profiles.get(uid)
        if not row or not row["structured"]:
            continue
        nick = (row["nickname_hint"] or "").strip() or f"user_id {uid}"
        age_days = row["age_days"]
        age_str = ""
        if age_days is not None:
            if age_days < 1:
                age_str = " (updated today)"
            elif age_days < 2:
                age_str = " (updated yesterday)"
            else:
                age_str = f" (updated {int(age_days)}d ago)"

        body = _render_profile_sections(row["structured"], query_tokens, max_each)
        if not body:
            continue
        sketch_parts.append(f"— {nick} (id {uid}){age_str}:\n{body}")

    if not sketch_parts:
        return ""

    header = (
        "Internal sketches from saved chat only—not quotes or proof.\n"
        "Use them for tone, personal facts, and continuity. For what anyone actually said, use the excerpts below.\n"
        "If a user asks a personal question (pets, hobbies, family, etc.), check the sketch AND the excerpts for the answer.\n"
        "Answer in your normal voice; do not present these as a formal profile or read them like a list to the user."
    )
    out = header + "\n\n" + "\n\n".join(sketch_parts)
    if len(out) > max_total:
        out = out[: max_total - 1] + "…"
    return (
        "--- Member sketches (approximate; excerpts below win for facts) ---\n"
        + out
        + "\n--- end sketches ---\n\n"
    )

