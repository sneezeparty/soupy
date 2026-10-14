"""The prompt files behind the dashboard's prompt editor.

Each prompt resolves the way ``soupy/prompts.py`` resolves it for the bot:
the legacy env var, then ``prompts/<name>.txt``, then the shipped
``prompts/<name>.default.txt``. The editor only ever writes
``prompts/<name>.txt``, and keeps the version it replaces under
``prompts/backups/``.

Kept apart from ``web.app`` so the tests can use it without starting the panel.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

BACKUP_DIR = "backups"
# Per prompt. Older backups are deleted as new ones are made.
BACKUPS_KEPT = 20


@dataclass(frozen=True)
class PromptSpec:
    name: str  # file stem under prompts/, and the name soupy.prompts.load_prompt() takes
    label: str
    used_for: str
    group: str  # "personality" or "images"
    env_var: Optional[str] = None  # legacy .env-stable key that overrides the file while set


# Every prompt the bot loads. tests/test_web_prompts.py fails when the code
# loads one that is missing here, or when a legacy env var drifts from
# soupy/prompts.py.
PROMPTS: Tuple[PromptSpec, ...] = (
    PromptSpec("behaviour", "Chat personality", "The system prompt for every chat reply.", "personality", "BEHAVIOUR"),
    PromptSpec("behaviour_search", "Search answers", "/soupysearch summaries.", "personality", "BEHAVIOUR_SEARCH"),
    PromptSpec(
        "behaviour_daily_post",
        "Article posts",
        "Commentary on the daily article post.",
        "personality",
        "BEHAVIOUR_DAILY_POST",
    ),
    PromptSpec("nineball", "9-ball", "/9ball answers.", "personality", "9BALL"),
    PromptSpec(
        "fancy",
        "Fancy rewrite (Stable Diffusion)",
        "Rewrites an image prompt for the Fancy button.",
        "images",
        "FANCY",
    ),
    PromptSpec("fancy_flux", "Fancy rewrite (Flux)", "Rewrites an image prompt for the Fancy button.", "images"),
    PromptSpec(
        "randomprompt",
        "Random prompt (Stable Diffusion)",
        "Turns random keywords into an image prompt.",
        "images",
        "RANDOMPROMPT",
    ),
    PromptSpec("randomprompt_flux", "Random prompt (Flux)", "Turns random keywords into an image prompt.", "images"),
    PromptSpec(
        "sd_negative_prompt",
        "Negative prompt (Stable Diffusion)",
        "Sent with every Stable Diffusion request. Flux has no negative prompt and never sees it.",
        "images",
        "SD_NEGATIVE_PROMPT",
    ),
)

_BY_NAME: Dict[str, PromptSpec] = {p.name: p for p in PROMPTS}


class PromptError(Exception):
    """A read, save or reset that can't go ahead. ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def spec(name: str) -> PromptSpec:
    found = _BY_NAME.get(str(name or "").strip().lower())
    if found is None:
        raise PromptError(f"Unknown prompt: {name!r}", 404)
    return found


def _read(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _modified(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def read_state(name: str, prompts_dir: Path, env: Mapping[str, str]) -> dict:
    """One prompt as the bot will load it at its next start, plus where that text comes from.

    ``env`` is the parsed ``.env-stable``. ``source`` is ``env``, ``custom``,
    ``default`` or ``missing``.
    """
    s = spec(name)
    custom_path = prompts_dir / f"{s.name}.txt"
    default_path = prompts_dir / f"{s.name}.default.txt"
    custom = _read(custom_path)
    default = _read(default_path)
    env_value = env.get(s.env_var, "") if s.env_var else ""
    modified_at = None
    if env_value:
        source, value = "env", env_value
    elif custom is not None:
        source, value, modified_at = "custom", custom, _modified(custom_path)
    elif default is not None:
        source, value, modified_at = "default", default, _modified(default_path)
    else:
        source, value = "missing", ""
    return {
        "name": s.name,
        "label": s.label,
        "used_for": s.used_for,
        "group": s.group,
        "env_var": s.env_var,
        "source": source,
        "value": value,
        "default": default,
        "has_custom": custom is not None,
        "modified_at": modified_at,
        "file": f"prompts/{s.name}.txt",
    }


def list_states(prompts_dir: Path, env: Mapping[str, str]) -> List[dict]:
    return [read_state(p.name, prompts_dir, env) for p in PROMPTS]


def _backup(target: Path, prompts_dir: Path) -> str:
    """Copy ``target`` into the backups folder and return the copy's path relative to the repo root."""
    folder = prompts_dir / BACKUP_DIR
    folder.mkdir(parents=True, exist_ok=True)
    # The names sort by time. The glob needs the dot so "fancy" doesn't claim "fancy_flux".
    existing = sorted(folder.glob(f"{target.stem}.2*.txt"))
    base = f"{target.stem}.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    name = f"{base}.txt"
    if existing and existing[-1].name.startswith(base):
        # A second save in the same second. "_" sorts after the "." of the first.
        numbered = re.search(r"_(\d+)\.txt$", existing[-1].name)
        name = f"{base}_{(int(numbered.group(1)) if numbered else 1) + 1:02d}.txt"
    shutil.copy2(target, folder / name)
    for old in (existing + [folder / name])[:-BACKUPS_KEPT]:
        old.unlink(missing_ok=True)
    return f"prompts/{BACKUP_DIR}/{name}"


def _overridden(s: PromptSpec, env: Mapping[str, str]) -> None:
    if s.env_var and env.get(s.env_var):
        raise PromptError(
            f"{s.env_var} is set in .env-stable and overrides prompts/{s.name}.txt. "
            "Remove that line (tools/migrate_prompts.py does it for you), then save here.",
            409,
        )


def save(name: str, text: str, prompts_dir: Path, env: Mapping[str, str]) -> dict:
    """Write ``prompts/<name>.txt`` exactly as given, keeping the version it replaces.

    Returns ``{"changed", "backup"}``. Saving text identical to the stored
    file writes nothing.
    """
    s = spec(name)
    _overridden(s, env)
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    target = prompts_dir / f"{s.name}.txt"
    previous = _read(target)
    if previous == text:
        return {"changed": False, "backup": None}
    prompts_dir.mkdir(parents=True, exist_ok=True)
    backup = _backup(target, prompts_dir) if previous is not None else None
    tmp = target.with_name(target.name + ".tmp")
    # Bytes, so no platform rewrites the line endings on the way out.
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, target)
    return {"changed": True, "backup": backup}


def reset(name: str, prompts_dir: Path, env: Mapping[str, str]) -> dict:
    """Go back to the shipped default by moving ``prompts/<name>.txt`` into the backups folder."""
    s = spec(name)
    _overridden(s, env)
    target = prompts_dir / f"{s.name}.txt"
    if not target.is_file():
        raise PromptError(f"prompts/{s.name}.txt doesn't exist; the shipped default is already in use.", 409)
    if not (prompts_dir / f"{s.name}.default.txt").is_file():
        raise PromptError(f"There is no shipped default for {s.name}.", 409)
    backup = _backup(target, prompts_dir)
    target.unlink()
    return {"changed": True, "backup": backup}
