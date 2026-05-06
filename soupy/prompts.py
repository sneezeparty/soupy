"""Prompt loading.

Each prompt the bot uses (`BEHAVIOUR`, `BEHAVIOUR_SEARCH`, `9BALL`, etc.)
is resolved in this order:

1. The legacy environment variable (`BEHAVIOUR=...` in `.env-stable`).
   Honoured for backward compatibility — installs that customised the
   prompt in `.env-stable` keep working unchanged. A one-time deprecation
   warning logs on first lookup.
2. `prompts/<name>.txt` — gitignored, the place to drop a custom prompt
   without touching `.env-stable`.
3. `prompts/<name>.default.txt` — tracked, ships with the repo, contains
   the same content that used to live in `.env-stable.example`.
4. The `fallback` argument passed by the caller (last-resort safety net).

This means old installs keep their custom prompts, new installs see the
shipped defaults via the file system, and nobody has to deal with a 5KB
quoted multi-line env-var assignment ever again.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# prompts/ lives at the repo root, one directory up from this file
# (we're at soupy/prompts.py, the prompts directory is sibling to soupy/).
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Prompt name -> legacy env var name. The "name" is the file stem under
# prompts/ (e.g. "behaviour" -> prompts/behaviour.default.txt).
_LEGACY_ENV: Dict[str, str] = {
    "behaviour": "BEHAVIOUR",
    "behaviour_search": "BEHAVIOUR_SEARCH",
    "behaviour_daily_post": "BEHAVIOUR_DAILY_POST",
    "nineball": "9BALL",
    "fancy": "FANCY",
    "randomprompt": "RANDOMPROMPT",
    "sd_negative_prompt": "SD_NEGATIVE_PROMPT",
}

# Cache the resolved value per prompt for the lifetime of the process.
# `clear_cache()` invalidates it for /reload_env.
_cache: Dict[str, str] = {}
_warned_legacy: set[str] = set()


def _read_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("failed reading prompt file %s: %s", path, e)
        return None


def load_prompt(name: str, *, fallback: str = "") -> str:
    """Resolve the named prompt.

    `name` is lowercase, no extension (e.g. "behaviour"). See module
    docstring for the resolution order.
    """
    if name in _cache:
        return _cache[name]

    # 1. Legacy env var (warn once, then keep using it).
    env_var = _LEGACY_ENV.get(name)
    if env_var:
        env_value = os.getenv(env_var, "")
        if env_value:
            if name not in _warned_legacy:
                logger.info(
                    "Prompt '%s' loaded from legacy env var %s. "
                    "Future versions will read it from prompts/%s.txt instead — "
                    "you can migrate by moving the value into that file and "
                    "removing %s from your .env-stable.",
                    name,
                    env_var,
                    name,
                    env_var,
                )
                _warned_legacy.add(name)
            _cache[name] = env_value
            return env_value

    # 2. User-customised file (gitignored).
    custom = _read_file(PROMPTS_DIR / f"{name}.txt")
    if custom is not None:
        _cache[name] = custom
        return custom

    # 3. Shipped default.
    default = _read_file(PROMPTS_DIR / f"{name}.default.txt")
    if default is not None:
        _cache[name] = default
        return default

    # 4. Caller-supplied fallback.
    logger.warning(
        "no prompt found for '%s' (env=%s, prompts/%s.txt, prompts/%s.default.txt); "
        "using caller-supplied fallback (%d chars)",
        name,
        env_var or "—",
        name,
        name,
        len(fallback),
    )
    _cache[name] = fallback
    return fallback


def clear_cache() -> None:
    """Invalidate the in-memory cache (for /reload_env)."""
    _cache.clear()
    _warned_legacy.clear()
