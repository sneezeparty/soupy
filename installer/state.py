"""Resumable wizard state.

Persists answers to `.install-state.json` so a crashed or aborted wizard can
pick up where it left off. Secret values (tokens, passwords, API keys) are
NEVER written to disk — they're replaced with a placeholder string and
re-prompted on the next run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

PLACEHOLDER = "<set in earlier step>"

# Substring match — any env-style key containing one of these is treated
# as a secret. Match is case-insensitive on the upper-cased key.
SECRET_HINTS = ("TOKEN", "PASSWORD", "SECRET", "API_KEY")


def is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(hint in upper for hint in SECRET_HINTS)


def _scrub(data: Any) -> Any:
    """Recursively replace secret values with the placeholder."""
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(k, str) and is_secret_key(k) and v not in (None, "", PLACEHOLDER):
                out[k] = PLACEHOLDER
            else:
                out[k] = _scrub(v)
        return out
    if isinstance(data, list):
        return [_scrub(item) for item in data]
    return data


def save(path: Path, data: Dict[str, Any]) -> None:
    scrubbed = _scrub(data)
    path.write_text(json.dumps(scrubbed, indent=2, sort_keys=True), encoding="utf-8")


def load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def clear(path: Path) -> None:
    if path.exists():
        path.unlink()


def needs_reprompt(state: Dict[str, Any], keys: Iterable[str]) -> list[str]:
    """Return the subset of `keys` whose value in state is missing or a placeholder."""
    out = []
    for k in keys:
        v = state.get(k)
        if v in (None, "", PLACEHOLDER):
            out.append(k)
    return out
