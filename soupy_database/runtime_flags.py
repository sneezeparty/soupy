"""
Runtime toggles shared by the web UI and the Discord bot (no restart required).
Stored as JSON next to the project under data/runtime_flags.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "rag_enabled": False,
    "disabled_commands": [],  # list of command names (e.g. ["sd", "soupysearch"])
}

# Cached read: invalidate via mtime
_cache: Dict[str, Any] = {"mtime": 0.0, "data": None}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def flags_path() -> Path:
    return _project_root() / "data" / "runtime_flags.json"


def read_runtime_flags() -> Dict[str, Any]:
    path = flags_path()
    merged = dict(_DEFAULTS)
    if not path.exists():
        return merged
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k, v in raw.items():
                merged[k] = v
    except json.JSONDecodeError as e:
        logger.warning("runtime_flags: invalid JSON (%s), using defaults", e)
    return merged


def write_runtime_flags(updates: Dict[str, Any]) -> Dict[str, Any]:
    path = flags_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = read_runtime_flags()
    for k, v in updates.items():
        current[k] = v
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _cache["mtime"] = 0.0
    _cache["data"] = None
    return current


def is_command_disabled(command_name: str) -> bool:
    """Check if a slash command is disabled via the dashboard toggle."""
    path = flags_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False  # all commands enabled by default
    if _cache["data"] is not None and mtime <= _cache["mtime"]:
        disabled = _cache["data"].get("disabled_commands", [])
    else:
        data = read_runtime_flags()
        _cache["data"] = data
        _cache["mtime"] = mtime
        disabled = data.get("disabled_commands", [])
    return command_name.lower() in [c.lower() for c in disabled]


def is_rag_enabled() -> bool:
    """Read rag_enabled from disk; uses mtime cache to reduce I/O per message."""
    path = flags_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return bool(_DEFAULTS["rag_enabled"])
    if _cache["data"] is not None and mtime <= _cache["mtime"]:
        return bool(_cache["data"].get("rag_enabled", False))
    data = read_runtime_flags()
    _cache["data"] = data
    _cache["mtime"] = mtime
    return bool(data.get("rag_enabled", False))
