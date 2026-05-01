"""Render `.env-stable` from the example template + user-supplied updates.

Preserves comments, blank lines, and ordering from `.env-stable.example`.
Only the values for keys we know about are rewritten; everything else
(quoted multi-line `BEHAVIOUR`, all the `WEB_COLOR_*` defaults, etc.) is
copied through verbatim.

Mirrors the behaviour of `web/services/env_store.py` but stripped down —
the installer doesn't need that module's full quote-handling because we
only ever rewrite simple, single-line values in the wizard. Multi-line
prompts stay untouched.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .state import is_secret_key


_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _line_key(line: str) -> Optional[str]:
    """Return the env key on a line, or None if the line is a comment / blank
    / inside a quoted multi-line block. We treat any line whose content
    before `=` looks like a normal env key as a candidate, but we only
    rewrite lines whose value is simple (not the start of a multi-line
    quoted string)."""
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        return None
    m = _ASSIGN_RE.match(line)
    if not m:
        return None
    return m.group(1)


def _value_needs_quotes(value: str) -> bool:
    if value == "":
        return False
    if any(ch.isspace() for ch in value):
        return True
    return any(ch in value for ch in ('#', '"', "'", "\\"))


def _format_value(value: str) -> str:
    """Render a value for the right-hand side of KEY=...

    Single-line only. The wizard never collects multi-line values, so we
    just escape what we need to and quote when the bare form would break."""
    if _value_needs_quotes(value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return value


def _is_simple_assignment(line: str) -> bool:
    """A line is rewritable if `KEY=value` fits on it.

    We bail out on lines whose value starts with a quote but doesn't close
    on the same line (multi-line BEHAVIOUR, BEHAVIOUR_SEARCH, 9BALL, FANCY,
    RANDOMPROMPT, SD_NEGATIVE_PROMPT, etc. fit this case)."""
    m = _ASSIGN_RE.match(line)
    if not m:
        return False
    val = m.group(2).strip()
    if not val:
        return True
    first = val[0]
    if first in ('"', "'"):
        # Closes on same line?
        if len(val) >= 2 and val.endswith(first):
            return True
        return False
    return True


def render(example_path: Path, updates: Dict[str, str]) -> str:
    """Return the rewritten `.env-stable` content as a string."""
    if not example_path.exists():
        raise FileNotFoundError(example_path)
    src = example_path.read_text(encoding="utf-8")
    lines = src.splitlines()

    remaining = dict(updates)
    out: List[str] = []

    in_multiline = False
    multiline_quote = ""

    for line in lines:
        if in_multiline:
            out.append(line)
            if line.rstrip().endswith(multiline_quote):
                in_multiline = False
                multiline_quote = ""
            continue

        key = _line_key(line)
        if key is None:
            out.append(line)
            continue

        if not _is_simple_assignment(line):
            # Start of a multi-line quoted value — keep verbatim.
            out.append(line)
            m = _ASSIGN_RE.match(line)
            if m:
                val = m.group(2).strip()
                if val and val[0] in ('"', "'"):
                    multiline_quote = val[0]
                    in_multiline = True
            continue

        if key in remaining:
            new_val = remaining.pop(key)
            out.append(f"{key}={_format_value(new_val)}")
        else:
            out.append(line)

    if remaining:
        if out and out[-1].strip() != "":
            out.append("")
        out.append("# Added by installer")
        for k, v in remaining.items():
            out.append(f"{k}={_format_value(v)}")

    return "\n".join(out) + "\n"


def write(
    example_path: Path,
    target_path: Path,
    updates: Dict[str, str],
    *,
    backup: bool = True,
) -> Optional[Path]:
    """Write the rendered config to `target_path`. Returns the backup path
    if one was made, else None."""
    rendered = render(example_path, updates)
    backup_path: Optional[Path] = None
    if backup and target_path.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = target_path.with_suffix(target_path.suffix + f".bak.{ts}")
        shutil.copy2(target_path, backup_path)
    target_path.write_text(rendered, encoding="utf-8")
    return backup_path


def parse_existing(path: Path) -> Dict[str, str]:
    """Read an existing env file and return a flat key->value dict.

    Used to compute a diff before overwriting. Multi-line quoted values
    are joined with `\n`."""
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    n = len(raw_lines)
    while i < n:
        line = raw_lines[i]
        m = _ASSIGN_RE.match(line)
        if not m or line.lstrip().startswith("#"):
            i += 1
            continue
        key = m.group(1)
        val = m.group(2).strip()
        if val and val[0] in ('"', "'"):
            quote = val[0]
            content = val[1:]
            if content.endswith(quote) and len(val) >= 2:
                out[key] = content[:-1]
                i += 1
                continue
            # Multi-line — accumulate until close
            acc = [content]
            i += 1
            while i < n:
                seg = raw_lines[i]
                if seg.rstrip().endswith(quote):
                    acc.append(seg.rstrip()[:-1])
                    i += 1
                    break
                acc.append(seg)
                i += 1
            out[key] = "\n".join(acc)
            continue
        out[key] = val
        i += 1
    return out


def diff(
    existing: Dict[str, str],
    updates: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """Return a list of (key, old_masked, new_masked) for keys whose value is
    changing. Secrets are masked as `****`."""
    rows: List[Tuple[str, str, str]] = []
    for k, new in updates.items():
        old = existing.get(k, "")
        if old == new:
            continue
        if is_secret_key(k):
            old_disp = "****" if old else ""
            new_disp = "****" if new else ""
        else:
            old_disp = old
            new_disp = new
        rows.append((k, old_disp, new_disp))
    rows.sort(key=lambda r: r[0])
    return rows
