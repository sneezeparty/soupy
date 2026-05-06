#!/usr/bin/env python3
"""One-shot helper: move legacy prompt env vars into prompts/<name>.txt files.

After M2 the bot resolves prompts in this order:

    env var (legacy)  ->  prompts/<name>.txt  ->  prompts/<name>.default.txt

Existing installs that customised `BEHAVIOUR` / `9BALL` / etc. in
`.env-stable` keep working, but each lookup logs a one-time deprecation
hint. This script does the migration:

  1. Reads the seven legacy keys from `.env-stable`.
  2. For each that's set to a non-empty, non-default value, writes
     `prompts/<name>.txt` with the same content.
  3. Comments out the env-var line (with a `# migrated to prompts/...`
     trailer) so a future re-run doesn't re-process it.
  4. Backs up the old `.env-stable` to `.env-stable.bak.<timestamp>`
     before any edit.

Idempotent: re-running picks up only env vars still set to non-empty
non-default values. Use `--dry-run` first to see what it would do.

Usage:
    python tools/migrate_prompts.py [--env .env-stable] [--prompts prompts/] [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# (env-var name, prompt file stem) — same mapping as soupy_prompts._LEGACY_ENV.
LEGACY_PROMPTS: List[Tuple[str, str]] = [
    ("BEHAVIOUR", "behaviour"),
    ("BEHAVIOUR_SEARCH", "behaviour_search"),
    ("BEHAVIOUR_DAILY_POST", "behaviour_daily_post"),
    ("9BALL", "nineball"),
    ("FANCY", "fancy"),
    ("RANDOMPROMPT", "randomprompt"),
    ("SD_NEGATIVE_PROMPT", "sd_negative_prompt"),
]


def _parse_env_value(raw_lines: List[str], idx: int) -> Tuple[str, int]:
    """Given an env-file line that starts a `KEY="..."` assignment, return
    (decoded value, line-index immediately after the closing quote)."""
    line = raw_lines[idx]
    after_eq = line.split("=", 1)[1]
    if not after_eq:
        return "", idx + 1
    if after_eq[0] not in ('"', "'"):
        # Unquoted — single line, simple value
        return after_eq.strip(), idx + 1
    quote = after_eq[0]
    body = after_eq[1:]
    if body.endswith(quote) and len(body) > 0:
        return _unescape(body[:-1]), idx + 1
    # Multi-line quoted value
    parts = [body]
    j = idx + 1
    while j < len(raw_lines):
        seg = raw_lines[j]
        if seg.endswith(quote):
            parts.append(seg[:-1])
            return _unescape("\n".join(parts)), j + 1
        parts.append(seg)
        j += 1
    return _unescape("\n".join(parts)), j


def _unescape(s: str) -> str:
    """Decode common .env escape sequences (\\n, \\t, \\")."""
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
        .replace("\\'", "'")
        .replace("\\\\", "\\")
    )


def _read_env(path: Path) -> Tuple[List[str], Dict[str, Tuple[str, int, int]]]:
    """Return (raw lines, {key: (value, start_line, end_line_exclusive)}).

    Only captures the seven legacy prompt keys we care about.
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    found: Dict[str, Tuple[str, int, int]] = {}
    interesting = {k for k, _ in LEGACY_PROMPTS}

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            i += 1
            continue
        key = line.split("=", 1)[0].strip()
        if key in interesting:
            value, end = _parse_env_value(raw_lines, i)
            found[key] = (value, i, end)
            i = end
        else:
            i += 1
    return raw_lines, found


def _is_default_or_blank(name: str, value: str) -> bool:
    """Return True if `value` is empty, whitespace, or matches the shipped default."""
    if not value or not value.strip():
        return True
    default_path = REPO_ROOT / "prompts" / f"{name}.default.txt"
    if not default_path.is_file():
        return False
    try:
        default_text = default_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return value.strip() == default_text.strip()


def _comment_out(raw_lines: List[str], start: int, end: int) -> None:
    """Prefix every line in [start, end) with `# `."""
    for k in range(start, end):
        if not raw_lines[k].lstrip().startswith("#"):
            raw_lines[k] = "# " + raw_lines[k]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".env-stable")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "prompts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.env.exists():
        print(f"error: {args.env} not found", file=sys.stderr)
        return 1
    if not args.prompts.is_dir():
        print(f"error: {args.prompts} is not a directory", file=sys.stderr)
        return 1

    raw_lines, found = _read_env(args.env)

    actions: List[Tuple[str, str, str]] = []  # (env_key, prompt_stem, status)

    for env_key, stem in LEGACY_PROMPTS:
        info = found.get(env_key)
        if info is None:
            actions.append((env_key, stem, "absent — nothing to migrate"))
            continue
        value, _start, _end = info
        if _is_default_or_blank(stem, value):
            actions.append((env_key, stem, "blank-or-default — comment out"))
            continue
        target = args.prompts / f"{stem}.txt"
        if target.exists():
            actions.append(
                (env_key, stem, f"prompts/{stem}.txt already exists — leaving env var; please reconcile manually")
            )
            continue
        actions.append((env_key, stem, f"-> prompts/{stem}.txt ({len(value)} chars), comment out env var"))

    print(f"plan ({args.env}):")
    for env_key, _stem, status in actions:
        print(f"  {env_key:<22} {status}")

    if args.dry_run:
        print()
        print("dry-run — nothing changed.")
        return 0

    # Backup
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = args.env.with_suffix(args.env.suffix + f".bak.{ts}")
    shutil.copy2(args.env, backup)
    print(f"\nbacked up {args.env.name} -> {backup.name}")

    # Apply
    new_lines = list(raw_lines)
    wrote_files: List[Path] = []
    commented_keys: List[str] = []

    # Sort so we comment-out from the bottom up (line indices stay valid).
    by_line_desc = sorted(
        ((env_key, stem, found.get(env_key)) for env_key, stem in LEGACY_PROMPTS),
        key=lambda triple: -1 if triple[2] is None else triple[2][1],
        reverse=True,
    )

    for env_key, stem, info in by_line_desc:
        if info is None:
            continue
        value, start, end = info
        if _is_default_or_blank(stem, value):
            _comment_out(new_lines, start, end)
            commented_keys.append(env_key)
            continue
        target = args.prompts / f"{stem}.txt"
        if target.exists():
            continue
        target.write_text(value, encoding="utf-8")
        wrote_files.append(target)
        # Add a trailer line so a future reader knows where it went
        new_lines.insert(end, f"# (migrated to prompts/{stem}.txt by tools/migrate_prompts.py on {ts})")
        _comment_out(new_lines, start, end)  # comment out the original block
        commented_keys.append(env_key)

    args.env.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"updated {args.env.name}")
    for p in wrote_files:
        print(f"  wrote {p.relative_to(REPO_ROOT)}")
    for k in commented_keys:
        print(f"  commented out env var {k}")

    print()
    print("Restart the bot to pick up the new prompt files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
