from __future__ import annotations

import codecs
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class EnvLine:
    raw: str
    key: Optional[str] = None
    value: Optional[str] = None


def parse_env(path: Path) -> Tuple[List[EnvLine], Dict[str, str]]:
    lines: List[EnvLine] = []
    kv: Dict[str, str] = {}
    if not path.exists():
        return [], {}
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    n = len(raw_lines)
    while i < n:
        raw = raw_lines[i]
        stripped = raw.strip()
        # Comments/blank
        if not stripped or stripped.startswith("#") or "=" not in raw:
            lines.append(EnvLine(raw=raw))
            i += 1
            continue
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.lstrip()
        # Handle quoted (possibly multi-line) values
        if val.startswith('"') or val.startswith("'"):
            quote = val[0]
            content = val[1:]
            # Check if closes on same line
            if content.endswith(quote):
                content = content[:-1]
                # Decode escape sequences (\\n -> \n, \\" -> ", etc.)
                # Use codecs.decode for more reliable escape sequence handling
                try:
                    content = codecs.decode(content, 'unicode_escape')
                except (UnicodeDecodeError, ValueError):
                    # Fallback: manual replacement for common cases
                    content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
                lines.append(EnvLine(raw=raw, key=key, value=content))
                kv[key] = content
                i += 1
                continue
            # Accumulate subsequent lines until closing quote encountered
            i += 1
            acc = [content]
            closed = False
            while i < n:
                seg = raw_lines[i]
                if seg.endswith(quote):
                    acc.append(seg[:-1])
                    closed = True
                    i += 1
                    break
                else:
                    acc.append(seg)
                    i += 1
            combined = "\n".join(acc)
            # Decode escape sequences for multiline values too
            try:
                combined = codecs.decode(combined, 'unicode_escape')
            except (UnicodeDecodeError, ValueError):
                # Fallback: manual replacement for common cases
                combined = combined.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
            lines.append(EnvLine(raw=raw, key=key, value=combined))
            kv[key] = combined
            # If not closed, we still record combined; remaining lines already advanced
            continue
        else:
            unquoted = val.strip()
            lines.append(EnvLine(raw=raw, key=key, value=unquoted))
            kv[key] = unquoted
            i += 1
            continue
    return lines, kv


def _needs_quotes(value: str) -> bool:
    if value == "":
        return False
    # Quote if spaces or special characters likely to break parsing
    return ("\n" in value) or any(ch.isspace() for ch in value) or any(ch in value for ch in ['#', '"', "'", '\\'])


def _escape_for_env(value: str) -> str:
    """Escape special characters for .env file format, preserving newlines as \\n"""
    # Replace actual newlines with \n escape sequences for single-line quoted format
    # The parser will convert \n back to actual newlines when reading
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def write_env(path: Path, updates: Dict[str, str]) -> None:
    # Backup
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak." + datetime.now().strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(path, backup)

    lines, current = parse_env(path)

    # Track which keys updated/created
    remaining = dict(updates)

    out_lines: List[str] = []
    for item in lines:
        if item.key is None:
            out_lines.append(item.raw)
            continue
        if item.key in remaining:
            new_val = remaining.pop(item.key)
            if new_val == item.value:
                # Value unchanged — preserve original raw line exactly
                out_lines.append(item.raw)
            elif _needs_quotes(new_val):
                # Escape newlines and other special chars for single-line quoted format
                escaped = _escape_for_env(new_val)
                out_lines.append(f"{item.key}=\"{escaped}\"")
            else:
                out_lines.append(f"{item.key}={new_val}")
        else:
            # Key not in updates — preserve original raw line exactly
            out_lines.append(item.raw)

    # Append any new keys
    if remaining:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append("# Added by web UI")
        for k, v in remaining.items():
            if _needs_quotes(v):
                escaped = _escape_for_env(v)
                out_lines.append(f"{k}=\"{escaped}\"")
            else:
                out_lines.append(f"{k}={v}")

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


