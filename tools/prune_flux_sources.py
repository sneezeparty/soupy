#!/usr/bin/env python3
"""Remove the copies of archived pictures from media/flux_sources.

The Flux cog used to save every result twice: once in `media/images` (the
archive) and once in `media/flux_sources`, for the 🖼️ button to start from.
The button now reads the archive, so the second copy is dead weight. Uploaded
source images are still saved in `media/flux_sources`, and those stay.

A file is removed only when both hold:

  1. It is byte-for-byte the same as a file in `media/images`.
  2. No row in `media/images/index.jsonl` names it as a `source_image`
     (a result made from an upload that happens to match an archived picture
     still needs its source for Remix).

Without `--delete` nothing is removed; the script prints what it would do.

Usage:
    python tools/prune_flux_sources.py [--media media/] [--delete]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def referenced_sources(index_path: Path) -> Set[str]:
    """File names in flux_sources that an archive row names as its source image."""
    names: Set[str] = set()
    if not index_path.exists():
        return names
    for line in index_path.read_text(encoding="utf-8").splitlines():
        try:
            ref = json.loads(line).get("source_image") or ""
        except (ValueError, AttributeError):
            continue
        folder, _, name = str(ref).partition("/")
        if folder == "flux_sources" and name:
            names.add(name)
    return names


def find_duplicates(media: Path) -> List[Path]:
    """Files in flux_sources that are safe to remove, oldest name first."""
    sources = media / "flux_sources"
    images = media / "images"
    if not sources.is_dir() or not images.is_dir():
        return []
    keep = referenced_sources(images / "index.jsonl")

    # Hashing 4 GB of archive is slow, so only archive files of a size some
    # candidate has are read.
    by_size: Dict[int, List[Path]] = defaultdict(list)
    for path in sources.iterdir():
        if path.is_file() and path.name not in keep:
            by_size[path.stat().st_size].append(path)
    archived: Set[str] = set()
    for path in images.iterdir():
        if path.is_file() and path.stat().st_size in by_size:
            archived.add(_digest(path))

    return sorted(
        (path for group in by_size.values() for path in group if _digest(path) in archived),
        key=lambda path: path.name,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--media", type=Path, default=REPO_ROOT / "media", help="the media folder (default: media/)")
    parser.add_argument("--delete", action="store_true", help="remove the files; without it, only report")
    args = parser.parse_args(argv)

    sources = args.media / "flux_sources"
    if not sources.is_dir():
        print(f"{sources} does not exist; nothing to do.")
        return 0
    everything = [p for p in sources.iterdir() if p.is_file()]
    duplicates = find_duplicates(args.media)
    size = sum(p.stat().st_size for p in duplicates)
    print(
        f"{len(duplicates)} of {len(everything)} files in {sources} are copies of archived pictures "
        f"({size / 1024**2:.0f} MB)."
    )
    if not duplicates:
        return 0
    if not args.delete:
        print("Nothing removed. Run again with --delete to remove them.")
        return 0
    for path in duplicates:
        path.unlink()
    print(f"Removed {len(duplicates)} files. {len(everything) - len(duplicates)} uploaded sources kept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
