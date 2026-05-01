"""Step 6 — Write `.env-stable`.

Reads `.env-stable.example` as the template and replaces values for the
keys the wizard collected. Comments, ordering, and blank lines are
preserved. If `.env-stable` already exists, copy it to a timestamped
backup and print a unified diff (with secrets masked) before writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .. import env_writer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_PATH = REPO_ROOT / ".env-stable.example"
TARGET_PATH = REPO_ROOT / ".env-stable"

# These are the keys we let the wizard collect. Every value handed to
# env_writer.write must be one of these; everything else stays at its
# .env-stable.example default.
WIZARD_KEYS = [
    "DISCORD_TOKEN",
    "OWNER_IDS",
    "GUILD_ID",
    "CHANNEL_IDS",
    "OPENAI_BASE_URL",
    "LOCAL_CHAT",
    "RAG_EMBEDDING_MODEL",
    "ENABLE_VISION",
    "VISION_MODEL",
    "BLUESKY_HANDLE",
    "BLUESKY_APP_PASSWORD",
    "BLUESKY_AUTO_REPLY",
    "DAILY_POST_ENABLED",
    "DAILY_POST_CHANNELS",
    "SD_SERVER_URL",
    "SD_IMG2IMG_URL",
    "SD_INPAINT_URL",
    "REMOVE_BG_API_URL",
]


def _gather_updates(state: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in WIZARD_KEYS:
        if k in state and state[k] not in (None, ""):
            out[k] = str(state[k])
    return out


def _print_diff(diff_rows: List, ui) -> None:
    if not diff_rows:
        ui.info("(no value changes — existing config matches the wizard answers)")
        return
    ui.info("changes:")
    for key, old, new in diff_rows:
        if old:
            ui.info(f"  {key}: {old}  ->  {new}")
        else:
            ui.info(f"  {key}: (unset)  ->  {new}")


def run(state: Dict, ui) -> Dict:
    ui.header("Step 6 — Write .env-stable")

    if not EXAMPLE_PATH.exists():
        ui.fail(f"{EXAMPLE_PATH} not found — repo is incomplete")
        raise SystemExit(2)

    updates = _gather_updates(state)
    existing = env_writer.parse_existing(TARGET_PATH)
    diff_rows = env_writer.diff(existing, updates)
    _print_diff(diff_rows, ui)

    dry_run = bool(state.get("__dry_run__"))
    if dry_run:
        ui.info(f"[dry-run] would write {TARGET_PATH}")
        return {"__config_written__": False}

    if TARGET_PATH.exists() and not ui.confirm(
        f"Overwrite {TARGET_PATH.name} (a timestamped .bak will be saved)?",
        default=True,
    ):
        ui.warn("skipping config write")
        return {"__config_written__": False}

    backup_path = env_writer.write(EXAMPLE_PATH, TARGET_PATH, updates, backup=True)
    ui.ok(f"wrote {TARGET_PATH}")
    if backup_path:
        ui.info(f"  backup: {backup_path.name}")

    return {"__config_written__": True}
