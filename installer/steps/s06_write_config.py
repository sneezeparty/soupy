"""Step 6 — Write `.env-stable`.

Takes a template and replaces values for the keys the wizard collected.
Comments, ordering, and blank lines are preserved. On a first install the
template is `.env-stable.example`. When `.env-stable` already exists it is
its own template, so every setting the wizard doesn't ask about (the
persona prompts, colours, tuning knobs, keys added by hand) carries over;
the file is copied to a timestamped backup and the changes are listed
(with secrets masked) before writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .. import env_writer
from ..state import PLACEHOLDER

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_PATH = REPO_ROOT / ".env-stable.example"
TARGET_PATH = REPO_ROOT / ".env-stable"

# These are the keys we let the wizard collect. Every value handed to
# env_writer.write must be one of these; everything else keeps the value
# it has in the template.
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
    "FLUX_ENABLED",
    "FLUX_SERVER_URL",
    "FLUX_SERVER_HOST",
    "FLUX_SERVER_PORT",
    "FLUX_SERVER_AUTOSTART",
    "FLUX_EDIT_ENABLED",
    "AVAILABLE_MODELS",
]


def _switched_off(state: Dict) -> Dict[str, str]:
    """An explicit "false" for each feature declined in step 1.

    The steps only ever write "true". That was enough while every write
    started from the example, where these default to off. Over an existing
    config, saying no has to turn the feature off rather than leave it as
    it was."""
    off: Dict[str, str] = {}
    if not state.get("vision"):
        off["ENABLE_VISION"] = "false"
    if not state.get("daily_posts"):
        off["DAILY_POST_ENABLED"] = "false"
    if state.get("image_gen_mode", "none") != "local_flux":
        off["FLUX_ENABLED"] = "false"
    if not state.get("bluesky"):
        off["BLUESKY_AUTO_REPLY"] = "false"
    return off


def _gather_updates(state: Dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in WIZARD_KEYS:
        # PLACEHOLDER is what a secret looks like after a trip through the
        # state file. It is never a value to write.
        if k in state and state[k] not in (None, "", PLACEHOLDER):
            out[k] = str(state[k])
    out.update(_switched_off(state))
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

    template = TARGET_PATH if TARGET_PATH.exists() else EXAMPLE_PATH
    if not template.exists():
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

    if TARGET_PATH.exists() and not diff_rows:
        # Nothing to write, so no backup to add to the pile either.
        ui.ok(f"{TARGET_PATH.name} left as it is")
        return {"__config_written__": True}

    if TARGET_PATH.exists():
        ui.info(f"Only the values listed above change. Everything else in {TARGET_PATH.name} is kept.")
        # The label is what a --non-interactive answers file keys on; keep it as is.
        if not ui.confirm(
            f"Overwrite {TARGET_PATH.name} (a timestamped .bak will be saved)?",
            default=True,
        ):
            ui.warn("skipping config write")
            return {"__config_written__": False}

    backup_path = env_writer.write(template, TARGET_PATH, updates, backup=True)
    ui.ok(f"wrote {TARGET_PATH}")
    if backup_path:
        ui.info(f"  backup: {backup_path.name}")

    return {"__config_written__": True}
