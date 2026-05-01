"""Step 8 — Handoff.

Print where things landed, mention `--resume` and `/soupyscan`, and offer
to exec `python run_all.py` so logs stream cleanly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(state: Dict, ui) -> Dict:
    ui.header("Step 8 — Handoff")

    ui.info(f"config: {REPO_ROOT / '.env-stable'}")
    ui.info(f"web panel will be at: http://127.0.0.1:4941")
    ui.info("re-run the wizard later with:  python install.py --resume")
    ui.hr()

    ui.info("First-run reminder:")
    ui.info("  After the bot is running, hop into your server as an owner and run /soupyscan.")
    ui.info("  The first scan can take hours — or days on a busy server. See soupy_database/SETUP.md")
    ui.info("  for tuning (FIRST_SCAN_LOOKBACK_DAYS, SCAN_EXCLUDE_CHANNEL_IDS).")
    ui.hr()

    if state.get("__dry_run__"):
        ui.info("[dry-run] would offer to launch python run_all.py")
        return {"__done__": True}

    if not ui.confirm("Launch `python run_all.py` now?", default=True):
        ui.info("done. start the bot manually whenever you're ready.")
        return {"__done__": True}

    run_all = REPO_ROOT / "run_all.py"
    if not run_all.exists():
        ui.fail(f"could not find {run_all}")
        return {"__done__": True}

    # exec replaces the current process so Ctrl-C goes straight to the bot.
    ui.info("launching...")
    os.execvp(sys.executable, [sys.executable, str(run_all)])
    # Unreachable.
    return {"__done__": True}
