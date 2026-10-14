"""Step 8 — Handoff.

Print where things landed, how to change the answers later, and the
`/soupyscan` reminder, then offer to exec `run_all.py` so logs stream cleanly.

The launch uses `.venv`'s interpreter. The wizard itself runs on whatever
Python the user started it with, which on a fresh clone has none of the
bot's dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

from ..ui import wizard_command
from .s02_environment import BOT_VENV, SD_ENTRYPOINTS, SD_VENV, venv_python

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _relative(path: Path) -> str:
    """`path` as typed from the repo root (falls back to the full path)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run(state: Dict, ui) -> Dict:
    ui.header("Step 8 — Handoff")

    bot_python = venv_python(BOT_VENV)
    launch_cmd = f"{_relative(bot_python)} run_all.py"

    ui.info(f"config: {REPO_ROOT / '.env-stable'}")
    ui.info("web panel will be at: http://127.0.0.1:4941")
    ui.info(f"change these answers later with:  {wizard_command()}")
    ui.hr()

    mode = state.get("image_gen_mode")
    if mode == "local_flux":
        ui.info("Flux backend:")
        if str(state.get("FLUX_SERVER_AUTOSTART") or "").strip().lower() == "true":
            ui.info("  /flux runs against a separate local server, flux_server.py. The web panel starts it,")
            ui.info("  and has Start / Stop / Restart for it on the Overview tab. Its log is logs/flux_server.log.")
        else:
            ui.info("  /flux runs against a separate local server. Launch it in another terminal:")
            ui.info(f"      {_relative(bot_python)} flux_server.py")
            ui.info("  (Or set FLUX_SERVER_AUTOSTART=true and the web panel starts it.)")
        ui.info("  (First request downloads weights; mflux holds the model resident after.)")
        ui.hr()
    elif mode in SD_ENTRYPOINTS:
        ui.info("Stable Diffusion backend:")
        ui.info("  /sd runs against a separate local server on port 8000. Launch it in another terminal:")
        ui.info("      cd sd-api")
        ui.info(f"      {os.path.join(os.pardir, _relative(venv_python(SD_VENV)))} {SD_ENTRYPOINTS[mode]}")
        ui.hr()

    ui.info("First-run reminder:")
    ui.info("  After the bot is running, hop into your server as an owner and run /soupyscan.")
    ui.info("  The first scan can take hours — or days on a busy server. See soupy_database/SETUP.md")
    ui.info("  for tuning (FIRST_SCAN_LOOKBACK_DAYS, SCAN_EXCLUDE_CHANNEL_IDS).")
    ui.hr()

    if state.get("__dry_run__"):
        ui.info(f"[dry-run] would offer to launch {launch_cmd}")
        return {"__done__": True}

    # The label is what a --non-interactive answers file keys on; keep it as is.
    if not ui.confirm("Launch `python run_all.py` now?", default=True):
        ui.info(f"done. start the bot whenever you're ready with:  {launch_cmd}")
        return {"__done__": True}

    run_all = REPO_ROOT / "run_all.py"
    if not run_all.exists():
        ui.fail(f"could not find {run_all}")
        return {"__done__": True}
    if not bot_python.exists():
        ui.fail(f"{BOT_VENV.name} has no Python in it — run the wizard again so step 2 can build it")
        return {"__done__": True}

    # exec replaces the current process so Ctrl-C goes straight to the bot.
    # run_all.py starts uvicorn on `web.app:app`, which only resolves from the
    # repo root, wherever the wizard happened to be started from.
    ui.info(f"launching {launch_cmd} ...")
    sys.stdout.flush()  # exec drops whatever is still buffered
    os.chdir(REPO_ROOT)
    os.execvp(str(bot_python), [str(bot_python), str(run_all)])
    # Unreachable.
    return {"__done__": True}
