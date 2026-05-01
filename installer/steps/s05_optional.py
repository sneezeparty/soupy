"""Step 5 — Optional integrations.

Run the sub-steps the user opted into in step 1: Bluesky credentials,
remote Stable Diffusion endpoint, autonomous daily-post channel map.
"""

from __future__ import annotations

import json
from typing import Dict

from .. import validators


def _bluesky(state: Dict, ui) -> Dict:
    ui.step("Bluesky")
    ui.info("Generate an app password at https://bsky.app/settings/app-passwords (NOT your main password).")
    handle = ui.prompt(
        "Bluesky handle (e.g. yourname.bsky.social)",
        default=state.get("BLUESKY_HANDLE", ""),
    )
    password = ui.prompt(
        "Bluesky app password",
        secret=True,
        default=state.get("BLUESKY_APP_PASSWORD", "") or None,
    )
    auto_reply = ui.confirm(
        "Enable the autonomous Bluesky loop now? (you can flip this later in the web panel)",
        default=False,
    )
    return {
        "BLUESKY_HANDLE": handle,
        "BLUESKY_APP_PASSWORD": password,
        "BLUESKY_AUTO_REPLY": "true" if auto_reply else "false",
    }


def _remote_sd(state: Dict, ui) -> Dict:
    ui.step("Stable Diffusion (remote)")
    while True:
        url = ui.prompt(
            "SD backend base URL (e.g. http://10.0.0.4:8000)",
            default=state.get("SD_SERVER_URL", "http://your-sd-host:8000"),
        )
        if not validators.url_looks_valid(url):
            ui.fail("URL must start with http:// or https://")
            continue
        result = validators.sd_backend(url)
        if result.ok:
            ui.ok(result.message)
            break
        ui.fail(result.message)
        choice = ui.retry_skip_abort("How do you want to handle this?")
        if choice == "retry":
            continue
        if choice == "skip":
            ui.warn("storing SD URL without verification")
            break
        raise SystemExit("aborted at SD backend probe")

    base = url.rstrip("/")
    return {
        "SD_SERVER_URL": base + "/",
        "SD_IMG2IMG_URL": f"{base}/sd_img2img",
        "SD_INPAINT_URL": f"{base}/sd_inpaint",
        "REMOVE_BG_API_URL": f"{base}/remove_background",
    }


def _local_sd(ui) -> Dict:
    ui.step("Stable Diffusion (local)")
    ui.info("SD will run on this machine via .venv-sd. Default port is 8000.")
    base = "http://localhost:8000"
    return {
        "SD_SERVER_URL": base + "/",
        "SD_IMG2IMG_URL": f"{base}/sd_img2img",
        "SD_INPAINT_URL": f"{base}/sd_inpaint",
        "REMOVE_BG_API_URL": f"{base}/remove_background",
    }


def _daily_posts(state: Dict, ui) -> Dict:
    ui.step("Daily posts")
    ui.info("Map channel IDs to short topic hints. Soupy uses these to bias article picking.")
    ui.info("Press enter on an empty channel id to finish.")
    channel_map: Dict[str, str] = {}
    # If state has a previous map, seed from there but still let the user re-enter.
    existing_raw = state.get("DAILY_POST_CHANNELS", "")
    if existing_raw:
        try:
            channel_map = dict(json.loads(existing_raw))
        except (ValueError, TypeError):
            channel_map = {}

    while True:
        cid = ui.prompt("channel id (blank to stop)", allow_blank=True)
        if not cid:
            break
        if not validators.is_snowflake(cid):
            ui.fail(f"{cid!r} is not a Discord snowflake; skipping")
            continue
        hint = ui.prompt(f"topic hint for {cid}", allow_blank=True)
        channel_map[cid] = hint

    return {
        "DAILY_POST_ENABLED": "true",
        "DAILY_POST_CHANNELS": json.dumps(channel_map),
    }


def run(state: Dict, ui) -> Dict:
    ui.header("Step 5 — Optional integrations")
    out: Dict = {}

    image_gen_mode = state.get("image_gen_mode", "none")
    if image_gen_mode == "remote":
        out.update(_remote_sd(state, ui))
    elif image_gen_mode in ("local_cuda", "local_mps"):
        out.update(_local_sd(ui))
    else:
        ui.info("image generation: none (skipping SD configuration)")

    if state.get("bluesky"):
        out.update(_bluesky(state, ui))
    else:
        ui.info("bluesky: disabled (skipping)")

    if state.get("daily_posts"):
        out.update(_daily_posts(state, ui))
    else:
        ui.info("daily posts: disabled (skipping)")

    return out
