"""Step 3 — Discord app.

Print the manual portal checklist, open the browser, then prompt for token
+ IDs. Token is verified live against `GET /users/@me`. Snowflake IDs are
regex-validated.
"""

from __future__ import annotations

import webbrowser
from typing import Dict

from .. import validators


PORTAL_URL = "https://discord.com/developers/applications"


def _open_portal(ui) -> None:
    ui.step(f"opening {PORTAL_URL}")
    try:
        webbrowser.open(PORTAL_URL)
    except Exception:  # noqa: BLE001 — headless / no browser is fine
        ui.warn("could not open browser; navigate there manually")


def _checklist(ui) -> None:
    ui.info("Discord setup checklist (do these in the developer portal):")
    ui.info("  1. New Application -> name it whatever you like")
    ui.info("  2. Bot tab -> Add Bot, then Reset Token (copy it; you'll paste it below)")
    ui.info("  3. Privileged Gateway Intents -> enable Message Content Intent (required)")
    ui.info("                                  -> enable Server Members Intent (recommended)")
    ui.info("  4. OAuth2 -> URL Generator: scope `bot` + `applications.commands`")
    ui.info("     Permissions: Read/Send Messages, Embed Links, Attach Files,")
    ui.info("                  Use Slash Commands, Read Message History")
    ui.info("  5. Open the generated URL and invite the bot to your server")
    ui.info("  6. Note your own Discord user id (Settings -> Advanced -> Developer Mode,")
    ui.info("     then right-click your name -> Copy User ID)")


def _prompt_token(state: Dict, ui) -> str:
    existing = state.get("DISCORD_TOKEN", "")
    while True:
        if existing and existing != "<set in earlier step>":
            token = ui.prompt("Discord bot token", default=existing, secret=True)
        else:
            token = ui.prompt("Discord bot token", secret=True)
        result = validators.discord_token(token)
        if result.ok:
            ui.ok(result.message)
            return token
        ui.fail(result.message)
        choice = ui.retry_skip_abort("How do you want to handle this?")
        if choice == "retry":
            existing = ""
            continue
        if choice == "skip":
            ui.warn("storing token without verification — fix it before launching the bot")
            return token
        raise SystemExit("aborted at Discord token validation")


def _prompt_snowflake_list(label: str, default: str, ui, *, allow_blank: bool) -> str:
    while True:
        raw = ui.prompt(label, default=default or None, allow_blank=allow_blank)
        if not raw and allow_blank:
            return ""
        try:
            ids = validators.parse_snowflake_list(raw)
        except ValueError as e:
            ui.fail(str(e))
            continue
        return ",".join(ids)


def _prompt_single_snowflake(label: str, default: str, ui, *, allow_blank: bool) -> str:
    while True:
        raw = ui.prompt(label, default=default or None, allow_blank=allow_blank)
        if not raw and allow_blank:
            return ""
        if not validators.is_snowflake(raw):
            ui.fail(f"{raw!r} is not a Discord snowflake (17-20 digits)")
            continue
        return raw


def run(state: Dict, ui) -> Dict:
    ui.header("Step 3 — Discord app")
    _checklist(ui)
    ui.hr()
    if ui.confirm("Open the Discord developer portal in your browser now?", default=True):
        _open_portal(ui)
    ui.hr()

    token = _prompt_token(state, ui)
    owner_ids = _prompt_snowflake_list(
        "Owner Discord user id(s), comma-separated",
        state.get("OWNER_IDS", ""),
        ui,
        allow_blank=False,
    )
    guild_id = _prompt_single_snowflake(
        "Primary Discord server (guild) id",
        state.get("GUILD_ID", ""),
        ui,
        allow_blank=False,
    )
    channel_ids = _prompt_snowflake_list(
        "Channel id(s) Soupy listens in, comma-separated (blank for none)",
        state.get("CHANNEL_IDS", ""),
        ui,
        allow_blank=True,
    )

    return {
        "DISCORD_TOKEN": token,
        "OWNER_IDS": owner_ids,
        "GUILD_ID": guild_id,
        "CHANNEL_IDS": channel_ids,
    }
