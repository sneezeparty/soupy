"""Step 7 — Verification.

Re-run the live validations against whatever's now in state and print a
green/red summary. Failures offer retry / skip / abort. We don't write
back to `.env-stable` here — this is a final smoke test.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .. import validators
from ..state import PLACEHOLDER


def _verify_discord(state: Dict, ui) -> Tuple[bool, str]:
    token = state.get("DISCORD_TOKEN", "")
    if not token or token == PLACEHOLDER:
        return False, "no token in state"
    result = validators.discord_token(token)
    return result.ok, result.message


def _verify_lm_studio(state: Dict, ui) -> List[Tuple[str, bool, str]]:
    out: List[Tuple[str, bool, str]] = []
    base = state.get("OPENAI_BASE_URL", "")
    if not base:
        out.append(("LM Studio reachable", False, "no base URL in state"))
        return out
    result = validators.lm_studio_models(base)
    out.append(("LM Studio reachable", result.ok, result.message))
    if not result.ok:
        return out
    loaded = set(result.data or [])
    chat = state.get("LOCAL_CHAT", "")
    embed = state.get("RAG_EMBEDDING_MODEL", "")
    out.append(("chat model loaded", chat in loaded, chat or "(unset)"))
    out.append(("embedding model loaded", embed in loaded, embed or "(unset)"))
    if state.get("ENABLE_VISION") == "true":
        vision = state.get("VISION_MODEL", "")
        out.append(("vision model loaded", vision in loaded, vision or "(unset)"))
    return out


def _verify_sd(state: Dict, ui) -> List[Tuple[str, bool, str]]:
    mode = state.get("image_gen_mode", "none")
    if mode in ("none", "local_flux"):
        return []
    base = state.get("SD_SERVER_URL", "")
    if not base:
        return [("SD backend reachable", False, "no SD URL in state")]
    result = validators.sd_backend(base)
    if result.ok or mode == "remote":
        return [("SD backend reachable", result.ok, result.message)]
    # A local backend is started by hand once the wizard is done (the handoff
    # step prints the command), so "not up yet" is expected here. Same
    # treatment as Flux below.
    return [("SD backend reachable", True, f"not running yet — the next step shows how to start it ({result.message})")]


def _verify_flux(state: Dict, ui) -> List[Tuple[str, bool, str]]:
    if state.get("image_gen_mode", "none") != "local_flux":
        return []
    base = state.get("FLUX_SERVER_URL", "")
    if not base:
        return [("Flux backend configured", False, "no FLUX_SERVER_URL in state")]
    # flux_server.py is launched separately and may well not be running yet
    # during install — soft-probe and downgrade a failure to a warning row
    # rather than a hard fail.
    result = validators.sd_backend(base)
    if result.ok:
        return [("Flux backend reachable", True, result.message)]
    if str(state.get("FLUX_SERVER_AUTOSTART") or "").strip().lower() == "true":
        return [("Flux backend reachable", True, f"not running yet — the web panel starts it ({result.message})")]
    return [("Flux backend reachable", True, f"not running yet — launch with `python flux_server.py` ({result.message})")]


def _render_row(label: str, ok: bool, msg: str, ui) -> None:
    if ok:
        ui.ok(f"{label}: {msg}")
    else:
        ui.fail(f"{label}: {msg}")


def run(state: Dict, ui) -> Dict:
    ui.header("Step 7 — Verification")

    rows: List[Tuple[str, bool, str]] = []

    ok, msg = _verify_discord(state, ui)
    rows.append(("discord token", ok, msg))

    rows.extend(_verify_lm_studio(state, ui))
    rows.extend(_verify_sd(state, ui))
    rows.extend(_verify_flux(state, ui))

    for label, ok, msg in rows:
        _render_row(label, ok, msg, ui)

    failed = [(label, msg) for label, ok, msg in rows if not ok]
    if not failed:
        ui.ok("all verifications passed")
        return {"__verified__": True}

    ui.warn(f"{len(failed)} verification(s) failed")
    choice = ui.retry_skip_abort("How do you want to handle this?")
    if choice == "abort":
        raise SystemExit("aborted at verification step")
    if choice == "retry":
        return run(state, ui)
    return {"__verified__": False}
