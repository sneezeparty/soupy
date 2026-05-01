"""Step 7 — Verification.

Re-run the live validations against whatever's now in state and print a
green/red summary. Failures offer retry / skip / abort. We don't write
back to `.env-stable` here — this is a final smoke test.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .. import validators


def _verify_discord(state: Dict, ui) -> Tuple[bool, str]:
    token = state.get("DISCORD_TOKEN", "")
    if not token:
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
    out.append(
        ("chat model loaded", chat in loaded, chat or "(unset)")
    )
    out.append(
        ("embedding model loaded", embed in loaded, embed or "(unset)")
    )
    if state.get("ENABLE_VISION") == "true":
        vision = state.get("VISION_MODEL", "")
        out.append(("vision model loaded", vision in loaded, vision or "(unset)"))
    return out


def _verify_sd(state: Dict, ui) -> List[Tuple[str, bool, str]]:
    if state.get("image_gen_mode", "none") == "none":
        return []
    base = state.get("SD_SERVER_URL", "")
    if not base:
        return [("SD backend reachable", False, "no SD URL in state")]
    result = validators.sd_backend(base)
    return [("SD backend reachable", result.ok, result.message)]


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
