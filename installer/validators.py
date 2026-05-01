"""Live validators for user-supplied config.

Each validator returns a `Result` so callers can render a short message
without re-running the network call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import http


SNOWFLAKE_RE = re.compile(r"^\d{17,20}$")


@dataclass
class Result:
    ok: bool
    message: str
    data: Any = None


def is_snowflake(value: str) -> bool:
    return bool(SNOWFLAKE_RE.match(value.strip()))


def parse_snowflake_list(value: str) -> List[str]:
    """Split on commas, trim whitespace, validate each entry. Raises ValueError on first bad id."""
    out = []
    for part in value.split(","):
        s = part.strip()
        if not s:
            continue
        if not is_snowflake(s):
            raise ValueError(f"{s!r} is not a Discord snowflake (17-20 digits)")
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def discord_token(token: str) -> Result:
    """Verify a bot token by hitting GET /users/@me on the Discord API."""
    if not token or token.strip() in ("", "your_discord_bot_token_here"):
        return Result(False, "token is blank or still the placeholder")
    url = "https://discord.com/api/v10/users/@me"
    try:
        status, body = http.get_json(
            url,
            headers={"Authorization": f"Bot {token.strip()}", "User-Agent": "soupy-installer"},
            timeout=10.0,
        )
    except http.HttpError as e:
        return Result(False, f"network error: {e}")

    if status == 200 and isinstance(body, dict) and body.get("id"):
        return Result(
            True,
            f"authenticated as {body.get('username', '?')} (id {body['id']})",
            data=body,
        )
    if status == 401:
        return Result(False, "401 unauthorised — token is invalid or revoked")
    return Result(False, f"unexpected response (HTTP {status}): {body!r}")


# ---------------------------------------------------------------------------
# LM Studio (OpenAI-compatible)
# ---------------------------------------------------------------------------

def lm_studio_models(base_url: str) -> Result:
    """GET {base}/models. Returns the list of model ids on success."""
    base = base_url.rstrip("/")
    url = f"{base}/models"
    try:
        status, body = http.get_json(url, timeout=10.0)
    except http.HttpError as e:
        return Result(False, f"could not reach {url}: {e}")
    if status != 200:
        return Result(False, f"HTTP {status} from {url}: {body!r}")
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return Result(False, f"unexpected response shape from {url}")
    ids: List[str] = []
    for entry in body["data"]:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            ids.append(entry["id"])
    if not ids:
        return Result(False, f"no models loaded at {url} — load chat + embedding models in LM Studio")
    return Result(True, f"{len(ids)} model(s) loaded", data=ids)


# ---------------------------------------------------------------------------
# Stable Diffusion backend
# ---------------------------------------------------------------------------

def sd_backend(base_url: str) -> Result:
    """Ping the SD backend's /health endpoint (sd-api/sd_api.py exposes it)."""
    base = base_url.rstrip("/")
    # /health is the canonical readiness probe; fall back to / if absent.
    for suffix in ("/health", "/"):
        url = f"{base}{suffix}"
        try:
            status, body = http.get_json(url, timeout=10.0)
        except http.HttpError as e:
            last_err = str(e)
            continue
        if status == 200:
            model = ""
            if isinstance(body, dict):
                model = body.get("model") or body.get("message") or ""
            return Result(True, f"reachable at {url}" + (f" ({model})" if model else ""), data=body)
        last_err = f"HTTP {status}"
    return Result(False, f"could not reach SD backend: {last_err}")


# ---------------------------------------------------------------------------
# Helpers used at prompt time, not networked
# ---------------------------------------------------------------------------

def url_looks_valid(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")
