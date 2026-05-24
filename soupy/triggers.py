"""Pure functions used by the chat-respond decision.

Extracted from `soupy_remastered_stablediffusion.py` so they're testable
in isolation. The bot-aware predicate `should_bot_respond_to_message`
stays in the main bot file for now because it touches the `bot`
instance and the user-stats side effect.

The function bodies here are byte-identical to the originals; only
their location changed. Imports + DEFAULT_TRIGGER_KEYWORDS are
preserved so removing them from the main file doesn't break anything.
"""

from __future__ import annotations

import os
import random
import re

DEFAULT_TRIGGER_KEYWORDS = ["soup", "gumbo"]


def get_trigger_keywords() -> list[str]:
    """Return literal chat keywords that trigger a reply outside allowed channels."""
    # WHY: env is read on every call (not cached) so the dashboard's
    # SOUPY_TRIGGER_KEYWORDS edit takes effect without a bot restart.
    # The function is called per-message, but is cheap (one os.getenv +
    # a comma split) so the missed cache is not worth re-architecting.
    raw = os.getenv("SOUPY_TRIGGER_KEYWORDS", ",".join(DEFAULT_TRIGGER_KEYWORDS))
    keywords = [kw.strip() for kw in raw.split(",") if kw.strip()]
    return keywords or DEFAULT_TRIGGER_KEYWORDS


def message_contains_trigger_keyword(content: str) -> bool:
    """Case-insensitive literal keyword match. Preserves old soup -> soupy behavior."""
    return any(re.search(re.escape(keyword), content or "", re.IGNORECASE) for keyword in get_trigger_keywords())


def should_randomly_respond(probability=None) -> bool:
    """
    Returns True with the given probability (default from RANDOM_RESPONSE_RATE env, or 5%).
    """
    if probability is None:
        probability = float(os.getenv("RANDOM_RESPONSE_RATE", "0.05"))
    return random.random() < probability
