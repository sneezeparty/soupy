"""Shared test setup.

Importing ``soupy_remastered_stablediffusion`` runs module-level guards that
``raise ValueError`` when required env vars are missing (DISCORD_TOKEN is the
exception — it's checked but the test env sets a dummy), and it logs heavily at
import time. pytest imports every ``conftest.py`` before any test module, so
setting these here means individual test files can ``import
soupy_remastered_stablediffusion`` without repeating the preamble.

``setdefault`` is used throughout so a real shell environment (if you happen to
run the suite with a populated ``.env``) is never clobbered.
"""

from __future__ import annotations

import logging
import os

# Required by module-level guards in the bot — harmless localhost stand-ins.
os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("REMOVE_BG_API_URL", "http://localhost:8000/remove_background")
os.environ.setdefault("SD_SERVER_URL", "http://localhost:8000/")
os.environ.setdefault("SD_IMG2IMG_URL", "http://localhost:8000/sd_img2img")
os.environ.setdefault("SD_INPAINT_URL", "http://localhost:8000/sd_inpaint")

# Used by the chat path; harmless defaults so nothing reads an unset value.
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1234/v1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOCAL_CHAT", "test-model")

# The bot module is chatty at import and per-call. Keep test output readable.
logging.getLogger("soupy_prompts").setLevel(logging.ERROR)
logging.getLogger("soupy_remastered_stablediffusion").setLevel(logging.ERROR)
