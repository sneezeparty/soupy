"""Centralised, typed settings.

Single source of truth for everything the bot reads from `.env-stable`.
Loaded once at import, lazily parses each value the first time it's
accessed. Existing `os.getenv()` call sites are intentionally untouched
in this commit — this module is **additive**. New code should prefer
`from soupy_settings import settings; settings.X` over inline
`os.getenv()` calls; old call sites get migrated incrementally in
follow-up PRs.

The reason for laziness (rather than reading every variable in
`__post_init__`): the bot has 200+ env vars, many specific to features
not enabled in every install. Eager reads would either need to handle
every "missing or malformed" case at startup, or would crash an
otherwise-working install on a typo in an unrelated variable.
Per-property reads localise the failure to the call site that needs
the value.

`settings.reload()` re-reads `.env-stable` (called by `/reload_env`).
"""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed parsers
# ---------------------------------------------------------------------------


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        logger.warning("env var %s=%r is not an int; falling back to %d", name, raw, default)
        return default
    # WHY fall back to default (not clamp) on out-of-range: a clamp silently
    # changes the operator's intent; the default is a known-good value and the
    # warning tells them what happened. Consistent with the malformed-value path.
    if minimum is not None and val < minimum:
        logger.warning("env var %s=%d is below minimum %d; falling back to %d", name, val, minimum, default)
        return default
    if maximum is not None and val > maximum:
        logger.warning("env var %s=%d is above maximum %d; falling back to %d", name, val, maximum, default)
        return default
    return val


def _env_float(name: str, default: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        logger.warning("env var %s=%r is not a float; falling back to %f", name, raw, default)
        return default
    if minimum is not None and val < minimum:
        logger.warning("env var %s=%g is below minimum %g; falling back to %g", name, val, minimum, default)
        return default
    if maximum is not None and val > maximum:
        logger.warning("env var %s=%g is above maximum %g; falling back to %g", name, val, maximum, default)
        return default
    return val


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on")


def _env_int_list(name: str, default: Optional[List[int]] = None) -> List[int]:
    raw = os.getenv(name, "")
    if not raw:
        return list(default or [])
    out: List[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError:
            logger.warning("env var %s contains non-int %r; skipping", name, s)
    return out


def _env_str_list(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.getenv(name, "")
    if not raw:
        return list(default or [])
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# The Settings object
# ---------------------------------------------------------------------------


class Settings:
    """Lazy-loaded typed view of `.env-stable`.

    Each property reads its env var on first access and caches the parsed
    value via `functools.cached_property`. Use `settings.reload()` to
    invalidate the cache.
    """

    # ----- Discord -------------------------------------------------------

    @cached_property
    def discord_token(self) -> str:
        return _env_str("DISCORD_TOKEN")

    @cached_property
    def owner_ids(self) -> List[int]:
        return _env_int_list("OWNER_IDS")

    @cached_property
    def guild_id(self) -> Optional[int]:
        raw = _env_str("GUILD_ID")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @cached_property
    def channel_ids(self) -> List[int]:
        return _env_int_list("CHANNEL_IDS")

    @cached_property
    def memory_channel_ids(self) -> List[int]:
        return _env_int_list("MEMORY_CHANNEL_IDS")

    @cached_property
    def scan_exclude_channel_ids(self) -> List[int]:
        return _env_int_list("SCAN_EXCLUDE_CHANNEL_IDS")

    @cached_property
    def trigger_keywords(self) -> List[str]:
        return _env_str_list("SOUPY_TRIGGER_KEYWORDS", default=["soup", "gumbo"])

    @cached_property
    def first_scan_lookback_days(self) -> int:
        return _env_int("FIRST_SCAN_LOOKBACK_DAYS", 365)

    @cached_property
    def random_response_rate(self) -> float:
        return _env_float("RANDOM_RESPONSE_RATE", 0.05, minimum=0.0, maximum=1.0)

    # ----- LLM -----------------------------------------------------------

    @cached_property
    def openai_base_url(self) -> str:
        return _env_str("OPENAI_BASE_URL", "http://localhost:1234/v1")

    @cached_property
    def openai_api_key(self) -> str:
        # OPENAI_API_KEY takes precedence; LOCAL_KEY is the legacy name.
        return _env_str("OPENAI_API_KEY") or _env_str("LOCAL_KEY", "lm-studio")

    @cached_property
    def local_chat(self) -> str:
        return _env_str("LOCAL_CHAT")

    @cached_property
    def chat_temperature(self) -> float:
        return _env_float("CHAT_TEMPERATURE", 0.65)

    @cached_property
    def chat_num_candidates(self) -> int:
        return _env_int("CHAT_NUM_CANDIDATES", 1, minimum=1)

    @cached_property
    def chat_frequency_penalty(self) -> float:
        return _env_float("CHAT_FREQUENCY_PENALTY", 0.6)

    @cached_property
    def chat_presence_penalty(self) -> float:
        return _env_float("CHAT_PRESENCE_PENALTY", 0.3)

    @cached_property
    def max_tokens(self) -> int:
        return _env_int("MAX_TOKENS", 4096, minimum=1)

    # ----- Search --------------------------------------------------------

    @cached_property
    def search_blocked_domains(self) -> List[str]:
        """Extra hosts to filter from /soupysearch results (additive to the
        built-in dictionary blocklist)."""
        return _env_str_list("SEARCH_BLOCKED_DOMAINS")

    @cached_property
    def search_select_temperature(self) -> float:
        return _env_float("SEARCH_SELECT_TEMPERATURE", 0.3)

    @cached_property
    def search_summary_temperature(self) -> float:
        return _env_float("SEARCH_SUMMARY_TEMPERATURE", 0.7)

    @cached_property
    def recent_message_limit(self) -> int:
        return _env_int("RECENT_MESSAGE_LIMIT", 15, minimum=1)

    @cached_property
    def context_window_tokens(self) -> int:
        return _env_int("CONTEXT_WINDOW_TOKENS", 16000, minimum=512)

    # ----- Vision --------------------------------------------------------

    @cached_property
    def enable_vision(self) -> bool:
        return _env_bool("ENABLE_VISION", default=False)

    @cached_property
    def vision_model(self) -> str:
        return _env_str("VISION_MODEL")

    @cached_property
    def vision_temperature(self) -> float:
        return _env_float("VISION_TEMPERATURE", 0.7)

    @cached_property
    def vision_max_tokens(self) -> int:
        return _env_int("VISION_MAX_TOKENS", 300)

    @cached_property
    def vision_prompt(self) -> str:
        return _env_str("VISION_PROMPT", "What is in this image? Describe it concisely.")

    # ----- RAG -----------------------------------------------------------

    @cached_property
    def rag_embedding_model(self) -> str:
        return _env_str("RAG_EMBEDDING_MODEL")

    @cached_property
    def rag_embed_max_concurrent(self) -> int:
        return _env_int("RAG_EMBED_MAX_CONCURRENT", 2, minimum=1)

    @cached_property
    def rag_reindex_interval_hours(self) -> int:
        return _env_int("RAG_REINDEX_INTERVAL_HOURS", 6, minimum=1)

    @cached_property
    def rag_min_sim(self) -> float:
        # Cosine-similarity floor for vector RAG chunks; matches below this are
        # dropped rather than padded into the prompt. 0 disables the floor.
        return _env_float("RAG_MIN_SIM", 0.45, minimum=0.0, maximum=1.0)

    # ----- Stable Diffusion ----------------------------------------------

    @cached_property
    def sd_server_url(self) -> str:
        return _env_str("SD_SERVER_URL")

    @cached_property
    def sd_img2img_url(self) -> str:
        return _env_str("SD_IMG2IMG_URL")

    @cached_property
    def sd_inpaint_url(self) -> str:
        return _env_str("SD_INPAINT_URL")

    @cached_property
    def remove_bg_api_url(self) -> str:
        return _env_str("REMOVE_BG_API_URL")

    @cached_property
    def sd_steps(self) -> int:
        return _env_int("SD_STEPS", 20, minimum=1)

    @cached_property
    def sd_guidance(self) -> float:
        return _env_float("SD_GUIDANCE", 5.0)

    @cached_property
    def sd_default_width(self) -> int:
        return _env_int("SD_DEFAULT_WIDTH", 1024, minimum=64)

    @cached_property
    def sd_default_height(self) -> int:
        return _env_int("SD_DEFAULT_HEIGHT", 1024, minimum=64)

    # ----- Flux (local mflux backend) ------------------------------------
    # A second, *local* image backend running on the Mac via mflux (MLX). Unlike
    # the remote SD server, flux_server.py runs on the same host. Model is chosen
    # by FLUX_MODEL so swapping FLUX.1 schnell -> FLUX.2 klein is config-only.

    @cached_property
    def flux_enabled(self) -> bool:
        return _env_bool("FLUX_ENABLED", default=False)

    @cached_property
    def flux_server_url(self) -> str:
        # Base URL of the local mflux HTTP server (e.g. http://127.0.0.1:4942).
        return _env_str("FLUX_SERVER_URL")

    @cached_property
    def flux_img2img_url(self) -> str:
        # Optional explicit img2img endpoint; cog falls back to {server}/flux_img2img.
        return _env_str("FLUX_IMG2IMG_URL")

    @cached_property
    def flux_model(self) -> str:
        # mflux model name: "schnell" / "dev" (FLUX.1) or a FLUX.2 klein/dev id.
        return _env_str("FLUX_MODEL", "schnell")

    @cached_property
    def flux_quantize(self) -> int:
        # Quantization bits for mflux (4 or 8). 4-bit keeps schnell ~8-12GB.
        return _env_int("FLUX_QUANTIZE", 4)

    @cached_property
    def flux_steps(self) -> int:
        # schnell is step-distilled (2-4 steps); dev/klein want more.
        return _env_int("FLUX_STEPS", 4, minimum=1)

    @cached_property
    def flux_guidance(self) -> float:
        # schnell is guidance-distilled, so 0.0 is correct; dev wants ~3.5.
        return _env_float("FLUX_GUIDANCE", 0.0)

    @cached_property
    def flux_default_strength(self) -> float:
        # Default img2img strength when /flux is given an input image.
        return _env_float("FLUX_DEFAULT_STRENGTH", 0.35, minimum=0.0, maximum=1.0)

    # ----- Flux edit pipeline (FLUX.2-Klein only) -----------------------
    # When enabled, /flux with an attached image routes to flux_server.py's
    # /flux_edit endpoint, which uses Flux2KleinEdit (reference-image-token
    # conditioning) instead of the noise-mix /flux_img2img path. Much better
    # prompt following for "make this anime" / "set background on fire" edits.
    # Costs ~5-8 GB additional RAM on the flux_server host once loaded.

    @cached_property
    def flux_edit_enabled(self) -> bool:
        return _env_bool("FLUX_EDIT_ENABLED", default=False)

    @cached_property
    def flux_edit_url(self) -> str:
        # Optional explicit edit endpoint; cog falls back to {server}/flux_edit.
        return _env_str("FLUX_EDIT_URL")

    @cached_property
    def flux_edit_model(self) -> str:
        # FLUX.2 klein variant for the edit pipeline (4b/9b, base or distilled).
        return _env_str("FLUX_EDIT_MODEL", "flux2-klein-4b")

    @cached_property
    def flux_edit_steps(self) -> int:
        # Distilled klein-edit runs in 4 steps; base variants want more.
        return _env_int("FLUX_EDIT_STEPS", 4, minimum=1)

    @cached_property
    def flux_edit_guidance(self) -> float:
        # Distilled klein-edit requires 1.0; base variants accept >1.0.
        return _env_float("FLUX_EDIT_GUIDANCE", 1.0, minimum=0.0)

    # ----- Bluesky -------------------------------------------------------

    @cached_property
    def bluesky_handle(self) -> str:
        return _env_str("BLUESKY_HANDLE")

    @cached_property
    def bluesky_app_password(self) -> str:
        return _env_str("BLUESKY_APP_PASSWORD")

    @cached_property
    def bluesky_auto_reply(self) -> bool:
        return _env_bool("BLUESKY_AUTO_REPLY", default=False)

    @cached_property
    def bluesky_replies_min(self) -> int:
        return _env_int("BLUESKY_REPLIES_MIN", 4)

    @cached_property
    def bluesky_replies_max(self) -> int:
        return _env_int("BLUESKY_REPLIES_MAX", 7)

    @cached_property
    def bluesky_reposts_per_day(self) -> int:
        return _env_int("BLUESKY_REPOSTS_PER_DAY", 1)

    @cached_property
    def bluesky_posts_per_day(self) -> int:
        return _env_int("BLUESKY_POSTS_PER_DAY", 1)

    @cached_property
    def bluesky_min_gap_minutes(self) -> int:
        return _env_int("BLUESKY_MIN_GAP_MINUTES", 45)

    @cached_property
    def bluesky_max_likes_per_day(self) -> int:
        return _env_int("BLUESKY_MAX_LIKES_PER_DAY", 10)

    @cached_property
    def bluesky_max_follows_per_day(self) -> int:
        return _env_int("BLUESKY_MAX_FOLLOWS_PER_DAY", 2)

    @cached_property
    def bluesky_article_freshness_days(self) -> int:
        return _env_int("BLUESKY_ARTICLE_FRESHNESS_DAYS", 14)

    # ----- Daily posts ---------------------------------------------------

    @cached_property
    def daily_post_enabled(self) -> bool:
        return _env_bool("DAILY_POST_ENABLED", default=False)

    @cached_property
    def daily_post_max_age_days(self) -> int:
        return _env_int("DAILY_POST_MAX_AGE_DAYS", 21)

    @cached_property
    def daily_post_channels(self) -> str:
        """Raw JSON string mapping channel-id -> topic hint. Parse at use site."""
        return _env_str("DAILY_POST_CHANNELS", "{}")

    @cached_property
    def daily_post_active_start(self) -> int:
        return _env_int("DAILY_POST_ACTIVE_START", 8, minimum=0, maximum=23)

    @cached_property
    def daily_post_active_end(self) -> int:
        return _env_int("DAILY_POST_ACTIVE_END", 18, minimum=0, maximum=23)

    @cached_property
    def daily_post_interval_hours(self) -> int:
        return _env_int("DAILY_POST_INTERVAL_HOURS", 24, minimum=1)

    @cached_property
    def daily_post_topic_dedup_sim(self) -> float:
        return _env_float("DAILY_POST_TOPIC_DEDUP_SIM", 0.65, minimum=0.0, maximum=1.0)

    @cached_property
    def daily_post_topic_dedup_days(self) -> int:
        return _env_int("DAILY_POST_TOPIC_DEDUP_DAYS", 10)

    @cached_property
    def daily_post_reject_no_date(self) -> bool:
        return _env_bool("DAILY_POST_REJECT_NO_DATE", default=False)

    @cached_property
    def daily_post_fallback_to_top_rated(self) -> bool:
        return _env_bool("DAILY_POST_FALLBACK_TO_TOP_RATED", default=True)

    # ----- Musings -------------------------------------------------------

    # Note: musing inline defaults below intentionally differ from the
    # values in .env-stable.example. The example ships the *recommended*
    # values; the inline defaults match what the bot's existing
    # os.getenv("MUSING_*", "...") calls used so behaviour is preserved
    # for installs that don't set these explicitly.

    @cached_property
    def musing_enabled(self) -> bool:
        return _env_bool("MUSING_ENABLED", default=False)

    @cached_property
    def musing_chance(self) -> float:
        return _env_float("MUSING_CHANCE", 0.10, minimum=0.0, maximum=1.0)

    @cached_property
    def musing_poll_minutes_min(self) -> int:
        return _env_int("MUSING_POLL_MINUTES_MIN", 10, minimum=1)

    @cached_property
    def musing_poll_minutes_max(self) -> int:
        return _env_int("MUSING_POLL_MINUTES_MAX", 20, minimum=1)

    @cached_property
    def musing_channel_id(self) -> Optional[int]:
        raw = _env_str("MUSING_CHANNEL_ID")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    # ----- URL processing ------------------------------------------------

    @cached_property
    def url_fetch_timeout_ms(self) -> int:
        return _env_int("URL_FETCH_TIMEOUT", 15000)

    @cached_property
    def url_max_content_length(self) -> int:
        return _env_int("URL_MAX_CONTENT_LENGTH", 2000)

    @cached_property
    def max_urls_per_message(self) -> int:
        return _env_int("MAX_URLS_PER_MESSAGE", 3)

    @cached_property
    def url_cache_ttl_seconds(self) -> int:
        return _env_int("URL_CACHE_TTL_SECONDS", 3600)

    @cached_property
    def url_include_domain(self) -> bool:
        return _env_bool("URL_INCLUDE_DOMAIN", default=True)

    # ----- Logging -------------------------------------------------------

    @cached_property
    def log_level(self) -> str:
        return _env_str("LOG_LEVEL", "INFO").upper()

    @cached_property
    def soupy_log_max_bytes(self) -> int:
        return _env_int("SOUPY_LOG_MAX_BYTES", 5 * 1024 * 1024)

    @cached_property
    def soupy_log_backup_count(self) -> int:
        return _env_int("SOUPY_LOG_BACKUP_COUNT", 5)

    # ----- Web panel -----------------------------------------------------

    @cached_property
    def web_host(self) -> str:
        return _env_str("SOUPY_WEB_HOST", "0.0.0.0")

    @cached_property
    def web_port(self) -> int:
        return _env_int("SOUPY_WEB_PORT", 4941)

    @cached_property
    def autostart_bot(self) -> bool:
        return _env_bool("SOUPY_AUTOSTART_BOT", default=False)

    @cached_property
    def timezone(self) -> str:
        return _env_str("TIMEZONE", "America/Los_Angeles")

    @cached_property
    def web_control_panel_title(self) -> str:
        return _env_str("WEB_CONTROL_PANEL_TITLE", "Soupy Control")

    @cached_property
    def soupy_db_dir(self) -> str:
        """Optional override for the per-guild database directory.
        Defaults to soupy_database/databases/ at the repo root."""
        return _env_str("SOUPY_DB_DIR")

    @cached_property
    def channel_names_raw(self) -> str:
        """Comma-separated `id:name` pairs (legacy format)."""
        return _env_str("CHANNEL_NAMES")

    @cached_property
    def channel_names_json(self) -> str:
        """Raw JSON object mapping channel id (str) -> name (str)."""
        return _env_str("CHANNEL_NAMES_JSON")

    # ----- Self-knowledge ------------------------------------------------

    @cached_property
    def self_md_enabled(self) -> bool:
        return _env_bool("SELF_MD_ENABLED", default=True)

    @cached_property
    def self_md_anchor_max_chars(self) -> int:
        return _env_int("SELF_MD_ANCHOR_MAX_CHARS", 600)

    # ----- Lifecycle -----------------------------------------------------

    def reload(self) -> None:
        """Re-read `.env-stable` and clear the cache.

        Called by `/reload_env`. Note: this clears every cached_property
        on this object so the next access re-reads from the (now-updated)
        environment. We don't load `.env-stable` here directly; the
        caller is expected to have already called dotenv.load_dotenv()
        before calling settings.reload().
        """
        cls = type(self)
        # Drop cached_property values from __dict__
        for name in list(self.__dict__):
            descriptor = getattr(cls, name, None)
            if isinstance(descriptor, cached_property):
                del self.__dict__[name]


# Module-level singleton. Import as: `from soupy_settings import settings`.
settings = Settings()


# ---------------------------------------------------------------------------
# Shared OpenAI client
# ---------------------------------------------------------------------------


def openai_client():
    """Return a configured OpenAI SDK client pointed at the LM Studio (or
    other OpenAI-compatible) server. Uses settings.openai_base_url and
    settings.openai_api_key.

    Each caller gets its own instance — the OpenAI SDK's client is cheap
    to construct and isn't async-safe to share across event loops in
    every version. Future work: bake retry / circuit-breaker policy here
    so every cog gets the same robustness for free.
    """
    from openai import OpenAI

    return OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
