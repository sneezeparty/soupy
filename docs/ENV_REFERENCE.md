# Environment Variable Reference

Every variable in `.env-stable.example`, grouped by section the way they appear in that file. Each entry shows the variable name, the default from the example, and a one-line description. Most can be edited from the web panel's Environment Editor; a few are launcher / web-panel specific and are noted at the bottom.

## Web Control Panel

- **`WEB_CONTROL_PANEL_TITLE`** — `"Soupy Universal Control Panel"` — Browser tab and heading text for the control panel.

### Color scheme

- **`WEB_COLOR_PAGE_BG`** — `#1e1010` — Page background.
- **`WEB_COLOR_CARD_BG`** — `#1a2332` — Card background.
- **`WEB_COLOR_CARD_BORDER`** — `#374151` — Card border.
- **`WEB_COLOR_TEXT_PRIMARY`** — `#e5e7eb` — Primary text color.
- **`WEB_COLOR_TEXT_SECONDARY`** — `#9ca3af` — Secondary text color.
- **`WEB_COLOR_TEXT_MUTED`** — `#6b7280` — Muted text color.
- **`WEB_COLOR_TAB_INACTIVE_BG`** — `#e5e7eb` — Inactive tab background.
- **`WEB_COLOR_TAB_INACTIVE_TEXT`** — `#6b7280` — Inactive tab text color.
- **`WEB_COLOR_TAB_INACTIVE_BORDER`** — `#d1d5db` — Inactive tab border.
- **`WEB_COLOR_TAB_ACTIVE_BG`** — `#1f2937` — Active tab background.
- **`WEB_COLOR_TAB_ACTIVE_TEXT`** — `#f9fafb` — Active tab text color.
- **`WEB_COLOR_TAB_ACTIVE_BORDER`** — `#374151` — Active tab border.
- **`WEB_COLOR_TAB_CONTENT_BG`** — `#1f2937` — Tab content area background.
- **`WEB_COLOR_TAB_CONTENT_TEXT`** — `#e5e7eb` — Tab content text color.
- **`WEB_COLOR_TAB_CONTENT_BORDER`** — `#374151` — Tab content area border.
- **`WEB_COLOR_CONSOLE_BG`** — `#0b1020` — Console drawer background.
- **`WEB_COLOR_CONSOLE_TEXT`** — `#c8d1f3` — Console drawer text color.
- **`WEB_COLOR_CONSOLE_BORDER`** — `#272b42` — Console drawer border.
- **`WEB_COLOR_STATUS_RUNNING`** — `#2ecc71` — Color for "running" status indicator.
- **`WEB_COLOR_STATUS_STOPPED`** — `#e74c3c` — Color for "stopped" status indicator.
- **`WEB_COLOR_ENV_FIELD_BG`** — `#ffffff` — Env editor field background.
- **`WEB_COLOR_ENV_FIELD_TEXT`** — `#111827` — Env editor field text color.
- **`WEB_COLOR_ENV_FIELD_BORDER`** — `#e5e7eb` — Env editor field border.
- **`WEB_COLOR_ENV_POPOVER_BG`** — `#ffffff` — Env editor tooltip popover background.
- **`WEB_COLOR_ENV_POPOVER_TEXT`** — `#111827` — Env editor tooltip popover text color.
- **`WEB_COLOR_ENV_POPOVER_BORDER`** — `#e5e7eb` — Env editor tooltip popover border.
- **`WEB_COLOR_ENV_TAB_ACTIVE_BG`** — `#111827` — Env editor active-tab background.
- **`WEB_COLOR_ENV_TAB_ACTIVE_TEXT`** — `#ffffff` — Env editor active-tab text color.

## Discord Bot Configuration

- **`DISCORD_TOKEN`** — `your_discord_bot_token_here` — Discord bot token from the Developer Portal. Required.
- **`OWNER_IDS`** — `000000000000000000` — Comma-separated Discord user IDs allowed to run owner-only commands.
- **`CHANNEL_IDS`** — *(empty)* — Comma-separated channel IDs Soupy actively listens in and chats freely.
- **`GUILD_ID`** — *(empty)* — Primary Discord server ID; enables fast (per-guild) slash-command sync.
- **`SPECIAL_GUILD_ID`** — *(commented out)* — Optional secondary guild ID for special handling.
- **`MEMORY_CHANNEL_IDS`** — *(empty)* — Comma-separated channel IDs whose history is included in memory/RAG.
- **`SCAN_EXCLUDE_CHANNEL_IDS`** — *(empty)* — Comma-separated channel IDs to skip during `/soupyscan`.
- **`FIRST_SCAN_LOOKBACK_DAYS`** — `365` — How far back the very first `/soupyscan` reaches; subsequent scans are always incremental.

## LLM Configuration

- **`OPENAI_BASE_URL`** — `http://localhost:1234/v1` — LM Studio (or other OpenAI-compatible) base URL.
- **`LOCAL_KEY`** — `lm-studio` — Auto-mapped to `OPENAI_API_KEY` if that's unset; usually fine for local servers.
- **`LOCAL_CHAT`** — `google/gemma-3-27b` — Exact chat-model id loaded in LM Studio.
- **`AVAILABLE_MODELS`** — `"google/gemma-3-27b, qwen/qwen2.5-vl-7b, text-embedding-nomic-embed-text-v1.5"` — Comma-separated list of models offered in the panel's model dropdown.

## Vision Model Configuration

- **`ENABLE_VISION`** — `false` — Master switch for image-attachment understanding via a vision-capable LLM.
- **`VISION_MODEL`** — `qwen/qwen2.5-vl-7b` — Exact id of the vision-capable model loaded in LM Studio.
- **`VISION_PROMPT`** — `"What is in this image? Describe it concisely."` — Prompt sent alongside the image.
- **`VISION_TEMPERATURE`** — `0.7` — Sampling temperature for vision calls.
- **`VISION_MAX_TOKENS`** — `300` — Max tokens for the vision description.

## Logging

- **`LOG_LEVEL`** — `INFO` — Console log level (DEBUG / INFO / WARNING / ERROR). The file at `logs/soupy.log` always captures DEBUG regardless.

## Chat Behavior & Responses

- **`CHAT_TEMPERATURE`** — `0.65` — Sampling temperature for chat replies.
- **`CHAT_NUM_CANDIDATES`** — `1` — Number of candidate replies generated per turn (1 disables the candidate-and-judge pass).
- **`CHAT_FREQUENCY_PENALTY`** — `0.6` — Frequency penalty passed to LM Studio (range -2.0 to 2.0).
- **`CHAT_PRESENCE_PENALTY`** — `0.3` — Presence penalty passed to LM Studio (range -2.0 to 2.0).
- **`SOUPY_TRIGGER_KEYWORDS`** — `soup,gumbo` — Comma-separated literal keywords that trigger replies outside `CHANNEL_IDS`. Case-insensitive; `soup` also matches `soupy`.
- **`RANDOM_RESPONSE_RATE`** — `0.05` — Probability (0–1) that Soupy spontaneously replies to a non-triggering message. Read per message so it can be tuned live.
- **`MAX_TOKENS`** — `4096` — LLM response length limit.
- **`RECENT_MESSAGE_LIMIT`** — `15` — How many recent messages to pull for chat context.
- **`UPDATE_INTERVAL_MINUTES`** — `61` — Periodic background-update interval, in minutes.
- **`TIMEZONE`** — `America/Los_Angeles` — Timezone for scheduled loops and timestamps.
- **`BEHAVIOUR`** — *(long string)* — Main personality system prompt; see [CUSTOMIZATION.md](CUSTOMIZATION.md).
- **`BEHAVIOUR_SEARCH`** — *(long string)* — Voice used for `/soupysearch` summaries.
- **`MUSING_ENABLED`** — `true` — Enable the once-daily "thinking out loud" musing.
- **`MUSING_CHANNEL_ID`** — *(empty)* — Channel ID where Soupy posts musings.
- **`MUSING_HOUR_MIN`** — `6` — Earliest local hour the daily musing can fire (0–23, inclusive).
- **`MUSING_HOUR_MAX`** — `20` — Latest local hour the daily musing can fire (1–24, exclusive; 24 means midnight).
- **`9BALL`** — *(long string)* — Response style for `/9ball`.

## Stable Diffusion Configuration

- **`SD_SERVER_URL`** — `http://your-sd-host:8000/` — Base URL of the Stable Diffusion FastAPI backend.
- **`SD_IMG2IMG_URL`** — `http://your-sd-host:8000/sd_img2img` — img2img endpoint URL.
- **`SD_INPAINT_URL`** — `http://your-sd-host:8000/sd_inpaint` — Inpaint endpoint URL.
- **`REMOVE_BG_API_URL`** — `http://your-sd-host:8000/remove_background` — Background-removal endpoint URL.
- **`SD_STEPS`** — `20` — Diffusion steps per image.
- **`SD_GUIDANCE`** — `5` — Classifier-free guidance scale.
- **`SD_NEGATIVE_PROMPT`** — *(long default)* — Default negative prompt for image generation.
- **`SD_DEFAULT_WIDTH`** — `1024` — Default image width (must be divisible by 64).
- **`SD_DEFAULT_HEIGHT`** — `1024` — Default image height (must be divisible by 64).
- **`SD_WIDE_WIDTH`** — `1440` — Width preset for wide-aspect generation.
- **`SD_WIDE_HEIGHT`** — `1024` — Height preset for wide-aspect generation.
- **`SD_TALL_WIDTH`** — `1024` — Width preset for tall-aspect generation.
- **`SD_TALL_HEIGHT`** — `1440` — Height preset for tall-aspect generation.

## Flux (Local) Configuration

The `/flux` command's local mflux/MLX backend. Unlike SD (remote GPU host), the
Flux model runs on the same Mac as the bot via `flux_server.py`. The bot only
talks to it over HTTP, exactly like SD.

- **`FLUX_ENABLED`** — `false` — Soft toggle for the Flux feature (the cog also no-ops if `FLUX_SERVER_URL` is unset).
- **`FLUX_SERVER_URL`** — `http://127.0.0.1:4942` — Base URL of the local `flux_server.py` mflux backend.
- **`FLUX_IMG2IMG_URL`** — *(empty)* — Optional explicit img2img endpoint; defaults to `{FLUX_SERVER_URL}/flux_img2img`.
- **`FLUX_MODEL`** — `schnell` — mflux model: `schnell`/`dev` (FLUX.1), or a FLUX.2 klein/dev id later. Server-side; swapping models needs no code change.
- **`FLUX_QUANTIZE`** — `4` — Quantization bits (4 or 8). 4-bit keeps FLUX.1 schnell ~8-12GB.
- **`FLUX_LOW_RAM`** — `0` — Release text encoders between runs to save RAM (needed for tight memory / FLUX.2).
- **`FLUX_STEPS`** — `4` — Inference steps. schnell is step-distilled (2-4); dev/klein want more.
- **`FLUX_GUIDANCE`** — `0.0` — Guidance scale. schnell is guidance-distilled (0); klein/dev want `1.0+`. `flux_server.py` auto-substitutes `1.0` when this is left at `0.0` on a guided model.
- **`FLUX_DEFAULT_STRENGTH`** — `0.35` — Default img2img strength when `/flux` is given an input image.
- **`FLUX_SERVER_HOST`** — `127.0.0.1` — Bind host for `flux_server.py`.
- **`FLUX_SERVER_PORT`** — `4942` — Bind port for `flux_server.py`.

### Flux Klein-Edit Pipeline (FLUX.2 only)

When enabled, `/flux` with an attached image routes to `flux_server.py`'s
`/flux_edit` endpoint, which uses `Flux2KleinEdit`. Reference-image latent
tokens are concatenated with the noise latents inside the transformer — much
stronger prompt-driven editing than the noise-mix `/flux_img2img` path.
Strength is intentionally ignored here. First request loads klein weights
(~5 GB at `FLUX_QUANTIZE=4`, ~15 GB unquantized) and adds ~5-8 GB peak RAM
alongside the `FLUX_MODEL` already loaded.

- **`FLUX_EDIT_ENABLED`** — `false` — Auto-route `/flux <prompt> <image>` to `/flux_edit` instead of `/flux_img2img`.
- **`FLUX_EDIT_URL`** — *(empty)* — Optional explicit edit endpoint; defaults to `{FLUX_SERVER_URL}/flux_edit`.
- **`FLUX_EDIT_MODEL`** — `flux2-klein-4b` — FLUX.2 klein variant for editing (`flux2-klein-4b`/`9b`, plus `*-base-*` non-distilled variants).
- **`FLUX_EDIT_STEPS`** — `4` — Distilled klein-edit is step-distilled; base variants want more.
- **`FLUX_EDIT_GUIDANCE`** — `1.0` — Distilled FLUX.2-klein requires `1.0`; base variants accept `>1.0`.

## Rate Limiting & Permissions

- **`MAX_INTERACTIONS_PER_MINUTE`** — `4` — Per-user rate limit for interactive commands.
- **`LIMIT_EXCEPTION_ROLES`** — `mod,owner` — Comma-separated role names exempt from rate limiting.

## Image Generation Prompts

- **`FANCY`** — *(long string)* — Template used by the Fancy button to expand a short prompt into a detailed CLIP-style prompt.
- **`RANDOMPROMPT`** — *(long string)* — Template used by Random buttons to generate a full image prompt from keywords.

## Content Categories

- **`OVERALL_THEMES`** — `soupy_themes.txt` — Path to the themes/setting keyword file.
- **`CHARACTER_CONCEPTS`** — `soupy_characters.txt` — Path to the character/subject keyword file.
- **`ARTISTIC_RENDERING_STYLES`** — `soupy_styles.txt` — Path to the art-style keyword file.
- **`SD_KEYWORDS`** — `sd_keywords.txt` — Path to the SD-specific quality/modifier keyword file.

## URL Processing

- **`URL_FETCH_TIMEOUT`** — `15000` — Timeout for URL fetches, in milliseconds.
- **`MAX_URLS_PER_MESSAGE`** — `3` — Max number of URLs Soupy will fetch per message.
- **`URL_MAX_CONTENT_LENGTH`** — `2000` — Max characters of fetched content used in chat context.
- **`URL_INCLUDE_DOMAIN`** — `true` — Whether to include the domain alongside the summary.

## Outpaint Configuration

- **`OUTPAINT_USE_CANNY`** — `false` — Use Canny ControlNet during outpaint.
- **`OUTPAINT_USE_DEPTH`** — `false` — Use Depth ControlNet during outpaint.
- **`OUTPAINT_CONTROL_WEIGHT`** — `0.8` — ControlNet weight for outpaint.
- **`OUTPAINT_HARMONIZE_STRENGTH`** — `0.00` — Strength of post-process harmonization between original and outpainted regions.
- **`OUTPAINT_USE_HIST_MATCH`** — `false` — Apply histogram matching to outpainted regions.
- **`OUTPAINT_LIGHTNESS_FIX`** — `false` — Apply a lightness/contrast correction pass.

## Optional Settings

- **`LORA_PATH`** — *(commented out)* — Path to a LoRA `.safetensors` file (set on the SD host, not the bot host).
- **`LORA_WEIGHT`** — *(commented out, default 1.0)* — Strength of the loaded LoRA.

## Temperature Settings

- **`RANDOM_PROMPT_TEMPERATURE`** — `0.65` — Temperature for the random-prompt builder.
- **`NINE_BALL_TEMPERATURE`** — `0.8` — Temperature for `/9ball` responses.
- **`FANCY_PROMPT_TEMPERATURE`** — `0.75` — Temperature for the Fancy-button prompt rewrite.
- **`SEARCH_SELECT_TEMPERATURE`** — `0.3` — Temperature for `/soupysearch` article selection.
- **`SEARCH_SUMMARY_TEMPERATURE`** — `0.5` — Temperature for `/soupysearch` final-answer summarization.
- **`SEARCH_BLOCKED_DOMAINS`** — *(commented out)* — Extra domains to block from `/soupysearch` results, on top of a built-in list of dictionary/glossary sites.
- **`OPENAI_API_KEY`** — *(commented out)* — Set only if your LLM endpoint enforces auth.
- **`FANCY_MAX_TOKENS`** — `300` — Max tokens for the Fancy-button rewrite.

## RAG (Retrieval-Augmented Generation)

- **`RAG_EMBEDDING_MODEL`** — `text-embedding-qwen3-embedding-0.6b` — Exact embedding-model id loaded in LM Studio. Rebuild the RAG index after changing this.
- **`RAG_REINDEX_INTERVAL_HOURS`** — `6` — How often the background loop consolidates and re-embeds RAG chunks.
- **`RAG_EMBED_MAX_CONCURRENT`** — `2` — Max concurrent embedding requests to LM Studio (a global semaphore — keeps reindex from starving live chat RAG).
- **`RAG_EMBED_BATCH_SIZE`** — `8` — Number of texts sent per embedding HTTP call.
- **`RAG_MAX_CHARS_PER_LINE`** — `800` — Per-message character cap when building a RAG chunk.
- **`RAG_LOG_FULL_CONTENT`** — `0` — Deep RAG debug: dump full retrieved chunk/profile content. Off by default.
- **`RAG_LOG_VERBOSE`** — `1` — Per-hit RAG details on INFO. Set to `0` for one-line retrieval summaries.

## Self-Context (`self.md`)

- **`SELF_MD_ENABLED`** — `true` — Enable the running self-knowledge document and reflection cycle.
- **`SELF_MD_ANCHOR_MAX_CHARS`** — `600` — Size cap on the always-on identity slug.
- **`SELF_MD_ANCHOR_TEMPERATURE`** — `0.5` — Distillation temperature for the anchor.
- **`SELF_MD_ANCHOR_MAX_TOKENS`** — `400` — Token cap for the anchor distillation step.
- **`SELF_MD_ANCHOR_FALLBACK_CHARS`** — `600` — Character cap on the truncated-core fallback used before the first reflection completes.

## Daily Article Posts (Discord)

- **`DAILY_POST_CHANNELS`** — `"{}"` — JSON map: channel ID → friendly slug used internally. Empty = no posts.
- **`DAILY_POST_ENABLED`** — `false` — Master switch for the daily-post loop.
- **`DAILY_POST_ACTIVE_START`** — `8` — Earliest hour (0–23) a daily post may fire.
- **`DAILY_POST_ACTIVE_END`** — `18` — Latest hour (0–23) a daily post may fire.
- **`DAILY_POST_INTERVAL_HOURS`** — `24` — Spacing target between posts, in hours.
- **`DAILY_POST_MAX_AGE_DAYS`** — `21` — Reject articles older than this when picking dailies.
- **`DAILY_POST_REJECT_NO_DATE`** — `false` — When `true`, reject any article whose publish date can't be extracted.
- **`DAILY_POST_FALLBACK_TO_TOP_RATED`** — `true` — When the LLM judge says SKIP at the final pick, fall back to the top-rated candidate instead of giving up.
- **`DAILY_POST_TOPIC_DEDUP_SIM`** — *(commented out)* — Topic-dedup cosine-similarity threshold (0–1).
- **`DAILY_POST_TOPIC_DEDUP_DAYS`** — *(commented out)* — Days of recent topics to dedup against.

## Bluesky Integration

- **`BLUESKY_HANDLE`** — `yourname.bsky.social` — Your Bluesky handle.
- **`BLUESKY_APP_PASSWORD`** — `xxxx-xxxx-xxxx-xxxx` — App password from Settings → App Passwords (not your main password).
- **`BLUESKY_AUTO_REPLY`** — `false` — Master switch for the autonomous Bluesky loop.
- **`BLUESKY_REPLIES_MIN`** — `4` — Lower bound on replies per day.
- **`BLUESKY_REPLIES_MAX`** — `7` — Upper bound on replies per day.
- **`BLUESKY_REPOSTS_PER_DAY`** — `1` — Quote-posts per day.
- **`BLUESKY_POSTS_PER_DAY`** — `1` — Original article posts per day.
- **`BLUESKY_MIN_GAP_MINUTES`** — `45` — Minimum gap, in minutes, between any two autonomous Bluesky actions. *Previously hardcoded.*
- **`BLUESKY_MAX_LIKES_PER_DAY`** — `10` — Daily cap on likes. *Previously hardcoded.*
- **`BLUESKY_MAX_FOLLOWS_PER_DAY`** — `2` — Daily cap on follows. *Previously hardcoded.*
- **`BLUESKY_ARTICLE_FRESHNESS_DAYS`** — `14` — Reject articles older than this when picking originals. *Previously hardcoded.*

## Context Budgeting

- **`CONTEXT_WINDOW_TOKENS`** — `16000` — Total context-window budget, in tokens.

## Misc Runtime Knobs

- **`URL_CACHE_TTL_SECONDS`** — `3600` — How long URL summaries are cached before being re-fetched. *Previously hardcoded.*
- **`SOUPY_LOG_MAX_BYTES`** — `5242880` — Bot log file rotation threshold, in bytes. *Previously hardcoded.*
- **`SOUPY_LOG_BACKUP_COUNT`** — `5` — How many old rotated log files to keep. *Previously hardcoded.*
- **`SOUPY_DB_DIR`** — *(empty → `soupy_database/databases/`)* — Override the directory holding per-guild SQLite databases. Both the bot and web panel must agree on this.
- **`CHANNEL_NAMES`** — *(empty)* — Optional channel-ID→name map for the stats panel as `"123:general,456:random"`. Falls back to names recorded in the archive.
- **`CHANNEL_NAMES_JSON`** — *(empty)* — Same mapping as JSON, e.g. `'{"123":"general"}'`. Takes precedence over `CHANNEL_NAMES`.

## Web Panel / Launcher (set outside `.env-stable`)

These aren't in `.env-stable.example` but affect the panel; set them in your shell or systemd unit:

- **`SOUPY_WEB_HOST`** — `0.0.0.0` — Web panel bind host.
- **`SOUPY_WEB_PORT`** — `4941` — Web panel bind port.
- **`SOUPY_AUTOSTART_BOT`** — Set to `1` by `run_all.py`; `0` to launch the web panel without auto-spawning the bot.
- **`SOUPY_BOT_ENTRY`** — Override the bot-script entrypoint resolution.
