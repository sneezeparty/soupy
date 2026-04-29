# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Soupy** is a fully-local Discord bot combining AI chat, image generation, web search, autonomous Discord posts, and an autonomous Bluesky presence. It runs entirely on-premises against LM Studio (or any OpenAI-compatible LLM server) and a Stable Diffusion backend. A FastAPI web control panel manages the bot process, streams logs, and edits configuration.

---

## Running the Project

```bash
source .venv/bin/activate

# Standard launch — web panel spawns the bot as a subprocess
python run_all.py

# Web panel only (set SOUPY_AUTOSTART_BOT=0 first to skip auto-spawning the bot)
python -m uvicorn web.app:app --host 0.0.0.0 --port 4941

# Bot only (skips the web UI; useful for direct stack traces)
python soupy_remastered_stablediffusion.py
```

Web panel binds to `0.0.0.0:4941` by default (override with `SOUPY_WEB_HOST` / `SOUPY_WEB_PORT`).

**Important:** `run_all.py` does not spawn the bot directly. It launches uvicorn; the web app's `startup` event calls `BotRunner.start()` when `SOUPY_AUTOSTART_BOT=1` (which `run_all.py` sets). The bot's stdout/stderr stream through a PTY in `bot_runner.py` to `logs/soupy.log` and the `/ws/logs` WebSocket. To restart the bot after code changes, use the web panel's restart button or `POST /api/bot/restart` — restarting uvicorn alone is not required.

`BotRunner._resolve_entrypoint()` picks the bot script in this order: `$SOUPY_BOT_ENTRY` → `soupy_remastered_stablediffusion.py` → `run_soupy.py` → `soupy_remastered.py`. On every start, `.env-stable` is re-parsed and merged into the child environment (so env edits take effect on bot restart, not web restart).

---

## Architecture

**Three tiers:**

1. **Discord bot** — `soupy_remastered_stablediffusion.py` is the entrypoint and owns chat handling, image generation (`SDQueue`), event handlers, and most slash commands. At `on_ready` it loads five discord.py extensions:
   - `soupy_search` — `/soupysearch`
   - `soupy_imagesearch` — `/soupyimage`
   - `soupy_dailypost` — `/soupypost` + autonomous daily article posts
   - `soupy_musings` — `/soupymuse` + periodic "thinking out loud"
   - `soupy_bluesky` — `/soupysky` + autonomous Bluesky activity

2. **Web control panel** (`web/`) — FastAPI app that manages the bot subprocess (`web/services/bot_runner.py`), streams its log via WebSocket (`web/services/log_stream.py`), reads/writes `.env-stable` (`web/services/env_store.py`), and exposes archive/stats/runtime-flag endpoints.

3. **External services** — LM Studio (chat + embeddings, OpenAI-compatible), Stable Diffusion FastAPI backend (image gen), DuckDuckGo (search), Bluesky AT Protocol (autonomous posts), and per-guild SQLite databases for message archive + RAG.

**Cross-module imports to be aware of:**
- `soupy_dailypost.py` imports `_fetch_og_image` and `_post_url` from `soupy_bluesky` for cross-posting articles to Bluesky.
- `soupy_bluesky.py` imports `_extract_date_from_html` and `_estimate_article_age_days` from `soupy_dailypost` for article freshness checks.
- All cogs share the same `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `LOCAL_CHAT` env vars and instantiate their own `OpenAI` client.

---

## Key Files

| File | Purpose |
|------|---------|
| [soupy_remastered_stablediffusion.py](soupy_remastered_stablediffusion.py) | Main bot — chat, image gen, core slash commands, event loop |
| [soupy_search.py](soupy_search.py) | Cog — DuckDuckGo web search + LLM summary |
| [soupy_imagesearch.py](soupy_imagesearch.py) | Cog — DuckDuckGo image search |
| [soupy_dailypost.py](soupy_dailypost.py) | Cog — autonomous daily article posts to Discord (+ optional Bluesky cross-post) |
| [soupy_musings.py](soupy_musings.py) | Cog — periodic reflective musings in a configured channel |
| [soupy_bluesky.py](soupy_bluesky.py) | Cog — autonomous Bluesky replies, quote-posts, original posts |
| [run_all.py](run_all.py) | Launcher: starts uvicorn with `SOUPY_AUTOSTART_BOT=1` |
| [web/app.py](web/app.py) | FastAPI app — routes, WebSocket, archive/stats endpoints |
| [web/services/bot_runner.py](web/services/bot_runner.py) | Spawns and manages the bot subprocess via PTY |
| [web/services/env_store.py](web/services/env_store.py) | Parses and rewrites `.env-stable` |
| [web/services/log_stream.py](web/services/log_stream.py) | WebSocket fan-out for live log streaming |
| [soupy_database/database.py](soupy_database/database.py) | SQLite schema + per-guild DB operations |
| [soupy_database/rag.py](soupy_database/rag.py) | RAG embeddings and cosine similarity retrieval |
| [soupy_database/user_profiles.py](soupy_database/user_profiles.py) | LLM-generated structured user profiles |
| [soupy_database/self_context.py](soupy_database/self_context.py) | Self-knowledge document reflection cycles |
| [soupy_database/runtime_flags.py](soupy_database/runtime_flags.py) | Shared bot↔web feature toggles |
| [.env-stable](.env-stable) | Master configuration |

---

## Configuration (`.env-stable`)

Critical variables — the bot will not start without `DISCORD_TOKEN`:

| Variable | Description |
|----------|-------------|
| `DISCORD_TOKEN` | Discord bot token |
| `OPENAI_BASE_URL` | LM Studio URL (e.g., `http://localhost:1234/v1` or `http://<lm-studio-host>:1234/v1`) |
| `OPENAI_API_KEY` | API key (can be `lm-studio` for local; `LOCAL_KEY` is auto-mapped if this is empty) |
| `SD_SERVER_URL` | Stable Diffusion endpoint |
| `OWNER_IDS` | Comma-separated Discord user IDs for admin commands |
| `CHANNEL_IDS` | Channels the bot actively monitors |
| `GUILD_ID` | Primary Discord server ID (used for fast slash-command sync) |
| `MAX_TOKENS` | LLM response length limit |
| `RAG_ENABLED` | Enable retrieval-augmented generation |
| `ENABLE_VISION` | Enable image understanding via LM Studio's vision-capable LLM (set `VISION_MODEL`) |
| `DAILY_POST_ENABLED` / `DAILY_POST_CHANNELS` | Autonomous Discord article posts |
| `MUSING_ENABLED` / `MUSING_CHANNEL_ID` / `MUSING_CHANCE` | Autonomous musings |
| `BLUESKY_HANDLE` / `BLUESKY_APP_PASSWORD` / `BLUESKY_AUTO_REPLY` | Bluesky integration |
| `SELF_MD_ENABLED` / `SELF_MD_REFLECT_INTERVAL_HOURS` | Self-knowledge reflection |
| `SOUPY_WEB_HOST` / `SOUPY_WEB_PORT` | Web panel binding (default: `0.0.0.0:4941`) |
| `SOUPY_AUTOSTART_BOT` | Set by `run_all.py`; `0` to launch the web panel without the bot |
| `SOUPY_BOT_ENTRY` | Override bot entrypoint resolution |

There are also 50+ `WEB_COLOR_*` variables for UI customization.

---

## Personality (`BEHAVIOUR`)

Soupy's personality is a single system prompt stored in the `BEHAVIOUR` variable in `.env-stable`. To change personality, edit that string. There is no per-guild or runtime preset selection — the same `BEHAVIOUR` is used everywhere.

Two related prompt variables exist:

| Variable | Used for |
|----------|----------|
| `BEHAVIOUR` | Default persona for chat replies, musings, autonomous interjections |
| `BEHAVIOUR_SEARCH` | Persona used when answering `/soupysearch` queries |
| `9BALL` | Response style for `/9ball` |

**Never edit these prompts via the web UI env editor** — they are long and the editor may truncate them. Edit `.env-stable` directly.

The lookup happens in `get_guild_behaviour()` in [soupy_remastered_stablediffusion.py](soupy_remastered_stablediffusion.py).

---

## Dependencies & Installation

```bash
python -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA 11.8 (required for GPU image generation)
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for the full dependency list. Vision (`ENABLE_VISION=true`) calls LM Studio's vision-capable LLM over the OpenAI-compatible chat endpoint — no extra GPU resources needed on the bot host (LM Studio runs separately).

---

## Discord Commands

| Command | Cog | Description |
|---------|-----|-------------|
| `/sd <prompt>` | main | Generate image via Stable Diffusion |
| `/img2img` | main | Transform an image with a prompt |
| `/inpaint` | main | Inpaint an image with a mask + prompt |
| `/outpaint <prompt> <direction>` | main | Extend image by ~25% |
| `/soupysearch <query>` | soupy_search | Web search + LLM summary |
| `/soupyimage <query>` | soupy_imagesearch | DuckDuckGo image search |
| `/soupypost` | soupy_dailypost | Force daily article post (owner) |
| `/soupymuse` | soupy_musings | Trigger a musing (owner) |
| `/soupysky [action] [url]` | soupy_bluesky | Bluesky reply/repost/post (owner) |
| `/soupyself [action]` | main | View/manage self-knowledge document (owner) |
| `/soupyscan` | main | Archive messages to database (owner) |
| `/soupystats` | main | Server and bot statistics |
| `/status` | main | Service health check |
| `/helpsoupy` | main | Command reference |
| `/whattime <location>` | main | Local time lookup |
| `/weather <location>` | main | Weather lookup |
| `/8ball` / `/9ball` | main | Magic 8-ball (classic / LLM-powered) |
| `/testurl` | main | Test URL extraction |

Image generation posts a 2×2 thumbnail grid (`ThumbnailSelectionView`) with Remix, R-Fancy, R-Keyword, Fancy, Edit, and Outpaint buttons.

---

## Data & Storage

- **Per-guild SQLite DBs**: `soupy_database/databases/guild_<id>.db` — message archive + RAG embeddings
- **Generated images**: `media/images/` (full) + `media/thumbs/` (thumbnails)
- **Message log**: `media/messages.jsonl`
- **User stats**: `user_stats.json`
- **Logs**: `logs/soupy.log` (rotating, 5MB max, 5 backups)

Under `data/`:
- `data/runtime_flags.json` — shared bot↔web feature toggles (RAG enabled, command disables, etc.)
- `data/bot_dashboard.json` — bot-written status the web panel reads
- `data/self_md/guild_<id>.md` + `_core.md` + `_archive.md` — per-guild self-knowledge documents
- `data/musings_archive.jsonl` — musings history
- `data/daily_post_history.json` / `daily_post_schedule.json` — daily-post state
- `data/bluesky_engage_history.json` / `bluesky_schedule.json` — Bluesky engagement state

---

## Key Code Patterns

- **Cog architecture** — five extensions loaded at `on_ready` (see Architecture). Editing a cog requires only a bot restart, not a web-panel restart.
- **Async/await throughout** — all I/O is non-blocking
- **Queue-based image generation** — `SDQueue` prevents backend overload
- **Per-user rate limiting** — 10 searches/minute
- **URL content caching** — 1-hour TTL to avoid repeated fetches
- **Embedding semaphore** — prevents LM Studio request stampedes
- **Graceful degradation** — services checked before operations; user-friendly error messages
- **Web front end** — whenever there is a change to the bot's functionality, check whether the web front-end needs an update too.

Key functions in the main bot:
- `async_chat_completion()` — LLM API calls
- `generate_sd_image()` — image generation pipeline
- `fetch_recent_messages()` — Discord context retrieval
- `build_rag_retrieval_query()` — RAG context assembly
- `process_image_attachment()` — sends Discord image attachments to LM Studio's vision-capable LLM and returns a description
- `get_guild_behaviour()` — reads the active `BEHAVIOUR` prompt from `.env-stable`
- `load_extensions()` — the canonical list of loaded cogs

---

## Keyword Files (Image Generation)

Random image prompt generation uses these text files (one keyword per line):
- [soupy_characters.txt](soupy_characters.txt) — 1000+ character/subject keywords
- [soupy_styles.txt](soupy_styles.txt) — art style keywords
- [soupy_themes.txt](soupy_themes.txt) — theme/setting keywords
- [sd_keywords.txt](sd_keywords.txt) — SD-specific quality/modifier keywords

Roughly 212 quadrillion possible keyword combinations.

---

## Notes & Quirks

- The `.env-stable` file accumulates timestamped backups (`.env-stable.bak.*`) on every web-UI save — safe to clean up old ones.
- The bot supports multi-guild operation; each Discord server gets its own SQLite database and self-knowledge document.
- Discord requires **Message Content Intent** to be enabled in the Discord Developer Portal.
- The production setup runs LM Studio and Stable Diffusion on separate machines on the LAN.
- The live scan-trigger workspace is `soupy_database/databases/scan_triggers/` (created at runtime; not in source control).
