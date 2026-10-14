# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Soupy** is a fully-local Discord bot combining AI chat, image generation, web search, autonomous Discord posts, and an autonomous Bluesky presence. It runs entirely on-premises against LM Studio (or any OpenAI-compatible LLM server) and a Stable Diffusion backend. A FastAPI web control panel manages the bot process, streams logs, and edits configuration.

**Maintainer docs:** [ARCHITECTURE.md](ARCHITECTURE.md) (system design, invariants, IPC), [CONTRIBUTING.md](CONTRIBUTING.md) (conventions + how to extend), and [ROADMAP.md](ROADMAP.md) (prioritized backlog). This file is the quick-reference; those go deeper.

---

## Installer

A first-run wizard lives at `install.py`, with helpers under `installer/`
(see `INSTALL.md` for end-user docs). It is a single-file entry point
plus a small package — stdlib-only at module level (`colorama` is the one
optional third-party import, with a graceful fallback) so it runs on a
fresh clone before `pip install` has happened. Each numbered step in
`installer/steps/sNN_*.py` exposes `run(state, ui) -> dict`. The wizard
persists progress to `.install-state.json` (gitignored), masks secrets
in that file, generates `.env-stable` from `.env-stable.example` (or
updates an existing `.env-stable` in place) while preserving comments /
ordering / multi-line values, and `os.execvp`s `run_all.py` with
`.venv`'s Python at the end. It installs `requirements.txt`;
`requirements.lock` is used only with `--locked`.

When a change touches the env-var surface, the Discord-bot connection,
LM Studio probe shape, or SD endpoints, also check whether `install.py`
or the steps under `installer/steps/` need a corresponding update.

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

1. **Discord bot** — `soupy_remastered_stablediffusion.py` is the entrypoint and owns chat handling, image generation (`SDQueue`), event handlers, and most slash commands. At `on_ready` it loads five discord.py extensions from the `soupy/cogs/` package:
   - `soupy.cogs.search` — `/soupysearch`
   - `soupy.cogs.imagesearch` — `/soupyimage`
   - `soupy.cogs.dailypost` — `/soupypost` + autonomous daily article posts
   - `soupy.cogs.musings` — `/soupymuse` + periodic "thinking out loud"
   - `soupy.cogs.bluesky` — `/soupysky` + autonomous Bluesky activity

2. **Web control panel** (`web/`) — FastAPI app that manages the bot subprocess (`web/services/bot_runner.py`), streams its log via WebSocket (`web/services/log_stream.py`), reads/writes `.env-stable` (`web/services/env_store.py`), and exposes archive/stats/runtime-flag endpoints.

3. **External services** — LM Studio (chat + embeddings, OpenAI-compatible), Stable Diffusion FastAPI backend (image gen), DuckDuckGo (search), Bluesky AT Protocol (autonomous posts), and per-guild SQLite databases for message archive + RAG.

**Cross-module imports to be aware of:**
- `soupy/cogs/dailypost.py` imports `_fetch_og_image` and `_post_url` from `soupy/cogs/bluesky.py` for cross-posting articles to Bluesky.
- `soupy/cogs/bluesky.py` imports `_extract_date_from_html` and `_estimate_article_age_days` from `soupy/cogs/dailypost.py` for article freshness checks.
- All cogs share the same `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `LOCAL_CHAT` env vars and instantiate their own `OpenAI` client.

---

## Key Files

| File | Purpose |
|------|---------|
| [soupy_remastered_stablediffusion.py](soupy_remastered_stablediffusion.py) | Main bot — chat, image gen, core slash commands, event loop |
| [soupy/cogs/search.py](soupy/cogs/search.py) | Cog — DuckDuckGo web search + LLM summary |
| [soupy/cogs/imagesearch.py](soupy/cogs/imagesearch.py) | Cog — DuckDuckGo image search |
| [soupy/cogs/dailypost.py](soupy/cogs/dailypost.py) | Cog — autonomous daily article posts to Discord (+ optional Bluesky cross-post) |
| [soupy/cogs/musings.py](soupy/cogs/musings.py) | Cog — periodic reflective musings in a configured channel |
| [soupy/cogs/bluesky.py](soupy/cogs/bluesky.py) | Cog — autonomous Bluesky replies, quote-posts, original posts |
| [soupy/settings.py](soupy/settings.py) | Typed, cached config accessors over `.env-stable` |
| [soupy/prompts.py](soupy/prompts.py) | Prompt loader (legacy env var → `prompts/<name>.txt` → `prompts/<name>.default.txt` → fallback) |
| [soupy/msn.py](soupy/msn.py) | Resolves an msn.com link to the article it was syndicated from |
| [soupy/slop.py](soupy/slop.py) | Anti-slop rules for Bluesky drafts: the prompt block and the tell detector |
| [soupy/triggers.py](soupy/triggers.py) | Pure respond/don't-respond predicates |
| [soupy/timeaware.py](soupy/timeaware.py) | Time awareness for chat: the clock line, age phrases, history markers |
| [soupy/scheduling.py](soupy/scheduling.py) | Shared local-time windows + atomic JSON state files for the autonomous loops |
| [run_all.py](run_all.py) | Launcher: starts uvicorn with `SOUPY_AUTOSTART_BOT=1` |
| [web/app.py](web/app.py) | FastAPI app — routes, WebSocket, archive/stats endpoints |
| [web/services/bot_runner.py](web/services/bot_runner.py) | Spawns and manages the bot subprocess via PTY |
| [web/services/flux_runner.py](web/services/flux_runner.py) | Starts, stops and watches `flux_server.py`; keeps `logs/flux_server.log` |
| [soupy/cogs/flux.py](soupy/cogs/flux.py) | Cog — `/flux`, its buttons and dialogs, and what a result remembers about how it was made |
| [flux_server.py](flux_server.py) | The local Flux backend (mflux/MLX): `/flux`, `/flux_img2img`, `/flux_edit`, `/health`. Its own process |
| [web/services/env_store.py](web/services/env_store.py) | Parses and rewrites `.env-stable` |
| [web/services/browse.py](web/services/browse.py) | Read-only queries behind the Database and Media tabs: messages, profiles, Soupy's memory, the image archive |
| [web/services/lm_studio.py](web/services/lm_studio.py) | The model server's list, and the chat-model switch: plan it, carry it out, undo it if LM Studio fails |
| [web/services/prompt_store.py](web/services/prompt_store.py) | The prompt files behind the panel's editor: the list of prompts, exact saves, backups, reset to default |
| [web/static/dashboard/](web/static/dashboard/) | The dashboard, no build step: `dashboard-core.js`, `dashboard-media.js`, `dashboard-database.js`, `dashboard-model.js`, `dashboard-app.js`, loaded in that order |
| [web/services/log_stream.py](web/services/log_stream.py) | WebSocket fan-out for live log streaming |
| [soupy_database/database.py](soupy_database/database.py) | SQLite schema + per-guild DB operations |
| [soupy_database/rag.py](soupy_database/rag.py) | RAG embeddings and cosine similarity retrieval |
| [soupy_database/user_profiles.py](soupy_database/user_profiles.py) | Member profiles: pass loop, profile jobs + nightly refresh, chat-time profile prefix |
| [soupy_database/profile_document.py](soupy_database/profile_document.py) | Profile document v2: sections, dated items, edits, caps, rendering |
| [soupy_database/profile_llm.py](soupy_database/profile_llm.py) | Profile-build prompt, edit-list schema, context budget from LM Studio |
| [soupy/llm_gate.py](soupy/llm_gate.py) | One-big-LLM-call-at-a-time gate shared by chat, profiles, reflection, cogs |
| [soupy_database/self_profile.py](soupy_database/self_profile.py) | Soupy's memory of itself: dated items from its own messages, built with the profile pass loop |
| [soupy_database/self_context.py](soupy_database/self_context.py) | SELF.MD files and the identity anchor in every reply |
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
| `FLUX_ENABLED` / `FLUX_SERVER_URL` / `FLUX_SERVER_AUTOSTART` | Local Flux backend: on/off (`/flux` is only registered when on), where the bot reaches it, and whether the web panel starts it |
| `OWNER_IDS` | Comma-separated Discord user IDs for admin commands |
| `CHANNEL_IDS` | Channels the bot actively monitors |
| `SOUPY_TRIGGER_KEYWORDS` | Comma-separated keywords that trigger replies outside monitored channels. Default: `soup,gumbo` |
| `GUILD_ID` | Primary Discord server ID (used for fast slash-command sync) |
| `MAX_TOKENS` | LLM response length limit |
| `RAG_ENABLED` | Enable retrieval-augmented generation |
| `ENABLE_VISION` | Enable image understanding via LM Studio's vision-capable LLM (set `VISION_MODEL`) |
| `DAILY_POST_ENABLED` / `DAILY_POST_CHANNELS` | Autonomous Discord article posts |
| `MUSING_ENABLED` / `MUSING_CHANNEL_ID` / `MUSING_HOUR_MIN` / `MUSING_HOUR_MAX` | Autonomous musings — one per day, at a random time in the local-hour window |
| `BLUESKY_HANDLE` / `BLUESKY_APP_PASSWORD` / `BLUESKY_AUTO_REPLY` | Bluesky integration |
| `SELF_MD_ENABLED` | Soupy's memory of itself, updated at the end of each profile refresh (nightly or manual) |
| `USER_PROFILE_NIGHTLY_ENABLED` / `USER_PROFILE_NIGHTLY_HOUR` | Nightly member-profile refresh (default on, 4 AM local, `USER_PROFILE_NIGHTLY_MAX_MINUTES` time limit) |
| `SOUPY_WEB_HOST` / `SOUPY_WEB_PORT` | Web panel binding (default: `0.0.0.0:4941`) |
| `SOUPY_AUTOSTART_BOT` | Set by `run_all.py`; `0` to launch the web panel without the bot |
| `SOUPY_BOT_ENTRY` | Override bot entrypoint resolution |

There are also 50+ `WEB_COLOR_*` variables for UI customization.

---

## Personality and prompts

Soupy's personality is one system prompt, `behaviour`, sent with every chat reply. There is no per-guild or runtime preset: the same prompt is used everywhere. Eight more prompts cover search answers, article posts, `/9ball` and the image prompts.

Each prompt is a text file. The bot loads `prompts/<name>.txt` (yours, gitignored) when it exists and the tracked `prompts/<name>.default.txt` otherwise, once, at startup. A legacy variable in `.env-stable` (`BEHAVIOUR`, `BEHAVIOUR_SEARCH`, `BEHAVIOUR_DAILY_POST`, `9BALL`, `FANCY`, `RANDOMPROMPT`, `SD_NEGATIVE_PROMPT`) still overrides its file when set; `tools/migrate_prompts.py` moves those into files. The template no longer sets any.

| Prompt file | Used for |
|-------------|----------|
| `behaviour` | The system prompt for every chat reply |
| `behaviour_search` | `/soupysearch` summaries |
| `behaviour_daily_post` | Commentary on the daily article post |
| `nineball` | `/9ball` answers |
| `fancy`, `fancy_flux` | Rewriting an image prompt for the Fancy button |
| `randomprompt`, `randomprompt_flux` | Turning random keywords into an image prompt |
| `sd_negative_prompt` | Sent with every Stable Diffusion request (Flux has no negative prompt) |

Edit a prompt in the panel (Model & Personality → Prompts) or in the file. The panel's editor saves the text exactly as typed and keeps the replaced version in `prompts/backups/`. **Do not put a prompt back into `.env-stable`**: it would override the file, and the env editor no longer offers fields for them.

`web/services/prompt_store.PROMPTS` is the list the editor shows. A prompt the bot loads that is missing from it fails `tests/test_web_prompts.py`.

The chat lookup is `get_guild_behaviour()` in [soupy_remastered_stablediffusion.py](soupy_remastered_stablediffusion.py).

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
| `/flux <description> [image] [strength] [fancy]` | soupy.cogs.flux | Generate image **locally** via Flux/mflux, backed by `flux_server.py`. With an image: edits it by instruction (Klein edit), or repaints it at `strength` when one is given (noise-mix img2img). `fancy` has Soupy write a fuller prompt first |
| `/img2img` | main | Transform an image with a prompt |
| `/inpaint` | main | Inpaint an image with a mask + prompt |
| `/outpaint <prompt> <direction>` | main | Extend image by ~25% |
| `/soupysearch <query>` | soupy.cogs.search | Web search + LLM summary |
| `/soupyimage <query>` | soupy.cogs.imagesearch | DuckDuckGo image search |
| `/soupypost` | soupy.cogs.dailypost | Force daily article post (owner) |
| `/soupymuse` | soupy.cogs.musings | Trigger a musing (owner) |
| `/soupysky [action] [url]` | soupy.cogs.bluesky | Bluesky reply/repost/post (owner) |
| `/soupyself [action]` | main | View, refresh, or reset Soupy's memory of itself (owner) |
| `/soupyscan` | main | Archive messages to database (owner) |
| `/soupystats` | main | Server and bot statistics |
| `/status` | main | Service health check |
| `/helpsoupy` | main | Command reference |
| `/whattime <location>` | main | Local time lookup |
| `/weather <location>` | main | Weather lookup |
| `/8ball` / `/9ball` | main | Magic 8-ball (classic / LLM-powered) |
| `/testurl` | main | Test URL extraction |

Image generation posts the result with a `SDRemixView` control panel — Remix, R-Fancy, R-Keyword, Fancy, Edit, Wide, Tall, and Outpaint buttons.

A `/flux` result gets `FluxRemixView`: Edit prompt, Fancy, Remix, Change image, R-Fancy, R-Keyword, Wide, Tall, Square. Change image starts from the picture it sits under (a chain of edits runs A to B to C). A picture made from another one remembers its source, so Remix and Edit prompt redo that step from the source; it gets no Fancy or shape buttons.

---

## Data & Storage

- **Per-guild SQLite DBs**: `soupy_database/databases/guild_<id>.db` — message archive + RAG embeddings
- **Generated images**: `media/images/` (full) + `media/thumbs/` (thumbnails). `media/images/index.jsonl` has one row per image; a Flux row also records model, steps, strength, source image and the prompt a Fancy rewrite started from
- **Uploaded Flux source images**: `media/flux_sources/` (named by content; results are not copied here)
- **Message log**: `media/messages.jsonl`
- **User stats**: `user_stats.json`
- **Logs**: `logs/soupy.log` (rotating, 5MB max, 5 backups); `logs/flux_server.log` when the panel runs the Flux server

Under `data/`:
- `data/runtime_flags.json` — shared bot↔web feature toggles (RAG enabled, command disables, time awareness)
- `data/bot_dashboard.json` — bot-written status the web panel reads
- `data/self_md/guild_<id>_self.json` — Soupy's memory per guild; `guild_<id>.md` + `_core.md` + `_anchor.md` are views rendered from it (`_archive.md` and `v1_backup/` hold the retired reflection's files)
- `data/musings_archive.jsonl` — musings history
- `data/musings_daily_state.json` — today's rolled musing time + whether the day has been handled
- `data/profile_nightly_state.json` — local date the nightly profile refresh was last queued
- `data/daily_post_history.json` / `daily_post_schedule.json` — daily-post state
- `data/bluesky_engage_history.json` / `bluesky_schedule.json` — Bluesky engagement state

---

## Key Code Patterns

- **Cog architecture** — five extensions loaded at `on_ready` (see Architecture). Editing a cog requires only a bot restart, not a web-panel restart.
- **Async/await throughout** — all I/O is non-blocking
- **One big LLM call at a time** — chat replies, profile-build passes (members and Soupy's own memory), and the cogs' LLM calls all take `soupy.llm_gate.llm_turn()`. Profile passes use most of LM Studio's loaded context window, and hitting that ceiling takes LM Studio down, so large prompts must never overlap. Chat goes first: the profile builder waits while `ChatQueue` has a reply queued or running (plus a 60 s grace) before each pass, but never aborts a pass in flight. Profile jobs run only in the bot process (the web panel just queues them) for the same reason.
- **Two single-consumer work queues** — `SDQueue` serializes image generation so the SD/flux backend isn't overloaded; `ChatQueue` serializes chat replies. Keep them separate: they were one queue once, and since each consumer awaits a job to completion, a single `/flux` blocked every reply bot-wide with no typing indicator anywhere. Both derive from `_WorkQueue`.
- **Time awareness** — when the `time_awareness` runtime flag is on (default), `process_chat_message` reads the clock once (`timeaware.now_local()`, in `TIMEZONE`) and passes that one value everywhere: the `TIME` block that ends the system prompt, `[chat from ...]` markers on stale stretches of history, and `now=` into RAG so retrieved conversations and profile items say how long ago they were. History entries carry a `ts` key that must be stripped (`label_history` / `strip_timestamps`) before anything reaches the LLM. Ages are worded in code; never hand the 12b raw dates to do arithmetic on.
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

- Comments in `.env-stable` and `.env-stable.example` go on their own line. The web panel's `parse_env` takes everything after `=` as the value, and under `run_all.py` its values are the ones the bot gets, so `KEY=6  # note` arrives as `"6  # note"`. `tests/test_env_template.py` checks the template.
- The dashboard API sends Discord IDs as strings. A browser reads a JSON number as a double and corrupts a snowflake, which broke every member filter in the old explorer. `tests/test_web_browse.py` checks it.
- The panel reads page code (`web/static/`, `web/templates/`) from disk on every load, but `web/app.py` and `web/services/` only when it starts. After a change to both, the page is ahead of the server until the panel restarts. The Model & Personality tab says so instead of breaking.
- A chat-model switch from the panel stops the bot, unloads and loads in LM Studio, writes `.env-stable`, then starts the bot. `CONTEXT_WINDOW_TOKENS` is the chat prompt budget, not the context length the model is loaded with; the budget has to stay under it. `tests/fake_lm_studio.py` stands in for LM Studio in tests and for trying the panel by hand.
- A Flux button click after a restart is handled by a blank template view. It reads the prompt, seed and size from the message and the rest (source image, strength, original prompt) from the picture's row in `media/images/index.jsonl`, found by the attachment's filename. Button `custom_id`s must not change: Discord stores them on every posted message. `tests/flux_fakes.py` has the stand-ins for testing the cog; `tests/test_flux_buttons.py` clicks with a blank view.
- `flux_server.py` is its own process. A change to it or to a server-side `FLUX_*` setting needs the Flux server restarted (the panel's Flux server card, or by hand); restarting the bot does not touch it. To try a change with real generations, run a second server on a spare port (`FLUX_SERVER_PORT=14942 .venv/bin/python flux_server.py`).
- The `.env-stable` file accumulates timestamped backups (`.env-stable.bak.*`) on every web-UI save — safe to clean up old ones.
- The bot supports multi-guild operation; each Discord server gets its own SQLite database and its own Soupy memory.
- Discord requires **Message Content Intent** to be enabled in the Discord Developer Portal.
- The production setup runs LM Studio and Stable Diffusion on separate machines on the LAN.
- The live scan-trigger workspace is `soupy_database/databases/scan_triggers/` (created at runtime; not in source control).
