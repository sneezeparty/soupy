

![Soupy Header](https://i.imgur.com/JNbVjY3.png)

![Soupy Remastered Header](https://i.imgur.com/AiCorTA.jpeg)

Please feel free to [Buy Me A Coffee](https://buymeacoffee.com/sneezeparty) to help support this project.  

Join [Soupy's Discord Server](https://discord.gg/YJeBgsMt) to try it out.

---

## What Soupy Is

Soupy is a fully-local Discord bot designed to feel like a real member of your server — one who reads the room, forms opinions, and shares things on its own. It chats with personality, remembers conversations, builds profiles of the people it talks to, and autonomously shares articles it finds interesting both on Discord and Bluesky. Everything runs on your own hardware against a local LLM (LM Studio or any OpenAI-compatible server) — no cloud APIs, no per-token costs, no data leaving your network.

**What it does, briefly:**

- **Talks like a person** — A configurable personality, RAG-backed memory of past conversations, LLM-generated user profiles, and a self-knowledge document that tracks the bot's own opinions, relationships, and running jokes.
- **Shares things on its own** — Twice a day across configured channels, Soupy reads each channel's audience, finds an article those specific people would care about, writes a brief take in its own voice, and posts it.
- **Lives on Bluesky too** — Autonomous replies, quote-posts, and original posts on a randomized daily schedule, all generated through a 3-candidate-plus-judge pipeline with strict fact-checking against the source article.
- **Reads URLs you drop in chat** — Fetches and summarizes links so it can respond to what they actually say.
- **Sees images** — Optional vision support routes Discord image attachments through a vision-capable LLM so Soupy can react to what's actually in the picture.
- **Searches the web** — DuckDuckGo-backed search and image search with LLM summaries.
- **Generates images** — Stable Diffusion backend with remix, outpaint, edit, and random-prompt buttons.
- **Web control panel** — A FastAPI dashboard for managing the bot: start/stop/restart, live log streaming, model and personality editing, per-loop on/off toggles, a categorized form-based editor for every variable in `.env-stable`, plus stats, media browsing, and a per-guild database explorer. Almost every cadence and threshold the bot uses is tunable from this panel without touching code.

## Table of Contents

- [What Soupy Is](#what-soupy-is)
- [Soupy Remastered](#soupy-remastered)
  - [Conversation and Personality](#conversation-and-personality)
  - [Daily Article Posts (Discord)](#daily-article-posts-discord)
  - [Bluesky Integration](#bluesky-integration)
  - [Search and Utilities](#search-and-utilities)
  - [Image Generation](#image-generation)
- [Web Control Panel](#web-control-panel)
- [Key Files and Components](#key-files-and-components)
- [Software Requirements](#software-requirements)
- [Hardware Requirements](#hardware-requirements)
- [Stable Diffusion Backend Setup](#stable-diffusion-backend-setup)
- [First Run Walkthrough](#first-run-walkthrough)
  - [Phase 1 — Local Services](#phase-1--local-services)
  - [Phase 2 — Bot Install](#phase-2--bot-install)
  - [Phase 3 — Discord App Setup](#phase-3--discord-app-setup)
  - [Phase 4 — Minimum Config](#phase-4--minimum-config)
  - [Phase 5 — Start the Bot](#phase-5--start-the-bot)
  - [Phase 6 — Initial Archive Scan (`/soupyscan`)](#phase-6--initial-archive-scan-soupyscan)
  - [Phase 7 — RAG Reindex and Verification](#phase-7--rag-reindex-and-verification)
- [Available Commands](#available-commands)
- [Web Control Panel Customization](#web-control-panel-customization)
- [Usage Examples](#usage-examples)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Support](#support)

---

# Soupy Remastered

## Conversation and Personality

Soupy is built to feel like a server regular, not a query-response bot. It joins conversations when its name comes up, occasionally interjects on its own, and brings real context to what it says.

| Capability | Description |
|---|---|
| **Chat** | Responds when @-mentioned, when "soup" appears in a message, or in any channel listed in `CHANNEL_IDS`. Outside of those triggers, Soupy will jump into a random message with a configurable probability (`RANDOM_RESPONSE_RATE`, default `0.05` = 5%). All triggers are tunable via the web Environment Editor. |
| **Personality** | A single system prompt (`BEHAVIOUR` in `.env-stable`) defines Soupy's voice. The web panel's **Model & Personality** tab has a dedicated editor for `BEHAVIOUR` and `BEHAVIOUR_SEARCH` (the voice used for `/soupysearch` summaries) with auto-formatting and a "Save preset + restart" button. For very long rewrites, editing `.env-stable` in a real text editor is still the safest path. |
| **RAG memory** | Retrieval-augmented generation pulls relevant past messages from the per-guild SQLite archive so Soupy actually remembers what people have said. Embeddings are computed by the local LLM server. Top-K, similarity thresholds, context budget, embedding concurrency, and reindex interval are all env-tunable (`RAG_TOP_K`, `RAG_CONTEXT_MAX_CHARS`, `RAG_EMBED_MAX_CONCURRENT`, `RAG_REINDEX_INTERVAL_HOURS`, etc.). The dashboard also has a runtime "RAG-on-every-reply" toggle that takes effect without restart. |
| **User profiles** | LLM-generated structured profiles (opinions, hobbies, interests) built from each user's message history, used to tailor responses. Profile sample size, minimum-message threshold, max-age before re-merge, and chunked-build threshold are all env-tunable; the Database Explorer tab triggers per-guild profile-batch rebuilds with a progress log. |
| **Self-knowledge** | Soupy maintains a running document about itself — opinions it's formed, relationships with regulars, jokes it's been part of — refreshed through reflection cycles. Tiered into core (always in the system prompt), full (RAG-indexed), and archive. Reflection interval, minimum-interactions trigger, max-words caps, and reflection temperature are all env-tunable (`SELF_MD_REFLECT_INTERVAL_HOURS`, `SELF_MD_MIN_INTERACTIONS`, `SELF_MD_MAX_WORDS`, etc.). |
| **URL awareness** | When someone drops a link, Soupy fetches the article and responds to the actual content. Fetched content is cached for an hour to avoid hammering sources. Fetch timeout, max URLs per message, and content length cap are env-tunable (`URL_FETCH_TIMEOUT`, `MAX_URLS_PER_MESSAGE`, `URL_MAX_CONTENT_LENGTH`). |
| **Vision** | Optional. With `ENABLE_VISION=true` and a vision-capable model loaded in LM Studio (`VISION_MODEL`), Soupy describes Discord image attachments and works that into its reply. Prompt template, max-tokens, and temperature are env-tunable. |
| **Musings** | Soupy occasionally "thinks out loud" in a configured channel, reflecting on past conversations or things it's been reading. Random poll interval (`MUSING_POLL_MINUTES_MIN`/`MAX`, defaults 10–20 minutes) and per-poll fire probability (`MUSING_CHANCE`, default `0.10` = 10%) are env-tunable from the web panel. Manual override via `/soupymuse`. |

## Daily Article Posts (Discord)

Twice a day across the channels you configure, Soupy autonomously shares an article it thinks the people in that channel would actually care about. Two posts are scheduled per day — a morning slot and an evening slot — and each one targets a different channel. The pipeline:

1. **Read the room** — Pulls the channel's top posters and loads their structured profiles, then samples recent messages from the channel.
2. **Build an audience brief** — LLM synthesizes the profiles and recent themes into a description of what this channel's audience is into right now.
3. **Search** — LLM generates 2–3 specific search queries; DuckDuckGo News finds candidate articles from the last 24 hours (falling back to the last week).
4. **Filter and rate** — Junk-URL and date pre-filtering rejects homepages, feeds, forums, and articles older than `DAILY_POST_MAX_AGE_DAYS` (default 21); LLM rates the survivors against the audience brief.
5. **Pick and write** — Fetches full content of the top 3, picks the best one, and writes a brief opinionated take in Soupy's voice.
6. **Post and notify** — Drops the post in the channel, optionally cross-posts it to Bluesky with a link card, and pings the owner in the musing channel.

If nothing good is found, Soupy retries up to a few times and then skips that slot for the day rather than posting filler. The day's schedule is persisted to disk and survives restarts.

**Configurable via the web Environment Editor** — enable/disable the loop (`DAILY_POST_ENABLED`, also a one-click toggle on the Overview tab), per-channel topic hints (`DAILY_POST_CHANNELS`, JSON map), active hours window (`DAILY_POST_ACTIVE_START` / `DAILY_POST_ACTIVE_END`, default 8–18 in your `TIMEZONE`), per-channel interval (`DAILY_POST_INTERVAL_HOURS`, default 24), maximum article age (`DAILY_POST_MAX_AGE_DAYS`, default 21), and the persona used to write the take (`BEHAVIOUR_DAILY_POST`).

`/soupypost` (owner-only) forces a post immediately to a chosen channel.

## Bluesky Integration

Soupy maintains an autonomous Bluesky presence via the AT Protocol, running alongside the Discord bot. Activity is randomized across the day so it doesn't look like a scheduled bot, with a minimum 45-minute gap between actions.

### Autonomous daily activity (6am–11pm Pacific)
- **Replies** — Default 4–7/day. Finds interesting posts from the timeline, trending topics, and thread exploration; reads the full comment thread; generates 3 candidate replies and picks the best.
- **Quote-posts** — Default 1/day. Shares someone else's post with brief commentary.
- **Original posts** — Default 1/day. Mines Bluesky and DuckDuckGo for articles, fetches full content, generates a take, and posts with a link card and og:image thumbnail.
- **Likes** — Likes 1–2 good comments per thread it replies to (currently capped at 10/day in code).
- **Follows** — May follow up to 2 interesting accounts per day based on their recent post quality (also currently a hardcoded daily cap).

The Overview tab on the web panel shows today's progress live (e.g. `5 replies · 1 post · 1 repost`) and includes a one-click toggle for the whole loop (`BLUESKY_AUTO_REPLY`).

### Quality controls
- **3-candidate generation** — Every reply, quote-post, and original post generates 3 candidates; an LLM judge picks the best one.
- **Fact-checking** — Posts and replies are checked against the source article before publishing. The checker explicitly allows opinion, framing, sarcasm, and editorial takes — it only blocks invented facts, wrong attribution, and contradictions.
- **Author diversity** — Won't reply to the same author twice in a row (last 5 authors blocked, next 10 penalized).
- **Article freshness** — Articles without a verifiable date are rejected outright; anything older than 14 days is filtered out (currently hardcoded).
- **Schedule persistence** — Daily schedule is saved to disk and survives restarts; completed actions are not re-scheduled.
- **Rate limiting** — Random delays between all API actions; daily caps on likes and follows.

### Configuration

Editable from the web Environment Editor (Bluesky Integration tab):

| Variable | Default | What it controls |
|---|---|---|
| `BLUESKY_HANDLE` | — | Your handle (e.g. `name.bsky.social`) |
| `BLUESKY_APP_PASSWORD` | — | App password from Settings → App Passwords (not your main password) |
| `BLUESKY_AUTO_REPLY` | `false` | Master enable/disable for the autonomous loop |
| `BLUESKY_REPLIES_MIN` | `4` | Lower bound on replies per day |
| `BLUESKY_REPLIES_MAX` | `7` | Upper bound on replies per day |
| `BLUESKY_REPOSTS_PER_DAY` | `1` | Quote-posts per day |
| `BLUESKY_POSTS_PER_DAY` | `1` | Original article posts per day |

The 45-minute minimum inter-action gap, 10 likes/day cap, 2 follows/day cap, and 14-day article freshness window are currently hardcoded in `soupy_bluesky.py`.

### Manual control (`/soupysky`)
- `/soupysky` or `/soupysky reply` — Find and reply to a post now.
- `/soupysky repost` — Quote-post something interesting now.
- `/soupysky post` — Find an article and post about it now.
- `/soupysky post url:https://...` — Post about a specific article you provide.

All actions report results to the configured musing channel and tag the owner.

## Search and Utilities

| Command | Description |
|---|---|
| `/soupysearch <query>` | DuckDuckGo search with LLM-powered summary and citations |
| `/soupyimage <query>` | Image search — returns a random result from the top 300 |
| `/whattime <location>` | Local time for a geographic location |
| `/weather <location>` | Current weather for a location |
| `/8ball` / `/9ball` | Classic and LLM-powered Magic 8-Ball |
| `/soupystats` | Server and bot statistics dashboard |
| `/status` | Service health check (LLM, SD backend) |

## Image Generation

Soupy includes a Stable Diffusion image generation pipeline through `/sd`, with interactive buttons for iterating on results:

- **Remix** — Regenerate with a new random seed
- **R-Fancy** — LLM generates a random prompt from keyword files (~212 quadrillion combinations)
- **R-Keyword** — Random keywords only (no LLM)
- **Fancy** — LLM elaborates on the current prompt
- **Edit** — Modify the prompt, dimensions, or seed
- **Outpaint** — Extend the image in any direction

`/img2img`, `/inpaint`, and `/outpaint` provide additional transformations. This is a self-contained feature — if you don't want it, leave the SD backend disconnected and the rest of the bot still works fine.

SD endpoints, default steps/guidance, default and preset dimensions (`SD_DEFAULT_*` / `SD_WIDE_*` / `SD_TALL_*`), the negative prompt, the "Fancy" prompt-rewrite template, and the outpaint ControlNet weights are all editable from the web Environment Editor (Stable Diffusion Setup, Prompt Templates & Files, and Outpaint tabs).

# Web Control Panel

Soupy ships with a FastAPI control panel that runs alongside the bot and manages it as a subprocess. **The panel is the primary way to operate Soupy** — start/stop/restart the bot, watch its log live, edit configuration, switch LLM models, and tune virtually every cadence and threshold the bot uses, all without touching the terminal. The recommended workflow is `python run_all.py`, then drive everything from the browser.

What the panel actually does:

- **Process management** — Start, stop, and restart the bot from a single button. The web app spawns the bot as a subprocess via PTY; restarting the bot does not restart uvicorn, and vice versa, so you can iterate on `.env-stable` or cog code rapidly.
- **Live log streaming** — Bot stdout/stderr is fanned out over a WebSocket (`/ws/logs`) and shown in a console drawer in real time. Also written to `logs/soupy.log` (rotating, 5MB, 5 backups).
- **Background-loop dashboard** — At-a-glance cards for each scheduled loop (Daily Post, Bluesky Engagement, Archive Scan, RAG Reindex, Self-Reflection, Musings) showing whether each is enabled, when it last ran, when it next runs, today's progress (e.g. `2 / 2 posts`, `5 replies · 1 post · 1 repost`), and a one-click toggle that flips the corresponding env flag.
- **Runtime feature flags** — Toggle RAG-on-every-reply and disable individual slash commands without a restart (these are stored in `data/runtime_flags.json` and read live by the bot).
- **Model & Personality tab** — Pick the active LM Studio model from a searchable dropdown, set the context-window size, and click "Switch model + restart" to push the change into LM Studio and reload the bot. A dedicated personality editor edits `BEHAVIOUR` and `BEHAVIOUR_SEARCH` directly with auto-formatting on load and a "Load raw" mode for the as-stored text. (Long prompts are still safest to edit in `.env-stable` directly — open the file in a real editor if you're rewriting from scratch.)
- **Environment Editor** — Categorized, form-based editor for every variable in `.env-stable`. Each field has a "[?]" tooltip with a description and a placeholder showing the in-code default. Every save creates a timestamped `.env-stable.bak.*` backup. Categories include Discord setup, LLM, Stable Diffusion, Chat Behavior, Rate Limits, Vision, RAG, User Profiles, Context Window, Self-Knowledge, Daily Posts, Bluesky, Musings, Outpaint, and Web Panel theming.
- **Stats Studio** — Server and bot statistics: message volume by category over time, top channels, top users, image counts, totals across 24h / 7d / 30d windows.
- **Media & Log tab** — Browse generated images as a thumbnail light-table; click for caption, user, prompt, and metadata. Live message stream alongside.
- **Database Explorer** — Per-guild SQLite browser. Inspect `messages`, `profiles`, `rag_chunks`, and `scan_metadata` tables. Trigger or schedule archive scans, kick off RAG reindex, and run profile-batch jobs (build/refresh user profiles for everyone in a guild) with progress logs.
- **Theming** — 30+ `WEB_COLOR_*` variables with a color-picker UI in the env editor. Set `WEB_CONTROL_PANEL_TITLE` to rebrand the page.

The panel binds to `0.0.0.0:4941` by default. Override with `SOUPY_WEB_HOST` and `SOUPY_WEB_PORT`. There are exactly two pages: `/` (the React dashboard with all the tabs above) and `/env` (the form-based env editor).

**Configurable without code changes.** Almost every cadence, cap, threshold, and probability mentioned in the feature sections above is an env var, which means it's editable in the web Environment Editor and applied on the next bot restart (one click in the panel). The handful of values that are currently hardcoded are called out explicitly where they appear.

## Key Files and Components

| File / Directory | Description |
|------|-------------|
| `soupy_remastered_stablediffusion.py` | Main Discord bot — chat, image generation, core slash commands, event loop |
| `soupy_search.py` | Cog — DuckDuckGo web search with LLM summary (`/soupysearch`) |
| `soupy_imagesearch.py` | Cog — DuckDuckGo image search (`/soupyimage`) |
| `soupy_dailypost.py` | Cog — autonomous daily article posts (Discord), with optional Bluesky cross-post |
| `soupy_musings.py` | Cog — periodic "thinking out loud" musings (`/soupymuse`) |
| `soupy_bluesky.py` | Cog — autonomous Bluesky replies, quote-posts, and original posts (`/soupysky`) |
| `run_all.py` | Launcher: starts the web panel which auto-spawns the bot |
| `web/` | FastAPI control panel: routes, templates, dashboard JS, env editor, WebSocket log fan-out |
| `soupy_database/` | Per-guild SQLite databases, RAG embeddings, user profiles, self-knowledge document |
| `sd-api/` | Reference Stable Diffusion FastAPI backend (run separately on a GPU host or Apple Silicon Mac — see [Stable Diffusion Backend Setup](#stable-diffusion-backend-setup)) |
| `.env-stable` | Master configuration file (see `.env-stable.example` for the template) |
| `soupy_characters.txt` / `soupy_styles.txt` / `soupy_themes.txt` / `sd_keywords.txt` | Keyword files used by the random-prompt buttons in `/sd` |

## Software Requirements

- **LM Studio** (or any OpenAI-compatible local server) for chat and embeddings. A vision-capable multimodal model is required if you enable `ENABLE_VISION`.
- **Stable Diffusion backend** (FastAPI) for image generation. A reference implementation lives in `sd-api/sd_api.py`; in the production setup it runs on a separate LAN machine with a GPU.
- **Python 3.10+**
- **PyTorch with CUDA** (only required on the host running the Stable Diffusion backend)
- **Bluesky account** with an app password (optional — for Bluesky integration)

## Hardware Requirements

- 16GB+ system RAM on the bot host (more if you also run LM Studio there)
- 24GB GPU on the SD backend host (defaults are tuned for SD 3.5 Medium with `qfloat8` quantization; smaller VRAM may work for SD 1.5 / smaller SDXL checkpoints — see below)
- LLM and SD backends can each run on a separate machine on the LAN; the bot itself is light

## Stable Diffusion Backend Setup

**Skip this entire section if you don't want image generation.** The bot runs fine without `/sd`, `/img2img`, `/inpaint`, and `/outpaint` — those commands will simply error out, and everything else (chat, search, daily posts, Bluesky, etc.) keeps working.

If you do want image generation, Soupy's `sd-api/` directory ships a reference FastAPI backend that the bot talks to over HTTP. It is **a separate process with its own dependencies and its own Python environment**, typically run on a different machine on the LAN — the GPU host — while the bot itself runs somewhere lighter.

### What's in `sd-api/`

| File | Purpose |
|---|---|
| `sd-api/sd_api.py` | Main FastAPI backend (Linux / NVIDIA CUDA) |
| `sd-api/sd_api-mac.py` | Apple Silicon variant — same API, uses MPS (Metal) instead of CUDA, falls back to CPU |
| `sd-api/requirements.txt` | Dependency list for the SD host (Linux/CUDA) |
| `sd-api/requirements-m1-mac.txt` | Dependency list for Apple Silicon |
| `sd-api/M1_MAC_SETUP.md` | Mac-specific setup notes — read this if you're running the backend on Apple Silicon |

### Install (Linux / NVIDIA)

The SD host needs its own venv. **Do not** install these into the bot's venv — the bot's `requirements.txt` deliberately excludes the heavy SD-only stack (`diffusers`, `optimum-quanto`, `rembg`, `transformers`, etc.).

```bash
# On the GPU host
git clone https://github.com/sneezeparty/soupy.git    # only sd-api/ is strictly needed
cd soupy
python -m venv .venv-sd
source .venv-sd/bin/activate

# PyTorch with CUDA 11.8 — install before sd-api/requirements.txt
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r sd-api/requirements.txt

python sd-api/sd_api.py
```

The first launch downloads the configured model from Hugging Face (default: `stabilityai/stable-diffusion-3.5-medium`) into the standard HF cache (`~/.cache/huggingface/`). This can be tens of GB and take a while depending on your connection. Subsequent launches use the cache.

### Install (Apple Silicon)

```bash
# On the Mac
git clone https://github.com/sneezeparty/soupy.git
cd soupy
python -m venv .venv-sd
source .venv-sd/bin/activate

# Standard (non-CUDA) PyTorch — MPS support is built in
pip install -r sd-api/requirements-m1-mac.txt

python sd-api/sd_api-mac.py
```

`sd_api-mac.py` auto-detects MPS at startup. See [`sd-api/M1_MAC_SETUP.md`](sd-api/M1_MAC_SETUP.md) for verification steps, performance notes, and troubleshooting.

### Ports and endpoints

The backend is **a single FastAPI app listening on port `8000`**. All endpoints — `/sd`, `/sd_img2img`, `/sd_inpaint`, `/outpaint_hybrid`, `/remove_background`, `/upscale`, `/health` — are served from that one process. The `host="0.0.0.0", port=8000` binding is currently hardcoded in `sd_api.py` / `sd_api-mac.py`; change them in code if you need a different port.

> **Note on the `:8001` defaults in `.env-stable.example`.** The example file ships with `SD_IMG2IMG_URL` and `SD_INPAINT_URL` pointing at `:8001`, which would be correct for a multi-process setup but does **not** match the bundled single-process backend. The bot has fallback logic that derives both URLs from `SD_SERVER_URL` if the dedicated values are unreachable, so things still work — but the cleanest config is to point all four bot-side env vars at the same `:8000` host.

### Bot-side configuration

In `.env-stable`, point all the SD endpoints at the backend host:

```bash
SD_SERVER_URL=http://<sd-host>:8000/
SD_IMG2IMG_URL=http://<sd-host>:8000/sd_img2img
SD_INPAINT_URL=http://<sd-host>:8000/sd_inpaint
REMOVE_BG_API_URL=http://<sd-host>:8000/remove_background
```

`<sd-host>` is the LAN hostname or IP of the GPU machine — `localhost` if you're running the bot and SD on the same box. If you change which machine runs SD, **update all four**.

`ANALYZE_IMAGE_API_URL` is in `.env-stable.example` but is not currently consumed by the bot or served by the backend; it is safe to leave at its default. Image vision goes through LM Studio's vision-capable model (set `ENABLE_VISION=true`), not through the SD backend.

### Models, LoRAs, GPU sizing

- **Model selection.** The model id is set as `StableDiffusionConfig.REPO_NAME` near the top of `sd-api/sd_api.py` (default `stabilityai/stable-diffusion-3.5-medium`, with `USE_SDXL = True`). A handful of community model ids are listed as commented-out alternatives. To change models, edit that constant in code — there is currently no env-var override for the model id, and the change requires restarting the SD process.
- **LoRA.** Set `LORA_PATH` (and optionally `LORA_WEIGHT`, default `1.0`) in the SD process's environment. The LoRA is loaded and fused into the pipeline at startup. Switching LoRAs requires a backend restart.
- **GPU sizing.** A 24GB-class GPU (RTX 3090 / 4090 / A5000-class) comfortably runs the default SD 3.5 Medium config with the on-startup `qfloat8` quantization (via `optimum-quanto`) plus `enable_model_cpu_offload()` and attention slicing. Smaller GPUs may work for SD 1.5 or smaller SDXL checkpoints with the same code path — verify by trying it; out-of-memory will surface in the SD host's log.
- **Apple Silicon.** Performance is generally slower than a discrete NVIDIA GPU but workable for casual use. M1/M2/M3 unified memory is shared between CPU and GPU; 32GB+ is recommended. See [`sd-api/M1_MAC_SETUP.md`](sd-api/M1_MAC_SETUP.md).

### Verify it's up

From the bot host (or anywhere on the LAN that can reach the SD host):

```bash
curl http://<sd-host>:8000/health
# {"status":"ok","model":"stabilityai/stable-diffusion-3.5-medium"}
```

A `200 OK` with the configured model id confirms the pipeline finished loading. From inside Discord, `/status` reports SD-backend reachability alongside the LLM, and `/sd a quick test` is the end-to-end smoke test.

## First Run Walkthrough

This is the path from `git clone` to a working bot with persistent memory. The bot will start and respond to messages after Phase 5 — but Soupy's memory (RAG) doesn't actually contain anything until you complete Phases 6 and 7. Plan accordingly.

### Phase 1 — Local Services

Soupy talks to two external services that you run yourself:

**LM Studio (required).** Install [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible server) on a machine with a capable GPU and start its local server. You need **two models loaded simultaneously**:

- A **chat model** (whatever you set in `LOCAL_CHAT`, e.g. `google/gemma-3-27b`) — used for replies, musings, daily-post writing, Bluesky candidates, etc.
- An **embedding model** (whatever you set in `RAG_EMBEDDING_MODEL`, e.g. `text-embedding-qwen3-embedding-0.6b`) — used by RAG to vectorize messages and queries. **This is separate from the chat model and must be loaded explicitly in LM Studio.** Without it, RAG will fail at runtime with a "RAG_EMBEDDING_MODEL is not set" or HTTP 4xx error.

Confirm the server is reachable from wherever the bot will run (`curl http://<lm-studio-host>:1234/v1/models`).

If you enable `ENABLE_VISION=true`, also load a vision-capable multimodal model (set `VISION_MODEL` to its exact id).

**Stable Diffusion backend (optional — only if you want image generation).** Stand up the bundled FastAPI backend on a GPU host (or an Apple Silicon Mac) per the [Stable Diffusion Backend Setup](#stable-diffusion-backend-setup) section. Skip this entirely if you don't need `/sd` and friends — the rest of the bot still works.

### Phase 2 — Bot Install

```bash
git clone https://github.com/sneezeparty/soupy.git
cd soupy
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# PyTorch with CUDA 11.8 — only needed on the host that runs the SD backend.
# Skip this on a CPU-only / non-image-gen install.
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

cp .env-stable.example .env-stable
```

### Phase 3 — Discord App Setup

1. **Create a Discord Application** at the [Discord Developer Portal](https://discord.com/developers/applications).
2. **Create a Bot** under the Bot tab and copy the token (you'll paste it into `.env-stable` in Phase 4).
3. **Enable Privileged Gateway Intents**: Message Content Intent (required), Server Members Intent (recommended).
4. **Invite the bot** to your server with these permissions: Read/Send Messages, Embed Links, Attach Files, Use Slash Commands, Read Message History.
5. Note your Discord user ID (Developer Mode → right-click yourself → Copy User ID) — that goes in `OWNER_IDS`.

### Phase 4 — Minimum Config

Open `.env-stable` and set, at minimum:

```bash
DISCORD_TOKEN=<token from the Discord Developer Portal>
OWNER_IDS=<your Discord user id, comma-separated for multiple owners>
GUILD_ID=<your Discord server id>                 # enables per-guild slash sync (commands appear in seconds, not up to an hour)
CHANNEL_IDS=<channel id>,<channel id>             # channels Soupy actively listens in and chats freely (without these, it only replies to @-mentions or "soup"/"soupy")
OPENAI_BASE_URL=http://<lm-studio-host>:1234/v1   # or http://localhost:1234/v1
LOCAL_CHAT=<exact model id loaded in LM Studio>
SD_SERVER_URL=http://<sd-host>:port               # only if using image generation
```

> **Why `GUILD_ID` and `CHANNEL_IDS` matter on day one.** Without `GUILD_ID`, slash commands fall back to Discord's global sync and can take up to an hour to appear the first time — so `/helpsoupy` will look broken even though the bot is fine. Without `CHANNEL_IDS`, Soupy will not initiate or randomly join conversations in any channel; it only responds when @-mentioned or when "soup" or "soupy" appears in a message. Set both before your first start to avoid the "is it even working?" period.

**If you want memory (RAG)** — and you almost certainly do — also set the embedding model id. RAG itself is toggled at runtime via the dashboard (it defaults to off and is *not* an env variable):

```bash
RAG_EMBEDDING_MODEL=<exact embedding model id loaded in LM Studio>
```

`OPENAI_API_KEY` can be left unset; the code defaults it to `lm-studio` for local servers.

For optional integrations:

```bash
# Bluesky
BLUESKY_HANDLE=yourname.bsky.social
BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx   # Settings → App Passwords (not your main password)
BLUESKY_AUTO_REPLY=true

# Autonomous daily article posts
DAILY_POST_ENABLED=true
DAILY_POST_CHANNELS={"123456789012345678": "tech and gaming"}  # JSON map, channel_id → topic hint
```

Almost every other knob in `.env-stable` is editable later from the web Environment Editor — you don't need to touch them for the first run.

### Phase 5 — Start the Bot

```bash
source .venv/bin/activate
python run_all.py
```

This launches the FastAPI web panel, which auto-spawns the bot as a subprocess. Open the panel:

```
http://127.0.0.1:4941     # local
http://<lan-ip>:4941      # from elsewhere on your network
```

You should see the bot show up online in your Discord server within a few seconds. Try `/helpsoupy` or just say "hey soupy" in a channel listed in `CHANNEL_IDS` to confirm chat works.

If you'd rather run components separately:

```bash
# Web panel only — set SOUPY_AUTOSTART_BOT=0 first if you don't want it to spawn the bot
python -m uvicorn web.app:app --host 0.0.0.0 --port 4941

# Bot only — useful for direct stack traces (no web panel)
python soupy_remastered_stablediffusion.py
```

To restart the bot after editing `.env-stable` or any cog, use the web panel's restart button (or `POST /api/bot/restart`). You don't need to restart uvicorn.

### Phase 6 — Initial Archive Scan (`/soupyscan`)

The per-guild SQLite database starts **empty**. The bot only knows what's in the database — meaning a freshly-installed Soupy has zero memory of your server's history. To populate the archive, run `/soupyscan` once per guild as an owner.

> [!WARNING]
> **The first scan can take a long time on busy servers.** It does two things in series:
> 1. Pulls historical messages from Discord (rate-limited by Discord's API — see below).
> 2. After messages are saved, it embeds every chunk via the LM Studio embedding model so RAG can retrieve them. **The embedding step is usually the bottleneck.**
>
> On a busy server (20+ active users, hundreds of messages/day, 365-day backlog), a full first scan can run for **multiple days, up to about a week**, depending on hardware.
>
> **Strongly recommended for new installs:** lower `FIRST_SCAN_LOOKBACK_DAYS` (default `365`) to something like `7`, `30`, or `60` for the first run. You can run another scan later — subsequent scans are incremental and only fetch messages since the last completed scan, so they finish quickly.

#### Rough order-of-magnitude estimates

These are very rough. Your mileage will vary based on total message count, embedding-model size, GPU vs. CPU embedding, network latency between bot and LM Studio host, image-description and URL-summarization volume during the scan, and Discord's own rate limiting. Treat these as ballparks, not guarantees.

| Server profile | Hardware | Wall-clock estimate |
|---|---|---|
| Quiet / personal server, ~1k messages | Modern GPU embedding | Minutes |
| Medium server, ~50k messages | Modern GPU embedding | A few hours |
| Busy server, ~500k messages | Modern GPU embedding | Around a day |
| Very busy server, 1M+ messages | Modern GPU embedding | Multiple days, up to about a week |
| Any of the above on **CPU-only embedding** | CPU | Multiply substantially — usually impractical for backfills |

The scan code itself sleeps between Discord message reads (a randomized 0.1–2.0 seconds depending on the message; longer pauses every few messages) so it doesn't hammer the Discord API. Channel-by-channel commits land every 50 messages, so an interrupted scan does not lose work.

#### Tuning the scan

In `.env-stable`:

- **`FIRST_SCAN_LOOKBACK_DAYS`** (default `365`) — how far back the very first scan reaches. Subsequent scans are always incremental and ignore this.
- **`SCAN_EXCLUDE_CHANNEL_IDS`** (comma-separated channel IDs) — skip bot-spam channels, off-topic dumps, voice-text channels, NSFW-only rooms, etc. to keep the corpus relevant. Scope the archive to channels where the conversation actually matters.
- **`RAG_EMBED_MAX_CONCURRENT`** (default `2`) — how many embedding requests run in parallel against LM Studio during the post-scan reindex. Raising this can speed things up if your LM Studio host has headroom; setting it too high will tip the embedding server into errors. Tune up gradually.

#### What "interrupted" actually means

If the scan dies mid-way (bot restart, network drop, LM Studio crash), you can re-run `/soupyscan` and it will pick up where it left off in practice — already-saved messages are deduped via `message_exists` and skipped, so nothing is double-inserted. **However**, because `last_scan_time` is only recorded at successful completion, a re-run after a crash treats the next attempt as another "first scan" (Discord re-fetched, dedup-skipped on insert). The work that's redone is the Discord pull, not the embedding. Acceptable, but it's another reason to keep `FIRST_SCAN_LOOKBACK_DAYS` modest on the first attempt.

#### Watching progress

- Live log: open the **web panel console drawer** for streaming scan output (channels processed, messages added, periodic checkpoints).
- **Database Explorer tab** shows per-guild row counts and scan history.
- The dashboard's **Archive Scan** and **RAG Reindex** cards show whether each loop is currently running, when it last completed, and (for the scheduled archive scan) when it'll run next.

### Phase 7 — RAG Reindex and Verification

After `/soupyscan` finishes, the bot **automatically kicks off a RAG reindex in the background** if any new messages were added. You don't have to trigger it manually for the normal case. The reindex consolidates raw messages into conversation chunks and embeds them via your LM Studio embedding model — populating the `rag_chunks` table.

If for any reason you want to force a reindex (changed embedding models, manual database edits, etc.), the **Database Explorer tab** has a "RAG Reindex" button per guild. There's also a periodic consolidation pass on a timer (`RAG_REINDEX_INTERVAL_HOURS`, default `6`).

**Turn on RAG.** RAG retrieval defaults to **off** and is controlled by a runtime flag stored in `data/runtime_flags.json`, not by an env variable. Toggle it from the dashboard's runtime-flags section ("RAG on every reply") — the bot picks up the change live, no restart needed.

**Verify it works.** Once the reindex finishes and RAG is toggled on, ask the bot something it should remember from earlier in the channel — a running joke, a topic that came up last week, a username it should recognize. If the response actually references that history, RAG retrieval is working. If it answers cold, double-check:

1. The Database Explorer shows non-zero rows in `rag_chunks` for that guild.
2. `RAG_EMBEDDING_MODEL` exactly matches the embedding model loaded in LM Studio.
3. The "RAG on every reply" toggle is on in the dashboard.
4. The bot was restarted (or the runtime flag flushed) after any env edits.

Subsequent maintenance is mostly hands-off: schedule periodic incremental scans via the Database Explorer (or just re-run `/soupyscan`), and the bot will keep its archive and RAG index current.

## Available Commands

| Command | Description |
|---------|-------------|
| `/sd <prompt>` | Generate image via Stable Diffusion |
| `/img2img` | Transform an image with a prompt |
| `/inpaint` | Inpaint with a mask and prompt |
| `/outpaint <prompt> <direction>` | Extend image by ~25% |
| `/soupysearch <query>` | Web search with LLM summary and citations |
| `/soupyimage <query>` | Image search |
| `/soupysky [action] [url]` | Bluesky engagement: reply, repost, or post (owner only) |
| `/soupypost [channel]` | Force a daily article post to Discord (owner only) |
| `/soupymuse` | Trigger a musing (owner only) |
| `/soupystats` | Server and bot statistics dashboard |
| `/soupyself [action]` | View/manage self-knowledge document (owner only) |
| `/soupyscan` | Archive messages to database (owner only) |
| `/status` | Check service health |
| `/helpsoupy` | Command reference |
| `/whattime <location>` | Local time lookup |
| `/weather <location>` | Weather lookup |
| `/8ball <question>` | Classic Magic 8-Ball |
| `/9ball <question>` | LLM-powered 8-Ball |

### Chat triggers

Soupy responds when:
- @-mentioned directly
- Someone says "soup" or "soupy" in a message
- A message lands in any channel listed in `CHANNEL_IDS`
- Randomly, with probability `RANDOM_RESPONSE_RATE` (default `0.05` = 5%)

`CHANNEL_IDS` and `RANDOM_RESPONSE_RATE` are both editable from the web Environment Editor (Discord Server Setup and Chat Behavior tabs).

## Web Control Panel Customization

The Web Control Panel category in the Environment Editor exposes every theming variable with a color-picker UI. Or edit `.env-stable` directly:

### Color scheme
```bash
WEB_COLOR_PAGE_BG=#1e1010
WEB_COLOR_CARD_BG=#1a2332
WEB_COLOR_TEXT_PRIMARY=#e5e7eb
# ... 30+ more WEB_COLOR_* variables in .env-stable.example
```

### Title and binding
```bash
WEB_CONTROL_PANEL_TITLE="My Custom Bot Control"
SOUPY_WEB_HOST=0.0.0.0
SOUPY_WEB_PORT=4941
```

## Usage Examples

**Chatting** — Soupy is content-aware and browses URLs:

![soupy chatting](https://i.imgur.com/RlV9HV6.png)

**`/soupysearch`** — Web search with LLM summary:

![search](https://i.imgur.com/UnfRKsC.png)

**`/sd` image generation:**

![basic image](https://i.imgur.com/ODIR9OT.png)

**Fancy button** — LLM elaborates on the prompt:

![fancy](https://i.imgur.com/naG52aN.png)

**Random button** — Keywords from text files + LLM:

![random](https://i.imgur.com/0eFjCSq.png)

---

## License

This project is licensed under the MIT License.

MIT License Copyright (c) 2024-2026 sneezeparty

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgements

- **LM Studio**: OpenAI-compatible local LLM server
- **Hugging Face**: Diffusers and community SD models
- **FastAPI**: Web control panel and SD backend
- **DuckDuckGo**: Search without API keys
- **Bluesky/AT Protocol**: Autonomous social media engagement — replies, posts, likes, follows

## Support

If you encounter any issues, open an issue at [GitHub Issues](https://github.com/sneezeparty/soupy/issues).

[Buy Me A Coffee](https://buymeacoffee.com/sneezeparty) to help support this project.
