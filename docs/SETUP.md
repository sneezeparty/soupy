# First Run Walkthrough

For most users the installer (`python install.py`) handles steps 1–5 of this walkthrough — see [INSTALL.md](../INSTALL.md). This doc is the manual breakdown if you want to know what the wizard is doing under the hood, plus the post-install archive scan and RAG verification steps that the installer does *not* do for you.

This is the path from `git clone` to a working bot with persistent memory. The bot will start and respond to messages after Phase 5 — but Soupy's memory (RAG) doesn't actually contain anything until you complete Phases 6 and 7. Plan accordingly.

## Phase 1 — Local Services

Soupy talks to two external services that you run yourself:

**LM Studio (required).** Install [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible server) on a machine with a capable GPU and start its local server. You need **two models loaded simultaneously**:

- A **chat model** (whatever you set in `LOCAL_CHAT`, e.g. `google/gemma-3-27b`) — used for replies, musings, daily-post writing, Bluesky candidates, etc.
- An **embedding model** (whatever you set in `RAG_EMBEDDING_MODEL`, e.g. `text-embedding-qwen3-embedding-0.6b`) — used by RAG to vectorize messages and queries. **This is separate from the chat model and must be loaded explicitly in LM Studio.** Without it, RAG will fail at runtime with a "RAG_EMBEDDING_MODEL is not set" or HTTP 4xx error.

Confirm the server is reachable from wherever the bot will run (`curl http://<lm-studio-host>:1234/v1/models`).

If you enable `ENABLE_VISION=true`, also load a vision-capable multimodal model (set `VISION_MODEL` to its exact id).

**Stable Diffusion backend (optional — only if you want image generation).** Stand up the bundled FastAPI backend on a GPU host (or an Apple Silicon Mac) per [HARDWARE.md](HARDWARE.md). Skip this entirely if you don't need `/sd` and friends — the rest of the bot still works.

## Phase 2 — Bot Install

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

## Phase 3 — Discord App Setup

1. **Create a Discord Application** at the [Discord Developer Portal](https://discord.com/developers/applications).
2. **Create a Bot** under the Bot tab and copy the token (you'll paste it into `.env-stable` in Phase 4).
3. **Enable Privileged Gateway Intents**: Message Content Intent (required), Server Members Intent (recommended).
4. **Invite the bot** to your server with these permissions: Read/Send Messages, Embed Links, Attach Files, Use Slash Commands, Read Message History.
5. Note your Discord user ID (Developer Mode → right-click yourself → Copy User ID) — that goes in `OWNER_IDS`.

## Phase 4 — Minimum Config

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

## Phase 5 — Start the Bot

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

## Phase 6 — Initial Archive Scan (`/soupyscan`)

The per-guild SQLite database starts **empty**. The bot only knows what's in the database — meaning a freshly-installed Soupy has zero memory of your server's history. To populate the archive, run `/soupyscan` once per guild as an owner.

> [!WARNING]
> **The first scan can take a long time on busy servers.** It does two things in series:
> 1. Pulls historical messages from Discord (rate-limited by Discord's API — see below).
> 2. After messages are saved, it embeds every chunk via the LM Studio embedding model so RAG can retrieve them. **The embedding step is usually the bottleneck.**
>
> On a busy server (20+ active users, hundreds of messages/day, 365-day backlog), a full first scan can run for **multiple days, up to about a week**, depending on hardware.
>
> **Strongly recommended for new installs:** lower `FIRST_SCAN_LOOKBACK_DAYS` (default `365`) to something like `7`, `30`, or `60` for the first run. You can run another scan later — subsequent scans are incremental and only fetch messages since the last completed scan, so they finish quickly.

### Rough order-of-magnitude estimates

These are very rough. Your mileage will vary based on total message count, embedding-model size, GPU vs. CPU embedding, network latency between bot and LM Studio host, image-description and URL-summarization volume during the scan, and Discord's own rate limiting. Treat these as ballparks, not guarantees.

- **Quiet / personal server, ~1k messages, modern GPU embedding** — minutes.
- **Medium server, ~50k messages, modern GPU embedding** — a few hours.
- **Busy server, ~500k messages, modern GPU embedding** — around a day.
- **Very busy server, 1M+ messages, modern GPU embedding** — multiple days, up to about a week.
- **Any of the above on CPU-only embedding** — multiply substantially; usually impractical for backfills.

The scan code itself sleeps between Discord message reads (a randomized 0.1–2.0 seconds depending on the message; longer pauses every few messages) so it doesn't hammer the Discord API. Channel-by-channel commits land every 50 messages, so an interrupted scan does not lose work.

### Tuning the scan

In `.env-stable`:

- **`FIRST_SCAN_LOOKBACK_DAYS`** (default `365`) — how far back the very first scan reaches. Subsequent scans are always incremental and ignore this.
- **`SCAN_EXCLUDE_CHANNEL_IDS`** (comma-separated channel IDs) — skip bot-spam channels, off-topic dumps, voice-text channels, NSFW-only rooms, etc. to keep the corpus relevant. Scope the archive to channels where the conversation actually matters.
- **`RAG_EMBED_MAX_CONCURRENT`** (default `2`) — how many embedding requests run in parallel against LM Studio during the post-scan reindex. Raising this can speed things up if your LM Studio host has headroom; setting it too high will tip the embedding server into errors. Tune up gradually.

### What "interrupted" actually means

If the scan dies mid-way (bot restart, network drop, LM Studio crash), you can re-run `/soupyscan` and it will pick up where it left off in practice — already-saved messages are deduped via `message_exists` and skipped, so nothing is double-inserted. **However**, because `last_scan_time` is only recorded at successful completion, a re-run after a crash treats the next attempt as another "first scan" (Discord re-fetched, dedup-skipped on insert). The work that's redone is the Discord pull, not the embedding. Acceptable, but it's another reason to keep `FIRST_SCAN_LOOKBACK_DAYS` modest on the first attempt.

### Watching progress

- Live log: open the **web panel console drawer** for streaming scan output (channels processed, messages added, periodic checkpoints).
- **Database Explorer tab** shows per-guild row counts and scan history.
- The dashboard's **Archive Scan** and **RAG Reindex** cards show whether each loop is currently running, when it last completed, and (for the scheduled archive scan) when it'll run next.

## Phase 7 — RAG Reindex and Verification

After `/soupyscan` finishes, the bot **automatically kicks off a RAG reindex in the background** if any new messages were added. You don't have to trigger it manually for the normal case. The reindex consolidates raw messages into conversation chunks and embeds them via your LM Studio embedding model — populating the `rag_chunks` table.

If for any reason you want to force a reindex (changed embedding models, manual database edits, etc.), the **Database Explorer tab** has a "RAG Reindex" button per guild. There's also a periodic consolidation pass on a timer (`RAG_REINDEX_INTERVAL_HOURS`, default `6`).

**Turn on RAG.** RAG retrieval defaults to **off** and is controlled by a runtime flag stored in `data/runtime_flags.json`, not by an env variable. Toggle it from the dashboard's runtime-flags section ("RAG on every reply") — the bot picks up the change live, no restart needed.

**Verify it works.** Once the reindex finishes and RAG is toggled on, ask the bot something it should remember from earlier in the channel — a running joke, a topic that came up last week, a username it should recognize. If the response actually references that history, RAG retrieval is working. If it answers cold, double-check:

1. The Database Explorer shows non-zero rows in `rag_chunks` for that guild.
2. `RAG_EMBEDDING_MODEL` exactly matches the embedding model loaded in LM Studio.
3. The "RAG on every reply" toggle is on in the dashboard.
4. The bot was restarted (or the runtime flag flushed) after any env edits.

Subsequent maintenance is mostly hands-off: schedule periodic incremental scans via the Database Explorer (or just re-run `/soupyscan`), and the bot will keep its archive and RAG index current.
