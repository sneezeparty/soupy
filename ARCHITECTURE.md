# Architecture

This document is the canonical technical design reference for Soupy. It explains
how the pieces fit together, where state lives, the invariants you must not break,
and the seams where new features attach. It is aimed at maintainers — for
operator-facing setup and tuning see [`docs/`](docs/), and for the day-to-day
contribution workflow see [CONTRIBUTING.md](CONTRIBUTING.md).

> **Note on line numbers.** Where this doc cites `file.py:NNN` the number is a
> hint, not a contract — the 6 K-line main file moves around. Function and symbol
> names are the durable references; grep for those.

---

## 1. The three tiers

Soupy is one repository running as **two cooperating processes** plus a set of
**external services** it talks to over HTTP.

```
                         ┌─────────────────────────────────────────────┐
                         │  run_all.py                                   │
                         │   └─ uvicorn (web/app.py)   ← process #1      │
                         │        │ startup hook                         │
                         │        ▼                                      │
                         │   BotRunner.start() ──spawn(PTY)──► bot       │
                         │                          (soupy_remastered_   │
                         │                           stablediffusion.py) │
                         │                                ← process #2   │
                         └─────────────────────────────────────────────┘
                              │  file-based IPC (data/*.json, .env-stable)
                              ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  LM Studio   │   │ Stable Diff. │   │  DuckDuckGo  │   │   Bluesky    │
   │ chat + embed │   │   backend    │   │ web + image  │   │ AT Protocol  │
   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### Tier 1 — the Discord bot (`soupy_remastered_stablediffusion.py`)

A single ~6,250-line module that is the bot process entrypoint. It owns:

- The **chat reply pipeline** (`on_message` → `process_chat_message`).
- **Image generation** (`SDQueue`, `generate_sd_image`, the `/sd` family, and the
  `SDRemixView` button UI).
- The **"main" cog** of inline slash/prefix commands (`/helpsoupy`, `/soupystats`,
  `/status`, `/8ball`, `/9ball`, `/whattime`, `/weather`, `/testurl`, `/soupyself`,
  `/soupyscan`, and the owner prefix commands `reloadenv` / `synccommands`).
- **Event handlers and background loops** started at `on_ready` (see §5).
- Loading the five feature cogs from `soupy/cogs/` via `load_extensions()`.

Feature behavior that isn't core chat/image lives in cogs (Tier 1b) and shared
helpers in the `soupy/` package (Tier 1c).

### Tier 1b — feature cogs (`soupy/cogs/`)

Five discord.py extensions, loaded at `on_ready`:

| Cog | Command(s) | Autonomous behavior |
|-----|-----------|---------------------|
| `soupy.cogs.search` | `/soupysearch` | — |
| `soupy.cogs.imagesearch` | `/soupyimage` | — |
| `soupy.cogs.dailypost` | `/soupypost` | Twice-daily article posts (+ optional Bluesky cross-post) |
| `soupy.cogs.musings` | `/soupymuse` | Periodic "thinking out loud" |
| `soupy.cogs.bluesky` | `/soupysky` | Autonomous replies / quote-posts / original posts |

### Tier 1c — shared package (`soupy/`)

Small, well-factored, and **unit-tested** (unlike the main file). This is the
"clean core":

- `soupy/settings.py` — typed, lazily-cached configuration. 87 `@cached_property`
  accessors over `.env-stable`. `openai_client()` builds a fresh OpenAI SDK client
  per caller (deliberately not shared — see §7).
- `soupy/prompts.py` — prompt resolution with a four-step fallback chain
  (legacy env var → `prompts/<name>.txt` → `prompts/<name>.default.txt` → caller
  fallback), cached, invalidated by `reloadenv`.
- `soupy/triggers.py` — pure predicates deciding whether the bot should respond
  (`message_contains_trigger_keyword`, `should_randomly_respond`). Reads env on
  **every call** so dashboard keyword edits take effect without restart.
- `soupy/log.py` — the single `ColoredFormatter` used by both processes.

### Tier 2 — web control panel (`web/`)

A FastAPI app (`web/app.py`, ~2,545 lines) that is the **parent process** and the
primary operator interface. It:

- Spawns and supervises the bot subprocess (`web/services/bot_runner.py`).
- Streams the bot's stdout/stderr to browsers over a WebSocket
  (`web/services/log_stream.py` + `/ws/logs`).
- Reads and rewrites `.env-stable` while preserving comments and ordering
  (`web/services/env_store.py`).
- Exposes ~40 `/api/*` endpoints: archive browsing, stats, env editing, LM Studio
  model switching, runtime-flag toggles, RAG/profile management, and bot lifecycle.

### Tier 3 — database tier (`soupy_database/`)

Per-guild SQLite plus the RAG / profile / self-knowledge machinery:

- `database.py` — schema + per-guild DB ops, scan triggers, the `active_scans`
  race guard.
- `rag.py` — chunk → embed → retrieve, with a per-guild reindex lock and a global
  embedding semaphore.
- `user_profiles.py` — structured + summary per-member profiles.
- `self_context.py` — the evolving self-knowledge document (anchor/core/full/archive
  tiers) and the reflection accumulator.
- `runtime_flags.py` — the mtime-cached bot↔web feature-toggle channel.
- `profile_batch.py` — supervisor for background profile-rebuild jobs.

---

## 2. The process model (read this before touching `run_all.py` or the web panel)

The single most surprising thing about Soupy's runtime: **`run_all.py` does not
start the bot.** It starts uvicorn with `SOUPY_AUTOSTART_BOT=1`; the web app's
`startup` event calls `BotRunner.start()`, which spawns the bot as a child process.

Consequences:

- **To pick up bot code changes**, restart the *bot* (web panel restart button or
  `POST /api/bot/restart`). Restarting uvicorn is neither necessary nor sufficient.
- **To pick up web code changes**, restart uvicorn (`run_all.py`).
- On every bot start, `.env-stable` is re-parsed and merged into the child env, so
  env edits take effect on **bot** restart.
- The bot runs under a **PTY** on POSIX (pipes on Windows) so colorama/rich emit
  ANSI as if attached to a terminal; the web layer strips ANSI before broadcasting
  to the log WebSocket.
- `BotRunner._resolve_entrypoint()` picks the script:
  `$SOUPY_BOT_ENTRY` → `soupy_remastered_stablediffusion.py` → `run_soupy.py` →
  `soupy_remastered.py`.
- **LOCAL_KEY shim:** if `OPENAI_API_KEY` is empty, `LOCAL_KEY` is mapped into it
  for the child. This lives in `bot_runner.start()` and must stay in sync with
  `soupy.settings`.

---

## 3. The chat reply pipeline

This is the most behavior-sensitive surface in the codebase. Trace:

```
on_message
  ├─ drop if author is self or any bot            (message.author.bot guard)
  ├─ process_image_attachment  → vision LLM        (if ENABLE_VISION)
  ├─ _index_message_realtime   → fire-and-forget RAG embed (plain-text msgs only)
  ├─ should_bot_respond_to_message  (mention | keyword | CHANNEL_IDS | random)
  └─ enqueue {"type": "chat"} onto bot.chat_queue
                              │
                              ▼
process_chat_message  (runs one-at-a-time off the queue)
  1. Assemble system prompt:  BEHAVIOUR  +  self-knowledge anchor  +  technical_instructions
  2. Token budget:  CONTEXT_WINDOW − MAX_TOKENS − safety buffer − system − current
                    split history 45% / RAG 35% / URL 15%
  3. Recent history       (fetch_recent_messages, trimmed to budget)
  4. RAG retrieval        (if is_rag_enabled(): build_rag_retrieval_query → fetch_rag_context_for_query)
  5. URL content          (history URLs + current-message URLs, cached w/ TTL)
  6. Image descriptions    (from the vision step)
  7. Merge consecutive same-role turns (Gemma-style strict-alternation compat)
  8. generate_parallel_candidates  (N completions at varied temperatures)
  9. judge_best_of_candidates       (LLM scores; skipped if N == 1)
 10. clean_response → split_message → send → archive_sent_message
 11. add_notable_interaction  (feeds the self-knowledge accumulator)
```

Key design decisions worth preserving:

- **Best-of-N with an LLM judge.** Small local models are inconsistent;
  generating a few candidates at spread temperatures and judging them lifts
  quality more than a single low-temperature call. Tunable via env.
- **Aggressive output scrubbing.** `clean_response()` strips RAG metadata,
  timestamps, and separators that local models tend to regurgitate verbatim. RAG
  context is injected behind a sentinel marker so it can be recognized and removed.
- **The `RESPOND TO THE MESSAGE BELOW` marker.** After history + RAG + URL content +
  image descriptions are merged, the actual trigger message can be buried 10 K+
  chars deep inside a single user turn (the same-role merge that strict-alternation
  models require). Heavy visual fences keep the trigger findable. Don't remove them.
- **Self-knowledge: anchor, not core.** `get_self_md_for_injection` injects the
  small (~600-char) *anchor* into every system prompt; the larger *core* and *full*
  documents are retrieved on demand via self-knowledge RAG. This keeps the per-reply
  system prompt around 6–7 K chars instead of ~12 K.

### Triggering

`should_bot_respond_to_message` returns true on **any** of: the bot is mentioned,
the message contains a `SOUPY_TRIGGER_KEYWORDS` keyword (default `soup,gumbo`), the
channel is in `CHANNEL_IDS`, or a probabilistic `should_randomly_respond` roll. The
`message.author.bot` guard short-circuits *before* any of this, so Soupy never
replies to other bots (this was a real bug — see CHANGELOG).

---

## 4. Image generation

`/sd` (and `/flux`, `/img2img`, `/inpaint`, `/outpaint`) enqueue work onto a single
`SDQueue`. The queue processor runs **one job at a time** — image generation can
take minutes on a Mac SD backend, and the queue exists precisely to prevent
overloading it. Routing is by `item["type"]` (`sd` / `flux` / `button` / `outpaint`)
and, for buttons, by `item["action"]`.

Chat replies ride a **separate** `ChatQueue` (§3). Both derive from `_WorkQueue`
and both are single-consumer, but they must stay distinct: they were one queue
once, and because the consumer awaits each job to completion, a single `/flux`
generation blocked every reply bot-wide until it finished — with no typing
indicator anywhere, since `process_chat_message` is what starts it. Chat hits LM
Studio and image generation hits the SD/flux server, so there is nothing to
serialize between them.

`generate_sd_image()` POSTs to `SD_SERVER_URL` (with pooled aiohttp connections and
a 600 s timeout), validates the returned image's magic bytes, archives it to
`media/`, and replies with an embed carrying the `SDRemixView` button panel
(Remix / Wide / Tall / Edit / Fancy / Outpaint).

Outpaint is a "hybrid" mode using Canny-edge and/or depth ControlNet conditioning
with a tunable harmonize strength (`OUTPAINT_*` env vars).

---

## 5. Background loops

Started fire-and-forget at `on_ready` and held by a module-level `_background_tasks`
set (without a strong reference, asyncio garbage-collects a running task mid-await):

| Loop | Cadence | Purpose |
|------|---------|---------|
| `SDQueue.process_queue` | continuous | Serialized image job processing |
| `ChatQueue.process_queue` | continuous | Serialized chat replies — independent of image work |
| `scan_trigger_loop` | ~5 s | Watches `soupy_database/databases/scan_triggers/` for web-requested scans |
| `archive_auto_scan_loop` | ~45 s poll | Incremental message archival per guild on its configured interval |
| `rag_reindex_loop` | `RAG_REINDEX_INTERVAL_HOURS` (6) | Consolidate + re-embed RAG chunks |
| `_dashboard_status_writer` | 15 s | Writes `data/bot_dashboard.json` for the web panel |
| `_self_md_reflection_loop` | daily at `SELF_MD_REFLECT_HOUR` local (3) | Runs a self-knowledge reflection cycle when enough interactions have accumulated |

Each cog additionally runs its own `@tasks.loop` (dailypost, musings, bluesky). All
of them re-read their enable flag per tick so they can be toggled live from the
panel.

---

## 6. State: where everything lives

Soupy has no central datastore. State is spread across SQLite, JSON/JSONL files, and
Markdown documents, each owned by exactly one module.

| Path | Owner | Contents |
|------|-------|----------|
| `soupy_database/databases/guild_<id>.db` | `database.py` | Messages, channels, `rag_chunks`, `user_profile_summaries`, self-chunks, profile-batch jobs, scan metadata |
| `data/runtime_flags.json` | `runtime_flags.py` | RAG enable + per-command disable toggles (bot↔web) |
| `data/bot_dashboard.json` | bot (write) / web (read) | Uptime, model, service health, loop timers |
| `data/self_md/guild_<id>{,_core,_anchor,_archive}.md` | `self_context.py` | Self-knowledge tiers |
| `data/self_md/accumulator.jsonl` | `self_context.py` | Pending interactions awaiting reflection (survives restart) |
| `data/daily_post_history.json` / `daily_post_schedule.json` | dailypost cog | Posted-article history + next-fire schedule |
| `data/musings_archive.jsonl` | musings cog | Last ~200 musings + topic tags for dedup |
| `data/bluesky_engage_history.json` | bluesky cog | Replies/likes/follows + daily counters |
| `.env-stable` | `env_store.py` (write) / bot (read on start) | All configuration |
| `media/images/`, `media/thumbs/` | main bot | Generated images |
| `user_stats.json` | main bot | Per-guild user counters |
| `logs/soupy.log` | both processes | Rotating log (5 MB × 5) |

**Atomic-write invariant.** Every JSON/JSONL state file is written via `tmp + os.replace`
so a crash mid-write can't truncate it. New code that persists state must do the same.

---

## 7. Concurrency model

- **Everything I/O is async.** Blocking work (PIL, the OpenAI SDK, sync HTTP) runs
  in a thread via `asyncio.to_thread`.
- **Fire-and-forget tasks** go through `_spawn_task`, which stores a strong ref in
  `_background_tasks` and discards it on completion.
- **`user_stats.json`** is serialized by `user_stats_lock`; always go through
  `read_user_stats` / `write_user_stats`.
- **SQLite** is opened fresh per call (`check_same_thread=False`) — no shared
  handles. The `active_scans` dict rejects a second `/soupyscan` on a guild already
  being scanned, so two scans can't race on the same DB.
- **RAG has two distinct guards, solving two distinct problems:** a *global*
  `_embed_sem` semaphore caps concurrent embedding calls to LM Studio (a shared
  external resource), while a *per-guild* `_reindex_lock` prevents two reindexes
  from racing on one guild's `rag_chunks` table. Reindexing guild A never blocks
  retrieval or reindex of guild B.
- **The self-knowledge accumulator** is guarded by `_acc_lock` and disk-backed so
  in-flight interactions survive a restart.
- **The OpenAI client is intentionally not shared.** Each cog calls
  `openai_client()` for its own instance — the SDK is cheap to build and sharing one
  across event loops/threads is unsafe.

---

## 8. Bot ↔ web IPC

The two processes share state through the filesystem only — there is no socket or
RPC between them. Five channels:

1. **`.env-stable`** — web writes (comment-preserving, with a timestamped backup);
   bot reads on next start. Configuration.
2. **`data/runtime_flags.json`** — web writes; bot reads on the hot path, cached by
   mtime so it's ~free per message and re-read only when the file changes. Live
   toggles (RAG on/off, per-command disables) with no restart.
3. **`data/bot_dashboard.json`** — bot writes every 15 s; web reads for the status
   panel.
4. **`scan_triggers/`** — web drops a trigger file; the bot's `scan_trigger_loop`
   picks it up. Manual scan requests.
5. **Bot stdout/stderr** — captured by `BotRunner` via PTY, ANSI-stripped, fanned
   out to browsers over `/ws/logs`.

Plus the web app reads the per-guild SQLite databases directly for stats, profile,
and RAG status endpoints.

---

## 9. Cross-module coupling to be aware of

- **dailypost ↔ bluesky cycle.** `dailypost` imports `_fetch_og_image` and
  `_post_url` from `bluesky` (to cross-post articles); `bluesky` imports
  `_extract_date_from_html` and `_estimate_article_age_days` from `dailypost` (for
  article-freshness checks). This is a deliberate code-reuse cycle, not an accident —
  but it means the two cogs cannot be split into separate packages without breaking
  the import, and a change to either function's signature ripples across both.
- **Main bot → database tier.** The chat path imports RAG, self-knowledge, and
  runtime-flag helpers directly from `soupy_database`.
- **All cogs → `soupy.settings` + `soupy.prompts`.** Shared config and persona
  loading.

---

## 10. External services and graceful degradation

| Service | Used for | Config | Degrades to |
|---------|----------|--------|-------------|
| LM Studio (OpenAI-compatible) | Chat, embeddings, vision | `OPENAI_BASE_URL`, `LOCAL_CHAT`, `VISION_MODEL` | Chat fails loudly; RAG/embeddings fall back to keyword-only |
| Stable Diffusion backend | Image gen | `SD_SERVER_URL` (+ img2img/inpaint/outpaint fallbacks) | `/sd` reports backend offline |
| DuckDuckGo | Web + image search | — (rotates backends on timeout) | Command reports failure |
| Bluesky AT Protocol | Autonomous posts | `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`, `BLUESKY_AUTO_REPLY` | Cog no-ops if credentials missing |

The bot fails fast at import on missing **critical** env (`DISCORD_TOKEN`,
`SD_SERVER_URL`, `REMOVE_BG_API_URL`). It does not currently health-check that those
URLs point at *live* services until first use — see [ROADMAP.md](ROADMAP.md).

---

## 11. Where to attach new work

| You want to… | Do it here |
|--------------|------------|
| Add a self-contained feature with its own command/loop | New cog in `soupy/cogs/` (see [CONTRIBUTING.md](CONTRIBUTING.md)) |
| Add a config knob | `@cached_property` in `soupy/settings.py` + document in `docs/ENV_REFERENCE.md` + `.env-stable.example` |
| Change a persona/prompt | `prompts/<name>.default.txt` (or `.env-stable` for the legacy `BEHAVIOUR*`) |
| Change the response decision | `soupy/triggers.py` (it's pure and tested) |
| Add a live web toggle | `runtime_flags.py` + an `/api/runtime-flags` field + bot-side read |
| Persist new state | A file under `data/`, written atomically, owned by one module |
| Change the chat pipeline | `process_chat_message` — tread carefully; add a test |
