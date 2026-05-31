# Roadmap

A prioritized, opinionated backlog for Soupy's future upgrades and polish. Items
come from a code audit (May 2026) plus the architectural seams described in
[ARCHITECTURE.md](ARCHITECTURE.md). This is a starting point, not a commitment —
prune, reorder, and reject freely.

**How to read this:** each item has an effort sense (S / M / L) and a rationale.
Effort is rough: **S** = an afternoon, **M** = a few days, **L** = a week-plus or
touches a load-bearing surface.

**Status of the basics (so we don't relitigate them):** CI already runs `ruff` +
`pytest` on Python 3.10/3.11/3.12 plus an installer smoke test; `requirements.lock`
is committed and the installer prefers it; `pre-commit` carries a `gitleaks` hook;
and `soupy/` core (triggers, settings, prompts, env writer, validators, state) is
unit-tested. The gaps below are real gaps, not missing fundamentals.

---

## Now — high value, low-to-medium effort

These pay for themselves quickly and reduce the risk of every future change.

### ~~1. Integration test for the chat pipeline~~ ✅ Shipped
Done: `tests/test_process_chat_message.py` drives `process_chat_message` with the
OpenAI client, RAG, DB, and Discord objects mocked (happy path, candidate/judge
selection, RAG inject/skip, metadata scrubbing, long-reply splitting). `tests/conftest.py`
centralizes the env-before-import preamble. See CHANGELOG. *Follow-on worth doing
later: assert specific token-budget numbers, and cover the URL-content and
image-description branches.*

### ~~2. Audit `except Exception: pass` in the web panel~~ ✅ Shipped
Done: file-read/operation swallow blocks in `web/app.py` now log with `exc_info`
(WARNING for config/file failures, DEBUG for polled dashboard reads); the
`SuppressNoisyAccess` fail-open is documented as deliberate; per-line JSONL skips
remain silent `continue`s. See CHANGELOG.

### ~~3. User-facing feedback on cog network failures~~ ✅ Shipped
Done: `perform_text_search` raises a typed `SearchBackendError` on total backend
failure so `/soupysearch` reports a network error vs. "no results"; `/soupyimage`
wraps its DDGS call likewise; `dailypost._fetch_article_content` now logs failures.

### ~~4. Atomic `.env-stable` writes from the web panel~~ ✅ Shipped
Done: `env_store.write_env` writes to a temp file and `os.replace`s it into place.

### ~~5. Document the env-var surface~~ ✅ Shipped (initial pass)
Done: the genuinely-undocumented vars (`RANDOM_RESPONSE_RATE`, RAG tuning knobs,
daily-post scheduling, `SOUPY_DB_DIR`, `CHANNEL_NAMES*`) are now in
`docs/ENV_REFERENCE.md` + `.env-stable.example`. The deeper goal — making
`soupy/settings.py` the single source for all 228 reads — remains as item 7.

---

## Next — structural cleanups that unlock everything after

### ~~6. Extract image generation into `soupy/cogs/sd.py`~~ ✅ Shipped
Done: ~2,050 lines (commands, button handlers, views, `generate_sd_image`) moved
out; main file dropped ~6,255 → ~4,200 lines. `SDQueue` stayed in the main module
(it also dispatches `"chat"` jobs and must run before cogs load) and reaches the
moved functions via a lazy `from soupy.cogs import sd` import. `tests/test_sd_cog_loads.py`
guards it. See CHANGELOG.
*Follow-on still open: move the utility commands (`/8ball`, `/9ball`, `/whattime`,
`/weather`, `/status`) into `soupy/cogs/utilities.py` to shrink the main file further.*

### 7. Centralize the remaining env reads into `soupy/settings.py` (M) — *partially done*
Follow-on to #5. `Settings` already covers 87 vars; the goal is for it to be the
*only* place env is read, with typed accessors and bounds validation.
**Done:** bounds validation (`minimum`/`maximum` on `_env_int`/`_env_float`, applied
to the key numeric settings — so `SD_DEFAULT_WIDTH=0` now falls back with a warning
instead of generating a broken request). `MAX_TOKENS=abc` was already handled.
**Still open:** the bulk migration of the main bot's inline `os.getenv` calls (and the
SD-config constants the new `sd.cog` reads off the live module) into `settings.*`.
*Deferred deliberately:* that migration heavily rewrites the 4,200-line main file and
re-touches the freshly-extracted SD cog; it should be done with the ability to
live-test the bot, not blind. Best tackled after image-gen is confirmed working live.

### 8. Decide the fate of the React rewrite (M, mostly a decision)
There are two front ends: the shipped, full-featured vanilla-JS dashboard
(`web/static/dashboard/dashboard-app.js`, ~3,800 lines) and a thin React+Vite
scaffold (`web/frontend/`) that only shows bot status and is **not served**. Either
commit to finishing the React migration (and set a feature-parity bar) or delete the
scaffold so it stops implying a migration that isn't happening. Carrying both is the
worst option — it's the kind of ambiguity that rots.
*Recommendation:* if nobody is actively porting features, delete `web/frontend/` and
keep the vanilla dashboard as the supported UI; revisit only if the dashboard's size
becomes a real maintenance problem.

### 9. Standardize aiohttp session lifecycles across cogs (S)
`search`/`imagesearch` cache one session per cog and close it in `cog_unload`;
`dailypost` and `bluesky` create a fresh `ClientSession` per fetch (15+ sites in
bluesky alone). Pick the cached-session pattern everywhere and add explicit
`ClientTimeout` to each — several external calls currently inherit a multi-minute
default, so a hung upstream can stall a command for minutes.

---

## Later — scalability and resilience (do when load justifies it)

### 10. Bot health supervision / auto-restart (M)
`BotRunner` reports the bot as "running" whenever the process exists, even if it has
hung or is wedged. Add a heartbeat (the bot already writes `bot_dashboard.json` every
15 s — treat a stale timestamp as unhealthy) and an optional bounded auto-restart
(e.g. restart on N crashes/hour with backoff, then give up and alert).

### 11. Incremental / resumable RAG reindex (L)
Reindex is a full table scan + re-embed. On a first setup with a large backlog it
runs for many minutes and, if interrupted, starts over. Track a watermark
(`last_indexed_message_id`) and embed only new messages on the common path, reserving
the full rebuild for embedding-model changes.
*Why later:* it works today; this is about first-run UX and large servers.

### 12. Recent-message / context caching on the chat hot path (M)
Each reply re-fetches recent history, re-estimates tokens, and (if enabled) issues an
embedding query. In a busy channel that's a lot of duplicated DB + embedding work per
minute. A small per-channel cache of the recent-message window, invalidated on new
messages, would cut it down. Pair with an explicit embedding-request timeout so a
slow LM Studio degrades gracefully instead of backing up the queue.

### 13. Per-guild SQLite schema versioning (M)
Schema changes are applied as idempotent `ALTER`-if-missing migrations on connect
(e.g. `user_profiles._migrate_profile_columns`). This works but has no version
record, so it's hard to reason about what migrations a given guild DB has seen. Add a
`schema_version` row and a small ordered migration runner invoked at `init_database`.
*Why later:* the current approach is correct; this is about future-proofing as the
schema grows.

### 14. Optional auth/IP-guard for the web panel (M)
`/api/*` is unauthenticated by design ("trusted LAN"), and the code says so. That's a
reasonable default, but `POST /api/bot/restart` and `POST /api/env/save` are fully
unprotected if the port is ever exposed. Add *optional* `WEB_API_KEY` / `WEB_ALLOWED_IPS`
middleware that defaults to off, so security-conscious operators can lock it down
without changing the default experience.

---

## Polish — low urgency, nice to have

- **Self-knowledge size cap (S).** `self_context` builds the injected document; add
  an explicit ceiling so a large core can never push the system prompt past the
  context window. (The anchor tier mostly addresses this, but a hard cap is cheap
  insurance.)
- **Code-block-aware message splitting (S).** `split_message` splits on newlines and
  can break a fenced ```code block``` across two Discord messages.
- **Log hygiene (S).** Standardize levels per subsystem (some cogs are chatty, some
  silent) and consider a lint rule discouraging emoji in log strings — they make
  `grep`/`awk` over `logs/soupy.log` harder.
- **Dashboard poll interval (S).** The profile-batch status endpoint is polled
  aggressively enough to need a log-suppression filter; raising the interval (or
  pushing over the existing WebSocket) removes the cause rather than hiding it.
- **State-file rotation (S).** `user_stats.json` and the JSONL archives grow
  unbounded; add a size cap or monthly rotation for long-lived deployments.
- **Cog reload integration test (S).** `cog_unload` closes sessions/cancels loops but
  isn't exercised by a test; a reload that hangs on `session.close()` would wedge the
  bot.

---

## Explicitly *not* doing (and why)

- **Multi-process bot scaling / sharding.** Soupy is a single-deployment,
  small-number-of-guilds bot. The per-guild SQLite model is fine at that scale;
  don't add a connection pool or a multi-tenant schema speculatively.
- **Swapping SQLite for a server DB.** Same reason — local-first is a feature, not a
  limitation to engineer away.
- **Pinning `requirements.txt` with `==`.** The `>=` pins plus the committed
  `requirements.lock` already give "auto-pick-up-fixes" *and* "reproducible when you
  want it." Leave it.

---

*Keep this file honest: when an item ships, move it to [CHANGELOG.md](CHANGELOG.md)
and delete it here. A roadmap full of done items is just a second changelog.*
