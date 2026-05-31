# Contributing to Soupy

This guide is for working *on* Soupy. It assumes you've read
[ARCHITECTURE.md](ARCHITECTURE.md) — the patterns below only make sense against
that picture. For installing and running the bot as an operator, see
[INSTALL.md](INSTALL.md) and [`docs/`](docs/).

---

## Dev environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # ruff, black, pytest, pre-commit, pip-tools
pre-commit install             # wire the gitleaks hook to git
pip install -r requirements.txt
```

Before every commit (CI runs the same on Python 3.10/3.11/3.12):

```bash
ruff check .
pytest
```

`pre-commit` runs a `gitleaks` secret scan on commit. Never commit `.env-stable`
or any `.env-stable.bak.*` — they're gitignored, keep it that way.

---

## The golden rules

These are the conventions that keep the codebase coherent. Most map directly to an
invariant in ARCHITECTURE.md.

1. **All I/O is async.** Blocking work (PIL, the OpenAI SDK, sync HTTP) goes through
   `asyncio.to_thread`. Never block the event loop.
2. **Persist state atomically.** Any file under `data/` is written via
   `tmp + os.replace`, never in place. A crash mid-write must not corrupt it.
3. **Read config through `soupy/settings.py`.** Add a `@cached_property`; don't
   sprinkle new `os.getenv` calls. (Exception: things that must change *without* a
   restart read env per-call on purpose — see `soupy/triggers.py`. Match the existing
   pattern, don't invent a third.)
4. **Load prompts through `soupy/prompts.py`.** Ship a default; let users override.
5. **One module owns each piece of state.** If you need new persistent state, give
   it a clear owner and a single read/write path.
6. **Fail fast on missing *critical* config, degrade gracefully on missing
   *optional* services.** A missing `DISCORD_TOKEN` should crash at import; a missing
   Bluesky credential should make the Bluesky cog a no-op, not break the bot.
7. **When you change bot behavior, check whether the web panel needs to follow.**
   New env var → env editor + `docs/ENV_REFERENCE.md`. New runtime toggle →
   `runtime_flags.py` + an `/api/runtime-flags` field. New command → `/helpsoupy`.

---

## Where code goes

| What you're building | Where it lives |
|----------------------|----------------|
| A feature with its own command and/or autonomous loop | New cog in `soupy/cogs/` |
| Core chat or image-gen behavior | `soupy_remastered_stablediffusion.py` (the main file) |
| A reusable helper / pure logic | `soupy/` package (so it can be unit-tested) |
| A config knob | `soupy/settings.py` |
| A persona / system prompt | `prompts/<name>.default.txt` |
| Database / RAG / profile / self-knowledge logic | `soupy_database/` |
| A web endpoint or panel feature | `web/app.py` (+ `web/static/dashboard/`) |

Prefer adding to a cog or the `soupy/` package over growing the main file — it's
already 6 K+ lines and the [roadmap](ROADMAP.md) calls for shrinking it, not feeding
it.

---

## How to add a cog

Cogs are the primary extension point. The shape is consistent — copy an existing one
(`soupy/cogs/search.py` is the simplest; `dailypost.py` is the fullest example with a
loop + persisted state + dashboard hook).

```python
# soupy/cogs/mything.py
"""One-paragraph module docstring: what this cog does, any cross-cog imports,
and the gotcha a future reader most needs to know."""

import discord
from discord import app_commands
from discord.ext import commands, tasks

from soupy.settings import openai_client, settings
from soupy import prompts as soupy_prompts

client = openai_client()  # one per cog — do NOT share across cogs (see ARCHITECTURE §7)


class MyThingCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._loop.start()              # only if you have an autonomous loop

    def cog_unload(self) -> None:
        self._loop.cancel()             # cancel loops, close aiohttp sessions here

    @tasks.loop(seconds=60)
    async def _loop(self) -> None:
        if not settings.mything_enabled:   # re-read the flag each tick (live toggle)
            return
        ...

    @_loop.before_loop
    async def _before_loop(self) -> None:
        await self.bot.wait_until_ready()

    @app_commands.command(name="mything", description="…")
    async def mything(self, interaction: discord.Interaction):
        await interaction.response.defer()      # defer before any slow work
        ...


async def setup(bot: commands.Bot) -> None:     # required by discord.py
    await bot.add_cog(MyThingCog(bot))
```

Then register it in `load_extensions()` in the main file:

```python
await bot.load_extension("soupy.cogs.mything")
```

Checklist:

- [ ] `setup(bot)` coroutine at module bottom.
- [ ] Loaded in `load_extensions()`.
- [ ] Config via `settings`; loop enable-flag re-read **per tick** so it's live-toggleable.
- [ ] Slow commands `defer()` first.
- [ ] `cog_unload` cancels loops and closes sessions.
- [ ] Persisted state (if any) under `data/`, written atomically.
- [ ] Owner-only commands gated on `OWNER_IDS`.
- [ ] If it should surface in the panel, push timer state to the dashboard like
      `dailypost`/`musings`/`bluesky` do, and add it to `/helpsoupy`.
- [ ] Cross-cog imports documented in the module docstring (see the
      dailypost ↔ bluesky cycle in ARCHITECTURE §9 — don't add a new cycle casually).

---

## How to add a config variable

1. Add a `@cached_property` to `soupy/settings.py` using the `_env_*` helpers
   (`_env_str`, `_env_int`, `_env_bool`, `_env_str_list`). Give it a sane default.
2. Read it as `settings.my_var` wherever you need it.
3. Add it to **`.env-stable.example`** with a comment (this is what the installer
   renders from — it is not auto-synced).
4. Document it in **`docs/ENV_REFERENCE.md`**.
5. If it should be editable from the panel, it already is — the env editor reads
   `.env-stable`. **Exception:** the long prompt vars (`BEHAVIOUR`, `BEHAVIOUR_SEARCH`,
   `9BALL`) must **never** be edited through the web form — it can truncate multi-line
   quoted values. Edit `.env-stable` directly. Leave a note if you add another long
   value.

The cached layer means changes to `.env-stable` aren't visible until the cache is
cleared (`reloadenv` / a bot restart). If a value genuinely must change without a
restart, read it per-call instead and document why (the `triggers.py` pattern).

---

## How to add a prompt

1. Ship the default as `prompts/<name>.default.txt` (tracked in git).
2. Load it: `soupy_prompts.load_prompt("<name>", fallback="…")`.
3. Users override by creating `prompts/<name>.txt` (gitignored).

Resolution order is legacy env var → `prompts/<name>.txt` → `prompts/<name>.default.txt`
→ caller fallback. Legacy env-var prompts emit a one-time deprecation warning.

---

## How to add a live (no-restart) toggle

For on/off switches the operator should flip from the panel without restarting the
bot:

1. Add the flag to `data/runtime_flags.json`'s defaults in
   `soupy_database/runtime_flags.py`.
2. Add a read helper (mirror `is_rag_enabled` / `is_command_disabled` — they're
   mtime-cached, so reading per message is ~free).
3. Read it on the bot side where it matters.
4. Add a field to the `/api/runtime-flags` GET/POST handlers in `web/app.py` and a
   control in the dashboard.

Use this channel — not `.env-stable` — for anything that needs to take effect *now*.

---

## Testing

Tests live in `tests/` and run under `pytest`. Current coverage is the `soupy/`
core: `triggers`, `settings`, `prompts`, the env writer, validators, and state
helpers. The conspicuous gaps (chat pipeline, web endpoints, RAG, cogs) are tracked
in [ROADMAP.md](ROADMAP.md).

Guidelines:

- New pure logic in `soupy/` should ship with a test — that's *why* it lives there
  rather than in the main file.
- Mock external services (OpenAI client, SQLite, aiohttp). Tests must not need a live
  LM Studio, SD backend, or network.
- If you touch `process_chat_message`, the token-budget math, or `clean_response`,
  add or extend a test — that path is fragile and unguarded end-to-end.
- Put shared fixtures in `tests/fixtures/`.

---

## The two-process gotcha (don't get caught by this)

`run_all.py` starts uvicorn, and uvicorn's startup hook starts the bot as a child
process (see ARCHITECTURE §2). So:

- **Changed bot code?** Restart the **bot** (panel button or `POST /api/bot/restart`).
  Restarting uvicorn does *not* reload bot code.
- **Changed web code?** Restart uvicorn (`run_all.py`).
- **Changed `.env-stable`?** Restart the bot — env is merged into the child on start.

For fast iteration on the bot alone (and clean stack traces), run it directly:

```bash
python soupy_remastered_stablediffusion.py
```

---

## Commits, changelog, and releases

- Keep [CHANGELOG.md](CHANGELOG.md) current under `## [Unreleased]`, grouped into
  **Added / Changed / Fixed / Docs**. Write entries that explain the *why*, matching
  the existing style — the changelog doubles as design rationale here.
- The installer surface is sensitive: when a change touches the env-var surface, the
  Discord connection, the LM Studio probe, or the SD endpoints, check whether
  `install.py` / `installer/steps/sNN_*.py` need a matching update (see
  [CLAUDE.md](CLAUDE.md)).
- Versioning is SemVer-ish but, per the changelog's own note, a guideline for a
  single-deployment bot rather than a contract.

---

## Touching the web front end (developers only)

End users never need Node — the dashboard's compiled JS ships in
`web/static/dashboard/`. The active UI is the vanilla-JS dashboard
(`web/static/dashboard/dashboard-app.js`). A React+Vite rewrite scaffold exists under
`web/frontend/` but is **not currently served**; its fate is an open question in
[ROADMAP.md](ROADMAP.md). Don't assume work there reaches users until that's
resolved. If you do work on it: `npm install` once in `web/frontend/`, then
`npm run dev` (hot reload) or `npm run build` (refresh the bundle). See
[`web/frontend/README.md`](web/frontend/README.md).
