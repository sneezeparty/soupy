# Installing Soupy

```bash
python install.py
```

The wizard establishes what you want Soupy to do, creates `.venv` (and
`.venv-sd` if you opted into local image generation), validates your
Discord token and LM Studio models live, and writes `.env-stable`. At the
end it offers to launch `python run_all.py`.

## Flags

- `--dry-run` — walk the prompts but don't create venvs, install
  packages, or write `.env-stable`. Print what would happen.
- `--resume` — read `.install-state.json` and skip steps that already
  finished. Use this after a crash, network blip, or `Ctrl-C`. Secrets
  are never written to the state file, so the wizard re-prompts for any
  it needs.
- `--minimal` — skip the optional-integrations step entirely. Defaults
  image generation, Bluesky, vision, and daily posts to off. Still
  prompts for Discord and LM Studio.
- `--non-interactive --config <path>` — read every answer from a JSON
  file instead of stdin. Object keys are the prompt labels (lower-cased,
  trailing `:` / `?` stripped). Useful for scripted installs.
- `--reset` — delete `.install-state.json` before starting.

## What it does, step by step

1. **Goals** — picks which downstream paths apply. Image gen can be
   *none*, *NVIDIA CUDA on this machine* (Stable Diffusion), *Apple
   Silicon on this machine* (Flux/mflux — recommended on a Mac, or
   Stable Diffusion via MPS for the older path), or *on a different
   machine* (Stable Diffusion over HTTP). Also asks about Bluesky,
   vision (image descriptions in chat), and autonomous daily article
   posts.
2. **Environment** — verifies Python ≥ 3.10, creates `.venv`, installs
   `requirements.txt`. If you picked Flux, also installs `mflux` +
   `python-multipart` into the same `.venv`. If you picked CUDA or MPS
   Stable Diffusion, builds a separate `.venv-sd` with the right PyTorch
   flavour and `sd-api/requirements*.txt`.
3. **Discord** — prints the developer-portal checklist, opens the portal
   in your browser, validates the token live against `GET /users/@me`.
4. **LM Studio** — probes `/v1/models`, lets you pick chat + embedding
   (and vision, if enabled) from whatever's loaded. If only one model is
   loaded, walks you through loading the second one. Populates
   `AVAILABLE_MODELS` (the web-UI dropdown) from the live list.
5. **Optional** — Bluesky credentials, remote SD URL, Flux host/port +
   Klein-Edit toggle, and daily-post channel map. Skipped entirely under
   `--minimal`.
6. **Write config** — renders `.env-stable` from `.env-stable.example`,
   preserving comments and ordering. Backs up any existing
   `.env-stable` to `.env-stable.bak.<timestamp>` first and prints a
   diff (with secrets masked) before writing.
7. **Verify** — re-runs every live check against the rendered config.
   Failures offer retry / skip / abort.
8. **Handoff** — prints where things landed and offers to exec
   `python run_all.py`.

## After installation

Run `/soupyscan` once per guild as an owner. The first scan archives
your server's history and embeds it for RAG. On a busy server this can
take hours or days — see `soupy_database/SETUP.md` for tuning.

If you picked the Flux backend, launch it separately in another
terminal — `flux_server.py` holds the model resident so the bot process
stays light:

```bash
source .venv/bin/activate
python flux_server.py
```

`/flux` will report "Flux is not configured" until that server is up.

## Reproducible installs (optional)

`requirements.txt` uses `>=` pins so a fresh install picks up bug-fix
releases automatically. If you want a frozen snapshot, generate
`requirements.lock` once and check it in:

```bash
pip install pip-tools
pip-compile --output-file=requirements.lock requirements.txt
```

The installer prefers `requirements.lock` when it's present and falls
back to `requirements.txt` otherwise.

## Development

For working on Soupy itself:

```bash
pip install -e ".[dev]"        # ruff, black, pytest, pre-commit, pip-tools
pre-commit install             # wire hooks to git
ruff check .                   # lint
pytest                         # run tests
```

## Frontend (developers only)

End users do not need Node.js — the dashboard's compiled JavaScript is
shipped in `web/static/dashboard/`, and the Vite production bundle in
`web/static/dashboard/dist/` is checked into git. If you are working on
the React rewrite under `web/frontend/`, run `npm install` once in that
directory and use `npm run dev` for hot-reload during development or
`npm run build` to refresh the production bundle. See
[`web/frontend/README.md`](web/frontend/README.md) for details. Running
`python install.py` is unaffected.

## Replaces

The interactive wizard replaces the manual 7-phase walkthrough that
previously lived in `README.md` (Phases 1–5 are now `python install.py`;
Phase 6 / archive scan is the post-install reminder above).
