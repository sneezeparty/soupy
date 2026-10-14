# Installing Soupy

```bash
python3 install.py        # Windows: python install.py
```

The wizard needs Python 3.10 or newer and checks that before asking
anything. It establishes what you want Soupy to do, creates `.venv` (and
`.venv-sd` if you opted into local Stable Diffusion), validates your
Discord token and LM Studio models live, and writes `.env-stable`. At the
end it offers to launch Soupy with `.venv`'s Python.

## Flags

- `--dry-run`: walk the prompts without creating venvs, installing
  packages, or writing `.env-stable`. Prints what would happen and leaves
  no `.install-state.json` behind.
- `--resume`: read `.install-state.json` and skip steps that already
  finished. Use this after a crash, network blip, or `Ctrl-C`. Secrets
  are never written to the state file. The wizard reads them back from
  `.env-stable` if an earlier session got as far as writing it. Otherwise
  it runs the step that collects them again, with the other answers
  filled in as defaults.
- `--minimal`: skip the optional-integrations step entirely. Defaults
  image generation, Bluesky, vision, and daily posts to off. Still
  prompts for Discord and LM Studio.
- `--locked`: install the pinned versions in `requirements.lock` instead
  of resolving `requirements.txt`. See "Reproducible installs" below.
- `--non-interactive --config <path>`: read every answer from a JSON
  file instead of stdin. Object keys are the prompt labels (lower-cased,
  trailing `:` / `?` stripped). Useful for scripted installs.
- `--reset`: delete `.install-state.json` before starting.

## What it does, step by step

1. **Goals.** Picks which downstream paths apply. Image gen can be
   *none*, *NVIDIA CUDA on this machine* (Stable Diffusion), *Apple
   Silicon on this machine* (Flux/mflux, recommended on a Mac, or Stable
   Diffusion via MPS for the older path), or *on a different machine*
   (Stable Diffusion over HTTP). Only the backends the machine can run
   are listed: the Apple Silicon options on an Apple Silicon Mac, and
   CUDA on Linux or Windows under Python 3.10 to 3.12 (its pinned PyTorch
   build has no wheels for newer Pythons). Also asks about Bluesky,
   vision (image descriptions in chat), and autonomous daily article
   posts.
2. **Environment.** Creates `.venv` and installs `requirements.txt`. If
   you picked Flux, also installs `mflux` + `python-multipart` into the
   same `.venv`. If you picked CUDA or MPS Stable Diffusion, builds a
   separate `.venv-sd` with the right PyTorch flavour and
   `sd-api/requirements*.txt`. On Debian/Ubuntu, creating a venv needs
   the `python3-venv` package; the wizard names it if creation fails.
3. **Discord.** Prints the developer-portal checklist, opens the portal
   in your browser, validates the token live against `GET /users/@me`.
4. **LM Studio.** Probes `/v1/models`, lets you pick chat + embedding
   (and vision, if enabled) from whatever's loaded. If only one model is
   loaded, walks you through loading the second one. Populates
   `AVAILABLE_MODELS` (the panel's fallback model list) from the live list.
5. **Optional.** Bluesky credentials, remote SD URL, Flux host/port +
   Klein-Edit toggle, and daily-post channel map. Skipped entirely under
   `--minimal`.
6. **Write config.** On a first install, renders `.env-stable` from
   `.env-stable.example`, preserving comments and ordering. When
   `.env-stable` already exists, updates it in place: only the values
   the wizard asked about change, and the rest of the file (the persona
   prompts, colours, tuning knobs, keys added by hand) stays. Features
   declined in step 1 are set to `false`. Lists the changes with secrets
   masked and saves the old file as `.env-stable.bak.<timestamp>`. With
   nothing to change, it writes nothing.
7. **Verify.** Re-runs every live check against the rendered config.
   Failures offer retry / skip / abort. A local Flux or Stable Diffusion
   server that isn't up yet is reported without counting as a failure,
   because those are started after the install.
8. **Handoff.** Prints where things landed and the launch command for
   any local image server, then offers to exec `run_all.py` with
   `.venv`'s Python.

## Running the wizard again

With a `.env-stable` already in place, `python3 install.py` offers three
choices:

- **reconfigure**: go through the prompts again. Each one defaults to
  what the file already says, so Enter keeps it (the Discord token
  included).
- **verify-only**: run step 7 against the file as it stands.
- **cancel**.

To start over from the template, move `.env-stable` out of the way first.

## After installation

Start Soupy with `.venv`'s Python:

```bash
.venv/bin/python run_all.py        # Windows: .venv\Scripts\python run_all.py
```

Run `/soupyscan` once per guild as an owner. The first scan archives
your server's history and embeds it for RAG. On a busy server this can
take hours or days — see `soupy_database/SETUP.md` for tuning.

If you picked the Flux backend, `flux_server.py` runs as its own
process and holds the model, so the bot process stays light. The web
panel starts it when you answered yes to that question
(`FLUX_SERVER_AUTOSTART=true`); its card on the Overview tab has Start,
Stop and Restart, and its log is `logs/flux_server.log`. Otherwise
launch it in another terminal:

```bash
source .venv/bin/activate
python flux_server.py
```

`/flux` reports that the Flux server is offline until it is up.

If you picked Stable Diffusion on this machine, its server also runs
separately, from `.venv-sd` (port 8000):

```bash
cd sd-api
../.venv-sd/bin/python sd_api.py        # sd_api-mac.py on Apple Silicon
```

## Reproducible installs (optional)

`requirements.txt` uses `>=` pins, so a fresh install resolves current
releases for whatever OS and Python it runs on. `requirements.lock` is a
frozen snapshot of one such resolution, and
`python3 install.py --locked` installs it.

A lock only fits the Python version and OS that generated it (its header
names the Python). The committed one was generated with Python 3.10,
before `matplotlib` joined `requirements.txt`. It pins `uvloop`, which
has no Windows build, and `numpy==2.2.6`, which has no wheel for Python
3.14. Regenerate it on the machine you deploy to:

```bash
pip install pip-tools
pip-compile --output-file=requirements.lock requirements.txt
```

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
`python3 install.py` is unaffected.

## Replaces

The interactive wizard replaces the manual 7-phase walkthrough that
previously lived in `README.md` (Phases 1–5 are now `python3 install.py`;
Phase 6 / archive scan is the post-install reminder above).
