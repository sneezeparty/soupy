![Soupy Header](https://i.imgur.com/JNbVjY3.png)

![Soupy Remastered Header](https://i.imgur.com/AiCorTA.jpeg)

A fully-local Discord bot with a configurable personality, retrieval-augmented memory of your server, autonomous Discord and Bluesky posting, web search, image understanding, and two independent image-generation backends — all running against your own LLM server with no cloud API in the loop.

<p align="center">
  <a href="https://discord.gg/GAv9umz5RB">
    <img src="https://img.shields.io/badge/Join-Soupy's%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join Soupy's Discord">
  </a>
  &nbsp;&nbsp;
  <a href="https://buymeacoffee.com/sneezeparty">
    <img src="https://img.shields.io/badge/Buy%20me-a%20coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee">
  </a>
</p>

---

## What Soupy Is

Soupy is a two-process application — a Discord bot and a FastAPI web control panel that supervises it — built to run entirely on hardware you own. Chat, embeddings, summarization, vision, and prompt expansion all go through [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible LLM server). Image generation has two paths: a remote Stable Diffusion backend reached over HTTP (`/sd`, `/img2img`, `/inpaint`, `/outpaint`) and a local Flux/MLX backend that runs on the same Mac as the bot via `flux_server.py` (`/flux`, with optional FLUX.2 Klein-Edit for genuine prompt-driven image editing). No cloud APIs, no per-token costs, no data leaving your network.

Personality is the point. Soupy maintains a self-knowledge document about itself that it edits during a nightly reflection pass, builds structured profiles of the users it sees, and pulls relevant context out of a per-guild SQLite archive of your server's history. Given the chance, it will decide on its own when to chime in, share an article, or post on Bluesky. The web panel is the operator surface: start/stop, live logs, env editing, per-loop toggles, archive browsing, model probing, and theming all live there.

## Features

- Conversational chat with a configurable personality and a self-knowledge document Soupy maintains and rewrites about itself.
- Retrieval-augmented memory pulled from a per-guild SQLite archive of your server's history, with a cosine-similarity floor so off-topic chunks don't get injected.
- LLM-generated user profiles built from each member's message history and used to tailor replies.
- Nightly self-reflection at a configurable hour (default 3 AM) that distills opinions, relationships, and a short identity anchor from the day's interactions.
- Autonomous daily article posts: reads the room, finds something the channel would actually care about, writes a take, posts it, optionally cross-posts to Bluesky.
- Autonomous Bluesky presence — replies, quote-posts, and original article posts on a randomized daily schedule, fact-checked against the source.
- Periodic "thinking out loud" musings in a configured channel, with dedupe against recent topics via keyword filtering and embedding similarity.
- DuckDuckGo web and image search with LLM-summarized results.
- Optional vision: routes Discord image attachments through LM Studio's vision-capable LLM for image understanding.
- **Two independent image-generation backends, sharing a single serial queue:**
  - **Stable Diffusion** (`/sd`, `/img2img`, `/inpaint`, `/outpaint`) over HTTP to a separate GPU host.
  - **Local Flux** (`/flux`) via mflux/MLX on Apple Silicon — text-to-image, noise-mix image-to-image, and FLUX.2 Klein-Edit for genuine prompt-driven editing — running on the same Mac as the bot via a small `flux_server.py` FastAPI process.
- Post-generation control panels with Remix, Fancy, R-Fancy, R-Keyword, Edit, Wide, and Tall buttons on both backends (plus Outpaint on `/sd`, img2img-from-display on `/flux`).
- FastAPI web control panel for process control, live logs, env editing, stats, theming, per-loop toggles, and a light-table view of every image the bot has generated.

## What Soupy Accesses In Your Server

Before inviting Soupy, here's what it actually does once it's in your server:

- **Listens** in the channels listed in `CHANNEL_IDS`. Outside those channels it only responds when @-mentioned or when a keyword from `SOUPY_TRIGGER_KEYWORDS` appears in a message. Defaults: `soup,gumbo` (`soup` also catches `soupy`).
- **Posts** in any channel where it's been triggered, plus the configured daily-post and musing channels for autonomous activity.
- **Archives** message history into a per-guild SQLite database (`soupy_database/databases/guild_<id>.db`) and embeds it for retrieval-augmented generation.
- **Profiles users** — generates structured summaries of opinions, hobbies, and interests from the message history of people it sees, and uses them to tailor replies.
- **Fetches URLs** dropped in chat and reads the article content so it can respond to what's actually there.
- **Reads images** through a vision-capable LLM if `ENABLE_VISION` is on. Off by default.
- **Posts on its own** — random in-channel interjections, twice-daily article posts, and Bluesky activity if Bluesky is configured.

To scope it tightly: keep `CHANNEL_IDS` short, leave the autonomous loops and vision off, and invite the bot only to channels where you want it. Every one of the above is toggleable from the web panel.

## Quick Start

Have [LM Studio](https://lmstudio.ai/) running with a chat model **and** an embedding model loaded. Then:

```bash
git clone https://github.com/sneezeparty/soupy.git
cd soupy
python install.py
```

The interactive installer walks you through Discord setup, LM Studio probing, optional integrations (Bluesky, Stable Diffusion, daily posts), and writes `.env-stable`. It validates the Discord token and the loaded LM Studio models live, then offers to launch `python run_all.py`. See [INSTALL.md](INSTALL.md) for flags (`--dry-run`, `--resume`, `--minimal`, `--non-interactive`).

After the bot is running, run `/soupyscan` once per guild as an owner to archive history. The first scan can run for hours or days on busy servers — see `docs/SETUP.md` for tuning.

## Requirements

- Python 3.10+
- LM Studio (or any OpenAI-compatible server) with a chat model and an embedding model loaded
- Optional: a separate GPU host for the Stable Diffusion backend (powers `/sd`, `/img2img`, `/inpaint`, `/outpaint`)
- Optional: Apple Silicon Mac with enough unified memory to run mflux (powers `/flux`; ~8-12 GB at `FLUX_QUANTIZE=4` for FLUX.1 schnell, more for FLUX.2 Klein-Edit). Runs on the same host as the bot.
- Optional: a vision-capable LLM in LM Studio (set `ENABLE_VISION=true` and `VISION_MODEL`)
- Optional: a Bluesky account with an app password

## Documentation

For contributors and maintainers:

- `ARCHITECTURE.md` — how the two processes, cogs, and database tier fit together
- `CONTRIBUTING.md` — dev setup, conventions, and how to add a cog / config var / prompt
- `ROADMAP.md` — prioritized backlog of planned upgrades and polish

For operators:

- `docs/SETUP.md` — full first-run walkthrough, archive scan tuning, RAG verification
- `docs/WEB_PANEL.md` — web control panel deep dive
- `docs/ENV_REFERENCE.md` — every variable in `.env-stable`
- `docs/BLUESKY.md` — Bluesky integration, schedule, quality controls
- `docs/HARDWARE.md` — Stable Diffusion backend setup, GPU sizing, Apple Silicon notes
- `docs/CUSTOMIZATION.md` — personality, theming, web panel rebranding
- `docs/EXAMPLES.md` — usage screenshots

## Commands

`/helpsoupy` lists every command.

## License

MIT License — Copyright (c) 2024-2026 sneezeparty. The software is provided "as is" without warranty of any kind. See [CHANGELOG.md](CHANGELOG.md) for release history and [GitHub Issues](https://github.com/sneezeparty/soupy/issues) for bugs.

## Links

- [Soupy's Discord Server](https://discord.gg/GAv9umz5RB) — try it out
- [Buy Me A Coffee](https://buymeacoffee.com/sneezeparty) — support the project
