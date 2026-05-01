![Soupy Header](https://i.imgur.com/JNbVjY3.png)

![Soupy Remastered Header](https://i.imgur.com/AiCorTA.jpeg)

A fully-local, autonomous Discord bot that chats with personality, remembers conversations, posts on its own, and generates images — all running against your own LLM server.

[Soupy's Discord Server](https://discord.gg/GAv9umz5RB) · [Buy Me A Coffee](https://buymeacoffee.com/sneezeparty)

---

## What Soupy Is

Soupy runs entirely on your own hardware against [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible LLM server) and an optional Stable Diffusion backend. No cloud APIs, no per-token costs, no data leaving your network. It's opinionated and autonomous — given the chance, it will decide on its own when to chime in, share an article, or post on Bluesky. A FastAPI web panel runs alongside the bot and is the primary way you operate it: start/stop, live logs, model switching, env editing, and per-loop toggles all live there.

## Features

- Conversational chat with a configurable personality and a self-knowledge document Soupy maintains about itself.
- Retrieval-augmented memory pulled from a per-guild SQLite archive of your server's history.
- LLM-generated user profiles built from each member's message history and used to tailor replies.
- Autonomous daily article posts: reads the room, finds something the channel would actually care about, writes a take, posts it.
- Autonomous Bluesky presence — replies, quote-posts, and original article posts on a randomized daily schedule, fact-checked against the source.
- Periodic "thinking out loud" musings in a configured channel.
- DuckDuckGo web and image search with LLM-summarized results.
- Stable Diffusion image generation with remix, outpaint, edit, and random-prompt buttons.
- Optional vision: routes Discord image attachments through a vision-capable LLM.
- FastAPI web control panel for process control, live logs, env editing, stats, theming, and per-loop toggles.

## What Soupy Accesses In Your Server

Before inviting Soupy, here's what it actually does once it's in your server:

- **Listens** in the channels listed in `CHANNEL_IDS`. Outside those channels it only responds when @-mentioned or when "soup" / "soupy" appears in a message.
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
- Optional: a separate GPU host for the Stable Diffusion backend
- Optional: a Bluesky account with an app password

## Documentation

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
