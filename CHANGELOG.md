# Changelog

All notable changes to Soupy will be recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project tries to use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though for a single-deployment Discord bot, that's a guideline, not a contract.

## [Unreleased]

### Added
- `LOG_LEVEL` env var (default `INFO`). Controls what shows up in the terminal and the web log stream. The file at `logs/soupy.log` still captures `DEBUG` regardless. Tunable from the web Environment Editor.
- Logging policy comment block above the logging setup in the main bot, documenting what belongs at INFO vs. DEBUG vs. WARNING vs. ERROR. Roughly 5–10 lines per chat reply at INFO.

### Changed
- Demoted noisy chat-path INFO calls to DEBUG so the terminal stays readable: per-message DB connect (`Database initialized for guild …`), token-budget breakdown, history-trim notice, per-URL fetch outcomes.
- Reclassified Bluesky og:image processing logs: per-image resize/UA-blocked details → DEBUG; "could not fetch page" / "download failed (all UAs)" / "still too large after recompression" → WARNING.

## [1.1.0] - 2026-04-30

### Added
- `CHAT_FREQUENCY_PENALTY` (default `0.6`) and `CHAT_PRESENCE_PENALTY` (default `0.3`) env vars, passed to the chat completion call. They discourage the model from falling into the same speech template reply after reply. Set either to `0` to disable. Tunable from the web Environment Editor.
- Few-shot example block in the `BEHAVIOUR` system prompt — ten short style samples (greetings, insults, recall questions, political rants, etc.) to give small local models concrete shape to follow.
- Pre-commit `gitleaks` hook (`.pre-commit-config.yaml`). Run `pre-commit install` after cloning to activate. Catches secret-bearing commits before they land. Server-side secret scanning + push protection are also enabled on the GitHub repo.

### Changed
- Consolidated duplicated RAG memory rules. The longer-term memory guidance now lives once in the system prompt's `technical_instructions`; the per-turn RAG context message is a short pointer back to those rules + the actual snippets. Net prompt length is slightly shorter despite the new few-shot block.

## [1.0.0] - 2026-04-29

### Added
- Initial public release of Soupy Remastered: chat with personality, RAG-backed memory, autonomous Discord article posts, autonomous Bluesky engagement, web search, vision, image generation via a separate Stable Diffusion backend, and a FastAPI web control panel for live config and monitoring.

[Unreleased]: https://github.com/sneezeparty/soupy/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/sneezeparty/soupy/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/sneezeparty/soupy/releases/tag/v1.0.0
