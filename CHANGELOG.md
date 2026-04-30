# Changelog

All notable changes to Soupy will be recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project tries to use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though for a single-deployment Discord bot, that's a guideline, not a contract.

## [Unreleased]

### Added
- New self-knowledge **anchor** tier (`data/self_md/guild_<id>_anchor.md`). The reflection cycle now distills a small (~600 char) timeless identity statement from the core after generating it. The anchor is what gets injected into the system prompt for every reply; the larger core is no longer always-on. Topical detail (specific people, jokes, opinions on specific things) continues to be retrieved on demand via the existing self-knowledge RAG.
- Env vars: `SELF_MD_ANCHOR_MAX_CHARS` (default 600), `SELF_MD_ANCHOR_TEMPERATURE` (0.5), `SELF_MD_ANCHOR_MAX_TOKENS` (400), `SELF_MD_ANCHOR_FALLBACK_CHARS` (600). All editable from the web Environment Editor.

### Changed
- `get_self_md_for_injection` now prefers the anchor file. Falls back to a paragraph-bounded truncation of the core (then full doc) when no anchor exists yet — so behavior is sensible from first run, and improves automatically once the next reflection cycle generates a real anchor. Net effect: system prompt drops from ~12k chars → ~6-7k chars per reply.
- The current trigger message is now wrapped in a `RESPOND TO THE MESSAGE BELOW` marker before the user/assistant merge step. Without it, the user/role merge that's needed for strict-alternation models like Gemma was concatenating the trigger onto any preceding URL content + RAG snippets + image descriptions, producing an 11k+ char user blob with the actual question buried at the end. The marker keeps the trigger findable.

## [1.1.1] - 2026-04-30

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

[Unreleased]: https://github.com/sneezeparty/soupy/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/sneezeparty/soupy/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/sneezeparty/soupy/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/sneezeparty/soupy/releases/tag/v1.0.0
