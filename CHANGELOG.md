# Changelog

All notable changes to Soupy will be recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project tries to use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) — though for a single-deployment Discord bot, that's a guideline, not a contract.

## [Unreleased]

### Added
- New self-knowledge **anchor** tier (`data/self_md/guild_<id>_anchor.md`). The reflection cycle now distills a small (~600 char) timeless identity statement from the core after generating it. The anchor is what gets injected into the system prompt for every reply; the larger core is no longer always-on. Topical detail (specific people, jokes, opinions on specific things) continues to be retrieved on demand via the existing self-knowledge RAG.
- Env vars: `SELF_MD_ANCHOR_MAX_CHARS` (default 600), `SELF_MD_ANCHOR_TEMPERATURE` (0.5), `SELF_MD_ANCHOR_MAX_TOKENS` (400), `SELF_MD_ANCHOR_FALLBACK_CHARS` (600). All editable from the web Environment Editor.
- Posted musings now feed into the self-reflection accumulator (the same one chat replies use). Each musing is logged as a `(self)` notable interaction tagged `[unprompted musing — mode=… — triggered by: …]`, so the next reflection cycle can grow opinions, relationships, and self-knowledge from what the bot mused about. Gated on `SELF_MD_ENABLED`; failures are swallowed so they cannot break musing posting.
- **Musings dedupe rework.** Each posted musing now persists a structured topic in two fields — `topic_subject` (what it's *about*: products, events, ideas, behaviors) and `topic_mentions` (people/handles named in it). Only `topic_subject` is used to filter candidate source messages, so people are no longer shadow-banned from the candidate pool just because they got name-checked in a recent musing.
- **Embedding-based similarity check.** When `RAG_EMBEDDING_MODEL` is set, candidate musings (archive seed, news headline, synthesis theme) are embedded and compared against the recent 15-musing window via cosine similarity; anything ≥ 0.88 is skipped before the LLM is invoked. Catches paraphrased near-duplicates that keyword filtering misses. Graceful fallback to keyword-only if the embeddings endpoint is unreachable.
- **Synthesis-mode time-bucket rotation.** Synthesis now picks its sampling window from three buckets (last 2 weeks 40%, 2-6 weeks 30%, 1.5-6 months 30%) and tells the prompt which window it drew from, so the framing matches the data it's pulling from.
- Cog-level `asyncio.Lock` serializes the entire post-and-persist pipeline so concurrent triggers (warmup, scheduled tick, `/soupymuse`) can't race on the JSONL archive.
- Background warmup task on cog load: backfills `topic_subject`/`topic_mentions` for legacy entries via one batched LLM call and pre-populates the embedding cache before the first scheduled tick fires.

### Changed
- `get_self_md_for_injection` now prefers the anchor file. Falls back to a paragraph-bounded truncation of the core (then full doc) when no anchor exists yet — so behavior is sensible from first run, and improves automatically once the next reflection cycle generates a real anchor. Net effect: system prompt drops from ~12k chars → ~6-7k chars per reply.
- The current trigger message is now wrapped in a `RESPOND TO THE MESSAGE BELOW` marker before the user/assistant merge step. Without it, the user/role merge that's needed for strict-alternation models like Gemma was concatenating the trigger onto any preceding URL content + RAG snippets + image descriptions, producing an 11k+ char user blob with the actual question buried at the end. The marker keeps the trigger findable.
- Musing prompts now require a concrete anchor — a name, a quoted phrase, a specific number, or the topic by name — so a reader can tell what Soupy is reacting to. The `news_react` mode previously forbade including the headline, which produced cryptic openers like "that ninety degrees thing keeps nagging at me" with no handle attached; that rule is now flipped to require the topic be worked in (without quoting the headline verbatim). Archive and random-thought prompts got the same anchoring instruction.
- Musing-mode weights rebalanced: `archive_reflect` 0.35 (was 0.30), `news_react` 0.20, `random_thought` 0.20 (was 0.15), `synthesis` 0.25 (was 0.35). Synthesis was over-firing on a thin archive; archive-reflect is the highest-signal mode now that subject/mentions split keeps its candidate pool diverse.
- Self-reflection feedback loop narrowed to **synthesis musings only**. The other three modes are reactions to specific external triggers and don't generalize well into self-knowledge; only the cross-message pattern-finding done by synthesis is worth memorizing. Tunable via the `_SELF_FEEDBACK_MODES` constant at the top of the cog.
- `MUSING_SYSTEM` prompt trimmed ~30%: removed duplicated anchor language (per-mode prompts say it too), all-lowercased the directives to match the persona, dropped redundant clauses.
- Topic extraction moved to *after* `channel.send()` so a slow LLM no longer delays the user-visible post; topics only need to exist by the next cycle.
- Archive writes use atomic `tmp + os.replace` instead of in-place rewrites, so a crash mid-write can't truncate the JSONL.

### Fixed
- Musings were repeating the same topic 3–4 times in a row. Root cause was the LLM "don't repeat these topics" hint being soft text the model ignored, combined with a wide-window random source pick that kept re-surfacing the same hot conversation. Hard-filter at the candidate-selection stage (keywords + embeddings) replaces the soft prompt hint.

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
