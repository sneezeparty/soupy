# Soupy Database Module

Handles message storage, RAG retrieval, user profiles, and self-knowledge for the Soupy Discord bot. Each Discord server gets its own SQLite database.

## Files

| File | Purpose |
|------|---------|
| `database.py` | Core schema, message storage, scan operations |
| `rag.py` | RAG embeddings, cosine similarity search, chunk indexing |
| `user_profiles.py` | Structured user profile generation via LLM |
| `self_profile.py` | Soupy's memory of itself, built from its own messages like a member profile |
| `self_context.py` | SELF.MD files and the identity anchor injected into replies |
| `runtime_flags.py` | Shared runtime state (command toggles, flags) |
| `profile_batch.py` | Batch profile generation worker |
| `helpers.py` | Image description and URL summarization helpers |
| `check_database.py` | Database statistics utility |
| `progress_report.py` | Scan progress reporting |
| `view_conversation.py` | Conversation viewing utility |
| `view_detailed.py` | Message viewing utility |

## Database Schema

Each guild gets a database at `databases/guild_{guild_id}.db` (or `SOUPY_DB_DIR`).

### Core Tables

**messages** — All archived messages
- `message_id`, `date`, `time`, `username`, `nickname`, `user_id`
- `message_content`, `channel_id`, `channel_name`
- `image_description`, `url_summary`, `created_at`

**channels** — Channel metadata
- `channel_id`, `channel_name`, `last_updated`

**scan_metadata** — Scan history
- `last_scan_time`, `scan_type`, `messages_scanned`

### RAG Tables

**rag_chunks** — Embedded message chunks for retrieval
- `chunk_id`, `message_id_lo`, `message_id_hi`, `channel_name`
- `chunk_text`, `embedding` (binary vector)

### User Profile Tables

**user_profiles** — Structured JSON profiles per user
- `user_id`, `guild_id`, `nickname_hint`, `structured` (JSON)
- `summary_text`, `messages_processed`, `updated_at`

## Key Functions

### RAG (`rag.py`)
- `embed_texts_lm_studio()` — Generate embeddings via LM Studio
- `search_rag_chunks()` — Cosine similarity search against embedded chunks
- `index_new_messages_to_rag()` — Index new messages into RAG
- `ensure_rag_schema()` — Create RAG tables if missing

### User Profiles (`user_profiles.py`)
- `_load_structured_profiles()` — Bulk-load profiles for a list of user IDs
- `refresh_user_profile()` — Fold a member's unread messages into their profile, one saved pass at a time (`run_edit_passes()` is the loop, shared with Soupy's memory)
- `ensure_user_profile_schema()` — Create profile tables if missing

### Soupy's Memory (`self_profile.py`, `self_context.py`)
- Built from Soupy's archived messages (with what members said just before) in passes, at the end of each profile job
- Dated, sectioned items: personality, lately, people, opinions, likes & dislikes, running jokes, moments, about me
- Rendered into `data/self_md/guild_<id>.md` / `_core.md` / `_anchor.md`; items embedded into `self_chunks` for chat

## Usage

Scanning is triggered via `/soupyscan` (owner-only). After scan, RAG reindexing runs automatically. Profiles are generated in batch via the web panel or during scan.

## Database Location

Default: `soupy_database/databases/`
Override: set `SOUPY_DB_DIR` in `.env-stable`
