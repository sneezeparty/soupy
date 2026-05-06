-- Soupy multi-tenant SQLite schema (v2).
--
-- Replaces the per-guild file layout (soupy_database/databases/guild_<id>.db) with a
-- single soupy.db.  Every table that used to live inside a per-guild file gains a
-- guild_id INTEGER NOT NULL column, and every per-guild uniqueness constraint is
-- carried across as a composite UNIQUE that includes guild_id.
--
-- Performance: rag_chunks is the hot table (full-table scan per RAG query under the
-- old layout).  The new layout MUST keep WHERE guild_id = ? cheap, so guild_id is
-- the leading column on every multi-column index that touches it.  See the explicit
-- pytest assertion in tools/migrate_to_single_db_test.py.
--
-- This file is read by tools/migrate_to_single_db.py and is also safe to apply by
-- hand (sqlite3 soupy.db < schema_v2.sql).  All statements are CREATE ... IF NOT
-- EXISTS so it can be re-run.

PRAGMA foreign_keys = OFF;

-- -----------------------------------------------------------------------------
-- channels
-- -----------------------------------------------------------------------------
-- Old PK: channel_id.  New uniqueness: (guild_id, channel_id) — the same channel
-- id can appear in many guilds in theory; even if it can't on Discord, the
-- composite key is the safe multi-tenant shape.
CREATE TABLE IF NOT EXISTS channels (
    guild_id      INTEGER NOT NULL,
    channel_id    INTEGER NOT NULL,
    channel_name  TEXT NOT NULL,
    last_updated  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, channel_id)
);
CREATE INDEX IF NOT EXISTS idx_channels_guild ON channels(guild_id);

-- -----------------------------------------------------------------------------
-- messages
-- -----------------------------------------------------------------------------
-- Old PK: message_id.  Discord snowflakes are globally unique, but we still scope
-- on (guild_id, message_id) so a stray cross-guild collision (or test fixture)
-- never silently overwrites the wrong row.
CREATE TABLE IF NOT EXISTS messages (
    guild_id           INTEGER NOT NULL,
    message_id         INTEGER NOT NULL,
    date               TEXT NOT NULL,
    time               TEXT NOT NULL,
    username           TEXT NOT NULL,
    nickname           TEXT,
    user_id            INTEGER NOT NULL,
    message_content    TEXT,
    channel_id         INTEGER NOT NULL,
    channel_name       TEXT NOT NULL,
    image_description  TEXT,
    url_summary        TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (guild_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_guild              ON messages(guild_id);
CREATE INDEX IF NOT EXISTS idx_messages_guild_channel      ON messages(guild_id, channel_id);
CREATE INDEX IF NOT EXISTS idx_messages_guild_user         ON messages(guild_id, user_id);
CREATE INDEX IF NOT EXISTS idx_messages_guild_created_at   ON messages(guild_id, created_at);

-- -----------------------------------------------------------------------------
-- scan_metadata
-- -----------------------------------------------------------------------------
-- AUTOINCREMENT row id stays local to each guild — we re-key on insert.  The
-- composite UNIQUE on (guild_id, last_scan_time, scan_type) lets the migration
-- run idempotently (re-importing the same row is a no-op).
CREATE TABLE IF NOT EXISTS scan_metadata (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id           INTEGER NOT NULL,
    last_scan_time     TIMESTAMP NOT NULL,
    scan_type          TEXT NOT NULL,
    messages_scanned   INTEGER DEFAULT 0,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (guild_id, last_scan_time, scan_type)
);
CREATE INDEX IF NOT EXISTS idx_scan_metadata_guild ON scan_metadata(guild_id);

-- -----------------------------------------------------------------------------
-- guild_metadata
-- -----------------------------------------------------------------------------
-- Already keyed on guild_id in the old schema, just carries over.
CREATE TABLE IF NOT EXISTS guild_metadata (
    guild_id                       INTEGER PRIMARY KEY,
    guild_name                     TEXT,
    last_updated                   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archive_scan_interval_minutes  INTEGER NOT NULL DEFAULT 0
);

-- -----------------------------------------------------------------------------
-- rag_chunks  (HOT TABLE)
-- -----------------------------------------------------------------------------
-- Old layout: full-table scan per query, fine because each guild was its own file.
-- New layout: every read goes through WHERE guild_id = ?.  The leading-guild_id
-- index below is what keeps the planner from full-scanning the merged table.
-- The (guild_id, channel_id) index preserves the per-channel filter pattern used
-- in index_new_messages().
CREATE TABLE IF NOT EXISTS rag_chunks (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id           INTEGER NOT NULL,
    first_message_id   INTEGER NOT NULL,
    last_message_id    INTEGER NOT NULL,
    channel_id         INTEGER NOT NULL,
    channel_name       TEXT NOT NULL,
    chunk_text         TEXT NOT NULL,
    embedding_dim      INTEGER NOT NULL,
    embedding          BLOB NOT NULL,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (guild_id, first_message_id, last_message_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_guild         ON rag_chunks(guild_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_guild_channel ON rag_chunks(guild_id, channel_id);

-- -----------------------------------------------------------------------------
-- user_profile_summaries
-- -----------------------------------------------------------------------------
-- Old PK: user_id.  Same Discord user can be a member of multiple guilds and
-- have a different profile per guild, so the new key is (guild_id, user_id).
CREATE TABLE IF NOT EXISTS user_profile_summaries (
    guild_id              INTEGER NOT NULL,
    user_id               INTEGER NOT NULL,
    nickname_hint         TEXT,
    summary               TEXT NOT NULL,
    source_message_count  INTEGER NOT NULL DEFAULT 0,
    source_max_message_id INTEGER,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used            TEXT,
    structured_json       TEXT,
    PRIMARY KEY (guild_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_user_profile_summaries_guild ON user_profile_summaries(guild_id);

-- -----------------------------------------------------------------------------
-- profile_batch_jobs
-- -----------------------------------------------------------------------------
-- Already keyed on guild_id in the old schema; carries over verbatim.
CREATE TABLE IF NOT EXISTS profile_batch_jobs (
    guild_id        INTEGER PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'idle',
    user_ids_json   TEXT NOT NULL DEFAULT '[]',
    next_index      INTEGER NOT NULL DEFAULT 0,
    total           INTEGER NOT NULL DEFAULT 0,
    stats_json      TEXT,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- profile_job_log_lines
-- -----------------------------------------------------------------------------
-- Old schema already had guild_id on each row + a (guild_id, id) index.  Carries
-- over with the same shape; AUTOINCREMENT id is re-keyed at migration time.
CREATE TABLE IF NOT EXISTS profile_job_log_lines (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id  INTEGER NOT NULL,
    line      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_job_log_guild_id ON profile_job_log_lines(guild_id, id);

-- -----------------------------------------------------------------------------
-- self_chunks
-- -----------------------------------------------------------------------------
-- Self-knowledge embeddings.  Old PK is AUTOINCREMENT id.  We add guild_id so the
-- merged table can serve every guild's self-document independently, and key the
-- index on (guild_id, source) since RAG retrieval filters by guild and by source.
CREATE TABLE IF NOT EXISTS self_chunks (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id       INTEGER NOT NULL,
    source         TEXT NOT NULL,
    section        TEXT NOT NULL DEFAULT '',
    chunk_text     TEXT NOT NULL,
    embedding_dim  INTEGER NOT NULL,
    embedding      BLOB NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_self_chunks_guild        ON self_chunks(guild_id);
CREATE INDEX IF NOT EXISTS idx_self_chunks_guild_source ON self_chunks(guild_id, source);
