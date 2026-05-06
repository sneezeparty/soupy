# Migrating to the single multi-tenant `soupy.db`

This is the operator runbook for consolidating Soupy's per-guild SQLite files
(`soupy_database/databases/guild_<id>.db`) into a single multi-tenant `soupy.db`.

The migration tool itself lives at
[tools/migrate_to_single_db.py](../tools/migrate_to_single_db.py).
The new schema lives at
[soupy_database/schema_v2.sql](schema_v2.sql).

> **Status:** the migration tool is ready.  The bot still reads the per-guild
> file layout — the cutover (point the bot at `soupy.db`) lands in a follow-up
> commit.  **TBD: bot-side cutover lands in a follow-up commit.**

---

## What changes

Every per-guild table gains a `guild_id INTEGER NOT NULL` column.  Every
per-guild uniqueness constraint becomes a composite `(guild_id, ...)` UNIQUE.

| Old table              | New primary / unique key                |
|------------------------|------------------------------------------|
| `channels`             | `(guild_id, channel_id)`                 |
| `messages`             | `(guild_id, message_id)`                 |
| `scan_metadata`        | `id` PK, `(guild_id, last_scan_time, scan_type)` UNIQUE |
| `guild_metadata`       | `guild_id` (unchanged)                   |
| `rag_chunks`           | `id` PK, `(guild_id, first_message_id, last_message_id)` UNIQUE |
| `user_profile_summaries` | `(guild_id, user_id)`                  |
| `profile_batch_jobs`   | `guild_id` (unchanged)                   |
| `profile_job_log_lines` | `id` PK (already had `guild_id`)        |
| `self_chunks`          | `id` PK, indexed on `(guild_id, source)` |

`rag_chunks` is the hot table.  Two indexes keep it fast under the new layout:
`(guild_id)` so `WHERE guild_id = ?` does an index search instead of a full
scan, and `(guild_id, channel_id)` so the per-channel filter inside
`index_new_messages()` continues to be index-driven.  There is a regression test
for both queries in
[tools/migrate_to_single_db_test.py](../tools/migrate_to_single_db_test.py).

---

## Runbook

### 1. Stop the bot

Use the web panel's stop button, or `POST /api/bot/stop`.  Do **not** run the
migration while the bot is writing to the per-guild files.

### 2. Back up your databases

```bash
cp -a soupy_database/databases/ soupy_database/databases.bak.$(date +%Y%m%d)
```

Keep this backup for at least 30 days after the cutover.  The migration is
non-destructive (it never touches the source files), but a snapshot is cheap
insurance.

### 3. Dry-run the migration

```bash
source .venv/bin/activate
python tools/migrate_to_single_db.py \
    --source soupy_database/databases/ \
    --target soupy.db \
    --dry-run
```

This walks every `guild_<id>.db`, reports row counts per (guild, table), and
exits without writing anything to the target file.  Inspect the output for
unexpected zero counts or missing tables.

### 4. Real run

```bash
python tools/migrate_to_single_db.py \
    --source soupy_database/databases/ \
    --target soupy.db
```

The tool is idempotent: `INSERT OR IGNORE` against the new composite UNIQUE
keys means re-running it never duplicates rows.  If the run is interrupted,
just re-run it.

Expect roughly seconds per 100k messages on a local SSD.  RAG chunks are
larger (embeddings are blobs); allow several minutes for guilds with a deep
reindex history.

### 5. Verify

```bash
python tools/migrate_to_single_db.py \
    --source soupy_database/databases/ \
    --target soupy.db \
    --verify
```

`--verify` re-counts every (table, guild) pair against the source and exits 1
on mismatch.  A clean run prints `verify: all (table, guild_id) row counts
match`.

### 6. Inspect

A quick smoke test against the merged file:

```bash
sqlite3 soupy.db "SELECT guild_id, COUNT(*) FROM messages GROUP BY guild_id;"
sqlite3 soupy.db "SELECT guild_id, COUNT(*) FROM rag_chunks GROUP BY guild_id;"
```

Numbers should match what the per-guild stats endpoint reported pre-migration.

### 7. Cutover

**TBD: bot-side cutover lands in a follow-up commit.**  Until that lands, the
bot continues to read the per-guild files; `soupy.db` sits alongside them as a
known-good replica that the new code path can switch over to.

When the cutover ships:

1. Start the bot with the new code path enabled.
2. If anything goes wrong, set the rollback env var (defined in the cutover
   commit) — the bot will go back to reading the per-guild files.  The cutover
   PR will document the exact name; until it's merged, it's not yet a real
   knob.

---

## Rollback

For at least the first 30 days after cutover:

* Keep the per-guild files (`soupy_database/databases/guild_*.db`) on disk.
* Keep the `databases.bak.YYYYMMDD` snapshot you made in step 2.
* The cutover commit will ship an env var (something like
  `SOUPY_DB_LAYOUT=per_guild|single`) so you can flip back without rolling
  code.  Until the cutover commit lands, the bot still uses per-guild files
  unconditionally.

After 30 days of healthy operation on the merged DB you can delete the
backup and the original `databases/` directory.

---

## Troubleshooting

**`source directory not found`**
Pass an absolute path, or run from the repo root.  The default in this repo is
`soupy_database/databases/`.

**`no guild_<id>.db files found`**
The tool only matches files named exactly `guild_<digits>.db`.  Anything else
(including `guild_<id>.db.bak.*`) is ignored on purpose.

**A specific guild fails to migrate, but others succeed**
The tool logs the failure and keeps going.  Re-run the migration after
fixing the source file (e.g. `sqlite3 guild_<id>.db "PRAGMA integrity_check;"`)
— `INSERT OR IGNORE` makes the partial state safe to top up.

**`--verify` reports a mismatch**
Don't cut over.  Re-run the migration (it's idempotent), then re-run
`--verify`.  If the mismatch persists, open the source file directly and
inspect the offending table — the source DB may be corrupt.

**`rag_chunks` queries feel slow after cutover**
Confirm the index is present:

```bash
sqlite3 soupy.db ".indexes rag_chunks"
```

You should see `idx_rag_chunks_guild` and `idx_rag_chunks_guild_channel`.  If
either is missing, re-apply the schema:

```bash
sqlite3 soupy.db < soupy_database/schema_v2.sql
```

Every statement is `CREATE ... IF NOT EXISTS`, so it's safe to apply against a
populated DB.
