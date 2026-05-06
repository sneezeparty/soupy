#!/usr/bin/env python3
"""
Migrate per-guild SQLite databases (soupy_database/databases/guild_<id>.db) into a
single multi-tenant soupy.db.

USAGE
    python tools/migrate_to_single_db.py \
        --source soupy_database/databases/ \
        --target soupy.db \
        [--dry-run] [--verify]

The new schema lives in soupy_database/schema_v2.sql.  Every per-guild table gains
a guild_id column; every per-guild uniqueness constraint becomes a composite
(guild_id, ...) UNIQUE.  Re-running the migration is safe — INSERT OR IGNORE
against the composite UNIQUE makes it idempotent.

Stdlib only: sqlite3, argparse, pathlib, logging, re, sys.
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "soupy_database" / "schema_v2.sql"

GUILD_DB_RE = re.compile(r"^guild_(\d+)\.db$")

# Per-table copy plan.
#
# - source_columns:   columns to read from the per-guild DB (in order).
# - target_columns:   columns to write into the merged DB (in order).
#                     Always begins with "guild_id"; remaining columns mirror
#                     source_columns, except for AUTOINCREMENT id columns which
#                     are dropped so SQLite re-keys them on the target side.
# - source_check:     a query that confirms the source table exists with the
#                     expected columns.  If a guild DB pre-dates a feature
#                     (e.g. self_chunks) the table just won't be there and we
#                     skip it cleanly.
# - order_by:         deterministic ordering for streaming progress.

@dataclass(frozen=True)
class TablePlan:
    name: str
    source_columns: Sequence[str]
    target_columns: Sequence[str]
    order_by: str = "rowid"
    # If True, every row for this guild is deleted from the target before insert.
    # Used for "rebuild" tables that have no natural unique key (rolling logs,
    # embeddings that get fully recomputed on every reindex).  Without this, a
    # re-run would duplicate rows.  With it, a re-run is still idempotent: the
    # source is a snapshot, the target ends up with exactly that snapshot.
    replace_per_guild: bool = False


TABLE_PLANS: Tuple[TablePlan, ...] = (
    TablePlan(
        name="channels",
        source_columns=("channel_id", "channel_name", "last_updated"),
        target_columns=("guild_id", "channel_id", "channel_name", "last_updated"),
        order_by="channel_id",
    ),
    TablePlan(
        name="messages",
        source_columns=(
            "message_id", "date", "time", "username", "nickname", "user_id",
            "message_content", "channel_id", "channel_name",
            "image_description", "url_summary", "created_at",
        ),
        target_columns=(
            "guild_id", "message_id", "date", "time", "username", "nickname", "user_id",
            "message_content", "channel_id", "channel_name",
            "image_description", "url_summary", "created_at",
        ),
        order_by="message_id",
    ),
    TablePlan(
        name="scan_metadata",
        # AUTOINCREMENT id is dropped so each row picks up a fresh id in the merged DB.
        source_columns=("last_scan_time", "scan_type", "messages_scanned", "created_at"),
        target_columns=(
            "guild_id", "last_scan_time", "scan_type", "messages_scanned", "created_at",
        ),
        order_by="id",
    ),
    TablePlan(
        name="guild_metadata",
        source_columns=(
            "guild_id", "guild_name", "last_updated", "archive_scan_interval_minutes",
        ),
        # guild_metadata already has guild_id; the migration overrides it with the
        # filename-derived id to defend against malformed source rows.
        target_columns=(
            "guild_id", "guild_name", "last_updated", "archive_scan_interval_minutes",
        ),
        order_by="guild_id",
    ),
    TablePlan(
        name="rag_chunks",
        # AUTOINCREMENT id is dropped.
        source_columns=(
            "first_message_id", "last_message_id", "channel_id", "channel_name",
            "chunk_text", "embedding_dim", "embedding", "created_at",
        ),
        target_columns=(
            "guild_id", "first_message_id", "last_message_id", "channel_id",
            "channel_name", "chunk_text", "embedding_dim", "embedding", "created_at",
        ),
        order_by="id",
    ),
    TablePlan(
        name="user_profile_summaries",
        source_columns=(
            "user_id", "nickname_hint", "summary", "source_message_count",
            "source_max_message_id", "updated_at", "model_used", "structured_json",
        ),
        target_columns=(
            "guild_id", "user_id", "nickname_hint", "summary", "source_message_count",
            "source_max_message_id", "updated_at", "model_used", "structured_json",
        ),
        order_by="user_id",
    ),
    TablePlan(
        name="profile_batch_jobs",
        source_columns=(
            "guild_id", "status", "user_ids_json", "next_index", "total",
            "stats_json", "updated_at",
        ),
        target_columns=(
            "guild_id", "status", "user_ids_json", "next_index", "total",
            "stats_json", "updated_at",
        ),
        order_by="guild_id",
    ),
    TablePlan(
        name="profile_job_log_lines",
        # AUTOINCREMENT id is dropped.  guild_id already exists; we override it
        # from the filename anyway to keep things consistent with the dropped id.
        # Rolling log: no natural unique key, so use replace-per-guild to keep
        # re-runs idempotent.
        source_columns=("guild_id", "line"),
        target_columns=("guild_id", "line"),
        order_by="id",
        replace_per_guild=True,
    ),
    TablePlan(
        name="self_chunks",
        # AUTOINCREMENT id is dropped.  Embeddings are rebuilt wholesale on every
        # self-knowledge reflection cycle, so there is no natural unique key.
        # replace-per-guild keeps re-runs idempotent.
        source_columns=(
            "source", "section", "chunk_text", "embedding_dim", "embedding", "created_at",
        ),
        target_columns=(
            "guild_id", "source", "section", "chunk_text", "embedding_dim",
            "embedding", "created_at",
        ),
        order_by="id",
        replace_per_guild=True,
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger("migrate_to_single_db")


def _setup_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def discover_guild_dbs(source_dir: Path) -> List[Tuple[int, Path]]:
    """Return [(guild_id, path), ...] sorted by guild_id."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")
    out: List[Tuple[int, Path]] = []
    for entry in sorted(source_dir.iterdir()):
        if not entry.is_file():
            continue
        m = GUILD_DB_RE.match(entry.name)
        if not m:
            continue
        out.append((int(m.group(1)), entry))
    out.sort(key=lambda x: x[0])
    return out


def load_schema_sql() -> str:
    return SCHEMA_FILE.read_text(encoding="utf-8")


def apply_schema(target_conn: sqlite3.Connection) -> None:
    """Apply schema_v2.sql.  Idempotent — every CREATE is IF NOT EXISTS."""
    target_conn.executescript(load_schema_sql())
    target_conn.commit()


def list_source_tables(source_conn: sqlite3.Connection) -> Dict[str, set]:
    """Return {table_name: {column_name, ...}} for every user table in the source DB."""
    out: Dict[str, set] = {}
    cur = source_conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    for (tname,) in cur.fetchall():
        cur.execute(f"PRAGMA table_info({tname})")
        cols = {row[1] for row in cur.fetchall()}
        out[tname] = cols
    return out


def _quote_columns(cols: Sequence[str]) -> str:
    return ", ".join(f'"{c}"' for c in cols)


def count_source_rows(source_conn: sqlite3.Connection, table: str) -> int:
    cur = source_conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    return int(cur.fetchone()[0])


def count_target_rows(target_conn: sqlite3.Connection, table: str, guild_id: int) -> int:
    cur = target_conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE guild_id = ?', (guild_id,))
    return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Per-table migration
# ---------------------------------------------------------------------------

@dataclass
class TableResult:
    table: str
    source_rows: int
    inserted: int
    skipped: bool = False
    error: Optional[str] = None


def migrate_table(
    *,
    plan: TablePlan,
    guild_id: int,
    source_conn: sqlite3.Connection,
    target_conn: Optional[sqlite3.Connection],
    source_columns_present: Dict[str, set],
    table_index: int,
    table_total: int,
    dry_run: bool,
    progress_every: int = 1000,
) -> TableResult:
    """Migrate one table for one guild.  target_conn is None when --dry-run."""
    if plan.name not in source_columns_present:
        logger.info(
            "  [%d/%d] %-25s skipped (table not present in source)",
            table_index, table_total, plan.name,
        )
        return TableResult(table=plan.name, source_rows=0, inserted=0, skipped=True)

    have_cols = source_columns_present[plan.name]
    missing = [c for c in plan.source_columns if c not in have_cols]
    if missing:
        msg = f"source table {plan.name} missing columns: {missing}"
        logger.error("  [%d/%d] %-25s ERROR: %s", table_index, table_total, plan.name, msg)
        return TableResult(table=plan.name, source_rows=0, inserted=0, error=msg)

    total = count_source_rows(source_conn, plan.name)
    if total == 0:
        logger.info(
            "  [%d/%d] %-25s 0 rows (empty source table)",
            table_index, table_total, plan.name,
        )
        return TableResult(table=plan.name, source_rows=0, inserted=0)

    select_sql = (
        f'SELECT {_quote_columns(plan.source_columns)} '
        f'FROM "{plan.name}" ORDER BY {plan.order_by}'
    )
    cur_src = source_conn.cursor()
    cur_src.execute(select_sql)

    insert_sql = (
        f'INSERT OR IGNORE INTO "{plan.name}" ({_quote_columns(plan.target_columns)}) '
        f'VALUES ({", ".join("?" * len(plan.target_columns))})'
    )

    # Tables without a natural unique key get a delete-then-insert pass per guild
    # so the migration stays idempotent on re-run.
    if plan.replace_per_guild and target_conn is not None and not dry_run:
        target_conn.execute(
            f'DELETE FROM "{plan.name}" WHERE guild_id = ?', (guild_id,)
        )
        target_conn.commit()

    inserted = 0
    seen = 0
    BATCH = 1000
    batch: List[Sequence] = []

    def flush() -> None:
        nonlocal inserted
        if not batch:
            return
        if dry_run or target_conn is None:
            inserted += len(batch)
        else:
            cur_tgt = target_conn.cursor()
            before = cur_tgt.execute(
                f'SELECT COUNT(*) FROM "{plan.name}" WHERE guild_id = ?', (guild_id,)
            ).fetchone()[0]
            cur_tgt.executemany(insert_sql, batch)
            after = cur_tgt.execute(
                f'SELECT COUNT(*) FROM "{plan.name}" WHERE guild_id = ?', (guild_id,)
            ).fetchone()[0]
            inserted += int(after - before)
            target_conn.commit()
        batch.clear()

    for src_row in cur_src:
        # Build target row: guild_id first if the plan starts with it; otherwise
        # the plan starts with guild_id from the source (e.g. guild_metadata) and
        # we override that position.
        if plan.target_columns[0] == "guild_id" and plan.source_columns[0] != "guild_id":
            tgt_row = (guild_id, *src_row)
        elif plan.target_columns[0] == "guild_id" and plan.source_columns[0] == "guild_id":
            tgt_row = (guild_id, *src_row[1:])
        else:
            # Defensive — every plan in TABLE_PLANS starts with guild_id.
            raise RuntimeError(
                f"plan for table {plan.name} does not lead with guild_id in target_columns"
            )
        batch.append(tgt_row)
        seen += 1
        if len(batch) >= BATCH:
            flush()
        if seen % progress_every == 0 or seen == total:
            pct = (seen * 100) // total if total else 100
            logger.info(
                "    migrated %s/%s rows from %s for guild %s (%s%% — table %d/%d)",
                f"{seen:,}", f"{total:,}", plan.name, guild_id, pct,
                table_index, table_total,
            )

    flush()
    return TableResult(table=plan.name, source_rows=total, inserted=inserted)


def migrate_guild(
    *,
    guild_id: int,
    db_path: Path,
    target_conn: Optional[sqlite3.Connection],
    dry_run: bool,
) -> List[TableResult]:
    logger.info("processing guild %s (%s)", guild_id, db_path.name)
    source_conn = sqlite3.connect(str(db_path))
    source_conn.row_factory = sqlite3.Row
    try:
        source_tables = list_source_tables(source_conn)
        results: List[TableResult] = []
        for idx, plan in enumerate(TABLE_PLANS, 1):
            res = migrate_table(
                plan=plan,
                guild_id=guild_id,
                source_conn=source_conn,
                target_conn=target_conn,
                source_columns_present=source_tables,
                table_index=idx,
                table_total=len(TABLE_PLANS),
                dry_run=dry_run,
            )
            results.append(res)
        return results
    finally:
        source_conn.close()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_migration(
    *,
    source_dir: Path,
    target_path: Path,
) -> Tuple[int, List[str]]:
    """Recount per-(table, guild) rows; return (mismatches, messages)."""
    mismatches = 0
    messages: List[str] = []
    target_conn = sqlite3.connect(str(target_path))
    try:
        for guild_id, path in discover_guild_dbs(source_dir):
            src = sqlite3.connect(str(path))
            try:
                src_tables = list_source_tables(src)
                for plan in TABLE_PLANS:
                    if plan.name not in src_tables:
                        continue
                    src_n = count_source_rows(src, plan.name)
                    tgt_n = count_target_rows(target_conn, plan.name, guild_id)
                    if src_n == tgt_n:
                        messages.append(
                            f"OK   guild={guild_id} table={plan.name:<25} rows={src_n}"
                        )
                    else:
                        mismatches += 1
                        messages.append(
                            f"FAIL guild={guild_id} table={plan.name:<25} "
                            f"source={src_n} target={tgt_n}"
                        )
            finally:
                src.close()
    finally:
        target_conn.close()
    return mismatches, messages


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Migrate per-guild SQLite DBs into a single multi-tenant soupy.db.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Directory containing guild_<id>.db files (e.g. soupy_database/databases/).",
    )
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="Destination soupy.db path.  Created if it does not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not modify the target; only report what would be inserted.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After migration, recount rows per (table, guild_id) and fail on mismatch.",
    )
    args = parser.parse_args(argv)

    source_dir: Path = args.source
    target_path: Path = args.target

    try:
        guild_dbs = discover_guild_dbs(source_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not guild_dbs:
        logger.error("no guild_<id>.db files found in %s", source_dir)
        return 1

    logger.info(
        "migration plan: %d guild DB(s) -> %s%s",
        len(guild_dbs),
        target_path,
        " [DRY RUN]" if args.dry_run else "",
    )
    for gid, p in guild_dbs:
        logger.info("  guild %s: %s", gid, p)

    target_conn: Optional[sqlite3.Connection] = None
    failures = 0
    try:
        if not args.dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_conn = sqlite3.connect(str(target_path))
            apply_schema(target_conn)

        for guild_id, path in guild_dbs:
            try:
                results = migrate_guild(
                    guild_id=guild_id,
                    db_path=path,
                    target_conn=target_conn,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                failures += 1
                logger.error(
                    "guild %s: migration failed: %s", guild_id, exc, exc_info=True
                )
                continue

            errors = [r for r in results if r.error]
            for r in errors:
                failures += 1
                logger.error("guild %s: %s — %s", guild_id, r.table, r.error)

            total_src = sum(r.source_rows for r in results)
            total_inserted = sum(r.inserted for r in results)
            logger.info(
                "guild %s: %s source row(s) -> %s inserted%s",
                guild_id,
                f"{total_src:,}",
                f"{total_inserted:,}",
                " (dry-run, target untouched)" if args.dry_run else "",
            )
    finally:
        if target_conn is not None:
            target_conn.close()

    if args.verify:
        if args.dry_run:
            logger.warning("--verify ignored in --dry-run mode (target was not written)")
        else:
            mismatches, messages = verify_migration(
                source_dir=source_dir, target_path=target_path
            )
            for line in messages:
                logger.info("verify: %s", line)
            if mismatches:
                logger.error("verify: %d mismatch(es) detected", mismatches)
                return 1
            logger.info("verify: all (table, guild_id) row counts match")

    if failures:
        logger.error("migration completed with %d failure(s)", failures)
        return 1
    logger.info("migration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
