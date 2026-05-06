# tools/

Standalone CLIs you run by hand. None of these are imported by the bot
or the web panel — they're maintenance scripts. Move imports here, not
the other way around.

| Script | Purpose |
|---|---|
| `view_conversation.py` | Group recent messages into conversation chunks (~15 min windows) and print them. |
| `view_detailed.py` | Dump every column of every row for a given user, plus a sample row. |
| `progress_report.py` | Print a per-guild summary of message counts, scan progress, and RAG coverage. |
| `check_database.py` | One-liner sanity check: row counts, top users, last scan timestamp. |
| `check_database.sh` | Shell version of the above using `sqlite3` directly — no Python required. |

Run from the repo root, e.g.:

```bash
python tools/view_conversation.py <guild_id> [user]
python tools/progress_report.py
./tools/check_database.sh <guild_id>
```
