# Viewing the Database

## Quick Commands

The database files are located in: `soupy_database/databases/guild_*.db`

### Open a database in SQLite CLI:
```bash
cd /path/to/cursor-project
sqlite3 soupy_database/databases/guild_<server_id>.db
```

### Useful SQLite Commands:

#### View all tables:
```sql
.tables
```

#### View table structure:
```sql
.schema messages
.schema channels
.schema scan_metadata
```

#### Count total messages:
```sql
SELECT COUNT(*) FROM messages;
```

#### View recent messages (last 10):
```sql
SELECT date, time, username, channel_name, 
       substr(message_content, 1, 50) as content_preview
FROM messages 
ORDER BY created_at DESC 
LIMIT 10;
```

#### View messages with image descriptions:
```sql
SELECT date, time, username, channel_name,
       substr(image_description, 1, 80) as image_desc
FROM messages 
WHERE image_description IS NOT NULL
ORDER BY created_at DESC 
LIMIT 10;
```

#### View messages with URL summaries:
```sql
SELECT date, time, username, channel_name,
       substr(url_summary, 1, 80) as url_summary
FROM messages 
WHERE url_summary IS NOT NULL
ORDER BY created_at DESC 
LIMIT 10;
```

#### View scan history:
```sql
SELECT scan_type, messages_scanned, last_scan_time, created_at
FROM scan_metadata
ORDER BY created_at DESC;
```

#### Count messages per channel:
```sql
SELECT channel_name, COUNT(*) as message_count
FROM messages
GROUP BY channel_name
ORDER BY message_count DESC;
```

#### Count messages per user:
```sql
SELECT username, COUNT(*) as message_count
FROM messages
GROUP BY username
ORDER BY message_count DESC
LIMIT 20;
```

#### View database file size:
```sql
PRAGMA page_count;
PRAGMA page_size;
```

#### Exit SQLite:
```sql
.quit
```

### One-liner Commands (from terminal, not SQLite):

#### Quick message count:
```bash
sqlite3 soupy_database/databases/guild_<server_id>.db "SELECT COUNT(*) FROM messages;"
```

#### View last 5 messages:
```bash
sqlite3 soupy_database/databases/guild_<server_id>.db "SELECT date, time, username, substr(message_content, 1, 60) FROM messages ORDER BY created_at DESC LIMIT 5;"
```

#### Check database file size:
```bash
ls -lh soupy_database/databases/guild_*.db
```

#### Watch message count in real-time (update every 2 seconds):
```bash
watch -n 2 'sqlite3 soupy_database/databases/guild_<server_id>.db "SELECT COUNT(*) FROM messages;"'
```

### Using a GUI Tool (Optional):

If you prefer a GUI, you can use:
- **DB Browser for SQLite** (free): https://sqlitebrowser.org/
- **TablePlus** (paid, free trial): https://tableplus.com/
- **DBeaver** (free): https://dbeaver.io/

Just open the `.db` file directly with any of these tools.

