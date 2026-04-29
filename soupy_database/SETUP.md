# Quick Setup Guide for /soupyscan

## Prerequisites

1. **Owner Access**: Your Discord user ID must be in `OWNER_IDS` in `.env-stable`
2. **LLM Backend**: LM Studio (or compatible) must be running for image descriptions and URL summaries
3. **Vision Model** (optional): If you want image descriptions, set `ENABLE_VISION=true` and configure `VISION_MODEL` in `.env-stable`

## First Time Setup

1. **Verify OWNER_IDS** in `.env-stable`:
   ```
   OWNER_IDS=000000000000000000
   ```
   (Add your Discord user ID here — enable Developer Mode in Discord and right-click your profile to copy your ID. Multiple owners can be comma-separated.)

2. **Optional: Set Custom Database Location** (for M1 Mac Mini):
   ```
   SOUPY_DB_DIR=/path/to/your/database/location
   ```
   If not set, databases will be stored in `soupy_database/databases/`

3. **Start the Bot**: Make sure the bot is running

## Using /soupyscan

1. **In Discord**, type `/soupyscan` in any server where the bot is present
2. **Wait for completion**: The scan will take a while, especially for the first scan (12 months of history)
3. **Check results**: You'll get an embed with statistics when the scan completes

## What Gets Scanned

- All text channels the bot can access
- Messages from the last 12 months (first scan) or since last scan (subsequent scans)
- Image attachments (described using vision model)
- URLs in messages (summarized using LLM)

## Database Location

Databases are stored as:
```
soupy_database/databases/guild_{server_id}.db
```

Each server gets its own database file.

## Troubleshooting

### "You don't have permission"
- Check that your user ID is in `OWNER_IDS` in `.env-stable`
- Restart the bot after changing `.env-stable`

### Scan is very slow
- This is normal! The scan is intentionally slow to avoid Discord rate limits
- First scan can take hours for large servers
- Subsequent scans are much faster (only new messages)

### Images/URLs not being processed
- Check that `ENABLE_VISION=true` for images
- Check that LM Studio is running and accessible
- Check `OPENAI_BASE_URL` in `.env-stable` points to your LM Studio instance

### Database errors
- Ensure the database directory is writable
- Check disk space
- Verify file permissions

