"""
Database module for Soupy message scanning.
Handles per-server SQLite databases for message storage.
"""

import sqlite3
import logging
import os
import asyncio
import time
import random
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict
import discord
from discord import app_commands

from .helpers import describe_image, extract_urls, extract_url_content
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Track active scans to prevent concurrent scans on the same server
# Key: guild_id, Value: asyncio.Task
active_scans: Dict[int, asyncio.Task] = {}

# Database directory - can be configured via environment variable
DB_DIR = os.getenv("SOUPY_DB_DIR", os.path.join(os.path.dirname(__file__), "databases"))
Path(DB_DIR).mkdir(parents=True, exist_ok=True)

# Scan trigger directory - for file-based triggers from web API
SCAN_TRIGGER_DIR = os.path.join(DB_DIR, "scan_triggers")
Path(SCAN_TRIGGER_DIR).mkdir(parents=True, exist_ok=True)


def get_db_path(guild_id: int) -> str:
    """Get the database file path for a specific guild."""
    return os.path.join(DB_DIR, f"guild_{guild_id}.db")


def ensure_rag_schema(conn: sqlite3.Connection) -> None:
    """RAG vector chunks over scanned messages (safe to call on any guild connection)."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_message_id INTEGER NOT NULL,
            last_message_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL,
            channel_name TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(first_message_id, last_message_id)
        )
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_channel
        ON rag_chunks(channel_id)
        """
    )
    conn.commit()


def init_database(guild_id: int) -> sqlite3.Connection:
    """
    Initialize or connect to the database for a specific guild.
    Creates tables if they don't exist.
    """
    db_path = get_db_path(guild_id)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    
    # Create channels table to track channel names and IDs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id INTEGER PRIMARY KEY,
            channel_name TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            username TEXT NOT NULL,
            nickname TEXT,
            user_id INTEGER NOT NULL,
            message_content TEXT,
            channel_id INTEGER NOT NULL,
            channel_name TEXT NOT NULL,
            image_description TEXT,
            url_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
        )
    """)
    
    # Create scan_metadata table to track last scan time
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_scan_time TIMESTAMP NOT NULL,
            scan_type TEXT NOT NULL,
            messages_scanned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create guild_metadata table to store guild information
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS guild_metadata (
            guild_id INTEGER PRIMARY KEY,
            guild_name TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            archive_scan_interval_minutes INTEGER NOT NULL DEFAULT 0
        )
    """)
    
    # Create index on channel_id for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_channel_id ON messages(channel_id)
    """)
    
    # Create index on user_id for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)
    """)
    
    # Create index on created_at for faster time-based queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)
    """)

    ensure_rag_schema(conn)
    ensure_guild_metadata_schedule_columns(conn)
    from .user_profiles import ensure_user_profile_schema

    ensure_user_profile_schema(conn)

    conn.commit()
    logger.debug(f"Database initialized for guild {guild_id} at {db_path}")
    return conn


def ensure_guild_metadata_schedule_columns(conn: sqlite3.Connection) -> None:
    """Add archive_scan_interval_minutes to existing DBs (SQLite migration)."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(guild_metadata)")
    cols = {row[1] for row in cursor.fetchall()}
    if "archive_scan_interval_minutes" not in cols:
        cursor.execute(
            """
            ALTER TABLE guild_metadata
            ADD COLUMN archive_scan_interval_minutes INTEGER NOT NULL DEFAULT 0
            """
        )
        conn.commit()


def get_archive_scan_interval_minutes(guild_id: int) -> int:
    """Minutes between automatic archive (message DB) scans; 0 = disabled."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return 0
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        ensure_guild_metadata_schedule_columns(conn)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT archive_scan_interval_minutes FROM guild_metadata WHERE guild_id = ?",
            (guild_id,),
        )
        row = cursor.fetchone()
        if not row:
            return 0
        return int(row["archive_scan_interval_minutes"] or 0)
    finally:
        conn.close()


def set_archive_scan_interval_minutes(guild_id: int, minutes: int) -> Dict[str, Any]:
    """
    Persist auto-archive interval for a guild (creates DB if missing).
    minutes 0 disables automatic scans; max 10080 (7 days).
    """
    if minutes < 0 or minutes > 10080:
        raise ValueError("archive_scan_interval_minutes must be between 0 and 10080")
    conn = init_database(guild_id)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT guild_id FROM guild_metadata WHERE guild_id = ?", (guild_id,))
        if cursor.fetchone():
            cursor.execute(
                """
                UPDATE guild_metadata
                SET archive_scan_interval_minutes = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE guild_id = ?
                """,
                (minutes, guild_id),
            )
        else:
            cursor.execute(
                """
                INSERT INTO guild_metadata (guild_id, guild_name, last_updated, archive_scan_interval_minutes)
                VALUES (?, '', CURRENT_TIMESTAMP, ?)
                """,
                (guild_id, minutes),
            )
        conn.commit()
        return {
            "guild_id": guild_id,
            "archive_scan_interval_minutes": minutes,
        }
    finally:
        conn.close()


def get_last_scan_time(guild_id: int) -> Optional[datetime]:
    """Get the timestamp of the last scan for a guild. Returns timezone-aware UTC datetime."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT last_scan_time FROM scan_metadata 
        ORDER BY last_scan_time DESC 
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        timestamp_str = row["last_scan_time"]
        # Parse the timestamp
        dt = datetime.fromisoformat(timestamp_str)
        # If it's naive, assume it's UTC and make it timezone-aware
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def update_channel_name(conn: sqlite3.Connection, channel_id: int, channel_name: str):
    """Update or insert channel name for a channel ID."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO channels (channel_id, channel_name, last_updated)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """, (channel_id, channel_name))
    conn.commit()


def insert_message(
    conn: sqlite3.Connection,
    message_id: int,
    date: str,
    time: str,
    username: str,
    nickname: Optional[str],
    user_id: int,
    message_content: Optional[str],
    channel_id: int,
    channel_name: str,
    image_description: Optional[str] = None,
    url_summary: Optional[str] = None
):
    """Insert a message into the database."""
    cursor = conn.cursor()
    
    # Update channel name mapping
    update_channel_name(conn, channel_id, channel_name)
    
    # Insert message (replace if exists to handle updates)
    cursor.execute("""
        INSERT OR REPLACE INTO messages (
            message_id, date, time, username, nickname, user_id,
            message_content, channel_id, channel_name,
            image_description, url_summary, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        message_id, date, time, username, nickname, user_id,
        message_content, channel_id, channel_name,
        image_description, url_summary
    ))
    
    conn.commit()


def message_exists(conn: sqlite3.Connection, message_id: int) -> bool:
    """Check if a message already exists in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM messages WHERE message_id = ?", (message_id,))
    return cursor.fetchone() is not None


def record_scan(conn: sqlite3.Connection, scan_type: str, messages_scanned: int):
    """Record a scan operation in the metadata table."""
    cursor = conn.cursor()
    # Store UTC timestamp as ISO format string
    utc_timestamp = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO scan_metadata (last_scan_time, scan_type, messages_scanned)
        VALUES (?, ?, ?)
    """, (utc_timestamp, scan_type, messages_scanned))
    conn.commit()


def get_stats(guild_id: int) -> Dict:
    """Get statistics about the database for a guild."""
    db_path = get_db_path(guild_id)
    if not os.path.exists(db_path):
        return {
            "exists": False,
            "total_messages": 0,
            "total_channels": 0,
            "last_scan": None,
            "guild_name": None,
            "archive_scan_interval_minutes": 0,
        }
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_guild_metadata_schedule_columns(conn)
    cursor = conn.cursor()

    # Count messages
    cursor.execute("SELECT COUNT(*) as count FROM messages")
    total_messages = cursor.fetchone()["count"]
    
    # Count channels
    cursor.execute("SELECT COUNT(*) as count FROM channels")
    total_channels = cursor.fetchone()["count"]
    
    # Get last scan time
    last_scan = get_last_scan_time(guild_id)
    
    # Get guild name from metadata
    cursor.execute(
        """
        SELECT guild_name, archive_scan_interval_minutes
        FROM guild_metadata WHERE guild_id = ?
        """,
        (guild_id,),
    )
    guild_name_row = cursor.fetchone()
    guild_name = guild_name_row["guild_name"] if guild_name_row else None
    archive_scan_interval_minutes = (
        int(guild_name_row["archive_scan_interval_minutes"] or 0)
        if guild_name_row
        else 0
    )

    conn.close()

    return {
        "exists": True,
        "total_messages": total_messages,
        "total_channels": total_channels,
        "last_scan": last_scan.isoformat() if last_scan else None,
        "guild_name": guild_name,
        "archive_scan_interval_minutes": archive_scan_interval_minutes,
    }


def setup_scan_command(bot, owner_ids: list):
    """
    Sets up the /soupyscan command on the bot.
    This should be called from the main bot file after the bot is created.
    """
    @bot.tree.command(name="soupyscan", description="Scan all channels and store messages in database (Owner only)")
    async def soupyscan_command(interaction: discord.Interaction):
        """
        Scans all channels the bot has access to and stores messages in a database.
        Only accessible by users in OWNER_IDS.
        Runs in background to allow other bot functions to continue working.
        """
        # Check if user is in OWNER_IDS
        if interaction.user.id not in owner_ids:
            await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
            logger.warning(f"Unauthorized attempt to use /soupyscan by {interaction.user}")
            return
        
        if not interaction.guild:
            await interaction.response.send_message("❌ This command can only be used in a server.", ephemeral=True)
            return
        
        guild_id = interaction.guild.id
        guild_name = interaction.guild.name
        
        # Check if a scan is already running for this server
        if guild_id in active_scans:
            task = active_scans[guild_id]
            if not task.done():
                await interaction.response.send_message(
                    f"⚠️ A scan is already running for **{guild_name}**. Please wait for it to complete.",
                    ephemeral=True
                )
                return
        
        # Defer response since this will take a while
        await interaction.response.defer(ephemeral=True)
        
        # Create background task for the scan
        scan_task = asyncio.create_task(
            run_scan_background(interaction, guild_id, guild_name)
        )
        active_scans[guild_id] = scan_task
        
        # Clean up task when done
        def cleanup_task(task):
            if guild_id in active_scans and active_scans[guild_id] == task:
                del active_scans[guild_id]
        
        scan_task.add_done_callback(cleanup_task)
        
        await interaction.followup.send(
            f"🔍 Started background scan for **{guild_name}**. The bot will continue to respond to other commands while scanning.",
            ephemeral=True
        )


async def run_scan_background(interaction: discord.Interaction, guild_id: int, guild_name: str):
    """
    Background task that performs the actual scan.
    This runs concurrently with other bot functions.
    """
    try:
        # Run database operations in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Initialize database (run in thread pool)
        conn = await loop.run_in_executor(None, init_database, guild_id)
        
        # Update guild name in metadata (run in thread pool)
        def update_guild_name(conn, guild_id, guild_name):
            cursor = conn.cursor()
            ensure_guild_metadata_schedule_columns(conn)
            cursor.execute(
                """
                INSERT INTO guild_metadata (guild_id, guild_name, last_updated)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id) DO UPDATE SET
                    guild_name = excluded.guild_name,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (guild_id, guild_name),
            )
            conn.commit()
        
        await loop.run_in_executor(None, update_guild_name, conn, guild_id, guild_name)
        
        # Get last scan time (run in thread pool)
        last_scan = await loop.run_in_executor(None, get_last_scan_time, guild_id)
        is_first_scan = last_scan is None
        
        # Determine time range
        from datetime import timezone
        if is_first_scan:
            # First scan lookback configurable via FIRST_SCAN_LOOKBACK_DAYS (default 365).
            try:
                lookback_days = int(os.getenv("FIRST_SCAN_LOOKBACK_DAYS", "365"))
            except ValueError:
                lookback_days = 365
            if lookback_days < 1:
                lookback_days = 1
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            scan_type = "initial"
            try:
                await interaction.followup.send(
                    f"🔍 Starting initial scan for {guild_name} (going back {lookback_days} day{'s' if lookback_days != 1 else ''})...",
                    ephemeral=True,
                )
            except:
                pass  # Interaction may have expired, continue anyway
        else:
            # Only scan messages since last scan
            # Ensure cutoff_time is timezone-aware (UTC)
            if last_scan.tzinfo is None:
                cutoff_time = last_scan.replace(tzinfo=timezone.utc)
            else:
                cutoff_time = last_scan
            scan_type = "incremental"
            try:
                # Display in UTC for consistency
                utc_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')
                await interaction.followup.send(f"🔍 Starting incremental scan for {guild_name} (since {utc_str})...", ephemeral=True)
            except:
                pass  # Interaction may have expired, continue anyway
        
        # Get all channels the bot can access, excluding specified channels
        all_channels = [ch for ch in interaction.guild.channels if isinstance(ch, discord.TextChannel)]
        
        # Get excluded channel IDs from environment variable
        excluded_channel_ids_str = os.getenv("SCAN_EXCLUDE_CHANNEL_IDS", "")
        excluded_channel_ids = []
        if excluded_channel_ids_str:
            excluded_channel_ids = [
                int(cid.strip()) 
                for cid in excluded_channel_ids_str.split(",") 
                if cid.strip().isdigit()
            ]
        
        # Filter out excluded channels
        channels = [ch for ch in all_channels if ch.id not in excluded_channel_ids]
        total_channels = len(channels)
        
        # Log excluded channels if any
        if excluded_channel_ids:
            excluded_channels = [ch for ch in all_channels if ch.id in excluded_channel_ids]
            excluded_names = [f"#{ch.name}" for ch in excluded_channels]
            logger.info(f"  ⏭️  Excluding {len(excluded_channel_ids)} channel(s): {', '.join(excluded_names)}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"🔍 STARTING SCAN: {guild_name}")
        logger.info("=" * 70)
        logger.info(f"  Channels: {total_channels}")
        logger.info(f"  Type: {scan_type.upper()}")
        if is_first_scan:
            logger.info(f"  Time range: 12 months (from {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')})")
        else:
            try:
                local_time = cutoff_time.astimezone()
                local_tz = local_time.strftime('%Z') or 'local'
                logger.info(f"  Time range: Since {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({local_time.strftime('%Y-%m-%d %H:%M:%S')} {local_tz})")
            except:
                logger.info(f"  Time range: Since {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 70)
        logger.info("")
        
        total_messages_scanned = 0
        total_messages_added = 0
        total_images_processed = 0
        total_urls_processed = 0
        total_images_skipped = 0
        total_urls_skipped = 0
        
        # Commit counter for periodic database commits (handles interruptions)
        commit_interval = 50  # Commit every 50 messages
        
        for channel_idx, channel in enumerate(channels, 1):
            try:
                channel_name = channel.name
                channel_id = channel.id
                
                logger.info(f"📁 [{channel_idx}/{total_channels}] #{channel_name}")
                
                # Check if we can read message history
                if not channel.permissions_for(interaction.guild.me).read_message_history:
                    logger.warning(f"  ⚠️  No permission to read messages - skipping")
                    continue
                
                messages_processed = 0
                messages_added = 0
                channel_start_time = time.time()
                
                # Fetch messages
                try:
                    # For incremental scan, get messages after last scan.
                    # For first scan, cutoff_time was set above from FIRST_SCAN_LOOKBACK_DAYS.
                    # Ensure after_time is timezone-aware (Discord requires this)
                    from datetime import timezone
                    if cutoff_time.tzinfo is None:
                        after_time = cutoff_time.replace(tzinfo=timezone.utc)
                    else:
                        after_time = cutoff_time
                    
                    # Show both UTC and local time for clarity
                    try:
                        local_time = after_time.astimezone()
                        local_tz = local_time.strftime('%Z') or 'local'
                        logger.info(f"  → Fetching messages after {after_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ({local_time.strftime('%Y-%m-%d %H:%M:%S')} {local_tz})...")
                    except:
                        logger.info(f"  → Fetching messages after {after_time.strftime('%Y-%m-%d %H:%M:%S UTC')}...")
                    
                    # Fetch messages - use limit=None to get all messages after cutoff_time
                    # Discord will paginate automatically, we process with rate limiting
                    async for message in channel.history(limit=None, after=after_time, oldest_first=True):
                        messages_processed += 1
                        total_messages_scanned += 1
                        
                        # Check if message already exists (run in thread pool)
                        exists = await loop.run_in_executor(None, message_exists, conn, message.id)
                        if exists:
                            if messages_processed % 100 == 0:
                                logger.debug(f"  → Message {messages_processed} already exists, skipping")
                            continue
                        
                        # Extract message data
                        msg_date = message.created_at.strftime("%Y-%m-%d")
                        msg_time = message.created_at.strftime("%H:%M:%S")
                        username = str(message.author)
                        nickname = message.author.display_name if hasattr(message.author, 'display_name') else None
                        user_id = message.author.id
                        
                        # Combine message content with embed content
                        message_parts = []
                        if message.content:
                            message_parts.append(message.content)
                        
                        # Add embed content (link previews, GIF metadata, etc.)
                        has_embeds = False
                        if message.embeds:
                            has_embeds = True
                            for embed in message.embeds:
                                if embed.title:
                                    message_parts.append(f"[Embed Title: {embed.title}]")
                                if embed.description:
                                    message_parts.append(f"[Embed Description: {embed.description}]")
                                if embed.url:
                                    message_parts.append(f"[Embed URL: {embed.url}]")
                                # Add field values from embeds
                                if embed.fields:
                                    for field in embed.fields:
                                        if field.name and field.value:
                                            message_parts.append(f"[Embed Field: {field.name} = {field.value}]")
                        
                        message_content = " ".join(message_parts) if message_parts else None
                        
                        # Show message preview periodically (every 10 messages or if it has interesting content)
                        show_preview = (messages_processed % 10 == 0) or has_embeds or (message.attachments and any(not att.filename.lower().endswith('.gif') for att in message.attachments))
                        if show_preview and message_content:
                            preview = message_content[:80] + "..." if len(message_content) > 80 else message_content
                            embed_info = f" (+{len(message.embeds)} embed)" if has_embeds else ""
                            attach_info = f" (+{len(message.attachments)} file)" if message.attachments else ""
                            logger.info(f"  📝 [{msg_date} {msg_time}] {username}: {preview}{embed_info}{attach_info}")
                        
                        # Process image attachments (skip GIFs)
                        image_description = None
                        if message.attachments:
                            for attachment in message.attachments:
                                filename = attachment.filename.lower() if attachment.filename else ""
                                # Only process static images, skip GIFs
                                if filename.endswith(".gif"):
                                    total_images_skipped += 1
                                    if messages_processed % 50 == 0:
                                        logger.debug(f"  → Skipping GIF attachment: {attachment.filename}")
                                    continue
                                
                                if any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                                    try:
                                        logger.debug(f"  → Processing image attachment: {attachment.filename}")
                                        desc = await describe_image(attachment.url, attachment.filename)
                                        if desc:
                                            image_description = desc
                                            total_images_processed += 1
                                            logger.info(f"  ✅ Image described: {desc[:50]}...")
                                            # Only process first image to avoid too many API calls
                                            break
                                        else:
                                            logger.debug(f"  → Image description returned None for {attachment.filename}")
                                    except Exception as e:
                                        logger.warning(f"  ⚠️  Failed to describe image {attachment.filename}: {e}")
                        
                        # Process URLs in message content AND embeds
                        url_summary = None
                        
                        # Collect URLs from message content and embeds
                        all_urls = []
                        if message_content:
                            all_urls.extend(extract_urls(message_content))
                        
                        # Also extract URLs from embeds
                        if message.embeds:
                            for embed in message.embeds:
                                if embed.url:
                                    all_urls.append(embed.url)
                                # Check embed description for URLs
                                if embed.description:
                                    all_urls.extend(extract_urls(embed.description))
                        
                        # Remove duplicates and filter out excluded domains (like tenor.com)
                        seen = set()
                        unique_urls = []
                        excluded_domains = ['tenor.com', 'www.tenor.com']
                        for url in all_urls:
                            if url not in seen:
                                # Check if URL is from an excluded domain
                                parsed = urlparse(url)
                                domain = parsed.netloc.lower()
                                if any(excluded in domain for excluded in excluded_domains):
                                    total_urls_skipped += 1
                                    logger.debug(f"  → Skipping URL from excluded domain: {url}")
                                    seen.add(url)  # Mark as seen so we don't process it
                                    continue
                                seen.add(url)
                                unique_urls.append(url)
                        
                        if unique_urls:
                            # Check if URL points to an image/GIF first
                            first_url = unique_urls[0]
                            parsed = urlparse(first_url)
                            path_lower = parsed.path.lower()
                            if any(path_lower.endswith(ext) for ext in ['.gif', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.svg']):
                                total_urls_skipped += 1
                                logger.debug(f"  → Skipping URL that points to image/GIF: {first_url}")
                            else:
                                # Process first URL only to avoid too many API calls
                                try:
                                    logger.debug(f"  → Processing URL: {first_url}")
                                    async with aiohttp.ClientSession() as session:
                                        content = await extract_url_content(first_url, session)
                                        if content:
                                            # Use LLM to summarize the extracted content
                                            base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
                                            endpoint = f"{base}/chat/completions"
                                            
                                            headers = {"Content-Type": "application/json"}
                                            api_key = os.getenv("OPENAI_API_KEY")
                                            if api_key:
                                                headers["Authorization"] = f"Bearer {api_key}"
                                            
                                            model_name = os.getenv("LOCAL_CHAT")
                                            if model_name:
                                                summary_prompt = f"Summarize the following web content concisely in 2-3 sentences:\n\n{content}"
                                                
                                                payload = {
                                                    "model": model_name,
                                                    "messages": [{
                                                        "role": "user",
                                                        "content": summary_prompt
                                                    }],
                                                    "max_tokens": 200,
                                                    "temperature": 0.7,
                                                }
                                                
                                                async with aiohttp.ClientSession() as llm_session:
                                                    async with llm_session.post(endpoint, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
                                                        if r.status == 200:
                                                            out = await r.json()
                                                            summary = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                                                            if summary:
                                                                url_summary = summary
                                                                total_urls_processed += 1
                                                                logger.info(f"  ✅ URL summarized: {summary[:50]}...")
                                        else:
                                            logger.debug(f"  → URL content extraction returned None for {first_url}")
                                except Exception as e:
                                    logger.warning(f"  ⚠️  Failed to process URL {first_url}: {e}")
                        
                        # Insert message into database (run in thread pool to avoid blocking)
                        await loop.run_in_executor(
                            None,
                            insert_message,
                            conn,
                            message.id,
                            msg_date,
                            msg_time,
                            username,
                            nickname,
                            user_id,
                            message_content,
                            channel_id,
                            channel_name,
                            image_description,
                            url_summary
                        )
                        
                        messages_added += 1
                        total_messages_added += 1
                        
                        # Periodic commit to handle interruptions gracefully
                        if messages_added % commit_interval == 0:
                            await loop.run_in_executor(None, conn.commit)
                            logger.info(f"  💾 Committed {messages_added} messages to database (checkpoint)")
                        
                        # Rate limiting: be discrete to avoid Discord rate limits
                        # Vary delays to appear more human-like
                        if messages_processed % 5 == 0:
                            # Longer delay every 5 messages (0.8-2.0 seconds)
                            await asyncio.sleep(random.uniform(0.8, 2.0))
                        elif messages_processed % 2 == 0:
                            # Medium delay every other message (0.3-0.7 seconds)
                            await asyncio.sleep(random.uniform(0.3, 0.7))
                        else:
                            # Short delay otherwise (0.1-0.4 seconds)
                            await asyncio.sleep(random.uniform(0.1, 0.4))
                        
                        # Progress update every 25 messages
                        if messages_processed % 25 == 0:
                            elapsed = time.time() - channel_start_time
                            rate = messages_processed / elapsed if elapsed > 0 else 0
                            logger.info(f"  📊 [{channel_idx}/{total_channels}] Progress: {messages_processed} scanned, {messages_added} new | Total: {total_messages_scanned} scanned, {total_messages_added} new | Images: {total_images_processed} | URLs: {total_urls_processed} | Rate: {rate:.1f} msg/sec")
                
                except discord.Forbidden:
                    logger.warning(f"  ⚠️  No permission to read messages in #{channel_name} - skipping")
                    continue
                except Exception as e:
                    logger.error(f"  ❌ Error scanning channel #{channel_name}: {e}", exc_info=True)
                    continue
                
                # Final commit for this channel
                await loop.run_in_executor(None, conn.commit)
                
                channel_elapsed = time.time() - channel_start_time
                logger.info(f"  ✅ #{channel_name}: {messages_processed} scanned, {messages_added} new in {channel_elapsed:.1f}s")
                if total_images_processed > 0 or total_urls_processed > 0:
                    logger.info(f"     └─ Images: {total_images_processed} | URLs: {total_urls_processed}")
                
                # Small delay between channels
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing channel: {e}")
                continue
        
        # Final summary before recording
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📊 SCAN SUMMARY for {guild_name}")
        logger.info("=" * 70)
        logger.info(f"  Messages: {total_messages_scanned} scanned, {total_messages_added} new")
        logger.info(f"  Images: {total_images_processed} processed, {total_images_skipped} skipped (GIFs)")
        logger.info(f"  URLs: {total_urls_processed} processed, {total_urls_skipped} skipped (images/GIFs)")
        logger.info(f"  Channels: {total_channels} processed")
        logger.info("=" * 70)
        logger.info("")
        
        # Record the scan (run in thread pool)
        await loop.run_in_executor(None, record_scan, conn, scan_type, total_messages_added)
        
        # Close connection (run in thread pool)
        await loop.run_in_executor(None, conn.close)

        # Get final stats (run in thread pool)
        stats = await loop.run_in_executor(None, get_stats, guild_id)
        
        # Send completion message
        embed = discord.Embed(
            title="✅ Scan Complete",
            description=f"Scan completed for **{guild_name}**",
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="📊 Messages Scanned", value=str(total_messages_scanned), inline=True)
        embed.add_field(name="➕ Messages Added", value=str(total_messages_added), inline=True)
        embed.add_field(name="📁 Channels Processed", value=str(total_channels), inline=True)
        embed.add_field(name="🖼️ Images Processed", value=str(total_images_processed), inline=True)
        embed.add_field(name="⏭️ Images Skipped (GIFs)", value=str(total_images_skipped), inline=True)
        embed.add_field(name="🔗 URLs Processed", value=str(total_urls_processed), inline=True)
        embed.add_field(name="⏭️ URLs Skipped (Images)", value=str(total_urls_skipped), inline=True)
        embed.add_field(name="💾 Total in Database", value=str(stats["total_messages"]), inline=True)
        embed.set_footer(text=f"Scan type: {scan_type}")
        
        try:
            await interaction.followup.send(embed=embed, ephemeral=True)
        except:
            # Interaction may have expired, log instead
            logger.info(f"✅ Scan completed for {guild_name}: {total_messages_added} new messages added")
        
        logger.info(f"✅ Scan completed for {guild_name}: {total_messages_added} new messages added")

        async def _reindex_after_archive(gid: int, gname: str, new_count: int) -> None:
            # Only reindex if the scan actually found new messages.
            # Scans that find 0 new messages don't need a heavy consolidation pass —
            # the 6-hour rag_reindex_loop handles periodic consolidation.
            if new_count == 0:
                logger.debug("RAG reindex skipped after scan guild=%s: 0 new messages", gid)
                return
            try:
                from .rag import index_new_messages

                rag_result = await index_new_messages(gid)
                logger.info("RAG incremental index after scan guild=%s (%d new msgs): %s", gid, new_count, rag_result)
            except Exception as rag_exc:
                logger.warning(
                    "RAG incremental index after scan failed guild=%s: %s",
                    gid,
                    rag_exc,
                    exc_info=True,
                )

        # RAG reindex runs in the background so the scan releases active_scans
        # and the event loop stays free for chat and commands.
        asyncio.create_task(_reindex_after_archive(guild_id, guild_name, total_messages_added))

    except Exception as e:
        error_msg = f"❌ Error during scan: {str(e)}"
        logger.error(f"Scan error: {e}", exc_info=True)
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except:
            # Interaction may have expired, just log
            logger.error(f"Could not send error message to user: {error_msg}")


def get_active_scans() -> Dict[int, asyncio.Task]:
    """Get the active scans dictionary for external access (e.g., web API)."""
    return active_scans


def create_scan_trigger(guild_id: int) -> tuple[bool, str]:
    """
    Create a file-based trigger for a database scan.
    The bot will pick this up and process it.
    Returns (success: bool, message: str)
    """
    try:
        # Check if a scan is already running
        if guild_id in active_scans:
            task = active_scans[guild_id]
            if not task.done():
                return False, f"A scan is already running for guild {guild_id}"
        
        # Check if trigger file already exists
        trigger_file = os.path.join(SCAN_TRIGGER_DIR, f"trigger_{guild_id}.txt")
        if os.path.exists(trigger_file):
            return False, f"Scan trigger already pending for guild {guild_id}"
        
        # Create trigger file
        with open(trigger_file, 'w') as f:
            f.write(f"{guild_id}\n{datetime.utcnow().isoformat()}\n")
        
        logger.info(f"Created scan trigger for guild {guild_id}")
        return True, f"Scan trigger created for guild {guild_id}. Bot will process it shortly."
        
    except Exception as e:
        logger.error(f"Error creating scan trigger: {e}", exc_info=True)
        return False, f"Error: {str(e)}"


async def trigger_scan_programmatic(bot, guild_id: int) -> tuple[bool, str]:
    """
    Trigger a database scan programmatically (without Discord interaction).
    This is called from within the bot process.
    Returns (success: bool, message: str)
    """
    try:
        # Check if bot is available
        if not bot or not bot.is_ready():
            return False, "Bot is not ready"
        
        # Get guild
        guild = bot.get_guild(guild_id)
        if not guild:
            return False, f"Guild {guild_id} not found or bot not in guild"
        
        guild_name = guild.name
        
        # Check if a scan is already running
        if guild_id in active_scans:
            task = active_scans[guild_id]
            if not task.done():
                return False, f"A scan is already running for {guild_name}"
        
        # Create a mock interaction-like object for compatibility
        class MockInteraction:
            def __init__(self, guild):
                self.guild = guild
                self.followup = MockFollowup()
        
        class MockFollowup:
            async def send(self, *args, **kwargs):
                # Just log instead of sending to Discord
                logger.info(f"[Scan] {args[0] if args else ''}")
        
        mock_interaction = MockInteraction(guild)
        
        # Create background task for the scan
        scan_task = asyncio.create_task(
            run_scan_background(mock_interaction, guild_id, guild_name)
        )
        active_scans[guild_id] = scan_task
        
        # Clean up task when done
        def cleanup_task(task):
            if guild_id in active_scans and active_scans[guild_id] == task:
                del active_scans[guild_id]
        
        scan_task.add_done_callback(cleanup_task)
        
        return True, f"Started scan for {guild_name}"
        
    except Exception as e:
        logger.error(f"Error triggering scan: {e}", exc_info=True)
        return False, f"Error: {str(e)}"


async def process_scan_triggers(bot):
    """
    Check for scan trigger files and process them.
    This should be called periodically from the bot (e.g., in a task loop).
    """
    try:
        if not os.path.exists(SCAN_TRIGGER_DIR):
            return
        
        # Find all trigger files
        trigger_files = list(Path(SCAN_TRIGGER_DIR).glob("trigger_*.txt"))
        
        for trigger_file in trigger_files:
            try:
                # Read guild ID from file
                with open(trigger_file, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    guild_id = int(lines[0].strip())
                
                # Delete trigger file
                trigger_file.unlink()
                
                # Trigger the scan
                success, message = await trigger_scan_programmatic(bot, guild_id)
                if success:
                    logger.info(f"Processed scan trigger for guild {guild_id}: {message}")
                else:
                    logger.warning(f"Failed to process scan trigger for guild {guild_id}: {message}")
                    
            except Exception as e:
                logger.error(f"Error processing trigger file {trigger_file}: {e}")
                # Delete corrupted trigger file
                try:
                    trigger_file.unlink()
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Error processing scan triggers: {e}", exc_info=True)

