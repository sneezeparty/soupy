#!/usr/bin/env python3
"""
Quick script to check what's in the database.
Shows usernames, content, channels, etc. in a readable format.

Usage:
    python check_database.py <guild_id>
"""

import argparse
import sqlite3
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Show a quick summary of a guild's Soupy database.")
parser.add_argument("guild_id", help="Discord guild/server ID (the database file is guild_<id>.db)")
args = parser.parse_args()

db_dir = os.getenv("SOUPY_DB_DIR", os.path.join(os.path.dirname(__file__), "databases"))
db_path = os.path.join(db_dir, f"guild_{args.guild_id}.db")

if not os.path.exists(db_path):
    print(f"❌ Database file not found: {db_path}")
    print(f"   Looking in: {db_dir}")
    sys.exit(1)

try:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 70)
    print("📊 DATABASE CHECK")
    print("=" * 70)
    print()
    
    # Total messages
    cursor.execute("SELECT COUNT(*) as count FROM messages")
    total = cursor.fetchone()["count"]
    print(f"📈 Total Messages: {total:,}")
    print()
    
    if total == 0:
        print("⚠️  Database is empty - scan may not have started yet or no messages found.")
        conn.close()
        sys.exit(0)
    
    # Channels
    print("📁 Channels Found:")
    cursor.execute("""
        SELECT channel_name, COUNT(*) as count 
        FROM messages 
        GROUP BY channel_name 
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        print(f"   #{row['channel_name']}: {row['count']:,} messages")
    print()
    
    # Users
    print("👥 Top 10 Users:")
    cursor.execute("""
        SELECT username, COUNT(*) as count 
        FROM messages 
        GROUP BY username 
        ORDER BY count DESC 
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"   {row['username']}: {row['count']:,} messages")
    print()
    
    # Sample messages
    print("📝 Sample Messages (Last 5):")
    print("-" * 70)
    cursor.execute("""
        SELECT 
            date || ' ' || time as timestamp,
            username,
            channel_name,
            message_content,
            image_description,
            url_summary
        FROM messages 
        ORDER BY created_at DESC 
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        print(f"\n⏰ {row['timestamp']}")
        print(f"   👤 {row['username']} in #{row['channel_name']}")
        
        content = row['message_content']
        if content:
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"   💬 {preview}")
        else:
            print(f"   💬 (no text content)")
        
        if row['image_description']:
            img_preview = row['image_description'][:80] + "..." if len(row['image_description']) > 80 else row['image_description']
            print(f"   🖼️  Image: {img_preview}")
        
        if row['url_summary']:
            url_preview = row['url_summary'][:80] + "..." if len(row['url_summary']) > 80 else row['url_summary']
            print(f"   🔗 URL: {url_preview}")
    
    print()
    print("-" * 70)
    print()
    
    # Stats
    cursor.execute("SELECT COUNT(*) as count FROM messages WHERE image_description IS NOT NULL")
    images = cursor.fetchone()["count"]
    print(f"🖼️  Messages with Image Descriptions: {images:,}")
    
    cursor.execute("SELECT COUNT(*) as count FROM messages WHERE url_summary IS NOT NULL")
    urls = cursor.fetchone()["count"]
    print(f"🔗 Messages with URL Summaries: {urls:,}")
    print()
    
    # Last scan
    cursor.execute("""
        SELECT scan_type, messages_scanned, last_scan_time, created_at
        FROM scan_metadata 
        ORDER BY created_at DESC 
        LIMIT 1
    """)
    scan = cursor.fetchone()
    if scan:
        print("📅 Last Scan Info:")
        print(f"   Type: {scan['scan_type']}")
        print(f"   Messages Scanned: {scan['messages_scanned']:,}")
        print(f"   Time: {scan['last_scan_time']}")
    print()
    
    # Database file size
    file_size = os.path.getsize(db_path)
    size_mb = file_size / (1024 * 1024)
    print(f"💾 Database Size: {size_mb:.2f} MB")
    print()
    
    print("=" * 70)
    print("✅ Check complete!")
    print("=" * 70)
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

