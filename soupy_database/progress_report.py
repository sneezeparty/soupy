#!/usr/bin/env python3
"""
Comprehensive progress report for the database.

Usage:
    python progress_report.py <guild_id>
"""

import argparse
import sqlite3
import os
import sys
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(description="Print a detailed progress report for a guild's Soupy database.")
parser.add_argument("guild_id", help="Discord guild/server ID (the database file is guild_<id>.db)")
args = parser.parse_args()

db_dir = os.getenv("SOUPY_DB_DIR", os.path.join(os.path.dirname(__file__), "databases"))
db_path = os.path.join(db_dir, f"guild_{args.guild_id}.db")

if not os.path.exists(db_path):
    print(f"❌ Database file not found: {db_path}")
    sys.exit(1)

try:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 80)
    print("📊 DATABASE PROGRESS REPORT")
    print("=" * 80)
    print()
    
    # Total messages
    cursor.execute("SELECT COUNT(*) as count FROM messages")
    total_messages = cursor.fetchone()["count"]
    print(f"📈 Total Messages: {total_messages:,}")
    print()
    
    if total_messages == 0:
        print("⚠️  Database is empty")
        conn.close()
        sys.exit(0)
    
    # Total channels
    cursor.execute("SELECT COUNT(DISTINCT channel_id) as count FROM messages")
    total_channels = cursor.fetchone()["count"]
    print(f"📁 Total Channels: {total_channels:,}")
    print()
    
    # Messages with images
    cursor.execute("SELECT COUNT(*) as count FROM messages WHERE image_description IS NOT NULL")
    messages_with_images = cursor.fetchone()["count"]
    image_percentage = (messages_with_images / total_messages * 100) if total_messages > 0 else 0
    print(f"🖼️  Messages with Image Descriptions: {messages_with_images:,} ({image_percentage:.1f}%)")
    
    # Messages with URLs
    cursor.execute("SELECT COUNT(*) as count FROM messages WHERE url_summary IS NOT NULL")
    messages_with_urls = cursor.fetchone()["count"]
    url_percentage = (messages_with_urls / total_messages * 100) if total_messages > 0 else 0
    print(f"🔗 Messages with URL Summaries: {messages_with_urls:,} ({url_percentage:.1f}%)")
    
    # Messages with both
    cursor.execute("SELECT COUNT(*) as count FROM messages WHERE image_description IS NOT NULL AND url_summary IS NOT NULL")
    messages_with_both = cursor.fetchone()["count"]
    print(f"📎 Messages with Both Image & URL: {messages_with_both:,}")
    print()
    
    # Time period
    cursor.execute("SELECT MIN(date || ' ' || time) as earliest, MAX(date || ' ' || time) as latest FROM messages")
    time_result = cursor.fetchone()
    if time_result["earliest"] and time_result["latest"]:
        earliest = datetime.strptime(time_result["earliest"], '%Y-%m-%d %H:%M:%S')
        latest = datetime.strptime(time_result["latest"], '%Y-%m-%d %H:%M:%S')
        days = (latest - earliest).days
        print(f"📅 Time Period Covered:")
        print(f"   Earliest: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Latest:   {latest.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Span:     {days} days")
        print()
    
    # Messages per channel
    print("📁 Messages per Channel:")
    cursor.execute("""
        SELECT channel_name, COUNT(*) as count 
        FROM messages 
        GROUP BY channel_name 
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        percentage = (row["count"] / total_messages * 100) if total_messages > 0 else 0
        print(f"   #{row['channel_name']:20} {row['count']:6,} messages ({percentage:5.1f}%)")
    print()
    
    # Top users
    print("👥 Top 10 Users:")
    cursor.execute("""
        SELECT username, COUNT(*) as count 
        FROM messages 
        GROUP BY username 
        ORDER BY count DESC 
        LIMIT 10
    """)
    for idx, row in enumerate(cursor.fetchall(), 1):
        percentage = (row["count"] / total_messages * 100) if total_messages > 0 else 0
        print(f"   {idx:2}. {row['username']:25} {row['count']:6,} messages ({percentage:5.1f}%)")
    print()
    
    # Unique users
    cursor.execute("SELECT COUNT(DISTINCT user_id) as count FROM messages")
    unique_users = cursor.fetchone()["count"]
    print(f"👤 Unique Users: {unique_users:,}")
    print()
    
    # Database file size
    file_size = os.path.getsize(db_path)
    size_mb = file_size / (1024 * 1024)
    size_kb = file_size / 1024
    if size_mb >= 1:
        print(f"💾 Database Size: {size_mb:.2f} MB")
    else:
        print(f"💾 Database Size: {size_kb:.2f} KB")
    print()
    
    # Last scan info
    cursor.execute("""
        SELECT scan_type, messages_scanned, datetime(last_scan_time) as last_scan, datetime(created_at) as created
        FROM scan_metadata 
        ORDER BY created_at DESC 
        LIMIT 1
    """)
    scan = cursor.fetchone()
    if scan:
        print("📅 Last Scan:")
        print(f"   Type: {scan['scan_type']}")
        print(f"   Messages Scanned: {scan['messages_scanned']:,}")
        print(f"   Scan Time: {scan['last_scan']}")
        print(f"   Recorded: {scan['created']}")
    print()
    
    # Scan history summary
    cursor.execute("SELECT COUNT(*) as count FROM scan_metadata")
    total_scans = cursor.fetchone()["count"]
    print(f"🔄 Total Scans Performed: {total_scans}")
    
    if total_scans > 1:
        cursor.execute("""
            SELECT scan_type, COUNT(*) as count, SUM(messages_scanned) as total_scanned
            FROM scan_metadata
            GROUP BY scan_type
        """)
        print("   Breakdown:")
        for row in cursor.fetchall():
            print(f"      {row['scan_type']}: {row['count']} scan(s), {row['total_scanned']:,} messages")
    print()
    
    # Messages per day (average)
    if days > 0:
        avg_per_day = total_messages / days
        print(f"📊 Average Messages per Day: {avg_per_day:.1f}")
        print()
    
    print("=" * 80)
    print("✅ Report complete!")
    print("=" * 80)
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

