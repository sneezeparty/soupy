#!/usr/bin/env python3
"""
Show detailed database entries - all fields for a specific user, plus a sample full row.

Usage:
    python view_detailed.py <guild_id> <username>
"""

import argparse
import sqlite3
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Print every archived message from a given username, plus one full sample row.")
parser.add_argument("guild_id", help="Discord guild/server ID (the database file is guild_<id>.db)")
parser.add_argument("username", help="Discord username to filter on (case-sensitive, exact match against the messages.username column)")
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
    
    username = args.username

    print("=" * 80)
    print(f"📋 ALL MESSAGES FROM: {username}")
    print("=" * 80)
    print()
    
    cursor.execute("""
        SELECT 
            message_id,
            date,
            time,
            username,
            nickname,
            user_id,
            message_content,
            channel_id,
            channel_name,
            image_description,
            url_summary,
            created_at
        FROM messages 
        WHERE username = ?
        ORDER BY created_at ASC
    """, (username,))
    
    messages = cursor.fetchall()
    
    if not messages:
        print(f"❌ No messages found for {username}")
        conn.close()
        sys.exit(0)
    
    print(f"Found {len(messages)} message(s):\n")
    
    for idx, msg in enumerate(messages, 1):
        print("─" * 80)
        print(f"Message #{idx}")
        print("─" * 80)
        print(f"📅 Date:        {msg['date']}")
        print(f"⏰ Time:        {msg['time']}")
        print(f"🆔 Message ID:   {msg['message_id']}")
        print(f"👤 Username:     {msg['username']}")
        print(f"🏷️  Nickname:     {msg['nickname'] or '(none)'}")
        print(f"🔢 User ID:       {msg['user_id']}")
        print(f"📁 Channel:      #{msg['channel_name']} (ID: {msg['channel_id']})")
        print(f"💾 Created At:   {msg['created_at']}")
        print()
        print(f"💬 Message Content:")
        if msg['message_content']:
            # Show full content, wrapped
            content = msg['message_content']
            if len(content) > 200:
                print(f"   {content[:200]}...")
                print(f"   ... ({len(content)} total characters)")
            else:
                print(f"   {content}")
        else:
            print("   (no text content)")
        print()
        
        if msg['image_description']:
            print(f"🖼️  Image Description:")
            img_desc = msg['image_description']
            if len(img_desc) > 200:
                print(f"   {img_desc[:200]}...")
                print(f"   ... ({len(img_desc)} total characters)")
            else:
                print(f"   {img_desc}")
            print()
        else:
            print("🖼️  Image Description: (none)")
            print()
        
        if msg['url_summary']:
            print(f"🔗 URL Summary:")
            url_sum = msg['url_summary']
            if len(url_sum) > 200:
                print(f"   {url_sum[:200]}...")
                print(f"   ... ({len(url_sum)} total characters)")
            else:
                print(f"   {url_sum}")
            print()
        else:
            print("🔗 URL Summary: (none)")
            print()
    
    print("=" * 80)
    print()
    
    # Also show a sample of ALL fields from one message (any user)
    print("=" * 80)
    print("📊 SAMPLE: ONE COMPLETE DATABASE ENTRY (any user)")
    print("=" * 80)
    print()
    
    cursor.execute("""
        SELECT * FROM messages 
        ORDER BY created_at DESC 
        LIMIT 1
    """)
    
    sample = cursor.fetchone()
    if sample:
        print("All fields in the database:")
        print("-" * 80)
        for key in sample.keys():
            value = sample[key]
            if value is None:
                value_str = "(NULL)"
            elif isinstance(value, str) and len(value) > 100:
                value_str = f"{value[:100]}... ({len(value)} chars)"
            else:
                value_str = str(value)
            
            print(f"{key:20} = {value_str}")
    
    print()
    print("=" * 80)
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

