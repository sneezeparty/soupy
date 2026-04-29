#!/usr/bin/env python3
"""
Show conversations from the database - groups of messages within 15 minutes of each other.

Usage:
    python view_conversation.py <guild_id>
"""

import argparse
import sqlite3
import os
import sys
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description="Show example conversations reconstructed from a guild's Soupy database.")
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
    
    # Find conversations - groups of messages within 15 minutes
    print("=" * 80)
    print("💬 FINDING CONVERSATIONS (messages within 15 minutes)")
    print("=" * 80)
    print()
    
    # Get all messages ordered by time
    cursor.execute("""
        SELECT 
            message_id,
            date || ' ' || time as timestamp,
            username,
            nickname,
            message_content,
            image_description,
            url_summary,
            channel_name
        FROM messages 
        ORDER BY date, time
    """)
    
    all_messages = cursor.fetchall()
    
    if not all_messages:
        print("No messages found in database")
        conn.close()
        sys.exit(0)
    
    # Group messages into conversations (30 minute window, must have 2+ users)
    conversations = []
    current_conversation = []
    last_time = None
    time_window_minutes = 30  # Allow up to 30 minutes between messages
    
    for msg in all_messages:
        msg_time = datetime.strptime(msg['timestamp'], '%Y-%m-%d %H:%M:%S')
        
        if last_time is None:
            # First message starts a conversation
            current_conversation = [msg]
            last_time = msg_time
        else:
            # Check if this message is within time window of the last one
            time_diff = msg_time - last_time
            
            if time_diff <= timedelta(minutes=time_window_minutes) and msg['channel_name'] == current_conversation[0]['channel_name']:
                # Same conversation
                current_conversation.append(msg)
                last_time = msg_time
            else:
                # New conversation - save the old one if it has 2+ messages and 2+ users
                if len(current_conversation) >= 2:
                    unique_users = set(m['username'] for m in current_conversation)
                    if len(unique_users) >= 2:
                        conversations.append(current_conversation)
                # Start new conversation
                current_conversation = [msg]
                last_time = msg_time
    
    # Don't forget the last conversation
    if len(current_conversation) >= 2:
        unique_users = set(m['username'] for m in current_conversation)
        if len(unique_users) >= 2:
            conversations.append(current_conversation)
    
    print(f"Found {len(conversations)} conversation(s) with 2+ messages and 2+ users (within {time_window_minutes} minutes)")
    print()
    
    if not conversations:
        print(f"No conversations found (need at least 2 messages from 2+ users within {time_window_minutes} minutes)")
        conn.close()
        sys.exit(0)
    
    # Sort by length (longest first) to show more interesting conversations
    conversations.sort(key=len, reverse=True)
    
    # Show a few example conversations
    print("=" * 80)
    print(f"📋 EXAMPLE CONVERSATIONS (showing up to 5, sorted by length)")
    print("=" * 80)
    print()
    
    for conv_idx, conversation in enumerate(conversations[:5], 1):
        print("─" * 80)
        print(f"Conversation #{conv_idx}")
        print("─" * 80)
        
        # Get time span
        first_time = datetime.strptime(conversation[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
        last_time = datetime.strptime(conversation[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
        duration = last_time - first_time
        
        unique_users = set(m['username'] for m in conversation)
        print(f"📅 Channel: #{conversation[0]['channel_name']}")
        print(f"⏰ Started: {first_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ Ended:   {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {duration}")
        print(f"💬 Messages: {len(conversation)}")
        print(f"👥 Participants: {len(unique_users)} ({', '.join(sorted(unique_users))})")
        print()
        print("Messages:")
        print()
        
        for msg in conversation:
            msg_time = datetime.strptime(msg['timestamp'], '%Y-%m-%d %H:%M:%S')
            nickname_display = f" ({msg['nickname']})" if msg['nickname'] else ""
            
            print(f"[{msg_time.strftime('%H:%M:%S')}] {msg['username']}{nickname_display}:")
            
            if msg['message_content']:
                content = msg['message_content']
                # Wrap long content
                if len(content) > 200:
                    print(f"   {content[:200]}...")
                    print(f"   ... ({len(content)} total characters)")
                else:
                    print(f"   {content}")
            else:
                print("   (no text content)")
            
            if msg['image_description']:
                img_desc = msg['image_description']
                if len(img_desc) > 150:
                    print(f"   🖼️  [Image: {img_desc[:150]}...]")
                else:
                    print(f"   🖼️  [Image: {img_desc}]")
            
            if msg['url_summary']:
                url_sum = msg['url_summary']
                if len(url_sum) > 150:
                    print(f"   🔗 [URL: {url_sum[:150]}...]")
                else:
                    print(f"   🔗 [URL: {url_sum}]")
            
            print()
        
        if len(conversation) > 25:
            print(f"... and {len(conversation) - 25} more messages")
        print()
    
    print("=" * 80)
    print(f"📊 CONVERSATION STATISTICS")
    print("=" * 80)
    print()
    
    # Stats
    total_conv_messages = sum(len(c) for c in conversations)
    avg_conv_length = total_conv_messages / len(conversations) if conversations else 0
    longest_conv = max(len(c) for c in conversations) if conversations else 0
    
    print(f"Total conversations: {len(conversations)}")
    print(f"Total messages in conversations: {total_conv_messages:,}")
    print(f"Average conversation length: {avg_conv_length:.1f} messages")
    print(f"Longest conversation: {longest_conv} messages")
    print()
    
    # Show longest conversation
    if conversations:
        longest = max(conversations, key=len)
        print("=" * 80)
        print(f"📋 LONGEST CONVERSATION ({len(longest)} messages)")
        print("=" * 80)
        print()
        
        first_time = datetime.strptime(longest[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
        last_time = datetime.strptime(longest[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
        duration = last_time - first_time
        
        print(f"📅 Channel: #{longest[0]['channel_name']}")
        print(f"⏰ {first_time.strftime('%Y-%m-%d %H:%M:%S')} to {last_time.strftime('%Y-%m-%d %H:%M:%S')} ({duration})")
        print()
        
        for msg in longest[:10]:  # Show first 10 messages
            msg_time = datetime.strptime(msg['timestamp'], '%Y-%m-%d %H:%M:%S')
            nickname_display = f" ({msg['nickname']})" if msg['nickname'] else ""
            
            print(f"[{msg_time.strftime('%H:%M:%S')}] {msg['username']}{nickname_display}:")
            
            if msg['message_content']:
                content = msg['message_content']
                if len(content) > 150:
                    print(f"   {content[:150]}...")
                else:
                    print(f"   {content}")
            else:
                print("   (no text)")
            
            if msg['image_description']:
                print(f"   🖼️  [Has image description]")
            
            if msg['url_summary']:
                print(f"   🔗 [Has URL summary]")
            
            print()
        
        if len(longest) > 10:
            print(f"... and {len(longest) - 10} more messages")
        print()
    
    print("=" * 80)
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

