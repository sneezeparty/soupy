#!/bin/bash
# Quick script to check what's in the database
#
# Usage: ./check_database.sh <guild_id>

if [ -z "$1" ]; then
    echo "Usage: $0 <guild_id>"
    echo "  guild_id: Discord guild/server ID (the database file is guild_<id>.db)"
    exit 1
fi

DB_PATH="soupy_database/databases/guild_$1.db"

if [ ! -f "$DB_PATH" ]; then
    echo "❌ Database file not found: $DB_PATH"
    exit 1
fi

echo "=========================================="
echo "📊 DATABASE CHECK"
echo "=========================================="
echo ""

echo "📈 Total Messages:"
sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM messages;"
echo ""

echo "📁 Channels Found:"
sqlite3 "$DB_PATH" "SELECT channel_name, COUNT(*) as count FROM messages GROUP BY channel_name ORDER BY count DESC;"
echo ""

echo "👥 Users Found:"
sqlite3 "$DB_PATH" "SELECT username, COUNT(*) as count FROM messages GROUP BY username ORDER BY count DESC LIMIT 10;"
echo ""

echo "📝 Sample Messages (Last 5):"
echo "----------------------------------------"
sqlite3 -header -column "$DB_PATH" <<EOF
SELECT 
    date || ' ' || time as timestamp,
    username,
    channel_name,
    CASE 
        WHEN length(message_content) > 60 THEN substr(message_content, 1, 60) || '...'
        ELSE message_content
    END as content,
    CASE 
        WHEN image_description IS NOT NULL THEN '✅'
        ELSE '❌'
    END as has_image,
    CASE 
        WHEN url_summary IS NOT NULL THEN '✅'
        ELSE '❌'
    END as has_url
FROM messages 
ORDER BY created_at DESC 
LIMIT 5;
EOF
echo ""

echo "🖼️ Messages with Image Descriptions:"
sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM messages WHERE image_description IS NOT NULL;"
echo ""

echo "🔗 Messages with URL Summaries:"
sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM messages WHERE url_summary IS NOT NULL;"
echo ""

echo "📅 Last Scan Info:"
sqlite3 -header -column "$DB_PATH" "SELECT scan_type, messages_scanned, datetime(last_scan_time) as last_scan FROM scan_metadata ORDER BY created_at DESC LIMIT 1;"
echo ""

echo "=========================================="
echo "✅ Check complete!"
echo "=========================================="

