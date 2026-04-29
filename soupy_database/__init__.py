"""
Soupy Database Module
"""

from .database import (
    init_database,
    get_last_scan_time,
    insert_message,
    message_exists,
    record_scan,
    get_stats,
    get_db_path,
    get_archive_scan_interval_minutes,
    set_archive_scan_interval_minutes,
    setup_scan_command,
    get_active_scans,
    trigger_scan_programmatic,
    create_scan_trigger,
    process_scan_triggers,
)

__all__ = [
    "init_database",
    "get_last_scan_time",
    "insert_message",
    "message_exists",
    "record_scan",
    "get_stats",
    "get_db_path",
    "get_archive_scan_interval_minutes",
    "set_archive_scan_interval_minutes",
    "setup_scan_command",
    "get_active_scans",
    "trigger_scan_programmatic",
    "create_scan_trigger",
    "process_scan_triggers",
]

