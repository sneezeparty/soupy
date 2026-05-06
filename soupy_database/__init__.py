"""
Soupy Database Module
"""

from .database import (
    create_scan_trigger,
    get_active_scans,
    get_archive_scan_interval_minutes,
    get_db_path,
    get_last_scan_time,
    get_stats,
    init_database,
    insert_message,
    message_exists,
    process_scan_triggers,
    record_scan,
    set_archive_scan_interval_minutes,
    setup_scan_command,
    trigger_scan_programmatic,
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

