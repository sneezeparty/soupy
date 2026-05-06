"""Shared logging formatter.

The bot, web app, and (formerly) the soupy_database tools each had their
own near-duplicate `ColoredFormatter` class. This module is the single
canonical implementation. Import `ColoredFormatter` from here and use
it directly:

    from soupy_logging import ColoredFormatter, PLAIN_FORMAT

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())

Existing call sites can keep their own subclass names (e.g.
`CustomFormatter`, `_WebFormatter`) by aliasing this class — that
preserves the names other code might `isinstance()` against.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

# ANSI escape sequences. Kept as module constants so other modules can
# style ad-hoc messages outside the logging system.
RESET = "\033[0m"
TIMESTAMP_COLOR = "\033[36m"  # cyan
ARROW_COLOR = "\033[90m"  # grey
NAME_COLOR = "\033[94m"  # blue

LEVEL_COLORS = {
    "DEBUG": "\033[95m",  # purple
    "INFO": "\033[92m",  # bright green
    "WARNING": "\033[93m",  # yellow
    "ERROR": "\033[91m",  # red
    "CRITICAL": "\033[41m",  # red background
}

# The plain-text format used for log files (no colour). Kept here so the
# file and console handlers agree on field order even if one drops colour.
PLAIN_FORMAT = "[%(asctime)s] (%(levelname)s) %(name)s => %(message)s"
PLAIN_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Coloured `[ts] (LEVEL) name => message` formatter.

    Replaces the previous `CustomFormatter` (main bot) and
    `_WebFormatter` (web app), which were copy-pastes of each other
    that drifted slightly over time.

    `datefmt` may include the literal `,f` to splice in milliseconds,
    matching the style the bot has used since v1.0.
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level_color = LEVEL_COLORS.get(record.levelname, "")
        colored_level = f"{level_color}({record.levelname}){RESET}"
        out = (
            f"{TIMESTAMP_COLOR}[{timestamp}]{RESET} "
            f"{colored_level} "
            f"{NAME_COLOR}{record.name}{RESET} "
            f"{ARROW_COLOR}=>{RESET} "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            out = f"{out}\n{self.formatException(record.exc_info)}"
        return out

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        ct = self.converter(record.created)
        if datefmt:
            msec = int((record.created - int(record.created)) * 1000)
            s = time.strftime(datefmt, ct)
            return s.replace(",f", f",{msec:03d}")
        return time.strftime(self.default_time_format, ct)
