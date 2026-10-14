"""
FastAPI web control panel for Soupy.

Owns the bot subprocess (via :class:`web.services.bot_runner.BotRunner`),
streams its log over a WebSocket (via :class:`web.services.log_stream.WebsocketManager`),
edits ``.env-stable`` (via :mod:`web.services.env_store`), and exposes
archive/stats/runtime-flag JSON endpoints for the dashboard React app.

How it fits together:

* ``run_all.py`` launches uvicorn with ``SOUPY_AUTOSTART_BOT=1``. The
  ``startup`` event below then calls ``BotRunner.start()``, which is the
  *only* way the bot subprocess gets spawned in normal operation. Restarting
  uvicorn alone won't restart the bot — use ``POST /api/bot/restart`` (or
  the dashboard button) for that.
* The web app and the bot read ``.env-stable`` independently. Edits via
  ``POST /api/env/save`` rewrite the file and take effect on the next bot
  restart. Runtime flags (RAG enabled, command disables) live in
  ``data/runtime_flags.json`` and are mtime-cached on the bot side, so
  those changes propagate without a restart.
* Bot status JSON is written by the bot to ``data/bot_dashboard.json``
  every ~15s; this app reads it for the dashboard. The web app is *not*
  in-process with the bot — they only share files + the PTY.

Cross-module:

* ``soupy_database.database`` / ``soupy_database.rag`` /
  ``soupy_database.user_profiles`` — read directly (per-guild SQLite) to
  serve stats and trigger management actions.
* ``soupy.settings`` — typed env access for web-only settings
  (``web_control_panel_title``, ``timezone``, ``WEB_COLOR_*``).

Gotchas:

* The ``LOG_LEVEL`` env var only filters the *console* handler; the
  rotating file handler always captures DEBUG. Web log lines, like bot
  log lines, end up in ``logs/soupy.log``.
* The ``SuppressNoisyAccess`` log filter inside ``create_app`` drops
  thumb 304s and the dashboard's batch-status polling — without it,
  uvicorn's access log is unreadable.
* ANSI escape codes are stripped from bot output before it reaches the
  WebSocket so the browser console isn't full of raw escape sequences.
* All ``/api/*`` endpoints are unauthenticated — assume the panel runs
  on a trusted LAN. Do *not* expose port 4941 to the public internet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Coloured logging — uses the canonical formatter from soupy_logging so
# the web app and the bot agree on output format.
# ---------------------------------------------------------------------------
from soupy.log import ColoredFormatter

_web_fmt = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S,f")

# Plain-text formatter for file (no ANSI colors)
_file_fmt = logging.Formatter(
    "[%(asctime)s] (%(levelname)s) %(name)s => %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_web_handler = logging.StreamHandler(sys.stdout)
_web_handler.setFormatter(_web_fmt)

# Also write to the bot's log file so everything is in one place
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)
try:
    from logging.handlers import RotatingFileHandler

    _web_file_handler = RotatingFileHandler(
        _log_dir / "soupy.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _web_file_handler.setFormatter(_file_fmt)
except Exception:
    _web_file_handler = None

logger = logging.getLogger("web.app")
logger.setLevel(logging.DEBUG)
logger.addHandler(_web_handler)
if _web_file_handler:
    logger.addHandler(_web_file_handler)
logger.propagate = False

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for Python < 3.9
    from backports.zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from soupy.settings import settings

from .services import browse, lm_studio, prompt_store
from .services.bot_runner import BotRunner
from .services.env_store import parse_env, write_env
from .services.flux_runner import FluxRunner
from .services.log_stream import WebsocketManager

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"


def _configure_soupy_database_console_logging() -> None:
    """Route soupy_database.* INFO logs to stderr (root is often WARNING-only)."""
    lg = logging.getLogger("soupy_database")
    if getattr(lg, "_soupy_console_configured", False):
        return
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(h)
    lg.propagate = False
    lg._soupy_console_configured = True  # type: ignore[attr-defined]


# Same env file as the Discord bot (RAG reindex, OPENAI_BASE_URL, etc.)
load_dotenv(BASE_DIR / ".env-stable")


def _sqlite_guild_db_paths(base_dir: Path, server_id: str | None) -> list[Path]:
    """Paths to per-guild message archive DBs (SQLite)."""
    db_dir = Path(settings.soupy_db_dir or str(base_dir / "soupy_database" / "databases"))
    if not db_dir.is_dir():
        return []
    if server_id:
        sid = "".join(c for c in str(server_id).strip() if c.isdigit())
        if not sid:
            return []
        p = db_dir / f"guild_{sid}.db"
        return [p] if p.is_file() else []
    return sorted(db_dir.glob("guild_*.db"), key=lambda x: x.name)


def _merge_sqlite_message_stats_for_summary(
    *,
    base_dir: Path,
    server_id: str | None,
    days_list: list[str],
    limit: int,
) -> dict[str, Any] | None:
    """
    Aggregate message counts and leaders from scanned Discord archive DBs.
    This is the authoritative source for historic channel/message totals; messages.jsonl
    only records a subset of bot/archiver events.
    """
    import sqlite3
    from datetime import timedelta

    paths = _sqlite_guild_db_paths(base_dir, server_id)
    if not paths:
        return None

    now_naive_utc = datetime.utcnow()
    cut24 = (now_naive_utc - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    cut48 = (now_naive_utc - timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")
    cut7 = (now_naive_utc - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    cut30 = (now_naive_utc - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    now_epoch = now_naive_utc.replace(tzinfo=timezone.utc).timestamp()

    total_messages = 0
    last24 = last48 = last7d = last30d = 0
    messages_by_day = {d: 0 for d in days_list}
    hourly_msg = [0] * 24
    user_acc: dict[int, list[Any]] = {}
    ch_acc: dict[int, dict[str, Any]] = {}

    for db_path in paths:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM messages")
            n_msgs = int(cur.fetchone()["n"])
            total_messages += n_msgs
            if n_msgs == 0:
                continue

            cur.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE (date || ' ' || time) >= ?",
                (cut24,),
            )
            last24 += int(cur.fetchone()["n"])
            cur.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE (date || ' ' || time) >= ?",
                (cut48,),
            )
            last48 += int(cur.fetchone()["n"])
            cur.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE (date || ' ' || time) >= ?",
                (cut7,),
            )
            last7d += int(cur.fetchone()["n"])
            cur.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE (date || ' ' || time) >= ?",
                (cut30,),
            )
            last30d += int(cur.fetchone()["n"])

            if days_list:
                placeholders = ",".join(["?"] * len(days_list))
                cur.execute(
                    f"SELECT date AS d, COUNT(*) AS c FROM messages WHERE date IN ({placeholders}) GROUP BY date",
                    days_list,
                )
                for r in cur.fetchall():
                    dk = r["d"]
                    if dk in messages_by_day:
                        messages_by_day[dk] += int(r["c"])

            cur.execute(
                "SELECT date, time FROM messages WHERE (date || ' ' || time) >= ?",
                (cut24,),
            )
            for r in cur.fetchall():
                try:
                    dt = datetime.strptime(f'{r["date"]} {r["time"]}', "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    te = dt.timestamp()
                except Exception:
                    continue
                if te < now_epoch - 24 * 3600:
                    continue
                slot = int((now_epoch - te) // 3600)
                if 0 <= slot < 24:
                    hourly_msg[23 - slot] += 1

            cur.execute("SELECT user_id, username, COUNT(*) AS n FROM messages GROUP BY user_id, username")
            for r in cur.fetchall():
                uid = int(r["user_id"])
                uname = r["username"] or "Unknown"
                n = int(r["n"])
                if uid not in user_acc:
                    user_acc[uid] = [n, uname, n]
                else:
                    user_acc[uid][0] += n
                    if n > user_acc[uid][2]:
                        user_acc[uid][1] = uname
                        user_acc[uid][2] = n

            cur.execute(
                "SELECT channel_id, channel_name, COUNT(*) AS n FROM messages GROUP BY channel_id, channel_name"
            )
            for r in cur.fetchall():
                cid = int(r["channel_id"])
                cname = (r["channel_name"] or "").strip() or "unknown"
                n = int(r["n"])
                if cid not in ch_acc:
                    ch_acc[cid] = {"count": n, "name": cname, "best_n": n}
                else:
                    ch_acc[cid]["count"] += n
                    if n > ch_acc[cid]["best_n"]:
                        ch_acc[cid]["name"] = cname
                        ch_acc[cid]["best_n"] = n
        except Exception:
            # One guild DB being unreadable shouldn't sink the whole summary.
            logger.debug("Skipped a guild DB while merging message stats", exc_info=True)
            continue
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass  # already failing; nothing useful to do

    if total_messages <= 0:
        return None

    users_ranked = sorted(
        ((v[1], v[0]) for v in user_acc.values()),
        key=lambda t: (-t[1], t[0].lower()),
    )[:limit]

    channels_ranked = sorted(ch_acc.items(), key=lambda item: -item[1]["count"])[:limit]
    channel_id_to_name = {str(cid): data["name"] for cid, data in ch_acc.items()}

    hour_labels = [(now_naive_utc - timedelta(hours=23 - i)).strftime("%m-%d %H:00") for i in range(24)]

    return {
        "total_messages": total_messages,
        "last24_messages": last24,
        "last48_messages": last48,
        "last7d_messages": last7d,
        "last30d_messages": last30d,
        "messages_by_day": messages_by_day,
        "hourly_msg": hourly_msg,
        "hour_labels": hour_labels,
        "users_by_messages": [{"username": u, "messages": c} for u, c in users_ranked],
        "channels_by_messages": [
            {"channel_id": str(cid), "channel_name": data["name"], "messages": data["count"]}
            for cid, data in channels_ranked
        ],
        "channel_id_to_name": channel_id_to_name,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Soupy Control", version="0.1.0")

    # Get control panel title from environment
    control_panel_title = settings.web_control_panel_title

    # Get timezone from environment (default to America/Los_Angeles for California)
    display_timezone_str = settings.timezone
    try:
        ZoneInfo(display_timezone_str)
    except Exception:
        # Fallback to UTC if timezone is invalid
        logging.warning(f"Invalid timezone '{display_timezone_str}', falling back to UTC")
        _display_timezone = ZoneInfo("UTC")
        display_timezone_str = "UTC"

    # Get color scheme from environment
    colors = {
        "page_bg": os.getenv("WEB_COLOR_PAGE_BG", "#11191f"),
        "card_bg": os.getenv("WEB_COLOR_CARD_BG", "#1a2332"),
        "card_border": os.getenv("WEB_COLOR_CARD_BORDER", "#374151"),
        "text_primary": os.getenv("WEB_COLOR_TEXT_PRIMARY", "#e5e7eb"),
        "text_secondary": os.getenv("WEB_COLOR_TEXT_SECONDARY", "#9ca3af"),
        "text_muted": os.getenv("WEB_COLOR_TEXT_MUTED", "#6b7280"),
        "tab_inactive_bg": os.getenv("WEB_COLOR_TAB_INACTIVE_BG", "#e5e7eb"),
        "tab_inactive_text": os.getenv("WEB_COLOR_TAB_INACTIVE_TEXT", "#6b7280"),
        "tab_inactive_border": os.getenv("WEB_COLOR_TAB_INACTIVE_BORDER", "#d1d5db"),
        "tab_active_bg": os.getenv("WEB_COLOR_TAB_ACTIVE_BG", "#1f2937"),
        "tab_active_text": os.getenv("WEB_COLOR_TAB_ACTIVE_TEXT", "#f9fafb"),
        "tab_active_border": os.getenv("WEB_COLOR_TAB_ACTIVE_BORDER", "#374151"),
        "tab_content_bg": os.getenv("WEB_COLOR_TAB_CONTENT_BG", "#1f2937"),
        "tab_content_text": os.getenv("WEB_COLOR_TAB_CONTENT_TEXT", "#e5e7eb"),
        "tab_content_border": os.getenv("WEB_COLOR_TAB_CONTENT_BORDER", "#374151"),
        "console_bg": os.getenv("WEB_COLOR_CONSOLE_BG", "#0b1020"),
        "console_text": os.getenv("WEB_COLOR_CONSOLE_TEXT", "#c8d1f3"),
        "console_border": os.getenv("WEB_COLOR_CONSOLE_BORDER", "#272b42"),
        "status_running": os.getenv("WEB_COLOR_STATUS_RUNNING", "#2ecc71"),
        "status_stopped": os.getenv("WEB_COLOR_STATUS_STOPPED", "#e74c3c"),
        "env_field_bg": os.getenv("WEB_COLOR_ENV_FIELD_BG", "#ffffff"),
        "env_field_text": os.getenv("WEB_COLOR_ENV_FIELD_TEXT", "#111827"),
        "env_field_border": os.getenv("WEB_COLOR_ENV_FIELD_BORDER", "#e5e7eb"),
        "env_popover_bg": os.getenv("WEB_COLOR_ENV_POPOVER_BG", "#ffffff"),
        "env_popover_text": os.getenv("WEB_COLOR_ENV_POPOVER_TEXT", "#111827"),
        "env_popover_border": os.getenv("WEB_COLOR_ENV_POPOVER_BORDER", "#e5e7eb"),
        "env_tab_active_bg": os.getenv("WEB_COLOR_ENV_TAB_ACTIVE_BG", "#111827"),
        "env_tab_active_text": os.getenv("WEB_COLOR_ENV_TAB_ACTIVE_TEXT", "#ffffff"),
    }

    # Templates and static
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    # Add global context for templates
    templates.env.globals["control_panel_title"] = control_panel_title
    templates.env.globals["colors"] = colors
    templates.env.globals["display_timezone"] = display_timezone_str
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    media_dir = BASE_DIR / "media"
    # Ensure media directories exist so mount is reliable
    try:
        (media_dir / "images").mkdir(parents=True, exist_ok=True)
        (media_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    except Exception:
        # Not fatal, but the /media mount below will 404 — make the cause visible.
        logger.warning("Could not create media directories under %s", media_dir, exc_info=True)
    app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")

    # Shared state
    ws_manager = WebsocketManager()

    async def fanout_output(message: str) -> None:
        # Mirror to server console and broadcast to connected web clients
        try:
            print(message, flush=True)
        except Exception:
            pass  # console print is best-effort; logging here could recurse
        # Strip ANSI color codes for the web UI
        ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        clean = ansi_re.sub("", message)
        await ws_manager.broadcast_text(clean)

    bot_runner = BotRunner(project_root=BASE_DIR, on_output=fanout_output)
    flux_runner = FluxRunner(project_root=BASE_DIR, on_output=fanout_output)
    media_archive = browse.MediaArchive(media_dir)

    def _open_guild_db(guild_id: str):
        """``(connection, None)`` for a guild's archive, or ``(None, error response)``."""
        from soupy_database.database import get_db_path

        gid = browse.parse_snowflake(guild_id)
        if gid is None:
            return None, JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        db_path = get_db_path(gid)
        if not os.path.exists(db_path):
            return None, JSONResponse({"ok": False, "message": "Database not found"}, status_code=404)
        return browse.connect(db_path), None

    def _channel_names(guild_id: str) -> Dict[str, str]:
        conn, _error = _open_guild_db(guild_id)
        if conn is None:
            return {}
        try:
            return {ch["id"]: ch["name"] for ch in browse.list_channels(conn)}
        except Exception:
            logger.debug("Could not read channel names for guild %s", guild_id, exc_info=True)
            return {}
        finally:
            conn.close()

    app.state.templates = templates
    app.state.ws_manager = ws_manager
    app.state.bot_runner = bot_runner
    app.state.flux_runner = flux_runner

    _configure_soupy_database_console_logging()

    # Reduce log noise: thumb 304s, the dashboard's polling endpoints
    try:

        class SuppressNoisyAccess(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    msg = record.getMessage()
                except Exception:
                    # Fail open: if we can't format the record, don't suppress it.
                    return True
                if "/media/thumbs/" in msg and " 304 " in msg:
                    return False
                if "GET /api/profiles/batch/status/" in msg:
                    return False
                # The dashboard's status bar polls these every 20 seconds.
                if "GET /api/bot/status " in msg or "GET /api/bot/dashboard-status " in msg:
                    return False
                if "GET /api/flux/status " in msg:
                    return False
                return True

        logging.getLogger("uvicorn.access").addFilter(SuppressNoisyAccess())
    except Exception:
        # Filter is a nicety; without it the access log is just noisier.
        logger.debug("Could not install uvicorn access-log filter", exc_info=True)

    # Add strong caching for media/static to reduce revalidation requests
    @app.middleware("http")
    async def _cache_headers(request: Request, call_next):
        response = await call_next(request)
        p = request.url.path
        if p.startswith("/media/") or p.startswith("/static/"):
            # One week, immutable to discourage revalidation
            response.headers.setdefault("Cache-Control", "public, max-age=604800, immutable")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        status = await bot_runner.status()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "status": status,
                "now": datetime.now(timezone.utc),
            },
        )

    # -----------------------------------------------------------------------
    # Archive endpoints — image gallery + message viewer for /media archive
    # -----------------------------------------------------------------------

    @app.get("/api/archive/images")
    async def api_archive_images(limit: int = 60, offset: int = 0, kind: str | None = None, q: str | None = None):
        """One page of the image archive, newest first. Vision rows carry the whole description."""
        page = media_archive.images(kind=(kind or "all").strip().lower(), search=q or "", limit=limit, offset=offset)
        browse.attach_channel_names(page["items"], _channel_names)
        return JSONResponse(page)

    @app.get("/api/archive/messages")
    async def api_archive_messages(limit: int = 50, offset: int = 0):
        return JSONResponse(media_archive.log(limit=limit, offset=offset))

    # -----------------------------------------------------------------------
    # Stats endpoints — user_stats.json + per-guild SQLite roll-ups
    # -----------------------------------------------------------------------

    @app.get("/api/stats/raw")
    async def api_stats_raw():

        stats_path = BASE_DIR / "user_stats.json"
        if not stats_path.exists():
            return JSONResponse({}, status_code=200)
        try:
            return JSONResponse(json.loads(stats_path.read_text(encoding="utf-8")))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/stats/summary")
    async def api_stats_summary(server_id: str | None = None, limit: int = 5):
        import datetime
        from collections import Counter

        stats_path = BASE_DIR / "user_stats.json"
        media_messages = media_dir / "messages.jsonl"
        images_index = media_dir / "images" / "index.jsonl"

        # Base structure
        result: dict[str, object] = {
            "totals": {
                "images": 0,
                "images_generated": 0,
                "images_analyzed": 0,
                "chats": 0,
                "messages": 0,
                "users": 0,
            },
            "last24_messages": 0,
            "last48_messages": 0,
            "last7d_messages": 0,
            "last30d_messages": 0,
            "series_7d": {
                "days": [],
                "messages": [],
                "images_generated": [],
                "images_analyzed": [],
                "other_messages": [],
            },
            "hourly_24h": {
                "labels": [],
                "messages": [],
                "images_generated": [],
                "images_analyzed": [],
            },
            "event_totals": {},
            "top": {
                "users_by_images": [],
                "users_by_vision": [],
                "users_by_messages": [],
                "channels_by_messages": [],
                "channels_by_images": [],
            },
            # Legacy fields for compatibility with existing UI (will keep for now)
            "top_images": [],
            "top_chats": [],
        }

        # 7-day window boundaries (UTC, inclusive of today)
        today = datetime.datetime.utcnow().date()
        days_list = [(today - datetime.timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
        result["series_7d"]["days"] = days_list
        messages_by_day = {d: 0 for d in days_list}
        gen_by_day = {d: 0 for d in days_list}
        vis_by_day = {d: 0 for d in days_list}

        # Aggregate from user_stats.json (known users and global totals)
        if stats_path.exists():
            try:
                data = json.loads(stats_path.read_text(encoding="utf-8"))
                result["totals"]["users"] = len(data)

                rows = []
                chat_rows = []
                for _uid, u in data.items():
                    servers = u.get("servers", {})
                    scope = servers.get(server_id) if server_id else servers.get("global") or {}
                    rows.append(
                        {
                            "username": u.get("username", "Unknown"),
                            "images_generated": int(scope.get("images_generated", 0)),
                        }
                    )
                    chat_rows.append(
                        {
                            "username": u.get("username", "Unknown"),
                            "chat_responses": int(scope.get("chat_responses", 0)),
                        }
                    )

                # Totals from global if available
                for _uid, u in data.items():
                    g = u.get("servers", {}).get("global", {})
                    result["totals"]["images_generated"] += int(g.get("images_generated", 0))
                    result["totals"]["chats"] += int(g.get("chat_responses", 0))

                result["top_images"] = sorted(rows, key=lambda r: r["images_generated"], reverse=True)[:limit]
                result["top_chats"] = sorted(chat_rows, key=lambda r: r["chat_responses"], reverse=True)[:limit]
                # Use long-term totals for top.users_by_images
                result["top"]["users_by_images"] = result["top_images"]
            except Exception:
                logger.debug("Failed to derive totals from user_stats", exc_info=True)

        # Optional channel name mapping
        channel_name_map: dict[str, str] = {}
        try:
            # File-based mapping
            ch_file = media_dir / "channel_names.json"
            if ch_file.exists():
                channel_name_map.update(json.loads(ch_file.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Could not parse channel_names.json", exc_info=True)
        try:
            # Env pair list: CHANNEL_NAMES = "123:general,456:random"
            raw_pairs = settings.channel_names_raw or None
            if raw_pairs:
                for pair in raw_pairs.split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    if ":" in pair:
                        cid, name = pair.split(":", 1)
                        channel_name_map[str(cid.strip())] = name.strip()
        except Exception:
            logger.warning("Could not parse CHANNEL_NAMES env var", exc_info=True)
        try:
            # Env JSON mapping: CHANNEL_NAMES_JSON = '{"123":"general"}'
            raw_json = settings.channel_names_json or None
            if raw_json:
                channel_name_map.update(json.loads(raw_json))
        except Exception:
            logger.warning("Could not parse CHANNEL_NAMES_JSON env var", exc_info=True)

        # Scan messages.jsonl for last 24h count, 7d series and top users/channels
        user_msg_counter: Counter[str] = Counter()
        channel_msg_counter: Counter[str] = Counter()
        user_vision_counter: Counter[str] = Counter()
        user_image_msg_counter: Counter[str] = Counter()

        hourly_msg = [0] * 24
        hourly_gen = [0] * 24
        hourly_vis = [0] * 24
        event_totals: Counter[str] = Counter()

        if media_messages.exists():
            total_messages = 0
            try:
                now_epoch = datetime.datetime.utcnow().timestamp()
                now_dt = datetime.datetime.utcfromtimestamp(now_epoch)
                cutoff_24 = now_epoch - 24 * 3600
                cutoff_48 = now_epoch - 48 * 3600
                cutoff_7d = now_epoch - 7 * 24 * 3600
                cutoff_30d = now_epoch - 30 * 24 * 3600
                with open(media_messages, "r", encoding="utf-8") as f:
                    for line in f:
                        # Count every line as a message
                        total_messages += 1
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        ts = obj.get("ts")
                        try:
                            tdt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
                        except Exception:
                            tdt = None
                        if tdt is None:
                            continue
                        t_epoch = tdt.timestamp()
                        et = obj.get("event_type")
                        et_key = str(et) if et is not None else "unknown"
                        event_totals[et_key] += 1

                        if t_epoch >= cutoff_24:
                            result["last24_messages"] += 1
                            slot = int((now_epoch - t_epoch) // 3600)
                            if 0 <= slot < 24:
                                bi = 23 - slot
                                hourly_msg[bi] += 1
                                if et == "image_generation":
                                    hourly_gen[bi] += 1
                                elif et == "vision":
                                    hourly_vis[bi] += 1
                        if t_epoch >= cutoff_48:
                            result["last48_messages"] += 1
                        if t_epoch >= cutoff_7d:
                            result["last7d_messages"] += 1
                        if t_epoch >= cutoff_30d:
                            result["last30d_messages"] += 1

                        day_key = tdt.date().isoformat()
                        if day_key in messages_by_day:
                            messages_by_day[day_key] += 1

                        username = obj.get("username") or "Unknown"
                        user_msg_counter[username] += 1
                        ch_id = str(obj.get("channel_id") or "unknown")
                        channel_msg_counter[ch_id] += 1

                        if et == "image_generation":
                            if day_key in gen_by_day:
                                gen_by_day[day_key] += 1
                            user_image_msg_counter[username] += 1
                        elif et == "vision":
                            if day_key in vis_by_day:
                                vis_by_day[day_key] += 1
                            user_vision_counter[username] += 1

                hour_labels = [(now_dt - datetime.timedelta(hours=23 - i)).strftime("%m-%d %H:00") for i in range(24)]
                result["hourly_24h"]["labels"] = hour_labels
                result["hourly_24h"]["messages"] = hourly_msg
                result["hourly_24h"]["images_generated"] = hourly_gen
                result["hourly_24h"]["images_analyzed"] = hourly_vis
                result["event_totals"] = dict(sorted(event_totals.items(), key=lambda kv: (-kv[1], kv[0])))
            except Exception:
                logger.warning("Failed while scanning %s for stats", media_messages, exc_info=True)
            # Set total messages after scan
            result["totals"]["messages"] = total_messages
        else:
            hour_now = datetime.datetime.utcnow()
            result["hourly_24h"]["labels"] = [
                (hour_now - datetime.timedelta(hours=23 - i)).strftime("%m-%d %H:00") for i in range(24)
            ]
            result["hourly_24h"]["messages"] = hourly_msg
            result["hourly_24h"]["images_generated"] = hourly_gen
            result["hourly_24h"]["images_analyzed"] = hourly_vis

        # Images index aggregation (totals and generated/analyzed breakdown)
        gen_total = 0
        vis_total = 0
        user_images_index_counter: Counter[str] = Counter()
        channel_images_index_counter: Counter[str] = Counter()
        if images_index.exists():
            try:
                with open(images_index, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            o = json.loads(line)
                        except Exception:
                            continue
                        et = o.get("event_type")
                        if et == "vision":
                            vis_total += 1
                        else:
                            gen_total += 1
                            # Only count generated images for user/channel tops here
                            user_images_index_counter[o.get("username") or "Unknown"] += 1
                            channel_images_index_counter[str(o.get("channel_id") or "unknown")] += 1
            except Exception:
                logger.warning("Failed to read image index %s for stats", images_index, exc_info=True)

        # If user_stats.json provided generated total, keep max of the two to avoid regressions
        result["totals"]["images_generated"] = max(result["totals"]["images_generated"], gen_total)
        result["totals"]["images_analyzed"] = vis_total
        # Total images = generated (best available) + analyzed
        result["totals"]["images"] = int(result["totals"]["images_generated"]) + int(
            result["totals"]["images_analyzed"]
        )

        # Finalize series arrays
        msgs7 = [messages_by_day[d] for d in days_list]
        gens7 = [gen_by_day[d] for d in days_list]
        vis7 = [vis_by_day[d] for d in days_list]
        result["series_7d"]["messages"] = msgs7
        result["series_7d"]["images_generated"] = gens7
        result["series_7d"]["images_analyzed"] = vis7
        result["series_7d"]["other_messages"] = [
            max(0, int(msgs7[i]) - int(gens7[i]) - int(vis7[i])) for i in range(len(msgs7))
        ]

        # Rolling-window counts from JSONL (bot/archiver stream); kept when merging SQLite so
        # recent activity is not shown as zero if the scanned DB is stale or has no rows in-window.
        jsonl_last24 = int(result.get("last24_messages", 0))
        jsonl_last48 = int(result.get("last48_messages", 0))
        jsonl_last7d = int(result.get("last7d_messages", 0))
        jsonl_last30d = int(result.get("last30d_messages", 0))
        jsonl_hourly = list(result["hourly_24h"].get("messages") or [0] * 24)
        jsonl_hourly_gen = list(result["hourly_24h"].get("images_generated") or [0] * 24)
        jsonl_hourly_vis = list(result["hourly_24h"].get("images_analyzed") or [0] * 24)

        sqlite_archive = _merge_sqlite_message_stats_for_summary(
            base_dir=BASE_DIR,
            server_id=server_id,
            days_list=days_list,
            limit=limit,
        )
        used_sqlite_messages = sqlite_archive is not None
        sqlite_ch_names: dict[str, str] = {}
        if sqlite_archive:
            sqlite_ch_names = sqlite_archive["channel_id_to_name"]
            result["stats_source"] = {"message_counts": "sqlite_archive"}
            result["totals"]["messages"] = sqlite_archive["total_messages"]
            for d, c in sqlite_archive["messages_by_day"].items():
                messages_by_day[d] = c
            msgs7 = [messages_by_day[d] for d in days_list]
            result["series_7d"]["messages"] = msgs7
            result["series_7d"]["other_messages"] = [
                max(0, int(msgs7[i]) - int(gens7[i]) - int(vis7[i])) for i in range(len(msgs7))
            ]
            result["last24_messages"] = max(int(sqlite_archive["last24_messages"]), jsonl_last24)
            result["last48_messages"] = max(int(sqlite_archive["last48_messages"]), jsonl_last48)
            result["last7d_messages"] = max(int(sqlite_archive["last7d_messages"]), jsonl_last7d)
            result["last30d_messages"] = max(int(sqlite_archive["last30d_messages"]), jsonl_last30d)
            result["top"]["users_by_messages"] = sqlite_archive["users_by_messages"]
            result["top"]["channels_by_messages"] = sqlite_archive["channels_by_messages"]
            result["hourly_24h"]["labels"] = sqlite_archive["hour_labels"]
            sql_h = sqlite_archive["hourly_msg"]
            result["hourly_24h"]["messages"] = [
                max(int(sql_h[i] if i < len(sql_h) else 0), int(jsonl_hourly[i] if i < len(jsonl_hourly) else 0))
                for i in range(24)
            ]
            result["hourly_24h"]["images_generated"] = jsonl_hourly_gen
            result["hourly_24h"]["images_analyzed"] = jsonl_hourly_vis

        # Derive period counters from daily buckets (jsonl only; archive uses SQL cutoffs above)
        try:
            if not used_sqlite_messages:
                result["last7d_messages"] = sum(messages_by_day.get(d, 0) for d in days_list)
                if len(days_list) >= 2:
                    result["last48_messages"] = sum(messages_by_day.get(d, 0) for d in days_list[-2:])
                else:
                    result["last48_messages"] = result.get("last24_messages", 0)
                if result.get("last30d_messages", 0) < result.get("last7d_messages", 0):
                    result["last30d_messages"] = result["last7d_messages"]
        except Exception:
            logger.debug("Failed to derive period message counters", exc_info=True)

        # Tops
        if not used_sqlite_messages:
            result["top"]["users_by_messages"] = [
                {"username": u, "messages": c} for u, c in user_msg_counter.most_common(limit)
            ]
        if not result["top"]["users_by_images"]:
            result["top"]["users_by_images"] = [
                {"username": u, "images_generated": c} for u, c in user_images_index_counter.most_common(limit)
            ] or [{"username": u, "images_generated": c} for u, c in user_image_msg_counter.most_common(limit)]
        result["top"]["users_by_vision"] = [
            {"username": u, "vision": c} for u, c in user_vision_counter.most_common(limit)
        ]
        if not used_sqlite_messages:
            result["top"]["channels_by_messages"] = [
                {"channel_id": ch, "channel_name": channel_name_map.get(ch), "messages": c}
                for ch, c in channel_msg_counter.most_common(limit)
            ]
        result["top"]["channels_by_images"] = [
            {
                "channel_id": ch,
                "channel_name": sqlite_ch_names.get(str(ch)) or channel_name_map.get(str(ch)),
                "images_generated": c,
            }
            for ch, c in channel_images_index_counter.most_common(limit)
        ]

        return JSONResponse(result)

    @app.delete("/api/archive/images/{filename}")
    async def api_delete_image(filename: str):
        import re
        from tempfile import NamedTemporaryFile

        # Basic filename validation to prevent traversal
        if not re.fullmatch(r"[A-Za-z0-9._-]+", filename or ""):
            return JSONResponse({"ok": False, "message": "Invalid filename"}, status_code=400)
        img_path = media_dir / "images" / filename
        thumb_path = media_dir / "thumbs" / filename
        idx_path = media_dir / "images" / "index.jsonl"
        deleted = False
        try:
            if img_path.exists():
                img_path.unlink()
                deleted = True
        except Exception as exc:
            return JSONResponse({"ok": False, "message": f"Failed to delete image: {exc}"}, status_code=500)
        try:
            if thumb_path.exists():
                thumb_path.unlink()
        except Exception:
            logger.debug("Could not delete thumbnail %s", thumb_path, exc_info=True)
        # Rewrite index.jsonl excluding this filename
        try:
            if idx_path.exists():
                with (
                    open(idx_path, "r", encoding="utf-8") as src,
                    NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp,
                ):
                    for line in src:
                        try:
                            obj = json.loads(line)
                            if obj.get("filename") == filename:
                                continue
                        except Exception:
                            # Keep lines we can't parse
                            pass
                        tmp.write(line)
                # Replace atomically
                Path(tmp.name).replace(idx_path)
        except Exception:
            logger.warning("Failed to rewrite image index after deleting %s", filename, exc_info=True)
        return JSONResponse({"ok": True, "deleted": deleted})

    # Environment editor
    # -----------------------------------------------------------------------
    # Environment editor — reads/rewrites .env-stable via env_store
    # WHY: takes effect on the NEXT bot restart (env is merged into the
    # subprocess env at spawn time, not on every uvicorn reload).
    # -----------------------------------------------------------------------

    @app.get("/env", response_class=HTMLResponse)
    async def env_page(request: Request):
        env_path = BASE_DIR / ".env-stable"
        _, kv = parse_env(env_path)
        return templates.TemplateResponse(
            request,
            "env.html",
            {"vars": kv, "env_path": str(env_path)},
        )

    @app.get("/api/env/get")
    async def api_env_get():
        env_path = BASE_DIR / ".env-stable"
        _, kv = parse_env(env_path)
        return JSONResponse({"vars": kv, "env_path": str(env_path)})

    @app.post("/api/env/save")
    async def env_save(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "message": "Invalid payload"}, status_code=400)
        env_path = BASE_DIR / ".env-stable"
        try:
            # Convert all values to strings; drop empty/None values so they don't
            # clobber defaults in code (e.g. SOUPY_DB_DIR="" breaks database paths).
            updates = {}
            skipped = []
            for k, v in payload.items():
                sv = "" if v is None else str(v).strip()
                if sv:  # only write non-empty values
                    updates[str(k)] = sv
                else:
                    skipped.append(str(k))
            if updates:
                write_env(env_path, updates)
            # The editor tells the user which fields were written and which emptied ones were not.
            return JSONResponse(
                {"ok": True, "message": "Saved", "saved": sorted(updates), "skipped_empty": sorted(skipped)}
            )
        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    # -----------------------------------------------------------------------
    # Prompts — the bot reads them from prompts/<name>.txt (soupy/prompts.py).
    # web/services/prompt_store.py lists them and does the reading and writing.
    # -----------------------------------------------------------------------

    prompts_dir = BASE_DIR / "prompts"

    def _env_values() -> dict:
        return parse_env(BASE_DIR / ".env-stable")[1]

    @app.get("/api/prompts/get")
    async def api_prompts_get():
        return JSONResponse({"ok": True, "prompts": prompt_store.list_states(prompts_dir, _env_values())})

    async def _prompt_change(request: Request, change) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "message": "Invalid payload"}, status_code=400)
        name = str(payload.get("name", ""))
        env = _env_values()
        try:
            result = change(name, payload, env)
            state = prompt_store.read_state(name, prompts_dir, env)
        except prompt_store.PromptError as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=exc.status)
        except OSError as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)
        return JSONResponse({"ok": True, **result, "prompt": state})

    @app.post("/api/prompts/save")
    async def api_prompts_save(request: Request):
        def change(name, payload, env):
            if not isinstance(payload.get("value"), str):
                raise prompt_store.PromptError("Missing 'value'")
            return prompt_store.save(name, payload["value"], prompts_dir, env)

        return await _prompt_change(request, change)

    @app.post("/api/prompts/reset")
    async def api_prompts_reset(request: Request):
        return await _prompt_change(request, lambda name, _payload, env: prompt_store.reset(name, prompts_dir, env))

    # -----------------------------------------------------------------------
    # Models — what the model server has, and switching the chat model.
    # web/services/lm_studio.py plans a switch and carries it out.
    # -----------------------------------------------------------------------

    switch_lock = asyncio.Lock()

    async def _model_listing(settings: lm_studio.Settings) -> dict:
        try:
            return {
                **(await lm_studio.list_models(settings.base_url, settings.api_key)),
                "reachable": True,
                "error": None,
            }
        except lm_studio.LMStudioError as exc:
            return {"native": False, "models": [], "reachable": False, "error": str(exc)}

    @app.get("/api/lm-studio/models")
    async def api_lm_studio_models():
        """The server's models, which are loaded, and what Soupy uses each for."""
        settings = lm_studio.settings_from_env(_env_values())
        listing = await _model_listing(settings)
        return JSONResponse(
            {
                "ok": True,
                "reachable": listing["reachable"],
                "error": listing["error"],
                "native": listing["native"],
                "models": lm_studio.describe(listing, settings),
                "settings": {
                    "chat": settings.chat,
                    "vision": settings.vision or None,
                    "vision_enabled": settings.vision_enabled,
                    "embedding": settings.embedding or None,
                    "chat_budget": settings.chat_budget,
                    "saved_list": settings.saved_list,
                },
            }
        )

    @app.post("/api/lm-studio/switch-model")
    async def api_lm_studio_switch_model(request: Request):
        """Change the chat model, its context length, or the chat prompt budget.

        With ``dry_run`` the answer is only the plan: what would be unloaded,
        loaded and written to .env-stable. Without it the plan is carried out,
        with the bot stopped for the duration.
        """
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "message": "Invalid payload"}, status_code=400)
        env_path = BASE_DIR / ".env-stable"
        settings = lm_studio.settings_from_env(_env_values())
        listing = await _model_listing(settings)
        try:
            plan = lm_studio.build_plan(
                listing,
                settings,
                model=payload.get("model", ""),
                context_length=payload.get("context_length"),
                chat_budget=payload.get("chat_budget", settings.chat_budget),
                vision_follows=bool(payload.get("vision_follows")),
                bot_running=bool((await bot_runner.status())["running"]),
            )
        except lm_studio.PlanError as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        if payload.get("dry_run"):
            return JSONResponse({"ok": True, "plan": plan})
        if plan["noop"]:
            return JSONResponse({"ok": True, "message": "Nothing to change.", "plan": plan})
        if switch_lock.locked():
            return JSONResponse({"ok": False, "message": "A model switch is already running."}, status_code=409)
        async with switch_lock:
            logger.info("🔄 Model switch: %s (context %s)", plan["model"], plan["context_length"])
            result = await lm_studio.apply_plan(
                plan,
                settings,
                stop_bot=bot_runner.stop,
                start_bot=bot_runner.start,
                write_env=lambda updates: write_env(env_path, updates),
                log=lambda text: logger.info("🔄 %s", text),
            )
        return JSONResponse(
            {**result, "plan": plan, **(await bot_runner.status())}, status_code=200 if result["ok"] else 502
        )

    # -----------------------------------------------------------------------
    # Bot status — reads data/bot_dashboard.json (written by the bot every
    # ~15s) and exposes activity feeds for the dashboard.
    # -----------------------------------------------------------------------

    def _written_before(updated_at, runner_status) -> bool:
        """True when a status file stamped ``updated_at`` predates the bot that is running now."""
        started = runner_status.get("start_time")
        if not (runner_status.get("running") and started and updated_at):
            return False
        try:
            return datetime.fromisoformat(str(updated_at)) < datetime.fromisoformat(str(started))
        except (TypeError, ValueError):
            return False

    @app.get("/api/bot/dashboard-status")
    async def api_bot_dashboard_status():
        """Extended status for the dashboard Bot Operations panel.
        Reads from data/bot_dashboard.json (written by the bot process).
        """
        import json as _json

        data = {"ok": True}

        # Bot running status
        runner_status = await bot_runner.status()
        data["running"] = runner_status.get("running", False)
        data["pid"] = runner_status.get("pid")

        # Read shared status file written by bot process
        status_path = os.path.join("data", "bot_dashboard.json")
        try:
            if os.path.exists(status_path):
                raw = _json.loads(open(status_path).read())
                # A file older than the running bot was written by the bot before it:
                # its model, uptime and timers describe a process that is gone.
                if isinstance(raw, dict) and not _written_before(raw.get("updated_at"), runner_status):
                    data.update(raw)
        except Exception:
            logger.debug("Could not read bot_dashboard.json", exc_info=True)

        # Daily post history (file-based, readable from any process)
        try:
            hist_path = os.path.join("data", "daily_post_history.json")
            if os.path.exists(hist_path):
                hist = _json.loads(open(hist_path).read())
                from datetime import date

                today = date.today().isoformat()
                posted_today = {}
                for ch_id, entries in hist.items():
                    for entry in entries:
                        if entry.get("date") == today:
                            posted_today[ch_id] = entry.get("title", "")
                data["daily_post_today"] = posted_today
        except Exception:
            logger.debug("Could not read daily_post_history.json for dashboard", exc_info=True)

        # Soupy's memory, per guild (data/self_md/guild_<id>_self.json, written by the bot's profile worker)
        try:
            from soupy_database import self_context
            from soupy_database.self_profile import memory_summary

            memories = {}
            if self_context.SELF_MD_DIR.is_dir():
                for p in self_context.SELF_MD_DIR.glob("guild_*_self.json"):
                    gid = p.name[len("guild_") : -len("_self.json")]
                    summary = memory_summary(int(gid)) if gid.isdigit() else None
                    if summary:
                        memories[gid] = {"items": summary["items"], "updated_at": summary["updated_at"]}
            data["self_memory"] = memories
            data["self_md_enabled"] = bool(settings.self_md_enabled)
        except Exception:
            logger.debug("Could not read Soupy's memory for dashboard", exc_info=True)

        return JSONResponse(data)

    @app.get("/api/bot/activity")
    async def api_bot_activity():
        """Return recent Bluesky and Daily Post activity from persisted history files.

        Query params:
          ?type=all|replies|posts|reposts|dailyposts  (default: all)
          ?limit=50  (default: 50, max: 200)
        """
        import json as _json
        from datetime import date

        _req = app.state._request if hasattr(app.state, "_request") else None
        # Parse from starlette request

        # We'll just return everything and let the frontend filter
        today = date.today().isoformat()
        result = {
            "ok": True,
            "today": today,
            "bluesky": {
                "replies_today": 0,
                "posts_today": 0,
                "reposts_today": 0,
                "replies": [],
                "posts": [],
                "reposts": [],
            },
            "daily_posts": {"posts_today": 0, "posts": []},
        }

        # Bluesky history
        try:
            bsky_path = os.path.join("data", "bluesky_engage_history.json")
            if os.path.exists(bsky_path):
                bh = _json.loads(open(bsky_path).read())

                comments = bh.get("comments", [])
                result["bluesky"]["replies"] = comments[-50:]
                result["bluesky"]["replies_today"] = sum(1 for c in comments if c.get("ts", "").startswith(today))

                posts = bh.get("posts", [])
                result["bluesky"]["posts"] = posts[-50:]
                result["bluesky"]["posts_today"] = sum(1 for p in posts if p.get("ts", "").startswith(today))

                reposts = bh.get("reposts", [])
                result["bluesky"]["reposts"] = reposts[-50:]
                result["bluesky"]["reposts_today"] = sum(1 for r in reposts if r.get("ts", "").startswith(today))
        except Exception:
            logger.debug("Could not read bluesky_engage_history.json", exc_info=True)

        # Daily post history
        try:
            dp_path = os.path.join("data", "daily_post_history.json")
            if os.path.exists(dp_path):
                dh = _json.loads(open(dp_path).read())
                all_posts = []
                for ch_id, entries in dh.items():
                    for entry in entries:
                        entry["channel_id"] = ch_id
                        all_posts.append(entry)
                # Sort by date descending
                all_posts.sort(key=lambda x: x.get("date", ""), reverse=True)
                result["daily_posts"]["posts"] = all_posts[:50]
                result["daily_posts"]["posts_today"] = sum(1 for p in all_posts if p.get("date") == today)
        except Exception:
            logger.debug("Could not read daily_post_history.json for activity", exc_info=True)

        return JSONResponse(result)

    # -----------------------------------------------------------------------
    # Runtime flags — the bot↔web IPC channel for live toggles.
    # WHY: stored in data/runtime_flags.json; bot reads via mtime-cached
    # lookup so changes propagate without a bot restart.
    # -----------------------------------------------------------------------

    @app.get("/api/runtime-flags")
    async def api_runtime_flags_get():
        from soupy_database.runtime_flags import read_runtime_flags

        return JSONResponse(read_runtime_flags())

    @app.get("/api/channels/{guild_id}")
    async def api_channels_list(guild_id: str):
        """Return all known channels for a guild from the SQLite archive."""
        conn, _error = _open_guild_db(guild_id)
        if conn is None:
            return JSONResponse({"ok": False, "channels": []})
        try:
            return JSONResponse({"ok": True, "channels": browse.list_channels(conn)})
        finally:
            conn.close()

    @app.post("/api/runtime-flags")
    async def api_runtime_flags_post(request: Request):
        from soupy_database.runtime_flags import write_runtime_flags

        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "message": "Invalid payload"}, status_code=400)
        updates = {}
        if "rag_enabled" in payload:
            updates["rag_enabled"] = bool(payload["rag_enabled"])
        if "time_awareness" in payload:
            updates["time_awareness"] = bool(payload["time_awareness"])
        if "disabled_commands" in payload:
            val = payload["disabled_commands"]
            if isinstance(val, list):
                updates["disabled_commands"] = [str(c).lower().strip() for c in val if c]
        if not updates:
            return JSONResponse({"ok": False, "message": "No supported keys"}, status_code=400)
        merged = write_runtime_flags(updates)
        return JSONResponse({"ok": True, **merged})

    # -----------------------------------------------------------------------
    # RAG + user profile management — trigger reindex; queue/pause/cancel profile jobs (the bot runs them).
    # All operations are per-guild.
    # -----------------------------------------------------------------------

    @app.get("/api/rag/status/{guild_id}")
    async def api_rag_status(guild_id: str):
        from soupy_database import get_stats
        from soupy_database.rag import get_rag_chunk_count

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        stats = get_stats(gid)
        if not stats.get("exists"):
            return JSONResponse({"ok": False, "message": "Database not found"}, status_code=404)
        chunks = get_rag_chunk_count(gid)
        return JSONResponse(
            {
                "ok": True,
                "guild_id": str(gid),
                "rag_chunks": chunks,
                "total_messages": stats.get("total_messages", 0),
            }
        )

    @app.post("/api/rag/reindex/{guild_id}")
    async def api_rag_reindex(guild_id: str):
        from soupy_database.rag import reindex_guild_rag

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        try:
            result = await reindex_guild_rag(gid)
            return JSONResponse({"ok": True, **result})
        except Exception as exc:
            logging.exception("RAG reindex failed")
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.post("/api/self-memory/refresh/{guild_id}")
    async def api_self_memory_refresh(guild_id: str):
        """Ask the bot to refresh Soupy's memory for a guild; its profile worker runs it at the next chance."""
        from soupy_database.self_profile import request_self_refresh

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        request_self_refresh(gid)
        return JSONResponse({"ok": True, "queued": True})

    @app.get("/api/profiles/status/{guild_id}")
    async def api_profiles_status(guild_id: str):
        from soupy_database.user_profiles import get_user_profile_stats

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        stats = get_user_profile_stats(gid)
        if not stats.get("exists"):
            return JSONResponse(stats, status_code=404)
        return JSONResponse(stats)

    @app.post("/api/profiles/reset/{guild_id}")
    async def api_profiles_reset(guild_id: str):
        """Delete all stored profiles and batch job state for a guild (the bot's worker stops on its next check)."""
        from soupy_database.user_profiles import clear_stored_profiles

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        result = clear_stored_profiles(gid)
        if not result.get("ok"):
            return JSONResponse(result, status_code=404)
        return JSONResponse({"ok": True, "cleared": True})

    @app.post("/api/profiles/batch/start/{guild_id}")
    async def api_profiles_batch_start(guild_id: str):
        """Queue a profile batch; the bot process runs it (see soupy_database.user_profiles.profile_jobs_loop)."""
        from soupy_database.user_profiles import start_profile_batch_job

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        result = start_profile_batch_job(gid)
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)

    @app.post("/api/profiles/batch/pause/{guild_id}")
    async def api_profiles_batch_pause(guild_id: str):
        from soupy_database.user_profiles import pause_profile_batch_job

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        r = pause_profile_batch_job(gid)
        return JSONResponse(r, status_code=200 if r.get("ok") else 400)

    @app.post("/api/profiles/batch/resume/{guild_id}")
    async def api_profiles_batch_resume(guild_id: str):
        from soupy_database.user_profiles import resume_profile_batch_job

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        r = resume_profile_batch_job(gid)
        return JSONResponse(r, status_code=200 if r.get("ok") else 400)

    @app.post("/api/profiles/batch/cancel/{guild_id}")
    async def api_profiles_batch_cancel(guild_id: str):
        from soupy_database.user_profiles import cancel_profile_batch_job

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        r = await cancel_profile_batch_job(gid)
        return JSONResponse(r, status_code=200 if r.get("ok") else 400)

    @app.get("/api/profiles/batch/status/{guild_id}")
    async def api_profiles_batch_status(guild_id: str):
        from soupy_database.profile_batch import profile_job_log_lines
        from soupy_database.user_profiles import get_profile_batch_status

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        data = get_profile_batch_status(gid)
        if not data.get("ok"):
            return JSONResponse(data, status_code=404)
        data["log_lines"] = profile_job_log_lines(gid, 500)
        return JSONResponse(data)

    @app.get("/api/profiles/list/{guild_id}")
    async def api_profiles_list(guild_id: str, search: str | None = None):
        """Stored member profiles without their bodies, plus Soupy's own memory when it exists."""
        from soupy_database.self_profile import load_self_state

        conn, error = _open_guild_db(guild_id)
        if conn is None:
            return error
        try:
            rows = browse.list_profiles(conn, search=search or "")
            total = browse.count_profiles(conn)
        finally:
            conn.close()
        try:
            self_row = browse.self_memory_row(load_self_state(int(guild_id)), search=search or "")
        except Exception:
            logger.debug("Could not read Soupy's memory for guild %s", guild_id, exc_info=True)
            self_row = None
        return JSONResponse({"ok": True, "guild_id": guild_id, "total": total, "rows": rows, "self": self_row})

    @app.get("/api/profiles/get/{guild_id}/{user_id}")
    async def api_profiles_get(guild_id: str, user_id: str):
        """One profile laid out in sections. ``user_id`` "self" is Soupy's memory of itself."""
        from soupy_database.self_profile import load_self_state

        conn, error = _open_guild_db(guild_id)
        if conn is None:
            return error
        try:
            if user_id == "self":
                detail = browse.self_memory_detail(load_self_state(int(guild_id)))
            else:
                uid = browse.parse_snowflake(user_id)
                if uid is None:
                    return JSONResponse({"ok": False, "message": "Invalid user ID"}, status_code=400)
                detail = browse.profile_detail(conn, uid)
        finally:
            conn.close()
        if detail is None:
            return JSONResponse({"ok": False, "message": "No profile stored for this member"}, status_code=404)
        return JSONResponse({"ok": True, "guild_id": guild_id, "profile": detail})

    @app.get("/api/database/members/{guild_id}")
    async def api_database_members(guild_id: str, limit: int = 400):
        """Posters by message count, for the member filter."""
        conn, error = _open_guild_db(guild_id)
        if conn is None:
            return error
        try:
            return JSONResponse({"ok": True, "guild_id": guild_id, "members": browse.list_members(conn, limit)})
        finally:
            conn.close()

    # Bot control APIs
    # -----------------------------------------------------------------------
    # Bot lifecycle — start, stop, restart the bot subprocess.
    # WHY: restart is the only way to pick up .env-stable edits or new code.
    # The bot's stdout/stderr stream through a PTY into the WebSocket fanout.
    # -----------------------------------------------------------------------

    @app.get("/api/bot/status")
    async def api_status():
        return JSONResponse(await bot_runner.status())

    @app.post("/api/bot/start")
    async def api_start():
        ok, message = await bot_runner.start()
        return JSONResponse({"ok": ok, "message": message, **(await bot_runner.status())})

    @app.post("/api/bot/stop")
    async def api_stop():
        ok, message = await bot_runner.stop()
        return JSONResponse({"ok": ok, "message": message, **(await bot_runner.status())})

    @app.post("/api/bot/restart")
    async def api_restart():
        await bot_runner.stop()
        ok, message = await bot_runner.start()
        return JSONResponse({"ok": ok, "message": message, **(await bot_runner.status())})

    # -----------------------------------------------------------------------
    # Flux server lifecycle — start, stop, restart flux_server.py, the local
    # image backend. A server started by hand is reported and left alone.
    # WHY: restart is how a FLUX_* edit or a change to flux_server.py takes
    # effect. Restarting the bot does not touch this process.
    # -----------------------------------------------------------------------

    @app.get("/api/flux/status")
    async def api_flux_status():
        return JSONResponse(await flux_runner.status())

    @app.get("/api/flux/log")
    async def api_flux_log(lines: int = 100):
        return JSONResponse({"ok": True, "lines": flux_runner.tail(lines)})

    @app.post("/api/flux/start")
    async def api_flux_start():
        ok, message = await flux_runner.start()
        return JSONResponse({"ok": ok, "message": message, **(await flux_runner.status())})

    @app.post("/api/flux/stop")
    async def api_flux_stop():
        ok, message = await flux_runner.stop()
        return JSONResponse({"ok": ok, "message": message, **(await flux_runner.status())})

    @app.post("/api/flux/restart")
    async def api_flux_restart():
        ok, message = await flux_runner.restart()
        return JSONResponse({"ok": ok, "message": message, **(await flux_runner.status())})

    # Database scan APIs
    # -----------------------------------------------------------------------
    # Database management — per-guild stats, scan triggers, schedule,
    # archive table browsing for the Database tab in the dashboard.
    # -----------------------------------------------------------------------

    @app.get("/api/database/status")
    async def api_database_status():
        """Get active scans and overall database status."""
        try:
            import os
            from pathlib import Path

            from soupy_database import get_stats

            # Get active scans (try to import, but handle if bot isn't running)
            active = {}
            try:
                from soupy_database import get_active_scans, get_stats

                active_scans = get_active_scans()
                for guild_id, task in active_scans.items():
                    if not task.done():
                        # Get guild name from database if available
                        stats = get_stats(guild_id)
                        guild_name = stats.get("guild_name") if stats.get("exists") else None
                        active[str(guild_id)] = {
                            "guild_id": str(guild_id),
                            "guild_name": guild_name,
                            "running": True,
                            "done": task.done(),
                            "cancelled": task.cancelled(),
                        }
            except (ImportError, AttributeError):
                # Bot module not available or active_scans not defined
                pass

            # Get all databases
            db_dir = settings.soupy_db_dir or os.path.join(BASE_DIR, "soupy_database", "databases")
            db_path = Path(db_dir)
            databases = []

            if db_path.exists():
                for db_file in db_path.glob("guild_*.db"):
                    try:
                        # Extract guild ID from filename
                        guild_id_str = db_file.stem.replace("guild_", "")
                        guild_id = int(guild_id_str)
                        stats = get_stats(guild_id)
                        if stats["exists"]:
                            databases.append(
                                {
                                    "guild_id": str(guild_id),
                                    "guild_name": stats.get("guild_name"),
                                    "total_messages": stats["total_messages"],
                                    "total_channels": stats["total_channels"],
                                    "last_scan": stats["last_scan"],
                                    "archive_scan_interval_minutes": stats.get("archive_scan_interval_minutes", 0),
                                    "file_size": db_file.stat().st_size if db_file.exists() else 0,
                                }
                            )
                    except (ValueError, Exception):
                        continue

            # Biggest archive first: the dashboard opens on the first one.
            databases.sort(key=lambda d: -int(d["total_messages"] or 0))

            return JSONResponse(
                {"ok": True, "active_scans": active, "databases": databases, "total_databases": len(databases)}
            )
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.get("/api/database/stats/{guild_id}")
    async def api_database_stats(guild_id: str):
        """Get detailed stats for a specific server database."""
        try:
            import os
            import sqlite3
            from collections import defaultdict
            from datetime import datetime

            from soupy_database import get_stats

            # Get display timezone (from app context or env)
            display_tz_str = settings.timezone
            try:
                display_tz = ZoneInfo(display_tz_str)
            except Exception:
                display_tz = ZoneInfo("UTC")
                display_tz_str = "UTC"

            guild_id_int = int(guild_id)
            stats = get_stats(guild_id_int)

            if not stats["exists"]:
                return JSONResponse({"ok": False, "message": "Database not found"}, status_code=404)

            # Get additional stats from database
            from soupy_database.database import get_db_path

            db_path = get_db_path(guild_id_int)

            detailed_stats = {
                "guild_id": guild_id,
                "total_messages": stats["total_messages"],
                "total_channels": stats["total_channels"],
                "last_scan": stats["last_scan"],
            }

            # Get file size
            if os.path.exists(db_path):
                detailed_stats["file_size"] = os.path.getsize(db_path)
            else:
                detailed_stats["file_size"] = 0

            # Get scan history
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT scan_type, last_scan_time, messages_scanned, created_at
                    FROM scan_metadata
                    ORDER BY last_scan_time DESC
                    LIMIT 10
                """)

                scan_history = []
                for row in cursor.fetchall():
                    scan_history.append(
                        {
                            "scan_type": row["scan_type"],
                            "last_scan_time": row["last_scan_time"],
                            "messages_scanned": row["messages_scanned"],
                            "created_at": row["created_at"],
                        }
                    )

                detailed_stats["scan_history"] = scan_history

                # Get total scan count
                cursor.execute("SELECT COUNT(*) as count FROM scan_metadata")
                detailed_stats["total_scans"] = cursor.fetchone()["count"]

                # Get image/URL stats
                cursor.execute("SELECT COUNT(*) as count FROM messages WHERE image_description IS NOT NULL")
                detailed_stats["messages_with_images"] = cursor.fetchone()["count"]

                cursor.execute("SELECT COUNT(*) as count FROM messages WHERE url_summary IS NOT NULL")
                detailed_stats["messages_with_urls"] = cursor.fetchone()["count"]

                cursor.execute(
                    "SELECT COUNT(*) as count FROM messages WHERE image_description IS NOT NULL AND url_summary IS NOT NULL"
                )
                detailed_stats["messages_with_both"] = cursor.fetchone()["count"]

                # Get unique users count
                cursor.execute("SELECT COUNT(DISTINCT user_id) as count FROM messages")
                detailed_stats["unique_users"] = cursor.fetchone()["count"]

                # Get channel stats (all channels, not just top 20)
                cursor.execute("""
                    SELECT channel_id, channel_name, COUNT(*) as message_count
                    FROM messages
                    GROUP BY channel_id, channel_name
                    ORDER BY message_count DESC
                """)

                channel_stats = []
                for row in cursor.fetchall():
                    channel_stats.append(
                        {
                            "channel_id": str(row["channel_id"]),
                            "channel_name": row["channel_name"],
                            "message_count": row["message_count"],
                        }
                    )

                detailed_stats["channels"] = channel_stats
                detailed_stats["top_channels"] = channel_stats[:20]  # Keep for backward compatibility

                # Get user stats (all users, not just top 20)
                cursor.execute("""
                    SELECT user_id, username, COUNT(*) as message_count
                    FROM messages
                    GROUP BY user_id, username
                    ORDER BY message_count DESC
                """)

                user_stats = []
                for row in cursor.fetchall():
                    user_stats.append(
                        {
                            "user_id": str(row["user_id"]),
                            "username": row["username"],
                            "message_count": row["message_count"],
                        }
                    )

                detailed_stats["users"] = user_stats
                detailed_stats["top_users"] = user_stats[:20]  # Keep for backward compatibility

                # Get date range
                cursor.execute("""
                    SELECT MIN(date || ' ' || time) as earliest, MAX(date || ' ' || time) as latest
                    FROM messages
                """)

                date_range = cursor.fetchone()
                if date_range and date_range["earliest"]:
                    # Parse dates (stored as UTC in database)
                    earliest_str = date_range["earliest"]
                    latest_str = date_range["latest"]
                    earliest_utc = datetime.strptime(earliest_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    latest_utc = datetime.strptime(latest_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

                    # Convert to display timezone
                    earliest_local = earliest_utc.astimezone(display_tz)
                    latest_local = latest_utc.astimezone(display_tz)

                    days_span = (latest_utc - earliest_utc).days
                    detailed_stats["date_range"] = {
                        "earliest": earliest_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "latest": latest_local.strftime("%Y-%m-%d %H:%M:%S"),
                        "earliest_utc": earliest_str,
                        "latest_utc": latest_str,
                        "timezone": display_tz_str,
                        "days_span": days_span,
                    }
                    detailed_stats["avg_messages_per_day"] = stats["total_messages"] / days_span if days_span > 0 else 0

                # Get messages over time (for charts) - daily aggregation
                # Convert dates from UTC to display timezone
                cursor.execute("""
                    SELECT date, COUNT(*) as message_count
                    FROM messages
                    GROUP BY date
                    ORDER BY date ASC
                """)

                daily_messages = []
                for row in cursor.fetchall():
                    # Date is stored as UTC date, but we want to show it in local timezone
                    # For display purposes, we'll keep the UTC date but note the timezone
                    daily_messages.append({"date": row["date"], "count": row["message_count"]})

                detailed_stats["daily_messages"] = daily_messages
                detailed_stats["daily_messages_timezone"] = display_tz_str

                # Get messages by hour of day (0-23) - convert from UTC to display timezone
                cursor.execute("""
                    SELECT date, time, COUNT(*) as message_count
                    FROM messages
                    WHERE time IS NOT NULL AND LENGTH(time) >= 2
                    GROUP BY date, time
                """)

                hourly_messages = [0] * 24
                for row in cursor.fetchall():
                    try:
                        # Parse UTC datetime from database
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        hour = dt_local.hour
                        if 0 <= hour < 24:
                            hourly_messages[hour] += row["message_count"]
                    except Exception:
                        pass

                detailed_stats["hourly_messages"] = hourly_messages
                detailed_stats["hourly_messages_timezone"] = display_tz_str

                # Get messages by day of week (0=Monday, 6=Sunday) - convert from UTC to display timezone
                cursor.execute("""
                    SELECT date, time, COUNT(*) as message_count
                    FROM messages
                    WHERE time IS NOT NULL AND LENGTH(time) >= 2
                    GROUP BY date, time
                """)

                weekday_messages = [0] * 7
                for row in cursor.fetchall():
                    try:
                        # Parse UTC datetime from database
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        weekday = dt_local.weekday()  # 0=Monday, 6=Sunday
                        weekday_messages[weekday] += row["message_count"]
                    except Exception:
                        pass

                detailed_stats["weekday_messages"] = weekday_messages
                detailed_stats["weekday_messages_timezone"] = display_tz_str

                # Get user activity patterns (hourly activity per user) - convert from UTC to display timezone
                cursor.execute("""
                    SELECT
                        user_id,
                        username,
                        date,
                        time,
                        COUNT(*) as message_count
                    FROM messages
                    WHERE time IS NOT NULL AND LENGTH(time) >= 2
                    GROUP BY user_id, username, date, time
                """)

                user_hourly_activity = defaultdict(lambda: defaultdict(int))
                for row in cursor.fetchall():
                    try:
                        user_id = str(row["user_id"])
                        # Parse UTC datetime from database
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        hour = dt_local.hour
                        if 0 <= hour < 24:
                            user_hourly_activity[user_id][hour] += row["message_count"]
                    except Exception:
                        pass

                # Convert to list format for JSON
                user_activity_list = []
                for user_id, hourly_data in user_hourly_activity.items():
                    hours = [0] * 24
                    for hour, count in hourly_data.items():
                        hours[hour] = count
                    # Get username from first occurrence
                    cursor.execute("SELECT username FROM messages WHERE user_id = ? LIMIT 1", (int(user_id),))
                    username_row = cursor.fetchone()
                    username = username_row["username"] if username_row else f"User {user_id}"
                    user_activity_list.append(
                        {
                            "user_id": user_id,
                            "username": username,
                            "hourly_activity": hours,
                            "total_messages": sum(hours),
                        }
                    )

                # Sort by total messages descending
                user_activity_list.sort(key=lambda x: x["total_messages"], reverse=True)
                detailed_stats["user_hourly_activity"] = user_activity_list[:20]  # Top 20 users
                detailed_stats["user_hourly_activity_timezone"] = display_tz_str

                # Get channel activity by hour - convert from UTC to display timezone
                cursor.execute("""
                    SELECT
                        channel_id,
                        channel_name,
                        date,
                        time,
                        COUNT(*) as message_count
                    FROM messages
                    WHERE time IS NOT NULL AND LENGTH(time) >= 2
                    GROUP BY channel_id, channel_name, date, time
                """)

                channel_hourly_activity = defaultdict(lambda: {"name": "", "hours": [0] * 24})
                for row in cursor.fetchall():
                    try:
                        channel_id = str(row["channel_id"])
                        channel_hourly_activity[channel_id]["name"] = row["channel_name"]
                        # Parse UTC datetime from database
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        hour = dt_local.hour
                        if 0 <= hour < 24:
                            channel_hourly_activity[channel_id]["hours"][hour] += row["message_count"]
                    except Exception:
                        pass

                # Convert to list format
                channel_activity_list = []
                for channel_id, data in channel_hourly_activity.items():
                    total = sum(data["hours"])
                    channel_activity_list.append(
                        {
                            "channel_id": channel_id,
                            "channel_name": data["name"],
                            "hourly_activity": data["hours"],
                            "total_messages": total,
                            "peak_hour": data["hours"].index(max(data["hours"])) if max(data["hours"]) > 0 else None,
                        }
                    )

                # Sort by total messages descending
                channel_activity_list.sort(key=lambda x: x["total_messages"], reverse=True)
                detailed_stats["channel_hourly_activity"] = channel_activity_list
                detailed_stats["channel_hourly_activity_timezone"] = display_tz_str

                # Get user activity by day of week - convert from UTC to display timezone
                cursor.execute("""
                    SELECT
                        user_id,
                        username,
                        date,
                        time,
                        COUNT(*) as message_count
                    FROM messages
                    WHERE time IS NOT NULL AND LENGTH(time) >= 2
                    GROUP BY user_id, username, date, time
                """)

                user_weekday_activity = defaultdict(lambda: {"username": "", "weekdays": [0] * 7})
                for row in cursor.fetchall():
                    user_id = str(row["user_id"])
                    user_weekday_activity[user_id]["username"] = row["username"]
                    try:
                        # Parse UTC datetime from database
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        weekday = dt_local.weekday()  # 0=Monday, 6=Sunday
                        user_weekday_activity[user_id]["weekdays"][weekday] += row["message_count"]
                    except Exception:
                        pass

                # Convert to list
                user_weekday_list = []
                for user_id, data in user_weekday_activity.items():
                    total = sum(data["weekdays"])
                    user_weekday_list.append(
                        {
                            "user_id": user_id,
                            "username": data["username"],
                            "weekday_activity": data["weekdays"],
                            "total_messages": total,
                        }
                    )

                user_weekday_list.sort(key=lambda x: x["total_messages"], reverse=True)
                detailed_stats["user_weekday_activity"] = user_weekday_list[:20]  # Top 20 users

                # Get most active channels by time period (morning, afternoon, evening, night)
                # Convert from UTC to display timezone
                cursor.execute("""
                    SELECT
                        channel_id,
                        channel_name,
                        date,
                        time,
                        COUNT(*) as message_count
                    FROM messages
                    WHERE time IS NOT NULL AND LENGTH(time) >= 2
                    GROUP BY channel_id, channel_name, date, time
                """)

                channel_time_periods = defaultdict(
                    lambda: {"name": "", "morning": 0, "afternoon": 0, "evening": 0, "night": 0}
                )
                for row in cursor.fetchall():
                    try:
                        channel_id = str(row["channel_id"])
                        channel_time_periods[channel_id]["name"] = row["channel_name"]
                        # Parse UTC datetime from database
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        hour = dt_local.hour
                        # Determine time period based on local time
                        if 6 <= hour <= 11:
                            period = "morning"
                        elif 12 <= hour <= 17:
                            period = "afternoon"
                        elif 18 <= hour <= 22:
                            period = "evening"
                        else:
                            period = "night"
                        channel_time_periods[channel_id][period] += row["message_count"]
                    except Exception:
                        pass

                channel_period_list = []
                for channel_id, data in channel_time_periods.items():
                    total = data["morning"] + data["afternoon"] + data["evening"] + data["night"]
                    channel_period_list.append(
                        {
                            "channel_id": channel_id,
                            "channel_name": data["name"],
                            "morning": data["morning"],
                            "afternoon": data["afternoon"],
                            "evening": data["evening"],
                            "night": data["night"],
                            "total_messages": total,
                            "most_active_period": max(
                                ["morning", "afternoon", "evening", "night"], key=lambda p: data[p]
                            ),
                        }
                    )

                channel_period_list.sort(key=lambda x: x["total_messages"], reverse=True)
                detailed_stats["channel_time_periods"] = channel_period_list
                detailed_stats["channel_time_periods_timezone"] = display_tz_str

                # Add display timezone to all stats for reference
                detailed_stats["display_timezone"] = display_tz_str

                # Get guild name from basic stats
                basic_stats = get_stats(int(guild_id))
                if basic_stats.get("guild_name"):
                    detailed_stats["guild_name"] = basic_stats["guild_name"]

                conn.close()
            except Exception as e:
                detailed_stats["error"] = str(e)
                import traceback

                detailed_stats["error_traceback"] = traceback.format_exc()
                # Ensure analytics fields exist even on error
                if "user_hourly_activity" not in detailed_stats:
                    detailed_stats["user_hourly_activity"] = []
                if "channel_hourly_activity" not in detailed_stats:
                    detailed_stats["channel_hourly_activity"] = []
                if "channel_time_periods" not in detailed_stats:
                    detailed_stats["channel_time_periods"] = []

            # Add guild name to stats if available
            basic_stats = get_stats(int(guild_id))
            if basic_stats.get("guild_name"):
                detailed_stats["guild_name"] = basic_stats["guild_name"]

            return JSONResponse({"ok": True, "stats": detailed_stats})
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.get("/api/database/archive-schedule/{guild_id}")
    async def api_database_archive_schedule_get(guild_id: str):
        """Return auto-archive interval (minutes) for a guild; 0 = off."""
        try:
            from soupy_database import get_archive_scan_interval_minutes, get_last_scan_time

            guild_id_int = int(guild_id)
            minutes = get_archive_scan_interval_minutes(guild_id_int)
            last = get_last_scan_time(guild_id_int)
            return JSONResponse(
                {
                    "ok": True,
                    "archive_scan_interval_minutes": minutes,
                    "last_scan": last.isoformat() if last else None,
                }
            )
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.post("/api/database/archive-schedule/{guild_id}")
    async def api_database_archive_schedule_post(guild_id: str, request: Request):
        """Set auto-archive interval in minutes (0 disables). Creates guild DB if needed."""
        try:
            from soupy_database import set_archive_scan_interval_minutes

            guild_id_int = int(guild_id)
            body = await request.json()
            if not isinstance(body, dict) or "minutes" not in body:
                return JSONResponse(
                    {"ok": False, "message": "JSON body must include minutes (integer)"},
                    status_code=400,
                )
            minutes = int(body["minutes"])
            result = set_archive_scan_interval_minutes(guild_id_int, minutes)
            return JSONResponse({"ok": True, **result})
        except ValueError as ve:
            return JSONResponse({"ok": False, "message": str(ve)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.post("/api/database/scan/{guild_id}")
    async def api_database_scan(guild_id: str):
        """Trigger an incremental database scan for a specific server."""
        try:
            from soupy_database import create_scan_trigger

            guild_id_int = int(guild_id)

            # Check if bot is running
            bot_status = await bot_runner.status()
            if not bot_status.get("running"):
                return JSONResponse({"ok": False, "message": "Bot is not running"}, status_code=400)

            # Create a file-based trigger (works even if bot is in separate process)
            success, message = create_scan_trigger(guild_id_int)

            if success:
                return JSONResponse({"ok": True, "message": message})
            else:
                return JSONResponse({"ok": False, "message": message}, status_code=400)

        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    @app.get("/api/database/explore/{guild_id}")
    async def api_database_explore(
        guild_id: str,
        table: str = "messages",
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
        channel_id: str | None = None,
        user_id: str | None = None,
        has_images: bool = False,
        has_urls: bool = False,
        after: str | None = None,
        before: str | None = None,
        order: str = "desc",
    ):
        """Browse archived messages (or the scan history) with a fixed set of filters."""
        table_name = (table or "messages").strip().lower()
        if table_name not in {"messages", "scan_metadata"}:
            return JSONResponse({"ok": False, "message": "Unsupported table"}, status_code=400)
        filters = {"channel_id": None, "user_id": None}
        for key, raw in (("channel_id", channel_id), ("user_id", user_id)):
            if raw and raw.strip():
                filters[key] = browse.parse_snowflake(raw)
                if filters[key] is None:
                    return JSONResponse({"ok": False, "message": f"Invalid {key}"}, status_code=400)

        conn, error = _open_guild_db(guild_id)
        if conn is None:
            return error
        try:
            if table_name == "scan_metadata":
                page = browse.recent_scans(conn, limit)
            else:
                page = browse.query_messages(
                    conn,
                    search=search or "",
                    has_images=has_images,
                    has_urls=has_urls,
                    after=(after or "").strip(),
                    before=(before or "").strip(),
                    newest_first=(order or "").strip().lower() != "asc",
                    limit=limit,
                    offset=offset,
                    **filters,
                )
            return JSONResponse({"ok": True, "table": table_name, **page})
        except Exception as exc:
            logger.warning("Database explore failed for guild %s", guild_id, exc_info=True)
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)
        finally:
            conn.close()

    @app.get("/api/database/context/{guild_id}/{message_id}")
    async def api_database_context(guild_id: str, message_id: str, before: int = 8, after: int = 8):
        """A message with its neighbours in the same channel, oldest first."""
        mid = browse.parse_snowflake(message_id)
        if mid is None:
            return JSONResponse({"ok": False, "message": "Invalid message ID"}, status_code=400)
        conn, error = _open_guild_db(guild_id)
        if conn is None:
            return error
        try:
            context = browse.message_context(conn, mid, before=before, after=after)
        finally:
            conn.close()
        if context is None:
            return JSONResponse({"ok": False, "message": "Message not found"}, status_code=404)
        return JSONResponse({"ok": True, **context})

    # Logs websocket
    @app.websocket("/ws/logs")
    async def logs_ws(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive. We don't expect messages from client.
                await websocket.receive_text()
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception:
            await ws_manager.disconnect(websocket)

    @app.on_event("startup")
    async def _startup():
        # Optional autostart if env flag is set
        if settings.autostart_bot:
            await bot_runner.start()
        # FLUX_SERVER_AUTOSTART: bring the image backend up with the panel.
        # start() declines when a hand-started server already answers.
        if flux_runner.autostart_wanted():
            await flux_runner.start()

    @app.on_event("shutdown")
    async def _shutdown():
        # Ensure bot is stopped when app shuts down
        await bot_runner.stop()
        # Only the panel's own Flux server; stop() leaves a hand-started one alone.
        await flux_runner.stop()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=4941,
        reload=True,
        log_level="info",
    )
