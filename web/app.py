from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import re
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Colored logging (matches bot's CustomFormatter style)
# ---------------------------------------------------------------------------

class _WebFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[95m", "INFO": "\033[92m",
        "WARNING": "\033[93m", "ERROR": "\033[91m", "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"
    TS = "\033[36m"      # cyan timestamp
    NAME = "\033[94m"    # blue logger name
    ARROW = "\033[90m"   # grey arrow

    def format(self, record):
        ts = self.formatTime(record, self.datefmt)
        lc = self.COLORS.get(record.levelname, "")
        return (
            f"{self.TS}[{ts}]{self.RESET} "
            f"{lc}({record.levelname}){self.RESET} "
            f"{self.NAME}{record.name}{self.RESET} "
            f"{self.ARROW}=>{self.RESET} "
            f"{record.getMessage()}"
            + (f"\n{self.formatException(record.exc_info)}" if record.exc_info else "")
        )

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        msec = int((record.created - int(record.created)) * 1000)
        s = _time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return f"{s},{msec:03d}"


_web_fmt = _WebFormatter()

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
        _log_dir / "soupy.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from dotenv import load_dotenv

from .services.bot_runner import BotRunner
from .services.log_stream import WebsocketManager
from .services.env_store import parse_env, write_env


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
    db_dir = Path(os.environ.get("SOUPY_DB_DIR", str(base_dir / "soupy_database" / "databases")))
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
                    dt = datetime.strptime(
                        f'{r["date"]} {r["time"]}', "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=timezone.utc)
                    te = dt.timestamp()
                except Exception:
                    continue
                if te < now_epoch - 24 * 3600:
                    continue
                slot = int((now_epoch - te) // 3600)
                if 0 <= slot < 24:
                    hourly_msg[23 - slot] += 1

            cur.execute(
                "SELECT user_id, username, COUNT(*) AS n FROM messages GROUP BY user_id, username"
            )
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
            continue
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    if total_messages <= 0:
        return None

    users_ranked = sorted(
        ((v[1], v[0]) for v in user_acc.values()),
        key=lambda t: (-t[1], t[0].lower()),
    )[:limit]

    channels_ranked = sorted(ch_acc.items(), key=lambda item: -item[1]["count"])[:limit]
    channel_id_to_name = {str(cid): data["name"] for cid, data in ch_acc.items()}

    hour_labels = [
        (now_naive_utc - timedelta(hours=23 - i)).strftime("%m-%d %H:00") for i in range(24)
    ]

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
    control_panel_title = os.getenv("WEB_CONTROL_PANEL_TITLE", "Soupy Control")
    
    # Get timezone from environment (default to America/Los_Angeles for California)
    display_timezone_str = os.getenv("TIMEZONE", "America/Los_Angeles")
    try:
        display_timezone = ZoneInfo(display_timezone_str)
    except Exception:
        # Fallback to UTC if timezone is invalid
        logging.warning(f"Invalid timezone '{display_timezone_str}', falling back to UTC")
        display_timezone = ZoneInfo("UTC")
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
        pass
    app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")

    # Shared state
    ws_manager = WebsocketManager()

    async def fanout_output(message: str) -> None:
        # Mirror to server console and broadcast to connected web clients
        try:
            print(message, flush=True)
        except Exception:
            pass
        # Strip ANSI color codes for the web UI
        ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        clean = ansi_re.sub("", message)
        await ws_manager.broadcast_text(clean)

    bot_runner = BotRunner(project_root=BASE_DIR, on_output=fanout_output)

    app.state.templates = templates
    app.state.ws_manager = ws_manager
    app.state.bot_runner = bot_runner

    _configure_soupy_database_console_logging()

    # Reduce log noise: thumb 304s, noisy dashboard polling endpoints
    try:
        class SuppressNoisyAccess(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    msg = record.getMessage()
                except Exception:
                    return True
                if "/media/thumbs/" in msg and " 304 " in msg:
                    return False
                if "GET /api/profiles/batch/status/" in msg:
                    return False
                return True

        logging.getLogger("uvicorn.access").addFilter(SuppressNoisyAccess())
    except Exception:
        pass

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
            "dashboard.html",
            {
                "request": request,
                "status": status,
                "now": datetime.now(timezone.utc),
            },
        )

    @app.get("/api/archive/images")
    async def api_archive_images(limit: int = 50, offset: int = 0, kind: str | None = None):
        import json
        from pathlib import Path
        idx = media_dir / "images" / "index.jsonl"
        all_items = []
        if idx.exists():
            try:
                with open(idx, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            all_items.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                pass
        # Newest last → reverse for newest first
        all_items.reverse()
        k = (kind or "").strip().lower()
        if k == "vision":
            all_items = [x for x in all_items if x.get("event_type") == "vision"]
        elif k == "generated":
            all_items = [x for x in all_items if x.get("event_type") != "vision"]
        total = len(all_items)
        items = all_items[offset : offset + limit]
        return JSONResponse({"items": items, "total": total, "kind": k or "all"})

    @app.get("/api/archive/messages")
    async def api_archive_messages(limit: int = 50, offset: int = 0):
        import json
        log = media_dir / "messages.jsonl"
        all_items = []
        if log.exists():
            try:
                with open(log, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            all_items.append(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                pass
        all_items.reverse()
        total = len(all_items)
        items = all_items[offset:offset + limit]
        return JSONResponse({"items": items, "total": total})

    @app.get("/api/archive/message_by_image")
    async def api_message_by_image(filename: str):
        """Return the most recent message entry that references the given image filename.

        This is primarily used to retrieve the full, non-truncated description for
        images that were analyzed (vision events) when the paginated preload misses it.
        """
        import json
        import re
        # Basic filename validation to prevent traversal or odd inputs
        if not re.fullmatch(r"[A-Za-z0-9._-]+", filename or ""):
            return JSONResponse({"ok": False, "message": "Invalid filename"}, status_code=400)

        log = media_dir / "messages.jsonl"
        if not log.exists():
            return JSONResponse({"ok": False, "message": "messages log not found"}, status_code=404)

        try:
            lines = log.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

        # Search newest-first
        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("image_filename") == filename:
                # Return minimal useful fields
                return JSONResponse({
                    "ok": True,
                    "content": obj.get("content"),
                    "event_type": obj.get("event_type"),
                    "ts": obj.get("ts"),
                    "username": obj.get("username"),
                })

        return JSONResponse({"ok": False, "message": "No matching message found"}, status_code=404)

    @app.get("/api/stats/raw")
    async def api_stats_raw():
        import json
        stats_path = BASE_DIR / "user_stats.json"
        if not stats_path.exists():
            return JSONResponse({}, status_code=200)
        try:
            return JSONResponse(json.loads(stats_path.read_text(encoding="utf-8")))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/stats/summary")
    async def api_stats_summary(server_id: str | None = None, limit: int = 5):
        import json
        import datetime
        from collections import Counter, defaultdict

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
                    scope = (servers.get(server_id) if server_id else servers.get("global") or {})
                    rows.append({
                        "username": u.get("username", "Unknown"),
                        "images_generated": int(scope.get("images_generated", 0)),
                    })
                    chat_rows.append({
                        "username": u.get("username", "Unknown"),
                        "chat_responses": int(scope.get("chat_responses", 0)),
                    })

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
                pass

        # Optional channel name mapping
        channel_name_map: dict[str, str] = {}
        try:
            # File-based mapping
            ch_file = media_dir / "channel_names.json"
            if ch_file.exists():
                channel_name_map.update(json.loads(ch_file.read_text(encoding="utf-8")))
        except Exception:
            pass
        try:
            # Env pair list: CHANNEL_NAMES = "123:general,456:random"
            raw_pairs = os.environ.get("CHANNEL_NAMES")
            if raw_pairs:
                for pair in raw_pairs.split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    if ":" in pair:
                        cid, name = pair.split(":", 1)
                        channel_name_map[str(cid.strip())] = name.strip()
        except Exception:
            pass
        try:
            # Env JSON mapping: CHANNEL_NAMES_JSON = '{"123":"general"}'
            raw_json = os.environ.get("CHANNEL_NAMES_JSON")
            if raw_json:
                channel_name_map.update(json.loads(raw_json))
        except Exception:
            pass

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

                hour_labels = [
                    (now_dt - datetime.timedelta(hours=23 - i)).strftime("%m-%d %H:00") for i in range(24)
                ]
                result["hourly_24h"]["labels"] = hour_labels
                result["hourly_24h"]["messages"] = hourly_msg
                result["hourly_24h"]["images_generated"] = hourly_gen
                result["hourly_24h"]["images_analyzed"] = hourly_vis
                result["event_totals"] = dict(sorted(event_totals.items(), key=lambda kv: (-kv[1], kv[0])))
            except Exception:
                pass
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
                pass

        # If user_stats.json provided generated total, keep max of the two to avoid regressions
        result["totals"]["images_generated"] = max(result["totals"]["images_generated"], gen_total)
        result["totals"]["images_analyzed"] = vis_total
        # Total images = generated (best available) + analyzed
        result["totals"]["images"] = int(result["totals"]["images_generated"]) + int(result["totals"]["images_analyzed"]) 

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
            pass

        # Tops
        if not used_sqlite_messages:
            result["top"]["users_by_messages"] = [
                {"username": u, "messages": c} for u, c in user_msg_counter.most_common(limit)
            ]
        if not result["top"]["users_by_images"]:
            result["top"]["users_by_images"] = [
                {"username": u, "images_generated": c} for u, c in user_images_index_counter.most_common(limit)
            ] or [
                {"username": u, "images_generated": c} for u, c in user_image_msg_counter.most_common(limit)
            ]
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
        import json
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
            pass
        # Rewrite index.jsonl excluding this filename
        try:
            if idx_path.exists():
                with open(idx_path, "r", encoding="utf-8") as src, NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
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
            pass
        return JSONResponse({"ok": True, "deleted": deleted})

    # Environment editor
    @app.get("/env", response_class=HTMLResponse)
    async def env_page(request: Request):
        env_path = BASE_DIR / ".env-stable"
        _, kv = parse_env(env_path)
        return templates.TemplateResponse(
            "env.html",
            {"request": request, "vars": kv, "env_path": str(env_path)},
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
            for k, v in payload.items():
                sv = "" if v is None else str(v).strip()
                if sv:  # only write non-empty values
                    updates[str(k)] = sv
            write_env(env_path, updates)
            return JSONResponse({"ok": True, "message": "Saved"})
        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.get("/api/lm-studio/models")
    async def api_lm_studio_models():
        """Fetch ALL downloaded models from LM Studio's native /api/v1/models endpoint.

        This returns every model file on disk, including all quantization variants,
        not just currently loaded models.
        """
        import aiohttp as _aiohttp

        env_path = BASE_DIR / ".env-stable"
        _, kv = parse_env(env_path)
        base_url = kv.get("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        # Native API is at /api/v1, not /v1
        api_base = base_url.replace("/v1", "/api/v1")
        logger.info("🔄 Fetching all downloaded models from %s/models", api_base)

        try:
            async with _aiohttp.ClientSession() as session:
                async with session.get(
                    f"{api_base}/models",
                    timeout=_aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        # Fallback: try the OpenAI-compatible endpoint (older LM Studio versions)
                        logger.info("🔄 Native API returned %d, falling back to /v1/models", resp.status)
                        async with session.get(
                            f"{base_url}/models",
                            timeout=_aiohttp.ClientTimeout(total=8),
                        ) as fallback_resp:
                            if fallback_resp.status != 200:
                                return JSONResponse(
                                    {"ok": False, "models": [], "message": f"LM Studio returned HTTP {fallback_resp.status}"},
                                    status_code=502,
                                )
                            fallback_data = await fallback_resp.json()
                            model_ids = [m.get("id", "") for m in fallback_data.get("data", []) if m.get("id")]
                            model_ids.sort(key=str.lower)
                            logger.info("🔄 Fallback found %d loaded model(s)", len(model_ids))
                            return JSONResponse({"ok": True, "models": model_ids})

                    data = await resp.json()

            # Native /api/v1/models returns {"models": [{"key": "...", "type": "llm", "quantization": {...}, ...}]}
            model_entries = []
            for m in data.get("models", data.get("data", [])):
                key = m.get("key") or m.get("id", "")
                if not key:
                    continue
                model_type = m.get("type", "llm")
                quant = m.get("quantization") or {}
                quant_name = quant.get("name", "")
                display = m.get("display_name", "")
                size_mb = round(m.get("size_bytes", 0) / (1024 * 1024))
                loaded = len(m.get("loaded_instances", [])) > 0

                # Build a display label: "google/gemma-3-27b (Q4_K_M, 17GB, llm)"
                label_parts = []
                if quant_name:
                    label_parts.append(quant_name)
                if size_mb > 0:
                    if size_mb >= 1024:
                        label_parts.append(f"{size_mb / 1024:.1f}GB")
                    else:
                        label_parts.append(f"{size_mb}MB")
                if model_type != "llm":
                    label_parts.append(model_type)
                if loaded:
                    label_parts.append("loaded")

                label = f"{key} ({', '.join(label_parts)})" if label_parts else key
                model_entries.append({
                    "key": key,
                    "label": label,
                    "type": model_type,
                    "quant": quant_name,
                    "size_mb": size_mb,
                    "loaded": loaded,
                })

            model_entries.sort(key=lambda e: e["key"].lower())
            logger.info("🔄 Found %d model(s) (%d LLM, %d embedding)",
                         len(model_entries),
                         sum(1 for e in model_entries if e["type"] == "llm"),
                         sum(1 for e in model_entries if e["type"] == "embedding"))
            return JSONResponse({"ok": True, "models": model_entries})

        except Exception as exc:
            logger.error("🔄 Failed to fetch models: %s", exc)
            return JSONResponse(
                {"ok": False, "models": [], "message": f"Could not reach LM Studio: {exc}"},
                status_code=502,
            )

    @app.post("/api/lm-studio/switch-model")
    async def api_lm_studio_switch_model(request: Request):
        """Unload the current model in LM Studio and load a new one with the specified context length.

        Also updates LOCAL_CHAT and CONTEXT_WINDOW_TOKENS in .env-stable.
        """
        import aiohttp as _aiohttp

        payload = await request.json()
        new_model = payload.get("model", "").strip()
        context_length = int(payload.get("context_length", 16000))

        if not new_model:
            return JSONResponse({"ok": False, "message": "No model specified"}, status_code=400)

        logger.info("🔄 ╔══════════════════════════════════════════════════╗")
        logger.info("🔄 ║  MODEL SWITCH: %s", new_model[:40])
        logger.info("🔄 ║  Context: %d tokens", context_length)
        logger.info("🔄 ╚══════════════════════════════════════════════════╝")

        env_path = BASE_DIR / ".env-stable"
        _, kv = parse_env(env_path)
        base_url = kv.get("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        # LM Studio load/unload endpoints use /api/v1 not /v1
        api_base = base_url.replace("/v1", "/api/v1")
        logger.info("🔄 LM Studio API: %s", api_base)

        # Models to NEVER unload (embedding, vision, etc.)
        protected_models = set()
        rag_model = kv.get("RAG_EMBEDDING_MODEL", "")
        vision_model = kv.get("VISION_MODEL", "")
        if rag_model:
            protected_models.add(rag_model)
        if vision_model:
            protected_models.add(vision_model)

        current_chat = kv.get("LOCAL_CHAT", "")
        logger.info("🔄 Current chat model: %s", current_chat or "(none)")
        logger.info("🔄 Protected models: %s", protected_models or "(none)")

        try:
            async with _aiohttp.ClientSession() as session:
                # Step 1: Get currently loaded models
                logger.info("🔄 Step 1: Querying loaded models...")
                async with session.get(
                    f"{base_url}/models",
                    timeout=_aiohttp.ClientTimeout(total=8),
                ) as resp:
                    loaded = []
                    if resp.status == 200:
                        data = await resp.json()
                        loaded = data.get("data", [])
                    else:
                        logger.warning("🔄 Could not query models: HTTP %d", resp.status)
                logger.info("🔄 Currently loaded: %s", [m.get("id", "?") for m in loaded])

                # Step 2: Unload ONLY the current chat model (protect embedding + vision)
                logger.info("🔄 Step 2: Unloading current chat model...")
                for m in loaded:
                    instance_id = m.get("id", "")
                    if not instance_id:
                        continue
                    if instance_id in protected_models:
                        logger.info("Keeping protected model loaded: %s", instance_id)
                        continue
                    # Only unload the current chat model (or the new model if already loaded)
                    if instance_id == current_chat or instance_id == new_model:
                        try:
                            async with session.post(
                                f"{api_base}/models/unload",
                                json={"instance_id": instance_id},
                                timeout=_aiohttp.ClientTimeout(total=15),
                            ) as unresp:
                                if unresp.status == 200:
                                    logger.info("Unloaded model: %s", instance_id)
                                else:
                                    body = await unresp.text()
                                    logger.warning("Unload %s returned %d: %s", instance_id, unresp.status, body[:200])
                        except Exception as e:
                            logger.warning("Unload %s failed: %s", instance_id, e)
                    else:
                        logger.info("Keeping unknown model loaded: %s", instance_id)

                # Step 3: Load the new model with the specified context length
                logger.info("🔄 Step 3: Loading %s (ctx=%d)...", new_model, context_length)
                load_payload = {
                    "model": new_model,
                    "context_length": context_length,
                    "flash_attention": True,
                    "echo_load_config": True,
                }
                async with session.post(
                    f"{api_base}/models/load",
                    json=load_payload,
                    timeout=_aiohttp.ClientTimeout(total=120),  # Loading can take a while
                ) as lresp:
                    if lresp.status != 200:
                        body = await lresp.text()
                        logger.error("🔄 ❌ Load failed: HTTP %d — %s", lresp.status, body[:300])
                        return JSONResponse(
                            {"ok": False, "message": f"LM Studio load failed ({lresp.status}): {body[:300]}"},
                            status_code=502,
                        )
                    load_result = await lresp.json()
                    logger.info("🔄 ✅ Model loaded successfully")

            # Step 4: Update .env-stable with new model and context window
            logger.info("🔄 Step 4: Updating .env-stable...")
            write_env(env_path, {
                "LOCAL_CHAT": new_model,
                "CONTEXT_WINDOW_TOKENS": str(context_length),
            })

            logger.info("🔄 ✅ Model switch complete: %s (%d ctx)", new_model, context_length)
            return JSONResponse({
                "ok": True,
                "message": f"Loaded {new_model} with {context_length} context",
                "load_config": load_result,
            })

        except Exception as exc:
            logger.error("🔄 ❌ Model switch failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"ok": False, "message": f"Model switch failed: {exc}"},
                status_code=502,
            )

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
                if isinstance(raw, dict):
                    data.update(raw)
        except Exception:
            pass

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
            pass

        # Self-knowledge pending (accumulator file is on disk)
        try:
            acc_path = os.path.join("data", "self_md", "accumulator.jsonl")
            if os.path.exists(acc_path):
                lines = [l for l in open(acc_path).read().splitlines() if l.strip()]
                data["self_md_pending"] = len(lines)
            from soupy_database.runtime_flags import read_runtime_flags
            flags = read_runtime_flags()
            data["self_md_enabled"] = bool(
                os.getenv("SELF_MD_ENABLED", "false").strip().lower() in ("1", "true", "yes")
            )
        except Exception:
            pass

        return JSONResponse(data)

    @app.get("/api/bot/activity")
    async def api_bot_activity():
        """Return recent Bluesky and Daily Post activity from persisted history files.

        Query params:
          ?type=all|replies|posts|reposts|dailyposts  (default: all)
          ?limit=50  (default: 50, max: 200)
        """
        import json as _json
        from datetime import date, datetime, timezone

        req = app.state._request if hasattr(app.state, "_request") else None
        # Parse from starlette request
        from starlette.requests import Request

        # We'll just return everything and let the frontend filter
        today = date.today().isoformat()
        result = {
            "ok": True,
            "today": today,
            "bluesky": {"replies_today": 0, "posts_today": 0, "reposts_today": 0,
                        "replies": [], "posts": [], "reposts": []},
            "daily_posts": {"posts_today": 0, "posts": []},
        }

        # Bluesky history
        try:
            bsky_path = os.path.join("data", "bluesky_engage_history.json")
            if os.path.exists(bsky_path):
                bh = _json.loads(open(bsky_path).read())

                comments = bh.get("comments", [])
                result["bluesky"]["replies"] = comments[-50:]
                result["bluesky"]["replies_today"] = sum(
                    1 for c in comments if c.get("ts", "").startswith(today))

                posts = bh.get("posts", [])
                result["bluesky"]["posts"] = posts[-50:]
                result["bluesky"]["posts_today"] = sum(
                    1 for p in posts if p.get("ts", "").startswith(today))

                reposts = bh.get("reposts", [])
                result["bluesky"]["reposts"] = reposts[-50:]
                result["bluesky"]["reposts_today"] = sum(
                    1 for r in reposts if r.get("ts", "").startswith(today))
        except Exception:
            pass

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
                result["daily_posts"]["posts_today"] = sum(
                    1 for p in all_posts if p.get("date") == today)
        except Exception:
            pass

        return JSONResponse(result)

    @app.get("/api/runtime-flags")
    async def api_runtime_flags_get():
        from soupy_database.runtime_flags import read_runtime_flags

        return JSONResponse(read_runtime_flags())

    @app.get("/api/channels/{guild_id}")
    async def api_channels_list(guild_id: int):
        """Return all known channels for a guild from the SQLite archive."""
        import sqlite3 as _sql
        from soupy_database.database import get_db_path
        db_path = get_db_path(guild_id)
        if not os.path.exists(db_path):
            return JSONResponse({"ok": False, "channels": []})
        conn = _sql.connect(db_path, check_same_thread=False)
        conn.row_factory = _sql.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT c.channel_id, c.channel_name, COUNT(m.message_id) AS msg_count
                FROM channels c
                LEFT JOIN messages m ON m.channel_id = c.channel_id
                GROUP BY c.channel_id
                ORDER BY msg_count DESC
                """
            )
            channels = [
                {"id": str(r["channel_id"]), "name": r["channel_name"] or str(r["channel_id"]), "messages": r["msg_count"]}
                for r in cur.fetchall()
            ]
        finally:
            conn.close()
        return JSONResponse({"ok": True, "channels": channels})

    @app.post("/api/runtime-flags")
    async def api_runtime_flags_post(request: Request):
        from soupy_database.runtime_flags import write_runtime_flags

        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "message": "Invalid payload"}, status_code=400)
        updates = {}
        if "rag_enabled" in payload:
            updates["rag_enabled"] = bool(payload["rag_enabled"])
        if "disabled_commands" in payload:
            val = payload["disabled_commands"]
            if isinstance(val, list):
                updates["disabled_commands"] = [str(c).lower().strip() for c in val if c]
        if not updates:
            return JSONResponse({"ok": False, "message": "No supported keys"}, status_code=400)
        merged = write_runtime_flags(updates)
        return JSONResponse({"ok": True, **merged})

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

    @app.post("/api/profiles/rebuild/{guild_id}")
    async def api_profiles_rebuild(guild_id: str):
        """Legacy one-shot batch (blocking). Prefer /api/profiles/batch/start + pause/resume."""
        from soupy_database import get_stats
        from soupy_database.user_profiles import refresh_profiles_for_guild_limited

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        st = get_stats(gid)
        if not st.get("exists"):
            return JSONResponse({"ok": False, "message": "Database not found"}, status_code=404)
        try:
            result = await refresh_profiles_for_guild_limited(gid)
            if not result.get("ok"):
                return JSONResponse({"ok": False, **result}, status_code=500)
            return JSONResponse({"ok": True, **result})
        except Exception as exc:
            logging.exception("Profile rebuild failed")
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.post("/api/profiles/reset/{guild_id}")
    async def api_profiles_reset(guild_id: str):
        """Delete all stored profiles and batch job state for a guild."""
        from soupy_database.profile_batch import cancel_profile_task
        from soupy_database.user_profiles import clear_stored_profiles

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        await cancel_profile_task(gid)
        result = clear_stored_profiles(gid)
        if not result.get("ok"):
            return JSONResponse(result, status_code=404)
        return JSONResponse({"ok": True, "cleared": True})

    @app.post("/api/profiles/batch/start/{guild_id}")
    async def api_profiles_batch_start(guild_id: str):
        from soupy_database.profile_batch import cancel_profile_task
        from soupy_database.user_profiles import start_profile_batch_job

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)
        await cancel_profile_task(gid)
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
    async def api_profiles_list(
        guild_id: str,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ):
        """Browse stored member profile summaries (RAG sketch rows) for a guild."""
        import sqlite3

        from soupy_database.database import get_db_path
        from soupy_database.user_profiles import ensure_user_profile_schema

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)

        db_path = get_db_path(gid)
        if not os.path.exists(db_path):
            return JSONResponse({"ok": False, "message": "Database not found"}, status_code=404)

        safe_limit = max(1, min(int(limit), 200))
        safe_offset = max(0, int(offset))

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            ensure_user_profile_schema(conn)
            cur = conn.cursor()
            if search and search.strip():
                term = f"%{search.strip()}%"
                cur.execute(
                    """
                    SELECT COUNT(*) AS c FROM user_profile_summaries
                    WHERE CAST(user_id AS TEXT) LIKE ?
                       OR COALESCE(nickname_hint, '') LIKE ?
                       OR COALESCE(summary, '') LIKE ?
                    """,
                    (term, term, term),
                )
                total = int(cur.fetchone()["c"])
                cur.execute(
                    """
                    SELECT user_id, nickname_hint, summary, structured_json, source_message_count,
                           source_max_message_id, updated_at, model_used
                    FROM user_profile_summaries
                    WHERE CAST(user_id AS TEXT) LIKE ?
                       OR COALESCE(nickname_hint, '') LIKE ?
                       OR COALESCE(summary, '') LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (term, term, term, safe_limit, safe_offset),
                )
            else:
                cur.execute("SELECT COUNT(*) AS c FROM user_profile_summaries")
                total = int(cur.fetchone()["c"])
                cur.execute(
                    """
                    SELECT user_id, nickname_hint, summary, structured_json, source_message_count,
                           source_max_message_id, updated_at, model_used
                    FROM user_profile_summaries
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_limit, safe_offset),
                )
            rows = [dict(r) for r in cur.fetchall()]
            return JSONResponse(
                {
                    "ok": True,
                    "guild_id": str(gid),
                    "total": total,
                    "limit": safe_limit,
                    "offset": safe_offset,
                    "rows": rows,
                }
            )
        finally:
            conn.close()

    @app.get("/api/database/user-picker/{guild_id}")
    async def api_database_user_picker(guild_id: str, archive_limit: int = 400):
        """
        Lists users for dashboard dropdowns: top posters from messages, and rows in user_profile_summaries.
        """
        import sqlite3

        from soupy_database.database import get_db_path
        from soupy_database.user_profiles import ensure_user_profile_schema

        try:
            gid = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)

        db_path = get_db_path(gid)
        if not os.path.exists(db_path):
            return JSONResponse({"ok": False, "message": "Database not found"}, status_code=404)

        lim = max(1, min(int(archive_limit), 800))

        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception as exc:
            logging.exception("user-picker connect failed guild=%s", gid)
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT user_id,
                       COUNT(*) AS message_count,
                       MAX(COALESCE(NULLIF(TRIM(nickname), ''), NULLIF(TRIM(username), ''))) AS label
                FROM messages
                GROUP BY user_id
                ORDER BY message_count DESC
                LIMIT ?
                """,
                (lim,),
            )
            from_messages: list[Dict[str, Any]] = []
            for r in cur.fetchall():
                uid = int(r["user_id"])
                lbl = (r["label"] or "").strip() or f"user {uid}"
                from_messages.append(
                    {
                        "user_id": uid,
                        "label": lbl,
                        "message_count": int(r["message_count"]),
                    }
                )

            ensure_user_profile_schema(conn)
            cur.execute("SELECT COUNT(*) AS c FROM user_profile_summaries")
            profile_row_count = int(cur.fetchone()["c"])
            cur.execute(
                """
                SELECT ups.user_id,
                       ups.nickname_hint,
                       ups.updated_at,
                       COALESCE(
                           NULLIF(TRIM(ups.nickname_hint), ''),
                           (SELECT MAX(
                               COALESCE(
                                   NULLIF(TRIM(m.nickname), ''),
                                   NULLIF(TRIM(m.username), '')
                               )
                            )
                            FROM messages m
                            WHERE m.user_id = ups.user_id)
                       ) AS label
                FROM user_profile_summaries ups
                ORDER BY LOWER(TRIM(COALESCE(ups.nickname_hint, ''))), ups.user_id
                """
            )
            from_profiles: list[Dict[str, Any]] = []
            for r in cur.fetchall():
                uid = int(r["user_id"])
                nh = (r["nickname_hint"] or "").strip()
                lbl = (r["label"] or "").strip() or nh
                from_profiles.append(
                    {
                        "user_id": uid,
                        "nickname_hint": nh,
                        "label": lbl,
                        "updated_at": r["updated_at"],
                    }
                )

            return JSONResponse(
                {
                    "ok": True,
                    "guild_id": str(gid),
                    "profile_row_count": profile_row_count,
                    "from_messages": from_messages,
                    "from_profiles": from_profiles,
                }
            )
        except Exception as exc:
            logging.exception("user-picker query failed guild=%s", gid)
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)
        finally:
            conn.close()

    # Bot control APIs
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

    # Database scan APIs
    @app.get("/api/database/status")
    async def api_database_status():
        """Get active scans and overall database status."""
        try:
            from soupy_database import get_stats
            import os
            from pathlib import Path
            
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
                            "cancelled": task.cancelled()
                        }
            except (ImportError, AttributeError):
                # Bot module not available or active_scans not defined
                pass
            
            # Get all databases
            db_dir = os.getenv("SOUPY_DB_DIR", os.path.join(BASE_DIR, "soupy_database", "databases"))
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
                            databases.append({
                                "guild_id": str(guild_id),
                                "guild_name": stats.get("guild_name"),
                                "total_messages": stats["total_messages"],
                                "total_channels": stats["total_channels"],
                                "last_scan": stats["last_scan"],
                                "archive_scan_interval_minutes": stats.get(
                                    "archive_scan_interval_minutes", 0
                                ),
                                "file_size": db_file.stat().st_size if db_file.exists() else 0
                            })
                    except (ValueError, Exception):
                        continue
            
            return JSONResponse({
                "ok": True,
                "active_scans": active,
                "databases": databases,
                "total_databases": len(databases)
            })
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    
    @app.get("/api/database/stats/{guild_id}")
    async def api_database_stats(guild_id: str):
        """Get detailed stats for a specific server database."""
        try:
            from soupy_database import get_stats, get_last_scan_time
            import sqlite3
            from datetime import datetime, timedelta
            from collections import defaultdict
            import os
            
            # Get display timezone (from app context or env)
            display_tz_str = os.getenv("TIMEZONE", "America/Los_Angeles")
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
                    scan_history.append({
                        "scan_type": row["scan_type"],
                        "last_scan_time": row["last_scan_time"],
                        "messages_scanned": row["messages_scanned"],
                        "created_at": row["created_at"]
                    })
                
                detailed_stats["scan_history"] = scan_history
                
                # Get total scan count
                cursor.execute("SELECT COUNT(*) as count FROM scan_metadata")
                detailed_stats["total_scans"] = cursor.fetchone()["count"]
                
                # Get image/URL stats
                cursor.execute("SELECT COUNT(*) as count FROM messages WHERE image_description IS NOT NULL")
                detailed_stats["messages_with_images"] = cursor.fetchone()["count"]
                
                cursor.execute("SELECT COUNT(*) as count FROM messages WHERE url_summary IS NOT NULL")
                detailed_stats["messages_with_urls"] = cursor.fetchone()["count"]
                
                cursor.execute("SELECT COUNT(*) as count FROM messages WHERE image_description IS NOT NULL AND url_summary IS NOT NULL")
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
                    channel_stats.append({
                        "channel_id": str(row["channel_id"]),
                        "channel_name": row["channel_name"],
                        "message_count": row["message_count"]
                    })
                
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
                    user_stats.append({
                        "user_id": str(row["user_id"]),
                        "username": row["username"],
                        "message_count": row["message_count"]
                    })
                
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
                    earliest_utc = datetime.strptime(earliest_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    latest_utc = datetime.strptime(latest_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    
                    # Convert to display timezone
                    earliest_local = earliest_utc.astimezone(display_tz)
                    latest_local = latest_utc.astimezone(display_tz)
                    
                    days_span = (latest_utc - earliest_utc).days
                    detailed_stats["date_range"] = {
                        "earliest": earliest_local.strftime('%Y-%m-%d %H:%M:%S'),
                        "latest": latest_local.strftime('%Y-%m-%d %H:%M:%S'),
                        "earliest_utc": earliest_str,
                        "latest_utc": latest_str,
                        "timezone": display_tz_str,
                        "days_span": days_span
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
                    daily_messages.append({
                        "date": row["date"],
                        "count": row["message_count"]
                    })
                
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
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", '%Y-%m-%d %H:%M:%S')
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
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", '%Y-%m-%d %H:%M:%S')
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
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", '%Y-%m-%d %H:%M:%S')
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
                    user_activity_list.append({
                        "user_id": user_id,
                        "username": username,
                        "hourly_activity": hours,
                        "total_messages": sum(hours)
                    })
                
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
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", '%Y-%m-%d %H:%M:%S')
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
                    channel_activity_list.append({
                        "channel_id": channel_id,
                        "channel_name": data["name"],
                        "hourly_activity": data["hours"],
                        "total_messages": total,
                        "peak_hour": data["hours"].index(max(data["hours"])) if max(data["hours"]) > 0 else None
                    })
                
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
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", '%Y-%m-%d %H:%M:%S')
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        weekday = dt_local.weekday()  # 0=Monday, 6=Sunday
                        user_weekday_activity[user_id]["weekdays"][weekday] += row["message_count"]
                    except:
                        pass
                
                # Convert to list
                user_weekday_list = []
                for user_id, data in user_weekday_activity.items():
                    total = sum(data["weekdays"])
                    user_weekday_list.append({
                        "user_id": user_id,
                        "username": data["username"],
                        "weekday_activity": data["weekdays"],
                        "total_messages": total
                    })
                
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
                
                channel_time_periods = defaultdict(lambda: {"name": "", "morning": 0, "afternoon": 0, "evening": 0, "night": 0})
                for row in cursor.fetchall():
                    try:
                        channel_id = str(row["channel_id"])
                        channel_time_periods[channel_id]["name"] = row["channel_name"]
                        # Parse UTC datetime from database
                        dt_utc = datetime.strptime(f"{row['date']} {row['time']}", '%Y-%m-%d %H:%M:%S')
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                        # Convert to display timezone
                        dt_local = dt_utc.astimezone(display_tz)
                        hour = dt_local.hour
                        # Determine time period based on local time
                        if 6 <= hour <= 11:
                            period = 'morning'
                        elif 12 <= hour <= 17:
                            period = 'afternoon'
                        elif 18 <= hour <= 22:
                            period = 'evening'
                        else:
                            period = 'night'
                        channel_time_periods[channel_id][period] += row["message_count"]
                    except Exception:
                        pass
                
                channel_period_list = []
                for channel_id, data in channel_time_periods.items():
                    total = data["morning"] + data["afternoon"] + data["evening"] + data["night"]
                    channel_period_list.append({
                        "channel_id": channel_id,
                        "channel_name": data["name"],
                        "morning": data["morning"],
                        "afternoon": data["afternoon"],
                        "evening": data["evening"],
                        "night": data["night"],
                        "total_messages": total,
                        "most_active_period": max(["morning", "afternoon", "evening", "night"], key=lambda p: data[p])
                    })
                
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
    ):
        """Browse database rows with safe, pre-defined filtering."""
        try:
            import sqlite3
            from soupy_database.database import get_db_path
        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

        try:
            guild_id_int = int(guild_id)
        except ValueError:
            return JSONResponse({"ok": False, "message": "Invalid guild ID"}, status_code=400)

        table_name = (table or "messages").strip().lower()
        if table_name not in {"messages", "scan_metadata"}:
            return JSONResponse({"ok": False, "message": "Unsupported table"}, status_code=400)

        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)

        db_path = get_db_path(guild_id_int)
        if not os.path.exists(db_path):
            return JSONResponse({"ok": False, "message": "Database not found"}, status_code=404)

        try:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            if table_name == "scan_metadata":
                cur.execute("SELECT COUNT(*) AS c FROM scan_metadata")
                total = int(cur.fetchone()["c"])
                cur.execute(
                    """
                    SELECT scan_type, last_scan_time, messages_scanned, created_at
                    FROM scan_metadata
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_limit, safe_offset),
                )
                rows = [dict(r) for r in cur.fetchall()]
                conn.close()
                return JSONResponse(
                    {
                        "ok": True,
                        "table": table_name,
                        "total": total,
                        "limit": safe_limit,
                        "offset": safe_offset,
                        "rows": rows,
                    }
                )

            where_parts: list[str] = []
            params: list[Any] = []
            if search:
                where_parts.append(
                    "(message_content LIKE ? OR username LIKE ? OR channel_name LIKE ? OR image_description LIKE ? OR url_summary LIKE ?)"
                )
                search_term = f"%{search.strip()}%"
                params.extend([search_term, search_term, search_term, search_term, search_term])
            if channel_id:
                where_parts.append("CAST(channel_id AS TEXT) = ?")
                params.append(channel_id)
            if user_id:
                where_parts.append("CAST(user_id AS TEXT) = ?")
                params.append(user_id)
            if has_images:
                where_parts.append("image_description IS NOT NULL AND TRIM(image_description) != ''")
            if has_urls:
                where_parts.append("url_summary IS NOT NULL AND TRIM(url_summary) != ''")

            where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

            count_sql = f"SELECT COUNT(*) AS c FROM messages {where_sql}"
            cur.execute(count_sql, params)
            total = int(cur.fetchone()["c"])

            data_sql = f"""
                SELECT
                    date, time, username, user_id, channel_name, channel_id,
                    message_content, image_description, url_summary
                FROM messages
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            data_params = [*params, safe_limit, safe_offset]
            cur.execute(data_sql, data_params)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()

            return JSONResponse(
                {
                    "ok": True,
                    "table": table_name,
                    "total": total,
                    "limit": safe_limit,
                    "offset": safe_offset,
                    "rows": rows,
                }
            )
        except Exception as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

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
        if os.environ.get("SOUPY_AUTOSTART_BOT", "0") in {"1", "true", "True"}:
            await bot_runner.start()

    @app.on_event("shutdown")
    async def _shutdown():
        # Ensure bot is stopped when app shuts down
        await bot_runner.stop()

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


