"""
Flux server manager: starts and stops ``flux_server.py`` and keeps its log.

The web app holds one :class:`FluxRunner`. It:

1. Spawns ``flux_server.py`` with ``.env-stable`` laid over the panel's
   environment, so a ``FLUX_*`` edit takes effect on the server's next start.
2. Writes every output line to ``logs/flux_server.log`` and passes it to
   ``on_output`` with a ``[flux]`` prefix (the panel's live log).
3. Reports whether a Flux server answers ``GET /health`` at
   ``FLUX_SERVER_URL``, whoever started it.

Gotchas:

* A server started by hand (``./flux_server.sh``) is reported as running
  outside the panel and left alone. Start refuses while anything answers at
  the URL, and Stop only ends the panel's own child.
* mflux draws progress bars with carriage returns. Those arrive as one line
  per update, at most one a second, so a model download doesn't flood the log.
* Stopping the server mid-generation loses that picture. The bot reports the
  failed request to whoever asked for it.
* The model loads on the first request after a start, not at start, so a
  freshly started server answers ``/health`` with ``loaded: false``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Awaitable, Callable, Deque, Dict, List, Optional, Tuple

import aiohttp

from .env_store import parse_env

OutputCallback = Callable[[str], Awaitable[None]]

HEALTH_TIMEOUT = 2.0
STOP_TIMEOUT = 10.0
# Seconds between progress-bar updates passed on.
PROGRESS_INTERVAL = 1.0
TAIL_LINES = 300

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TRUE = ("1", "true", "yes", "on")


def _is_on(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in _TRUE


class FluxRunner:
    """Manages the local Flux server's process and log."""

    def __init__(
        self,
        project_root: Path,
        on_output: OutputCallback,
        *,
        command: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self._project_root = Path(project_root)
        self._on_output = on_output
        # Overrides for tests; the defaults come from the project and .env-stable.
        self._command = command
        self._base_url = base_url
        self._log_path = Path(log_path) if log_path else self._project_root / "logs" / "flux_server.log"
        self._process: Optional[asyncio.subprocess.Process] = None
        self._start_time: Optional[datetime] = None
        self._reader: Optional[asyncio.Task] = None
        # True once Stop was asked for, so the exit that follows isn't shown as a crash.
        self._stop_requested = False
        self._tail: Deque[str] = deque(maxlen=TAIL_LINES)
        self._file_log: Optional[logging.Logger] = None

    # ---- configuration ---------------------------------------------------

    def _env_values(self) -> Dict[str, str]:
        _, values = parse_env(self._project_root / ".env-stable")
        return values

    def enabled(self) -> bool:
        return _is_on(self._env_values().get("FLUX_ENABLED"))

    def autostart_wanted(self) -> bool:
        values = self._env_values()
        return _is_on(values.get("FLUX_ENABLED")) and _is_on(values.get("FLUX_SERVER_AUTOSTART"))

    def base_url(self) -> str:
        """Where the bot reaches the server: ``FLUX_SERVER_URL``, else the bind host and port."""
        if self._base_url:
            return self._base_url.rstrip("/")
        values = self._env_values()
        url = (values.get("FLUX_SERVER_URL") or "").strip()
        if not url:
            host = (values.get("FLUX_SERVER_HOST") or "127.0.0.1").strip()
            port = (values.get("FLUX_SERVER_PORT") or "4942").strip()
            url = f"http://{host}:{port}"
        return url.rstrip("/")

    # ---- state -----------------------------------------------------------

    def _alive(self) -> bool:
        return bool(self._process and self._process.returncode is None)

    async def health(self) -> Optional[dict]:
        """The server's ``/health`` answer, or ``None`` when nothing answers."""
        try:
            timeout = aiohttp.ClientTimeout(total=HEALTH_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url()}/health") as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
                    return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def status(self) -> Dict[str, object]:
        health = await self.health()
        running = self._alive()
        return {
            "enabled": self.enabled(),
            "autostart": self.autostart_wanted(),
            "url": self.base_url(),
            # The panel's own child process.
            "running": running,
            "pid": self._process.pid if self._process and running else None,
            "start_time": self._start_time.isoformat() if self._start_time and running else None,
            "returncode": self._process.returncode if self._process else None,
            "stopped_by_panel": self._stop_requested and not running,
            # Whatever answers at the URL, the panel's child or not.
            "reachable": health is not None,
            "external": health is not None and not running,
            "model": (health or {}).get("model"),
            "loaded": (health or {}).get("loaded"),
            "edit_model": (health or {}).get("edit_model"),
            "edit_loaded": (health or {}).get("edit_loaded"),
            "cache_limit_gb": (health or {}).get("cache_limit_gb"),
        }

    def tail(self, lines: int = 100) -> List[str]:
        """The newest output lines since the panel started, oldest first."""
        lines = max(1, min(int(lines), TAIL_LINES))
        return list(self._tail)[-lines:]

    # ---- lifecycle -------------------------------------------------------

    async def start(self) -> Tuple[bool, str]:
        if self._alive():
            return False, "Flux server is already running"
        if await self.health() is not None:
            return False, f"A Flux server started outside the panel is already answering at {self.base_url()}"

        command = self._command or [sys.executable, "-u", str(self._project_root / "flux_server.py")]
        if not self._command and not (self._project_root / "flux_server.py").exists():
            return False, "flux_server.py not found"
        try:
            child_env = os.environ.copy()
            # The file wins, as it does for the bot: it is what the panel edits.
            child_env.update(self._env_values())
            child_env["PYTHONUNBUFFERED"] = "1"
            self._process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._project_root),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=child_env,
            )
        except Exception as exc:
            await self._emit(f"[web] Failed to start Flux server: {exc}")
            return False, f"Failed to start: {exc}"
        self._start_time = datetime.now(timezone.utc)
        self._stop_requested = False
        self._reader = asyncio.create_task(self._read_output(self._process))
        await self._emit(f"[web] Started Flux server pid={self._process.pid} at {self._start_time.isoformat()}")
        return True, "Flux server starting"

    async def stop(self) -> Tuple[bool, str]:
        if not self._alive():
            if await self.health() is not None:
                return False, "This Flux server was started outside the panel. Stop it where it was started."
            return False, "Flux server is not running"
        process = self._process
        assert process is not None
        self._stop_requested = True
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=STOP_TIMEOUT)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass
        except Exception as exc:
            await self._emit(f"[web] Failed to stop Flux server: {exc}")
            return False, f"Failed to stop: {exc}"
        # The reader ends at end of output and reports the exit.
        if self._reader:
            try:
                await asyncio.wait_for(self._reader, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._reader.cancel()
            self._reader = None
        return True, "Flux server stopped"

    async def restart(self) -> Tuple[bool, str]:
        if self._alive():
            ok, message = await self.stop()
            if not ok:
                return ok, message
        return await self.start()

    # ---- output ----------------------------------------------------------

    def _logger(self) -> Optional[logging.Logger]:
        if self._file_log is None:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(self._log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(message)s"))
                log = logging.getLogger(f"soupy.flux_server_log.{id(self)}")
                log.setLevel(logging.INFO)
                log.propagate = False
                log.addHandler(handler)
                self._file_log = log
            except Exception:
                return None
        return self._file_log

    async def _emit(self, line: str) -> None:
        """One line to the tail, the log file and the live log."""
        clean = _ANSI_RE.sub("", line).rstrip()
        if not clean:
            return
        self._tail.append(clean)
        log = self._logger()
        if log is not None:
            log.info(clean)
        try:
            await self._on_output(clean if clean.startswith("[web]") else f"[flux] {clean}")
        except Exception:
            pass

    async def _read_output(self, process: asyncio.subprocess.Process) -> None:
        """Pass the child's output on line by line until it exits, then report the exit."""
        assert process.stdout is not None
        pending = ""
        last_progress = 0.0
        try:
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                pending += chunk.decode(errors="replace")
                # A segment ending in "\r" is a progress-bar redraw, one in "\n" a finished line.
                while True:
                    match = re.search(r"[\r\n]", pending)
                    if not match:
                        break
                    segment, ended_by = pending[: match.start()], match.group()
                    pending = pending[match.end() :]
                    if ended_by == "\r" and not pending.startswith("\n"):
                        now = time.monotonic()
                        if now - last_progress < PROGRESS_INTERVAL:
                            continue
                        last_progress = now
                    await self._emit(segment)
            if pending.strip():
                await self._emit(pending)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self._emit(f"[web] Flux server output error: {exc}")
        code = await process.wait()
        await self._emit(f"[web] Flux server exited with code {code}")
