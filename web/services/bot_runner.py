from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Tuple

from .env_store import parse_env


OutputCallback = Callable[[str], Awaitable[None]]


@dataclass
class BotProcessStatus:
    running: bool
    pid: Optional[int]
    start_time: Optional[str]
    returncode: Optional[int]


class BotRunner:
    """Manages the Soupy bot process lifecycle and streams its output."""

    def __init__(self, project_root: Path, on_output: OutputCallback) -> None:
        self._project_root = Path(project_root)
        self._on_output = on_output
        self._process: Optional[asyncio.subprocess.Process] = None
        self._start_time: Optional[datetime] = None
        self._stdout_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._pty_master_fd: Optional[int] = None
        self._pty_task: Optional[asyncio.Task] = None

    def _resolve_entrypoint(self) -> Optional[str]:
        # Allow override via environment variable
        override = os.environ.get("SOUPY_BOT_ENTRY")
        if override:
            candidate = Path(override)
            if not candidate.is_absolute():
                candidate = self._project_root / candidate
            if candidate.exists():
                return str(candidate)

        # Preferred current main file
        preferred = self._project_root / "soupy_remastered_stablediffusion.py"
        if preferred.exists():
            return str(preferred)

        # Legacy/alternate names
        legacy = [
            self._project_root / "run_soupy.py",
            self._project_root / "soupy_remastered.py",
        ]
        for path in legacy:
            if path.exists():
                return str(path)

        return None

    async def start(self) -> Tuple[bool, str]:
        if self._process and self._process.returncode is None:
            return False, "Bot is already running"

        python_exe = sys.executable
        entrypoint = self._resolve_entrypoint()
        if not entrypoint or not os.path.exists(entrypoint):
            return False, f"Entrypoint not found"

        try:
            # Prepare environment; force colored output and ensure required vars
            child_env = os.environ.copy()
            child_env.setdefault("FORCE_COLOR", "1")
            child_env.setdefault("CLICOLOR", "1")
            child_env.setdefault("CLICOLOR_FORCE", "1")
            child_env.setdefault("PY_COLORS", "1")
            child_env.setdefault("TERM", "xterm-256color")

            # Merge .env-stable variables to child environment.
            # ALWAYS override from the file — the file is the source of truth,
            # especially after model switches or env edits via the web UI.
            env_path = self._project_root / ".env-stable"
            _, kv = parse_env(env_path)
            for k, v in kv.items():
                child_env[k] = v
            # Map LOCAL_KEY to OPENAI_API_KEY if missing/empty
            if not child_env.get("OPENAI_API_KEY") and child_env.get("LOCAL_KEY"):
                child_env["OPENAI_API_KEY"] = child_env["LOCAL_KEY"]

            use_pty = os.name == "posix"
            if use_pty:
                import pty
                import fcntl
                import termios

                master_fd, slave_fd = pty.openpty()
                # Non-blocking master to avoid potential hangs
                flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
                fcntl.fcntl(master_fd, fcntl.F_SETFL, flags)

                self._process = await asyncio.create_subprocess_exec(
                    python_exe,
                    "-u",
                    entrypoint,
                    cwd=str(self._project_root),
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    env=child_env,
                )
                # Parent does not need the slave end
                try:
                    os.close(slave_fd)
                except Exception:
                    pass
                self._pty_master_fd = master_fd
            else:
                self._process = await asyncio.create_subprocess_exec(
                    python_exe,
                    "-u",
                    entrypoint,
                    cwd=str(self._project_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=child_env,
                )

            self._start_time = datetime.now(timezone.utc)

            if use_pty and self._pty_master_fd is not None:
                self._pty_task = asyncio.create_task(self._stream_pty(self._pty_master_fd))
            else:
                # Stream outputs via pipes
                assert self._process.stdout and self._process.stderr
                self._stdout_task = asyncio.create_task(
                    self._stream(self._process.stdout, prefix="STDOUT")
                )
                self._stderr_task = asyncio.create_task(
                    self._stream(self._process.stderr, prefix="STDERR")
                )
            await self._on_output(
                f"[web] Started bot pid={self._process.pid} using {Path(entrypoint).name} at {self._start_time.isoformat()}"
            )
            return True, "Bot started"
        except Exception as exc:
            await self._on_output(f"[web] Failed to start bot: {exc}")
            return False, f"Failed to start: {exc}"

    async def stop(self) -> Tuple[bool, str]:
        if not self._process or self._process.returncode is not None:
            return False, "Bot is not running"
        try:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            await self._on_output("[web] Bot stopped")
            return True, "Bot stopped"
        except Exception as exc:
            await self._on_output(f"[web] Failed to stop bot: {exc}")
            return False, f"Failed to stop: {exc}"
        finally:
            self._cleanup_tasks()

    def _cleanup_tasks(self) -> None:
        for task in (self._stdout_task, self._stderr_task, self._pty_task):
            if task and not task.done():
                task.cancel()
        self._stdout_task = None
        self._stderr_task = None
        self._pty_task = None
        if self._pty_master_fd is not None:
            try:
                os.close(self._pty_master_fd)
            except Exception:
                pass
            self._pty_master_fd = None

    async def status(self) -> Dict[str, Optional[object]]:
        running = bool(self._process and self._process.returncode is None)
        return {
            "running": running,
            "pid": self._process.pid if self._process else None,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "returncode": self._process.returncode if self._process else None,
        }

    async def _stream(self, stream: asyncio.StreamReader, prefix: str) -> None:
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip("\n")
                await self._on_output(f"[{prefix}] {text}")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self._on_output(f"[web] Stream error: {exc}")

    async def _stream_pty(self, master_fd: int) -> None:
        loop = asyncio.get_running_loop()
        try:
            with os.fdopen(master_fd, "rb", buffering=0, closefd=False) as f:
                while True:
                    line = await loop.run_in_executor(None, f.readline)
                    if not line:
                        break
                    text = line.decode(errors="replace").rstrip("\n")
                    # PTY already includes combined stdout/stderr
                    await self._on_output(text)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self._on_output(f"[web] PTY stream error: {exc}")


