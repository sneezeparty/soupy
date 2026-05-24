"""
Standard launch script — starts the web control panel, which in turn
starts the Discord bot.

This is the production-style entry point. It does *not* spawn the bot
directly; it spawns uvicorn with ``SOUPY_AUTOSTART_BOT=1``, and
:func:`web.app.create_app`'s ``startup`` hook then calls
:meth:`web.services.bot_runner.BotRunner.start`. The bot's stdout/stderr
stream through a PTY in ``bot_runner.py`` into ``logs/soupy.log`` and the
``/ws/logs`` WebSocket.

Implications:

* Killing this process stops uvicorn, which terminates the bot subprocess.
* To restart the bot after code changes, use the web panel's restart
  button or ``POST /api/bot/restart`` — restarting uvicorn alone is *not*
  required to pick up bot code changes (and would lose connected
  WebSocket log viewers for no reason).
* To launch only the web panel (no bot), set ``SOUPY_AUTOSTART_BOT=0``
  before running uvicorn directly.
* To launch only the bot (no web), run
  ``python soupy_remastered_stablediffusion.py`` — useful for direct
  stack traces while debugging.

Environment knobs read here:

* ``SOUPY_WEB_HOST`` (default ``0.0.0.0``) — uvicorn bind host.
* ``SOUPY_WEB_PORT`` (default ``4941``) — uvicorn bind port.
* ``SOUPY_AUTOSTART_BOT`` — set to ``1`` if unset, so the bot autostarts.

uvicorn is invoked with ``--no-access-log --log-level warning`` so the
terminal isn't drowned in dashboard polling traffic; the bot's own
logging takes care of the interesting lines.
"""

import os
import subprocess
import sys


def main() -> int:
    env = os.environ.copy()
    # Ensure web app auto-starts the bot on startup
    env.setdefault("SOUPY_AUTOSTART_BOT", "1")

    host = env.get("SOUPY_WEB_HOST", "0.0.0.0")
    port = env.get("SOUPY_WEB_PORT", "4941")

    # Launch uvicorn serving the web app; inherit current venv
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "web.app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--no-access-log",
        "--log-level",
        "warning",
    ]
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


