import os
import subprocess
import sys
import time


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


