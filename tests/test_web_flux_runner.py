"""Tests for web.services.flux_runner: the panel starting, stopping and watching the Flux server.

The server here is a small script that prints what ``flux_server.py`` prints
(log lines, a progress bar drawn with carriage returns) and answers
``/health``. No model is involved.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import textwrap

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from web.services.flux_runner import FluxRunner

FAKE_SERVER = textwrap.dedent("""
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "serve"
    print("Starting Flux server on 127.0.0.1:%d (model=%s)" % (port, os.environ.get("FLUX_MODEL")), flush=True)
    if mode == "crash":
        sys.stderr.write("OSError: [Errno 48] Address already in use\\n")
        sys.exit(3)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"status": "ok", "model": os.environ.get("FLUX_MODEL"), "loaded": False,
                               "edit_model": "flux2-klein-4b", "edit_loaded": False, "cache_limit_gb": 2.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    # A progress bar: forty redraws in place, then the finished bar on its own line.
    for step in range(40):
        sys.stderr.write("\\r%3d%%|bar| %d/40" % (step * 100 // 40, step))
        sys.stderr.flush()
        time.sleep(0.002)
    sys.stderr.write("\\r100%|bar| 40/40\\n")
    sys.stderr.write("\\x1b[32mINFO\\x1b[0m ready\\n")
    sys.stderr.flush()
    server.serve_forever()
    """)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def project(tmp_path):
    """A project folder with a config, and a runner whose server is the fake one."""
    port = free_port()
    (tmp_path / "fake_flux.py").write_text(FAKE_SERVER, encoding="utf-8")
    (tmp_path / ".env-stable").write_text(
        f"FLUX_ENABLED=true\nFLUX_MODEL=klein\nFLUX_SERVER_URL=http://127.0.0.1:{port}\n", encoding="utf-8"
    )
    heard: list[str] = []

    async def on_output(line: str) -> None:
        heard.append(line)

    def runner(mode="serve"):
        command = [sys.executable, "-u", str(tmp_path / "fake_flux.py"), str(port), mode]
        return FluxRunner(tmp_path, on_output, command=command)

    return {"root": tmp_path, "port": port, "heard": heard, "runner": runner}


async def until(check, *, seconds=10.0):
    """Wait for ``check()`` (sync or async) to return something true."""
    deadline = asyncio.get_running_loop().time() + seconds
    while True:
        value = check()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return value
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.05)


def test_start_status_log_and_stop(project):
    async def scenario():
        runner = project["runner"]()
        before = await runner.status()
        ok, message = await runner.start()
        assert (ok, message) == (True, "Flux server starting")
        await until(runner.health)
        await until(lambda: any("ready" in line for line in runner.tail()))
        running = await runner.status()
        assert (await runner.start())[0] is False  # one at a time
        ok, message = await runner.stop()
        after = await runner.status()
        return before, running, (ok, message), after, runner.tail()

    before, running, stopped, after, tail = asyncio.run(scenario())
    assert (before["enabled"], before["running"], before["reachable"], before["pid"]) == (True, False, False, None)
    assert (running["running"], running["reachable"], running["external"]) == (True, True, False)
    assert running["pid"] and running["start_time"] and running["url"] == f"http://127.0.0.1:{project['port']}"
    # FLUX_MODEL reached the child from .env-stable, and came back through /health.
    assert (running["model"], running["loaded"], running["cache_limit_gb"]) == ("klein", False, 2.0)
    assert stopped == (True, "Flux server stopped")
    assert (after["running"], after["reachable"], after["pid"]) == (False, False, None)
    # A stop the panel asked for is not shown as a crash.
    assert after["stopped_by_panel"] is True and running["stopped_by_panel"] is False

    assert tail[0].startswith("[web] Started Flux server pid=")
    assert tail[1] == f"Starting Flux server on 127.0.0.1:{project['port']} (model=klein)"
    assert "INFO ready" in tail  # colour codes stripped
    assert tail[-1].startswith("[web] Flux server exited with code")
    # The live log gets the server's lines marked as the Flux server's.
    assert "[flux] INFO ready" in project["heard"] and project["heard"][0].startswith("[web] Started")
    # Kept on disk, where the terminal's scrollback used to be the only copy.
    saved = (project["root"] / "logs" / "flux_server.log").read_text(encoding="utf-8").splitlines()
    assert saved == tail


def test_a_progress_bar_is_thinned_out_and_its_last_frame_kept(project):
    async def scenario():
        runner = project["runner"]()
        await runner.start()
        await until(lambda: any("ready" in line for line in runner.tail()))
        await runner.stop()
        return runner.tail()

    bars = [line for line in asyncio.run(scenario()) if "|bar|" in line]
    assert bars[-1] == "100%|bar| 40/40"
    assert len(bars) <= 3  # forty redraws in well under a second


def test_a_server_started_outside_the_panel_is_reported_and_left_alone(project):
    async def scenario():
        async def health(_request):
            return web.json_response({"status": "ok", "model": "schnell", "loaded": True})

        outside = web.Application()
        outside.router.add_get("/health", health)
        async with TestServer(outside, port=project["port"]):
            runner = project["runner"]()
            status = await runner.status()
            return status, await runner.start(), await runner.stop()

    status, started, stopped = asyncio.run(scenario())
    assert (status["reachable"], status["external"], status["running"], status["model"]) == (
        True,
        True,
        False,
        "schnell",
    )
    assert started[0] is False and "outside the panel" in started[1]
    assert stopped[0] is False and "outside the panel" in stopped[1]


def test_a_server_that_dies_at_start_leaves_its_reason_in_the_log(project):
    async def scenario():
        runner = project["runner"]("crash")
        ok, _ = await runner.start()
        await until(lambda: any("exited with code" in line for line in runner.tail()))
        return ok, await runner.status(), runner.tail(), await runner.stop()

    ok, status, tail, stopped = asyncio.run(scenario())
    assert ok is True  # it did start; the status and the log say what happened next
    assert (status["running"], status["returncode"], status["reachable"]) == (False, 3, False)
    assert status["stopped_by_panel"] is False
    assert "OSError: [Errno 48] Address already in use" in tail
    assert tail[-1] == "[web] Flux server exited with code 3"
    assert stopped == (False, "Flux server is not running")


def test_restart_gives_a_new_process(project):
    async def scenario():
        runner = project["runner"]()
        await runner.start()
        await until(runner.health)
        first = (await runner.status())["pid"]
        ok, _ = await runner.restart()
        await until(runner.health)
        second = (await runner.status())["pid"]
        await runner.stop()
        return ok, first, second

    ok, first, second = asyncio.run(scenario())
    assert ok and first and second and first != second


@pytest.mark.parametrize(
    "config, enabled, autostart",
    [
        ("FLUX_ENABLED=true\nFLUX_SERVER_AUTOSTART=true\n", True, True),
        ("FLUX_ENABLED=true\n", True, False),
        ("FLUX_ENABLED=true\nFLUX_SERVER_AUTOSTART=false\n", True, False),
        # Autostart alone does nothing: Flux is off.
        ("FLUX_ENABLED=false\nFLUX_SERVER_AUTOSTART=true\n", False, False),
        ("", False, False),
    ],
)
def test_autostart_needs_flux_on_and_the_switch(tmp_path, config, enabled, autostart):
    (tmp_path / ".env-stable").write_text(config, encoding="utf-8")

    async def nothing(_line):
        return None

    runner = FluxRunner(tmp_path, nothing)
    assert (runner.enabled(), runner.autostart_wanted()) == (enabled, autostart)


def test_the_url_comes_from_the_config(tmp_path):
    async def nothing(_line):
        return None

    runner = FluxRunner(tmp_path, nothing)
    assert runner.base_url() == "http://127.0.0.1:4942"  # no config at all
    (tmp_path / ".env-stable").write_text("FLUX_SERVER_HOST=0.0.0.0\nFLUX_SERVER_PORT=5000\n", encoding="utf-8")
    assert runner.base_url() == "http://0.0.0.0:5000"
    (tmp_path / ".env-stable").write_text("FLUX_SERVER_URL=http://10.0.0.5:4942/\n", encoding="utf-8")
    assert runner.base_url() == "http://10.0.0.5:4942"
