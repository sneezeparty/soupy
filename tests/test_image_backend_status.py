"""The bot's report of whether its image servers answer, which the panel's status dots show.

``sd_server_online`` used to be set to ``True`` at startup and never touched
again, so the panel's SD dot was green with the server switched off, and
nothing reported Flux at all.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import soupy_remastered_stablediffusion as bot_module  # noqa: E402 (conftest sets env first)
from soupy.settings import settings


@pytest.fixture
def backends(monkeypatch):
    """Script what each health check answers. Returns the two answer queues."""
    answers = {"flux": [], "sd": []}

    async def flux():
        return answers["flux"].pop(0)

    async def sd(quiet=False):
        assert quiet, "the background check must not log a dead server every minute"
        return answers["sd"].pop(0)

    monkeypatch.setattr(bot_module, "check_flux_server_status", flux)
    monkeypatch.setattr(bot_module, "check_sd_server_status", sd)
    monkeypatch.setattr(bot_module, "flux_server_online", None)
    monkeypatch.setattr(bot_module, "sd_server_online", None)
    monkeypatch.setattr(bot_module, "_flux_check_failures", 0)
    monkeypatch.setitem(settings.__dict__, "flux_enabled", True)
    return answers


def refresh(**kwargs):
    asyncio.run(bot_module.refresh_image_backend_status(**kwargs))
    return bot_module.flux_server_online, bot_module.sd_server_online


def test_both_servers_are_reported_as_they_answer(backends):
    backends["flux"] += [True, False]
    backends["sd"] += [False, True]
    assert refresh() == (True, False)
    backends["flux"][:] = [True]
    assert refresh() == (True, True)


def test_a_server_that_was_never_up_is_down_at_once(backends):
    backends["flux"] += [False]
    backends["sd"] += [False]
    assert refresh() == (False, False)


def test_one_missed_flux_check_does_not_turn_the_dot_red(backends):
    """The server answers /health slowly while it generates."""
    backends["flux"] += [True, False, True, False, False, True]
    assert refresh(check_sd=False)[0] is True
    assert refresh(check_sd=False)[0] is True  # one miss
    assert refresh(check_sd=False)[0] is True  # answered again
    assert refresh(check_sd=False)[0] is True  # one miss
    assert refresh(check_sd=False)[0] is False  # two in a row
    assert refresh(check_sd=False)[0] is True


def test_sd_is_only_asked_when_told_to(backends):
    backends["flux"] += [True, True]
    backends["sd"] += [True]
    assert refresh(check_sd=False) == (True, None)
    assert refresh(check_sd=True) == (True, True)
    assert backends["sd"] == []


def test_with_flux_off_it_is_not_checked_or_reported(backends):
    settings.__dict__["flux_enabled"] = False
    backends["sd"] += [True]
    assert refresh() == (None, True)
    assert backends["flux"] == []


def test_the_health_check_reads_the_servers_own_answer():
    async def scenario():
        state = {"body": {"status": "ok"}, "status": 200}

        async def health(_request):
            return web.json_response(state["body"], status=state["status"])

        app = web.Application()
        app.router.add_get("/health", health)
        async with TestServer(app) as server:
            base = str(server.make_url("")).rstrip("/")
            up = await bot_module._health_ok(base + "/")
            state["body"] = {"status": "loading"}
            loading = await bot_module._health_ok(base)
            state["status"] = 500
            broken = await bot_module._health_ok(base)
        return up, loading, broken

    assert asyncio.run(scenario()) == (True, False, False)


def test_an_unreachable_flux_server_is_just_down(monkeypatch):
    monkeypatch.setitem(settings.__dict__, "flux_server_url", "http://127.0.0.1:9")
    assert asyncio.run(bot_module.check_flux_server_status()) is False
    monkeypatch.setitem(settings.__dict__, "flux_server_url", "")
    assert asyncio.run(bot_module.check_flux_server_status()) is False
