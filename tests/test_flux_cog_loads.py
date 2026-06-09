"""Guards the local Flux cog (soupy/cogs/flux.py).

Mirrors test_sd_cog_loads.py: the `/flux` command is dispatched through the same
shared ``SDQueue`` (``type:"flux"``) via a lazy import, and the cog binds shared
helpers off the LIVE main module using the same __main__-resolution trick as the
sd cog. This test loads the cog onto the real bot and asserts:

* the `/flux` slash command registers on the tree,
* the queue's dispatch target (`process_flux_image`) exists,
* the lazy ``from soupy.cogs import flux`` import the queue relies on works,
* the cog binds to the live bot instance (not a duplicate via by-name import).

It can't exercise real generation (needs a live Discord + the mflux server).
"""

from __future__ import annotations

import asyncio
import sys

import soupy_remastered_stablediffusion as bot_module  # noqa: E402 (conftest sets env first)


def _ensure_loaded():
    async def _load():
        # sd loads first in production (flux reuses sd.archive_image_bytes).
        if "soupy.cogs.sd" not in bot_module.bot.extensions:
            await bot_module.bot.load_extension("soupy.cogs.sd")
        if "soupy.cogs.flux" not in bot_module.bot.extensions:
            await bot_module.bot.load_extension("soupy.cogs.flux")

    asyncio.run(_load())
    return sys.modules["soupy.cogs.flux"]


def test_flux_extension_loads_and_registers_command():
    _ensure_loaded()
    names = {c.name for c in bot_module.bot.tree.get_commands()}
    assert "flux" in names


def test_queue_dispatch_target_exists_on_cog():
    flux = _ensure_loaded()
    for fn in ["process_flux_image", "generate_flux_image",
               "_handle_flux_fancy", "_handle_flux_random"]:
        assert hasattr(flux, fn), f"missing {fn}"


def test_views_and_modals_present():
    flux = _ensure_loaded()
    for cls in ["FluxRemixView", "FluxEditModal", "FluxImg2ImgModal"]:
        assert hasattr(flux, cls), f"missing {cls}"


def test_img2img_button_requires_display_source():
    """The 🖼️ img2img button appears only when a persisted display source exists."""
    flux = _ensure_loaded()

    def has_img2img(view):
        return any(getattr(c, "custom_id", None) == "fluxgen_img2img_button" for c in view.children)

    no_disp = flux.FluxRemixView(prompt="x", width=1024, height=1024, seed=1)
    with_disp = flux.FluxRemixView(prompt="x", width=1024, height=1024, seed=1, display_source_file="out.png")
    assert not has_img2img(no_disp), "view without a display source must NOT show img2img"
    assert has_img2img(with_disp), "view with a display source SHOULD show img2img"


def test_flux_edit_url_helper_derives_from_base():
    """_flux_edit_url() derives the edit endpoint from FLUX_SERVER_URL when no override is set."""
    flux = _ensure_loaded()
    from soupy.settings import settings

    # In the test env FLUX_EDIT_URL is unset, so the helper should append /flux_edit
    # to the base server URL. (When FLUX_EDIT_URL is set, the helper returns it
    # verbatim; that path is exercised by an explicit-override deployment.)
    assert flux._flux_edit_url().endswith("/flux_edit")
    if settings.flux_server_url:
        assert flux._flux_edit_url().startswith(settings.flux_server_url.rstrip("/"))


def test_prompt_builders_shared_from_sd():
    """The Flux Fancy/Random handlers reuse the sd cog's prompt builders."""
    _ensure_loaded()
    from soupy.cogs import sd as _sd

    assert hasattr(_sd, "build_fancy_prompt")
    assert hasattr(_sd, "build_random_prompt")
    assert hasattr(_sd, "build_flux_fancy_prompt")
    assert hasattr(_sd, "build_flux_random_prompt")


def test_lazy_import_used_by_queue_resolves():
    _ensure_loaded()
    # Exactly what SDQueue.process_queue does for type:"flux".
    from soupy.cogs import flux as _flux

    assert hasattr(_flux, "process_flux_image")


def test_cog_binds_to_the_live_bot_instance():
    flux = _ensure_loaded()
    assert flux.bot is bot_module.bot
