"""Guards the image-gen extraction (soupy/cogs/sd.py).

The SD subsystem was moved out of the main bot file into a cog, leaving the
shared ``SDQueue`` behind to dispatch into it via a lazy import. This test loads
the cog onto the real bot and asserts:

* the four slash commands register on the tree,
* every function ``SDQueue.process_queue`` dispatches to exists on the module,
* the lazy ``from soupy.cogs import sd`` import the queue relies on works.

It can't exercise real image generation (needs a live Discord + SD backend), but
it catches the failure modes a blind refactor introduces: missing imports,
unregistered commands, and dispatch names that drifted.
"""

from __future__ import annotations

import asyncio
import sys

import soupy_remastered_stablediffusion as bot_module  # noqa: E402 (conftest sets env first)

# Names SDQueue.process_queue dispatches to (must exist on the cog module).
_DISPATCH_FUNCS = [
    "process_sd_image",
    "generate_sd_image",
    "handle_outpaint",
    "handle_random",
    "handle_remix",
    "handle_fancy",
    "handle_wide",
    "handle_tall",
    "handle_edit",
    "handle_2x2_grid",
    "handle_thumbnail_upscale",
    "handle_regenerate_selected",
]


def _ensure_loaded():
    async def _load():
        if "soupy.cogs.sd" not in bot_module.bot.extensions:
            await bot_module.bot.load_extension("soupy.cogs.sd")

    asyncio.run(_load())
    return sys.modules["soupy.cogs.sd"]


def test_sd_extension_loads_and_registers_commands():
    _ensure_loaded()
    names = {c.name for c in bot_module.bot.tree.get_commands()}
    assert {"sd", "img2img", "inpaint", "outpaint"} <= names


def test_dispatch_targets_exist_on_cog():
    sd = _ensure_loaded()
    missing = [fn for fn in _DISPATCH_FUNCS if not hasattr(sd, fn)]
    assert not missing, f"SDQueue dispatch references missing from cog: {missing}"


def test_lazy_import_used_by_queue_resolves():
    _ensure_loaded()
    # This is exactly what SDQueue.process_queue does at runtime.
    from soupy.cogs import sd as _sd

    assert hasattr(_sd, "generate_sd_image")


def test_cog_binds_to_the_live_bot_instance():
    """The cog's `bot` must be the SAME object the running bot uses.

    Regression guard for the __main__ double-import trap: the bot runs as
    `python soupy_remastered_stablediffusion.py` (module name "__main__"), so a
    by-name `from soupy_remastered_stablediffusion import bot` would bind a
    DUPLICATE bot whose queue nothing drains — /sd would log "queued" then hang.
    The cog resolves the live module instead. If this identity ever breaks, image
    commands silently stop working.
    """
    sd = _ensure_loaded()
    assert sd.bot is bot_module.bot
