"""discord.py extensions. Each module in this directory exposes an
async ``setup(bot)`` function and is loaded by name from
``soupy_remastered_stablediffusion.py::load_extensions``.

The dotted load path is ``soupy.cogs.<name>`` — e.g.
``bot.load_extension("soupy.cogs.search")``.
"""
