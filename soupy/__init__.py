"""Soupy bot library — typed settings, prompt loading, logging, trigger
predicates, and the discord.py cogs that hang off them.

The bot's chat-flow entry point (``soupy_remastered_stablediffusion.py``)
stays at the repo root for two reasons: it's the script that gets
launched by ``run_all.py``, and the message-respond pipeline is the
single most behaviour-sensitive surface in the codebase. Everything
else lives here.
"""
