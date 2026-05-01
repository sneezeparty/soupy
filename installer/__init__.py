"""Soupy interactive installer.

Stdlib-only at import time. Steps may lazily import third-party packages
after the dependency-install step runs, but the top-level package itself
must not.
"""

__all__ = ["state", "ui", "http", "validators", "env_writer", "steps"]
