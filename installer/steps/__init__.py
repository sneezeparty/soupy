"""Ordered list of wizard steps.

Each step exposes `run(state, ui) -> dict` returning the keys to merge
back into the wizard's running state. Step IDs are used as the key in the
state file's `completed_steps` list — keep them stable.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from . import (
    s01_goals,
    s02_environment,
    s03_discord,
    s04_lm_studio,
    s05_optional,
    s06_write_config,
    s07_verify,
    s08_handoff,
)


StepFn = Callable[[Dict, "ui.UI"], Dict]  # type: ignore[name-defined]

STEPS: List[Tuple[str, str, StepFn]] = [
    ("goals", "Goals", s01_goals.run),
    ("environment", "Environment & dependencies", s02_environment.run),
    ("discord", "Discord app", s03_discord.run),
    ("lm_studio", "LM Studio", s04_lm_studio.run),
    ("optional", "Optional integrations", s05_optional.run),
    ("write_config", "Write .env-stable", s06_write_config.run),
    ("verify", "Verification", s07_verify.run),
    ("handoff", "Handoff", s08_handoff.run),
]
