"""Step 1 — Goals.

Establish which downstream paths the wizard needs to walk:
- image generation: none / on this machine (CUDA / Flux / MPS) / on a different machine
- bluesky integration: yes / no
- vision (image descriptions): yes / no
- autonomous daily article posts: yes / no

Only the image-generation backends this machine can run are offered.
"""

from __future__ import annotations

import platform
import sys
from typing import Dict, List, Tuple

IMAGE_GEN_OPTIONS = [
    ("none", "no image generation"),
    ("local_cuda", "image gen on this machine, NVIDIA CUDA (Stable Diffusion)"),
    ("local_flux", "image gen on this machine, Apple Silicon (Flux / mflux) — recommended on a Mac"),
    ("local_mps", "image gen on this machine, Apple Silicon (Stable Diffusion via MPS)"),
    ("remote", "image gen on a different machine (Stable Diffusion)"),
]

# Step 2 pins torch 2.4.0+cu118 for the CUDA backend, and PyTorch published
# that build for nothing newer than this.
CUDA_MAX_PYTHON = (3, 12)


def image_gen_options(
    os_name: str,
    machine: str,
    python: Tuple[int, int],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """The image-generation choices this machine can run, plus a note for each
    one left out for a reason the user can do something about."""
    apple_silicon = os_name == "darwin" and machine == "arm64"
    hidden = set()
    notes: List[str] = []
    if not apple_silicon:
        hidden.update({"local_flux", "local_mps"})
    if os_name == "darwin":
        hidden.add("local_cuda")
    elif python > CUDA_MAX_PYTHON:
        hidden.add("local_cuda")
        notes.append(
            "NVIDIA CUDA image generation is not offered: its PyTorch build stops at "
            f"Python {CUDA_MAX_PYTHON[0]}.{CUDA_MAX_PYTHON[1]} and this is Python {python[0]}.{python[1]}. "
            f"Run the wizard with python{CUDA_MAX_PYTHON[0]}.{CUDA_MAX_PYTHON[1]} to use it."
        )
    return [opt for opt in IMAGE_GEN_OPTIONS if opt[0] not in hidden], notes


def run(state: Dict, ui) -> Dict:
    ui.header("Step 1 — Goals")
    ui.info("Pick what you want Soupy to do. The wizard skips prompts you don't need.")
    ui.hr()

    options, notes = image_gen_options(sys.platform, platform.machine(), sys.version_info[:2])
    for note in notes:
        ui.info(note)
    image_gen_mode = ui.prompt_choice(
        "Image generation:",
        options,
        default=state.get("image_gen_mode", "none"),
    )
    bluesky = ui.confirm(
        "Bluesky integration?",
        default=bool(state.get("bluesky", False)),
    )
    vision = ui.confirm(
        "Enable vision (Soupy describes images users attach in chat)?",
        default=bool(state.get("vision", False)),
    )
    daily_posts = ui.confirm(
        "Enable autonomous daily article posts?",
        default=bool(state.get("daily_posts", False)),
    )

    return {
        "image_gen_mode": image_gen_mode,
        "bluesky": bluesky,
        "vision": vision,
        "daily_posts": daily_posts,
    }
