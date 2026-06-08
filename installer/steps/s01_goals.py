"""Step 1 — Goals.

Establish which downstream paths the wizard needs to walk:
- image generation: none / on this machine (CUDA / MPS) / on a different machine
- bluesky integration: yes / no
- vision (image descriptions): yes / no
- autonomous daily article posts: yes / no
"""

from __future__ import annotations

from typing import Dict

IMAGE_GEN_OPTIONS = [
    ("none", "no image generation"),
    ("local_cuda", "image gen on this machine, NVIDIA CUDA (Stable Diffusion)"),
    ("local_flux", "image gen on this machine, Apple Silicon (Flux / mflux) — recommended on a Mac"),
    ("local_mps", "image gen on this machine, Apple Silicon (Stable Diffusion via MPS)"),
    ("remote", "image gen on a different machine (Stable Diffusion)"),
]


def run(state: Dict, ui) -> Dict:
    ui.header("Step 1 — Goals")
    ui.info("Pick what you want Soupy to do. The wizard skips prompts you don't need.")
    ui.hr()

    image_gen_mode = ui.prompt_choice(
        "Image generation:",
        IMAGE_GEN_OPTIONS,
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
