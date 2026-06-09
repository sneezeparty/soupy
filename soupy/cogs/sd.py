"""Stable Diffusion image-generation cog.

Extracted from the main bot file (``soupy_remastered_stablediffusion.py``). Owns
the ``/sd``, ``/img2img``, ``/inpaint`` and ``/outpaint`` commands, the
Remix / Edit / Outpaint button UIs (``SDRemixView``, ``EditImageModal``), and
the SD HTTP pipeline (``generate_sd_image``).

WHY the queue isn't here: ``SDQueue`` stays in the main module because the same
queue also dispatches ``"chat"`` jobs and must be running before the first
``on_message`` (which fires before cogs finish loading). ``SDQueue.process_queue``
reaches the functions in this module via a lazy ``from soupy.cogs import sd``
import, so there is no import cycle at module load.

The slash commands are ``@app_commands.command`` objects registered on the tree
in ``setup()`` (when ``load_extensions()`` calls
``bot.load_extension("soupy.cogs.sd")`` at ``on_ready``, before the tree is
synced). Shared helpers and SD config constants are imported from the main module
— image generation stays intentionally coupled to them.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import sys
import time
from io import BytesIO
from typing import Optional

import aiohttp
import cv2
import discord
import numpy as np
from aiohttp import ClientConnectorError, ClientOSError, ServerTimeoutError
from discord import app_commands
from discord.ui import Modal, TextInput, View
from PIL import Image, ImageDraw

from soupy import prompts as soupy_prompts

# --- Resolve the LIVE main module ----------------------------------------------
# WHY this isn't a plain `from soupy_remastered_stablediffusion import ...`:
# the bot is launched as `python soupy_remastered_stablediffusion.py`, so its
# module name at runtime is "__main__". A by-name import would re-execute the
# file as a SECOND module with its own `bot`/`SDQueue` — the slash commands here
# would then enqueue onto a duplicate queue that nothing drains, and /sd would
# silently do nothing. So bind the shared symbols off whichever module object is
# actually running the bot: the script's __main__ if present, else the by-name
# module (tests / the web smoke import, where __main__ is pytest/uvicorn).
_mm = sys.modules.get("__main__")
if _mm is not None and hasattr(_mm, "SDQueue"):
    _main = _mm
else:  # imported as a module (tests, tooling) rather than run as the script
    import soupy_remastered_stablediffusion as _main  # noqa: E402

ARTISTIC_RENDERING_STYLES = _main.ARTISTIC_RENDERING_STYLES
CHARACTER_CONCEPTS = _main.CHARACTER_CONCEPTS
OUTPAINT_CONTROL_WEIGHT = _main.OUTPAINT_CONTROL_WEIGHT
OUTPAINT_HARMONIZE_STRENGTH = _main.OUTPAINT_HARMONIZE_STRENGTH
OUTPAINT_USE_CANNY = _main.OUTPAINT_USE_CANNY
OUTPAINT_USE_DEPTH = _main.OUTPAINT_USE_DEPTH
OVERALL_THEMES = _main.OVERALL_THEMES
RANDOMPROMPT = _main.RANDOMPROMPT
SD_DEFAULT_HEIGHT = _main.SD_DEFAULT_HEIGHT
SD_DEFAULT_WIDTH = _main.SD_DEFAULT_WIDTH
SD_IMG2IMG_URL = _main.SD_IMG2IMG_URL
SD_INPAINT_URL = _main.SD_INPAINT_URL
SD_KEYWORDS_LIST = _main.SD_KEYWORDS_LIST
SD_OUTPAINT_HYBRID_URL = _main.SD_OUTPAINT_HYBRID_URL
SD_SERVER_URL = _main.SD_SERVER_URL
SD_TALL_HEIGHT = _main.SD_TALL_HEIGHT
SD_TALL_WIDTH = _main.SD_TALL_WIDTH
SD_WIDE_HEIGHT = _main.SD_WIDE_HEIGHT
SD_WIDE_WIDTH = _main.SD_WIDE_WIDTH
_ensure_media_dirs = _main._ensure_media_dirs
archive_sent_message = _main.archive_sent_message
async_chat_completion = _main.async_chat_completion
bot = _main.bot
format_messages = _main.format_messages
increment_user_stat = _main.increment_user_stat
logger = _main.logger
universal_cooldown_check = _main.universal_cooldown_check


def get_random_terms():
    terms = {}

    if OVERALL_THEMES:
        num_themes = random.randint(1, 3)
        chosen_themes = random.sample(OVERALL_THEMES, num_themes)
        terms["Overall Theme"] = ", ".join(chosen_themes)

    if CHARACTER_CONCEPTS:
        rand_val = random.random()
        if rand_val < 0.05:  # 5% chance of no character
            pass  # Skip adding a character
        elif rand_val < 0.25:  # 20% chance of Grey Sphynx Cat (0.05 to 0.25)
            terms["Character Concept"] = "Grey Sphynx Cat"
        else:  # 75% chance of random character
            terms["Character Concept"] = random.choice(CHARACTER_CONCEPTS)

    if ARTISTIC_RENDERING_STYLES:
        # Randomly decide how many styles to pick (1-4)
        num_styles = random.randint(1, 4)
        # Get random styles without repeats
        chosen_styles = random.sample(ARTISTIC_RENDERING_STYLES, num_styles)
        terms["Artistic Rendering Style"] = ", ".join(chosen_styles)

    # Always include a handful of SD-specific keywords if available
    if SD_KEYWORDS_LIST:
        num_sd = min(4, len(SD_KEYWORDS_LIST))
        chosen_sd = random.sample(SD_KEYWORDS_LIST, num_sd)
        terms["SD Keywords"] = ", ".join(chosen_sd)

    return terms


async def handle_random(interaction, width, height, queue_size, direct_prompt=None):
    """
    Handles the generation of a random image by selecting random terms from categories
    and combining them with the base random prompt.

    Args:
        interaction: The Discord interaction
        width: Image width
        height: Image height
        queue_size: Current size of the queue
        direct_prompt: Optional direct prompt to use (for terms-only mode)
    """
    try:
        # Start timing for prompt generation
        prompt_start_time = time.perf_counter()
        selected_terms_str = None

        # Show typing indicator in the channel
        async with interaction.channel.typing():
            # Check if we have a direct prompt (terms-only mode)
            if direct_prompt:
                random_prompt = direct_prompt
                selected_terms_str = direct_prompt  # The terms are the prompt in this case
                logger.info(f"🔀 Using direct terms as prompt for {interaction.user}: {random_prompt}")
                # End timing for direct prompt case
                prompt_end_time = time.perf_counter()
                prompt_duration = prompt_end_time - prompt_start_time
            else:
                # Original random prompt generation logic
                if not RANDOMPROMPT:
                    if not interaction.response.is_done():
                        await interaction.response.send_message("❌ No RANDOMPROMPT found in .env.", ephemeral=True)
                    else:
                        await interaction.followup.send("❌ No RANDOMPROMPT found in .env.", ephemeral=True)
                    return

                # Get random terms first
                random_terms = get_random_terms()
                formatted_descriptors = "\n".join(
                    [f"**{category}:** {term}" for category, term in random_terms.items()]
                )
                logger.info(f"🔀 Selected Descriptors for {interaction.user}:\n{formatted_descriptors}")

                # Combine with base prompt, but emphasize artistic style
                art_style = random_terms.get("Artistic Rendering Style", "")
                other_terms = [
                    term for category, term in random_terms.items() if category != "Artistic Rendering Style"
                ]

                # Create a more detailed artistic style instruction
                style_emphasis = (
                    f"The image should be rendered combining these artistic styles: {art_style}. "
                    f"These artistic styles should be the dominant visual characteristics, "
                    f"blended together, with the following elements incorporated within these styles: {', '.join(other_terms)}"
                )

                # If SD keywords exist, append a clear instruction to leverage them
                sd_hint = ""
                if "SD Keywords" in random_terms:
                    sd_hint = f"\nFocus on these rendering/photographic cues as global style constraints: {random_terms['SD Keywords']}."
                combined_prompt = f"{RANDOMPROMPT} {style_emphasis}{sd_hint}"
                logger.info(f"🔀 Combined Prompt for {interaction.user}:\n{combined_prompt}")

                # Now send to LLM with modified system message
                system_msg = {
                    "role": "system",
                    "content": "You are an assistant that creates image prompts with strong emphasis on artistic style. "
                    "The artistic rendering style should be prominently featured in your prompt, affecting every element described.",
                }
                user_msg = {"role": "user", "content": combined_prompt}
                messages_for_llm = [system_msg, user_msg]

                # Add logging for the messages being sent to LLM
                formatted_messages = format_messages(messages_for_llm)
                logger.debug(f"📜 Sending the following messages to LLM for random prompt:\n{formatted_messages}")

                response = await async_chat_completion(
                    model=os.getenv("LOCAL_CHAT"),
                    messages=messages_for_llm,
                    temperature=float(os.getenv("RANDOM_PROMPT_TEMPERATURE", 0.8)),
                    max_tokens=325,
                )
                random_prompt = response.choices[0].message.content.strip()
                logger.info(f"🔀 Generated random prompt for {interaction.user}: {random_prompt}")

                # Capture the randomly chosen terms as a comma-separated string
                # Flatten the terms from the dictionary
                selected_terms_list = []
                for _category, terms in random_terms.items():
                    # Split by comma in case there are multiple terms in a single category
                    split_terms = [term.strip() for term in terms.split(",")]
                    selected_terms_list.extend(split_terms)
                selected_terms_str = ", ".join(selected_terms_list)

                # End timing for LLM prompt generation
                prompt_end_time = time.perf_counter()
                prompt_duration = prompt_end_time - prompt_start_time

        # Generate new seed for both direct and LLM-generated prompts
        new_seed = random.randint(0, 2**32 - 1)

        # Use generate_sd_image for both direct and LLM-generated prompts
        await generate_sd_image(
            interaction=interaction,
            prompt=random_prompt,
            width=width,
            height=height,
            seed=new_seed,
            action_name="Random",
            queue_size=queue_size,
            pre_duration=prompt_duration,
            selected_terms=selected_terms_str,
        )

        await increment_user_stat(interaction.user.id, "images_generated")

    except Exception as e:
        logger.error(f"🔀 Error generating random prompt for {interaction.user}: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message(f"❌ Error generating random prompt: {e}", ephemeral=True)
        else:
            await interaction.followup.send(f"❌ Error generating random prompt: {e}", ephemeral=True)

async def build_random_prompt(direct_prompt: Optional[str] = None):
    """Build a random image prompt. Returns ``(prompt, selected_terms_str, duration)``.

    Shared by ``handle_random`` (the SD R-Fancy / R-Keyword buttons) and the Flux
    cog's equivalents. With ``direct_prompt`` set it's terms-only (R-Keyword);
    otherwise it asks the LLM to elaborate random terms (R-Fancy). Raises
    ``RuntimeError`` when no ``RANDOMPROMPT`` is configured (non-direct path).
    Mirror of the inline logic in ``handle_random`` — keep the two in sync.
    """
    start = time.perf_counter()
    if direct_prompt:
        return direct_prompt, direct_prompt, time.perf_counter() - start

    if not RANDOMPROMPT:
        raise RuntimeError("No RANDOMPROMPT found in .env.")

    random_terms = get_random_terms()
    art_style = random_terms.get("Artistic Rendering Style", "")
    other_terms = [term for category, term in random_terms.items() if category != "Artistic Rendering Style"]
    style_emphasis = (
        f"The image should be rendered combining these artistic styles: {art_style}. "
        f"These artistic styles should be the dominant visual characteristics, "
        f"blended together, with the following elements incorporated within these styles: {', '.join(other_terms)}"
    )
    sd_hint = ""
    if "SD Keywords" in random_terms:
        sd_hint = f"\nFocus on these rendering/photographic cues as global style constraints: {random_terms['SD Keywords']}."
    combined_prompt = f"{RANDOMPROMPT} {style_emphasis}{sd_hint}"

    messages_for_llm = [
        {
            "role": "system",
            "content": "You are an assistant that creates image prompts with strong emphasis on artistic style. "
            "The artistic rendering style should be prominently featured in your prompt, affecting every element described.",
        },
        {"role": "user", "content": combined_prompt},
    ]
    response = await async_chat_completion(
        model=os.getenv("LOCAL_CHAT"),
        messages=messages_for_llm,
        temperature=float(os.getenv("RANDOM_PROMPT_TEMPERATURE", 0.8)),
        max_tokens=325,
    )
    random_prompt = response.choices[0].message.content.strip()

    selected_terms_list = []
    for _category, terms in random_terms.items():
        selected_terms_list.extend([term.strip() for term in terms.split(",")])
    selected_terms_str = ", ".join(selected_terms_list)
    return random_prompt, selected_terms_str, time.perf_counter() - start


async def build_flux_random_prompt(direct_prompt: Optional[str] = None):
    """Flux 2-tuned variant of :func:`build_random_prompt`.

    Loads ``prompts/randomprompt_flux*`` (falling back to ``RANDOMPROMPT``
    if no Flux variant is configured) and respects ``FLUX_RANDOM_MAX_TOKENS``
    (default 140) so the rewritten prompt stays a tight natural-language
    paragraph. Same return signature as ``build_random_prompt``.
    """
    start = time.perf_counter()
    if direct_prompt:
        return direct_prompt, direct_prompt, time.perf_counter() - start

    flux_randomprompt = soupy_prompts.load_prompt("randomprompt_flux", fallback=RANDOMPROMPT or "")
    if not flux_randomprompt:
        raise RuntimeError("No RANDOMPROMPT (or randomprompt_flux) found.")

    random_terms = get_random_terms()
    art_style = random_terms.get("Artistic Rendering Style", "")
    other_terms = [term for category, term in random_terms.items() if category != "Artistic Rendering Style"]
    style_emphasis = (
        f"The image should be rendered combining these artistic styles: {art_style}. "
        f"These artistic styles should be the dominant visual characteristics, "
        f"blended together, with the following elements incorporated within these styles: {', '.join(other_terms)}"
    )
    sd_hint = ""
    if "SD Keywords" in random_terms:
        sd_hint = f"\nFocus on these rendering/photographic cues as global style constraints: {random_terms['SD Keywords']}."
    combined_prompt = f"{flux_randomprompt} {style_emphasis}{sd_hint}"

    messages_for_llm = [
        {
            "role": "system",
            "content": "You are an assistant that creates image prompts with strong emphasis on artistic style. "
            "The artistic rendering style should be prominently featured in your prompt, affecting every element described.",
        },
        {"role": "user", "content": combined_prompt},
    ]
    response = await async_chat_completion(
        model=os.getenv("LOCAL_CHAT"),
        messages=messages_for_llm,
        temperature=float(os.getenv("RANDOM_PROMPT_TEMPERATURE", 0.8)),
        max_tokens=int(os.getenv("FLUX_RANDOM_MAX_TOKENS", 140)),
    )
    random_prompt = response.choices[0].message.content.strip()

    selected_terms_list = []
    for _category, terms in random_terms.items():
        selected_terms_list.extend([term.strip() for term in terms.split(",")])
    selected_terms_str = ", ".join(selected_terms_list)
    return random_prompt, selected_terms_str, time.perf_counter() - start


async def build_fancy_prompt(prompt: str):
    """LLM-rewrite ``prompt`` into a more elaborate "fancy" version.

    Returns ``(cleaned_prompt, duration)``. Used by the SD Fancy button path.
    Mirror of the inline logic in ``handle_fancy`` — keep in sync. The Flux
    Fancy button uses :func:`build_flux_fancy_prompt` instead.
    """
    fancy_instructions = soupy_prompts.load_prompt("fancy", fallback="")
    combined_instructions = f"{fancy_instructions}\n\nThe prompt you are elaborating on is: {prompt}"
    start = time.perf_counter()
    messages = [
        {"role": "system", "content": combined_instructions},
        {"role": "user", "content": "Please rewrite the above prompt accordingly."},
    ]
    response = await async_chat_completion(
        model=os.getenv("LOCAL_CHAT"),
        messages=messages,
        temperature=float(os.getenv("FANCY_PROMPT_TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("FANCY_MAX_TOKENS", 150)),
    )
    fancy_prompt = response.choices[0].message.content.strip()
    duration = time.perf_counter() - start
    cleaned_prompt = fancy_prompt.strip()
    while (cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"')) or (
        cleaned_prompt.startswith("'") and cleaned_prompt.endswith("'")
    ):
        cleaned_prompt = cleaned_prompt[1:-1].strip()
    return cleaned_prompt, duration


async def build_flux_fancy_prompt(prompt: str):
    """Flux 2-tuned variant of :func:`build_fancy_prompt`.

    Loads ``prompts/fancy_flux*`` and respects ``FLUX_FANCY_MAX_TOKENS``
    (default 120) so the rewritten prompt stays a tight natural-language
    paragraph instead of SD's longer CLIP-style tag dump.
    """
    fancy_instructions = soupy_prompts.load_prompt("fancy_flux", fallback="")
    combined_instructions = f"{fancy_instructions}\n\nThe prompt you are elaborating on is: {prompt}"
    start = time.perf_counter()
    messages = [
        {"role": "system", "content": combined_instructions},
        {"role": "user", "content": "Please rewrite the above prompt accordingly."},
    ]
    response = await async_chat_completion(
        model=os.getenv("LOCAL_CHAT"),
        messages=messages,
        temperature=float(os.getenv("FANCY_PROMPT_TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("FLUX_FANCY_MAX_TOKENS", 120)),
    )
    fancy_prompt = response.choices[0].message.content.strip()
    duration = time.perf_counter() - start
    cleaned_prompt = fancy_prompt.strip()
    while (cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"')) or (
        cleaned_prompt.startswith("'") and cleaned_prompt.endswith("'")
    ):
        cleaned_prompt = cleaned_prompt[1:-1].strip()
    return cleaned_prompt, duration


def enhance_img2img_prompt(prompt: str, strength: float) -> str:
    """Enhance img2img prompts based on strength for better results."""
    # Base quality enhancers
    quality_enhancers = ["high quality", "detailed", "sharp focus", "professional photography"]

    # Style enhancers based on strength
    if strength <= 0.3:
        # Low strength - subtle changes, preserve original style
        style_enhancers = ["subtle transformation", "preserving original composition", "enhanced details"]
    elif strength <= 0.6:
        # Medium strength - moderate changes
        style_enhancers = ["artistic transformation", "enhanced style", "improved composition"]
    else:
        # High strength - major changes
        style_enhancers = ["dramatic transformation", "complete style change", "artistic reinterpretation"]

    # Combine enhancers
    all_enhancers = quality_enhancers + style_enhancers
    enhanced_prompt = f"{prompt}, {', '.join(all_enhancers)}"

    return enhanced_prompt

@app_commands.command(name="img2img", description="Stable Diffusion 3.5: transform an image with a prompt.")
@app_commands.describe(
    prompt="What to transform the image into",
    strength="How much to deviate from the input (0.0-1.0) - lower values preserve more of original",
    steps="Inference steps (15-30 recommended)",
    guidance="CFG guidance scale (7.5-15 recommended for SD 3.5)",
)
async def img2img_cmd(
    interaction: discord.Interaction,
    prompt: str,
    strength: app_commands.Range[float, 0.0, 1.0] = 0.3,  # Better default - more subtle
    steps: app_commands.Range[int, 1, 50] = 20,  # Optimized for speed/quality
    guidance: app_commands.Range[float, 0.0, 20.0] = 7.5,  # Better for SD 3.5 Medium
):
    if not SD_IMG2IMG_URL:
        await interaction.response.send_message("❌ SD_IMG2IMG_URL not configured.", ephemeral=True)
        return

    # Get last image in channel history
    last_image = None
    async for msg in interaction.channel.history(limit=20, oldest_first=False):
        if msg.attachments:
            for att in msg.attachments:
                if any(att.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                    last_image = att
                    break
        if last_image:
            break

    if not last_image:
        await interaction.response.send_message(
            "❌ Attach an image (or have one recently in the channel) and try again.", ephemeral=True
        )
        return

    await interaction.response.defer()

    # Enhance the prompt for better img2img results
    enhanced_prompt = enhance_img2img_prompt(prompt, strength)
    logger.info(f"🖼️ Enhanced img2img prompt for {interaction.user}: '{enhanced_prompt}'")

    # Download the image
    async with aiohttp.ClientSession() as session:
        async with session.get(last_image.url) as resp:
            if resp.status != 200:
                await interaction.followup.send("❌ Failed to download the image.", ephemeral=True)
                return
            img_bytes = await resp.read()

        # Use the source image's dimensions (backend will adjust to valid multiples if needed)
        try:
            init_image = Image.open(BytesIO(img_bytes))
            src_w, src_h = init_image.size
        except Exception:
            src_w, src_h = 1024, 1024

        # Build multipart form
        form = aiohttp.FormData()
        form.add_field("image", img_bytes, filename="source.jpg", content_type="image/jpeg")
        form.add_field("prompt", enhanced_prompt)
        form.add_field("negative_prompt", soupy_prompts.load_prompt("sd_negative_prompt", fallback=""))
        form.add_field("steps", str(steps))
        form.add_field("guidance_scale", str(guidance))
        form.add_field("width", str(src_w))
        form.add_field("height", str(src_h))
        form.add_field("seed", str(-1))
        form.add_field("strength", str(strength))

        try:
            async with session.post(SD_IMG2IMG_URL, data=form) as r:
                if r.status != 200:
                    await interaction.followup.send(f"❌ Img2Img server error: HTTP {r.status}", ephemeral=True)
                    return
                out_bytes = await r.read()
        except (ClientConnectorError, asyncio.TimeoutError):
            # Retry against fallback derived from SD_SERVER_URL
            fallback_url = f"{SD_SERVER_URL.rstrip('/')}/sd_img2img"
            logger.warning(f"Primary SD_IMG2IMG_URL unreachable, retrying fallback: {fallback_url}")
            # Rebuild form for retry (FormData cannot be reused)
            form2 = aiohttp.FormData()
            form2.add_field("image", img_bytes, filename="source.jpg", content_type="image/jpeg")
            form2.add_field("prompt", enhanced_prompt)
            form2.add_field("negative_prompt", soupy_prompts.load_prompt("sd_negative_prompt", fallback=""))
            form2.add_field("steps", str(steps))
            form2.add_field("guidance_scale", str(guidance))
            form2.add_field("width", str(src_w))
            form2.add_field("height", str(src_h))
            form2.add_field("seed", str(-1))
            form2.add_field("strength", str(strength))
            async with session.post(fallback_url, data=form2) as r:
                if r.status != 200:
                    await interaction.followup.send(f"❌ Img2Img server error: HTTP {r.status}", ephemeral=True)
                    return
                out_bytes = await r.read()

    file = discord.File(BytesIO(out_bytes), filename="img2img.jpg")
    await interaction.followup.send(content=f"{interaction.user.mention} 🖼️ Img2Img result", file=file)


@app_commands.command(name="inpaint", description="Stable Diffusion 3.5: inpaint an image with a mask and prompt.")
@app_commands.describe(
    prompt="What to paint in the white areas of the mask",
    strength="How aggressively to change the masked areas (0.0-1.0) - higher for inpaint",
    steps="Inference steps (20-30 recommended)",
    guidance="CFG guidance scale (7.5-12 recommended for SD 3.5)",
)
async def inpaint_cmd(
    interaction: discord.Interaction,
    prompt: str,
    strength: app_commands.Range[float, 0.0, 1.0] = 0.8,  # Good for inpaint
    steps: app_commands.Range[int, 1, 50] = 20,  # Optimized
    guidance: app_commands.Range[float, 0.0, 20.0] = 7.5,  # Better for SD 3.5
):
    if not SD_INPAINT_URL:
        await interaction.response.send_message("❌ SD_INPAINT_URL not configured.", ephemeral=True)
        return

    # Expect two recent attachments: base image and mask (L mode white=edit)
    attachments = []
    async for msg in interaction.channel.history(limit=25, oldest_first=False):
        for att in msg.attachments:
            if any(att.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                attachments.append(att)
        if len(attachments) >= 2:
            break

    if len(attachments) < 2:
        await interaction.response.send_message(
            "❌ Please upload an image and a mask (white=edit, black=keep) and try again.", ephemeral=True
        )
        return

    image_att, mask_att = attachments[0], attachments[1]
    await interaction.response.defer()

    async with aiohttp.ClientSession() as session:
        async with session.get(image_att.url) as r1:
            if r1.status != 200:
                await interaction.followup.send("❌ Failed to download image.", ephemeral=True)
                return
            image_bytes = await r1.read()

        async with session.get(mask_att.url) as r2:
            if r2.status != 200:
                await interaction.followup.send("❌ Failed to download mask.", ephemeral=True)
                return
            mask_bytes = await r2.read()

        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename="base.jpg", content_type="image/jpeg")
        form.add_field("mask", mask_bytes, filename="mask.png", content_type="image/png")
        form.add_field("prompt", prompt)
        form.add_field("negative_prompt", soupy_prompts.load_prompt("sd_negative_prompt", fallback=""))
        form.add_field("steps", str(steps))
        form.add_field("guidance_scale", str(guidance))
        # Use source image dimensions
        try:
            base_img = Image.open(BytesIO(image_bytes))
            src_w, src_h = base_img.size
        except Exception:
            src_w, src_h = 1024, 1024
        form.add_field("width", str(src_w))
        form.add_field("height", str(src_h))
        form.add_field("seed", str(-1))
        form.add_field("strength", str(strength))

        try:
            async with session.post(SD_INPAINT_URL, data=form) as r:
                if r.status != 200:
                    await interaction.followup.send(f"❌ Inpaint server error: HTTP {r.status}", ephemeral=True)
                    return
                out_bytes = await r.read()
        except (ClientConnectorError, asyncio.TimeoutError):
            # Retry against fallback derived from SD_SERVER_URL
            fallback_url = f"{SD_SERVER_URL.rstrip('/')}/sd_inpaint"
            logger.warning(f"Primary SD_INPAINT_URL unreachable, retrying fallback: {fallback_url}")
            # Rebuild form for retry (FormData cannot be reused)
            form2 = aiohttp.FormData()
            form2.add_field("image", image_bytes, filename="base.jpg", content_type="image/jpeg")
            form2.add_field("mask", mask_bytes, filename="mask.png", content_type="image/png")
            form2.add_field("prompt", prompt)
            form2.add_field("negative_prompt", soupy_prompts.load_prompt("sd_negative_prompt", fallback=""))
            form2.add_field("steps", str(steps))
            form2.add_field("guidance_scale", str(guidance))
            form2.add_field("width", str(src_w))
            form2.add_field("height", str(src_h))
            form2.add_field("seed", str(-1))
            form2.add_field("strength", str(strength))
            async with session.post(fallback_url, data=form2) as r:
                if r.status != 200:
                    await interaction.followup.send(f"❌ Inpaint server error: HTTP {r.status}", ephemeral=True)
                    return
                out_bytes = await r.read()

    file = discord.File(BytesIO(out_bytes), filename="inpaint.jpg")
    await interaction.followup.send(content=f"{interaction.user.mention} 🖌️ Inpaint result", file=file)


@app_commands.command(name="outpaint", description="Stable Diffusion 3.5: extend an image by 25% in specified directions.")
@app_commands.describe(
    prompt="What should extend into the new areas (e.g., 'extend the landscape, continue the mountains')",
    direction="Which direction(s) to extend the image",
    strength="How aggressively to change the extended areas (0.0-1.0)",
    steps="Inference steps (20-30 recommended)",
    guidance="CFG guidance scale (7.5-12 recommended for SD 3.5)",
)
@app_commands.choices(
    direction=[
        app_commands.Choice(name="Horizontal (left + right)", value="horizontal"),
        app_commands.Choice(name="Vertical (top + bottom)", value="vertical"),
        app_commands.Choice(name="Both (all directions)", value="both"),
    ]
)
async def outpaint_cmd(
    interaction: discord.Interaction,
    prompt: str,
    direction: app_commands.Choice[str],
    strength: app_commands.Range[float, 0.0, 1.0] = 0.8,  # Higher for outpaint
    steps: app_commands.Range[int, 1, 50] = 20,  # Optimized
    guidance: app_commands.Range[float, 0.0, 20.0] = 7.5,  # Better for SD 3.5
):
    # Prefer hybrid outpaint endpoint if available
    if not SD_OUTPAINT_HYBRID_URL and not SD_INPAINT_URL:
        await interaction.response.send_message("❌ No outpaint endpoint configured.", ephemeral=True)
        return

    # Get the most recent image in channel history
    last_image = None
    async for msg in interaction.channel.history(limit=20, oldest_first=False):
        if msg.attachments:
            for att in msg.attachments:
                if any(att.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                    last_image = att
                    break
        if last_image:
            break

    if not last_image:
        await interaction.response.send_message("❌ Please upload an image and try again.", ephemeral=True)
        return

    logger.info(
        f"🖼️ Slash Command 'outpaint' invoked by {interaction.user} with prompt: '{prompt}', direction: '{direction.value}', strength: '{strength}'"
    )
    await interaction.response.send_message("🛠️ Your outpaint request has been queued...", ephemeral=True)

    # Determine dimensions based on direction
    direction_value = direction.value
    if direction_value == "horizontal":
        width = int(SD_DEFAULT_WIDTH * 1.25)
        height = SD_DEFAULT_HEIGHT
    elif direction_value == "vertical":
        width = SD_DEFAULT_WIDTH
        height = int(SD_DEFAULT_HEIGHT * 1.25)
    else:  # both
        width = int(SD_DEFAULT_WIDTH * 1.25)
        height = int(SD_DEFAULT_HEIGHT * 1.25)

    # Ensure dimensions are multiples of 64
    width = ((width + 63) // 64) * 64
    height = ((height + 63) // 64) * 64

    await bot.sd_queue.put(
        {
            "type": "outpaint",
            "interaction": interaction,
            "prompt": prompt,
            "direction": direction_value,
            "width": width,
            "height": height,
            "seed": -1,  # Random seed
            "strength": strength,
            "steps": steps,
            "guidance": guidance,
        }
    )
    logger.info(
        f"🖼️ Queued outpaint generation for {interaction.user}: prompt='{prompt}', direction='{direction_value}', strength='{strength}'"
    )

def archive_image_bytes(
    image_bytes: bytes,
    *,
    filename: str,
    prompt: str,
    user_id: int,
    username: str,
    width: int,
    height: int,
    seed: int,
    guild_id: int | None,
    channel_id: int | None,
    source: str = "sd",
) -> None:
    try:
        import json
        from datetime import datetime, timezone
        from io import BytesIO

        from PIL import Image

        media = _ensure_media_dirs()
        img_path = media / "images" / filename
        thumb_path = media / "thumbs" / filename

        # Save original
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        # Save thumbnail
        try:
            im = Image.open(BytesIO(image_bytes)).convert("RGB")
            im.thumbnail((400, 400))
            im.save(thumb_path, format="PNG")
        except Exception:
            # If thumbnail fails, ignore
            pass

        # Append metadata JSONL
        meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "prompt": prompt,
            "user_id": user_id,
            "username": username,
            "width": width,
            "height": height,
            "seed": seed,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "source": source,
        }
        index_path = media / "images" / "index.jsonl"
        with open(index_path, "a", encoding="utf-8") as idx:
            idx.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception as _e:
        logger.debug(f"archive_image_bytes failed: {_e}")

@app_commands.command(name="sd", description="Generates an image using Stable Diffusion.")
@app_commands.describe(
    description="Description of the image to generate", size="Size of the image", seed="Seed for random generation"
)
@app_commands.choices(
    size=[
        app_commands.Choice(name=f"Default ({SD_DEFAULT_WIDTH}x{SD_DEFAULT_HEIGHT})", value="default"),
        app_commands.Choice(name=f"Wide ({SD_WIDE_WIDTH}x{SD_WIDE_HEIGHT})", value="wide"),
        app_commands.Choice(name=f"Tall ({SD_TALL_WIDTH}x{SD_TALL_HEIGHT})", value="tall"),
        app_commands.Choice(name=f"Square ({SD_DEFAULT_WIDTH}x{SD_DEFAULT_HEIGHT})", value="square"),
    ]
)
async def sd(
    interaction: discord.Interaction,
    description: str,
    size: Optional[app_commands.Choice[str]] = None,
    seed: Optional[int] = None,
):
    size_value = size.value if size else "default"

    logger.info(
        f"🎨 Slash Command 'sd' invoked by {interaction.user} with description: '{description}', size: '{size_value}', seed: '{seed if seed else 'random'}'"
    )
    # Use defer + followup to avoid Unknown interaction issues
    try:
        if not interaction.response.is_done():
            # IMPORTANT: Do NOT defer ephemerally for /sd.
            # An ephemeral defer causes the generation to appear "only to the user",
            # which is not desired for regular /sd generations.
            await interaction.response.defer(thinking=True)
    except Exception as e:
        logger.debug(f"Defer failed: {e}")
    await bot.sd_queue.put(
        {
            "type": "sd",
            "interaction": interaction,
            "description": description,
            "size": size_value,
            "seed": seed,
        }
    )
    logger.info(
        f"🎨 Queued image generation for {interaction.user}: description='{description}', size='{size_value}', seed='{seed if seed else 'random'}'"
    )
    # Avoid sending extra followup ack to prevent 40060/10062 errors if interaction state changes

async def handle_remix(interaction, prompt, width, height, seed, queue_size):
    # Update to include server ID
    await increment_user_stat(interaction.user.id, "images_generated", interaction.guild_id)

    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Remix", queue_size=queue_size)


async def handle_wide(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, "images_generated")

    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Wide", queue_size=queue_size)


async def handle_tall(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, "images_generated")

    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Tall", queue_size=queue_size)


async def handle_edit(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, "images_generated")

    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Edit", queue_size=queue_size)


async def handle_outpaint(
    interaction, prompt, direction, width, height, seed, queue_size, strength=0.8, steps=None, guidance=None
):
    """Handle outpainting - extend image by 25% in specified directions using inpaint mask to preserve interior"""
    try:
        logger.info(
            f"🖼️ Starting outpainting for {interaction.user}: prompt='{prompt}', direction='{direction}', size={width}x{height}"
        )

        # Check if we need to send an initial response
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        # Prefer the image attached to the message where the button was clicked
        # Fall back to the last image in the channel (for slash command usage)
        last_image = None

        try:
            src_msg = getattr(interaction, "message", None)
            if src_msg and src_msg.attachments:
                for att in src_msg.attachments:
                    if any(att.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                        last_image = att
                        break
        except Exception:
            # If anything goes wrong, ignore and use channel fallback
            last_image = None

        # Fallback: search channel history for most recent image
        if not last_image:
            async for msg in interaction.channel.history(limit=20, oldest_first=False):
                if msg.attachments:
                    for att in msg.attachments:
                        if any(att.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                            last_image = att
                            break
                if last_image:
                    break

        if not last_image:
            # Interaction tokens can expire (long queues). Fallback to channel.send so the user still sees errors.
            try:
                await interaction.followup.send("❌ Please upload an image and try again.", ephemeral=True)
            except Exception as send_error:
                logger.warning(
                    f"Failed to send outpaint error via followup; falling back to channel.send: {send_error}"
                )
                await interaction.channel.send(f"{interaction.user.mention} ❌ Please upload an image and try again.")
            return

        # Use typing context manager for consistent behavior
        async with interaction.channel.typing():
            # Use optimized session with connection pooling and keep-alive
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                keepalive_timeout=30,  # Keep connections alive
                enable_cleanup_closed=True,
            )
            # Increased timeout for SD 3.5 Medium on Mac (can take 2-5 minutes)
            timeout = aiohttp.ClientTimeout(total=600, connect=10)

            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers={"Connection": "keep-alive"}
            ) as session:
                # Download the image
                async with session.get(last_image.url) as resp:
                    if resp.status != 200:
                        await interaction.followup.send("❌ Failed to download the image.", ephemeral=True)
                        return
                    img_bytes = await resp.read()

                # Create edge-extended outpaint canvas (avoid gray padding) and mask for new areas
                try:
                    original_image = Image.open(BytesIO(img_bytes))
                    original_width, original_height = original_image.size

                    # Calculate new dimensions (25% larger)
                    if direction == "horizontal":
                        new_width = int(original_width * 1.25)
                        new_height = original_height
                    elif direction == "vertical":
                        new_width = original_width
                        new_height = int(original_height * 1.25)
                    else:  # both
                        new_width = int(original_width * 1.25)
                        new_height = int(original_height * 1.25)

                    # Ensure dimensions are multiples of 64
                    new_width = ((new_width + 63) // 64) * 64
                    new_height = ((new_height + 63) // 64) * 64

                    # Compute pad sizes for edge extension
                    paste_x = (new_width - original_width) // 2
                    paste_y = (new_height - original_height) // 2
                    top = paste_y
                    bottom = new_height - (paste_y + original_height)
                    left = paste_x
                    right = new_width - (paste_x + original_width)

                    cv_img = cv2.cvtColor(np.array(original_image.convert("RGB")), cv2.COLOR_RGB2BGR)
                    extended = cv2.copyMakeBorder(cv_img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
                    # Do NOT blur the entire canvas; keep the original interior pixel-perfect
                    padded_image = Image.fromarray(cv2.cvtColor(extended, cv2.COLOR_BGR2RGB))

                    # Build mask that covers ONLY the new outpainted areas (not the original image)
                    # This ensures the original image remains untouched, preventing visible borders
                    mask_img = Image.new("L", (new_width, new_height), 0)
                    draw = ImageDraw.Draw(mask_img)

                    # Create a feather zone width for smooth blending at the border
                    # This will be applied only in the new area, not extending into original
                    feather_width = max(8, min(original_width, original_height) // 64)

                    # Mask should start exactly at the border of the original image
                    # We'll add a small feather zone that extends slightly into the new area for blending
                    if direction == "horizontal" or direction == "both":
                        if left > 0:
                            # Left side: mask from 0 to paste_x + feather_width (feather extends into new area only)
                            x2 = min(new_width - 1, paste_x + feather_width)
                            draw.rectangle([0, 0, x2, new_height - 1], fill=255)
                        if right > 0:
                            # Right side: mask starts feather_width pixels before the right edge of original
                            x1 = max(0, paste_x + original_width - feather_width)
                            draw.rectangle([x1, 0, new_width - 1, new_height - 1], fill=255)
                    if direction == "vertical" or direction == "both":
                        if top > 0:
                            # Top: mask from 0 to paste_y + feather_width
                            y2 = min(new_height - 1, paste_y + feather_width)
                            draw.rectangle([0, 0, new_width - 1, y2], fill=255)
                        if bottom > 0:
                            # Bottom: mask starts feather_width pixels before the bottom edge of original
                            y1 = max(0, paste_y + original_height - feather_width)
                            draw.rectangle([0, y1, new_width - 1, new_height - 1], fill=255)

                    # Now create a hard mask that protects the original image area completely
                    # This ensures no part of the original image gets modified
                    protection_mask = Image.new("L", (new_width, new_height), 255)
                    protect_draw = ImageDraw.Draw(protection_mask)
                    # The original image area should be black (protected) in the protection mask
                    protect_draw.rectangle(
                        [paste_x, paste_y, paste_x + original_width - 1, paste_y + original_height - 1], fill=0
                    )

                    # Apply feathering to the mask for smooth blending
                    try:
                        mask_np = np.array(mask_img)
                        protection_np = np.array(protection_mask)
                        # Use Gaussian blur for smooth feathering
                        sigma = 2.5
                        mask_np = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
                        # Ensure the original image area is completely protected (0 in final mask)
                        # by multiplying with the protection mask (inverted: 0=protected, 255=editable)
                        mask_np = np.minimum(mask_np, protection_np)
                        mask_img = Image.fromarray(mask_np).convert("L")
                    except Exception as e:
                        logger.warning(f"Mask blur failed, using hard mask: {e}")

                    padded_png_bytes = BytesIO()
                    padded_image.save(padded_png_bytes, format="PNG")
                    padded_png_bytes.seek(0)
                    padded_raw = padded_png_bytes.getvalue()

                    mask_png_bytes = BytesIO()
                    mask_img.save(mask_png_bytes, format="PNG")
                    mask_png_bytes.seek(0)
                    mask_raw = mask_png_bytes.getvalue()
                except Exception as e:
                    await interaction.followup.send(f"❌ Error processing image: {e}", ephemeral=True)
                    return

                # Enhance prompt for outpainting
                enhanced_prompt = f"outpaint, extend the image seamlessly, continue the scene naturally, maintain visual consistency, {prompt.lower()}"

                # Prefer hybrid endpoint; fallback to inpaint
                use_hybrid = SD_OUTPAINT_HYBRID_URL is not None
                out_bytes = None

                # Use provided parameters or fall back to env vars
                actual_steps = steps if steps is not None else int(os.getenv("SD_STEPS", 20))
                actual_guidance = guidance if guidance is not None else float(os.getenv("SD_GUIDANCE", 7.5))

                if use_hybrid:
                    form = aiohttp.FormData()
                    form.add_field("image", padded_raw, filename="padded.png", content_type="image/png")
                    form.add_field("mask", mask_raw, filename="mask.png", content_type="image/png")
                    form.add_field("prompt", enhanced_prompt)
                    form.add_field("negative_prompt", soupy_prompts.load_prompt("sd_negative_prompt", fallback=""))
                    # Use actual function parameters instead of env vars
                    form.add_field("steps", str(actual_steps))
                    form.add_field("guidance_scale", str(actual_guidance))
                    form.add_field("width", str(new_width))
                    form.add_field("height", str(new_height))
                    form.add_field("seed", str(seed))
                    form.add_field("strength", str(strength))
                    form.add_field("use_canny", str(OUTPAINT_USE_CANNY).lower())
                    form.add_field("use_depth", str(OUTPAINT_USE_DEPTH).lower())
                    form.add_field("control_weight", str(OUTPAINT_CONTROL_WEIGHT))
                    form.add_field("harmonize_strength", str(OUTPAINT_HARMONIZE_STRENGTH))
                    # Enable color matching by default for seamless blending
                    form.add_field("color_match", str(os.getenv("OUTPAINT_COLOR_MATCH", "true")).lower())

                    try:
                        async with session.post(SD_OUTPAINT_HYBRID_URL, data=form) as r:
                            if r.status == 200:
                                out_bytes = await r.read()
                            else:
                                logger.error(f"🖼️ Hybrid outpaint error for {interaction.user}: HTTP {r.status}")
                                use_hybrid = False
                    except (ClientConnectorError, ClientOSError, asyncio.TimeoutError):
                        logger.warning("Hybrid outpaint unreachable, falling back to SD_INPAINT")
                        use_hybrid = False

                if not use_hybrid:
                    form = aiohttp.FormData()
                    form.add_field("image", padded_raw, filename="padded.png", content_type="image/png")
                    form.add_field("mask", mask_raw, filename="mask.png", content_type="image/png")
                    form.add_field("prompt", enhanced_prompt)
                    form.add_field("negative_prompt", soupy_prompts.load_prompt("sd_negative_prompt", fallback=""))
                    form.add_field("steps", str(actual_steps))
                    form.add_field("guidance_scale", str(actual_guidance))
                    form.add_field("width", str(new_width))
                    form.add_field("height", str(new_height))
                    form.add_field("seed", str(seed))
                    form.add_field("strength", str(strength))
                    async with session.post(SD_INPAINT_URL, data=form) as r:
                        if r.status == 200:
                            out_bytes = await r.read()
                        else:
                            logger.error(f"🖼️ Outpaint server error for {interaction.user}: HTTP {r.status}")
                            await interaction.followup.send(
                                f"❌ Outpaint server error: HTTP {r.status}", ephemeral=True
                            )
                            return

                # Generate a unique filename
                random_number = random.randint(100000, 999999)
                safe_prompt = re.sub(r"\W+", "", prompt[:40]).lower()
                filename = f"{random_number}_{safe_prompt}_outpaint_{direction}.png"

                # If hybrid used, server already harmonized/blended; send directly
                final_bytes = out_bytes
                if not use_hybrid:
                    # Safeguard: composite the original interior back onto the result
                    try:
                        result_img = Image.open(BytesIO(out_bytes)).convert("RGB")
                        result_img.paste(original_image.convert("RGB"), (paste_x, paste_y))
                        final_io = BytesIO()
                        result_img.save(final_io, format="PNG")
                        final_io.seek(0)
                        final_bytes = final_io.getvalue()
                    except Exception as e:
                        logger.warning(f"Failed compositing original interior on outpaint result: {e}")
                        final_bytes = out_bytes

                # Create a Discord File object from the image bytes
                image_file = discord.File(BytesIO(final_bytes), filename=filename)

                # Create embed messages
                description_embed = discord.Embed(
                    description=f"**Prompt:** {prompt}\n**Direction:** {direction.title()}",
                    color=discord.Color.purple(),
                )
                details_embed = discord.Embed(color=discord.Color.green())

                queue_total = queue_size + 1
                details_text = f"🖼️ Outpaint {direction.title()} ⏱️ Extended ⏱️ 📋 {queue_total}"
                details_embed.description = details_text

                # Initialize the SDRemixView with new image parameters
                new_view = SDRemixView(prompt=prompt, width=new_width, height=new_height, seed=seed)

                # Send the result
                # Component interactions can expire if the queue is long; fall back to a normal channel send.
                try:
                    await interaction.followup.send(
                        content=f"{interaction.user.mention} 🖼️ Outpainted Image:",
                        embeds=[description_embed, details_embed],
                        file=image_file,
                        view=new_view,
                        ephemeral=False,
                    )
                except Exception as send_error:
                    logger.error(
                        f"❌ Failed to send outpaint result via followup, falling back to channel.send: {send_error}"
                    )
                    await interaction.channel.send(
                        content=f"{interaction.user.mention} 🖼️ Outpainted Image:",
                        embeds=[description_embed, details_embed],
                        file=image_file,
                        view=new_view,
                    )

                # Archive the outpaint generation
                try:
                    archive_image_bytes(
                        final_bytes,
                        filename=filename,
                        prompt=prompt,
                        user_id=interaction.user.id,
                        username=str(interaction.user),
                        width=new_width,
                        height=new_height,
                        seed=seed,
                        guild_id=(interaction.guild.id if interaction.guild else None),
                        channel_id=(interaction.channel.id if interaction.channel else None),
                    )
                    archive_sent_message(
                        content=f"Outpainted image ({direction}): {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                        user_id=interaction.user.id,
                        username=str(interaction.user),
                        guild_id=(interaction.guild.id if interaction.guild else None),
                        channel_id=(interaction.channel.id if interaction.channel else None),
                        image_filename=filename,
                        event_type="image_generation",
                    )
                except Exception:
                    pass

                logger.info(
                    f"🖼️ Outpainting completed for {interaction.user}: filename='{filename}', direction='{direction}'"
                )

    except Exception as e:
        logger.error(f"🖼️ Error in handle_outpaint for {interaction.user}: {e}")
        # Interaction tokens can expire; always fall back to channel.send on failures.
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Error during outpainting: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f"❌ Error during outpainting: {e}", ephemeral=True)
        except Exception as send_error:
            logger.error(
                f"❌ Failed to send outpaint error via interaction; falling back to channel.send: {send_error}"
            )
            try:
                await interaction.channel.send(f"{interaction.user.mention} ❌ Error during outpainting: {e}")
            except Exception:
                pass


fancy_instructions = soupy_prompts.load_prompt("fancy", fallback="")


async def handle_fancy(interaction, prompt, width, height, seed, queue_size):
    """Handle the 'Fancy' button click."""
    try:
        logger.info(f"'Fancy' button clicked by {interaction.user} for prompt: '{prompt}'")

        # If this is a new interaction (not a followup), defer it
        if not interaction.response.is_done():
            await interaction.response.defer()  # Remove thinking=True to make it visible to channel

        # Get the fancy instructions
        fancy_instructions = soupy_prompts.load_prompt("fancy", fallback="")
        logger.debug(f"📜 Retrieved 'FANCY' instructions: {fancy_instructions}")

        # Combine instructions with prompt
        combined_instructions = f"{fancy_instructions}\n\nThe prompt you are elaborating on is: {prompt}"
        logger.debug(f"📜 Combined rewriting instructions for {interaction.user}: {combined_instructions}")

        # Start timing for prompt generation
        prompt_start_time = time.perf_counter()

        # Generate fancy prompt
        messages = [
            {"role": "system", "content": combined_instructions},
            {"role": "user", "content": "Please rewrite the above prompt accordingly."},
        ]

        logger.debug(f"📜 Sending the following messages to LLM (Fancy):\n{format_messages(messages)}")

        response = await async_chat_completion(
            model=os.getenv("LOCAL_CHAT"),
            messages=messages,
            temperature=float(os.getenv("FANCY_PROMPT_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("FANCY_MAX_TOKENS", 150)),
        )

        fancy_prompt = response.choices[0].message.content.strip()
        logger.info(f"🪄 Fancy prompt generated for {interaction.user}: '{fancy_prompt}'")

        # Calculate prompt rewriting duration
        prompt_end_time = time.perf_counter()
        prompt_duration = prompt_end_time - prompt_start_time
        logger.info(f"⏱️ Prompt rewriting time for {interaction.user}: {prompt_duration:.2f} seconds")

        # Strip surrounding quotes from the prompt (don't use full clean_response — no truncation or RAG sanitization needed)
        cleaned_prompt = fancy_prompt.strip()
        while (cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"')) or (
            cleaned_prompt.startswith("'") and cleaned_prompt.endswith("'")
        ):
            cleaned_prompt = cleaned_prompt[1:-1].strip()
        logger.debug(f"🪄 Cleaned fancy prompt for {interaction.user}: '{cleaned_prompt}'")

        # Generate the image with the fancy prompt
        await generate_sd_image(
            interaction=interaction,
            prompt=cleaned_prompt,
            width=width,
            height=height,
            seed=seed,
            action_name="Fancy",
            queue_size=queue_size,
            pre_duration=prompt_duration,
        )

        logger.info(f"🪄 Passed cleaned fancy prompt to image generator for {interaction.user}")

    except Exception as e:
        error_msg = f"Error handling fancy button: {str(e)}"
        logger.error(error_msg)
        if not interaction.response.is_done():
            await interaction.response.send_message(error_msg, ephemeral=True)
        else:
            await interaction.followup.send(error_msg, ephemeral=True)


# ---------------------------------------------------------------------------
# Discord UI components: edit modal, remix buttons
# ---------------------------------------------------------------------------


class EditImageModal(Modal, title="🖌️ Edit Image Parameters"):
    def __init__(self, prompt: str, width: int, height: int, seed: int = None):
        super().__init__()
        self.prompt = prompt
        self.width_val = width
        self.height_val = height
        self.seed_val = seed

        self.image_description = TextInput(
            label="📝 Image Description",
            style=discord.TextStyle.paragraph,
            default=prompt,
            required=True,
            max_length=2000,
        )
        self.width_input = TextInput(
            label="📏 Width",
            style=discord.TextStyle.short,
            default=str(width),
            required=True,
            min_length=1,
            max_length=5,
        )
        self.height_input = TextInput(
            label="📐 Height",
            style=discord.TextStyle.short,
            default=str(height),
            required=True,
            min_length=1,
            max_length=5,
        )
        self.seed_input = TextInput(
            label="🌱 Seed",
            style=discord.TextStyle.short,
            default=str(seed) if seed is not None else "",
            required=False,
            max_length=10,
        )

        self.add_item(self.image_description)
        self.add_item(self.width_input)
        self.add_item(self.height_input)
        self.add_item(self.seed_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            await interaction.response.send_message("🛠️ Updating parameters...", ephemeral=True)
            new_prompt = self.image_description.value.strip()

            try:
                original_width = int(self.width_input.value.strip())
                original_height = int(self.height_input.value.strip())
            except ValueError:
                await interaction.followup.send("❌ Width and Height must be valid integers.", ephemeral=True)
                logger.warning("User provided invalid dimensions.")
                return

            def adjust_to_multiple_of_64(value: int) -> int:
                if value <= 0:
                    value = 64
                else:
                    value = ((value + 63) // 64) * 64
                return value

            new_width = adjust_to_multiple_of_64(original_width)
            new_height = adjust_to_multiple_of_64(original_height)

            seed_value = self.seed_input.value.strip()
            if seed_value.isdigit():
                new_seed = int(seed_value)
            else:
                new_seed = random.randint(0, 2**32 - 1)

            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "edit",
                    "prompt": new_prompt,
                    "width": new_width,
                    "height": new_height,
                    "seed": new_seed,
                }
            )

            logger.info(f"Edit requested: prompt='{new_prompt}', dimensions={new_width}x{new_height}, seed={new_seed}")
        except Exception as e:
            await interaction.followup.send("❌ An error occurred while processing your edit.", ephemeral=True)
            logger.error(f"Error in EditImageModal submission: {e}")


# UI view class for image remixing and manipulation
class SDRemixView(View):
    def __init__(self, prompt: str, width: int, height: int, seed: int = None):
        super().__init__(timeout=None)
        self.prompt = prompt
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.cleaned_prompt = self.parse_prompt(prompt)
        logger.debug(f"View initialized: prompt='{self.cleaned_prompt}', {self.width}x{self.height}, seed={self.seed}")

    def parse_prompt(self, prompt: str) -> str:
        # Clean and format the prompt text
        return prompt.strip()

    @discord.ui.button(label="✏️", style=discord.ButtonStyle.success, custom_id="flux_edit_button", row=0)
    @universal_cooldown_check()
    async def edit_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Edit' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            modal = EditImageModal(prompt=self.prompt, width=self.width, height=self.height, seed=self.seed)
            await interaction.response.send_modal(modal)
            logger.info(f"Opened Edit modal for {interaction.user}")
        except Exception as e:
            logger.error(f"Error opening Edit modal for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error opening edit dialog.", ephemeral=True)

    @discord.ui.button(label="🪄", style=discord.ButtonStyle.primary, custom_id="flux_fancy_button", row=0)
    @universal_cooldown_check()
    async def fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Fancy' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Making it fancy...", ephemeral=True)
            _queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "fancy",
                    "prompt": self.cleaned_prompt,  # The original prompt
                    "width": self.width,
                    "height": self.height,
                    "seed": self.seed,
                }
            )
            logger.info(
                f"Enqueued 'Fancy' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={self.width}x{self.height}, seed={self.seed}"
            )
        except Exception as e:
            logger.error(f"Error during fancy transformation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error during fancy transformation.", ephemeral=True)

    @discord.ui.button(label="🌱🎲", style=discord.ButtonStyle.primary, custom_id="flux_remix_button", row=0)
    @universal_cooldown_check()
    async def remix_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Remix' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Remixing...", ephemeral=True)
            _queue_size = bot.sd_queue.qsize()
            new_seed = random.randint(0, 2**32 - 1)
            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "remix",
                    "prompt": self.cleaned_prompt,
                    "width": self.width,
                    "height": self.height,
                    "seed": new_seed,
                }
            )
            logger.info(
                f"Enqueued 'Remix' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={self.width}x{self.height}, seed={new_seed}"
            )

            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, "images_generated")
        except Exception as e:
            logger.error(f"Error during remix for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error during remix.", ephemeral=True)

    @discord.ui.button(
        label="🎨 R-Fancy", style=discord.ButtonStyle.danger, custom_id="flux_random_fancy_button", row=1
    )
    @universal_cooldown_check()
    async def random_fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """
        Handler for the 'R-Fancy' button. Generates a new random prompt using LLM
        with random terms from categories. Also randomly selects image dimensions.
        """
        logger.info(f"🎨 'R-Fancy' button clicked by {interaction.user}.")
        try:
            await interaction.response.send_message("🛠️ Generating fancy random image...", ephemeral=True)

            # Randomly select dimensions with equal probability
            dimensions = [
                (SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT),  # Square
                (SD_WIDE_WIDTH, SD_WIDE_HEIGHT),  # Wide
                (SD_TALL_WIDTH, SD_TALL_HEIGHT),  # Tall
            ]
            width, height = random.choice(dimensions)

            # Use LLM-generated prompt (set prompt to None so handle_random generates it)
            prompt = None  # Will be generated in handle_random

            _queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "random",
                    "width": width,
                    "height": height,
                    "seed": None,  # Random will generate its own seed
                    "prompt": prompt,  # None for LLM-generated prompt
                }
            )
            logger.info(f"🎨 Enqueued 'R-Fancy' action for {interaction.user} with dimensions {width}x{height}")

            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, "images_generated")
        except Exception as e:
            logger.error(f"🎨 Error queueing fancy random generation for {interaction.user}: {e}")

    @discord.ui.button(
        label="🔤 R-Keyword", style=discord.ButtonStyle.danger, custom_id="flux_random_keyword_button", row=1
    )
    @universal_cooldown_check()
    async def random_keyword_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """
        Handler for the 'R-Keyword' button. Generates a new random prompt using only
        random keywords from categories. Also randomly selects image dimensions.
        """
        logger.info(f"🔤 'R-Keyword' button clicked by {interaction.user}.")
        try:
            await interaction.response.send_message("🛠️ Generating keyword random image...", ephemeral=True)

            # Randomly select dimensions with equal probability
            dimensions = [
                (SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT),  # Square
                (SD_WIDE_WIDTH, SD_WIDE_HEIGHT),  # Wide
                (SD_TALL_WIDTH, SD_TALL_HEIGHT),  # Tall
            ]
            width, height = random.choice(dimensions)

            # Get random terms and use them directly as the prompt
            random_terms = get_random_terms()
            # Flatten the terms from the dictionary into a comma-separated string
            terms_list = []
            for _category, terms in random_terms.items():
                # Split by comma in case there are multiple terms in a single category
                split_terms = [term.strip() for term in terms.split(",")]
                terms_list.extend(split_terms)
            prompt = ", ".join(terms_list)
            logger.info(f"🔤 Using only random terms for {interaction.user}: {prompt}")

            _queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "random",
                    "width": width,
                    "height": height,
                    "seed": None,  # Random will generate its own seed
                    "prompt": prompt,  # Direct terms-only prompt
                }
            )
            logger.info(f"🔤 Enqueued 'R-Keyword' action for {interaction.user} with dimensions {width}x{height}")

            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, "images_generated")
        except Exception as e:
            logger.error(f"🔤 Error queueing keyword random generation for {interaction.user}: {e}")

    @discord.ui.button(label="↔️", style=discord.ButtonStyle.primary, custom_id="flux_wide_button", row=1)
    @universal_cooldown_check()
    async def wide_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Wide' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Generating wide version...", ephemeral=True)
            _queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "wide",
                    "prompt": self.cleaned_prompt,
                    "width": SD_WIDE_WIDTH,
                    "height": SD_WIDE_HEIGHT,
                    "seed": self.seed,
                }
            )
            logger.info(
                f"Enqueued 'Wide' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={SD_WIDE_WIDTH}x{SD_WIDE_HEIGHT}, seed={self.seed}"
            )
        except Exception as e:
            logger.error(f"Error during wide generation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating wide version.", ephemeral=True)

    @discord.ui.button(label="↕️", style=discord.ButtonStyle.primary, custom_id="flux_tall_button", row=1)
    @universal_cooldown_check()
    async def tall_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Tall' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Generating tall version...", ephemeral=True)
            _queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "tall",
                    "prompt": self.cleaned_prompt,
                    "width": SD_TALL_WIDTH,
                    "height": SD_TALL_HEIGHT,
                    "seed": self.seed,
                }
            )
            logger.info(
                f"Enqueued 'Tall' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={SD_TALL_WIDTH}x{SD_TALL_HEIGHT}, seed={self.seed}"
            )
        except Exception as e:
            logger.error(f"Error during tall generation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating tall version.", ephemeral=True)

    @discord.ui.button(label="⤡ Outpaint", style=discord.ButtonStyle.success, custom_id="outpaint_both_button", row=0)
    @universal_cooldown_check()
    async def outpaint_both_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Outpaint Both' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Extending image in all directions...", ephemeral=True)
            _queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put(
                {
                    "type": "button",
                    "interaction": interaction,
                    "action": "outpaint",
                    "prompt": self.cleaned_prompt,
                    "direction": "both",
                    "width": self.width,
                    "height": self.height,
                    "seed": self.seed,
                    "strength": 0.8,  # Higher for outpaint
                }
            )
            logger.info(
                f"Enqueued 'Outpaint Both' action for {interaction.user}: prompt='{self.cleaned_prompt}', direction='both'"
            )
        except Exception as e:
            logger.error(f"Error during outpainting in all directions for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error extending image in all directions.", ephemeral=True)


# ---------------------------------------------------------------------------
# Image generation pipeline — sends prompts to the SD backend, fetches the
# result, archives it, and posts it back to Discord with the remix view.
# ---------------------------------------------------------------------------


async def generate_sd_image(
    interaction,
    prompt,
    width,
    height,
    seed,
    action_name="SD",
    queue_size=0,
    pre_duration=0,
    selected_terms: Optional[str] = None,  # New parameter
):
    try:
        # Check if we need to send an initial response
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        sd_server_url = SD_SERVER_URL.rstrip("/")  # Ensure no trailing slash
        num_steps = int(os.getenv("SD_STEPS", 20))
        guidance = float(os.getenv("SD_GUIDANCE", 7.5))
        negative_prompt = soupy_prompts.load_prompt("sd_negative_prompt", fallback="")
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": str(num_steps),
            "guidance_scale": str(guidance),
            "width": str(width),
            "height": str(height),
            "seed": str(seed),
        }

        # Use typing context manager for consistent behavior
        async with interaction.channel.typing():
            # Use optimized session with connection pooling and keep-alive
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                keepalive_timeout=30,  # Keep connections alive
                enable_cleanup_closed=True,
            )
            # Increased timeout for SD 3.5 Medium on Mac (can take 2-5 minutes)
            timeout = aiohttp.ClientTimeout(total=600, connect=10)

            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout, headers={"Connection": "keep-alive"}
            ) as session:
                # Start timing the image generation process
                image_start_time = time.perf_counter()

                async with session.post(f"{sd_server_url}/sd", data=payload) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "").lower()

                        # Handle different response formats
                        if content_type.startswith("application/json"):
                            # API might return JSON with base64 image
                            try:
                                data = await response.json()
                                if "image" in data:
                                    import base64

                                    if isinstance(data["image"], str):
                                        image_bytes = base64.b64decode(data["image"])
                                    else:
                                        image_bytes = data["image"]
                                elif "image_bytes" in data:
                                    image_bytes = data["image_bytes"]
                                else:
                                    raise ValueError(f"JSON response missing image data: {list(data.keys())}")
                                logger.debug(f"🖼️ Received image from JSON response: {len(image_bytes)} bytes")
                            except Exception as json_error:
                                logger.error(f"🖼️ Failed to parse JSON response: {json_error}")
                                error_text = await response.text()
                                raise ValueError(f"SD server returned JSON but couldn't parse: {error_text[:200]}")
                        elif content_type.startswith("image/"):
                            # Direct image response
                            image_bytes = await response.read()
                            logger.debug(f"🖼️ Received image bytes: {len(image_bytes)} bytes")
                        else:
                            # Unknown format, try to read as image anyway
                            logger.warning(f"🖼️ Unknown content type '{content_type}', attempting to read as image")
                            image_bytes = await response.read()

                        # Validate we got actual image data
                        if not image_bytes or len(image_bytes) < 100:
                            raise ValueError(
                                f"Received invalid image data: {len(image_bytes) if image_bytes else 0} bytes"
                            )

                        # Try to validate it's actually an image by checking magic bytes
                        if not (
                            image_bytes.startswith(b"\x89PNG")
                            or image_bytes.startswith(b"\xff\xd8\xff")
                            or image_bytes.startswith(b"GIF")
                        ):
                            logger.warning(
                                "🖼️ Image data doesn't start with PNG/JPEG/GIF magic bytes, but continuing anyway"
                            )

                        # End timing the image generation process
                        image_end_time = time.perf_counter()
                        image_generation_duration = image_end_time - image_start_time

                        # Calculate total duration
                        total_duration = pre_duration + image_generation_duration
                        logger.info(
                            f"⏱️ Total image generation time for {interaction.user}: {total_duration:.2f} seconds (Prompt: {pre_duration:.2f}s, Image: {image_generation_duration:.2f}s)"
                        )

                        # Generate a unique filename
                        random_number = random.randint(100000, 999999)
                        safe_prompt = re.sub(r"\W+", "", prompt[:40]).lower()
                        filename = f"{random_number}_{safe_prompt}.png"  # Changed to .png

                        # Archive to media for web dashboard
                        try:
                            archive_image_bytes(
                                image_bytes,
                                filename=filename,
                                prompt=prompt,
                                user_id=interaction.user.id,
                                username=str(interaction.user),
                                width=width,
                                height=height,
                                seed=seed,
                                guild_id=(interaction.guild.id if interaction.guild else None),
                                channel_id=(interaction.channel.id if interaction.channel else None),
                            )
                        except Exception as e:
                            logger.debug(f"Archive image failed: {e}")

                        # Create a Discord File object from the image bytes
                        image_file = discord.File(BytesIO(image_bytes), filename=filename)

                        # Create embed messages
                        if selected_terms and selected_terms != prompt:
                            # If selected_terms are provided and different from prompt, include them in the description
                            description_content = f"**Selected Terms:** {selected_terms}\n\n**Prompt:** {prompt}"
                        else:
                            # For simple prompts or when terms are the same as prompt, just show the prompt
                            description_content = f"**Prompt:** {prompt}"

                        description_embed = discord.Embed(description=description_content, color=discord.Color.blue())
                        details_embed = discord.Embed(color=discord.Color.green())

                        queue_total = queue_size + 1
                        details_text = f"🌱 {seed} 🔄 {action_name} ⏱️ {total_duration:.2f}s 📋 {queue_total}"

                        # Change this line to set the description instead of adding a field
                        details_embed.description = details_text

                        # Initialize the SDRemixView with current image parameters
                        new_view = SDRemixView(prompt=prompt, width=width, height=height, seed=seed)

                        # When sending the final message, use followup if the initial response was deferred
                        if interaction.response.is_done():
                            # Force non-ephemeral followup to ensure visibility in channel; fall back to channel.send on error
                            try:
                                await interaction.followup.send(
                                    content=f"{interaction.user.mention} 🖼️ Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view,
                                    ephemeral=False,
                                )
                            except Exception as send_error:
                                logger.error(
                                    f"❌ Failed to send follow-up message (falling back to channel.send): {send_error}"
                                )
                                await interaction.channel.send(
                                    content=f"{interaction.user.mention} 🖼️ Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view,
                                )
                        else:
                            _msg = await interaction.channel.send(
                                content=f"{interaction.user.mention} 🖼️ Generated Image:",
                                embeds=[description_embed, details_embed],
                                file=image_file,
                                view=new_view,
                            )

                        # Archive the image generation activity (do this for both branches)
                        try:
                            archive_sent_message(
                                content=f"Generated image: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                                user_id=interaction.user.id,
                                username=str(interaction.user),
                                guild_id=(interaction.guild.id if interaction.guild else None),
                                channel_id=(interaction.channel.id if interaction.channel else None),
                                image_filename=filename,
                                event_type="image_generation",
                            )
                        except Exception:
                            pass
                        logger.info(
                            f"🖼️ Image generation completed for {interaction.user}: filename='{filename}', total_duration={total_duration:.2f}s"
                        )
                    else:
                        logger.error(f"🖼️ SD server error for {interaction.user}: HTTP {response.status}")
                        try:
                            await interaction.followup.send(
                                f"❌ SD server error: HTTP {response.status}", ephemeral=True
                            )
                        except Exception as send_error:
                            logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except (ClientConnectorError, ClientOSError) as e:
        error_detail = str(e)
        logger.error(f"🖼️ SD server is offline or unreachable for {interaction.user}. Error: {error_detail}")
        logger.error(f"🖼️ SD server URL: {SD_SERVER_URL}")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    f"❌ The SD server is currently offline or unreachable.\n"
                    f"Server: {SD_SERVER_URL}\n"
                    f"Error: {error_detail}",
                    ephemeral=True,
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except ServerTimeoutError:
        logger.error(f"🖼️ SD server request timed out for {interaction.user}.")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    "❌ The SD server timed out while processing your request. Please try again later.", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f"🖼️ Unexpected error during image generation for {interaction.user}: {e}")
        logger.error(f"🖼️ Full traceback:\n{error_details}")
        if isinstance(interaction, discord.Interaction):
            try:
                error_msg = str(e) if str(e) else "Unknown error - check logs for details"
                await interaction.followup.send(
                    f"❌ An unexpected error occurred during image generation: {error_msg}", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")


async def process_sd_image(interaction: discord.Interaction, description: str, size: str, seed: Optional[int]):
    """Entry point for slash command /sd tasks to push work into generate_sd_image."""
    try:
        # Default dims for SD
        width, height = SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT

        if size == "wide":
            width, height = SD_WIDE_WIDTH, SD_WIDE_HEIGHT
        elif size == "tall":
            width, height = SD_TALL_WIDTH, SD_TALL_HEIGHT
        elif size == "square":
            width, height = SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        logger.info(
            f"Processing request: user={interaction.user}, prompt='{description}', size='{size}', dims={width}x{height}, seed={seed}"
        )

        # Update image generation count with server ID
        await increment_user_stat(interaction.user.id, "images_generated", interaction.guild_id)

        await generate_sd_image(interaction, description, width, height, seed, queue_size=bot.sd_queue.qsize())
    except Exception as e:
        try:
            await interaction.followup.send(f"❌ An error occurred: {str(e)}", ephemeral=True)
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")
        logger.error(f"Error in process_sd_image: {e}")


async def setup(bot):
    """discord.py extension entry point.

    Registers the image-gen slash commands on the tree. They're plain
    ``@app_commands.command`` objects (registering here, in setup, rather than at
    module import keeps a stray ``import soupy.cogs.sd`` from double-registering).
    The handlers, views and SD pipeline are module-level functions; the main
    module's ``SDQueue.process_queue`` dispatches into them via a lazy import.
    """
    bot.tree.add_command(sd)
    bot.tree.add_command(img2img_cmd)
    bot.tree.add_command(inpaint_cmd)
    bot.tree.add_command(outpaint_cmd)
    logger.info("✅ Loaded sd (image generation) extension")
