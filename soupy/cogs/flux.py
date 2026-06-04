"""Local Flux image-generation cog (`/flux`).

A second image backend that runs **locally** on the Mac via ``flux_server.py``
(an mflux/MLX HTTP server), as opposed to the remote Stable Diffusion server the
``soupy.cogs.sd`` cog talks to. ``/flux`` mirrors ``/sd`` (text-to-image with the
same default/wide/tall/square sizes and the same Remix button panel) but also
accepts an optional ``image`` attachment to run **image-to-image**.

Buttons under a result mirror ``/sd``'s ``SDRemixView`` — Edit, Fancy, Remix,
R-Fancy, R-Keyword, Wide, Tall, 2x2 — minus Outpaint (the Flux server has no
outpaint/inpaint endpoint; schnell/klein can't outpaint natively). img2img
results additionally get a 🎚️ Strength button that re-runs the img2img with a
new strength; all other buttons act as text2img on the prompt (matching /sd).

Design notes:
- Work is funnelled through the single ``SDQueue`` in the main module so Flux and
  SD generations run one-at-a-time. ``SDQueue.process_queue`` dispatches
  ``type:"flux"`` items here to ``process_flux_image``, which branches on
  ``item["action"]``.
- The LLM prompt builders (Fancy / R-Fancy / R-Keyword) are reused from
  ``soupy.cogs.sd`` (``build_fancy_prompt`` / ``build_random_prompt``) so prompt
  behaviour matches /sd exactly.
- img2img sources are persisted under ``media/flux_sources/`` so the Strength
  button can re-run from the original image after the Discord attachment URL
  expires.
"""

from __future__ import annotations

import asyncio
import random
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import discord
from aiohttp import ClientConnectorError, ClientOSError, ServerTimeoutError
from discord import app_commands
from discord.ui import Modal, TextInput, View
from PIL import Image

from soupy import prompts as soupy_prompts
from soupy.settings import settings

# --- Resolve the LIVE main module (same trick as soupy.cogs.sd) ----------------
_mm = sys.modules.get("__main__")
if _mm is not None and hasattr(_mm, "SDQueue"):
    _main = _mm
else:  # imported as a module (tests, tooling) rather than run as the script
    import soupy_remastered_stablediffusion as _main  # noqa: E402

SD_DEFAULT_WIDTH = _main.SD_DEFAULT_WIDTH
SD_DEFAULT_HEIGHT = _main.SD_DEFAULT_HEIGHT
SD_WIDE_WIDTH = _main.SD_WIDE_WIDTH
SD_WIDE_HEIGHT = _main.SD_WIDE_HEIGHT
SD_TALL_WIDTH = _main.SD_TALL_WIDTH
SD_TALL_HEIGHT = _main.SD_TALL_HEIGHT
bot = _main.bot
logger = _main.logger
archive_sent_message = _main.archive_sent_message
increment_user_stat = _main.increment_user_stat
universal_cooldown_check = _main.universal_cooldown_check
_ensure_media_dirs = _main._ensure_media_dirs

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_RANDOM_DIMS = [
    (SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT),  # square
    (SD_WIDE_WIDTH, SD_WIDE_HEIGHT),  # wide
    (SD_TALL_WIDTH, SD_TALL_HEIGHT),  # tall
]


# --- shared helpers ------------------------------------------------------------
def _archive_image_bytes(*args, **kwargs) -> None:
    """Reuse the sd cog's archiver (loaded first) without a hard import cycle."""
    try:
        from soupy.cogs.sd import archive_image_bytes

        archive_image_bytes(*args, **kwargs)
    except Exception as e:  # archiving is best-effort
        logger.debug(f"flux archive_image_bytes failed: {e}")


def _flux_img2img_url() -> str:
    """Explicit FLUX_IMG2IMG_URL, else derived from the base server URL."""
    explicit = settings.flux_img2img_url
    if explicit:
        return explicit
    base = settings.flux_server_url.rstrip("/")
    return f"{base}/flux_img2img"


def _flux_edit_url() -> str:
    """Explicit FLUX_EDIT_URL, else derived from the base server URL."""
    explicit = settings.flux_edit_url
    if explicit:
        return explicit
    base = settings.flux_server_url.rstrip("/")
    return f"{base}/flux_edit"


def _dims_for_size(size: str) -> tuple[int, int]:
    if size == "wide":
        return SD_WIDE_WIDTH, SD_WIDE_HEIGHT
    if size == "tall":
        return SD_TALL_WIDTH, SD_TALL_HEIGHT
    return SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT  # default / square


def _flux_sources_dir() -> Path:
    d = _ensure_media_dirs() / "flux_sources"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist_flux_source(data: bytes) -> str:
    """Save an img2img source so the Strength button can reuse it later. Returns the filename."""
    name = f"{random.randint(100000, 999999)}.png"
    (_flux_sources_dir() / name).write_bytes(data)
    return name


def _load_flux_source(name: str) -> Optional[bytes]:
    try:
        return (_flux_sources_dir() / name).read_bytes()
    except Exception as e:
        logger.warning(f"flux source '{name}' could not be read: {e}")
        return None


async def _read_image_response(response) -> bytes:
    """Accept either a raw image body or a JSON {image: base64} envelope."""
    content_type = response.headers.get("Content-Type", "").lower()
    if content_type.startswith("application/json"):
        import base64

        data = await response.json()
        if "image" in data:
            return base64.b64decode(data["image"]) if isinstance(data["image"], str) else data["image"]
        if "image_bytes" in data:
            return data["image_bytes"]
        raise ValueError(f"JSON response missing image data: {list(data.keys())}")
    return await response.read()


def _make_session() -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, keepalive_timeout=30, enable_cleanup_closed=True)
    timeout = aiohttp.ClientTimeout(total=600, connect=10)
    return aiohttp.ClientSession(connector=connector, timeout=timeout, headers={"Connection": "keep-alive"})


# --- /flux slash command -------------------------------------------------------
@app_commands.command(name="flux", description="Generate an image locally with Flux (text2img, or img2img if you attach an image).")
@app_commands.describe(
    description="Description of the image to generate",
    size="Size of the image (ignored when an image is attached — source dims are used)",
    image="Optional input image — attach one to run image-to-image instead of text-to-image",
    strength="img2img only: how much to deviate from the input (0.0-1.0, lower preserves more)",
    seed="Seed for reproducible generation",
)
@app_commands.choices(
    size=[
        app_commands.Choice(name=f"Default ({SD_DEFAULT_WIDTH}x{SD_DEFAULT_HEIGHT})", value="default"),
        app_commands.Choice(name=f"Wide ({SD_WIDE_WIDTH}x{SD_WIDE_HEIGHT})", value="wide"),
        app_commands.Choice(name=f"Tall ({SD_TALL_WIDTH}x{SD_TALL_HEIGHT})", value="tall"),
        app_commands.Choice(name=f"Square ({SD_DEFAULT_WIDTH}x{SD_DEFAULT_HEIGHT})", value="square"),
    ]
)
async def flux(
    interaction: discord.Interaction,
    description: str,
    size: Optional[app_commands.Choice[str]] = None,
    image: Optional[discord.Attachment] = None,
    strength: Optional[app_commands.Range[float, 0.0, 1.0]] = None,
    seed: Optional[int] = None,
):
    if not settings.flux_server_url:
        await interaction.response.send_message(
            "❌ Flux is not configured (set FLUX_SERVER_URL and run flux_server.py).", ephemeral=True
        )
        return

    size_value = size.value if size else "default"

    image_url: Optional[str] = None
    if image is not None:
        if not any(image.filename.lower().endswith(ext) for ext in _IMAGE_EXTS):
            await interaction.response.send_message(
                "❌ Attached file must be a PNG/JPG/WEBP image for img2img.", ephemeral=True
            )
            return
        image_url = image.url

    mode = "img2img" if image_url else "text2img"
    logger.info(
        f"⚡ Slash Command 'flux' ({mode}) invoked by {interaction.user}: description='{description}', "
        f"size='{size_value}', strength='{strength}', seed='{seed if seed else 'random'}'"
    )

    try:
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)
    except Exception as e:
        logger.debug(f"Defer failed: {e}")

    await bot.sd_queue.put(
        {
            "type": "flux",
            "interaction": interaction,
            "description": description,
            "size": size_value,
            "seed": seed,
            "image_url": image_url,
            "strength": float(strength) if strength is not None else None,
        }
    )
    logger.info(f"⚡ Queued flux {mode} generation for {interaction.user}")


# --- core generation -----------------------------------------------------------
async def generate_flux_image(
    interaction,
    prompt,
    width,
    height,
    seed,
    *,
    image_url: Optional[str] = None,
    source_file: Optional[str] = None,
    strength: Optional[float] = None,
    action_name: str = "Flux",
    queue_size: int = 0,
    selected_terms: Optional[str] = None,
    pre_duration: float = 0.0,
):
    """POST to the local Flux server and post the result to Discord.

    img2img is used when ``image_url`` (fresh attachment) or ``source_file`` (a
    persisted source, for the Strength button) is given; otherwise text2img.
    """
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        server_url = settings.flux_server_url.rstrip("/")
        steps = settings.flux_steps
        guidance = settings.flux_guidance
        negative_prompt = soupy_prompts.load_prompt("sd_negative_prompt", fallback="")
        if strength is None:
            strength = settings.flux_default_strength

        # Resolve the img2img source (persisted file first, else download + persist).
        src_bytes: Optional[bytes] = None
        if source_file:
            src_bytes = _load_flux_source(source_file)
        elif image_url:
            async with _make_session() as dl:
                async with dl.get(image_url) as src_resp:
                    if src_resp.status != 200:
                        await interaction.followup.send("❌ Failed to download the input image.", ephemeral=True)
                        return
                    src_bytes = await src_resp.read()
            if src_bytes:
                source_file = _persist_flux_source(src_bytes)
        is_img2img = src_bytes is not None
        # Edit mode = klein-edit endpoint with concatenated reference-image
        # tokens. Stronger prompt following than the noise-mix /flux_img2img
        # path, but no `strength` knob and FLUX.2-klein only.
        is_edit_mode = is_img2img and settings.flux_edit_enabled

        async with interaction.channel.typing():
            async with _make_session() as session:
                image_start_time = time.perf_counter()

                if is_img2img:
                    try:
                        src_w, src_h = Image.open(BytesIO(src_bytes)).size
                        width, height = src_w, src_h
                    except Exception:
                        pass
                    form = aiohttp.FormData()
                    form.add_field("image", src_bytes, filename="source.png", content_type="image/png")
                    form.add_field("prompt", prompt)
                    form.add_field("width", str(width))
                    form.add_field("height", str(height))
                    form.add_field("seed", str(seed))
                    if is_edit_mode:
                        # Klein-edit: no strength, no negative_prompt; its own
                        # steps/guidance defaults (4 / 1.0 for distilled klein).
                        form.add_field("steps", str(settings.flux_edit_steps))
                        form.add_field("guidance_scale", str(settings.flux_edit_guidance))
                        response_cm = session.post(_flux_edit_url(), data=form)
                    else:
                        form.add_field("negative_prompt", negative_prompt)
                        form.add_field("steps", str(steps))
                        form.add_field("guidance_scale", str(guidance))
                        form.add_field("strength", str(strength))
                        response_cm = session.post(_flux_img2img_url(), data=form)
                else:
                    payload = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "steps": str(steps),
                        "guidance_scale": str(guidance),
                        "width": str(width),
                        "height": str(height),
                        "seed": str(seed),
                    }
                    response_cm = session.post(f"{server_url}/flux", data=payload)

                async with response_cm as response:
                    if response.status != 200:
                        logger.error(f"⚡ Flux server error for {interaction.user}: HTTP {response.status}")
                        try:
                            await interaction.followup.send(
                                f"❌ Flux server error: HTTP {response.status}", ephemeral=True
                            )
                        except Exception as send_error:
                            logger.error(f"❌ Failed to send follow-up message: {send_error}")
                        return
                    image_bytes = await _read_image_response(response)

                if not image_bytes or len(image_bytes) < 100:
                    raise ValueError(f"Received invalid image data: {len(image_bytes) if image_bytes else 0} bytes")
                if not (
                    image_bytes.startswith(b"\x89PNG")
                    or image_bytes.startswith(b"\xff\xd8\xff")
                    or image_bytes.startswith(b"GIF")
                ):
                    logger.warning("⚡ Flux image data lacks PNG/JPEG/GIF magic bytes, continuing anyway")

                total_duration = pre_duration + (time.perf_counter() - image_start_time)
                logger.info(f"⏱️ Flux generation time for {interaction.user}: {total_duration:.2f}s")

                random_number = random.randint(100000, 999999)
                safe_prompt = re.sub(r"\W+", "", prompt[:40]).lower()
                filename = f"{random_number}_{safe_prompt}.png"
                archive_source = (
                    "flux-edit" if is_edit_mode
                    else "flux-img2img" if is_img2img
                    else "flux"
                )
                _archive_image_bytes(
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
                    source=archive_source,
                )

                image_file = discord.File(BytesIO(image_bytes), filename=filename)
                if selected_terms and selected_terms != prompt:
                    desc = f"**Selected Terms:** {selected_terms}\n\n**Prompt:** {prompt}"
                else:
                    desc = f"**Prompt:** {prompt}"
                description_embed = discord.Embed(description=desc, color=discord.Color.blue())
                details_embed = discord.Embed(color=discord.Color.green())
                queue_total = queue_size + 1
                effective_action_name = action_name
                if is_edit_mode and action_name == "Flux":
                    effective_action_name = "Edit"
                details_text = (
                    f"🌱 {seed} ⚡ {effective_action_name} "
                    f"⏱️ {total_duration:.2f}s 📋 {queue_total}"
                )
                if is_img2img and not is_edit_mode:
                    details_text += f" 🎚️ {strength}"
                details_embed.description = details_text

                new_view = FluxRemixView(
                    prompt=prompt, width=width, height=height, seed=seed,
                    is_img2img=is_img2img, source_file=source_file,
                    is_edit_mode=is_edit_mode,
                )

                content = f"{interaction.user.mention} ⚡ Flux Image:"
                if interaction.response.is_done():
                    try:
                        await interaction.followup.send(
                            content=content, embeds=[description_embed, details_embed],
                            file=image_file, view=new_view, ephemeral=False,
                        )
                    except Exception as send_error:
                        logger.error(f"❌ Flux follow-up failed (falling back to channel.send): {send_error}")
                        await interaction.channel.send(
                            content=content, embeds=[description_embed, details_embed],
                            file=image_file, view=new_view,
                        )
                else:
                    await interaction.channel.send(
                        content=content, embeds=[description_embed, details_embed],
                        file=image_file, view=new_view,
                    )

                try:
                    archive_sent_message(
                        content=f"Generated flux image: {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
                        user_id=interaction.user.id,
                        username=str(interaction.user),
                        guild_id=(interaction.guild.id if interaction.guild else None),
                        channel_id=(interaction.channel.id if interaction.channel else None),
                        image_filename=filename,
                        event_type="image_generation",
                    )
                except Exception:
                    pass
                logger.info(f"⚡ Flux generation completed for {interaction.user}: filename='{filename}'")

    except (ClientConnectorError, ClientOSError) as e:
        logger.error(f"⚡ Flux server offline/unreachable for {interaction.user}: {e}")
        try:
            await interaction.followup.send(
                f"❌ The Flux server is currently offline or unreachable.\n"
                f"Server: {settings.flux_server_url}\nError: {e}",
                ephemeral=True,
            )
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except (ServerTimeoutError, asyncio.TimeoutError):
        logger.error(f"⚡ Flux server request timed out for {interaction.user}.")
        try:
            await interaction.followup.send(
                "❌ The Flux server timed out while processing your request. Please try again later.", ephemeral=True
            )
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except Exception as e:
        import traceback

        logger.error(f"⚡ Unexpected error during flux generation for {interaction.user}: {e}")
        logger.error(f"⚡ Full traceback:\n{traceback.format_exc()}")
        try:
            await interaction.followup.send(
                f"❌ An unexpected error occurred during flux generation: {e or 'see logs'}", ephemeral=True
            )
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")


# --- queue dispatch ------------------------------------------------------------
async def process_flux_image(item: dict):
    """Entry point for ``type:"flux"`` queue items; branches on ``item["action"]``."""
    interaction = item["interaction"]
    action = item.get("action", "flux")
    try:
        if action == "fancy":
            await _handle_flux_fancy(item)
            return
        if action == "random":
            await _handle_flux_random(item)
            return
        if action == "2x2_grid":
            await handle_flux_2x2_grid(item)
            return

        # Basic generations: flux / edit / remix / wide / tall / strength / regenerate_selected
        seed = item.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        if item.get("width") and item.get("height"):
            width, height = item["width"], item["height"]
        else:
            width, height = _dims_for_size(item.get("size", "default"))

        await increment_user_stat(interaction.user.id, "images_generated", interaction.guild_id)
        await generate_flux_image(
            interaction,
            item.get("description") or item.get("prompt"),
            width,
            height,
            seed,
            image_url=item.get("image_url"),
            source_file=item.get("source_file"),
            strength=item.get("strength"),
            action_name=item.get("action_name", "Flux"),
            queue_size=bot.sd_queue.qsize(),
        )
    except Exception as e:
        logger.error(f"Error in process_flux_image (action={action}): {e}")
        try:
            await interaction.followup.send(f"❌ An error occurred: {e}", ephemeral=True)
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")


async def _handle_flux_fancy(item: dict):
    interaction = item["interaction"]
    from soupy.cogs.sd import build_fancy_prompt

    if not interaction.response.is_done():
        await interaction.response.defer()
    cleaned_prompt, duration = await build_fancy_prompt(item["prompt"])
    await increment_user_stat(interaction.user.id, "images_generated", interaction.guild_id)
    await generate_flux_image(
        interaction, cleaned_prompt, item["width"], item["height"], item["seed"],
        action_name="Fancy", pre_duration=duration, queue_size=bot.sd_queue.qsize(),
    )


async def _handle_flux_random(item: dict):
    interaction = item["interaction"]
    from soupy.cogs.sd import build_random_prompt

    if not interaction.response.is_done():
        await interaction.response.defer()
    try:
        prompt, selected_terms, duration = await build_random_prompt(item.get("prompt"))
    except RuntimeError as e:
        await interaction.followup.send(f"❌ {e}", ephemeral=True)
        return
    new_seed = random.randint(0, 2**32 - 1)
    await increment_user_stat(interaction.user.id, "images_generated", interaction.guild_id)
    await generate_flux_image(
        interaction, prompt, item["width"], item["height"], new_seed,
        action_name="Random", selected_terms=selected_terms,
        pre_duration=duration, queue_size=bot.sd_queue.qsize(),
    )


async def handle_flux_2x2_grid(item: dict):
    """Generate four low-step candidates, composite a 2x2 grid, post selection buttons."""
    interaction = item["interaction"]
    prompt = item.get("prompt") or item.get("description")
    try:
        tw = th = 1024
        base_seed = item.get("seed")
        if base_seed is None:
            base_seed = random.randint(0, 2**32 - 1)
        thumb_seeds = [base_seed + i for i in range(4)]
        logger.info(f"🔲 Flux 2x2 grid for {interaction.user}: seeds={thumb_seeds}")

        server_url = settings.flux_server_url.rstrip("/")
        steps = settings.flux_steps
        guidance = settings.flux_guidance
        negative_prompt = soupy_prompts.load_prompt("sd_negative_prompt", fallback="")

        thumbnail_data: List[Dict] = []
        thumbnail_images: List[bytes] = []
        async with interaction.channel.typing():
            async with _make_session() as session:
                for i, thumb_seed in enumerate(thumb_seeds):
                    payload = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "steps": str(steps),
                        "guidance_scale": str(guidance),
                        "width": str(tw),
                        "height": str(th),
                        "seed": str(thumb_seed),
                    }
                    async with session.post(f"{server_url}/flux", data=payload) as response:
                        if response.status != 200:
                            raise Exception(f"Failed to generate candidate {i+1}/4: HTTP {response.status}")
                        image_bytes = await _read_image_response(response)
                    thumbnail_images.append(image_bytes)
                    thumbnail_data.append({"image_bytes": image_bytes, "seed": thumb_seed, "width": tw, "height": th})

        combined = Image.new("RGB", (tw * 2, th * 2))
        positions = [(0, 0), (tw, 0), (0, th), (tw, th)]
        for image_bytes, pos in zip(thumbnail_images, positions, strict=False):
            combined.paste(Image.open(BytesIO(image_bytes)).convert("RGB"), pos)
        out = BytesIO()
        combined.save(out, format="PNG")
        out.seek(0)

        random_number = random.randint(100000, 999999)
        safe_prompt = re.sub(r"\W+", "", prompt[:40]).lower()
        filename = f"{random_number}_{safe_prompt}_2x2grid.png"
        description_embed = discord.Embed(
            description=f"**Prompt:** {prompt}\n**Seeds:** {', '.join(map(str, thumb_seeds))}",
            color=discord.Color.purple(),
        )
        details_embed = discord.Embed(
            description=f"🔲 2x2 Grid ⏱️ {steps} steps each 📋 {bot.sd_queue.qsize() + 1}",
            color=discord.Color.green(),
        )
        view = FluxThumbnailSelectionView(prompt=prompt, thumbnail_data=thumbnail_data)
        await interaction.followup.send(
            content=f"{interaction.user.mention} 🔲 Flux 2x2 Grid:",
            embeds=[description_embed, details_embed],
            file=discord.File(out, filename=filename),
            view=view,
        )
        await increment_user_stat(interaction.user.id, "images_generated", interaction.guild_id)
    except Exception as e:
        logger.error(f"🔲 Error generating flux 2x2 grid for {interaction.user}: {e}")
        try:
            await interaction.followup.send(f"❌ Error generating 2x2 grid: {e}", ephemeral=True)
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")


# --- UI: modals + views --------------------------------------------------------
class FluxEditModal(Modal, title="⚡ Edit Flux Parameters"):
    def __init__(self, prompt: str, width: int, height: int, seed: Optional[int] = None):
        super().__init__()
        self.image_description = TextInput(
            label="📝 Image Description", style=discord.TextStyle.paragraph, default=prompt, required=True, max_length=2000
        )
        self.width_input = TextInput(
            label="📏 Width", style=discord.TextStyle.short, default=str(width), required=True, min_length=1, max_length=5
        )
        self.height_input = TextInput(
            label="📐 Height", style=discord.TextStyle.short, default=str(height), required=True, min_length=1, max_length=5
        )
        self.seed_input = TextInput(
            label="🌱 Seed", style=discord.TextStyle.short, default=str(seed) if seed is not None else "", required=False, max_length=10
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
                req_w = int(self.width_input.value.strip())
                req_h = int(self.height_input.value.strip())
            except ValueError:
                await interaction.followup.send("❌ Width and Height must be valid integers.", ephemeral=True)
                return

            def _mult64(v: int) -> int:
                return 64 if v <= 0 else ((v + 63) // 64) * 64

            seed_value = self.seed_input.value.strip()
            new_seed = int(seed_value) if seed_value.isdigit() else random.randint(0, 2**32 - 1)
            await bot.sd_queue.put(
                {
                    "type": "flux",
                    "interaction": interaction,
                    "action": "edit",
                    "prompt": new_prompt,
                    "width": _mult64(req_w),
                    "height": _mult64(req_h),
                    "seed": new_seed,
                    "action_name": "Edit",
                }
            )
        except Exception as e:
            logger.error(f"Error in FluxEditModal submission: {e}")
            await interaction.followup.send("❌ An error occurred while processing your edit.", ephemeral=True)


class FluxStrengthModal(Modal, title="🎚️ Change img2img Strength"):
    def __init__(self, prompt: str, width: int, height: int, seed: int, source_file: str, current: float):
        super().__init__()
        self.prompt = prompt
        self.width = width
        self.height = height
        self.seed = seed
        self.source_file = source_file
        self.strength_input = TextInput(
            label="🎚️ Strength (0.0 - 1.0)",
            style=discord.TextStyle.short,
            default=str(current),
            required=True,
            min_length=1,
            max_length=5,
            placeholder="e.g. 0.35 — lower keeps more of the original",
        )
        self.add_item(self.strength_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            raw = self.strength_input.value.strip()
            try:
                value = float(raw)
            except ValueError:
                await interaction.response.send_message("❌ Strength must be a number between 0.0 and 1.0.", ephemeral=True)
                return
            if not (0.0 <= value <= 1.0):
                await interaction.response.send_message("❌ Strength must be between 0.0 and 1.0.", ephemeral=True)
                return

            await interaction.response.send_message(f"🛠️ Re-running img2img at strength {value}...", ephemeral=True)
            await bot.sd_queue.put(
                {
                    "type": "flux",
                    "interaction": interaction,
                    "action": "strength",
                    "prompt": self.prompt,
                    "width": self.width,
                    "height": self.height,
                    "seed": self.seed,
                    "source_file": self.source_file,
                    "strength": value,
                    "action_name": "Strength",
                }
            )
        except Exception as e:
            logger.error(f"Error in FluxStrengthModal submission: {e}")
            try:
                await interaction.followup.send("❌ An error occurred while changing strength.", ephemeral=True)
            except Exception:
                pass


class FluxThumbnailSelectionView(View):
    """Pick one of the four 2x2 candidates to regenerate at full size/steps via Flux."""

    def __init__(self, prompt: str, thumbnail_data: List[Dict]):
        super().__init__(timeout=None)
        self.prompt = prompt
        self.thumbnail_data = thumbnail_data

    async def _select(self, interaction: discord.Interaction, index: int):
        try:
            await interaction.response.send_message("🛠️ Regenerating the selected candidate at full size...", ephemeral=True)
            chosen = self.thumbnail_data[index]
            await bot.sd_queue.put(
                {
                    "type": "flux",
                    "interaction": interaction,
                    "action": "regenerate_selected",
                    "prompt": self.prompt,
                    "width": chosen.get("width", 1024),
                    "height": chosen.get("height", 1024),
                    "seed": chosen["seed"],
                    "action_name": f"Selected #{index + 1}",
                }
            )
        except Exception as e:
            logger.error(f"Error during flux thumbnail selection for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating selected image.", ephemeral=True)

    @discord.ui.button(label="1", style=discord.ButtonStyle.primary, custom_id="fluxgen_thumb_1", row=0)
    @universal_cooldown_check()
    async def t1(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._select(interaction, 0)

    @discord.ui.button(label="2", style=discord.ButtonStyle.primary, custom_id="fluxgen_thumb_2", row=0)
    @universal_cooldown_check()
    async def t2(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._select(interaction, 1)

    @discord.ui.button(label="3", style=discord.ButtonStyle.primary, custom_id="fluxgen_thumb_3", row=1)
    @universal_cooldown_check()
    async def t3(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._select(interaction, 2)

    @discord.ui.button(label="4", style=discord.ButtonStyle.primary, custom_id="fluxgen_thumb_4", row=1)
    @universal_cooldown_check()
    async def t4(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._select(interaction, 3)


class FluxRemixView(View):
    """Full button panel for /flux results — mirrors SDRemixView (minus Outpaint).

    img2img results additionally show a 🎚️ Strength button (re-runs the img2img);
    every other button acts as text2img on the prompt.
    """

    def __init__(self, prompt: str, width: int, height: int, seed: Optional[int] = None,
                 *, is_img2img: bool = False, source_file: Optional[str] = None,
                 is_edit_mode: bool = False):
        super().__init__(timeout=None)
        self.prompt = prompt.strip()
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.is_img2img = is_img2img
        self.source_file = source_file
        self.is_edit_mode = is_edit_mode
        # The Strength button only makes sense for the noise-mix img2img path —
        # not text2img (no source) and not klein-edit (no strength knob).
        show_strength = is_img2img and bool(source_file) and not is_edit_mode
        if not show_strength:
            for child in list(self.children):
                if getattr(child, "custom_id", None) == "fluxgen_strength_button":
                    self.remove_item(child)

    async def _enqueue(self, interaction: discord.Interaction, msg: str, item: dict):
        await interaction.response.send_message(msg, ephemeral=True)
        item.update({"type": "flux", "interaction": interaction})
        await bot.sd_queue.put(item)

    # ----- row 0 -----
    @discord.ui.button(label="✏️", style=discord.ButtonStyle.success, custom_id="fluxgen_edit_button", row=0)
    @universal_cooldown_check()
    async def edit_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(
                FluxEditModal(prompt=self.prompt, width=self.width, height=self.height, seed=self.seed)
            )
        except Exception as e:
            logger.error(f"Error opening Flux edit modal for {interaction.user}: {e}")

    @discord.ui.button(label="🪄", style=discord.ButtonStyle.primary, custom_id="fluxgen_fancy_button", row=0)
    @universal_cooldown_check()
    async def fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._enqueue(
            interaction, "🛠️ Making it fancy...",
            {"action": "fancy", "prompt": self.prompt, "width": self.width, "height": self.height, "seed": self.seed},
        )

    @discord.ui.button(label="🌱🎲", style=discord.ButtonStyle.primary, custom_id="fluxgen_remix_button", row=0)
    @universal_cooldown_check()
    async def remix_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._enqueue(
            interaction, "🛠️ Remixing...",
            {"action": "remix", "prompt": self.prompt, "width": self.width, "height": self.height,
             "seed": random.randint(0, 2**32 - 1), "action_name": "Remix"},
        )

    @discord.ui.button(label="🎚️ Strength", style=discord.ButtonStyle.secondary, custom_id="fluxgen_strength_button", row=0)
    @universal_cooldown_check()
    async def strength_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(
                FluxStrengthModal(
                    prompt=self.prompt, width=self.width, height=self.height, seed=self.seed,
                    source_file=self.source_file, current=settings.flux_default_strength,
                )
            )
        except Exception as e:
            logger.error(f"Error opening Flux strength modal for {interaction.user}: {e}")

    # ----- row 1 -----
    @discord.ui.button(label="🎨 R-Fancy", style=discord.ButtonStyle.danger, custom_id="fluxgen_rfancy_button", row=1)
    @universal_cooldown_check()
    async def random_fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        w, h = random.choice(_RANDOM_DIMS)
        await self._enqueue(
            interaction, "🛠️ Generating fancy random image...",
            {"action": "random", "prompt": None, "width": w, "height": h, "seed": None},
        )

    @discord.ui.button(label="🔤 R-Keyword", style=discord.ButtonStyle.danger, custom_id="fluxgen_rkeyword_button", row=1)
    @universal_cooldown_check()
    async def random_keyword_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        from soupy.cogs.sd import get_random_terms

        w, h = random.choice(_RANDOM_DIMS)
        terms_list: List[str] = []
        for _category, terms in get_random_terms().items():
            terms_list.extend([t.strip() for t in terms.split(",")])
        await self._enqueue(
            interaction, "🛠️ Generating keyword random image...",
            {"action": "random", "prompt": ", ".join(terms_list), "width": w, "height": h, "seed": None},
        )

    @discord.ui.button(label="↔️", style=discord.ButtonStyle.primary, custom_id="fluxgen_wide_button", row=1)
    @universal_cooldown_check()
    async def wide_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._enqueue(
            interaction, "🛠️ Generating wide version...",
            {"action": "wide", "prompt": self.prompt, "width": SD_WIDE_WIDTH, "height": SD_WIDE_HEIGHT,
             "seed": self.seed, "action_name": "Wide"},
        )

    @discord.ui.button(label="↕️", style=discord.ButtonStyle.primary, custom_id="fluxgen_tall_button", row=1)
    @universal_cooldown_check()
    async def tall_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._enqueue(
            interaction, "🛠️ Generating tall version...",
            {"action": "tall", "prompt": self.prompt, "width": SD_TALL_WIDTH, "height": SD_TALL_HEIGHT,
             "seed": self.seed, "action_name": "Tall"},
        )

    @discord.ui.button(label="🔲 2x2", style=discord.ButtonStyle.secondary, custom_id="fluxgen_2x2_button", row=1)
    @universal_cooldown_check()
    async def grid_2x2_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._enqueue(
            interaction, "🛠️ Generating 2x2 grid...",
            {"action": "2x2_grid", "prompt": self.prompt, "width": self.width, "height": self.height, "seed": self.seed},
        )


async def setup(bot):
    """discord.py extension entry point — registers /flux on the tree."""
    bot.tree.add_command(flux)
    logger.info("✅ Loaded flux (local Flux image generation) extension")
