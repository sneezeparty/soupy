"""Local Flux image-generation cog (`/flux`).

A second image backend that runs **locally** on the Mac via ``flux_server.py``
(an mflux/MLX HTTP server), as opposed to the remote Stable Diffusion server the
``soupy.cogs.sd`` cog talks to. ``/flux`` mirrors ``/sd``'s Remix button panel
and also accepts an optional ``image`` attachment to run **image-to-image**.

**Dimensions are Klein-specific, not SD's.** FLUX.2-Klein was trained on a
~1 MP aspect-ratio bucket list (1024x1024, 1392x752, 752x1392, ...). The
``FLUX_*`` env vars in ``.env-stable`` map default/wide/tall to those buckets
— going off-distribution (or above ~1.05 MP) softens output and, on
klein-edit, trips Metal's max-buffer cap because the reference-token area
goes quadratic in attention.

Buttons under a result: Edit prompt, Fancy, Remix, Change image, R-Fancy,
R-Keyword, Wide, Tall, Square. There is no Outpaint (the Flux server has no
outpaint/inpaint endpoint; schnell/klein can't outpaint natively).

**An image plus a prompt runs one of two pipelines**, picked the same way for
``/flux`` with an attachment and for the 🖼️ Change image button:

- No strength: Klein edit (``/flux_edit``, needs ``FLUX_EDIT_ENABLED``). The
  prompt is an instruction about the picture ("make it a theme park", "add an
  alien") and the rest of the picture is kept.
- A strength: noise-mix img2img (``/flux_img2img``). The prompt describes the
  picture wanted and strength says how far to move toward it. This path can't
  follow an instruction.

Without the edit pipeline, an image always takes the noise-mix path, at
``FLUX_DEFAULT_STRENGTH`` when no strength is given.

**A result remembers what made it.** Its row in ``media/images/index.jsonl``
records the source image, the strength and the prompt a Fancy rewrite started
from, and a click finds that row by the attachment's filename. So on a picture
made from another one, Remix and Edit prompt redo the same change on the same
source, and Fancy rewrites the prompt the user typed, not its own last rewrite.
Fancy and the shape buttons are left off such a picture: its prompt is an
instruction and its shape is the source's.

Design notes:
- Work is funnelled through the single ``SDQueue`` in the main module so Flux and
  SD generations run one-at-a-time. ``SDQueue.process_queue`` dispatches
  ``type:"flux"`` items here to ``process_flux_image``, which branches on
  ``item["action"]``.
- The LLM prompt builders are imported from ``soupy.cogs.sd``. R-Keyword reuses
  ``build_random_prompt`` directly. Fancy and R-Fancy use Flux-tuned variants
  (``build_flux_fancy_prompt`` / ``build_flux_random_prompt``) that emit a
  shorter natural-language paragraph — Flux 2 handles those better than the
  longer CLIP-style output SD's Remix view still uses.
- A source image is named by its path under ``media/``: ``images/<file>`` for
  an archived result, ``flux_sources/<file>`` for an upload. Results are not
  copied into ``flux_sources``; the archive already has them.
- A click is answered with its place in the queue (``_Workload``), and that
  note is deleted when the job ends.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import aiohttp
import discord
from aiohttp import ClientConnectorError, ClientOSError, ServerTimeoutError
from discord import app_commands
from discord.ui import Label, Modal, Select, TextInput, View
from PIL import Image

from soupy.settings import settings

# --- Resolve the LIVE main module (same trick as soupy.cogs.sd) ----------------
_mm = sys.modules.get("__main__")
if _mm is not None and hasattr(_mm, "SDQueue"):
    _main = _mm
else:  # imported as a module (tests, tooling) rather than run as the script
    import soupy_remastered_stablediffusion as _main  # noqa: E402

bot = _main.bot
logger = _main.logger
archive_sent_message = _main.archive_sent_message
increment_user_stat = _main.increment_user_stat
universal_cooldown_check = _main.universal_cooldown_check
_ensure_media_dirs = _main._ensure_media_dirs

# Flux uses its own dimension presets — not the SD cog's — because FLUX.2-Klein
# was trained on a ~1 MP aspect-ratio bucket list and goes off-distribution
# (and trips Metal's max-buffer on klein-edit) above ~1.05 MP. Defaults map to
# the closest BFL training buckets: 1024x1024 (1:1), 1392x752 (~16:9 wide),
# 752x1392 (~9:16 tall).
FLUX_DEFAULT_WIDTH = settings.flux_default_width
FLUX_DEFAULT_HEIGHT = settings.flux_default_height
FLUX_WIDE_WIDTH = settings.flux_wide_width
FLUX_WIDE_HEIGHT = settings.flux_wide_height
FLUX_TALL_WIDTH = settings.flux_tall_width
FLUX_TALL_HEIGHT = settings.flux_tall_height

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_RANDOM_DIMS = [
    (FLUX_DEFAULT_WIDTH, FLUX_DEFAULT_HEIGHT),  # square
    (FLUX_WIDE_WIDTH, FLUX_WIDE_HEIGHT),  # wide
    (FLUX_TALL_WIDTH, FLUX_TALL_HEIGHT),  # tall
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
        return FLUX_WIDE_WIDTH, FLUX_WIDE_HEIGHT
    if size == "tall":
        return FLUX_TALL_WIDTH, FLUX_TALL_HEIGHT
    return FLUX_DEFAULT_WIDTH, FLUX_DEFAULT_HEIGHT  # default / square


# Mirrors flux_server.py's defaults — kept in sync so the cog's embed/buttons
# reflect the size flux_server will actually generate. The server is still the
# authority and clamps again on its side, but pre-clamping here means a phone
# photo doesn't display "4032x3024" on a result the server downsized to 1024.
_FLUX_MAX_PIXELS = 1024 * 1024
_FLUX_MAX_DIMENSION = 1536


def _clamp_src_dims(w: int, h: int) -> tuple[int, int]:
    """Fit (w, h) into the flux area+dim budget, snapped to multiples of 16."""
    if w <= 0 or h <= 0:
        return FLUX_DEFAULT_WIDTH, FLUX_DEFAULT_HEIGHT
    scale = 1.0
    if max(w, h) > _FLUX_MAX_DIMENSION:
        scale = _FLUX_MAX_DIMENSION / max(w, h)
    pixels = (w * scale) * (h * scale)
    if pixels > _FLUX_MAX_PIXELS:
        scale *= (_FLUX_MAX_PIXELS / pixels) ** 0.5
    new_w = max(16, (int(w * scale) // 16) * 16)
    new_h = max(16, (int(h * scale) // 16) * 16)
    return new_w, new_h


# --- source images -------------------------------------------------------------
# A source image is named by its path under media/: "images/<file>" for an
# archived result, "flux_sources/<file>" for an upload.
_SOURCE_DIRS = ("images", "flux_sources")
# Archive `source` values of a picture that was made from another picture.
_FROM_IMAGE_SOURCES = ("flux-edit", "flux-img2img")


def _source_path(ref: Optional[str]) -> Optional[Path]:
    """The file a source reference names, or None when it isn't one of ours."""
    folder, _, name = (ref or "").partition("/")
    if folder not in _SOURCE_DIRS or not name or Path(name).name != name:
        return None
    return _ensure_media_dirs() / folder / name


def _load_source(ref: Optional[str]) -> Optional[bytes]:
    path = _source_path(ref)
    if path is None:
        return None
    try:
        return path.read_bytes()
    except Exception as e:
        logger.warning(f"flux source '{ref}' could not be read: {e}")
        return None


def _persist_flux_source(data: bytes) -> str:
    """Save an uploaded source image. Returns its reference.

    Named by content, so the same upload twice is one file.
    """
    folder = _ensure_media_dirs() / "flux_sources"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{hashlib.sha256(data).hexdigest()[:16]}.png"
    if not (folder / name).exists():
        (folder / name).write_bytes(data)
    return f"flux_sources/{name}"


def _attachment_filename(message) -> str:
    attachments = getattr(message, "attachments", None) or []
    return str(getattr(attachments[0], "filename", "") or "") if attachments else ""


def _archived_source(filename: str) -> Optional[str]:
    """The reference for an archived result, when its file is still there."""
    ref = f"images/{filename}"
    path = _source_path(ref)
    return ref if path is not None and path.is_file() else None


def _find_record(filename: str) -> Optional[dict]:
    """The archive row for a posted result (see ``find_archived_image``)."""
    try:
        from soupy.cogs.sd import find_archived_image

        return find_archived_image(filename)
    except Exception as e:
        logger.debug(f"flux: archive lookup failed: {e}")
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


async def _error_detail(response) -> str:
    """Why the Flux server refused: FastAPI's ``detail``, else the start of the body."""
    try:
        text = await response.text()
    except Exception:
        return "no details"
    try:
        import json

        detail = json.loads(text).get("detail")
        if detail:
            text = detail if isinstance(detail, str) else json.dumps(detail)
    except Exception:
        pass
    text = " ".join(str(text).split())
    return text[:300] or "no details"


def _make_session() -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, keepalive_timeout=30, enable_cleanup_closed=True)
    timeout = aiohttp.ClientTimeout(total=600, connect=10)
    return aiohttp.ClientSession(connector=connector, timeout=timeout, headers={"Connection": "keep-alive"})


# --- the queue, as a click sees it ---------------------------------------------
class _Workload:
    """What Flux has queued and running, so a click can be told how many jobs are ahead of it."""

    def __init__(self) -> None:
        self.pending: List[dict] = []
        self.running: Optional[dict] = None

    def add(self) -> tuple[dict, int]:
        """Register a job. Returns ``(job, jobs ahead of it)``."""
        job: dict = {}
        ahead = len(self.pending) + (self.running is not None)
        self.pending.append(job)
        return job, ahead

    def start(self, job: Optional[dict]) -> None:
        if job is None:
            return
        self.pending = [j for j in self.pending if j is not job]
        self.running = job

    def finish(self, job: Optional[dict]) -> None:
        if job is None:
            return
        self.pending = [j for j in self.pending if j is not job]
        if self.running is job:
            self.running = None


_workload = _Workload()


async def _submit(interaction: discord.Interaction, item: dict, doing: str) -> None:
    """Answer a click with its place in the queue, then queue the job.

    The answer is a private note; ``process_flux_image`` deletes it when the
    job ends, so it doesn't outlive the picture it was about.
    """
    job, ahead = _workload.add()
    note = f"🛠️ {doing}"
    if ahead:
        note += f" {ahead} ahead of you."
    try:
        await interaction.response.send_message(note, ephemeral=True)
    except Exception:
        _workload.finish(job)
        raise
    item.update({"type": "flux", "interaction": interaction, "job": job, "ack": True})
    await bot.sd_queue.put(item)


async def _clear_ack(interaction: discord.Interaction) -> None:
    try:
        await interaction.delete_original_response()
    except Exception as e:  # already dismissed, or the 15-minute token ran out
        logger.debug(f"flux: could not delete the queue note: {e}")


# --- /flux slash command -------------------------------------------------------
@app_commands.command(name="flux", description="Generate an image locally with Flux (text2img, or img2img if you attach an image).")
@app_commands.describe(
    description="Description of the image to generate",
    size="Size of the image (ignored when an image is attached — source dims are used)",
    image="Optional input image — attach one to run image-to-image instead of text-to-image",
    strength="With an image: empty edits it by instruction; 0.0-1.0 repaints it (lower keeps more)",
    seed="Seed for reproducible generation",
    fancy="Have Soupy write a fuller prompt from the description first (text-to-image only)",
)
@app_commands.choices(
    size=[
        app_commands.Choice(name=f"Wide ({FLUX_WIDE_WIDTH}x{FLUX_WIDE_HEIGHT})", value="wide"),
        app_commands.Choice(name=f"Tall ({FLUX_TALL_WIDTH}x{FLUX_TALL_HEIGHT})", value="tall"),
        app_commands.Choice(name=f"Square ({FLUX_DEFAULT_WIDTH}x{FLUX_DEFAULT_HEIGHT})", value="square"),
    ]
)
async def flux(
    interaction: discord.Interaction,
    description: str,
    size: Optional[app_commands.Choice[str]] = None,
    image: Optional[discord.Attachment] = None,
    strength: Optional[app_commands.Range[float, 0.0, 1.0]] = None,
    seed: Optional[int] = None,
    fancy: Optional[bool] = None,
):
    if not settings.flux_server_url:
        await interaction.response.send_message(
            "❌ Flux is not configured (set FLUX_SERVER_URL and run flux_server.py).", ephemeral=True
        )
        return

    # Default to wide when no size is specified — matches the button panel's default.
    size_value = size.value if size else "wide"

    image_url: Optional[str] = None
    if image is not None:
        if not any(image.filename.lower().endswith(ext) for ext in _IMAGE_EXTS):
            await interaction.response.send_message(
                "❌ Attached file must be a PNG/JPG/WEBP image for img2img.", ephemeral=True
            )
            return
        image_url = image.url

    # A rewrite is for a description. With an image the text is an instruction.
    use_fancy = bool(fancy) and not image_url
    mode = "img2img" if image_url else "fancy text2img" if use_fancy else "text2img"
    logger.info(
        f"⚡ Slash Command 'flux' ({mode}) invoked by {interaction.user}: description='{description}', "
        f"size='{size_value}', strength='{strength}', seed='{seed if seed else 'random'}'"
    )

    try:
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)
    except Exception as e:
        logger.debug(f"Defer failed: {e}")

    if use_fancy:
        width, height = _dims_for_size(size_value)
        item = {
            "type": "flux",
            "interaction": interaction,
            "action": "fancy",
            "prompt": description,
            "width": width,
            "height": height,
            "seed": seed if seed is not None else random.randint(0, 2**32 - 1),
        }
    else:
        item = {
            "type": "flux",
            "interaction": interaction,
            "description": description,
            "size": size_value,
            "seed": seed,
            "image_url": image_url,
            "strength": float(strength) if strength is not None else None,
        }
    item["job"] = _workload.add()[0]
    await bot.sd_queue.put(item)
    logger.info(f"⚡ Queued flux {mode} generation for {interaction.user}")


# --- core generation -----------------------------------------------------------
def _prompt_text(prompt: str, *, original: Optional[str] = None, selected_terms: Optional[str] = None) -> str:
    """The first embed of a result. ``_view_state_from_message`` reads the prompt back out of it."""
    if selected_terms and selected_terms != prompt:
        return f"**Selected Terms:** {selected_terms}\n\n**Prompt:** {prompt}"
    if original and original != prompt:
        # The full text is in the archive row; this copy only has to fit the embed.
        shown = original if len(original) <= 1000 else original[:999].rstrip() + "…"
        return f"**Original:** {shown}\n\n**Prompt:** {prompt}"
    return f"**Prompt:** {prompt}"


async def generate_flux_image(
    interaction,
    prompt,
    width,
    height,
    seed,
    *,
    image_url: Optional[str] = None,
    source_image: Optional[str] = None,
    strength: Optional[float] = None,
    action_name: str = "Flux",
    queue_size: int = 0,
    selected_terms: Optional[str] = None,
    original_prompt: Optional[str] = None,
    pre_duration: float = 0.0,
):
    """POST to the local Flux server and post the result to Discord.

    A source image, given as ``source_image`` (a path under ``media/``) and/or
    ``image_url`` (a Discord attachment), makes this an image-to-image run;
    otherwise it is text2img. With a source, ``strength=None`` edits by
    instruction when the Klein edit pipeline is on, and a number runs noise-mix
    img2img at that strength. See the module docstring.

    ``original_prompt`` is the prompt a Fancy rewrite started from. It is shown
    above the rewrite and kept in the archive row.
    """
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        server_url = settings.flux_server_url.rstrip("/")
        steps = settings.flux_steps
        guidance = settings.flux_guidance

        # Resolve the source image: the saved file first, then the attachment.
        wants_source = bool(source_image or image_url)
        src_bytes: Optional[bytes] = None
        if source_image:
            src_bytes = _load_source(source_image)
        if not src_bytes and image_url:
            async with _make_session() as dl:
                async with dl.get(image_url) as src_resp:
                    if src_resp.status != 200:
                        await interaction.followup.send("❌ Failed to download the input image.", ephemeral=True)
                        return
                    src_bytes = await src_resp.read()
            if src_bytes:
                source_image = _persist_flux_source(src_bytes)
        if wants_source and not src_bytes:
            # Never fall through to text2img here: the prompt is about a picture,
            # and on its own it would generate something unrelated.
            logger.error(f"⚡ Flux source image missing for {interaction.user} (source={source_image!r})")
            await interaction.followup.send(
                "❌ Couldn't load the image this request starts from. It may have been deleted.", ephemeral=True
            )
            return
        is_img2img = bool(src_bytes)
        # Edit mode = klein-edit endpoint with concatenated reference-image
        # tokens: it follows an instruction and keeps the rest of the picture,
        # has no `strength` knob, and is FLUX.2-klein only. Asking for a
        # strength is how a caller picks the noise-mix path instead.
        is_edit_mode = is_img2img and settings.flux_edit_enabled and strength is None
        # What the archive row keeps: None for an edit, which has no strength.
        asked_strength = strength
        if strength is None:
            strength = settings.flux_default_strength

        async with interaction.channel.typing():
            async with _make_session() as session:
                image_start_time = time.perf_counter()

                # Flux has no negative prompt (klein and schnell run without
                # CFG), so sd_negative_prompt is not sent.
                if is_img2img:
                    try:
                        src_w, src_h = Image.open(BytesIO(src_bytes)).size
                        # Cap source-derived dims so a 12-megapixel phone photo
                        # doesn't ask flux_server for a 12-megapixel generation
                        # (Klein-Edit would OOM at attention time; the server
                        # also clamps defensively).
                        width, height = _clamp_src_dims(src_w, src_h)
                    except Exception:
                        pass
                    form = aiohttp.FormData()
                    form.add_field("image", src_bytes, filename="source.png", content_type="image/png")
                    form.add_field("prompt", prompt)
                    form.add_field("width", str(width))
                    form.add_field("height", str(height))
                    form.add_field("seed", str(seed))
                    if is_edit_mode:
                        # Klein-edit: no strength; its own steps/guidance
                        # defaults (4 / 1.0 for distilled klein).
                        steps = settings.flux_edit_steps
                        guidance = settings.flux_edit_guidance
                        form.add_field("steps", str(steps))
                        form.add_field("guidance_scale", str(guidance))
                        response_cm = session.post(_flux_edit_url(), data=form)
                    else:
                        form.add_field("steps", str(steps))
                        form.add_field("guidance_scale", str(guidance))
                        form.add_field("strength", str(strength))
                        response_cm = session.post(_flux_img2img_url(), data=form)
                else:
                    payload = {
                        "prompt": prompt,
                        "steps": str(steps),
                        "guidance_scale": str(guidance),
                        "width": str(width),
                        "height": str(height),
                        "seed": str(seed),
                    }
                    response_cm = session.post(f"{server_url}/flux", data=payload)

                async with response_cm as response:
                    if response.status != 200:
                        detail = await _error_detail(response)
                        logger.error(
                            f"⚡ Flux server error for {interaction.user}: HTTP {response.status}: {detail}"
                        )
                        try:
                            await interaction.followup.send(
                                f"❌ Flux server error (HTTP {response.status}): {detail}", ephemeral=True
                            )
                        except Exception as send_error:
                            logger.error(f"❌ Failed to send follow-up message: {send_error}")
                        return
                    image_bytes = await _read_image_response(response)
                    # The server names the model it ran; FLUX_MODEL on the bot's
                    # side can be out of date (./flux_server.sh <model>).
                    model_name = response.headers.get("X-Flux-Model") or (
                        settings.flux_edit_model if is_edit_mode else settings.flux_model
                    )

                if not image_bytes or len(image_bytes) < 100:
                    raise ValueError(f"Received invalid image data: {len(image_bytes) if image_bytes else 0} bytes")
                if not (
                    image_bytes.startswith(b"\x89PNG")
                    or image_bytes.startswith(b"\xff\xd8\xff")
                    or image_bytes.startswith(b"GIF")
                ):
                    logger.warning("⚡ Flux image data lacks PNG/JPEG/GIF magic bytes, continuing anyway")

                server_seconds = time.perf_counter() - image_start_time
                total_duration = pre_duration + server_seconds
                logger.info(f"⏱️ Flux generation time for {interaction.user}: {total_duration:.2f}s")

                random_number = random.randint(100000, 999999)
                # ASCII only: a click finds this picture's archive row by the
                # attachment's filename, and Discord rewrites other characters.
                safe_prompt = re.sub(r"\W+", "", prompt[:40], flags=re.ASCII).lower()
                filename = f"{random_number}_{safe_prompt}.png"
                archive_source = (
                    "flux-edit" if is_edit_mode
                    else "flux-img2img" if is_img2img
                    else "flux"
                )
                effective_action_name = "Edit" if is_edit_mode else action_name
                if original_prompt == prompt:
                    original_prompt = None
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
                    extra={
                        "action": effective_action_name,
                        "model": model_name,
                        "steps": steps,
                        "guidance": guidance,
                        "strength": asked_strength if is_img2img and not is_edit_mode else None,
                        "source_image": source_image if is_img2img else None,
                        "original_prompt": original_prompt,
                        "seconds": round(total_duration, 1),
                    },
                )

                image_file = discord.File(BytesIO(image_bytes), filename=filename)
                description_embed = discord.Embed(
                    description=_prompt_text(prompt, original=original_prompt, selected_terms=selected_terms),
                    color=discord.Color.blue(),
                )
                details_embed = discord.Embed(color=discord.Color.green())
                queue_total = queue_size + 1
                details_text = (
                    f"🌱 {seed} ⚡ {effective_action_name} "
                    f"⏱️ {total_duration:.2f}s 📋 {queue_total}"
                )
                if is_img2img and not is_edit_mode:
                    details_text += f" 🎚️ {strength}"
                details_embed.description = details_text

                new_view = FluxRemixView(
                    prompt=prompt, width=width, height=height, seed=seed,
                    from_image=is_img2img, source_image=source_image if is_img2img else None,
                    strength=asked_strength if is_img2img and not is_edit_mode else None,
                    original=original_prompt,
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
    _workload.start(item.get("job"))
    try:
        if action == "fancy":
            await _handle_flux_fancy(item)
            return
        if action == "random":
            await _handle_flux_random(item)
            return
        # Basic generations: flux / reprompt / remix / wide / tall / img2img
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
            source_image=item.get("source_image"),
            strength=item.get("strength"),
            action_name=item.get("action_name", "Flux"),
            original_prompt=item.get("original"),
            queue_size=bot.sd_queue.qsize(),
        )
    except Exception as e:
        logger.error(f"Error in process_flux_image (action={action}): {e}")
        try:
            await interaction.followup.send(f"❌ An error occurred: {e}", ephemeral=True)
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")
    finally:
        _workload.finish(item.get("job"))
        if item.get("ack"):
            await _clear_ack(interaction)


async def _handle_flux_fancy(item: dict):
    interaction = item["interaction"]
    from soupy.cogs.sd import build_flux_fancy_prompt

    if not interaction.response.is_done():
        await interaction.response.defer()
    # Always rewrite what the user typed: a rewrite of a rewrite drifts.
    original = item.get("original") or item["prompt"]
    cleaned_prompt, duration = await build_flux_fancy_prompt(original)
    await increment_user_stat(interaction.user.id, "images_generated", interaction.guild_id)
    await generate_flux_image(
        interaction, cleaned_prompt, item["width"], item["height"], item["seed"],
        action_name="Fancy", original_prompt=original,
        pre_duration=duration, queue_size=bot.sd_queue.qsize(),
    )


async def _handle_flux_random(item: dict):
    interaction = item["interaction"]
    from soupy.cogs.sd import build_flux_random_prompt

    if not interaction.response.is_done():
        await interaction.response.defer()
    try:
        prompt, selected_terms, duration = await build_flux_random_prompt(item.get("prompt"))
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


# --- UI: modals + views --------------------------------------------------------
def _shape_options(width: int, height: int) -> List[discord.SelectOption]:
    """Wide / Tall / Square for the Edit prompt dialog, with the picture's own shape preselected."""
    presets = [
        ("wide", "Wide", FLUX_WIDE_WIDTH, FLUX_WIDE_HEIGHT),
        ("tall", "Tall", FLUX_TALL_WIDTH, FLUX_TALL_HEIGHT),
        ("square", "Square", FLUX_DEFAULT_WIDTH, FLUX_DEFAULT_HEIGHT),
    ]
    current = next((value for value, _name, w, h in presets if (w, h) == (width, height)), None)
    options = [
        discord.SelectOption(label=f"{name} ({w}×{h})", value=value, default=value == current)
        for value, name, w, h in presets
    ]
    if current is None:
        # An older picture, or one whose sizes were set differently in .env-stable.
        options.insert(0, discord.SelectOption(label=f"Same as now ({width}×{height})", value="keep", default=True))
    return options


class FluxEditModal(Modal):
    """Edit prompt: run the picture's recipe again with a changed prompt, shape or seed.

    For a picture made from another one (``from_image``) the recipe is a change
    to that earlier picture, so this redoes the step from there and there is no
    shape to pick. Changing the clicked picture itself is the 🖼️ button's job,
    and the dialog's wording keeps the two apart.
    """

    def __init__(self, prompt: str, width: int, height: int, seed: Optional[int] = None, *,
                 original: Optional[str] = None, from_image: bool = False,
                 source_image: Optional[str] = None, strength: Optional[float] = None):
        super().__init__(title="✏️ Redo from the previous image" if from_image else "✏️ Edit prompt")
        self.prompt = prompt
        self.width = width
        self.height = height
        self.original = original
        self.from_image = from_image
        self.source_image = source_image
        self.strength = strength
        self.image_description = TextInput(
            style=discord.TextStyle.paragraph, default=prompt[:2000], required=True, max_length=2000
        )
        if from_image:
            self.add_item(Label(
                text="📝 Change to make to the previous image" if strength is None else "📝 Prompt",
                description="Starts again from the image this one was made from. 🖼️ Change image starts from this one.",
                component=self.image_description,
            ))
        else:
            self.add_item(Label(text="📝 Prompt", component=self.image_description))
        self.shape_select: Optional[Select] = None
        if not from_image:
            self.shape_select = Select(options=_shape_options(width, height), required=True)
            self.add_item(Label(text="📐 Shape", component=self.shape_select))
        self.seed_input = TextInput(
            style=discord.TextStyle.short, default=str(seed) if seed is not None else "", required=False, max_length=10
        )
        self.add_item(Label(
            text="🌱 Seed", description="Blank picks a random one.", component=self.seed_input,
        ))

    def _chosen_dims(self) -> tuple[int, int]:
        chosen = self.shape_select.values[0] if self.shape_select and self.shape_select.values else "keep"
        if chosen == "keep":
            return self.width, self.height
        return _dims_for_size(chosen)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            new_prompt = self.image_description.value.strip()
            if not new_prompt:
                await interaction.response.send_message("❌ Prompt cannot be empty.", ephemeral=True)
                return
            seed_value = self.seed_input.value.strip()
            new_seed = int(seed_value) if seed_value.isdigit() else random.randint(0, 2**32 - 1)
            if self.from_image:
                await _submit(
                    interaction,
                    {
                        "action": "img2img",
                        "prompt": new_prompt,
                        "strength": self.strength,
                        "seed": new_seed,
                        "action_name": "Img2Img",
                        "source_image": self.source_image,
                    },
                    "Redoing it from the previous image...",
                )
                return
            width, height = self._chosen_dims()
            await _submit(
                interaction,
                {
                    "action": "reprompt",
                    "prompt": new_prompt,
                    "width": width,
                    "height": height,
                    "seed": new_seed,
                    "action_name": "Reprompt",
                    # Still the rewrite of what the user first typed, unless they changed it.
                    "original": self.original if new_prompt == self.prompt.strip() else None,
                },
                "Generating...",
            )
        except Exception as e:
            logger.error(f"Error in FluxEditModal submission: {e}")
            try:
                await interaction.followup.send("❌ An error occurred while processing your edit.", ephemeral=True)
            except Exception:
                pass


class FluxImg2ImgModal(Modal, title="🖼️ Change this image"):
    """What to do with the displayed image, and optionally how hard to push.

    With the Klein edit pipeline on (``edit_available``), the text is an
    instruction and Strength is left blank. Filling Strength in switches to the
    noise-mix path, where the text describes the picture wanted. Without the
    edit pipeline only the noise-mix path exists, so Strength is required and
    starts at ``default_strength``.
    """

    def __init__(self, source_image: Optional[str], default_strength: float,
                 source_url: Optional[str] = None, *, edit_available: bool = False):
        super().__init__()
        # The archived copy of the displayed image and/or the message
        # attachment's URL. generate_flux_image tries the file first, then the URL.
        self.source_image = source_image
        self.source_url = source_url
        self.default_strength = default_strength
        self.edit_available = edit_available
        if edit_available:
            self.prompt_input = TextInput(
                label="📝 What should change?",
                style=discord.TextStyle.paragraph,
                required=True,
                max_length=2000,
                placeholder="make it a theme park / add an alien to the scene / in the style of a pixar movie",
            )
            self.strength_input = TextInput(
                label="🎚️ Strength (leave blank to edit)",
                style=discord.TextStyle.short,
                required=False,
                max_length=5,
                placeholder="Blank follows the instruction. 0.1-1.0 repaints toward the text; low keeps more.",
            )
        else:
            self.prompt_input = TextInput(
                label="📝 Prompt",
                style=discord.TextStyle.paragraph,
                required=True,
                max_length=2000,
                placeholder="Describe what you want the new image to look like",
            )
            self.strength_input = TextInput(
                label="🎚️ Strength (0.0 - 1.0)",
                style=discord.TextStyle.short,
                default=str(default_strength),
                required=True,
                min_length=1,
                max_length=5,
                placeholder="e.g. 0.6. Lower keeps more of the source",
            )
        self.add_item(self.prompt_input)
        self.add_item(self.strength_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            new_prompt = self.prompt_input.value.strip()
            if not new_prompt:
                await interaction.response.send_message("❌ Prompt cannot be empty.", ephemeral=True)
                return
            # None = no strength given: edit by instruction (see generate_flux_image).
            strength: Optional[float] = None
            raw_strength = self.strength_input.value.strip()
            if raw_strength:
                try:
                    strength = float(raw_strength)
                except ValueError:
                    strength = -1.0
                if not (0.0 <= strength <= 1.0):
                    await interaction.response.send_message(
                        "❌ Strength must be a number between 0.0 and 1.0, or blank.", ephemeral=True
                    )
                    return
            elif not self.edit_available:
                strength = self.default_strength

            mode = "edit" if strength is None else f"strength {strength}"
            logger.info(f"🖼️ Flux img2img requested by {interaction.user}: {mode}, prompt='{new_prompt}'")
            await _submit(
                interaction,
                {
                    "action": "img2img",
                    "prompt": new_prompt,
                    "strength": strength,
                    "seed": random.randint(0, 2**32 - 1),
                    "action_name": "Img2Img",
                    "source_image": self.source_image,
                    "image_url": self.source_url,
                },
                "Editing the image..." if strength is None else f"Running img2img at strength {strength}...",
            )
        except Exception as e:
            logger.error(f"Error in FluxImg2ImgModal submission: {e}")
            try:
                await interaction.followup.send("❌ An error occurred while starting img2img.", ephemeral=True)
            except Exception:
                pass


class FluxRemixView(View):
    """The buttons under a /flux result.

    A picture made from another one (``from_image``) gets no Fancy and no shape
    buttons: its prompt is an instruction and its shape is the source's. The
    handlers still answer those buttons, because results posted before this
    rule have them.
    """

    def __init__(self, prompt: str, width: int, height: int, seed: Optional[int] = None,
                 *, from_image: bool = False, source_image: Optional[str] = None,
                 strength: Optional[float] = None, original: Optional[str] = None):
        super().__init__(timeout=None)
        self.prompt = prompt.strip()
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        # What made the picture, for a click before the next restart. After one,
        # the persistent template handles the click with none of this, and
        # _effective_state reads it from the message and the archive.
        self.from_image = from_image
        self.source_image = source_image
        self.strength = strength
        self.original = original
        if from_image:
            for button in (self.fancy_button, self.wide_button, self.tall_button, self.square_button):
                self.remove_item(button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if settings.flux_enabled:
            return True
        await interaction.response.send_message("❌ Flux is turned off on this bot (FLUX_ENABLED).", ephemeral=True)
        return False

    def _effective_state(self, interaction: discord.Interaction) -> dict:
        """What made the clicked picture: prompt, seed and size, and its source if it had one.

        The message gives the prompt, seed and size (the embed format matches
        sd.py's, so its parser works here). The archive row gives the rest. The
        view's own fields cover a picture whose row is missing.
        """
        from soupy.cogs.sd import _view_state_from_message

        parsed = _view_state_from_message(interaction.message)
        record = _find_record(_attachment_filename(interaction.message))
        if record:
            from_image = record.get("source") in _FROM_IMAGE_SOURCES
            source_image, strength = record.get("source_image"), record.get("strength")
            original = record.get("original_prompt")
        else:
            from_image, source_image, strength = self.from_image, self.source_image, self.strength
            original = parsed.get("original") or self.original
        return {
            "prompt": parsed["prompt"] or self.prompt,
            "width": parsed["width"] or self.width or FLUX_DEFAULT_WIDTH,
            "height": parsed["height"] or self.height or FLUX_DEFAULT_HEIGHT,
            "seed": parsed["seed"] if parsed["seed"] is not None else (self.seed or random.randint(0, 2**32 - 1)),
            "original": original,
            "from_image": from_image,
            "source_image": source_image,
            "strength": strength,
        }

    async def _no_prompt(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "❌ Couldn't recover the original prompt from this message.", ephemeral=True
        )

    async def _text_only(self, interaction: discord.Interaction, what: str) -> None:
        await interaction.response.send_message(
            f"This picture was made by changing another one, so {what} doesn't apply to it. "
            "🎲 Remix redoes that change with a new seed, and 🖼️ Change image changes this picture.",
            ephemeral=True,
        )

    async def _source_unknown(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "❌ This picture was made from another one before Soupy kept track of which, so the change "
            "can't be redone. 🖼️ Change image still works on it.",
            ephemeral=True,
        )

    # ----- row 0 -----
    @discord.ui.button(label="✏️ Edit prompt", style=discord.ButtonStyle.success, custom_id="fluxgen_edit_button", row=0)
    @universal_cooldown_check()
    async def edit_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        state = self._effective_state(interaction)
        if state["from_image"] and not state["source_image"]:
            await self._source_unknown(interaction)
            return
        try:
            await interaction.response.send_modal(
                FluxEditModal(
                    prompt=state["prompt"] or "", width=state["width"], height=state["height"], seed=state["seed"],
                    original=state["original"], from_image=state["from_image"],
                    source_image=state["source_image"], strength=state["strength"],
                )
            )
        except Exception as e:
            logger.error(f"Error opening Flux edit modal for {interaction.user}: {e}")

    @discord.ui.button(label="🪄 Fancy", style=discord.ButtonStyle.primary, custom_id="fluxgen_fancy_button", row=0)
    @universal_cooldown_check()
    async def fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        state = self._effective_state(interaction)
        if not state["prompt"]:
            await self._no_prompt(interaction)
            return
        if state["from_image"]:
            await self._text_only(interaction, "🪄 Fancy")
            return
        # The clicked picture's shape and seed; the prompt the user typed.
        await _submit(
            interaction,
            {"action": "fancy", "prompt": state["prompt"], "original": state["original"],
             "width": state["width"], "height": state["height"], "seed": state["seed"]},
            "Making it fancy...",
        )

    @discord.ui.button(label="🎲 Remix", style=discord.ButtonStyle.primary, custom_id="fluxgen_remix_button", row=0)
    @universal_cooldown_check()
    async def remix_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        state = self._effective_state(interaction)
        if not state["prompt"]:
            await self._no_prompt(interaction)
            return
        new_seed = random.randint(0, 2**32 - 1)
        if state["from_image"]:
            if not state["source_image"]:
                await self._source_unknown(interaction)
                return
            # The same change to the same source, with a new seed.
            await _submit(
                interaction,
                {"action": "img2img", "prompt": state["prompt"], "strength": state["strength"],
                 "seed": new_seed, "action_name": "Remix", "source_image": state["source_image"]},
                "Redoing the change with a new seed...",
            )
            return
        await _submit(
            interaction,
            {"action": "remix", "prompt": state["prompt"], "original": state["original"],
             "width": state["width"], "height": state["height"], "seed": new_seed, "action_name": "Remix"},
            "Remixing...",
        )

    @discord.ui.button(label="🖼️ Change image", style=discord.ButtonStyle.secondary, custom_id="fluxgen_img2img_button", row=0)
    @universal_cooldown_check()
    async def img2img_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Always the displayed picture, so a chain of changes runs A -> B -> C
        # and never back from A. It is in the archive under the attachment's
        # name; the attachment itself covers an archive file that was deleted.
        message = interaction.message
        source_image = _archived_source(_attachment_filename(message))
        source_url = None
        if message and message.attachments:
            source_url = message.attachments[0].url
        if not source_image and not source_url:
            await interaction.response.send_message(
                "❌ No source image available for img2img.", ephemeral=True
            )
            return
        try:
            await interaction.response.send_modal(
                FluxImg2ImgModal(
                    source_image=source_image,
                    default_strength=settings.flux_default_strength,
                    source_url=source_url,
                    edit_available=settings.flux_edit_enabled,
                )
            )
        except Exception as e:
            logger.error(f"Error opening Flux img2img modal for {interaction.user}: {e}")

    # ----- row 1 -----
    @discord.ui.button(label="🎨 R-Fancy", style=discord.ButtonStyle.secondary, custom_id="fluxgen_rfancy_button", row=1)
    @universal_cooldown_check()
    async def random_fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # A new, unrelated picture: wide, whatever the clicked one was.
        await _submit(
            interaction,
            {"action": "random", "prompt": None, "width": FLUX_WIDE_WIDTH, "height": FLUX_WIDE_HEIGHT, "seed": None},
            "Generating fancy random image...",
        )

    @discord.ui.button(label="🔤 R-Keyword", style=discord.ButtonStyle.secondary, custom_id="fluxgen_rkeyword_button", row=1)
    @universal_cooldown_check()
    async def random_keyword_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        from soupy.cogs.sd import get_random_terms

        terms_list: List[str] = []
        for _category, terms in get_random_terms().items():
            terms_list.extend([t.strip() for t in terms.split(",")])
        await _submit(
            interaction,
            {"action": "random", "prompt": ", ".join(terms_list),
             "width": FLUX_WIDE_WIDTH, "height": FLUX_WIDE_HEIGHT, "seed": None},
            "Generating keyword random image...",
        )

    async def _reshape(self, interaction: discord.Interaction, size: str, name: str) -> None:
        state = self._effective_state(interaction)
        if not state["prompt"]:
            await self._no_prompt(interaction)
            return
        if state["from_image"]:
            await self._text_only(interaction, f"a {size} version")
            return
        width, height = _dims_for_size(size)
        await _submit(
            interaction,
            {"action": size, "prompt": state["prompt"], "original": state["original"],
             "width": width, "height": height, "seed": state["seed"], "action_name": name},
            f"Generating {size} version...",
        )

    @discord.ui.button(label="↔️ Wide", style=discord.ButtonStyle.primary, custom_id="fluxgen_wide_button", row=1)
    @universal_cooldown_check()
    async def wide_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._reshape(interaction, "wide", "Wide")

    @discord.ui.button(label="↕️ Tall", style=discord.ButtonStyle.primary, custom_id="fluxgen_tall_button", row=1)
    @universal_cooldown_check()
    async def tall_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._reshape(interaction, "tall", "Tall")

    @discord.ui.button(label="🟦 Square", style=discord.ButtonStyle.primary, custom_id="fluxgen_square_button", row=1)
    @universal_cooldown_check()
    async def square_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._reshape(interaction, "square", "Square")


async def setup(bot):
    """discord.py extension entry point — registers /flux on the tree."""
    if settings.flux_enabled:
        bot.tree.add_command(flux)
    else:
        logger.info("⚡ Flux is off (FLUX_ENABLED is not true): /flux is not registered")
    # Persistent-view template: lets buttons on messages posted before the last
    # restart keep working. The blank state is fine — callbacks recover the real
    # prompt/seed/dims from the clicked message via _effective_state(). It is
    # registered with Flux off too, so an old button answers instead of failing.
    bot.add_view(FluxRemixView(prompt="", width=FLUX_DEFAULT_WIDTH, height=FLUX_DEFAULT_HEIGHT, seed=0))
    logger.info("✅ Loaded flux (local Flux image generation) extension")
