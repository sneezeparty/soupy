"""Local Flux image-generation HTTP server (mflux / MLX, Apple Silicon).

A small, self-contained FastAPI backend that runs Flux **locally** on the Mac and
exposes the same form-field contract the Soupy bot already speaks to its remote
Stable Diffusion server. The ``soupy.cogs.flux`` cog POSTs here exactly like
``soupy.cogs.sd`` POSTs to ``SD_SERVER_URL`` — this process owns the heavy model
so the bot process stays light.

Run it (separately from the bot) on the Mac that has the GPU/unified memory:

    source .venv/bin/activate
    pip install mflux fastapi uvicorn python-multipart pillow   # one-time
    python flux_server.py

Endpoints (all PNG endpoints return raw PNG bytes):
    POST /flux           form: prompt, [negative_prompt], steps, guidance_scale,
                               width, height, seed
    POST /flux_img2img   the above + image (file) + strength  (noise-mix img2img)
    POST /flux_edit      image + prompt, steps, guidance_scale, width, height, seed
                         (FLUX.2-Klein only; uses Flux2KleinEdit — concatenated
                         reference-image tokens, strong prompt-driven editing.
                         No `strength` field.)
    GET  /health         {"status": "ok", "model", "loaded", "edit_model", "edit_loaded"}

The img2img ``strength`` form field uses standard Stable-Diffusion semantics
(higher = deviate more from the input), matching the bot/SD contract. mflux's
own ``image_strength`` is the INVERSE (higher = preserve the input), so the
``/flux_img2img`` handler flips it before calling mflux — see the comment there.

The ``/flux_edit`` endpoint is FLUX.2-Klein-only and uses a separate model
instance (``Flux2KleinEdit``) lazily loaded on first request, then held
resident alongside the txt2img/img2img model. Same on-disk weights as
``Flux2Klein`` but a different Python class; expect ~5–8 GB additional RAM
on top of the txt2img model when both are loaded.

Model selection is by env var, so swapping FLUX.1 schnell -> FLUX.2 klein is
config + a weights download, not a code change:
    FLUX_MODEL             mflux model name ("schnell" / "dev", or a FLUX.2 id)  [schnell]
    FLUX_QUANTIZE          quantization bits, 4 or 8                             [4]
    FLUX_LOW_RAM           "1"/"true" to release encoders between runs           [0]
    FLUX_EDIT_MODEL        FLUX.2-Klein variant used by /flux_edit               [flux2-klein-4b]
    FLUX_EDIT_MAX_PIXELS   max area (W*H) the edit pipeline is allowed to run    [1048576]
    FLUX_MAX_DIMENSION     max long side fed to mflux for img2img/edit           [1536]
    FLUX_SERVER_HOST       bind host                                             [127.0.0.1]
    FLUX_SERVER_PORT       bind port                                             [4942]

The server also auto-bumps `guidance` to 1.0 when a caller passes the legacy
`guidance_scale=0.0` to a guided model (klein / dev). Schnell stays at 0.0
since it is CFG-distilled.

Reference images for /flux_edit are resized (preserving aspect) to fit
``FLUX_EDIT_MAX_PIXELS`` and ``FLUX_MAX_DIMENSION`` before mflux is called —
Flux2KleinEdit concatenates VAE-encoded reference tokens with the output
latents, so a large reference (a 12-megapixel phone photo) sends transformer
attention into multi-gigabyte territory and trips Metal's max-buffer limit.
The same cap is applied to ``/flux_img2img`` output dims for parity.

NOTE: mflux's Python API has shifted across releases. This file targets the
``Flux1.from_name(...).generate_image(..., config=Config(...))`` surface and the
FLUX.2 ``Flux2`` class where available, with a clear error if the installed mflux
differs — verify against the version you ``pip install``.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Load .env-stable so the server shares the bot's config when run standalone.
    for _candidate in (".env-stable", ".env"):
        if Path(_candidate).exists():
            load_dotenv(_candidate)
            break
except Exception:
    pass

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("flux_server")

FLUX_MODEL = os.getenv("FLUX_MODEL", "schnell")
FLUX_QUANTIZE = int(os.getenv("FLUX_QUANTIZE", "4"))
FLUX_LOW_RAM = os.getenv("FLUX_LOW_RAM", "0").lower() in ("1", "true", "yes")
FLUX_SERVER_HOST = os.getenv("FLUX_SERVER_HOST", "127.0.0.1")
FLUX_SERVER_PORT = int(os.getenv("FLUX_SERVER_PORT", "4942"))

# Edit-mode model (FLUX.2-klein only). Loaded lazily on first /flux_edit request
# and kept resident; runs alongside the txt2img/img2img model.
FLUX_EDIT_MODEL = os.getenv("FLUX_EDIT_MODEL", "flux2-klein-4b")

# Area cap (W*H) for the Klein-Edit pipeline. 1024*1024 = ~1M pixels is the
# size klein was trained at; larger references trip Metal's buffer limit
# because Flux2KleinEdit concatenates the reference tokens with the output
# latents and attention scales quadratically with the total token count.
FLUX_EDIT_MAX_PIXELS = int(os.getenv("FLUX_EDIT_MAX_PIXELS", str(1024 * 1024)))

# Hard cap on the long side for any img2img/edit request. Bound separately
# from area so a freakishly thin aspect (e.g. 4000x256) is also clamped.
FLUX_MAX_DIMENSION = int(os.getenv("FLUX_MAX_DIMENSION", "1536"))

app = FastAPI(title="Soupy Flux Server")

# Generation is single-threaded: one model, one GPU. The bot's SDQueue already
# serializes requests, but this lock makes the server safe on its own too.
_model = None
_model_lock = threading.Lock()

# Edit pipeline shares the same single-flight discipline but has its own slot
# so the txt2img/img2img model stays loaded across edit calls (and vice versa).
_edit_model = None
_edit_model_lock = threading.Lock()


def _load_model():
    """Construct the mflux model once (lazily) and cache it."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        logger.info(f"Loading Flux model '{FLUX_MODEL}' (quantize={FLUX_QUANTIZE}, low_ram={FLUX_LOW_RAM})...")
        _model = _construct_mflux()
        logger.info("Flux model loaded.")
        return _model


def _model_registry():
    """Map FLUX_MODEL names to (model class, ModelConfig factory) for mflux 0.17.x.

    Both Flux1 (FLUX.1 schnell/dev) and Flux2Klein (FLUX.2 klein) share the same
    constructor (quantize, model_config) and generate_image() signature, so the
    server treats them uniformly. Verified against mflux 0.17.5.
    """
    try:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1
    except Exception as e:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "mflux is not installed or its API changed. Install on the Mac with "
            "`pip install mflux`. Original error: " + repr(e)
        )

    registry = {
        "schnell": (Flux1, ModelConfig.schnell),
        "dev": (Flux1, ModelConfig.dev),
    }
    try:
        from mflux.models.flux2.variants import Flux2Klein

        registry.update(
            {
                "flux2-klein-4b": (Flux2Klein, ModelConfig.flux2_klein_4b),
                "flux2-klein-9b": (Flux2Klein, ModelConfig.flux2_klein_9b),
                "flux2-klein-base-4b": (Flux2Klein, ModelConfig.flux2_klein_base_4b),
                "flux2-klein-base-9b": (Flux2Klein, ModelConfig.flux2_klein_base_9b),
            }
        )
    except Exception as e:
        logger.info(f"FLUX.2 (Flux2Klein) not available in this mflux build: {e}")
    return registry


def _edit_model_registry():
    """Map edit-model names to (Flux2KleinEdit class, ModelConfig factory).

    Edit pipeline is FLUX.2-klein only; mflux ships no FLUX.1 equivalent.
    """
    try:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants import Flux2KleinEdit
    except Exception as e:
        raise RuntimeError(
            "FLUX.2 Klein edit pipeline requires mflux with Flux2KleinEdit. "
            "Install/upgrade mflux on the Mac. Original error: " + repr(e)
        )

    return {
        "flux2-klein-4b": (Flux2KleinEdit, ModelConfig.flux2_klein_4b),
        "flux2-klein-9b": (Flux2KleinEdit, ModelConfig.flux2_klein_9b),
        "flux2-klein-base-4b": (Flux2KleinEdit, ModelConfig.flux2_klein_base_4b),
        "flux2-klein-base-9b": (Flux2KleinEdit, ModelConfig.flux2_klein_base_9b),
    }


# Friendly aliases -> canonical registry keys.
_MODEL_ALIASES = {
    "flux1-schnell": "schnell",
    "flux1-dev": "dev",
    "klein": "flux2-klein-4b",
    "klein-4b": "flux2-klein-4b",
    "klein-9b": "flux2-klein-9b",
    "flux2-klein": "flux2-klein-4b",
}


def _construct_mflux():
    """Build the mflux model object for FLUX_MODEL (FLUX.1 or FLUX.2 klein)."""
    registry = _model_registry()
    key = FLUX_MODEL.strip().lower().replace("_", "-")
    key = _MODEL_ALIASES.get(key, key)
    if key not in registry:
        known = sorted(set(registry) | set(_MODEL_ALIASES))
        raise RuntimeError(f"Unknown FLUX_MODEL '{FLUX_MODEL}'. Known names: {known}")
    cls, model_config_factory = registry[key]
    logger.info(f"Constructing {cls.__name__} for '{key}' (quantize={FLUX_QUANTIZE})")
    return cls(quantize=FLUX_QUANTIZE, model_config=model_config_factory())


def _construct_mflux_edit():
    """Build the Flux2KleinEdit object for FLUX_EDIT_MODEL."""
    registry = _edit_model_registry()
    key = FLUX_EDIT_MODEL.strip().lower().replace("_", "-")
    key = _MODEL_ALIASES.get(key, key)
    if key not in registry:
        known = sorted(set(registry) | set(_MODEL_ALIASES))
        raise RuntimeError(
            f"Unknown FLUX_EDIT_MODEL '{FLUX_EDIT_MODEL}'. Known names: {known}"
        )
    cls, model_config_factory = registry[key]
    logger.info(f"Constructing {cls.__name__} for '{key}' (quantize={FLUX_QUANTIZE})")
    return cls(quantize=FLUX_QUANTIZE, model_config=model_config_factory())


def _load_edit_model():
    """Lazily build the edit model on first /flux_edit request and cache it."""
    global _edit_model
    if _edit_model is not None:
        return _edit_model
    with _edit_model_lock:
        if _edit_model is not None:
            return _edit_model
        logger.info(
            f"Loading Flux edit model '{FLUX_EDIT_MODEL}' (quantize={FLUX_QUANTIZE})..."
        )
        _edit_model = _construct_mflux_edit()
        logger.info("Flux edit model loaded.")
        return _edit_model


def _resolve_guidance(model, requested: float, *, default_for_guided: float = 1.0) -> float:
    """Override `requested` only when it's the legacy 0.0 default on a model that
    needs guidance.

    Schnell is CFG-distilled (`supports_guidance=False`) — 0.0 is correct.
    Distilled klein and dev expect guidance >= 1.0; passing 0.0 yields muddy
    output. To stay back-compatible with users still on the old `FLUX_GUIDANCE=0.0`
    default, substitute `default_for_guided` when the request explicitly hit 0.0
    on a guided model. Anything non-zero is honoured verbatim.
    """
    supports = bool(getattr(getattr(model, "model_config", None), "supports_guidance", False))
    if supports and requested == 0.0:
        return default_for_guided
    return requested


def _snap16_floor(v: int) -> int:
    """Floor-snap to a positive multiple of 16 (the VAE patch size mflux uses).
    Always rounds down so the result never exceeds the input — important when
    enforcing an area cap."""
    return max(16, (int(v) // 16) * 16)


def _fit_within(w: int, h: int, *, max_pixels: int, max_dim: int) -> tuple[int, int]:
    """Shrink (w, h) to satisfy both an area cap and a long-side cap, preserving
    aspect. Output is floor-snapped to multiples of 16 so it never exceeds the
    budget. Dims that already fit are still snapped (mflux requires /16)."""
    if w <= 0 or h <= 0:
        return 1024, 1024
    scale = 1.0
    long_side = max(w, h)
    if long_side > max_dim:
        scale = max_dim / long_side
    pixels = (w * scale) * (h * scale)
    if pixels > max_pixels:
        scale *= (max_pixels / pixels) ** 0.5
    return _snap16_floor(w * scale), _snap16_floor(h * scale)


def _prepare_image_for_mflux(
    raw: bytes,
    *,
    requested_w: int,
    requested_h: int,
    max_pixels: int,
    max_dim: int = FLUX_MAX_DIMENSION,
) -> tuple[str, int, int]:
    """Decode, EXIF-rotate, and resize `raw` to a sane size for mflux. Returns
    (tmp_path, target_w, target_h). The target matches the requested output
    dims after clamping — feeding mflux a same-sized reference avoids any
    internal resize and keeps the Klein-Edit token count predictable.
    """
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    src_w, src_h = img.size
    base_w = requested_w if requested_w > 0 else src_w
    base_h = requested_h if requested_h > 0 else src_h
    target_w, target_h = _fit_within(base_w, base_h, max_pixels=max_pixels, max_dim=max_dim)
    if (img.width, img.height) != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)
    fd, out_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(out_path, format="PNG")
    return out_path, target_w, target_h


def _to_png_bytes(result) -> bytes:
    """Extract PNG bytes from whatever mflux returns (GeneratedImage or PIL)."""
    pil = getattr(result, "image", result)  # GeneratedImage.image, else assume PIL
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _generate(*, prompt: str, width: int, height: int, steps: int, guidance: float, seed: int,
              init_image_path: str | None = None, strength: float | None = None) -> bytes:
    """Blocking generation — call inside a worker thread."""
    model = _load_model()
    with _model_lock:
        # mflux seeds are non-negative ints; map -1/None to a random-ish seed.
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        effective_guidance = _resolve_guidance(model, guidance)
        if effective_guidance != guidance:
            logger.info(
                f"Overriding guidance {guidance} -> {effective_guidance} for guided model "
                f"'{FLUX_MODEL}' (caller passed the schnell-era default)."
            )
        kwargs = dict(
            seed=int(seed),
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=effective_guidance,
        )
        if init_image_path is not None:
            kwargs["image_path"] = init_image_path
            if strength is not None:
                kwargs["image_strength"] = strength
        result = model.generate_image(**kwargs)
        png = _to_png_bytes(result)
        if FLUX_LOW_RAM:
            # Free MLX's buffer cache between runs to keep peak RAM down.
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
        return png


def _generate_edit(*, prompt: str, width: int, height: int, steps: int, guidance: float,
                   seed: int, image_path: str) -> bytes:
    """Blocking Klein-edit generation — call inside a worker thread.

    Flux2KleinEdit conditions the transformer on the reference image via
    concatenated tokens rather than noise blending, so there's no `strength`.
    Distilled klein requires guidance=1.0.
    """
    model = _load_edit_model()
    with _edit_model_lock:
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(4), "big")
        effective_guidance = _resolve_guidance(model, guidance, default_for_guided=1.0)
        if effective_guidance != guidance:
            logger.info(
                f"Overriding edit guidance {guidance} -> {effective_guidance} for "
                f"'{FLUX_EDIT_MODEL}'."
            )
        result = model.generate_image(
            seed=int(seed),
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance=effective_guidance,
            image_paths=[image_path],
        )
        png = _to_png_bytes(result)
        if FLUX_LOW_RAM:
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass
        return png


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "ok",
            "model": FLUX_MODEL,
            "quantize": FLUX_QUANTIZE,
            "loaded": _model is not None,
            "edit_model": FLUX_EDIT_MODEL,
            "edit_loaded": _edit_model is not None,
        }
    )


@app.post("/flux")
async def flux_text2img(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),  # accepted for contract parity; Flux has no CFG negative
    steps: int = Form(4),
    guidance_scale: float = Form(0.0),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
):
    import anyio

    try:
        png = await anyio.to_thread.run_sync(
            lambda: _generate(
                prompt=prompt, width=width, height=height, steps=steps,
                guidance=guidance_scale, seed=seed,
            )
        )
    except Exception as e:
        logger.exception("flux text2img failed")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=png, media_type="image/png")


@app.post("/flux_img2img")
async def flux_img2img(
    image: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(4),
    guidance_scale: float = Form(0.0),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
    strength: float = Form(0.35),
):
    import anyio

    raw = await image.read()

    # Translate the bot's standard Stable-Diffusion strength into mflux's INVERTED
    # image_strength. The bot (and the SD server this endpoint mirrors) treats
    # strength as "how much to deviate from the input" — higher = bigger change.
    # mflux's image_strength is the opposite: init_time_step = num_steps *
    # image_strength, and it only denoises range(init_time_step, num_steps), so a
    # HIGH image_strength starts late, runs few steps, and the prompt barely moves
    # the picture (it reads image_strength as "how much of the input to preserve").
    # Flip it so a high request strength actually transforms the image toward the
    # prompt. Clamp to a small positive floor so it stays img2img (image_strength
    # of exactly 0 makes mflux ignore the init image entirely).
    mflux_strength = max(0.05, min(1.0, 1.0 - strength))
    logger.info(
        f"img2img: request strength={strength} -> mflux image_strength={mflux_strength:.3f} "
        f"(steps={steps})"
    )

    tmp_path = None
    try:
        tmp_path, eff_w, eff_h = _prepare_image_for_mflux(
            raw, requested_w=width, requested_h=height, max_pixels=FLUX_EDIT_MAX_PIXELS,
        )
        if (eff_w, eff_h) != (width, height):
            logger.info(
                f"img2img: clamped output {width}x{height} -> {eff_w}x{eff_h} "
                f"(max_pixels={FLUX_EDIT_MAX_PIXELS}, max_dim={FLUX_MAX_DIMENSION})"
            )
        png = await anyio.to_thread.run_sync(
            lambda: _generate(
                prompt=prompt, width=eff_w, height=eff_h, steps=steps,
                guidance=guidance_scale, seed=seed,
                init_image_path=tmp_path, strength=mflux_strength,
            )
        )
    except Exception as e:
        logger.exception("flux img2img failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return Response(content=png, media_type="image/png")


@app.post("/flux_edit")
async def flux_edit(
    image: UploadFile,
    prompt: str = Form(...),
    steps: int = Form(4),
    guidance_scale: float = Form(1.0),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
):
    """FLUX.2-Klein edit pipeline.

    Uses Flux2KleinEdit, which concatenates VAE-encoded reference image tokens
    with the noise latents inside the transformer — a stronger prompt-driven
    edit than the noise-mix /flux_img2img path. No `strength` knob.
    """
    import anyio

    raw = await image.read()

    tmp_path = None
    try:
        tmp_path, eff_w, eff_h = _prepare_image_for_mflux(
            raw, requested_w=width, requested_h=height, max_pixels=FLUX_EDIT_MAX_PIXELS,
        )
        if (eff_w, eff_h) != (width, height):
            logger.info(
                f"flux_edit: clamped reference + output {width}x{height} -> {eff_w}x{eff_h} "
                f"(max_pixels={FLUX_EDIT_MAX_PIXELS}, max_dim={FLUX_MAX_DIMENSION})"
            )
        png = await anyio.to_thread.run_sync(
            lambda: _generate_edit(
                prompt=prompt, width=eff_w, height=eff_h, steps=steps,
                guidance=guidance_scale, seed=seed, image_path=tmp_path,
            )
        )
    except Exception as e:
        logger.exception("flux edit failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return Response(content=png, media_type="image/png")


if __name__ == "__main__":
    logger.info(
        f"Starting Flux server on {FLUX_SERVER_HOST}:{FLUX_SERVER_PORT} "
        f"(model={FLUX_MODEL}, edit_model={FLUX_EDIT_MODEL})"
    )
    uvicorn.run(app, host=FLUX_SERVER_HOST, port=FLUX_SERVER_PORT)
