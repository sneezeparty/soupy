# ================================
# Stable Diffusion Image Generation API with LoRA Support
# ================================

import logging
import time
import threading
from io import BytesIO
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from rembg import remove
from colorama import init, Fore, Style

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)
import cv2
import numpy as np
# Optional pipelines for img2img/inpaint (import if available)
try:
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionXLImg2ImgPipeline,
    )
except Exception:
    StableDiffusionImg2ImgPipeline = None
    StableDiffusionXLImg2ImgPipeline = None

try:
    from diffusers import (
        StableDiffusion3Img2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionInpaintPipeline,
    )
except Exception:
    StableDiffusion3Img2ImgPipeline = None
    StableDiffusionXLInpaintPipeline = None
    StableDiffusionInpaintPipeline = None
from optimum.quanto import freeze, qfloat8, quantize

# Optional ControlNet imports for SDXL inpaint hybrid
try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
except Exception:
    ControlNetModel = None
    StableDiffusionXLControlNetInpaintPipeline = None

# Optional histogram matching
try:
    from skimage.exposure import match_histograms
except Exception:
    match_histograms = None

import requests
from queue import Queue
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Initialize Colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom logging formatter with colors."""
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    
    COLOR_MAP = {
        logging.DEBUG: Fore.BLUE + LOG_FORMAT + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + LOG_FORMAT + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + LOG_FORMAT + Style.RESET_ALL,
        logging.ERROR: Fore.RED + LOG_FORMAT + Style.RESET_ALL,
        logging.CRITICAL: Fore.MAGENTA + LOG_FORMAT + Style.RESET_ALL,
    }
    
    def format(self, record):
        log_fmt = self.COLOR_MAP.get(record.levelno, self.LOG_FORMAT)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    """Configure logging with colored output."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = ColoredFormatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.DEBUG)
    logging.getLogger("fastapi").setLevel(logging.DEBUG)
    logging.getLogger("gradio").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

setup_logging()

def get_device():
    """Detect and return the appropriate device for PyTorch.
    Prioritizes MPS (Metal Performance Shaders) on Apple Silicon,
    then CUDA on NVIDIA GPUs, otherwise falls back to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Metal Performance Shaders) device for Apple Silicon")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device")
        return device
    else:
        device = torch.device("cpu")
        logging.warning("No GPU acceleration available, using CPU (this will be slow)")
        return device

# Global device variable
DEVICE = get_device()

class StableDiffusionConfig:
    """Configuration for Stable Diffusion model."""
    
    # Recommended uncensored community models - Choose one:
    # REPO_NAME = "Lykon/DreamShaper"  # High quality, minimal censorship, artistic
    # REPO_NAME = "SG161222/Realistic_Vision_V5.1"  # Ultra-realistic characters
    # REPO_NAME = "XpucT/Deliberate"  # Balanced realism/artistic (updated URL)
    # REPO_NAME = "RunDiffusion/Juggernaut-XL-v9"  # SDXL model, ultra-realistic
    # REPO_NAME = "Linaqruf/anything-v3.0"  # High-quality anime style
    # REPO_NAME = "RunDiffusion/ChilloutMix"  # Photo-quality Asian portraits
    # REPO_NAME = "waifu-diffusion/f222"  # Beautiful female portraits
    # REPO_NAME = "runwayml/stable-diffusion-v1-5"  # Classic base model
    # REPO_NAME = "stabilityai/stable-diffusion-2-1"  # SD 2.1 base model
    REPO_NAME = "stabilityai/stable-diffusion-3.5-medium"
    
    REVISION = None
    USE_SDXL = True  # Set to True for SDXL models (SD 3.5 Medium)

def load_scheduler(repo, revision, use_sdxl=False):
    """Load the appropriate scheduler."""
    logging.info("Loading scheduler...")
    try:
        if use_sdxl:
            scheduler = EulerDiscreteScheduler.from_pretrained(
                repo, subfolder="scheduler", revision=revision
            )
        else:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                repo, subfolder="scheduler", revision=revision
            )
        logging.info("Scheduler loaded successfully.")
        return scheduler
    except Exception as e:
        logging.error(f"Failed to load scheduler from {repo}: {e}")
        # Fallback to default scheduler
        scheduler = DPMSolverMultistepScheduler.from_config({
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
        })
        logging.info("Using fallback scheduler.")
        return scheduler

def quantize_and_freeze_model(model, model_name):
    """Apply quantization and freeze model."""
    logging.info(f"Starting quantization for {model_name}...")
    start_time = time.time()
    
    try:
        quantize(model, weights=qfloat8)
        freeze(model)
        end_time = time.time()
        logging.info(f"{model_name} quantized and frozen in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        logging.warning(f"Quantization failed for {model_name}: {e}")
        logging.info(f"Continuing without quantization for {model_name}.")

def initialize_sd_pipeline(scheduler, use_sdxl=False):
    """Initialize the Stable Diffusion pipeline."""
    logging.info("Initializing Stable Diffusion pipeline...")
    
    repo = StableDiffusionConfig.REPO_NAME
    revision = StableDiffusionConfig.REVISION
    
    try:
        if use_sdxl:
            # Use StableDiffusion3Pipeline for SD 3.5 Medium as it has a different architecture
            if "stable-diffusion-3.5-medium" in repo:
                # SD 3.5 Medium requires specific handling - try with trust_remote_code
                try:
                    pipeline = StableDiffusion3Pipeline.from_pretrained(
                        repo,
                        torch_dtype=torch.bfloat16,
                        revision=revision,
                        trust_remote_code=True,
                        force_download=True
                    )
                except Exception as e:
                    logging.warning(f"Failed with force_download=True, trying without: {e}")
                    # Fallback without force download
                    pipeline = StableDiffusion3Pipeline.from_pretrained(
                        repo,
                        torch_dtype=torch.bfloat16,
                        revision=revision,
                        trust_remote_code=True
                    )
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    repo,
                    scheduler=scheduler,
                    torch_dtype=torch.bfloat16,
                    revision=revision
                )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                repo,
                scheduler=scheduler,
                torch_dtype=torch.bfloat16,
                revision=revision
            )
        
        logging.info("Pipeline loaded successfully.")
        
        # Enable memory optimizations
        # Note: On M1 Macs, enable_model_cpu_offload() may not work as expected
        # with MPS. We'll use a different approach for MPS devices.
        if DEVICE.type == "mps":
            # For MPS, move pipeline to device and use attention slicing
            # MPS has unified memory, so CPU offload is less critical
            try:
                pipeline = pipeline.to(DEVICE)
                logging.info("Pipeline moved to MPS device")
            except Exception as e:
                logging.warning(f"Could not move pipeline to MPS device: {e}")
                logging.info("Falling back to CPU offload for MPS")
                pipeline.enable_model_cpu_offload()
        else:
            pipeline.enable_model_cpu_offload()
        
        pipeline.enable_attention_slicing()
        
        return pipeline
        
    except Exception as e:
        logging.error(f"Failed to load pipeline: {e}")
        raise e

def load_lora_weights(pipeline, lora_path, weight=1.0):
    """Load LoRA weights into the pipeline."""
    if not lora_path or not Path(lora_path).exists():
        logging.warning(f"LoRA path not found: {lora_path}")
        return pipeline
    
    try:
        logging.info(f"Loading LoRA weights from {lora_path} with weight {weight}")
        pipeline.load_lora_weights(lora_path)
        pipeline.fuse_lora(lora_scale=weight)
        logging.info("LoRA weights loaded successfully.")
        return pipeline
    except Exception as e:
        logging.error(f"Failed to load LoRA weights: {e}")
        return pipeline

def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed, pipeline):
    """Generate an image using Stable Diffusion."""
    generation_start = time.perf_counter()
    logging.info(f"Starting image generation for prompt: '{prompt}'")
    
    with pipeline_lock:
        try:
            # Ensure dimensions are valid (SD3/SDXL require multiples of 16)
            try:
                orig_w, orig_h = int(width), int(height)
                divisor = 16
                width = max(64, (orig_w // divisor) * divisor)
                height = max(64, (orig_h // divisor) * divisor)
                if (width, height) != (orig_w, orig_h):
                    logging.info(f"Adjusted size to {width}x{height} (was {orig_w}x{orig_h}) to meet model constraints")
            except Exception:
                pass

            # Handle seed
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                logging.info(f"Generated random seed: {seed}")
            else:
                seed = int(seed) % (2**32)
                logging.info(f"Using seed: {seed}")
            
            generator = torch.Generator().manual_seed(seed)
            
            logging.info(f"Generating with {steps} steps, guidance {guidance_scale}, size {width}x{height}")
            
            # Generate image
            pipeline_start = time.perf_counter()
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            pipeline_end = time.perf_counter()
            
            image = output.images[0]
            total_time = time.perf_counter() - generation_start
            
            logging.info(f"Generation completed in {total_time:.2f} seconds")
            return image
            
        except Exception as e:
            logging.error(f"Error during image generation: {e}")
            raise HTTPException(status_code=500, detail="Image generation failed.")

def remove_background_image(image: Image.Image) -> Image.Image:
    """Remove background from image."""
    logging.info("Removing background...")
    start_time = time.time()
    
    try:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        
        output_bytes = remove(image_bytes)
        output_image = Image.open(BytesIO(output_bytes)).convert("RGBA")
        
        end_time = time.time()
        logging.info(f"Background removal completed in {end_time - start_time:.2f} seconds.")
        return output_image
        
    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        raise HTTPException(status_code=500, detail="Background removal failed.")

# FastAPI Setup
app = FastAPI(
    title="Stable Diffusion Image Generation API",
    description="API for generating images using Stable Diffusion with LoRA support.",
    version="1.0.0"
)

# Global variables
pipeline = None
pipeline_lock = Lock()
generation_queue = Queue()
executor = ThreadPoolExecutor(max_workers=1)
pipeline_img2img = None

# Lazy-loaded ControlNet models (SDXL)
controlnet_canny_sdxl = None
controlnet_depth_sdxl = None

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    global pipeline
    logging.info("Starting Stable Diffusion API...")
    
    try:
        # Load scheduler
        scheduler = load_scheduler(
            StableDiffusionConfig.REPO_NAME, 
            StableDiffusionConfig.REVISION,
            StableDiffusionConfig.USE_SDXL
        )
        
        # Initialize pipeline
        pipeline = initialize_sd_pipeline(scheduler, StableDiffusionConfig.USE_SDXL)
        
        # Load LoRA if specified
        lora_path = os.getenv("LORA_PATH", "")
        lora_weight = float(os.getenv("LORA_WEIGHT", "1.0"))
        if lora_path:
            pipeline = load_lora_weights(pipeline, lora_path, lora_weight)
        
        logging.info("Stable Diffusion API is ready!")
        
    except Exception as e:
        logging.error(f"Startup failed: {e}")
        raise e

def _adjust_dimensions(orig_w: int, orig_h: int) -> (int, int):
    """Ensure dimensions are valid (multiples of 16 and >=64)."""
    try:
        divisor = 16
        width = max(64, (int(orig_w) // divisor) * divisor)
        height = max(64, (int(orig_h) // divisor) * divisor)
        return width, height
    except Exception:
        return orig_w, orig_h

def _init_img2img_pipeline():
    """Initialize and cache an img2img pipeline matching the configured model."""
    global pipeline_img2img
    if pipeline_img2img is not None:
        return pipeline_img2img
    repo = StableDiffusionConfig.REPO_NAME
    revision = StableDiffusionConfig.REVISION
    use_sdxl = StableDiffusionConfig.USE_SDXL
    try:
        if StableDiffusion3Img2ImgPipeline is not None and "stable-diffusion-3" in repo:
            pipeline_img2img = StableDiffusion3Img2ImgPipeline.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                revision=revision,
                trust_remote_code=True,
            )
        elif use_sdxl and StableDiffusionXLImg2ImgPipeline is not None:
            pipeline_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                revision=revision,
            )
        elif StableDiffusionImg2ImgPipeline is not None:
            pipeline_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                revision=revision,
            )
        else:
            raise RuntimeError("No compatible Img2Img pipeline available. Update diffusers.")

        # Apply device-specific memory optimizations
        if DEVICE.type == "mps":
            try:
                pipeline_img2img = pipeline_img2img.to(DEVICE)
                logging.info("Img2Img pipeline moved to MPS device")
            except Exception as e:
                logging.warning(f"Could not move img2img pipeline to MPS: {e}")
                pipeline_img2img.enable_model_cpu_offload()
        else:
            pipeline_img2img.enable_model_cpu_offload()
        
        pipeline_img2img.enable_attention_slicing()
        logging.info("Img2Img pipeline initialized.")
        return pipeline_img2img
    except Exception as e:
        logging.error(f"Failed to initialize img2img pipeline: {e}")
        raise HTTPException(status_code=500, detail="Img2Img pipeline initialization failed.")

def _load_controlnet_sdxl(canny: bool, depth: bool):
    """Lazy-load SDXL ControlNet models as requested."""
    global controlnet_canny_sdxl, controlnet_depth_sdxl
    cn_list = []
    if canny:
        if controlnet_canny_sdxl is None:
            if ControlNetModel is None:
                logging.warning("ControlNetModel not available; skipping canny ControlNet.")
            else:
                try:
                    # Official SDXL canny controlnet from diffusers hub
                    controlnet_canny_sdxl = ControlNetModel.from_pretrained(
                        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.bfloat16
                    )
                    # Move to device if MPS
                    if DEVICE.type == "mps":
                        try:
                            controlnet_canny_sdxl = controlnet_canny_sdxl.to(DEVICE)
                        except Exception:
                            pass  # Some models may not support MPS directly
                    logging.info("Loaded SDXL ControlNet (canny)")
                except Exception as e:
                    logging.error(f"Failed to load SDXL ControlNet canny: {e}")
        if controlnet_canny_sdxl is not None:
            cn_list.append(controlnet_canny_sdxl)
    if depth:
        if controlnet_depth_sdxl is None:
            if ControlNetModel is None:
                logging.warning("ControlNetModel not available; skipping depth ControlNet.")
            else:
                try:
                    controlnet_depth_sdxl = ControlNetModel.from_pretrained(
                        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.bfloat16
                    )
                    # Move to device if MPS
                    if DEVICE.type == "mps":
                        try:
                            controlnet_depth_sdxl = controlnet_depth_sdxl.to(DEVICE)
                        except Exception:
                            pass  # Some models may not support MPS directly
                    logging.info("Loaded SDXL ControlNet (depth)")
                except Exception as e:
                    logging.error(f"Failed to load SDXL ControlNet depth: {e}")
        if controlnet_depth_sdxl is not None:
            cn_list.append(controlnet_depth_sdxl)
    return cn_list

def _build_canny_map(pil_image: Image.Image, zero_in_mask: Image.Image = None) -> Image.Image:
    """Generate a canny edge map image suitable for SDXL ControlNet Canny.
    If zero_in_mask is provided (L mode), set edges to 0 where mask==255 so we don't
    over-constrain new outpaint regions and force mirroring.
    """
    try:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        if zero_in_mask is not None:
            m = np.array(zero_in_mask.convert('L'))
            edges[m > 0] = 0
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    except Exception as e:
        logging.warning(f"Canny preprocessing failed: {e}")
        # Return a blank control image to avoid crash
        return Image.new("RGB", pil_image.size, color=(0, 0, 0))

def _apply_histogram_match(source_band: Image.Image, reference_ring: Image.Image) -> Image.Image:
    if match_histograms is None:
        return source_band
    try:
        src = np.array(source_band)
        ref = np.array(reference_ring)
        try:
            # Newer scikit-image (>=0.19) uses channel_axis
            matched = match_histograms(src, ref, channel_axis=-1)
        except TypeError:
            # Older versions used multichannel
            matched = match_histograms(src, ref, multichannel=True)
        return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8))
    except Exception as e:
        logging.warning(f"Histogram matching failed: {e}")
        return source_band

def _match_lightness_lab(source_seam: Image.Image, reference_ring: Image.Image) -> Image.Image:
    """Match average lightness (L channel) of seam to reference using LAB space.
    Applies a conservative gain on L to avoid extreme brightness changes.
    """
    try:
        src = np.array(source_seam.convert('RGB'))
        ref = np.array(reference_ring.convert('RGB'))
        src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)
        src_L = src_lab[:,:,0].astype(np.float32)
        ref_L = ref_lab[:,:,0].astype(np.float32)
        
        # More robust mean calculation - exclude very dark and very bright pixels
        src_valid = src_L[(src_L > 10) & (src_L < 245)]
        ref_valid = ref_L[(ref_L > 10) & (ref_L < 245)]
        
        if len(src_valid) == 0 or len(ref_valid) == 0:
            return source_seam
            
        src_mean = float(src_valid.mean())
        ref_mean = float(ref_valid.mean())
        
        if src_mean <= 1e-3 or ref_mean <= 1e-3:
            return source_seam
            
        gain = ref_mean / src_mean
        
        # Much more conservative gain limits to prevent extreme changes
        gain = float(np.clip(gain, 0.95, 1.05))  # Only allow 5% adjustment max
        
        # Apply gain only to pixels that aren't already very dark or very bright
        mask = (src_L > 10) & (src_L < 245)
        src_L_adjusted = src_L.copy()
        src_L_adjusted[mask] = np.clip(src_L[mask] * gain, 0, 255)
        src_L_adjusted = src_L_adjusted.astype(np.uint8)
        
        src_lab[:,:,0] = src_L_adjusted
        out_rgb = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(out_rgb)
    except Exception as e:
        logging.warning(f"Lightness match failed: {e}")
        return source_seam

def _poisson_seam_blend(src_img: Image.Image, dst_img: Image.Image, seam_mask: Image.Image, mode: int = cv2.NORMAL_CLONE) -> Image.Image:
    """Use OpenCV seamlessClone to blend src onto dst within seam_mask.
    mode: cv2.NORMAL_CLONE or cv2.MIXED_CLONE
    """
    try:
        src = cv2.cvtColor(np.array(src_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        dst = cv2.cvtColor(np.array(dst_img.convert('RGB')), cv2.COLOR_RGB2BGR)
        mask = np.array(seam_mask.convert('L'))
        # Use a higher threshold to avoid black lines - threshold of 50 instead of 10
        # This ensures we only blend where there's significant mask coverage
        _, mask_bin = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        center = (dst.shape[1] // 2, dst.shape[0] // 2)
        blended = cv2.seamlessClone(src, dst, mask_bin, center, mode)
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logging.warning(f"Poisson blending failed, returning dst: {e}")
        return dst_img

def _seed_generator(seed: int):
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logging.info(f"Generated random seed: {seed}")
    else:
        seed = int(seed) % (2**32)
        logging.info(f"Using seed: {seed}")
    return seed, torch.Generator().manual_seed(seed)

@app.post("/sd")
async def sd_endpoint(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(20, ge=1, le=50),
    guidance_scale: float = Form(7.5, gt=0.0, le=20.0),
    width: int = Form(512, ge=64, le=2048),
    height: int = Form(512, ge=64, le=2048),
    seed: int = Form(-1)
):
    """Generate an image using Stable Diffusion."""
    endpoint_start = time.perf_counter()
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    
    try:
        image = generate_image(
            prompt, negative_prompt, steps, guidance_scale, 
            width, height, seed, pipeline
        )
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=False)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": "inline"
            }
        )
        
    except Exception as e:
        logging.error(f"Error in SD endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_background")
async def remove_background_endpoint(file: UploadFile = File(...)):
    """Remove background from uploaded image."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGBA")
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    try:
        output_image = remove_background_image(image)
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        raise HTTPException(status_code=500, detail="Background removal failed.")
    
    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")

@app.get("/")
def read_root():
    return {
        "message": "Stable Diffusion Image Generation API with LoRA support",
        "endpoints": ["/sd", "/remove_background", "/upscale", "/sd_img2img", "/sd_inpaint"]
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "model": StableDiffusionConfig.REPO_NAME}

def upscale_image_with_opencv(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Upscale image using OpenCV's high-quality interpolation methods."""
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Calculate scale factors
        scale_x = target_width / image.width
        scale_y = target_height / image.height
        
        # Choose interpolation method based on scale factor for optimal quality
        if scale_x <= 2.0 and scale_y <= 2.0:
            # For smaller scale factors, use INTER_LANCZOS4 (highest quality)
            interpolation = cv2.INTER_LANCZOS4
            logging.info(f"Using INTER_LANCZOS4 for scale factors: {scale_x:.2f}x{scale_y:.2f}")
        else:
            # For larger scale factors, use INTER_CUBIC (good balance of quality/speed)
            interpolation = cv2.INTER_CUBIC
            logging.info(f"Using INTER_CUBIC for scale factors: {scale_x:.2f}x{scale_y:.2f}")
        
        upscaled_cv = cv2.resize(cv_image, (target_width, target_height), interpolation=interpolation)
        
        # Convert back to PIL
        upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB))
        
        logging.info(f"OpenCV upscaling completed: {image.width}x{image.height} -> {target_width}x{target_height}")
        return upscaled_image
        
    except Exception as e:
        logging.error(f"OpenCV upscaling failed: {e}")
        # Fallback to PIL's LANCZOS resampling
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def upscale_image_with_ai(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Upscale image using AI super-resolution (Real-ESRGAN style)."""
    try:
        # For now, fall back to OpenCV LANCZOS4 as AI models require additional dependencies
        # In the future, you could add Real-ESRGAN or similar AI upscaling models here
        logging.info("AI upscaling requested, using high-quality OpenCV fallback")
        
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Use INTER_LANCZOS4 for highest quality
        upscaled_cv = cv2.resize(cv_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert back to PIL
        upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB))
        
        logging.info(f"AI-style upscaling completed: {image.width}x{image.height} -> {target_width}x{target_height}")
        return upscaled_image
        
    except Exception as e:
        logging.error(f"AI upscaling failed: {e}")
        # Fallback to PIL's LANCZOS resampling
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def preprocess_img2img_image(image: Image.Image) -> Image.Image:
    """Preprocess image for better img2img results."""
    try:
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Apply slight sharpening to improve detail preservation
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        # Apply sharpening filter
        sharpened = cv2.filter2D(img_array, -1, kernel)
        
        # Blend original with sharpened (70% original, 30% sharpened)
        processed_array = cv2.addWeighted(img_array, 0.7, sharpened, 0.3, 0)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(processed_array)
        
        logging.info("Image preprocessing completed for img2img")
        return processed_image
        
    except Exception as e:
        logging.warning(f"Image preprocessing failed, using original: {e}")
        return image

@app.post("/upscale")
async def upscale_endpoint(
    image: UploadFile = File(...),
    scale_factor: float = Form(2.0, ge=1.0, le=4.0),
    target_width: int = Form(None),
    target_height: int = Form(None),
    method: str = Form("opencv", description="Upscaling method: 'opencv' or 'ai'")
):
    """Upscale an image using OpenCV's high-quality interpolation."""
    try:
        contents = await image.read()
        input_image = Image.open(BytesIO(contents)).convert("RGB")
        orig_w, orig_h = input_image.size
        
        # Determine target dimensions
        if target_width and target_height:
            tgt_w, tgt_h = target_width, target_height
        else:
            # Use scale factor
            tgt_w = int(orig_w * scale_factor)
            tgt_h = int(orig_h * scale_factor)
        
        logging.info(f"Upscaling image from {orig_w}x{orig_h} to {tgt_w}x{tgt_h} using method: {method}")
        
        # Choose upscaling method
        if method.lower() == "ai":
            # Use AI super-resolution (if available)
            upscaled_image = upscale_image_with_ai(input_image, tgt_w, tgt_h)
        else:
            # Use OpenCV for high-quality upscaling that preserves content
            upscaled_image = upscale_image_with_opencv(input_image, tgt_w, tgt_h)
        
        buffer = BytesIO()
        upscaled_image.save(buffer, format="JPEG", quality=95, optimize=False)
        buffer.seek(0)
        
        logging.info(f"Upscaling completed successfully")
        return StreamingResponse(buffer, media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in upscale endpoint: {e}")
        raise HTTPException(status_code=500, detail="Upscaling failed.")

@app.post("/sd_img2img")
async def sd_img2img_endpoint(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(20, ge=1, le=50),           # Better default for SD 3.5
    guidance_scale: float = Form(7.5, gt=0.0, le=20.0),  # Better for SD 3.5 Medium
    width: int = Form(None),
    height: int = Form(None),
    seed: int = Form(-1),
    strength: float = Form(0.3),                 # More subtle default
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    try:
        contents = await image.read()
        init_image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Preprocess image for better img2img results
        init_image = preprocess_img2img_image(init_image)
        
        src_w, src_h = init_image.size
        tgt_w = width or src_w
        tgt_h = height or src_h
        tgt_w, tgt_h = _adjust_dimensions(tgt_w, tgt_h)

        _p = _init_img2img_pipeline()
        seed_val, generator = _seed_generator(seed)

        output = _p(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=int(tgt_w),
            height=int(tgt_h),
            generator=generator,
        )
        out_img = output.images[0]

        buffer = BytesIO()
        out_img.save(buffer, format="JPEG", quality=85, optimize=False)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in sd_img2img endpoint: {e}")
        raise HTTPException(status_code=500, detail="Img2Img failed.")

@app.post("/sd_inpaint")
async def sd_inpaint_endpoint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(20, ge=1, le=50),           # Better default for SD 3.5
    guidance_scale: float = Form(7.5, gt=0.0, le=20.0),  # Better for SD 3.5 Medium
    width: int = Form(None),
    height: int = Form(None),
    seed: int = Form(-1),
    strength: float = Form(0.8),                 # Good for inpaint
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    try:
        base_bytes = await image.read()
        mask_bytes = await mask.read()
        base_img = Image.open(BytesIO(base_bytes)).convert("RGB")
        mask_img = Image.open(BytesIO(mask_bytes)).convert("L")

        src_w, src_h = base_img.size
        tgt_w = width or src_w
        tgt_h = height or src_h
        tgt_w, tgt_h = _adjust_dimensions(tgt_w, tgt_h)

        repo = StableDiffusionConfig.REPO_NAME
        revision = StableDiffusionConfig.REVISION
        use_sdxl = StableDiffusionConfig.USE_SDXL

        _p = None
        if StableDiffusion3Pipeline is not None and "stable-diffusion-3" in repo:
            # SD3.5 inpaint pipeline may be unavailable; fallback to img2img with mask unsupported
            _p = _init_img2img_pipeline()
        elif use_sdxl and StableDiffusionXLInpaintPipeline is not None:
            _p = StableDiffusionXLInpaintPipeline.from_pretrained(
                repo, torch_dtype=torch.bfloat16, revision=revision
            )
            # Apply device-specific memory optimizations
            if DEVICE.type == "mps":
                try:
                    _p = _p.to(DEVICE)
                except Exception:
                    _p.enable_model_cpu_offload()
            else:
                _p.enable_model_cpu_offload()
            _p.enable_attention_slicing()
        elif StableDiffusionInpaintPipeline is not None:
            _p = StableDiffusionInpaintPipeline.from_pretrained(
                repo, torch_dtype=torch.bfloat16, revision=revision
            )
            # Apply device-specific memory optimizations
            if DEVICE.type == "mps":
                try:
                    _p = _p.to(DEVICE)
                except Exception:
                    _p.enable_model_cpu_offload()
            else:
                _p.enable_model_cpu_offload()
            _p.enable_attention_slicing()
        else:
            _p = _init_img2img_pipeline()

        seed_val, generator = _seed_generator(seed)
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_img,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=int(tgt_w),
            height=int(tgt_h),
            generator=generator,
        )
        try:
            output = _p(mask_image=mask_img, **kwargs)
        except TypeError:
            output = _p(**kwargs)

        out_img = output.images[0]
        buffer = BytesIO()
        out_img.save(buffer, format="JPEG", quality=85, optimize=False)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in sd_inpaint endpoint: {e}")
        raise HTTPException(status_code=500, detail="Inpaint failed.")

@app.post("/outpaint_hybrid")
async def outpaint_hybrid_endpoint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(24, ge=1, le=50),
    guidance_scale: float = Form(6.5, gt=0.0, le=20.0),
    width: int = Form(None),
    height: int = Form(None),
    seed: int = Form(-1),
    strength: float = Form(0.6),  # seam band denoise
    use_canny: bool = Form(True),
    use_depth: bool = Form(False),
    control_weight: float = Form(0.75),
    harmonize_strength: float = Form(0.0),  # SD3.5 img2img light pass (default off; can be slow/unstable)
    color_match: bool = Form(True),  # Enable conservative lightness matching for seamless blending
):
    """
    Hybrid outpaint:
     1) SDXL Inpaint over the seam band with optional ControlNet Canny/Depth for structure continuity.
     2) Composite original interior back.
     3) Poisson (seamlessClone) blend over the seam region.
     4) Harmonize with SD 3.5 img2img at low strength to match SD3.5 look.
    Expects `image` as the padded canvas and `mask` as L-mode mask (white=new editable areas).
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")

    try:
        # Read inputs
        base_bytes = await image.read()
        mask_bytes = await mask.read()
        padded_img = Image.open(BytesIO(base_bytes)).convert("RGB")
        seam_mask = Image.open(BytesIO(mask_bytes)).convert("L")

        # Determine target size
        src_w, src_h = padded_img.size
        tgt_w = width or src_w
        tgt_h = height or src_h
        tgt_w, tgt_h = _adjust_dimensions(tgt_w, tgt_h)

        # 1) SDXL Inpaint + optional ControlNet (avoid constraining masked area)
        result_img = None
        try:
            controlnets = _load_controlnet_sdxl(use_canny, use_depth)
            if controlnets and StableDiffusionXLControlNetInpaintPipeline is not None:
                base_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    base_model_id,
                    controlnet=controlnets if len(controlnets) > 1 else controlnets[0],
                    torch_dtype=torch.bfloat16,
                )
                # Apply device-specific memory optimizations
                if DEVICE.type == "mps":
                    try:
                        pipe = pipe.to(DEVICE)
                        logging.info("ControlNet inpaint pipeline moved to MPS device")
                    except Exception as e:
                        logging.warning(f"Could not move ControlNet pipeline to MPS: {e}")
                        pipe.enable_model_cpu_offload()
                else:
                    pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()

                control_images = []
                if use_canny:
                    control_images.append(_build_canny_map(padded_img, zero_in_mask=seam_mask))
                if use_depth:
                    # Placeholder: reuse canny-prepared control masking for depth as well to avoid guiding new band
                    control_images.append(_build_canny_map(padded_img, zero_in_mask=seam_mask))

                # If only one controlnet is present, pass a single image; else list
                control_image_arg = control_images[0] if len(control_images) == 1 else control_images

                seed_val, generator = _seed_generator(seed)
                out = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=padded_img,
                    mask_image=seam_mask,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    width=int(tgt_w),
                    height=int(tgt_h),
                    controlnet_conditioning_scale=control_weight if len(controlnets) == 1 else [control_weight]*len(controlnets),
                    generator=generator,
                    strength=float(strength),
                    control_image=control_image_arg,
                )
                result_img = out.images[0]
            else:
                # Fallback to SDXL Inpaint without ControlNet
                if StableDiffusionXLInpaintPipeline is None:
                    raise RuntimeError("SDXL Inpaint pipeline not available.")
                else:
                    xlp = StableDiffusionXLInpaintPipeline.from_pretrained(
                        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        torch_dtype=torch.bfloat16,
                    )
                    # Apply device-specific memory optimizations
                    if DEVICE.type == "mps":
                        try:
                            xlp = xlp.to(DEVICE)
                            logging.info("SDXL inpaint pipeline moved to MPS device")
                        except Exception as e:
                            logging.warning(f"Could not move SDXL inpaint pipeline to MPS: {e}")
                            xlp.enable_model_cpu_offload()
                    else:
                        xlp.enable_model_cpu_offload()
                    xlp.enable_attention_slicing()
                    seed_val, generator = _seed_generator(seed)
                    out = xlp(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=padded_img,
                        mask_image=seam_mask,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance_scale),
                        width=int(tgt_w),
                        height=int(tgt_h),
                        generator=generator,
                        strength=float(strength),
                    )
                    result_img = out.images[0]
        except Exception as e:
            logging.error(f"SDXL inpaint phase failed: {e}")
            raise HTTPException(status_code=500, detail="SDXL inpaint phase failed.")

        # 2) Composite: use generated content only in seam, keep original interior
        try:
            inv_mask = Image.eval(seam_mask, lambda v: 255 - v).convert('L')
            base_rgb = padded_img.convert('RGB')
            result_rgb = result_img.convert('RGB')
            # Want result in seam, base in interior:
            # composite(im1, im2, mask) takes from im1 where mask > 0 else from im2
            comp = Image.composite(result_rgb, base_rgb, seam_mask)
        except Exception as e:
            logging.warning(f"Compositing original interior failed, using result as-is: {e}")
            comp = result_img

        # 3) Use Poisson blending to seamlessly blend the seam region
        # This creates a smooth transition without visible borders
        try:
            blended = _poisson_seam_blend(result_rgb, base_rgb, seam_mask, mode=cv2.NORMAL_CLONE)
            logging.info("Applied Poisson blending for seamless seam transition")
        except Exception as e:
            logging.warning(f"Poisson blending failed, falling back to composite: {e}")
            blended = comp

        # 4) Apply conservative lightness matching to match colors at the border
        # This uses a very conservative 5% max adjustment to prevent black borders
        # while still matching the color/contrast between original and outpainted regions
        if color_match:
            try:
                # Build a thin interior ring mask (just inside the original image border)
                inv_mask_np = 255 - np.array(seam_mask)
                # Erode to get interior, then subtract to get a thin ring
                ring = cv2.erode(inv_mask_np, np.ones((7,7), np.uint8), iterations=1)
                ring = inv_mask_np - ring
                ring_img = Image.fromarray(ring).convert('L')
                
                # Create a mask for the seam region (where we want to adjust colors)
                # Use a slightly dilated version of the seam mask to include the border area
                seam_dilated = cv2.dilate(np.array(seam_mask), np.ones((5,5), np.uint8), iterations=1)
                seam_mask_dilated = Image.fromarray(seam_dilated).convert('L')
                
                # Extract the seam region and the reference ring
                seam_rgb = Image.composite(blended, Image.new('RGB', blended.size, (0,0,0)), seam_mask_dilated)
                ref = Image.composite(base_rgb, Image.new('RGB', base_rgb.size, (0,0,0)), ring_img)
                
                # Apply conservative lightness matching (max 5% adjustment)
                matched_seam = _match_lightness_lab(seam_rgb, ref)
                
                # Composite the matched seam back into the blended image
                blended = Image.composite(matched_seam, blended, seam_mask_dilated)
                logging.info("Applied conservative lightness matching for color consistency")
            except Exception as e:
                logging.warning(f"Lightness matching failed, using blended result as-is: {e}")
        else:
            logging.info("Skipping color matching (color_match=False)")

        # 5) Harmonize with SD3.5 img2img (optional)
        if float(harmonize_strength) and float(harmonize_strength) > 0.0:
            try:
                _p = _init_img2img_pipeline()
                seed_val, generator = _seed_generator(seed)
                out2 = _p(
                    prompt=f"{prompt}",
                    negative_prompt=negative_prompt,
                    image=blended.convert('RGB'),
                    strength=float(harmonize_strength),
                    num_inference_steps=int(max(10, min(28, steps))),
                    guidance_scale=float(max(4.5, min(8.0, guidance_scale))),
                    width=int(tgt_w),
                    height=int(tgt_h),
                    generator=generator,
                )
                final_img = out2.images[0]
            except Exception as e:
                logging.warning(f"SD3.5 harmonization failed, returning blended result: {e}")
                final_img = blended
        else:
            logging.info("Skipping SD3.5 harmonization (harmonize_strength<=0).")
            final_img = blended

        buffer = BytesIO()
        final_img.save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in outpaint_hybrid endpoint: {e}")
        raise HTTPException(status_code=500, detail="Outpaint hybrid failed.")

if __name__ == "__main__":
    import uvicorn
    import os
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="debug"
    )
