# ================================
# FLUX Image Generation and Background Removal API with Gradio Interface
# ================================

# Import Standard Libraries
import logging
import time
import threading
from io import BytesIO

# Import Third-Party Libraries
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from rembg import remove
from colorama import init, Fore, Style

# Import Optimum and Diffusers Modules
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    FluxPipeline
)
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

# Import Transformers Modules
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast
)

# Import Gradio and Requests
import gradio as gr
import requests
from queue import Queue
from threading import Lock
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Initialize Colorama for Colored Logging in Windows
init(autoreset=True)

# Custom Logging Formatter with Colors
class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors based on the log level.
    """
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
    """
    Configures the logging settings with colored output and ensures all loggers use the ColoredFormatter.
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger level to DEBUG

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler with debug level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create and set formatter
    formatter = ColoredFormatter()
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Set specific loggers to DEBUG
    logging.getLogger("uvicorn").setLevel(logging.DEBUG)
    logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
    logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
    logging.getLogger("fastapi").setLevel(logging.DEBUG)
    
    # Set loggers to WARNING or higher to suppress excessive output
    logging.getLogger("gradio").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("wait_for_fastapi").setLevel(logging.WARNING)

# Initialize Logging Early
setup_logging()

# =========================
# Model Configuration Setup
# =========================
class ModelConfig:
    """
    Configuration for model repository and revision details.
    """
    
    # REPO_NAME = "Freepik/flux.1-lite-8B-alpha"
    # REVISION = None
    
    REPO_NAME = "black-forest-labs/FLUX.1-schnell"
    REVISION = "refs/pr/1"
    
    # REPO_NAME = "black-forest-labs/FLUX.1-dev"
    
    # REPO_NAME = "sayakpaul/FLUX.1-merged"
    # REVISION = None
    
    # REPO_NAME = "ostris/OpenFLUX.1"
    # REVISION = None
    

# =======================
# Vision Configuration and Image Analysis
# =======================

# Model Loading and Setup
def load_scheduler(repo, revision):
    """
    Loads the scheduler from the specified repository and revision.
    """
    logging.info("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo, subfolder="scheduler", revision=revision
    )
    logging.info("Scheduler loaded successfully.")
    return scheduler

def load_text_encoder_and_tokenizer(dtype):
    """
    Loads the primary CLIP text encoder and tokenizer.
    """
    logging.info("Loading primary text encoder and tokenizer...")
    text_enc = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=dtype
    )
    logging.info("Primary text encoder and tokenizer loaded successfully.")
    return text_enc, tokenizer

def load_secondary_text_encoder_and_tokenizer(repo, revision, dtype):
    """
    Loads the secondary T5 text encoder and tokenizer.
    """
    logging.info("Loading secondary text encoder and tokenizer...")
    text_enc_2 = T5EncoderModel.from_pretrained(
        repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision
    )
    tokenizer_2 = T5TokenizerFast.from_pretrained(
        repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision
    )
    logging.info("Secondary text encoder and tokenizer loaded successfully.")
    return text_enc_2, tokenizer_2

def load_vae_model(repo, revision, dtype):
    """
    Loads the Variational Autoencoder (VAE) model.
    """
    logging.info("Loading VAE model...")
    vae = AutoencoderKL.from_pretrained(
        repo, subfolder="vae", torch_dtype=dtype, revision=revision
    )
    logging.info("VAE model loaded successfully.")
    return vae

def load_transformer_model(repo, revision, dtype):
    """
    Loads the Flux Transformer 2D model.
    """
    logging.info("Loading transformer model...")
    transformer = FluxTransformer2DModel.from_pretrained(
        repo, subfolder="transformer", torch_dtype=dtype, revision=revision
    )
    logging.info("Transformer model loaded successfully.")
    return transformer

# Quantization and Freezing of Models
def quantize_and_freeze_model(model, model_name):
    """
    Applies 8-bit quantization and freezes the model to reduce memory usage.
    """
    logging.info(f"Starting 8-bit quantization for {model_name}...")
    start_time = time.time()
    
    quantize(model, weights=qfloat8)
    logging.info(f"Quantization completed for {model_name}.")
    
    logging.info(f"Freezing {model_name}...")
    freeze(model)
    end_time = time.time()
    
    logging.info(f"{model_name} quantized and frozen successfully in {end_time - start_time:.2f} seconds.")

# Pipeline Initialization and Setup
def initialize_flux_pipeline(
    scheduler,
    text_encoder,
    tokenizer,
    text_encoder_2,
    tokenizer_2,
    vae,
    transformer
):
    """
    Initializes the FluxPipeline with the provided components.
    """
    logging.info("Initializing FluxPipeline with loaded models...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    pipeline = FluxPipeline(
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=None,  # Placeholder; will assign below
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=None,  # Placeholder; will assign below
    )
    
    # Assign additional components
    pipeline.text_encoder_2 = text_encoder_2
    pipeline.transformer = transformer
    
    if device == "cuda":
        # Enable smart memory management while keeping generation on GPU
        pipeline.enable_sequential_cpu_offload()
        # Enable memory efficient attention
        pipeline.enable_attention_slicing(slice_size="auto")
        # Enable VAE slicing for memory efficiency
        pipeline.enable_vae_slicing()
        logging.info("Enabled memory optimization features for GPU usage")
    else:
        pipeline.to(device)
        logging.info(f"Pipeline set to run on {device}")
    
    logging.info("FluxPipeline initialized successfully.")
    return pipeline

# Image Generation Functionality
def generate_image(prompt, steps, guidance_scale, width, height, seed, pipeline):
    """
    Generates an image based on input parameters using the FluxPipeline.
    Thread-safe implementation using a lock.
    """
    generation_start = time.perf_counter()
    logging.info(f"Starting image generation for prompt: '{prompt}'")
    
    with pipeline_lock:
        try:
            # Handle seed generation and validation
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                logging.info(f"No seed provided. Generated random seed: {seed}")
            else:
                # Ensure seed is within valid range
                try:
                    seed = int(seed) % (2**32)
                    logging.info(f"Using provided seed (normalized): {seed}")
                except (ValueError, TypeError):
                    seed = torch.randint(0, 2**32 - 1, (1,)).item()
                    logging.info(f"Invalid seed provided. Using random seed: {seed}")
            
            # Create generator on the same device as the pipeline's text encoder
            device = pipeline.text_encoder.device
            generator = torch.Generator(device=device.type).manual_seed(seed)
            
            logging.info(
                f"Starting image generation with {steps} steps, "
                f"guidance scale of {guidance_scale}, "
                f"width: {width}, height: {height}."
            )
            
            # Add timing for the actual pipeline generation
            pipeline_start = time.perf_counter()
            output = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=guidance_scale,
            )
            pipeline_end = time.perf_counter()
            logging.info(f"Pipeline generation completed in {pipeline_end - pipeline_start:.2f} seconds")
            
            # Add timing for image conversion
            conversion_start = time.perf_counter()
            image = output.images[0]
            conversion_end = time.perf_counter()
            logging.info(f"Image conversion completed in {conversion_end - conversion_start:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error during image generation: {e}")
            raise HTTPException(status_code=500, detail="Image generation failed.")
    
    total_time = time.perf_counter() - generation_start
    logging.info(f"Total image generation completed in {total_time:.2f} seconds")
    
    return image

# Background Removal Functionality
def remove_background_image(image: Image.Image) -> Image.Image:
    """
    Removes the background from the given PIL.Image using rembg.
    """
    logging.info("Removing background from the image...")
    start_time = time.time()
    
    try:
        # Ensure the image is in RGBA format
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        # Convert PIL.Image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        
        # Remove the background
        output_bytes = remove(image_bytes)
        
        # Convert bytes back to PIL.Image
        output_image = Image.open(BytesIO(output_bytes)).convert("RGBA")
    except Exception as e:
        logging.error(f"Error during background removal: {e}")
        raise HTTPException(status_code=500, detail="Background removal failed.")
    
    end_time = time.time()
    logging.info(f"Background removal completed in {end_time - start_time:.2f} seconds.")
    
    return output_image  # Return PIL.Image directly

# =======================
# Lifespan Event Handler
# =======================
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI. Handles startup and shutdown events.
    """
    global pipeline
    logging.info("Starting up the Flux Image Generation and Background Removal API...")

    try:
        # Define data type for tensors
        tensor_dtype = torch.bfloat16

        # Model repository configuration
        repo = ModelConfig.REPO_NAME
        revision = ModelConfig.REVISION

        # Load scheduler
        scheduler = load_scheduler(repo, revision)

        # Load text encoders and tokenizers
        primary_text_encoder, primary_tokenizer = load_text_encoder_and_tokenizer(tensor_dtype)
        secondary_text_encoder, secondary_tokenizer = load_secondary_text_encoder_and_tokenizer(
            repo, revision, tensor_dtype
        )

        # Load VAE and Transformer models
        vae_model = load_vae_model(repo, revision, tensor_dtype)
        transformer_model = load_transformer_model(repo, revision, tensor_dtype)

        # Apply quantization and freeze models to optimize performance
        quantize_and_freeze_model(transformer_model, "Transformer Model")
        quantize_and_freeze_model(secondary_text_encoder, "Secondary Text Encoder")

        # Initialize the FluxPipeline with loaded models
        pipeline = initialize_flux_pipeline(
            scheduler,
            primary_text_encoder,
            primary_tokenizer,
            secondary_text_encoder,
            secondary_tokenizer,
            vae_model,
            transformer_model
        )

        logging.info("Flux Image Generation and Background Removal API is ready to receive requests.")
        yield  # Indicates that the startup phase is complete

    except Exception as e:
        logging.error(f"Error during startup: {e}")
        raise e  # Prevent the app from starting if there's an error

    finally:
        logging.info("Shutting down the Flux Image Generation and Background Removal API...")

# =======================
# FastAPI Setup with Lifespan Events
# =======================

app = FastAPI(
    title="Flux Image Generation and Background Removal API",
    description="API for generating images using FluxPipeline and removing backgrounds from images.",
    version="1.0.0",
    lifespan=lifespan  # Pass the lifespan function directly
)

# Global variable to hold the pipeline
pipeline = None

# Add these after the pipeline global variable
pipeline_lock = Lock()
generation_queue = Queue()
executor = ThreadPoolExecutor(max_workers=1)  # Limit to 1 worker to prevent GPU conflicts

# =======================
# Endpoint Definitions
# =======================

@app.post("/flux")
async def flux_endpoint(
    prompt: str = Form(...),
    steps: int = Form(4, ge=1),
    guidance_scale: float = Form(3.5, gt=0.0),
    width: int = Form(1024, ge=64, le=1920),
    height: int = Form(1024, ge=64, le=1920),
    seed: int = Form(-1)
):
    endpoint_start = time.perf_counter()
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized.")
    
    try:
        # Execute the generation
        image = generate_image(prompt, steps, guidance_scale, width, height, seed, pipeline)
        
        # Optimize the image conversion process
        buffer = BytesIO()
        # Use lower quality and compression for faster transfer
        image.save(
            buffer, 
            format="JPEG",  # Change to JPEG for faster processing
            quality=85,     # Slightly reduced quality for better performance
            optimize=False  # Disable optimization for speed
        )
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
        logging.error(f"Error in flux endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove_background", summary="Remove the background from an uploaded image.")
async def remove_background_endpoint(
    file: UploadFile = File(..., description="The image file from which to remove the background.")
):
    """
    Endpoint to remove the background from the uploaded image.
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGBA")
    except Exception as e:
        logging.error(f"Error reading the uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    try:
        output_image = remove_background_image(image)
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Unexpected error during background removal: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during background removal.")
    
    # Convert PIL.Image to bytes
    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="image/png")

@app.get("/", summary="Flux Image Generation and Background Removal API Root")
def read_root():
    return {
        "message": "Welcome to the Flux Image Generation and Background Removal API. "
                   "Use /flux to generate images and /remove_background to remove image backgrounds."
    }

@app.get("/health", summary="Health Check Endpoint")
def health_check():
    logging.info("Health check endpoint was called.")
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request, call_next):
    """
    Middleware to log and handle invalid HTTP requests
    """
    try:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logging.debug(
            f"Request: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Duration: {process_time:.3f}s"
        )
        return response
    except Exception as e:
        logging.warning(
            f"Invalid request received: {str(e)} "
            f"Method: {getattr(request, 'method', 'UNKNOWN')} "
            f"Path: {getattr(request, 'url', 'UNKNOWN')}"
        )
        return None

# =======================
# Function to Run FastAPI Server
# =======================
def run_fastapi():
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="debug" 
    )

# =======================
# Gradio Interface
# =======================
def gradio_flux_interface(prompt, steps, guidance_scale, width, height, seed):
    interface_start = time.perf_counter()
    url = "http://localhost:8000/flux"
    
    data = {
        "prompt": prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": seed
    }
    
    try:
        # Use session with keep-alive and optimized settings
        session = requests.Session()
        session.headers.update({
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        response = session.post(
            url, 
            data=data, 
            timeout=120,
            stream=True  # Enable streaming
        )
        response.raise_for_status()
        
        # Stream the image data directly
        image_data = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                image_data.write(chunk)
        
        image_data.seek(0)
        image = Image.open(image_data).convert("RGB")
        
        total_time = time.perf_counter() - interface_start
        logging.info(f"Gradio interface total time: {total_time:.2f} seconds")
        
        return image
            
    except Exception as e:
        logging.error(f"Error in Gradio interface: {e}")
        return f"❌ Error: {str(e)}"

def gradio_remove_background_interface(image):
    """
    Gradio interface function to remove background via FastAPI /remove_background endpoint.
    """
    url = "http://localhost:8000/remove_background" 
    # Convert PIL.Image to bytes
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    files = {"file": ("input.png", buffer, "image/png")}
    try:
        response = requests.post(url, files=files, timeout=120) 
        response.raise_for_status()
        output_image = Image.open(BytesIO(response.content)).convert("RGBA")
        logging.info("Background removed via Gradio interface.")
        return output_image
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred in remove_background interface: {http_err}")
        return f"❌ HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred in remove_background interface: {conn_err}")
        return f"❌ Connection error occurred: {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred in remove_background interface: {timeout_err}")
        return f"❌ Timeout error occurred: {timeout_err}"
    except Exception as e:
        logging.error(f"Unexpected error in Gradio remove_background interface: {e}")
        return f"❌ Unexpected error occurred: {e}"

def wait_for_fastapi(url, endpoints=["/flux", "/remove_background"], timeout=300):
    """
    Waits until the FastAPI server is up and running by periodically sending GET requests to specified endpoints.
    """
    start_time = time.time()
    attempts = 0
    
    # Initial message - using print instead of logging
    print("Waiting for FastAPI server to start...")
    
    while True:
        try:
            for endpoint in endpoints:
                target_url = url + endpoint if endpoint != "/" else url
                response = requests.get(target_url)
                if response.status_code not in [200, 405]:
                    raise Exception(f"Endpoint {endpoint} returned status code {response.status_code}")
            print("\nFastAPI server is up and running!")
            break
        except Exception:
            attempts += 1
            print(f"\rAttempting to load FastAPI server (this may take a few minutes)... Attempt {attempts}...", end="", flush=True)
            if time.time() - start_time > timeout:
                print("\nTimeout while waiting for FastAPI server to start.")
                break
            time.sleep(5)

def launch_gradio():
    """
    Function to launch Gradio interface.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Flux Image Generation and Background Removal")
        
        with gr.Tab("Generate Image"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=2)
            with gr.Row():
                steps = gr.Slider(1, 100, value=4, step=1, label="Inference Steps")
                guidance_scale = gr.Slider(0.1, 20.0, value=3.5, step=0.1, label="Guidance Scale")
            with gr.Row():
                width = gr.Slider(64, 1920, value=512, step=64, label="Width")
                height = gr.Slider(64, 1920, value=512, step=64, label="Height")
            with gr.Row():
                seed = gr.Number(label="Seed", value=-1, precision=0)
            generate_btn = gr.Button("Generate Image")
            output_image = gr.Image(type="pil", label="Generated Image")
            generate_btn.click(
                gradio_flux_interface, 
                inputs=[prompt, steps, guidance_scale, width, height, seed], 
                outputs=output_image
            )
        
        with gr.Tab("Remove Background"):
            with gr.Row():
                input_image = gr.Image(type="pil", label="Input Image")
            remove_btn = gr.Button("Remove Background")
            output_bg_removed = gr.Image(type="pil", label="Image without Background")
            remove_btn.click(
                gradio_remove_background_interface, 
                inputs=input_image, 
                outputs=output_bg_removed
            )
        
        gr.Markdown("## API Endpoints")
        gr.Markdown("""
        - **/flux**: POST endpoint to generate images.
        - **/remove_background**: POST endpoint to remove image backgrounds.
        """)
    
    # Modify the launch parameters to allow external access
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False             # Creates a public URL (optional)
    )

# =======================
# Main Block to Run Both Servers
# =======================
if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Wait for FastAPI to start using health check
    logging.info("Waiting for FastAPI server to start...")
    wait_for_fastapi("http://localhost:8000", timeout=600)  # Increased timeout to 10 minutes if necessary
    
    # Launch Gradio interface
    launch_gradio()
