# ================================
# FLUX Image Generation Pipeline
# ================================

# Import Standard Libraries
import logging
import time

# Import Third-Party Libraries
import torch
import gradio as gr
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
    Configures the logging settings with colored output.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger().handlers[0].setFormatter(ColoredFormatter())

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
    logging.info("FluxPipeline initialized successfully.")
    
    # Enable model offloading to CPU to save memory
    logging.info("Enabling model offloading to CPU...")
    pipeline.enable_model_cpu_offload()
    logging.info("Model offloading enabled.")
    
    return pipeline

# Image Generation Functionality
def generate_image(prompt, steps, guidance_scale, width, height, seed, pipeline):
    """
    Generates an image based on input parameters using the FluxPipeline.
    """
    logging.info(f"Received image generation request with prompt: '{prompt}'")
    start_time = time.time()
    
    # Handle seed generation
    if seed == -1:
        seed = torch.seed()
        logging.info(f"No seed provided. Generated random seed: {seed}")
    else:
        logging.info(f"Using provided seed: {seed}")
    
    generator = torch.Generator().manual_seed(int(seed))
    
    logging.info(
        f"Starting image generation with {steps} steps, "
        f"guidance scale of {guidance_scale}, "
        f"width: {width}, height: {height}."
    )
    
    # Generate the image
    output = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=guidance_scale,
    )
    image = output.images[0]
    
    end_time = time.time()
    logging.info(f"Image generation completed in {end_time - start_time:.2f} seconds.")
    
    return image

# ================================
# Gradio Interface Setup
# ================================
def setup_gradio_interface(pipeline):
    """
    Sets up and launches the Gradio interface for image generation.
    """
    logging.info("Setting up Gradio interface...")
    
    # Define the image generation function with pipeline as a closure
    def gradio_generate(prompt, steps, guidance_scale, width, height, seed):
        return generate_image(prompt, steps, guidance_scale, width, height, seed, pipeline)
    
    # Define input components
    inputs = [
        gr.Textbox(label="Prompt", lines=2, placeholder="Enter your image prompt here..."),
        gr.Number(label="Number of Steps", value=4, precision=0),
        gr.Number(label="Guidance Scale", value=3, precision=1),
        gr.Slider(label="Width", minimum=0, maximum=1920, value=1024, step=2),
        gr.Slider(label="Height", minimum=0, maximum=1920, value=1024, step=2),
        gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
    ]
    
    # Define output component
    output = gr.Image(label="Generated Image")
    
    # Create the Gradio interface
    interface = gr.Interface(
        fn=gradio_generate,
        inputs=inputs,
        outputs=output,
        title="FLUX Image Generator",
        description="Generate images based on your prompts using the FLUX model pipeline."
    )
    
    # Launch the interface
    logging.info("Launching Gradio interface at http://0.0.0.0:7860...")
    interface.launch(server_name="0.0.0.0")

# =======================
# Main Execution Workflow
# =======================
def main():
    """
    Main function to execute the image generation pipeline setup and launch.
    """
    # Setup logging
    setup_logging()
    
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
    
    # Setup and launch the Gradio interface
    setup_gradio_interface(pipeline)

# Entry point
if __name__ == "__main__":
    main()
