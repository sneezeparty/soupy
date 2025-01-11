"""
Soupy-flux
Repository: https://github.com/sneezeparty/soupy
Licensed under the MIT License.

MIT License

Copyright (c) 2024 sneezeparty

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import os
import discord
from discord.ext import commands
from discord.ui import View, Modal, TextInput
import logging
import aiohttp
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import re
import random
from datetime import datetime
import asyncio
import base64
import time
from functools import wraps
from collections import defaultdict
import sys
from colorlog import ColoredFormatter

# Define the log format with color support
LOG_FORMAT = "%(log_color)s[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

# Define color schemes for different log levels
COLOR_SCHEMES = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

# Create a ColoredFormatter
formatter = ColoredFormatter(
    LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors=COLOR_SCHEMES,
    reset=True,
    style="%",
)

# Set up the console handler with the colored formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # Capture all levels
console_handler.setFormatter(formatter)

# (Optional) Set up a file handler for persistent logs
file_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler = logging.FileHandler("soupy_flux.log")
file_handler.setLevel(logging.INFO)  # Adjust as needed
file_handler.setFormatter(file_formatter)

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all types of log messages
    handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

load_dotenv()

# Define cooldown duration in seconds
COOLDOWN_DURATION = 15  # Example: 15 seconds

# Initialize a dictionary to track user cooldowns
user_cooldowns = defaultdict(lambda: 0)

def cooldown_check():
    """Decorator to check and set user cooldowns."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, interaction: discord.Interaction, button: discord.ui.Button):
            current_time = time.time()
            last_used = user_cooldowns[interaction.user.id]
            if current_time < last_used + COOLDOWN_DURATION:
                remaining = int((last_used + COOLDOWN_DURATION) - current_time)
                await interaction.response.send_message(
                    f"You're on cooldown! Please wait {remaining} more second(s) before using this button again.",
                    ephemeral=True
                )
                logger.warning(f"User {interaction.user} attempted to spam buttons. Cooldown in effect.")
                return
            # Set the new cooldown
            user_cooldowns[interaction.user.id] = current_time
            # Proceed with the original button handler
            await func(self, interaction, button)
        return wrapper
    return decorator

# Helper functions
def parse_image_size(prompt):
    """
    Parses the prompt to extract image size modifiers and removes them from the prompt.
    Returns the cleaned prompt and the corresponding image size.
    """
    size = "1024x1024"  # Default size
    if "--wide" in prompt:
        size = "1920x1024"
        prompt = prompt.replace("--wide", "").strip()
    elif "--square" in prompt:
        size = "1024x1024"
        prompt = prompt.replace("--square", "").strip()
    elif "--tall" in prompt:
        size = "1024x1920"
        prompt = prompt.replace("--tall", "").strip()
    elif "--small" in prompt:
        size = "512x512"
        prompt = prompt.replace("--small", "").strip()
    return prompt, size

def generate_unique_filename(prompt, extension=".png"):
    """
    Generates a unique filename for the image based on the prompt and a timestamp.
    """
    base_filename = re.sub(r'\W+', '', prompt[:80]).lower()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{base_filename}_{timestamp}{extension}"
    return unique_filename

async def encode_discord_image(image_url):
    """
    Downloads an image from Discord and encodes it in base64.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status == 200:
                image_data = await response.read()
                image = Image.open(BytesIO(image_data)).convert('RGB')
                if max(image.size) > 1000:
                    image.thumbnail((1000, 1000))
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                logger.error(f"Failed to download image. HTTP status code: {response.status}")
                return None

def format_error_message(e):
    """
    Formats error messages to be user-friendly.
    """
    return f"An error occurred: {str(e)}"

def parse_modifiers(prompt):
    """
    Parses the prompt to extract modifiers and removes them from the prompt.
    Returns the cleaned prompt and a list of modifiers.
    """
    modifiers = []
    possible_modifiers = ['--wide', '--tall', '--small', '--seed']
    value_modifiers = ['--seed']

    # Extract modifiers with values
    for mod in value_modifiers:
        pattern = rf'({mod}) (\d+)'
        match = re.search(pattern, prompt)
        if match:
            modifiers.append((mod, match.group(2)))
            prompt = re.sub(pattern, '', prompt).strip()

    # Extract modifiers without values
    for mod in possible_modifiers:
        if mod in value_modifiers:
            continue
        if mod in prompt:
            modifiers.append((mod, None))
            prompt = prompt.replace(mod, '').strip()

    return prompt, modifiers

# Flux queue for handling image generation tasks
flux_queue = asyncio.Queue()

@bot.event
async def on_ready():
    logger.info(f'🔵 Logged in as {bot.user} (ID: {bot.user.id})')
    bot.loop.create_task(process_flux_queue())

async def process_flux_queue():
    while True:
        task = await flux_queue.get()
        try:
            if task['type'] == 'flux':
                ctx = task['ctx']
                description = task['description']
                await process_flux_image(ctx, description)
            elif task['type'] == 'button':
                interaction = task['interaction']
                action = task['action']
                prompt = task['prompt']
                width = task['width']
                height = task['height']
                seed = task['seed']

                if action == 'remix':
                    await handle_remix(interaction, prompt, width, height, seed, flux_queue.qsize())
                elif action == 'wide':
                    await handle_wide(interaction, prompt, width, height, seed, flux_queue.qsize())
                elif action == 'tall':
                    await handle_tall(interaction, prompt, width, height, seed, flux_queue.qsize())
                elif action == 'edit':
                    await handle_edit(interaction, prompt, width, height, seed, flux_queue.qsize())
                else:
                    logger.error(f"❌ Unknown button action: {action}")
                    await interaction.followup.send(f"Unknown action: {action}", ephemeral=True)
            else:
                logger.error(f"❌ Unknown task type: {task['type']}")
        except Exception as e:
            if task['type'] == 'flux':
                ctx = task['ctx']
                await ctx.send(f"❌ An error occurred: {str(e)}")
            elif task['type'] == 'button':
                interaction = task['interaction']
                await interaction.followup.send(f"❌ An error occurred: {str(e)}", ephemeral=True)
            logger.error(f"❌ Error processing task: {e}")
        finally:
            flux_queue.task_done()

# Function to continuously trigger typing
async def trigger_typing_loop(ctx, interval: float = 9.0):
    """
    Continuously triggers typing until the task is cancelled.
    """
    try:
        while True:
            await ctx.trigger_typing()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass

# !flux command
@bot.command(
    name='flux',
    help='Generates an image using the Flux model.\n\n'
         'Options:\n'
         '--wide: Generates a wide image (1920x1024).\n'
         '--tall: Generates a tall image (1024x1920).\n'
         '--small: Generates a smaller image (512x512).\n\n'
         'Default size is 1024x1024.'
)
async def flux_image(ctx, *, description: str):
    flux_server_url = "http://192.168.1.96:7860"

    # Check if the Flux server is online
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(flux_server_url) as response:
                if response.status != 200:
                    await ctx.send("⚠️ The !flux command is currently offline.")
                    logger.warning("Flux server is offline.")
                    return
        except aiohttp.ClientError:
            await ctx.send("⚠️ The !flux command is currently offline.")
            logger.warning("Flux server is offline due to client error.")
            return

    # Enqueue the task with 'flux' type
    await flux_queue.put({
        'type': 'flux',
        'ctx': ctx,
        'description': description,
    })
    logger.info(f"🟢 Enqueued flux image generation for: '{description}'")

class EditImageModal(Modal, title="🖌️ Edit Image Parameters"):
    def __init__(self, prompt: str, width: int, height: int, seed: int = None):
        super().__init__()
        # Image Description Field
        self.image_description = TextInput(
            label="📝 Image Description",
            style=discord.TextStyle.paragraph,
            default=prompt,
            required=True,
            max_length=2000,
            placeholder="Enter the new image description."
        )
        self.add_item(self.image_description)

        # Width Field
        self.width_input = TextInput(
            label="📏 Width",
            style=discord.TextStyle.short,
            default=str(width),
            required=True,
            min_length=1,
            max_length=5,
            placeholder="Enter the width (e.g., 1024)."
        )
        self.add_item(self.width_input)

        # Height Field
        self.height_input = TextInput(
            label="📐 Height",
            style=discord.TextStyle.short,
            default=str(height),
            required=True,
            min_length=1,
            max_length=5,
            placeholder="Enter the height (e.g., 1024)."
        )
        self.add_item(self.height_input)

        # Seed Field
        self.seed_input = TextInput(
            label="🔢 Seed",
            style=discord.TextStyle.short,
            default=str(seed) if seed is not None else "",
            required=False,
            min_length=1,
            max_length=10,
            placeholder="Enter a seed number or leave blank for random."
        )
        self.add_item(self.seed_input)

    async def on_submit(self, interaction: discord.Interaction):
        """Handles the submission of the modal."""
        try:
            # Retrieve and sanitize input values
            new_prompt = self.image_description.value.strip()
            new_width = int(self.width_input.value.strip())
            new_height = int(self.height_input.value.strip())
            new_seed = int(self.seed_input.value.strip()) if self.seed_input.value.strip().isdigit() else random.randint(0, 2**32 - 1)

            # Validate width and height
            if new_width <= 0 or new_height <= 0:
                await interaction.response.send_message("❌ Width and Height must be positive integers.", ephemeral=True)
                logger.warning(f"User {interaction.user} provided invalid dimensions: {new_width}x{new_height}")
                return

            # Enqueue the edit task
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'edit',
                'prompt': new_prompt,
                'width': new_width,
                'height': new_height,
                'seed': new_seed,
            })

            await interaction.response.send_message("🟢 Your edited image is being generated!", ephemeral=True)
            logger.info(f"🖌️ Edit action enqueued with prompt: '{new_prompt}', width: {new_width}, height: {new_height}, seed: {new_seed}")
        except ValueError:
            await interaction.response.send_message("❌ Width and Height must be valid integers.", ephemeral=True)
            logger.error("Invalid input for width or height.")
        except Exception as e:
            await interaction.response.send_message("❌ An error occurred while processing your edit.", ephemeral=True)
            logger.error(f"Error in EditImageModal submission: {e}")

class FluxRemixView(View):
    """A custom Discord UI view for handling image remixing with multiple buttons."""

    def __init__(self, prompt: str, width: int, height: int, seed: int = None):
        super().__init__(timeout=None)  # Persistent view
        self.prompt = prompt
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.cleaned_prompt = self.parse_prompt(prompt)
        logger.debug(f"FluxRemixView initialized with cleaned_prompt: '{self.cleaned_prompt}', dimensions: {self.width}x{self.height}, seed: {self.seed}")

    def parse_prompt(self, prompt: str) -> str:
        """
        Removes all modifiers from the prompt and returns the cleaned prompt.
        """
        possible_modifiers = ['--wide', '--tall', '--small', '--seed']
        prompt = re.sub(r'(--seed\s+\d+)', '', prompt)
        for mod in possible_modifiers:
            prompt = prompt.replace(mod, '')
        prompt = re.sub(r'\s+', ' ', prompt)
        return prompt.strip()

    @discord.ui.button(label="✏️ Edit", style=discord.ButtonStyle.success, custom_id="flux_edit_button")
    @cooldown_check()
    async def edit_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Edit' button click by opening a modal dialog."""
        try:
            # Remove disabling buttons and editing the message here
            # Instead, directly send the modal
            modal = EditImageModal(prompt=self.prompt, width=self.width, height=self.height, seed=self.seed)
            await interaction.response.send_modal(modal)
            logger.info(f"🖌️ Edit modal sent to user {interaction.user} for prompt: '{self.prompt}'")
        except Exception as e:
            await interaction.followup.send("❌ An error occurred while opening the edit dialog.", ephemeral=True)
            logger.error(f"Error opening edit modal: {e}")


    @discord.ui.button(label="🌱 Remix", style=discord.ButtonStyle.primary, custom_id="flux_remix_button")
    @cooldown_check()
    async def remix_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Remix' button click by enqueuing the action with a new random seed."""
        try:
            await interaction.response.defer(thinking=True)
            queue_size = flux_queue.qsize()
            new_seed = random.randint(0, 2**32 - 1)  # Generate a new random seed
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'remix',
                'prompt': self.cleaned_prompt,
                'width': self.width,
                'height': self.height,
                'seed': new_seed,  # Use the new seed
            })
            logger.info(f"🌱 Remix action enqueued for prompt: '{self.cleaned_prompt}' with new seed: {new_seed}")
        except Exception as e:
            await interaction.followup.send("❌ An error occurred during remix image generation.", ephemeral=True)
            logger.error(f"Error during remix image generation: {e}")

    @discord.ui.button(label="↔️ Wide", style=discord.ButtonStyle.secondary, custom_id="flux_wide_button")
    @cooldown_check()
    async def wide_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Wide' button click by enqueuing the action."""
        try:
            await interaction.response.defer(thinking=True)
            queue_size = flux_queue.qsize()
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'wide',
                'prompt': self.cleaned_prompt,
                'width': 1920,
                'height': 1024,
                'seed': self.seed,
            })
            logger.info(f"↔️ Wide action enqueued for prompt: '{self.cleaned_prompt}'")
        except Exception as e:
            await interaction.followup.send("❌ An error occurred during wide remix.", ephemeral=True)
            logger.error(f"Error during wide remix: {e}")

    @discord.ui.button(label="↕️ Tall", style=discord.ButtonStyle.secondary, custom_id="flux_tall_button")
    @cooldown_check()
    async def tall_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Tall' button click by enqueuing the action."""
        try:
            await interaction.response.defer(thinking=True)
            queue_size = flux_queue.qsize()
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'tall',
                'prompt': self.cleaned_prompt,
                'width': 1024,
                'height': 1920,
                'seed': self.seed,
            })
            logger.info(f"↕️ Tall action enqueued for prompt: '{self.cleaned_prompt}'")
        except Exception as e:
            await interaction.followup.send("❌ An error occurred during tall remix.", ephemeral=True)
            logger.error(f"Error during tall remix: {e}")

# Helper functions for handling button actions
async def handle_remix(interaction, prompt, width, height, seed, queue_size):
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Remix", queue_size=queue_size)

async def handle_wide(interaction, prompt, width, height, seed, queue_size):
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Wide", queue_size=queue_size)

async def handle_tall(interaction, prompt, width, height, seed, queue_size):
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Tall", queue_size=queue_size)

async def handle_edit(interaction, prompt, width, height, seed, queue_size):
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Edit", queue_size=queue_size)

async def generate_flux_image(interaction_or_ctx, prompt, width, height, seed, action_name="Flux", queue_size=0):
    flux_server_url = "http://192.168.1.96:7860"

    # Calculate queue position
    queue_position = 1  # Since we are processing this task now
    queue_total = queue_size + 1  # Including the current task

    # Define image generation parameters
    num_steps = 4
    guidance = 3.5

    payload = {
        "data": [
            prompt,
            num_steps,
            guidance,
            width,
            height,
            seed
        ]
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post(f"{flux_server_url}/api/predict", json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extract key parts of the response
                    image_url = result['data'][0]['url'] if 'data' in result else None
                    duration = result.get('duration', 0)

                    if image_url:
                        image_url = image_url.replace("\\", "/")

                        # Fetch the image from the URL
                        async with session.get(image_url) as image_response:
                            if image_response.status == 200:
                                image_bytes = await image_response.read()

                                # Generate a unique filename
                                random_number = random.randint(100000, 999999)
                                safe_prompt = re.sub(r'\W+', '', prompt[:40]).lower()
                                filename = f"{random_number}_{safe_prompt}.webp"

                                # Prepare the image to be sent to Discord
                                image_file = discord.File(BytesIO(image_bytes), filename=filename)

                                # Create embeds
                                description_embed = discord.Embed(
                                    description=prompt,
                                    color=discord.Color.blue()
                                )

                                details_embed = discord.Embed(
                                    color=discord.Color.green()
                                )
                                details_embed.add_field(name="🔢 Seed", value=f"{seed}", inline=True)
                                details_embed.add_field(name="🔄 Action", value=action_name, inline=True)
                                details_embed.add_field(name="⏱️ Time", value=f"{duration:.1f} seconds", inline=True)
                                details_embed.add_field(name="📋 Queue", value=f"{queue_position} of {queue_total}", inline=True)

                                # Create a new FluxRemixView for the new image
                                new_view = FluxRemixView(prompt=prompt, width=width, height=height, seed=seed)

                                # Determine user mention and name based on object type
                                if isinstance(interaction_or_ctx, discord.Interaction):
                                    user_mention = interaction_or_ctx.user.mention
                                    user_name = interaction_or_ctx.user
                                elif isinstance(interaction_or_ctx, commands.Context):
                                    user_mention = interaction_or_ctx.author.mention
                                    user_name = interaction_or_ctx.author
                                else:
                                    user_mention = "Unknown User"
                                    user_name = "Unknown User"

                                # Send the new image with embeds and the buttons
                                await send_message(
                                    interaction_or_ctx,
                                    content=f"{user_mention} 🖼️ Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view
                                )
                                logger.info(f"🖼️ Image generated and sent to user {user_name} for prompt: '{prompt}'")
                            else:
                                await send_message(
                                    interaction_or_ctx,
                                    content="❌ Failed to fetch the generated image from the provided URL.",
                                    ephemeral=True
                                )
                                logger.error("❌ Failed to fetch the generated image.")
                    else:
                        await send_message(
                            interaction_or_ctx,
                            content="❌ Failed to generate image.",
                            ephemeral=True
                        )
                        logger.error("❌ Failed to generate image URL.")
                else:
                    await send_message(
                        interaction_or_ctx,
                        content=f"❌ Flux server error: {response.status}",
                        ephemeral=True
                    )
                    logger.error(f"❌ Flux server error: {response.status}")
    except Exception as e:
        await send_message(
            interaction_or_ctx,
            content=f"❌ An error occurred during image generation: {e}",
            ephemeral=True
        )
        logger.error(f"❌ Error in generate_flux_image: {e}")


async def send_message(interaction_or_ctx, content=None, embeds=None, file=None, view=None, ephemeral=False):
    if isinstance(interaction_or_ctx, discord.Interaction):
        # For Interaction objects, use followup.send
        await interaction_or_ctx.followup.send(
            content=content, embeds=embeds, file=file, view=view, ephemeral=ephemeral
        )
    else:
        # For Context objects, use send
        await interaction_or_ctx.send(
            content=content, embeds=embeds, file=file, view=view
        )

# Process and generate images using the Flux model, handling various modifiers and iterations
async def process_flux_image(ctx, description: str):
    async with ctx.typing():
        try:
            logger.info(f"🟢 Processing !flux command from {ctx.author.name} in channel {ctx.channel.name}")
            logger.debug(f"📝 Original Prompt: '{description}'")

            # Parse modifiers from the description
            description, modifiers = parse_modifiers(description)
            logger.debug(f"📝 Cleaned Prompt: '{description}'")
            logger.debug(f"🔧 Modifiers Found: {modifiers}")

            # Default image dimensions
            width = 1024
            height = 1024

            # Set default values for modifiers
            seed = random.randint(0, 2**32 - 1)
            # Process modifiers
            for mod, value in modifiers:
                if mod == '--wide':
                    width = 1920
                    height = 1024
                    logger.debug("🔧 Modifier: Wide (1920x1024)")
                elif mod == '--tall':
                    width = 1024
                    height = 1920
                    logger.debug("🔧 Modifier: Tall (1024x1920)")
                elif mod == '--small':
                    width = 512
                    height = 512
                    logger.debug("🔧 Modifier: Small (512x512)")
                elif mod == '--seed':
                    seed = int(value)
                    logger.debug(f"🔧 Seed: {seed} (specific)")

            # Generate the image
            await generate_flux_image(ctx, description, width, height, seed, queue_size=flux_queue.qsize())

        except Exception as e:
            await ctx.send(f"❌ An error occurred: {str(e)}")
            logger.error(f"❌ Error in process_flux_image: {str(e)}")

# Run the bot
if __name__ == "__main__":
    discord_bot_token = os.getenv("DISCORD_TOKEN")
    if discord_bot_token is None:
        logger.critical("🚨 No Discord bot token found. Set the DISCORD_TOKEN environment variable.")
        raise ValueError("No Discord bot token found. Set the DISCORD_TOKEN environment variable.")
    bot.run(discord_bot_token)
