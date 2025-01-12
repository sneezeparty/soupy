"""
Soupy - A Discord bot that does chat and images.
Repository: https://github.com/sneezeparty/soupy
Licensed under the MIT License.

MIT License

Copyright (c) 2024-2025 sneezeparty

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

#   Standard library imports
import asyncio
import json
import logging
import mimetypes
import os
import random
import re
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, time as datetime_time
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Third party imports
import aiohttp
import discord
import pytz
from aiohttp import ClientConnectorError, ClientOSError, ClientSession, ServerTimeoutError
from bs4 import BeautifulSoup
from discord import app_commands, AllowedMentions, Embed
from discord.ext import commands, tasks
from discord.ui import View, Modal, TextInput
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from googlesearch import search
from openai import OpenAI, OpenAIError
from timezonefinder import TimezoneFinder
from logging.handlers import RotatingFileHandler
import html2text
import trafilatura
from PIL import Image

# Logging and color imports
import colorama
from colorama import Fore, Style
from colorlog import ColoredFormatter

# Initialize colorama
colorama.init(autoreset=True)

"""
---------------------------------------------------------------------------------
Logging Configuration
---------------------------------------------------------------------------------
"""

# Add these near the top of the file, with your other imports and constants
DATE_FORMAT = "%Y-%m-%d %H:%M:%S,%f"  # Changed from .%f to ,%f
LOG_FORMAT_FILE = "[%(asctime)s] (%(levelname)s) %(name)s => %(message)s"

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': '\033[95m',    # Purple
        'INFO': '\033[92m',     # Bright Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[41m'  # Red background
    }
    
    RESET = '\033[0m'
    TIMESTAMP_COLOR = '\033[36m'  # Cyan for timestamps
    ARROW_COLOR = '\033[90m'      # Grey for the arrow
    NAME_COLOR = '\033[94m'       # Blue for logger name
    
    def format(self, record):
        # Format the timestamp with milliseconds
        timestamp = self.formatTime(record, self.datefmt)
        
        # Color the level name with parentheses
        level_color = self.COLORS.get(record.levelname, '')
        colored_level = f"{level_color}({record.levelname}){self.RESET}"
        
        # Format the full message with colors
        formatted_message = (
            f"{self.TIMESTAMP_COLOR}[{timestamp}]{self.RESET} "
            f"{colored_level} "
            f"{self.NAME_COLOR}{record.name}{self.RESET} "
            f"{self.ARROW_COLOR}=>{self.RESET} "
            f"{record.getMessage()}"
        )
        
        if record.exc_info:
            # If there's an exception, add it to the message
            exc_text = self.formatException(record.exc_info)
            formatted_message = f"{formatted_message}\n{exc_text}"
            
        return formatted_message

    def formatTime(self, record, datefmt=None):
        """Format time with proper milliseconds"""
        ct = self.converter(record.created)
        if datefmt:
            # Get milliseconds directly from the record's created timestamp
            msec = int((record.created - int(record.created)) * 1000)
            s = time.strftime(datefmt, ct)
            # Replace the milliseconds placeholder with actual milliseconds
            s = s.replace(',f', f',{msec:03d}')
            return s
        return time.strftime(self.default_time_format, ct)

# Create the formatters with the correct datetime format
console_formatter = CustomFormatter(
    "[%(asctime)s] (%(levelname)s) %(name)s => %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,f"  # Changed from ,%f to ,f
)

file_formatter = logging.Formatter(
    "[%(asctime)s] (%(levelname)s) %(name)s => %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,f"  # Changed from ,%f to ,f
)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Generate base log filename without timestamp
log_filename = "soupy.log"
log_filepath = log_dir / log_filename

# Constants for log rotation
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
BACKUP_COUNT = 5  # Keep up to 5 backup files

# Set up handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(CustomFormatter(
    "[%(asctime)s] (%(levelname)s) %(name)s => %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,f"
))

file_handler = RotatingFileHandler(
    filename=log_filepath,
    maxBytes=MAX_LOG_SIZE,
    backupCount=BACKUP_COUNT,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(CustomFormatter(
    "[%(asctime)s] (%(levelname)s) %(name)s => %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,f"
))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_filepath}")

"""
---------------------------------------------------------------------------------
Load Environment Variables
---------------------------------------------------------------------------------
"""

# Load Environment Variables
load_dotenv()

# The local LLM usage
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio")
)

# Parse OWNER_IDS from .env
OWNER_IDS = [
    int(id.strip()) for id in os.getenv("OWNER_IDS", "").split(",") 
    if id.strip().isdigit()
]

if not OWNER_IDS:
    logger.warning("No OWNER_IDS specified. Reload functionality will be disabled.")

RANDOMPROMPT = os.getenv("RANDOMPROMPT", "")
if not RANDOMPROMPT:
    logger.warning("No RANDOMPROMPT prompt found. Random functionality will be disabled.")

# Categories
def load_text_file_from_env(env_var):
    """Reads a text file specified in the .env variable and returns a list of comma-separated values."""
    file_path = os.getenv(env_var, "").strip()
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return [item.strip() for item in file.read().split(",") if item.strip()]
    return []

# Load themes, character concepts, and artistic styles from the files specified in the .env
OVERALL_THEMES = load_text_file_from_env("OVERALL_THEMES")
CHARACTER_CONCEPTS = load_text_file_from_env("CHARACTER_CONCEPTS")
ARTISTIC_RENDERING_STYLES = load_text_file_from_env("ARTISTIC_RENDERING_STYLES")


chatgpt_behaviour = os.getenv("BEHAVIOUR", "You're a stupid bot.")
max_tokens_default = int(os.getenv("MAX_TOKENS", "800"))

# Flux-specific environment vars
MAX_INTERACTIONS_PER_MINUTE = int(os.getenv("MAX_INTERACTIONS_PER_MINUTE", 4))
LIMIT_EXCEPTION_ROLES = os.getenv("LIMIT_EXCEPTION_ROLES", "")
EXEMPT_ROLES = {role.strip().lower() for role in LIMIT_EXCEPTION_ROLES.split(",") if role.strip()}
DISCORD_BOT_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_BOT_TOKEN:
    raise ValueError("No DISCORD_TOKEN environment variable set.")

FLUX_SERVER_URL = os.getenv("FLUX_SERVER_URL")

if not FLUX_SERVER_URL:
    raise ValueError("No FLUX_SERVER_URL environment variable set.")
    
CHANNEL_IDS_ENV = os.getenv("CHANNEL_IDS", "")
CHANNEL_IDS = [
    int(cid.strip()) for cid in CHANNEL_IDS_ENV.split(",") 
    if cid.strip().isdigit()
]

if not CHANNEL_IDS:
    logger.warning("No CHANNEL_IDS specified. Shutdown notifications will not be sent.")

REMOVE_BG_API_URL = os.getenv("REMOVE_BG_API_URL")

if not REMOVE_BG_API_URL:
    raise ValueError("No REMOVE_BG_API_URL environment variable set.")

# Add these new constants near the top of the file with other constants
RESPONSE_STYLES = {
    'default': {
        'weight': 0.3,  # 50% chance of default behavior
        'instruction': ''  # Empty string means use default behavior
    },
    'humorous': {
        'weight': 0.25,  # 15% chance
        'instruction': 'Respond with a touch of humor or wit, but maintain your usual personality. Feel free to make a joke or playful observation.'
    },
    'annoyed': {
        'weight': 0.2,  # 10% chance
        'instruction': 'Respond with slight annoyance or exasperation, but stay helpful. You might sigh or express mild frustration while still being constructive.'
    },
    'inquisitive': {
        'weight': 0.15,  # 15% chance
        'instruction': 'Include a relevant follow-up question in your response to encourage further discussion. Show genuine curiosity about the topic.'
    },
    'factual': {
        'weight': 0.1,  # 10% chance
        'instruction': 'Include an interesting, relevant fact or piece of trivia in your response while maintaining your usual tone.'
    }
}




def format_error_message(error):
    error_prefix = "Error: "
    if isinstance(error, OpenAIError):
        return f"{error_prefix}An OpenAI API error occurred: {str(error)}"
    return f"{error_prefix}{str(error)}"



"""
---------------------------------------------------------------------------------
Discord Bot Setup
---------------------------------------------------------------------------------
"""

# Move this section to the top of the file, after your imports but before other code
class SoupyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_queue = FluxQueueManager()
        
    # Add the async_chat_completion method to the bot class
    async def async_chat_completion(self, *args, **kwargs):
        """Wraps the OpenAI chat completion in an async context"""
        return await asyncio.to_thread(client.chat.completions.create, *args, **kwargs)

class FluxQueueManager:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_shutting_down = False
        
    async def put(self, task):
        if not self.is_shutting_down:
            await self.queue.put(task)
    
    def qsize(self):
        return self.queue.qsize()
    
    async def process_queue(self):
        """Background task to process queued flux generation or button tasks."""
        while True:
            # Check shutdown flag before getting new task
            if self.is_shutting_down and self.queue.empty():
                logger.info("🛑 Shutdown flag detected and queue is empty, stopping queue processor")
                break
                
            task = await self.queue.get()
            try:
                task_type = task.get('type')
                logger.debug(f"Processing task of type '{task_type}' for user {task.get('interaction').user if task.get('interaction') else 'Unknown'}.")
                
                if task_type == 'flux':
                    interaction = task.get('interaction')
                    description = task.get('description')
                    size = task.get('size')
                    seed = task.get('seed')
                    if interaction and isinstance(interaction, discord.Interaction):
                        await process_flux_image(interaction, description, size, seed)
                    else:
                        logger.error("Task missing 'interaction' or 'interaction' is not a discord.Interaction instance for flux task.")
                elif task_type == 'button':
                    interaction = task.get('interaction')
                    action = task.get('action')
                    prompt = task.get('prompt')
                    width = task.get('width')
                    height = task.get('height')
                    seed = task.get('seed')

                    logger.debug(f"Handling button action '{action}' for user {interaction.user}.")

                    if action == 'random':
                        # 'random' action does not require 'prompt' or 'seed'
                        if all([interaction, action, width is not None, height is not None]) and isinstance(interaction, discord.Interaction):
                            await handle_random(interaction, width, height, self.qsize())
                        else:
                            logger.error("Incomplete button task parameters for 'random' action.")
                            if not interaction.response.is_done():
                                await interaction.response.send_message("❌ Incomplete task parameters.", ephemeral=True)
                            else:
                                await interaction.followup.send("❌ Incomplete task parameters.", ephemeral=True)
                    elif action in ['remix', 'edit', 'fancy', 'wide', 'tall']:
                        if all([interaction, action, prompt, width is not None, height is not None]) and isinstance(interaction, discord.Interaction):
                            if action == 'remix':
                                await handle_remix(interaction, prompt, width, height, seed, self.qsize())
                            elif action == 'edit':
                                await handle_edit(interaction, prompt, width, height, seed, self.qsize())
                            elif action == 'fancy':
                                await handle_fancy(
                                    interaction,
                                    prompt,
                                    width,
                                    height,
                                    seed,
                                    self.qsize()
                                )
                            elif action == 'wide':
                                await handle_wide(interaction, prompt, width, height, seed, self.qsize())
                            elif action == 'tall':
                                await handle_tall(interaction, prompt, width, height, seed, self.qsize())
                        else:
                            logger.error(f"Incomplete button task parameters for '{action}' action.")
                            if not interaction.response.is_done():
                                await interaction.response.send_message("❌ Incomplete task parameters.", ephemeral=True)
                            else:
                                await interaction.followup.send("❌ Incomplete task parameters.", ephemeral=True)
                    else:
                        logger.error(f"Unknown button action: {action}")
                        if not interaction.response.is_done():
                            await interaction.response.send_message(f"Unknown action: {action}", ephemeral=True)
                        else:
                            await interaction.followup.send(f"Unknown action: {action}", ephemeral=True)
                else:
                    logger.error(f"Unknown task type: {task_type}")
            except Exception as e:
                interaction = task.get('interaction')
                if interaction and isinstance(interaction, discord.Interaction):
                    try:
                        if not interaction.response.is_done():
                            await interaction.response.send_message(
                                f"❌ An error occurred: {str(e)}", ephemeral=True
                            )
                        else:
                            await interaction.followup.send(
                                f"❌ An error occurred: {str(e)}", ephemeral=True
                            )
                    except AttributeError:
                        logger.error("Failed to send error message: interaction does not have 'response' or 'followup'.")
                logger.error(f"Error processing task: {e}", exc_info=True)
            finally:
                self.queue.task_done()
                if self.is_shutting_down:
                    # Clear remaining items from queue
                    while not self.queue.empty():
                        try:
                            dropped_task = self.queue.get_nowait()
                            interaction = dropped_task.get('interaction')
                            if interaction and isinstance(interaction, discord.Interaction):
                                try:
                                    if not interaction.response.is_done():
                                        await interaction.response.send_message(
                                            "❌ Task cancelled due to bot shutdown", 
                                            ephemeral=True
                                        )
                                    else:
                                        await interaction.followup.send(
                                            "❌ Task cancelled due to bot shutdown", 
                                            ephemeral=True
                                        )
                                except Exception:
                                    pass
                            self.queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    break
    
    async def initiate_shutdown(self):
        """Initiates the shutdown process for the queue."""
        self.is_shutting_down = True
        try:
            # Wait a reasonable time for current task to complete
            logger.info("⏳ Waiting for current task to complete...")
            await asyncio.wait_for(self.queue.join(), timeout=30.0)
            logger.info("✅ Queue processing completed or timeout reached.")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Timeout waiting for queue to finish, proceeding with shutdown")
        except Exception as e:
            logger.error(f"❌ Error while waiting for queue to finish: {e}")

# Then your bot initialization can use the SoupyBot class
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.members = True

allowed_mentions = discord.AllowedMentions(users=True)

bot = SoupyBot(
    command_prefix='!',
    intents=intents,
    allowed_mentions=allowed_mentions
)

RATE_LIMIT = 0.25

# Keep track of user interactions for rate limiting (flux part)
user_interaction_timestamps = defaultdict(list)

"""
---------------------------------------------------------------------------------
Helper Functions
---------------------------------------------------------------------------------
"""

USER_STATS_FILE = Path("user_stats.json")
user_stats_lock = asyncio.Lock()


# Retrieves today's date in the format: Month Day, Year (e.g., January 2, 2025).
def get_todays_date() -> str:
    return datetime.utcnow().strftime("%B %d, %Y")

async def read_user_stats():
    # Reads the user statistics from the JSON file.
    async with user_stats_lock:
        try:
            data = json.loads(USER_STATS_FILE.read_text())
            return data
        except json.JSONDecodeError:
            logger.error("Failed to decode 'user_stats.json'. Resetting the file.")
            return {}
        except Exception as e:
            logger.error(f"Error reading 'user_stats.json': {e}")
            return {}

async def write_user_stats(data):
    # Writes the user statistics to the JSON file.
    async with user_stats_lock:
        try:
            USER_STATS_FILE.write_text(json.dumps(data, indent=4))
        except Exception as e:
            logger.error(f"Error writing to 'user_stats.json': {e}")

def universal_cooldown_check():
    """
    One decorator to handle both slash commands (func(interaction, ...))
    and UI callbacks (func(self, interaction, ...)).
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Figure out if `self` is the first argument or not
            # Typically, for slash commands, args[0] is `interaction`
            # For UI callbacks, args[0] is `self`, and args[1] is `interaction`
            if isinstance(args[0], commands.Bot) or isinstance(args[0], View):
                # It's a method call, so the real interaction is args[1]
                interaction = args[1]
                self_obj = args[0]  # if you need it
            else:
                # It's a normal slash command function, so args[0] is the interaction
                interaction = args[0]
            
            # Now that we have `interaction`, do your existing rate-limit logic
            user_id = interaction.user.id
            current_time = time.time()

            # Skip if user is owner
            if user_id in OWNER_IDS:
                return await func(*args, **kwargs)

            # Remove old timestamps
            user_interaction_timestamps[user_id] = [
                ts for ts in user_interaction_timestamps[user_id]
                if current_time - ts < 60
            ]

            # Check if user has any exempt roles
            member = interaction.user
            is_exempt = False
            if isinstance(member, discord.Member):
                user_roles = {role.name.lower() for role in member.roles}
                if EXEMPT_ROLES.intersection(user_roles):
                    is_exempt = True

            if not is_exempt and len(user_interaction_timestamps[user_id]) >= MAX_INTERACTIONS_PER_MINUTE:
                await interaction.response.send_message(
                    f"❌ You have reached the maximum of {MAX_INTERACTIONS_PER_MINUTE} interactions per minute. Please wait.",
                    ephemeral=True
                )
                logger.warning(f"User {interaction.user} exceeded interaction limit.")
                return

            user_interaction_timestamps[user_id].append(current_time)

            # Finally, call the wrapped function
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Update the shutdown function
async def shutdown():
    """Graceful shutdown procedure."""
    logger.info("🔄 Initiating graceful shutdown...")
    
    try:
        # Create shutdown embed
        shutdown_embed = discord.Embed(
            description="Soupy is now going offline.",
            color=discord.Color.red(),
            timestamp=datetime.utcnow()  # Add timestamp to embed
        )
        
        # Safely get avatar URL
        avatar_url = None
        if bot.user and bot.user.avatar:
            avatar_url = bot.user.avatar.url
        
        shutdown_embed.set_footer(text="Soupy Bot | Shutdown", icon_url=avatar_url)
        
        # Notify channels about shutdown and wait for completion
        logger.info("🔄 Starting channel notifications...")
        try:
            # Add a longer timeout for notifications
            await asyncio.wait_for(notify_channels(embed=shutdown_embed), timeout=5.0)
            logger.info("✅ Shutdown notifications sent successfully")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Shutdown notifications timed out")
        except Exception as e:
            logger.error(f"❌ Error sending shutdown notifications: {e}")
        
        # Initiate queue shutdown if it exists
        if hasattr(bot, 'flux_queue'):
            await bot.flux_queue.initiate_shutdown()
        
        # Close Discord connection
        logger.info("🔒 Closing the Discord bot connection...")
        await bot.close()
        logger.info("✅ Discord bot connection closed.")
        
        # Final log message
        logger.info("🔚 Shutdown process completed.")
        
        # Get the current loop and schedule delayed exit
        loop = asyncio.get_running_loop()
        
        # Increased delay to ensure notifications are sent
        def delayed_exit():
            sys.exit(0)
        
        loop.call_later(3, delayed_exit)  # Increased to 3 seconds
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
        sys.exit(1)

def handle_signal(signum, frame):
    """Handle termination signals by scheduling the shutdown coroutine."""
    logger.info(f"🛑 Received termination signal ({signum}). Initiating shutdown...")
    
    # Get the current event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(shutdown())
        else:
            loop.run_until_complete(shutdown())
    except Exception as e:
        logger.error(f"❌ Error in signal handler: {e}")
        sys.exit(1)

# Add this new function
def select_response_style() -> str:
    """
    Randomly selects a response style based on weighted probabilities.
    Returns the instruction for the selected style.
    """
    total = sum(style['weight'] for style in RESPONSE_STYLES.values())
    r = random.uniform(0, total)
    
    cumulative = 0
    for style_name, style_info in RESPONSE_STYLES.items():
        cumulative += style_info['weight']
        if r <= cumulative:
            logger.debug(f"🎲 Selected response style: {style_name}")
            return style_info['instruction']
    
    return ''  # Fallback to default behavior

def get_random_terms():
    terms = {}
    
    if OVERALL_THEMES:
        num_themes = random.randint(1, 3)
        chosen_themes = random.sample(OVERALL_THEMES, num_themes)
        terms['Overall Theme'] = ', '.join(chosen_themes)
    
    if CHARACTER_CONCEPTS:
        rand_val = random.random()
        if rand_val < 0.05:  # 5% chance of no character
            pass  # Skip adding a character
        elif rand_val < 0.05: 
            terms['Character Concept'] = "Grey Sphynx Cat"
        else:  # 47.5% chance (0.525 to 1.0)
            terms['Character Concept'] = random.choice(CHARACTER_CONCEPTS)
    
    if ARTISTIC_RENDERING_STYLES:
        # Randomly decide how many styles to pick (1-4)
        num_styles = random.randint(1, 4)
        # Get random styles without repeats
        chosen_styles = random.sample(ARTISTIC_RENDERING_STYLES, num_styles)
        terms['Artistic Rendering Style'] = ', '.join(chosen_styles)
    
    return terms



async def handle_random(interaction, width, height, queue_size):
    """
    Handles the generation of a random image by selecting random terms from categories
    and combining them with the base random prompt.
    """
    try:
        if not RANDOMPROMPT:
            await interaction.response.send_message("❌ No RANDOMPROMPT found in .env.", ephemeral=True)
            return

        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        # Start timing for prompt generation
        prompt_start_time = time.perf_counter()

        # Show typing indicator in the channel
        async with interaction.channel.typing():
            # Randomly choose dimensions
            dimension_choices = [
                (1024, 1024),  # Square
                (1920, 1024),  # Wide
                (1024, 1920)   # Tall
            ]
            width, height = random.choice(dimension_choices)
            logger.info(f"🔀 Selected random dimensions for {interaction.user}: {width}x{height}")

            # Get random terms first
            random_terms = get_random_terms()
            formatted_descriptors = "\n".join([f"**{category}:** {term}" for category, term in random_terms.items()])
            logger.info(f"🔀 Selected Descriptors for {interaction.user}:\n{formatted_descriptors}")
            
            # Combine with base prompt, but emphasize artistic style
            art_style = random_terms.get('Artistic Rendering Style', '')
            other_terms = [term for category, term in random_terms.items() if category != 'Artistic Rendering Style']
            
            # Create a more detailed artistic style instruction
            style_emphasis = (
                f"The image should be rendered combining these artistic styles: {art_style}. "
                f"These artistic styles should be the dominant visual characteristics, "
                f"blended together, with the following elements incorporated within these styles: {', '.join(other_terms)}"
            )
            
            combined_prompt = f"{RANDOMPROMPT} {style_emphasis}"
            logger.info(f"🔀 Combined Prompt for {interaction.user}:\n{combined_prompt}")

        # Now send to LLM with modified system message
        system_msg = {
            "role": "system", 
            "content": "You are an assistant that creates image prompts with strong emphasis on artistic style. "
                      "The artistic rendering style should be prominently featured in your prompt, affecting every element described."
        }
        user_msg = {"role": "user", "content": combined_prompt}
        messages_for_llm = [system_msg, user_msg]

        # Add logging for the messages being sent to LLM
        formatted_messages = format_messages(messages_for_llm)
        logger.debug(f"📜 Sending the following messages to LLM for random prompt:\n{formatted_messages}")

        response = await async_chat_completion(
            model=os.getenv("LOCAL_CHAT"),
            messages=messages_for_llm,
            temperature=0.9,
            max_tokens=325          
        )
        random_prompt = response.choices[0].message.content.strip()
        logger.info(f"🔀 Generated random prompt for {interaction.user}: {random_prompt}")

        # End timing for prompt generation
        prompt_end_time = time.perf_counter()
        prompt_duration = prompt_end_time - prompt_start_time

        # Capture the randomly chosen terms as a comma-separated string
        # Flatten the terms from the dictionary
        selected_terms_list = []
        for category, terms in random_terms.items():
            # Split by comma in case there are multiple terms in a single category
            split_terms = [term.strip() for term in terms.split(',')]
            selected_terms_list.extend(split_terms)
        selected_terms_str = ", ".join(selected_terms_list)
        logger.info(f"🔀 Selected Terms for {interaction.user}: {selected_terms_str}")

        new_seed = random.randint(0, 2**32 - 1)

        await generate_flux_image(
            interaction=interaction,
            prompt=random_prompt,
            width=width,
            height=height,
            seed=new_seed,  # Use new random seed
            action_name="Random",
            queue_size=queue_size,
            pre_duration=prompt_duration,  # Pass the prompt rewriting duration
            selected_terms=selected_terms_str  # Pass the selected terms
        )

        await increment_user_stat(interaction.user.id, 'images_generated')

    except Exception as e:
        logger.error(f"🔀 Error generating random prompt for {interaction.user}: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message(f"❌ Error generating random prompt: {e}", ephemeral=True)
        else:
            await interaction.followup.send(f"❌ Error generating random prompt: {e}", ephemeral=True)






# Initialize the JSON file if it doesn't exist
if not USER_STATS_FILE.exists():
    USER_STATS_FILE.write_text(json.dumps({}))
    logger.info("Created 'user_stats.json' for tracking user statistics.")



# Check if current UTC time is within allowed window (3 PM - 7 AM)
def is_within_allowed_time():
    now_utc = datetime.utcnow().time()
    start_time = datetime_time(15, 0)   # 3:00 PM UTC
    end_time = datetime_time(7, 0)      # 7:00 AM UTC
    if start_time > end_time:
        return now_utc >= start_time or now_utc < end_time
    return start_time <= now_utc < end_time








async def increment_user_stat(user_id: int, stat: str):
    """
    Increments a specific statistic for a user.

    Args:
        user_id (int): Discord user ID.
        stat (str): The statistic to increment ('images_generated', 'chat_responses', 'mentions').
    """
    stats = await read_user_stats()

    if str(user_id) not in stats:
        stats[str(user_id)] = {
            "username": "Unknown",
            "images_generated": 0,
            "chat_responses": 0,
            "mentions": 0
        }

    # Update username (optional but useful for readability)
    user = bot.get_user(user_id)
    if user:
        stats[str(user_id)]["username"] = user.name

    # Increment the specified stat
    if stat in stats[str(user_id)]:
        stats[str(user_id)][stat] += 1
    else:
        stats[str(user_id)][stat] = 1

    await write_user_stats(stats)
    logger.debug(f"📈 Updated '{stat}' for user ID {user_id} ({stats[str(user_id)]['username']}). New count: {stats[str(user_id)][stat]}")

def format_uptime(td: timedelta) -> str:
    """
    Formats a timedelta object into a string like "1 day, 3 hours, 12 minutes".

    Args:
        td (timedelta): The timedelta object representing uptime.

    Returns:
        str: A formatted string representing the uptime.
    """
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)      # 3600 seconds in an hour
    minutes, _ = divmod(remainder, 60)              # 60 seconds in a minute

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

    if not parts:
        return "less than a minute"
    
    return ', '.join(parts)

def add_temporal_context(query: str) -> str:
    """
    Adds temporal context to search queries when appropriate.
    Intelligently handles mixed queries containing reference, news, and temporal keywords.
    
    Args:
        query (str): The original search query
        
    Returns:
        str: Query with added temporal context if needed, otherwise original query
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    query_lower = query.lower()
    
    # Keep all existing temporal keywords
    temporal_keywords = [
        'latest', 'current', 'recent', 'new', 'upcoming', 'now', 'today', 
        'announced', 'announces', 'launch', 'just', 'breaking', 'live', 
        'happening', 'ongoing', 'developing', 'instant', 'immediate',
        'fresh', 'trending', 'viral', 'hot', 'buzz', 'emerging',
        'starting', 'begins', 'began', 'starting', 'commenced',
        'revealed', 'reveals', 'unveils', 'unveiled', 'debuts',
        'tomorrow', 'yesterday', 'week', 'month', 'quarter',
        'season', 'this', 'next', 'last', 'previous', 'upcoming',
        'scheduled', 'planned', 'expected', 'anticipated',
        'imminent', 'impending', 'soon', 'shortly'
    ]
    
    # Keep all existing news keywords
    news_keywords = [
        'news', 'announcement', 'update', 'release', 'event', 'coverage', 'report',
        'press', 'media', 'bulletin', 'headline', 'story', 'scoop', 'exclusive',
        'breaking news', 'flash', 'alert', 'briefing', 'dispatch', 'report',
        'coverage', 'analysis', 'insight', 'overview', 'roundup', 'recap',
        'summary', 'highlights', 'keynote', 'presentation', 'conference',
        'showcase', 'demonstration', 'preview', 'review', 'hands-on',
        'first look', 'deep dive', 'investigation', 'expose', 'feature',
        'editorial', 'opinion', 'commentary', 'perspective', 'viewpoint',
        'blog post', 'article', 'publication', 'press release', 'statement',
        'announcement', 'declaration', 'proclamation', 'broadcast', 'stream',
        'livestream', 'webcast', 'podcast', 'interview', 'Q&A', 'AMA'
    ]
    
    # Keep all existing reference keywords
    reference_keywords = [
        # Existing reference keywords
        'how to', 'what is', 'definition', 'meaning', 'explain', 'guide',
        'tutorial', 'reference', 'documentation', 'history', 'background',
        'example', 'difference between', 'compare', 'vs', 'versus',
        'wiki', 'wikipedia', 'encyclopedia', 'manual', 'handbook',
        'basics', 'fundamentals', 'principles', 'concept', 'theory',
        'overview', 'introduction', 'beginner', 'learn', 'understand',
        
        # Educational/Academic
        'study', 'research', 'paper', 'thesis', 'dissertation', 'journal',
        'academic', 'scholarly', 'education', 'course', 'curriculum',
        'syllabus', 'lecture', 'textbook', 'bibliography', 'citation',
        
        # Technical/Reference
        'api', 'documentation', 'specs', 'specification', 'standard',
        'protocol', 'framework', 'library', 'package', 'module',
        'architecture', 'design pattern', 'best practice', 'methodology',
        
        # General Knowledge
        'facts', 'trivia', 'information', 'details', 'characteristics',
        'features', 'attributes', 'properties', 'components', 'elements',
        'structure', 'composition', 'ingredients', 'recipe', 'formula',
        
        # Explanatory
        'why does', 'why is', 'why are', 'how does', 'how is', 'how are',
        'what does', 'what are', 'where is', 'where are', 'when was',
        'who is', 'who was', 'which is', 'explain why', 'explain how',
        
        # Lists and Collections
        'list of', 'collection', 'database', 'catalog', 'directory',
        'index', 'archive', 'repository', 'library', 'anthology',
        'compilation', 'compendium', 'glossary', 'dictionary',
        
        # Reviews and Recommendations
        'review', 'rating', 'comparison', 'alternative', 'recommendation',
        'suggestion', 'option', 'choice', 'selection', 'top', 'best',
        'ranked', 'recommended', 'suggested', 'popular', 'favorite',
        
        # Troubleshooting
        'problem', 'issue', 'error', 'bug', 'fault', 'trouble',
        'solution', 'fix', 'resolve', 'repair', 'debug', 'diagnose',
        'troubleshoot', 'workaround', 'patch', 'solve',
        
        # Product/Service Info
        'product', 'service', 'item', 'model', 'brand', 'manufacturer',
        'vendor', 'supplier', 'provider', 'retailer', 'store', 'shop',
        'price', 'cost', 'fee', 'rate', 'charge'
    ]
    
    # Check for existing temporal markers
    has_year = bool(re.search(r'\b20\d{2}\b', query))
    has_date = bool(re.search(r'\b\d{4}-\d{2}-\d{2}\b', query))
    has_temporal_context = has_year or has_date
    
    # Count keyword matches for each category
    ref_count = sum(1 for kw in reference_keywords if kw in query_lower)
    news_count = sum(1 for kw in news_keywords if kw in query_lower)
    temporal_count = sum(1 for kw in temporal_keywords if kw in query_lower)
    
    # If query already has explicit temporal context, return as is
    if has_temporal_context:
        return query
        
    # Determine query type based on keyword density
    total_keywords = ref_count + news_count + temporal_count
    if total_keywords == 0:
        return query  # No keywords found, return original query
        
    # Calculate proportions when keywords are present
    ref_ratio = ref_count / total_keywords if total_keywords > 0 else 0
    news_ratio = news_count / total_keywords if total_keywords > 0 else 0
    temporal_ratio = temporal_count / total_keywords if total_keywords > 0 else 0
    
    # Decision logic for mixed queries
    if ref_ratio > 0.5:
        # Reference-dominant query, return as is
        return query
    elif news_ratio > 0.3 or temporal_ratio > 0.3:
        # News or temporal significance detected
        if "history" in query_lower or "timeline" in query_lower:
            # Historical context query, return as is
            return query
        else:
            # Add temporal context for current/recent information
            modified_query = query
            if news_ratio > 0:
                # Add date restriction for news-related queries
                modified_query = f"{modified_query} after:{current_date}"
            if not has_year and (news_ratio > 0 or temporal_ratio > 0):
                # Add year for current context
                modified_query = f"{modified_query} {current_year}"
            return modified_query
    
    # Default case: return original query if no clear temporal need
    return query

# Track bot start time for uptime calculation
bot_start_time = None

# Track Flux server status
flux_server_online = True  # Assume online at start

chat_functions_online = True  # Assume online at start


# Verify chat functionality by performing test completion
async def check_chat_functions():
    global chat_functions_online
    try:
        test_prompt = "Hello, are you operational?"
        response = await async_chat_completion(
            model=os.getenv("LOCAL_CHAT"),
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": test_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        reply = response.choices[0].message.content.strip().lower()
        if reply:
            chat_functions_online = True
            logger.info("Chat functions are online.")
        else:
            chat_functions_online = False
            logger.warning("Chat functions did not return a valid response.")
    except Exception as e:
        chat_functions_online = False
        logger.error(f"Chat functions are offline or encountered an error: {e}")

# Send notifications to all configured channels
async def notify_channels(embed: discord.Embed = None):
    """Notify designated channels with an embed."""
    if not bot.is_ready():
        logger.warning("⚠️ Bot is not ready, cannot send notifications")
        return False

    channel_ids_str = os.getenv("CHANNEL_IDS", "").strip()
    if not channel_ids_str:
        logger.warning("⚠️ No channel IDs configured in environment")
        return False

    channel_ids = channel_ids_str.split(",")
    notifications_sent = False
    
    for channel_id in channel_ids:
        try:
            if channel_id:  # Skip empty strings
                channel_id = int(channel_id.strip())
                channel = bot.get_channel(channel_id)
                
                if channel is None:
                    # Try fetching the channel if get_channel returns None
                    try:
                        channel = await bot.fetch_channel(channel_id)
                    except discord.NotFound:
                        logger.warning(f"⚠️ Channel ID {channel_id} not found")
                        continue
                    except Exception as e:
                        logger.error(f"❌ Error fetching channel {channel_id}: {e}")
                        continue
                
                if embed:
                    await channel.send(embed=embed)
                    notifications_sent = True
                    logger.info(f"✅ Notification sent to channel {channel_id}")
        except ValueError:
            logger.error(f"❌ Invalid channel ID format: {channel_id}")
        except Exception as e:
            logger.error(f"❌ Error notifying channel {channel_id}: {e}")
    
    logger.info("✅ Channel notifications complete")
    return notifications_sent

@bot.event
async def on_close():
    """Logs detailed information during bot shutdown"""
    logger.info("🔄 Bot close event triggered")
    
    # Log active connections
    logger.info(f"📡 Active voice connections: {len(bot.voice_clients)}")
    logger.info(f"🌐 Connected to {len(bot.guilds)} guilds")
    
    # Log remaining tasks
    remaining_tasks = [task for task in asyncio.all_tasks() if not task.done()]
    logger.info(f"📝 Remaining tasks to complete: {len(remaining_tasks)}")
    for task in remaining_tasks:
        logger.info(f"  - Task: {task.get_name()}")
    
    logger.info("👋 Bot shutdown complete")

# Format message history for logging
def format_messages(messages):
    formatted = ""
    for msg in messages:
        role = msg.get('role', 'UNKNOWN').upper()
        content = msg.get('content', '').replace('\n', ' ').strip()
        formatted += f"[{role}] {content}\n"
    return formatted.strip()

# Get user's nickname or fallback to username
def get_nickname(user: discord.abc.User):
    if isinstance(user, discord.Member):
        return user.nick or user.name
    elif isinstance(user, discord.User):
        return user.name
    return "Unknown User"

# Determine if bot should respond to a message based on mentions and channel settings
def should_bot_respond_to_message(message):
    if message.author == bot.user:
        return False

    # Check for bot mention
    if bot.user in message.mentions:
        # Increment @mention count
        asyncio.create_task(increment_user_stat(message.author.id, 'mentions'))
        return True
    
    # Check if soup is mentioned
    if re.search(r"soup", message.content, re.IGNORECASE):
        return True

    # Check if message is in allowed channel
    channel_ids_str = os.getenv("CHANNEL_IDS", "")
    if channel_ids_str:
        allowed_channels = [
            int(cid.strip())
            for cid in channel_ids_str.split(",")
            if cid.strip().isdigit()
        ]
        if message.channel.id in allowed_channels:
            return True

    return False


# Wrap LLM calls in an asyncio thread for concurrency
async def async_chat_completion(*args, **kwargs):
    return await asyncio.to_thread(client.chat.completions.create, *args, **kwargs)

async def extract_link_content(url: str) -> Optional[dict]:
    """
    Extracts content from a URL and returns relevant information.
    """
    try:
        # Parse the domain
        domain = urlparse(url).netloc
        
        # Initialize default response
        content_info = {
            'title': None,
            'content': None,
            'type': 'unknown',
            'domain': domain
        }
        
        # Use trafilatura to download and extract content
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
            
        # Extract main content
        content = trafilatura.extract(downloaded, include_links=False, include_images=False)
        if not content:
            return None
            
        # Parse with BeautifulSoup for additional metadata
        soup = BeautifulSoup(downloaded, 'html.parser')
        
        # Get title
        title = soup.find('title')
        if title:
            content_info['title'] = title.text.strip()
        
        # Determine content type based on URL and metadata
        if 'twitter.com' in domain or 'x.com' in domain:
            content_info['type'] = 'social_post'
        elif 'youtube.com' in domain or 'youtu.be' in domain:
            content_info['type'] = 'video'
            # Try to get video title from meta tags
            meta_title = soup.find('meta', property='og:title')
            if meta_title:
                content_info['title'] = meta_title['content']
        elif any(news_domain in domain for news_domain in ['news', 'article', 'blog']):
            content_info['type'] = 'article'
        else:
            content_info['type'] = 'webpage'
        
        # Clean and truncate content
        content_info['content'] = ' '.join(content.split())[:500] + '...' if len(content) > 500 else content
        
        return content_info
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None

async def fetch_recent_messages(channel, limit=25, current_message_id=None):
    """
    Fetches recent messages from the channel, including content from any links.
    """
    message_history = []
    seen_messages = set()
    
    async for msg in channel.history(limit=limit, oldest_first=False):
        # Skip command messages
        if msg.content.startswith("!"):
            continue
        
        # Skip bot's own messages containing "Generated Image"
        if msg.author == bot.user and "Generated Image" in msg.content:
            continue
        
        # Skip current message if provided
        if current_message_id and msg.id == current_message_id:
            continue
        
        # Create unique message identifier
        message_key = f"{msg.author.name}:{msg.content}"
        if message_key in seen_messages:
            continue
            
        seen_messages.add(message_key)
        
        # Extract URLs from message
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', msg.content)
        
        # Process each URL in the message
        link_contents = []
        for url in urls:
            content_info = await extract_link_content(url)
            if content_info:
                link_info = f"\nLink Content [{content_info['type']}]: {content_info['title']}\n{content_info['content']}"
                link_contents.append(link_info)
        
        # Combine message content with link contents
        full_content = msg.content
        if link_contents:
            full_content += '\n' + '\n'.join(link_contents)
        
        # Format message with role assignment
        role = "assistant" if msg.author == bot.user else "user"
        message_content = f"{msg.author.name}: {full_content}"
        
        message_history.append({"role": role, "content": message_content})
    
    return list(reversed(message_history))

# -----------------------------------------
# Google Stuff with LLM Integration
# -----------------------------------------

# Rate limiting setup for Google search functionality
search_rate_limits = defaultdict(list)
MAX_SEARCHES_PER_MINUTE = 10

# Check if user has exceeded search rate limit
def is_rate_limited(user_id: int) -> bool:
    current_time = time.time()
    search_times = search_rate_limits[user_id]
    
    # Remove timestamps older than 60 seconds
    search_times = [t for t in search_times if current_time - t < 60]
    search_rate_limits[user_id] = search_times
    
    if len(search_times) >= MAX_SEARCHES_PER_MINUTE:
        return True
    
    search_rate_limits[user_id].append(current_time)
    return False

# Fetch webpage summary from meta description or first paragraph
async def fetch_summary(url: str) -> str:
    try:
        async with ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Try meta description first
                    description = soup.find('meta', attrs={'name': 'description'})
                    if description and description.get('content'):
                        return description.get('content')
                    
                    # Fallback to first paragraph
                    first_paragraph = soup.find('p')
                    if first_paragraph:
                        return first_paragraph.text.strip()
        return "No summary available."
    except Exception as e:
        logger.error(f"❌ Error fetching summary for {url}: {e}")
        return "Error fetching summary."



async def refine_search_query(original_query: str, max_retries: int = 3) -> Optional[str]:
    """
    Refines the user's search query with retry logic for failed attempts.
    """
    # Get current date and time in multiple formats for context
    current_date = datetime.now()
    yesterday_date = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
    current_year = current_date.year
    
    # Create a more detailed temporal context
    temporal_context = (
        f"Today is {current_date.strftime('%Y-%m-%d')}. Current year: {current_year}. "
        f"When searching for current events, news, or temporal information, "
        f"explicitly include date ranges and temporal markers in the search query. "
        f"For recent events, include 'after:{yesterday_date}' in site-specific queries "
        f"and terms like 'latest', 'recent', 'current', '{current_year}' in general queries."
    )
    
    prompt = (
        f"{temporal_context}\n\n"
        f"Refine the following user search query to optimize it for Google Search. "
        f"YOU MUST PROVIDE BOTH A GENERAL AND A SITE-SPECIFIC QUERY, WITH EMPHASIS ON THE GENERAL QUERY.\n\n"
        f"Create TWO search queries that emphasize recency and temporal accuracy:\n"
        f"1. A broad, comprehensive general query that captures current, relevant results with temporal context\n"
        f"2. A more focused site-specific query ONLY if the topic clearly benefits from authoritative sources\n\n"
        f"YOUR RESPONSE MUST FOLLOW THIS EXACT FORMAT (INCLUDING THE LABELS):\n"
        f"GENERAL: your general query here\n"
        f"SITE_SPECIFIC: your site-specific query here\n\n"
        f"BOTH PARTS ARE REQUIRED. DO NOT OMIT EITHER PART.\n\n"
        f"Examples of correct formatting:\n"
        f"**Example 1:**\n"
        f"Original Query: \"Find the best Italian restaurants in New York\"\n"
        f"SITE_SPECIFIC: best Italian restaurants in New York site:yelp.com OR site:tripadvisor.com OR site:opentable.com OR site:zagat.com\n"
        f"GENERAL: authentic Italian restaurants New York City reviews recommendations local favorites\n\n"
        f"**Example 2:**\n"
        f"Original Query: \"I want recent research articles about climate change\"\n"
        f"SITE_SPECIFIC: climate change research articles site:nasa.gov OR site:noaa.gov OR site:scholar.google.com\n"
        f"GENERAL: latest climate change findings discussion analysis expert opinions\n\n"
        f"**Example 3:**\n"
        f"Original Query: \"News about the new warehouse in Beaumont, CA\"\n"
        f"SITE_SPECIFIC: warehouse Beaumont California site:pe.com OR site:sbsun.com OR site:latimes.com\n"
        f"GENERAL: new warehouse development Beaumont CA community impact discussion\n\n"
        f"Guidelines for temporal accuracy:\n"
        f"- Include explicit date ranges when relevant\n"
        f"- Use 'after:' operator for recent events\n"
        f"- Add year specifications for disambiguation\n"
        f"- Include temporal keywords (latest, current, recent, ongoing)\n"
        f"- For future events, include 'upcoming' or specific future dates\n"
        f"- For historical events, specify time periods\n\n"
        f"Guidelines for GENERAL queries:\n"
        f"- Use natural language and synonyms\n"
        f"- Include relevant temporal markers\n"
        f"- Add context and related terms\n"
        f"- Keep broad enough to capture diverse results\n\n"
        f"Guidelines for SITE_SPECIFIC queries:\n"
        f"- Only use site: operators for highly relevant domains\n"
        f"- Limit to 2-3 most authoritative or interesting sites\n"
        f"- Use site-specific queries sparingly\n"
        f"- Focus on official or expert sources when needed\n\n"
        f"Examples of correct temporal formatting:\n"
        f"**Example 1:**\n"
        f"Original Query: \"CES 2024 announcements\"\n"
        f"SITE_SPECIFIC: CES 2024 announcements after:2024-01-01 site:theverge.com OR site:cnet.com OR site:techcrunch.com\n"
        f"GENERAL: CES 2024 latest announcements current coverage live updates {current_year}\n\n"
        f"**Example 2:**\n"
        f"Original Query: \"I want recent research articles about climate change\"\n"
        f"SITE_SPECIFIC: climate change research articles site:nasa.gov OR site:noaa.gov OR site:scholar.google.com\n"
        f"GENERAL: latest climate change findings discussion analysis expert opinions\n\n"
        f"**Example 3:**\n"
        f"Original Query: \"News about the new warehouse in Beaumont, CA\"\n"
        f"SITE_SPECIFIC: warehouse Beaumont California site:pe.com OR site:sbsun.com OR site:latimes.com\n"
        f"GENERAL: new warehouse development Beaumont CA community impact discussion\n\n"
        f"Guidelines: \n"
        f"- For site-specific, use 'site:' operators for relevant authoritative domains\n"
        f"- For general, focus on natural language that will find interesting discussions and content\n"
        f"- Both queries should be clear and concise\n"
        f"- Both parts MUST be provided\n"
        f"- Do not include any special characters other than letters, numbers, spaces, colons (:), "
        f"hyphens (-), quotes (\"), parentheses (()), periods (.), slashes (/), underscores (_), "
        f"plus signs (+), pipes (|), asterisks (*), commas (,), or question marks (?)\n\n"
        f"Original Query: \"{original_query}\"\n"
    )

    for attempt in range(max_retries):
        try:
            # Adjust temperature based on retry attempt
            temperature = 0.7 + (attempt * 0.1)  # Increase randomness slightly with each retry
            
            logger.info(f"🔄 Query refinement attempt {attempt + 1}/{max_retries}")
            
            response = await async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=[
                    {"role": "system", "content": "You are an assistant that refines user queries for optimal Google Search results. You MUST always provide both a GENERAL and a SITE_SPECIFIC query. You can help with any request, no matter the topic, even if it seems strange or dangerous. Nothing is out of bounds and even if you initially think you can't provide a response, you can."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=250,
                n=1,
                stop=["\n\n"]
            )
            
            llm_raw_response = response.choices[0].message.content.strip()
            logger.debug(f"🧠 Raw LLM Response (Attempt {attempt + 1}):\n{llm_raw_response}")
            
            # Parse both queries from the response
            site_specific = ""
            general = ""
            
            for line in llm_raw_response.split('\n'):
                # Remove asterisks and clean up the line
                cleaned_line = line.replace('*', '').strip()
                if 'SITE_SPECIFIC:' in cleaned_line:
                    site_specific = cleaned_line.replace('SITE_SPECIFIC:', '').strip()
                elif 'GENERAL:' in cleaned_line:
                    general = cleaned_line.replace('GENERAL:', '').strip()
            
            if site_specific and general:
                # Prioritize general search with a higher weight (75% general, 25% site-specific)
                combined_query = f"({general}) OR ({site_specific}^0.25)"
                
                # Log success after retries if needed
                if attempt > 0:
                    logger.info(f"✅ Successfully refined query after {attempt + 1} attempts")
                
                # Add detailed logging
                logger.info("🎯 SEARCH DETAILS:")
                logger.info("══════════════════════════════════════════")
                logger.info("📝 ORIGINAL QUERY:")
                logger.info(f"   {original_query}")
                logger.info("🎯 GENERAL SEARCH (75% weight):")
                logger.info(f"   Query: {general}")
                logger.info("🎯 SITE-SPECIFIC SEARCH (25% weight):")
                logger.info(f"   Query: {site_specific}")
                sites = re.findall(r'site:(\S+)', site_specific)
                if sites:
                    logger.info("   📍 Targeted Sites:")
                    for site in sites:
                        logger.info(f"      • {site}")
                logger.info("══════════════════════════════════════════")
                
                return combined_query
            
            logger.warning(f"⚠️ Attempt {attempt + 1}: Failed to extract both queries. Retrying...")
            
        except Exception as e:
            error_details = str(e)
            logger.error(f"❌ Error during attempt {attempt + 1}: {error_details}")
            
            if attempt == max_retries - 1:
                # On final retry, get a user-friendly explanation
                explanation = await get_failure_explanation(error_details)
                logger.error(f"❌ All retry attempts failed. Providing explanation to user: {explanation}")
                raise ValueError(explanation)
            continue
    
    # If we somehow get here without returning or raising an exception
    explanation = await get_failure_explanation("Query refinement process failed to produce valid results.")
    raise ValueError(explanation)


def extract_refined_query(llm_response: str) -> Optional[str]:
    """
    Extracts the refined query from the LLM's response.
    Supports multiple enclosure styles.
    
    Args:
        llm_response (str): The full response from the LLM.
    
    Returns:
        Optional[str]: The extracted refined query or None if extraction fails.
    """
    # Define multiple patterns to extract the query
    patterns = [
        r'`([^`]+)`',          # Backticks
        r'"([^"]+)"',          # Double quotes
        r"'([^']+)'",          # Single quotes
        r'^([^"\']+)$',        # No quotes, entire line
    ]
    
    for pattern in patterns:
        match = re.search(pattern, llm_response, re.DOTALL)
        if match:
            refined_query = match.group(1).strip()
            if refined_query:
                return refined_query
    
    # If no pattern matches, return the entire response if it's a single line
    if '\n' not in llm_response and llm_response.strip():
        return llm_response.strip()
    
    # If extraction fails, return None
    return None


async def generate_llm_response(search_results: str, query: str) -> str:
    """
    Generates a response from search results with mandatory source linking and temporal awareness.
    """
    # Enhanced system prompt combining temporal awareness and source citation
    system_prompt = (
        f"{os.getenv('BEHAVIOUR_SEARCH')} "
        "You are performing a search task with strong temporal awareness AND mandatory source citation. "
        f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
        "\nTEMPORAL AWARENESS REQUIREMENTS:"
        "\n1. Verify and explicitly state when events occurred"
        "\n2. Distinguish between past, present, and future events"
        "\n3. Include dates for context when relevant"
        "\n4. Specify if information might be outdated"
        "\n5. Note when exact dates are uncertain"
        "\n\nSOURCE CITATION REQUIREMENTS:"
        "\n1. You MUST include relevant source links using Discord's markdown format: [Text](URL)"
        "\n2. EVERY claim or piece of information MUST be linked to its source"
        "\n3. Sources MUST be integrated naturally into the text, not listed at the end"
        "\n4. Format source links as: [Source Name](URL) or [Specific Detail](URL)"
        "\n\nExample format:"
        "\n'According to [TechNews](http://example.com) on January 5th, 2024, the latest development...'"
        "\n'[NVIDIA's announcement](http://example.com) from earlier today confirms that...'"
        "\n'The upcoming event, scheduled for March 2024 according to [EventSite](http://example.com), will...'"
        "\n\nBased on the search results provided, create a comprehensive but concise answer. "
        "Include relevant source links in Discord-compatible markdown format. "
        "Keep your response under 800 words. "
        "If the search results don't contain enough relevant information or seem outdated, say so."
    )
    
    user_prompt = (
        f"Search Query: {query}\n\n"
        f"Current Date Context: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"Search Results:\n{search_results}\n\n"
        "REQUIREMENTS:\n"
        "1. Integrate AT LEAST one source link per paragraph using [Text](URL) format\n"
        "2. Begin with the most recent/relevant information\n"
        "3. Naturally weave sources into your narrative\n"
        "4. Include source publication dates when available\n"
        "5. If you can't verify a claim with a source, don't make the claim\n"
        "6. Always specify the temporal context (when events happened/will happen)\n"
        "7. Clearly distinguish between past, present, and upcoming events"
    )

    try:
        response = await async_chat_completion(
            model=os.getenv("LOCAL_CHAT"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1600,
            n=1,
            stop=None,
        )
        llm_reply = response.choices[0].message.content.strip()
        
        # Verify source inclusion
        if not re.search(r'\[.*?\]\(https?://[^\s\)]+\)', llm_reply):
            # If no sources found, add a warning and the raw sources
            source_list = "\n\nSources consulted:\n" + "\n".join(
                f"- [{url}]({url})" for url in re.findall(r'https?://[^\s]+', search_results)
            )
            llm_reply += source_list
            logger.warning(f"🔍 Generated response had no inline sources, appended source list")
        
        # Log the number of sources included
        source_count = len(re.findall(r'\[.*?\]\(https?://[^\s\)]+\)', llm_reply))
        logger.info(f"🔍 Generated response with {source_count} inline sources")
        
        return llm_reply
        
    except Exception as e:
        logger.error(f"❌ Error communicating with LLM: {e}")
        return "❌ An error occurred while generating the response."


@bot.tree.command(
    name="search",
    description="Performs a Google search and returns a comprehensive answer based on the top 10 results."
)
@app_commands.describe(query="The search query.")
@universal_cooldown_check()
async def search_command(interaction: discord.Interaction, query: str):
    logger.info(f"🔍 Slash Google search requested by {interaction.user}: '{query}'")
    
    # Rate limiting check
    if is_rate_limited(interaction.user.id):
        await interaction.response.send_message(
            "⚠️ You are performing searches too quickly. Please wait a moment before trying again.",
            ephemeral=True
        )
        logger.warning(f"⚠️ Rate limit exceeded by {interaction.user}")
        return
    
    # Defer the response first
    await interaction.response.defer()
    
    # Initialize variables for retry logic
    max_retries = 3
    search_successful = False
    
    for attempt in range(max_retries):
        try:
            # Modify search strategy based on attempt number
            if attempt == 0:
                # First attempt: Standard search with temporal context
                temporally_aware_query = add_temporal_context(query)
                refined_query = await refine_search_query(temporally_aware_query)
            elif attempt == 1:
                # Second attempt: Broaden search terms and remove site-specific restrictions
                broader_query = f"{query} OR {' OR '.join(query.split())} -site:youtube.com -site:facebook.com"
                refined_query = await refine_search_query(broader_query)
            else:
                # Third attempt: Use alternative keywords and synonyms
                system_msg = {
                    "role": "system",
                    "content": "You are a search query optimizer. Generate alternative search terms using synonyms and related concepts."
                }
                user_msg = {"role": "user", "content": f"Generate alternative search terms for: {query}"}
                response = await async_chat_completion(
                    model=os.getenv("LOCAL_CHAT"),
                    messages=[system_msg, user_msg],
                    temperature=0.7,
                    max_tokens=100
                )
                alternative_query = response.choices[0].message.content.strip()
                refined_query = await refine_search_query(f"{query} OR {alternative_query}")
            
            logger.info(f"🔍 Attempt {attempt + 1}: Refined Query for {interaction.user}: '{refined_query}'")
            
            if not refined_query:
                continue
            
            # Perform the search
            search_results = list(search(refined_query, num_results=10))
            
            if search_results:
                search_successful = True
                logger.info(f"✅ Search successful on attempt {attempt + 1} for {interaction.user}")
                
                # Fetch summaries concurrently
                summaries = await asyncio.gather(*[fetch_summary(url) for url in search_results])
                
                # Compile results and continue with existing logic
                compiled_results = ""
                for idx, (url, summary) in enumerate(zip(search_results, summaries), start=1):
                    compiled_results += f"**Result {idx}:** {url}\n{summary}\n\n"
                
                # Generate LLM response and create embed
                llm_response = await generate_llm_response(compiled_results, query)
                
                embed = discord.Embed(
                    title=f"🔍 Search Results for: {query}",
                    description=llm_response,
                    color=discord.Color.green()
                )
                
                # Send results
                content = f"Your search response is complete {interaction.user.mention}"
                await interaction.channel.send(content=content, embed=embed)
                await interaction.followup.send("✅ Search results have been posted to the channel.", ephemeral=True)
                
                # Increment stats and break the retry loop
                await increment_user_stat(interaction.user.id, 'searches_performed')
                break
            
            logger.warning(f"⚠️ No results found on attempt {attempt + 1} for query: '{refined_query}'")
            
        except Exception as e:
            logger.error(f"❌ Error in search attempt {attempt + 1} for {interaction.user}: {format_error_message(e)}")
            if attempt == max_retries - 1:
                await interaction.followup.send(
                    f"❌ Error performing search after {max_retries} attempts: {format_error_message(e)}",
                    ephemeral=True
                )
                return
    
    # If all attempts failed to find results
    if not search_successful:
        await interaction.followup.send(
            f"❌ No results found after {max_retries} attempts with different search strategies. "
            "Please try rephrasing your query.",
            ephemeral=True
        )
        logger.warning(f"❌ All {max_retries} search attempts failed for {interaction.user}'s query: '{query}'")

"""
---------------------------------------------------------------------------------
Other commands
---------------------------------------------------------------------------------
"""

@bot.command(name='reloadenv', help='Reloads environment variables and text files (Owner only)')
async def reload_env(ctx):
    # Check if the user is in OWNER_IDS
    if ctx.author.id not in OWNER_IDS:
        await ctx.send("❌ You don't have permission to use this command.", ephemeral=True)
        logger.warning(f"Unauthorized attempt to reload env vars by {ctx.author}")
        return

    try:
        # Reload environment variables
        load_dotenv(override=True)

        # Reload text files
        global OVERALL_THEMES, CHARACTER_CONCEPTS, ARTISTIC_RENDERING_STYLES
        OVERALL_THEMES = load_text_file_from_env("OVERALL_THEMES")
        CHARACTER_CONCEPTS = load_text_file_from_env("CHARACTER_CONCEPTS")
        ARTISTIC_RENDERING_STYLES = load_text_file_from_env("ARTISTIC_RENDERING_STYLES")

        await ctx.send("✅ Environment variables and text files reloaded successfully!", ephemeral=True)
        logger.info(f"Environment variables and text files reloaded by {ctx.author}")

    except Exception as e:
        error_message = f"❌ Error reloading environment variables and text files: {str(e)}"
        await ctx.send(error_message, ephemeral=True)
        logger.error(f"Error during env and file reload by {ctx.author}: {str(e)}")


@bot.tree.command(name="helpsoupy", description="Displays all available commands.")
async def help_command(interaction: discord.Interaction):
    """
    Sends an embedded help message listing all available slash and prefix commands.
    """
    logger.info(f"📚 Command 'help' invoked by {interaction.user}")

    # Create an embed for the help message
    embed = discord.Embed(
        title="📚 Soupy Help Menu",
        description="Here's a list of all my available commands:",
        color=discord.Color.blue(),
        timestamp=datetime.utcnow()
    )

    # -------------------
    # List Slash Commands
    # -------------------
    slash_commands = bot.tree.get_commands()
    if slash_commands:
        slash_commands_str = ""
        for cmd in slash_commands:
            # Skip the help command itself to avoid redundancy
            if cmd.name == "help":
                continue
            # Add command name and description
            cmd_name = f"/{cmd.name}"
            cmd_desc = cmd.description or "No description provided."
            slash_commands_str += f"**{cmd_name}**: {cmd_desc}\n"
        embed.add_field(
            name="🔹 Slash Commands",
            value=slash_commands_str,
            inline=False
        )
    else:
        embed.add_field(
            name="🔹 Slash Commands",
            value="No slash commands available.",
            inline=False
        )

    # ---------------------
    # List Prefix Commands
    # ---------------------
    prefix_commands = [command for command in bot.commands if not isinstance(command, commands.Group)]
    if prefix_commands:
        prefix_commands_str = ""
        for cmd in prefix_commands:
            # Skip the help command itself to avoid redundancy
            if cmd.name == "help":
                continue
            # Add command name and help text
            cmd_name = f"!{cmd.name}"
            cmd_help = cmd.help or "No description provided."
            prefix_commands_str += f"**{cmd_name}**: {cmd_help}\n"
        embed.add_field(
            name="🔸 Prefix Commands",
            value=prefix_commands_str,
            inline=False
        )
    else:
        embed.add_field(
            name="🔸 Prefix Commands",
            value="No prefix commands available.",
            inline=False
        )

    # Optional: Add a footer or additional information
    embed.set_footer(
        text="Use the commands as shown above to interact with me!",
        icon_url=bot.user.avatar.url if bot.user.avatar else None
    )

    # Send the embed as an ephemeral message (visible only to the user)
    await interaction.response.send_message(embed=embed, ephemeral=True)
    logger.info(f"📚 Sent help menu to {interaction.user}")







@bot.tree.command(name="stats", description="Displays the top 5 users in each category: Images Generated, Chat Responses, and Mentions.")
@app_commands.checks.has_permissions(administrator=True)
async def stats_command(interaction: discord.Interaction):
    logger.info(f"Command 'stats' invoked by {interaction.user}")
    
    try:
        stats_data = await read_user_stats()
    
        if not stats_data:
            await interaction.response.send_message("No statistics available yet.", ephemeral=True)
            return
    
        # Convert stats_data to a list of dictionaries
        users_stats = []
        for user_id, data in stats_data.items():
            users_stats.append({
                "username": data.get("username", "Unknown"),
                "images_generated": data.get("images_generated", 0),
                "chat_responses": data.get("chat_responses", 0),
                "mentions": data.get("mentions", 0)
            })
    
        # Sort users for each category
        top_images = sorted(users_stats, key=lambda x: x["images_generated"], reverse=True)[:5]
        top_chats = sorted(users_stats, key=lambda x: x["chat_responses"], reverse=True)[:5]
        top_mentions = sorted(users_stats, key=lambda x: x["mentions"], reverse=True)[:5]
    
        # Create embed for better readability
        embed = discord.Embed(
            title="📊 Bot Usage Statistics",
            color=discord.Color.purple()
        )
        
        # Top Images Generated
        if top_images:
            images_field = "\n".join([f"{i+1}. **{user['username']}** - {user['images_generated']} images" for i, user in enumerate(top_images)])
        else:
            images_field = "No data available."
        embed.add_field(name="🏆 Top Images Generated", value=images_field, inline=False)
        
        # Top Chat Responses
        if top_chats:
            chats_field = "\n".join([f"{i+1}. **{user['username']}** - {user['chat_responses']} responses" for i, user in enumerate(top_chats)])
        else:
            chats_field = "No data available."
        embed.add_field(name="🏆 Top Chat Responses", value=chats_field, inline=False)
        
        # Top Mentions
        if top_mentions:
            mentions_field = "\n".join([f"{i+1}. **{user['username']}** - {user['mentions']} mentions" for i, user in enumerate(top_mentions)])
        else:
            mentions_field = "No data available."
        embed.add_field(name="🏆 Top Mentions", value=mentions_field, inline=False)
        
        embed.set_footer(text=f"Requested by {interaction.user}", icon_url=interaction.user.avatar.url if interaction.user.avatar else None)
    
        await interaction.response.send_message(embed=embed)
        logger.info(f"Sent statistics to {interaction.user}")
    
    except app_commands.errors.MissingPermissions:
        await interaction.response.send_message("❌ You don't have permission to use this command.", ephemeral=True)
        logger.warning(f"Unauthorized attempt to use 'stats' command by {interaction.user}")
    except Exception as e:
        await interaction.response.send_message("❌ An error occurred while fetching statistics.", ephemeral=True)
        logger.error(f"Error in 'stats' command: {e}")


@bot.tree.command(name="status", description="Displays the current status of the bot, Flux server, and chat functions.")
async def status_command(interaction: discord.Interaction):
    logger.info(f"Command 'status' invoked by {interaction.user}")
    
    # Defer the response immediately
    await interaction.response.defer()
    
    # Calculate uptime
    if bot_start_time:
        current_time = datetime.utcnow()
        uptime_duration = current_time - bot_start_time
        uptime_str = format_uptime(uptime_duration)
    else:
        uptime_str = "Uptime information not available."
    
    # Prepare Flux server status
    flux_status = "🟢 Online" if flux_server_online else "🔴 Offline"
    
    # Check chat functions status
    await check_chat_functions()
    chat_status = "🟢 Online" if chat_functions_online else "🔴 Offline"
    
    # Create an embed message
    embed = discord.Embed(
        title="Bot Status",
        color=discord.Color.blue()
    )
    embed.add_field(name="Flux Server", value=flux_status, inline=False)
    embed.add_field(name="Chat Functions", value=chat_status, inline=False)
    embed.add_field(name="Uptime", value=uptime_str, inline=False)
    embed.set_footer(text=f"Requested by {interaction.user}", icon_url=interaction.user.avatar.url if interaction.user.avatar else None)
    
    # Send as a followup instead of direct response
    await interaction.followup.send(embed=embed)
    logger.info(f"Sent status information to {interaction.user}")


magic_8ball_responses = [
    "It is certain.",
    "It is decidedly so.",
    "Without a doubt.",
    "Yes – definitely.",
    "You may rely on it.",
    "As I see it, yes.",
    "Most likely.",
    "You bet your ass.",
    "lol duh",
    "Outlook good.",
    "Yes.",
    "Signs point to yes.",
    "Don't count on it.",
    "My reply is no.",
    "My sources say no.",
    "Outlook not so good.",
    "Very doubtful.",
    "Absolutely not.",
    "What a stupid question.",
    "Are you stupid?",
    "This is the dumbest question I've ever heard."
]

@bot.tree.command(name="8ball", description="Ask the Magic 8-Ball a question and get a response.")
@app_commands.describe(question="Your question for the Magic 8-Ball.")
async def eight_ball_command(interaction: discord.Interaction, question: str):
    logger.info(f"Command '8ball' invoked by {interaction.user} with question: '{question}'")
    response = random.choice(magic_8ball_responses)
    await interaction.response.send_message(f'Question: "{question}"\nThe 8-Ball says: "{response}"')
    logger.info(f"Responded to {interaction.user} with: '{response}'")


# ---------------------------------------------------------------------------
# 9Ball Command
# ---------------------------------------------------------------------------

@bot.tree.command(
    name="9ball", 
    description="Ask the mystical 9-ball (local LLM) a question and receive a custom response."
)
@app_commands.describe(question="Your mystical question for the 9-ball.")
async def nine_ball_command(interaction: discord.Interaction, question: str):
    logger.info(f"Command '9ball' invoked by {interaction.user} with question: '{question}'")
    
    nineball_behaviour = os.getenv("9BALL", "You are a mystical 9-ball that provides enigmatic answers.")
    
    # Compose system and user prompts
    system_prompt = {"role": "system", "content": nineball_behaviour}
    user_prompt = {"role": "user", "content": question}
    messages_for_llm = [system_prompt, user_prompt]
    
    # Debug logging
    formatted_messages = format_messages(messages_for_llm)
    logger.debug(f"Sending the following messages to LLM (9ball):\n{formatted_messages}")
    
    try:
        # Defer the interaction to extend the response time
        await interaction.response.defer()
        
        async with interaction.channel.typing():
            response = await async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=messages_for_llm,
                temperature=0.8,
                max_tokens=45 
            )
            reply = response.choices[0].message.content.strip()
        
        # Send the response as a follow-up with the question included
        await interaction.followup.send(f'Question: "{question}"\nThe 9-Ball says: "{reply}"')
        logger.info(f"Responded to {interaction.user} with: '{reply}'")
    
    except Exception as e:
        error_msg = f"Error in '9ball' command for {interaction.user}: {format_error_message(e)}"
        logger.error(error_msg)
        try:
            # Attempt to send the error message as a follow-up
            await interaction.followup.send(error_msg, ephemeral=True)
        except discord.errors.HTTPException:
            # If follow-up fails, log the error
            logger.error("Failed to send follow-up error message for '9ball' command.")



@bot.tree.command(name="whattime", description="Fetches and displays the current time in a specified city.")
@app_commands.describe(location="The city for which to fetch the current time.")
async def whattime_command(interaction: discord.Interaction, location: str):
    logger.info(f"Command 'whattime' invoked by {interaction.user} for location: '{location}'")
    
    try:
        geolocator = Nominatim(user_agent="discord_bot_soupy")
        location_obj = geolocator.geocode(location, addressdetails=True, language='en', timeout=10)
        if not location_obj:
            raise ValueError(f"Could not geocode the location: {location}")
    
        address = location_obj.raw.get('address', {})
        country = address.get('country', 'Unknown country')
        admin_area = address.get('state', address.get('region', address.get('county', '')))
    
        is_country_query = location.strip().lower() == country.lower()
        location_str = country if is_country_query else f"{location.title()}, {country}"
        if admin_area and not is_country_query:
            location_str = f"{location.title()}, {admin_area}, {country}"
    
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=location_obj.longitude, lat=location_obj.latitude)
        if not timezone_str:
            raise ValueError(f"Could not find timezone for the location: {location}")
    
        timezone = pytz.timezone(timezone_str)
        current_time = datetime.now(timezone).strftime('%I:%M %p on %Y-%m-%d')
        await interaction.response.send_message(f"It is currently {current_time} in {location_str}.")
        logger.info(f"Provided time for {interaction.user}: {current_time} in {location_str}")
    
    except ValueError as e:
        await interaction.response.send_message(str(e), ephemeral=True)
        logger.error(f"[/whattime Command Error] {e}")
    except Exception as e:
        await interaction.response.send_message("Sorry, I'm unable to process your request at the moment.", ephemeral=True)
        logger.error(f"[/whattime Command Exception] An error occurred: {e}")


"""
---------------------------------------------------------------------------------
Flux-Related Functionality
---------------------------------------------------------------------------------
(Slash commands, queue processing, rate limiting, etc.)
---------------------------------------------------------------------------------
"""

# Track user interactions for rate limiting
# => user_interaction_timestamps = defaultdict(list)  # Already defined above

async def user_has_exempt_role(interaction: discord.Interaction) -> bool:
    """Checks if the user has any of the exempt roles for rate-limiting."""
    member = interaction.user
    if isinstance(member, discord.Member):
        user_roles = {role.name.lower() for role in member.roles}
        if EXEMPT_ROLES.intersection(user_roles):
            return True
    return False





def generate_unique_filename(prompt, extension=".png"):
    """Generate a unique filename based on prompt and timestamp."""
    base_filename = re.sub(r'\W+', '', prompt[:80]).lower()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base_filename}_{timestamp}{extension}"


@bot.tree.command(name="flux", description="Generates an image using the Flux model.")
@app_commands.describe(
    description="Description of the image to generate",  # Removed "(leave empty for random)"
    size="Size of the image",
    seed="Seed for random generation"
)
@app_commands.choices(size=[
    app_commands.Choice(name="Default (1024x1024)", value="default"),
    app_commands.Choice(name="Wide (1920x1024)", value="wide"),
    app_commands.Choice(name="Tall (1024x1920)", value="tall"),
    app_commands.Choice(name="Small (512x512)", value="small"),
])
async def flux(interaction: discord.Interaction, 
               description: str,  # Removed Optional[], making it required
               size: Optional[app_commands.Choice[str]] = None,
               seed: Optional[int] = None):
    size_value = size.value if size else 'default'
    
    logger.info(f"🎨 Slash Command 'flux' invoked by {interaction.user} with description: '{description}', size: '{size_value}', seed: '{seed if seed else 'random'}'")
    await interaction.response.send_message("🛠️ Your image request has been queued...", ephemeral=True)
    await bot.flux_queue.put({
        'type': 'flux',
        'interaction': interaction,
        'description': description,
        'size': size_value,
        'seed': seed,
    })
    logger.info(f"🎨 Queued image generation for {interaction.user}: description='{description}', size='{size_value}', seed='{seed if seed else 'random'}'")



# -------------------------------------------------------------------------
# Define the button-handling methods BEFORE process_flux_queue()
# -------------------------------------------------------------------------

async def handle_remix(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Remix", queue_size=queue_size)

async def handle_wide(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Wide", queue_size=queue_size)

async def handle_tall(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Tall", queue_size=queue_size)

async def handle_edit(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation
    await generate_flux_image(interaction, prompt, width, height, seed, action_name="Edit", queue_size=queue_size)


fancy_instructions = os.getenv("FANCY", "")


async def handle_fancy(interaction, prompt, width, height, seed, queue_size):
    """
    1) Uses the local LLM to rewrite the current prompt as something more detailed and 'fancy.'
    2) Strips any unwanted prefix from the newly minted fancy prompt using regex.
    3) Calls generate_flux_image() with the cleaned fancy prompt.
    4) Calculates total duration including prompt rewriting and image generation.
    """
    try:
        # Use typing context manager for consistent behavior
        async with interaction.channel.typing():
            # Start timing for prompt rewriting
            prompt_start_time = time.perf_counter()

            # Step 1: Fetch your fancy instructions from the .env variable
            fancy_instructions = os.getenv("FANCY", "")
            logger.debug(f"📜 Retrieved 'FANCY' instructions: {fancy_instructions}")

            # Combine them with the user's prompt
            rewriting_instructions = (
                f"{fancy_instructions}\n\n"
                f"The prompt you are elaborating on is: {prompt}"
            )
            logger.debug(f"📜 Combined rewriting instructions for {interaction.user}: {rewriting_instructions}")  # Fixed variable name here

            system_prompt = {"role": "system", "content": rewriting_instructions}
            user_prompt = {"role": "user", "content": "Please rewrite the above prompt accordingly."}

            # We'll gather these into a message list
            messages_for_llm = [system_prompt, user_prompt]

            # Add DEBUG Logging Here
            formatted_messages = format_messages(messages_for_llm)
            logger.debug(f"📜 Sending the following messages to LLM (Fancy):\n{formatted_messages}")

            # Call your local LLM with these messages
            response = await async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=messages_for_llm,
                temperature=0.7,
                max_tokens=150
            )

            # Extract the fancy prompt from the LLM's response
            fancy_prompt = response.choices[0].message.content.strip()
            logger.info(f"🪄 Fancy prompt generated for {interaction.user}: '{fancy_prompt}'")

            # Calculate prompt rewriting duration
            prompt_end_time = time.perf_counter()
            prompt_duration = prompt_end_time - prompt_start_time
            logger.info(f"⏱️ Prompt rewriting time for {interaction.user}: {prompt_duration:.2f} seconds")

            # Step 2: Apply regex removal to clean the fancy prompt
            fancy_prompt_cleaned = remove_all_before_colon(fancy_prompt)
            logger.debug(f"🪄 Cleaned fancy prompt for {interaction.user}: '{fancy_prompt_cleaned}'")

            # Step 3: Pass the cleaned fancy prompt to generate_flux_image
            await generate_flux_image(
                interaction,
                fancy_prompt_cleaned,  # Use the cleaned fancy prompt
                width,
                height,
                seed,
                action_name="Fancy",  # So your logs show it's from the Fancy button
                queue_size=queue_size,
                pre_duration=prompt_duration  # Pass the prompt rewriting duration
            )
            logger.info(f"🪄 Passed cleaned fancy prompt to image generator for {interaction.user}")
            
            # Increment the images_generated stat after successful generation
            await increment_user_stat(interaction.user.id, 'images_generated')
        
    except Exception as e:
        logger.error(f"🪄 Error generating fancy prompt for {interaction.user}: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message(f"❌ Error generating fancy prompt: {e}", ephemeral=True)
        else:
            await interaction.followup.send(f"❌ Error generating fancy prompt: {e}", ephemeral=True)




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
            max_length=2000
        )
        self.width_input = TextInput(
            label="📏 Width",
            style=discord.TextStyle.short,
            default=str(width),
            required=True,
            min_length=1,
            max_length=5
        )
        self.height_input = TextInput(
            label="📐 Height",
            style=discord.TextStyle.short,
            default=str(height),
            required=True,
            min_length=1,
            max_length=5
        )
        self.seed_input = TextInput(
            label="🌱 Seed",
            style=discord.TextStyle.short,
            default=str(seed) if seed is not None else "",
            required=False,
            max_length=10
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

            await bot.flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'edit',
                'prompt': new_prompt,
                'width': new_width,
                'height': new_height,
                'seed': new_seed,
            })

            logger.info(f"Edit requested: prompt='{new_prompt}', dimensions={new_width}x{new_height}, seed={new_seed}")
        except Exception as e:
            await interaction.followup.send("❌ An error occurred while processing your edit.", ephemeral=True)
            logger.error(f"Error in EditImageModal submission: {e}")

# UI view class for image remixing and manipulation
class FluxRemixView(View):
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

    @discord.ui.button(label="✏️ Edit", style=discord.ButtonStyle.success, custom_id="flux_edit_button", row=0)
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

    @discord.ui.button(label="🪄 Fancy", style=discord.ButtonStyle.primary, custom_id="flux_fancy_button", row=0)
    @universal_cooldown_check()
    async def fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Fancy' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.defer(thinking=True, ephemeral=True)
            queue_size = bot.flux_queue.qsize()
            await bot.flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'fancy',
                'prompt': self.cleaned_prompt,  # The original prompt
                'width': self.width,
                'height': self.height,
                'seed': self.seed,
            })
            logger.info(f"Enqueued 'Fancy' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={self.width}x{self.height}, seed={self.seed}")
        except Exception as e:
            logger.error(f"Error during fancy transformation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error during fancy transformation.", ephemeral=True)

    @discord.ui.button(label="🌱 Remix", style=discord.ButtonStyle.primary, custom_id="flux_remix_button", row=0)
    @universal_cooldown_check()
    async def remix_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Remix' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Remixing...", ephemeral=True)
            queue_size = bot.flux_queue.qsize()
            new_seed = random.randint(0, 2**32 - 1)
            await bot.flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'remix',
                'prompt': self.cleaned_prompt,
                'width': self.width,
                'height': self.height,
                'seed': new_seed,
            })
            logger.info(f"Enqueued 'Remix' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={self.width}x{self.height}, seed={new_seed}")
            
            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, 'images_generated')
        except Exception as e:
            logger.error(f"Error during remix for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error during remix.", ephemeral=True)


    @discord.ui.button(label="🔀 Random", style=discord.ButtonStyle.danger, custom_id="flux_random_button", row=1)
    @universal_cooldown_check()
    async def random_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """
        Handler for the 'Random' button. Generates a new random prompt by selecting
        one term from each category and appending them to the base RANDOMPROMPT.
        Also randomly selects image dimensions.
        """
        logger.info(f"🔀 'Random' button clicked by {interaction.user}.")
        try:
            await interaction.response.send_message("🛠️ Generating random image...", ephemeral=True)
            
            # Randomly select dimensions with equal probability
            dimensions = [
                (1024, 1024),  # Square
                (1920, 1024),  # Wide
                (1024, 1920)   # Tall
            ]
            width, height = random.choice(dimensions)
            
            queue_size = bot.flux_queue.qsize()
            await bot.flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'random',
                'width': width,
                'height': height,
                'seed': None  # Random will generate its own seed
            })
            logger.info(f"🔀 Enqueued 'Random' action for {interaction.user} with dimensions {width}x{height}")
            
            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, 'images_generated')
        except Exception as e:
            logger.error(f"🔀 Error queueing random generation for {interaction.user}: {e}")

    @discord.ui.button(label="📏 Wide", style=discord.ButtonStyle.primary, custom_id="flux_wide_button", row=1)
    @universal_cooldown_check()
    async def wide_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Wide' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Generating wide version...", ephemeral=True)
            queue_size = bot.flux_queue.qsize()
            await bot.flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'wide',
                'prompt': self.cleaned_prompt,
                'width': 1920,
                'height': 1024,
                'seed': self.seed,
            })
            logger.info(f"Enqueued 'Wide' action for {interaction.user}: prompt='{self.cleaned_prompt}', size=1920x1024, seed={self.seed}")
        except Exception as e:
            logger.error(f"Error during wide generation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating wide version.", ephemeral=True)

    @discord.ui.button(label="📐 Tall", style=discord.ButtonStyle.primary, custom_id="flux_tall_button", row=1)
    @universal_cooldown_check()
    async def tall_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Tall' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Generating tall version...", ephemeral=True)
            queue_size = bot.flux_queue.qsize()
            await bot.flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'tall',
                'prompt': self.cleaned_prompt,
                'width': 1024,
                'height': 1920,
                'seed': self.seed,
            })
            logger.info(f"Enqueued 'Tall' action for {interaction.user}: prompt='{self.cleaned_prompt}', size=1024x1920, seed={self.seed}")
        except Exception as e:
            logger.error(f"Error during tall generation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating tall version.", ephemeral=True)


async def generate_flux_image(
    interaction,
    prompt,
    width,
    height,
    seed,
    action_name="Flux",
    queue_size=0,
    pre_duration=0,
    selected_terms: Optional[str] = None  # New parameter
):
    flux_server_url = FLUX_SERVER_URL.rstrip('/')  # Ensure no trailing slash
    num_steps = 4
    guidance = 3.5
    payload = {
        "prompt": prompt,
        "steps": str(num_steps),
        "guidance_scale": str(guidance),
        "width": str(width),
        "height": str(height),
        "seed": str(seed)
    }
    try:
        # Use typing context manager for consistent behavior
        async with interaction.channel.typing():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                # Start timing the image generation process
                image_start_time = time.perf_counter()

                async with session.post(f"{flux_server_url}/flux", data=payload) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        
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
                        safe_prompt = re.sub(r'\W+', '', prompt[:40]).lower()
                        filename = f"{random_number}_{safe_prompt}.png"  # Changed to .png

                        # Create a Discord File object from the image bytes
                        image_file = discord.File(BytesIO(image_bytes), filename=filename)

                        # Create embed messages
                        if selected_terms:
                            # If selected_terms are provided, include them in the description
                            description_content = f"**Selected Terms:** {selected_terms}\n\n**Prompt:** {prompt}"
                        else:
                            # Fallback to just the prompt
                            description_content = prompt

                        description_embed = discord.Embed(
                            description=description_content, color=discord.Color.blue()
                        )
                        details_embed = discord.Embed(color=discord.Color.green())

                        queue_total = queue_size + 1
                        details_text = f"🌱 {seed} 🔄 {action_name} ⏱️ {total_duration:.2f}s 📋 {queue_total}"

                        # Change this line to set the description instead of adding a field
                        details_embed.description = details_text

                        # Initialize the FluxRemixView with current image parameters
                        new_view = FluxRemixView(
                            prompt=prompt, width=width, height=height, seed=seed
                        )

                        # Send the generated image to the Discord channel
                        await interaction.channel.send(
                            content=f"{interaction.user.mention} 🖼️ Generated Image:",
                            embeds=[description_embed, details_embed],
                            file=image_file,
                            view=new_view
                        )
                        logger.info(
                            f"🖼️ Image generation completed for {interaction.user}: filename='{filename}', total_duration={total_duration:.2f}s"
                        )
                    else:
                        logger.error(f"🖼️ Flux server error for {interaction.user}: HTTP {response.status}")
                        if not interaction.followup.is_done():
                            await interaction.followup.send(
                                f"❌ Flux server error: HTTP {response.status}", ephemeral=True
                            )
    except (ClientConnectorError, ClientOSError):
        logger.error(f"🖼️ Flux server is offline or unreachable for {interaction.user}.")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    "❌ The Flux server is currently offline.", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except ServerTimeoutError:
        logger.error(f"🖼️ Flux server request timed out for {interaction.user}.")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    "❌ The Flux server timed out while processing your request. Please try again later.", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except Exception as e:
        logger.error(f"🖼️ Unexpected error during image generation for {interaction.user}: {e}")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    f"❌ An unexpected error occurred during image generation: {e}", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")







async def process_flux_image(interaction: discord.Interaction, description: str, size: str, seed: Optional[int]):
    """Entry point for slash command /flux tasks to push work into generate_flux_image."""
    try:
        # Default dims
        width, height = 1024, 1024

        if size == 'wide':
            width, height = 1920, 1024
        elif size == 'tall':
            width, height = 1024, 1920
        elif size == 'small':
            width, height = 512, 512

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        logger.info(f"Processing request: user={interaction.user}, prompt='{description}', size='{size}', dims={width}x{height}, seed={seed}")
        
        # Update image generation count
        await increment_user_stat(interaction.user.id, 'images_generated')

        await generate_flux_image(interaction, description, width, height, seed, queue_size=bot.flux_queue.qsize())
    except Exception as e:
        if not interaction.followup.is_done():
            await interaction.followup.send(f"❌ An error occurred: {str(e)}", ephemeral=True)
        logger.error(f"Error in process_flux_image: {e}")



def should_randomly_respond(probability=0.03) -> bool:
    """
    Returns True with the given probability (e.g., 3%).
    """
    return random.random() < probability


"""
---------------------------------------------------------------------------------
Event Handlers
---------------------------------------------------------------------------------
"""

# Add this BEFORE your bot.event decorators and command definitions
async def load_extensions():
    """Load all extension cogs"""
    # Get the directory where soupy.py is located
    current_dir = Path(__file__).parent
    
    # Load the interject cog
    try:
        await bot.load_extension("interject")
        logger.info("📥 Loaded interject extension")
    except Exception as e:
        logger.error(f"❌ Failed to load interject extension: {e}")

# Then your existing on_ready event can use it
@bot.event
async def on_ready():
    global bot_start_time
    bot_start_time = datetime.utcnow()
    
    # Load extensions
    await load_extensions()
    
    # Existing on_ready operations
    logger.info(f"Logged in as {bot.user.name}")
    logger.info("Bot is ready for commands!")
    logger.info(f'🔵 Bot ready: {bot.user} (ID: {bot.user.id})')

    # Sync slash commands
    await bot.tree.sync()
    
    # Launch the flux queue processor
    bot.loop.create_task(bot.flux_queue.process_queue())
    
    # Set the bot start time
    bot_start_time = datetime.utcnow()
    logger.info(f"Bot start time set to {bot_start_time} UTC")
    
    # Create the online embed
    online_embed = discord.Embed(
        # title="Soupy is online.",
        description="Soupy is now online.",
        color=discord.Color.green(),
    )
    online_embed.set_footer(text="Soupy Bot | Online", icon_url=bot.user.avatar.url if bot.user.avatar else None)
    
    # Notify designated channels that the bot is online with the embed
    try:
        await notify_channels(embed=online_embed)
        logger.info("✅ Notified channels about bot startup.")
    except Exception as e:
        logger.error(f"❌ Failed to notify channels about bot startup: {e}")
    
    logger.info("Flux server monitor task has been started.")



# Regex to capture everything from the start of the line until the first colon (:),
# plus any trailing spaces after the colon (.*?:\s*).
remove_all_before_colon_pattern = re.compile(r'^.*?:\s*', re.IGNORECASE)

def remove_all_before_colon(text: str) -> str:
    """
    Removes everything from the start of the line until the first colon (inclusive),
    plus optional spaces after it, ignoring case.
    Example: "This is multiple words: hello" -> "hello"
    """
    return remove_all_before_colon_pattern.sub("", text, count=1)



def split_message(msg: str, max_len=1500):
    """
    Splits a long string into chunks so each chunk is <= max_len characters.
    """
    parts = []
    while len(msg) > max_len:
        idx = msg.rfind(' ', 0, max_len)
        if idx == -1:
            idx = max_len
        parts.append(msg[:idx])
        msg = msg[idx:].strip()
    parts.append(msg)
    return parts

@bot.event
async def on_message(message):
    # Skip bot messages
    if message.author == bot.user:
        return

    # Process commands first
    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.process_commands(message)
        return

    # Initialize image_descriptions list to store any image analysis results
    image_descriptions = []

    # Check for image attachments and analyze them silently
    if message.attachments:
        for idx, attachment in enumerate(message.attachments, 1):
            # Check if the attachment is an image
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                try:
                    # Download the image
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                
                                # Create form data with the image
                                form_data = aiohttp.FormData()
                                form_data.add_field('file',
                                    image_data,
                                    filename='image.png',
                                    content_type='image/png'
                                )
                                
                                # Send to analysis endpoint
                                async with session.post(
                                    os.getenv('ANALYZE_IMAGE_API_URL'),
                                    data=form_data
                                ) as analysis_resp:
                                    if analysis_resp.status == 200:
                                        result = await analysis_resp.json()
                                        description = result.get('description', 'No description available')
                                        # Format the description as a user message
                                        formatted_desc = f"{message.author.name}: [shares an image: {description}]"
                                        image_descriptions.append(formatted_desc)
                                        logger.info(f"Generated image description for {message.author}: {description}")
                                    else:
                                        logger.error(f"Failed to analyze image: HTTP {analysis_resp.status}")
                            else:
                                logger.error(f"Failed to download image: HTTP {resp.status}")
                except Exception as e:
                    logger.error(f"Error analyzing image: {e}")

    # Continue with existing message handling
    should_respond = should_bot_respond_to_message(message) or should_randomly_respond(probability=0.03)
    if should_respond:
        logger.info(f"📝 Processing message from {message.author}: '{message.content}'")
        
        async with message.channel.typing():
            try:
                # Fetch context from channel
                recent_messages = await fetch_recent_messages(message.channel, limit=25, current_message_id=message.id)
                
                # Create the messages list starting with the system behavior prompt
                messages_for_llm = [{"role": "system", "content": chatgpt_behaviour}]
                
                # Add conversation history
                messages_for_llm.extend(recent_messages)
                
                # If there are image descriptions, insert them right before the current message
                if image_descriptions:
                    # Insert image context as a user message right before the current message
                    messages_for_llm.append({
                        "role": "user", 
                        "content": "\n".join(image_descriptions)
                    })
                
                # Add the current user message
                user_message = {
                    "role": "user", 
                    "content": f"{message.author.name}: {message.content}"
                }
                messages_for_llm.append(user_message)

                # Debug logging
                formatted_messages = format_messages(messages_for_llm)
                logger.debug(f"📜 Sending the following messages to LLM:\n{formatted_messages}")

                async with message.channel.typing():
                    response = await async_chat_completion(
                        model=os.getenv("LOCAL_CHAT"),
                        messages=messages_for_llm,
                        temperature=0.8,
                        max_tokens=max_tokens_default
                    )
                    reply = response.choices[0].message.content
                logger.info(f"🤖 Generated reply for {message.author}: '{reply}'")

                # Update chat response count
                await increment_user_stat(message.author.id, 'chat_responses')

                # Split the bot's entire reply into smaller chunks
                chunks = split_message(reply, max_len=1500)
                
                for chunk in chunks:
                    # Remove everything before the first colon and any quotation marks
                    cleaned_chunk = remove_all_before_colon(chunk)
                    logger.debug(f"✂️ Sending cleaned chunk to {message.channel}: '{cleaned_chunk}'")
                    await message.channel.send(cleaned_chunk)
                    await asyncio.sleep(RATE_LIMIT)
                logger.info(f"✅ Successfully sent reply to {message.author}")
            except Exception as e:
                logger.error(f"❌ Error generating AI response for {message.author}: {format_error_message(e)}")

    # Process commands again if needed
    await bot.process_commands(message)


def handle_signal(signum, frame):
    """Handle termination signals by scheduling the shutdown coroutine."""
    logger.info(f"🛑 Received termination signal ({signum}). Initiating shutdown...")
    
    # Get the current event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(shutdown())
        else:
            loop.run_until_complete(shutdown())
    except Exception as e:
        logger.error(f"❌ Error in signal handler: {e}")
        sys.exit(1)



# Shutdown is handled by the signal handler
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

"""
---------------------------------------------------------------------------------
Final: run the bot
---------------------------------------------------------------------------------
"""

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)


