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
import base64
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
from datetime import datetime, timedelta, time as datetime_time, timezone
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Tuple
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
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.adapters import AdapterHTTPError
from openai import OpenAI, OpenAIError
from timezonefinder import TimezoneFinder
from logging.handlers import RotatingFileHandler
import html2text
import trafilatura
from PIL import Image, ImageDraw
import soupy_search
import soupy_imagesearch
import aiohttp
import numpy as np
import cv2
from soupy_database import setup_scan_command, get_active_scans, process_scan_triggers
from soupy_database.helpers import extract_urls, extract_url_content
from soupy_database.runtime_flags import is_rag_enabled
from soupy_database.rag import (
    build_rag_retrieval_query,
    fetch_rag_context_for_query,
    strip_rag_gate_word,
    strip_rag_query_invocations,
)
from soupy_database.self_context import (
    is_self_md_enabled,
    get_self_md_for_injection,
    add_notable_interaction,
    reflect_and_update as self_md_reflect,
    pending_interaction_count,
    load_self_md,
)

# Logging and color imports
import colorama
from colorama import Fore, Style
from colorlog import ColoredFormatter

# Initialize colorama
colorama.init(autoreset=True)

# URL processing cache and constants
url_cache: Dict[str, Tuple[Optional[str], float]] = {}
URL_CACHE_TTL = 3600  # 1 hour in seconds

# URL processing environment variables:
# MAX_URLS_PER_MESSAGE - Maximum number of URLs to process per message (default: 3)
# URL_FETCH_TIMEOUT - Timeout for URL requests in milliseconds (default: 15000)
# URL_MAX_CONTENT_LENGTH - Maximum length of extracted content (default: 800)
# URL_INCLUDE_DOMAIN - Whether to include domain in URL content (default: true)
# Note: extract_urls and extract_url_content are now imported from soupy_database.helpers

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

# ---------------------------------------------------------------------------
# Logging policy
#
# Console output is filtered by LOG_LEVEL (default INFO) so the terminal stays
# readable while the bot is running. The file handler always captures DEBUG,
# so full detail is recoverable from logs/soupy.log when investigating.
#
# Conventions used throughout the codebase:
#   INFO    — what the bot is doing right now: lifecycle (start/stop/cog load),
#             per-reply trigger and final text, loop cycle start/end, retries,
#             fallbacks, summary numbers. Roughly 5–10 lines per chat reply.
#   DEBUG   — internals: per-row RAG hits, per-message DB connect, token-budget
#             breakdowns, cache hits, candidate dumps, function traces.
#   WARNING — recovered or skipped: empty results, retry exhausted, fallback engaged.
#   ERROR   — failed: exceptions, send failures, backend unreachable.
#
# When adding a new log line, ask: "would the user need this every time, or
# only when investigating something specific?" If the latter, use DEBUG.
# ---------------------------------------------------------------------------

# Pre-load .env-stable so LOG_LEVEL can drive the handler on the next line.
# (load_dotenv runs again below with override=True for the rest of the env.)
load_dotenv('.env-stable')

# Set up handlers
console_handler = logging.StreamHandler(sys.stdout)
_console_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
console_handler.setLevel(_console_log_level)
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

# Suppress noisy debug logs from external libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("discord.client").setLevel(logging.INFO)
logging.getLogger("discord.gateway").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Console level: {logging.getLevelName(_console_log_level)} (set LOG_LEVEL to change). File: {log_filepath} (DEBUG).")

"""
---------------------------------------------------------------------------------
Load Environment Variables
---------------------------------------------------------------------------------
"""

# Load Environment Variables
load_dotenv('.env-stable', override=True)

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

# Load SD-specific keyword list (optional but recommended for SD3.5)
SD_KEYWORDS_LIST = load_text_file_from_env("SD_KEYWORDS")


chatgpt_behaviour = os.getenv("BEHAVIOUR", "You're a stupid bot.")
max_tokens_default = int(os.getenv("MAX_TOKENS", "800"))

# Stable Diffusion-specific environment vars
MAX_INTERACTIONS_PER_MINUTE = int(os.getenv("MAX_INTERACTIONS_PER_MINUTE", 4))
LIMIT_EXCEPTION_ROLES = os.getenv("LIMIT_EXCEPTION_ROLES", "")
EXEMPT_ROLES = {role.strip().lower() for role in LIMIT_EXCEPTION_ROLES.split(",") if role.strip()}
DISCORD_BOT_TOKEN = os.getenv("DISCORD_TOKEN")

# Image resolution settings
SD_DEFAULT_WIDTH = int(os.getenv("SD_DEFAULT_WIDTH", 1024))
SD_DEFAULT_HEIGHT = int(os.getenv("SD_DEFAULT_HEIGHT", 1024))
SD_WIDE_WIDTH = int(os.getenv("SD_WIDE_WIDTH", 1920))
SD_WIDE_HEIGHT = int(os.getenv("SD_WIDE_HEIGHT", 1088))
SD_TALL_WIDTH = int(os.getenv("SD_TALL_WIDTH", 1088))
SD_TALL_HEIGHT = int(os.getenv("SD_TALL_HEIGHT", 1920))

if not DISCORD_BOT_TOKEN:
    raise ValueError("No DISCORD_TOKEN environment variable set.")

SD_SERVER_URL = os.getenv("SD_SERVER_URL")

if not SD_SERVER_URL:
    raise ValueError("No SD_SERVER_URL environment variable set.")

# Img2Img/Inpaint/Hybrid Outpaint URLs
SD_IMG2IMG_URL = os.getenv("SD_IMG2IMG_URL")
SD_INPAINT_URL = os.getenv("SD_INPAINT_URL")
SD_OUTPAINT_HYBRID_URL = os.getenv("SD_OUTPAINT_HYBRID_URL")
    
# Provide fallbacks to the main SD server if specific endpoints are not configured
if SD_SERVER_URL:
    base_url = SD_SERVER_URL.rstrip('/')
    if not SD_IMG2IMG_URL:
        SD_IMG2IMG_URL = f"{base_url}/sd_img2img"
        logger.info(f"Using fallback SD_IMG2IMG_URL: {SD_IMG2IMG_URL}")
    if not SD_INPAINT_URL:
        SD_INPAINT_URL = f"{base_url}/sd_inpaint"
        logger.info(f"Using fallback SD_INPAINT_URL: {SD_INPAINT_URL}")
    if not SD_OUTPAINT_HYBRID_URL:
        SD_OUTPAINT_HYBRID_URL = f"{base_url}/outpaint_hybrid"
        logger.info(f"Using fallback SD_OUTPAINT_HYBRID_URL: {SD_OUTPAINT_HYBRID_URL}")

# Hybrid outpaint tuning (env-configurable)
OUTPAINT_USE_CANNY = os.getenv("OUTPAINT_USE_CANNY", "true").lower() == "true"
OUTPAINT_USE_DEPTH = os.getenv("OUTPAINT_USE_DEPTH", "false").lower() == "true"
OUTPAINT_CONTROL_WEIGHT = float(os.getenv("OUTPAINT_CONTROL_WEIGHT", 0.75))
OUTPAINT_HARMONIZE_STRENGTH = float(os.getenv("OUTPAINT_HARMONIZE_STRENGTH", 0.0))
    
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

async def get_guild_behaviour(guild_id: str) -> str:
    """Get the behaviour prompt. Single behaviour for all guilds.
    Reads the BEHAVIOUR env var from .env-stable.
    """
    return os.getenv("BEHAVIOUR", "You're a stupid bot.")

def format_error_message(error):
    error_prefix = "Error: "
    if isinstance(error, OpenAIError):
        return f"{error_prefix}An OpenAI API error occurred: {str(error)}"
    return f"{error_prefix}{str(error)}"

# Dictionary to store image descriptions for context
image_descriptions = []

# Persistent storage for image descriptions by message ID
# Maps message_id -> list of formatted image descriptions
message_image_descriptions: Dict[int, List[str]] = {}

"""
---------------------------------------------------------------------------------
Discord Bot Setup
---------------------------------------------------------------------------------
"""


# Move this section to the top of the file, after your imports but before other code
class SoupyBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sd_queue = SDQueue()
        
    # Add the async_chat_completion method to the bot class
    async def async_chat_completion(self, *args, **kwargs):
        """Wraps the OpenAI chat completion in an async context"""
        return await asyncio.to_thread(client.chat.completions.create, *args, **kwargs)

# First, let's add a proper Queue class to manage the image generation queue
class SDQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._shutdown = False
        self.current_size = 0

    async def put(self, item):
        self.current_size += 1
        await self._queue.put(item)

    async def get(self):
        item = await self._queue.get()
        self.current_size -= 1
        return item

    def qsize(self):
        return self.current_size

    async def initiate_shutdown(self):
        self._shutdown = True
        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def process_queue(self):
        """Process items in the queue."""
        while not self._shutdown:
            try:
                item = await self.get()

                if item['type'] == 'sd':
                    await process_sd_image(
                        item['interaction'],
                        item['description'],
                        item['size'],
                        item['seed']
                    )
                elif item['type'] == 'outpaint':
                    await handle_outpaint(
                        item['interaction'],
                        item['prompt'],
                        item['direction'],
                        item['width'],
                        item['height'],
                        item['seed'],
                        self.qsize(),
                        item.get('strength', 0.8),
                        item.get('steps'),
                        item.get('guidance')
                    )
                elif item['type'] == 'button':
                    if item['action'] == 'random':
                        # Pass the prompt directly to handle_random if it exists
                        await handle_random(
                            item['interaction'],
                            item['width'],
                            item['height'],
                            self.qsize(),
                            direct_prompt=item.get('prompt')
                        )
                    elif item['action'] == 'remix':
                        await handle_remix(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize()
                        )
                    elif item['action'] == 'fancy':
                        await handle_fancy(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize()
                        )
                    elif item['action'] == 'wide':
                        await handle_wide(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize()
                        )
                    elif item['action'] == 'tall':
                        await handle_tall(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize()
                        )
                    elif item['action'] == 'edit':
                        await handle_edit(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize()
                        )
                    elif item['action'] == '2x2_grid':
                        await handle_2x2_grid(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize()
                        )
                    elif item['action'] == 'outpaint_horizontal':
                        await handle_outpaint(
                            item['interaction'],
                            item['prompt'],
                            'horizontal',
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize(),
                            item.get('strength', 0.8),
                            item.get('steps'),
                            item.get('guidance')
                        )
                    elif item['action'] == 'outpaint_vertical':
                        await handle_outpaint(
                            item['interaction'],
                            item['prompt'],
                            'vertical',
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize(),
                            item.get('strength', 0.8),
                            item.get('steps'),
                            item.get('guidance')
                        )
                    elif item['action'] == 'outpaint_both':
                        await handle_outpaint(
                            item['interaction'],
                            item['prompt'],
                            'both',
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize(),
                            item.get('strength', 0.8),
                            item.get('steps'),
                            item.get('guidance')
                        )
                    elif item['action'] == 'thumbnail_upscale':
                        await handle_thumbnail_upscale(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['thumbnail_data'],
                            self.qsize(),
                            item['thumbnail_index']
                        )
                    elif item['action'] == 'regenerate_selected':
                        await handle_regenerate_selected(
                            item['interaction'],
                            item['prompt'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize(),
                            item['thumbnail_index']
                        )
                    elif item['action'] == 'outpaint':
                        await handle_outpaint(
                            item['interaction'],
                            item['prompt'],
                            item['direction'],
                            item['width'],
                            item['height'],
                            item['seed'],
                            self.qsize(),
                            item.get('strength', 0.8),
                            item.get('steps'),
                            item.get('guidance')
                        )
                elif item['type'] == 'chat':
                    await process_chat_message(
                        item['message'],
                        item['image_descriptions']
                    )

            except Exception as e:
                logger.error(f"Error processing queue item: {e}")


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


@bot.tree.interaction_check
async def global_command_toggle_check(interaction: discord.Interaction) -> bool:
    """Block slash commands that have been disabled via the dashboard."""
    from soupy_database.runtime_flags import is_command_disabled
    cmd_name = interaction.command.name if interaction.command else None
    if cmd_name and is_command_disabled(cmd_name):
        await interaction.response.send_message(
            f"the `/{cmd_name}` command is currently disabled.", ephemeral=True
        )
        return False
    return True


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


async def read_user_stats():
    # Reads the user statistics from the JSON file.
    async with user_stats_lock:
        try:
            data = json.loads(USER_STATS_FILE.read_text())
            # Convert old format to new format if necessary
            if data and not any('servers' in user_data for user_data in data.values()):
                new_data = {}
                for user_id, stats in data.items():
                    new_data[user_id] = {
                        'username': stats.get('username', 'Unknown'),
                        'servers': {
                            'global': {  # Store old stats as global stats
                                'images_generated': stats.get('images_generated', 0),
                                'chat_responses': stats.get('chat_responses', 0),
                                'mentions': stats.get('mentions', 0)
                            }
                        }
                    }
                return new_data
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
        # Commented out notification code
        '''
        # Create shutdown embed
        shutdown_embed = discord.Embed(
            description="Soupy is now going offline.",
            color=discord.Color.red(),
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
        '''
        # Initiate queue shutdown if it exists
        if hasattr(bot, 'sd_queue'):
            await bot.sd_queue.initiate_shutdown()
        
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
        elif rand_val < 0.25:  # 20% chance of Grey Sphynx Cat (0.05 to 0.25)
            terms['Character Concept'] = "Grey Sphynx Cat"
        else:  # 75% chance of random character
            terms['Character Concept'] = random.choice(CHARACTER_CONCEPTS)
    
    if ARTISTIC_RENDERING_STYLES:
        # Randomly decide how many styles to pick (1-4)
        num_styles = random.randint(1, 4)
        # Get random styles without repeats
        chosen_styles = random.sample(ARTISTIC_RENDERING_STYLES, num_styles)
        terms['Artistic Rendering Style'] = ', '.join(chosen_styles)

    # Always include a handful of SD-specific keywords if available
    if SD_KEYWORDS_LIST:
        num_sd = min(4, len(SD_KEYWORDS_LIST))
        chosen_sd = random.sample(SD_KEYWORDS_LIST, num_sd)
        terms['SD Keywords'] = ', '.join(chosen_sd)
    
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
                
                # If SD keywords exist, append a clear instruction to leverage them
                sd_hint = ""
                if 'SD Keywords' in random_terms:
                    sd_hint = f"\nFocus on these rendering/photographic cues as global style constraints: {random_terms['SD Keywords']}."
                combined_prompt = f"{RANDOMPROMPT} {style_emphasis}{sd_hint}"
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
                    temperature=float(os.getenv("RANDOM_PROMPT_TEMPERATURE", 0.8)),
                    max_tokens=325          
                )
                random_prompt = response.choices[0].message.content.strip()
                logger.info(f"🔀 Generated random prompt for {interaction.user}: {random_prompt}")

                # Capture the randomly chosen terms as a comma-separated string
                # Flatten the terms from the dictionary
                selected_terms_list = []
                for category, terms in random_terms.items():
                    # Split by comma in case there are multiple terms in a single category
                    split_terms = [term.strip() for term in terms.split(',')]
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
            selected_terms=selected_terms_str
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











async def increment_user_stat(user_id: int, stat: str, server_id: Optional[int] = None):
    """
    Increments a specific statistic for a user, optionally for a specific server.

    Args:
        user_id (int): Discord user ID
        stat (str): The statistic to increment ('images_generated', 'chat_responses', 'mentions')
        server_id (Optional[int]): The Discord server ID. If None, increments global stats.
    """
    stats = await read_user_stats()
    str_user_id = str(user_id)
    
    # Initialize user entry if it doesn't exist
    if str_user_id not in stats:
        stats[str_user_id] = {
            'username': 'Unknown',
            'servers': {
                'global': {
                    'images_generated': 0,
                    'chat_responses': 0,
                    'mentions': 0
                }
            }
        }
    
    # Update username if possible
    user = bot.get_user(user_id)
    if user:
        stats[str_user_id]['username'] = user.name
    
    # Initialize server stats if needed
    if server_id:
        str_server_id = str(server_id)
        if 'servers' not in stats[str_user_id]:
            stats[str_user_id]['servers'] = {}
        if str_server_id not in stats[str_user_id]['servers']:
            stats[str_user_id]['servers'][str_server_id] = {
                'images_generated': 0,
                'chat_responses': 0,
                'mentions': 0
            }
    
    # Increment both global and server-specific stats
    if 'global' not in stats[str_user_id]['servers']:
        stats[str_user_id]['servers']['global'] = {
            'images_generated': 0,
            'chat_responses': 0,
            'mentions': 0
        }
    
    # Increment global stat
    stats[str_user_id]['servers']['global'][stat] += 1
    
    # Increment server-specific stat if applicable
    if server_id:
        str_server_id = str(server_id)
        stats[str_user_id]['servers'][str_server_id][stat] += 1
    
    await write_user_stats(stats)
    logger.debug(f"📈 Updated '{stat}' for user ID {user_id} (server ID: {server_id})")

# Format uptime
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


# Track bot start time for uptime calculation
bot_start_time = None

# Track SD server status
sd_server_online = True  # Assume online at start

chat_functions_online = True  # Assume online at start

# Timer tracking for dashboard — stores last/next run times for all background loops
timer_state = {
    "archive_scan": {"last_run": None, "next_run": None, "interval": None, "enabled": True},
    "rag_reindex": {"last_run": None, "next_run": None, "interval": None, "enabled": True},
    "self_reflect": {"last_run": None, "next_run": None, "interval": None, "enabled": False},
    "daily_post": {"last_run": None, "next_run": None, "interval": None, "enabled": False,
                    "schedule": [], "posts_today": 0, "last_failure": None, "last_title": None, "last_channel": None},
    "bluesky": {"last_run": None, "next_run": None, "interval": None, "enabled": False,
                "schedule": [], "last_reply": None, "last_post": None, "last_repost": None,
                "last_failure": None, "replies_today": 0, "posts_today": 0, "reposts_today": 0},
}


# Verify chat functionality by performing test completion
async def check_chat_functions():
    global chat_functions_online
    try:
        test_prompt = "Hello, are you operational?"
        response = await async_chat_completion(
            model=os.getenv("LOCAL_CHAT"),
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": test_prompt}],
            temperature=float(os.getenv("HEALTHCHECK_TEMPERATURE", 0.0)),
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

# Injected before the current user turn when RAG runs; also used to tag that block in LLM log summaries.
RAG_CONTEXT_MESSAGE_SENTINEL = "Below are snippets from earlier messages in this server"

# Format message history for logging
def format_messages(messages):
    formatted = ""
    for msg in messages:
        role = msg.get('role', 'UNKNOWN').upper()
        content = msg.get('content', '').replace('\n', ' ').strip()
        formatted += f"[{role}] {content}\n"
    return formatted.strip()


def summarize_messages_for_llm_log(messages: list, preview_chars: int = 120) -> str:
    """One line per message: role, length, short preview (readable in terminals)."""
    lines: List[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "") or ""
        # Only tag as RAG if it's a user-role message that begins with the sentinel.
        # System messages also contain the sentinel string in their instructions, so we must
        # check role first to avoid mislabeling them.
        is_rag = role == "user" and (
            content.lstrip().startswith(RAG_CONTEXT_MESSAGE_SENTINEL)
            or content.lstrip().startswith("Below are excerpts retrieved from this server's saved message history")
        )
        label = "rag_context" if is_rag else role
        prev = content.replace("\n", " ").strip()
        if len(prev) > preview_chars:
            prev = prev[:preview_chars] + "…"
        lines.append(f"  [{i}] {label}: {len(content)} chars | {prev}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token count: chars / 3.5, conservative for English chat text."""
    return max(1, int(len(text or "") / 3.5))


def estimate_messages_tokens(messages: list) -> int:
    """Estimate total tokens for a list of role/content dicts (~4 tokens overhead per message)."""
    return sum(estimate_tokens(m.get("content") or "") + 4 for m in messages)


def trim_messages_to_token_budget(messages: list, budget_tokens: int) -> list:
    """Drop oldest messages until the list fits within budget_tokens. Always keeps at least one."""
    if not messages or estimate_messages_tokens(messages) <= budget_tokens:
        return messages
    trimmed = list(messages)
    while len(trimmed) > 1 and estimate_messages_tokens(trimmed) > budget_tokens:
        trimmed.pop(0)
    return trimmed

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
    """Wraps the OpenAI chat completion in an async context and cleans the response."""
    response = await asyncio.to_thread(client.chat.completions.create, *args, **kwargs)
    # Log token usage if available
    if response.usage:
        logger.info(
            "📊 Tokens — prompt: %d | completion: %d | total: %d",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )
    return response

def clean_response(text: str) -> str:
    text = text.strip()
    while (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Strip em dashes — the bot should never use them
    text = text.replace("—", ",").replace("–", ",")

    # --- RAG data leak sanitizer ---
    # The LLM sometimes regurgitates raw RAG context (timestamps, separators, message metadata).
    # Strip all of it aggressively.

    # Remove any line starting with "---" (RAG separator artifacts with or without label text)
    text = re.sub(r"^---\s*[^\n]*$", "", text, flags=re.MULTILINE)
    # Also catch "---" mid-line as sentence starters (LLM sometimes puts them after a period)
    text = re.sub(r"\s---\s", " ", text)

    # Remove raw timestamps like "cool: 2026-08-15 07:40:39" or "[2026-03-28 19:30:53]"
    text = re.sub(r"^\s*\w[\w\s]{0,30}:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]?", "", text)

    # Remove message_id references
    text = re.sub(r"\[?message_id=\d+[^\]]*\]?", "", text)

    # Remove channel references like "#channel-name, 2026-03-28"
    text = re.sub(r"#[\w-]+,\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", "", text)

    # Remove embed metadata like "[Embed Title: ...]", "[Embed Description: ...]", "[Embed URL: ...]"
    text = re.sub(r"\[Embed\s+(?:Title|Description|URL|Author)[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Remove "[image: ...]" vision descriptions leaked from context
    text = re.sub(r"\[image:[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Remove "[link: ...]" URL content descriptions leaked from context
    text = re.sub(r"\[link:[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Remove "--- Member sketches" or "--- end sketches ---" or "--- related conversation" headers
    text = re.sub(r"---\s*(?:Member sketches|end sketches|related conversation|your own memories)[^\n]*", "", text, flags=re.IGNORECASE)

    # Remove self-knowledge section headers leaked: "[self-knowledge: ...]"
    text = re.sub(r"\[self-knowledge:[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Collapse multiple blank lines left by removals
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Collapse multiple spaces left by inline removals
    text = re.sub(r"  +", " ", text)

    # Final cleanup
    text = text.strip()

    # --- Repetition loop / runaway length guard ---
    # The BEHAVIOUR prompt caps responses at ~80 words. If the LLM generates far more
    # than that, it's stuck in a repetition loop. Truncate to a sane maximum and cut
    # at the last sentence boundary.
    try:
        max_words = int(os.getenv("CHAT_MAX_RESPONSE_WORDS", "80"))
    except ValueError:
        max_words = 120
    words = text.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        # Cut at the last sentence-ending punctuation
        for end_char in [". ", "! ", "? "]:
            last = truncated.rfind(end_char)
            if last > len(truncated) // 2:
                truncated = truncated[:last + 1]
                break
        text = truncated.strip()
        logger.warning("clean_response: truncated runaway response from %d to %d words", len(words), len(text.split()))

    return text

async def generate_parallel_candidates(messages: list, model: str, temperature: float, max_tokens: int, num_candidates: int = 2) -> list:
    """
    Generate multiple candidate replies concurrently using the chat model.
    Uses slightly varied temperatures for diversity.
    Returns a list of candidate strings (length <= num_candidates).
    """
    # Get temperature variation settings from environment variables
    # Handle empty strings by using defaults
    def get_float_env(key: str, default: float) -> float:
        value = os.getenv(key, "")
        return float(value) if value and value.strip() else default
    
    temp_variation_down = get_float_env("CHAT_TEMP_VARIATION_DOWN", 0.15)  # How much to subtract for lower temp
    temp_variation_up = get_float_env("CHAT_TEMP_VARIATION_UP", 0.15)      # How much to add for higher temp
    temp_variation_up2 = get_float_env("CHAT_TEMP_VARIATION_UP2", 0.25)   # Additional variation for most creative
    temp_min = get_float_env("CHAT_TEMP_MIN", 0.3)                        # Minimum temperature
    temp_max = get_float_env("CHAT_TEMP_MAX", 1.0)                        # Maximum temperature
    frequency_penalty = get_float_env("CHAT_FREQUENCY_PENALTY", 0.6)
    presence_penalty = get_float_env("CHAT_PRESENCE_PENALTY", 0.3)
    
    # Create temperature variations for diversity (capped at temp_max to avoid hallucinations)
    # Base temp, slightly lower (more focused), slightly higher (more creative), highest (but capped)
    temp_variations = [
        temperature,                                    # Candidate A: base temperature
        max(temp_min, temperature - temp_variation_down),  # Candidate B: slightly more focused
        min(temp_max, temperature + temp_variation_up),  # Candidate C: slightly more creative  
        min(temp_max, temperature + temp_variation_up2),  # Candidate D: most creative (capped at temp_max)
    ]
    

    async def generate_with_retry(candidate_idx: int, retries: int = 2, backoff_seconds: float = 2.0):
        attempt = 0
        # Use temperature variation for this candidate
        temp = temp_variations[candidate_idx % len(temp_variations)]
        while True:
            try:
                return await async_chat_completion(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
            except Exception as e:
                err_text = str(e)
                logger.error(f"❌ Candidate #{candidate_idx} (temp={temp:.2f}) attempt {attempt+1} failed: {format_error_message(e)}")
                # Retry on model_not_found or HTTP 404 from gateway warm-up
                should_retry = (
                    'model_not_found' in err_text or '404' in err_text or 'Not Found' in err_text
                ) and attempt < retries
                if not should_retry:
                    raise
                await asyncio.sleep(backoff_seconds * (attempt + 1))
                attempt += 1

    generation_tasks = [
        asyncio.create_task(generate_with_retry(i))
        for i in range(max(1, num_candidates))
    ]

    results = await asyncio.gather(*generation_tasks, return_exceptions=True)

    candidate_replies = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"❌ Error generating candidate #{idx+1}: {format_error_message(result)}")
            continue
        try:
            msg = result.choices[0].message
            candidate_text = msg.content or ""
            if not candidate_text.strip():
                logger.warning(f"⚠ Candidate #{idx+1} generated tokens but content is empty (thinking model may have used all tokens on reasoning)")
                continue
            candidate_replies.append(candidate_text)
            temp_used = temp_variations[idx % len(temp_variations)]
            logger.debug(f"✅ Candidate #{idx+1} generated (temp={temp_used:.2f})")
        except Exception as parse_error:
            logger.error(f"❌ Failed to parse candidate #{idx+1} response: {format_error_message(parse_error)}")

    return candidate_replies

async def judge_best_of_candidates(messages_context: list, candidates: list, model: str) -> int:
    """
    Ask the model to choose the best candidate among the provided list.
    Prioritizes relevance to the most recent message.
    Returns the index of the selected candidate (defaults to 0 on parse errors).
    Supports up to 4 candidates (A-D).
    """
    if not candidates:
        return 0

    # Get more context for better judgment (last 12 messages)
    recent_context = messages_context[-12:] if len(messages_context) > 12 else messages_context
    formatted_context = format_messages(recent_context)
    
    # Extract the most recent user message to emphasize what we're responding to
    last_user_msg = None
    for msg in reversed(recent_context):
        if msg.get('role') == 'user':
            last_user_msg = msg.get('content', '').strip()
            break

    labels = ["A", "B", "C", "D"][:len(candidates)]
    allowed = ", ".join(labels)

    system_prompt = (
        "You are an expert judge evaluating which response best continues a conversation. "
        "Your job is to pick the candidate that fits NATURALLY into the conversation flow.\n\n"
        "Evaluation criteria (in priority order):\n"
        "1. RELEVANCE - Directly addresses what the user just said (not off-topic or tangential)\n"
        "2. CONVERSATIONAL FIT - Flows naturally from the recent back-and-forth (not a non-sequitur)\n"
        "3. CONTEXTUAL AWARENESS - Shows understanding of the ongoing topic and dynamic\n"
        "4. TONE MATCHING - Matches the energy and style of the conversation\n\n"
        "IMPORTANT: You're evaluating how each candidate fits INTO the conversation, not which sounds best in isolation.\n"
        "A witty response that ignores context is worse than a simple response that's on-topic.\n\n"
        f"Reply with ONLY one letter: {allowed}"
    )

    # Build the user content listing the context and each candidate
    candidate_sections = []
    for label, text in zip(labels, candidates):
        candidate_sections.append(f"Candidate {label}:\n{text}")
    candidates_block = "\n\n".join(candidate_sections)

    # Build prompt with emphasis on conversation flow
    user_prompt = "Read this conversation and see how it's flowing:\n\n"
    user_prompt += f"CONVERSATION:\n{formatted_context}\n\n"
    if last_user_msg:
        user_prompt += f"MOST RECENT MESSAGE to respond to:\n{last_user_msg}\n\n"
    user_prompt += f"Now evaluate how well EACH candidate continues this conversation:\n\n{candidates_block}\n\n"
    user_prompt += f"Which candidate best fits the conversation flow and addresses what was just said? Reply with only: {allowed}"

    try:
        response = await async_chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Very low but not 0 to allow slight variation
            max_tokens=5
        )
        raw_choice = response.choices[0].message.content.strip().upper()
        # Try to extract the letter from the response
        for idx, label in enumerate(labels):
            if label in raw_choice:
                logger.debug(f"Judge selected '{label}' from response: '{raw_choice}'")
                return idx
        logger.warning(f"Judge returned unexpected response: '{raw_choice}', defaulting to candidate A")
        return 0
    except Exception as e:
        logger.error(f"❌ Error during judge selection: {format_error_message(e)}")
        return 0

def enhance_img2img_prompt(prompt: str, strength: float) -> str:
    """Enhance img2img prompts based on strength for better results."""
    # Base quality enhancers
    quality_enhancers = [
        "high quality", "detailed", "sharp focus", "professional photography"
    ]
    
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


# Fetch "limit" recent messages from the channel, including content from any images.
async def fetch_recent_messages(channel, limit=int(os.getenv("RECENT_MESSAGE_LIMIT", 25)), current_message_id=None):
    """
    Fetches recent messages from the channel, focusing on the most recent conversation.
    Limited context to keep responses relevant and focused.
    """
    message_history = []
    seen_messages = set()
    current_topic_messages = []  # Track messages in current topic
    background_messages = []     # Track older context messages
    
    # Create a single session for all URL requests
    async with aiohttp.ClientSession() as session:
        async for msg in channel.history(limit=limit, oldest_first=False):
            # Skip command messages and bot's image generation messages
            if msg.content.startswith("!") or (msg.author == bot.user and "Generated Image" in msg.content):
                continue
                
            # Skip current message if provided
            if current_message_id and msg.id == current_message_id:
                continue
            
            # Create base message content (no URL processing for historical messages)
            message_content = msg.content
            
            # Check if this message has image descriptions stored
            image_desc_for_msg = message_image_descriptions.get(msg.id, [])
            
            # Create unique message identifier
            message_key = f"{msg.author.display_name}:{message_content}"
            if message_key in seen_messages:
                continue
                
            seen_messages.add(message_key)
            
            # Format message with role assignment and author
            # Bot messages: role="assistant", no name prefix (role already indicates it's the bot)
            # User messages: role="user", include nickname/display name prefix (multiple users can speak)
            role = "assistant" if msg.author == bot.user else "user"
            if role == "assistant":
                # Don't include bot name - the "assistant" role makes it clear
                formatted_content = message_content
            else:
                # Include user's display name (nickname) to distinguish between multiple users
                formatted_content = f"{msg.author.display_name}: {message_content}"
            
            # If there are image descriptions for this message, prepend them
            if image_desc_for_msg:
                image_context = "\n".join(image_desc_for_msg)
                formatted_content = f"{image_context}\n{formatted_content}" if formatted_content else image_context
            
            formatted_message = {"role": role, "content": formatted_content}
            
            # Prioritize recent messages much more heavily
            if len(current_topic_messages) < 8:  # Keep most recent 8 messages for current topic
                current_topic_messages.append(formatted_message)
            elif len(background_messages) < 4:  # Only keep 4 background messages maximum
                background_messages.append(formatted_message)
    
    # Clean up old URL cache entries
    current_time = time.time()
    expired_urls = [url for url, (_, timestamp) in url_cache.items() 
                   if current_time - timestamp > URL_CACHE_TTL]
    for url in expired_urls:
        del url_cache[url]
    
    # Clean up old image descriptions to prevent memory bloat
    # Keep only descriptions for message IDs that were actually seen in recent history
    seen_message_ids = {msg.id async for msg in channel.history(limit=limit*2, oldest_first=False)}
    message_ids_to_remove = [msg_id for msg_id in message_image_descriptions.keys() 
                             if msg_id not in seen_message_ids]
    for msg_id in message_ids_to_remove:
        del message_image_descriptions[msg_id]
    
    # Put minimal background context first, then focus heavily on recent messages
    message_history = list(reversed(background_messages)) + list(reversed(current_topic_messages))
    return message_history  # This would have oldest → newest order


# ---------------------------------------------
# Vision: Process images using LM Studio vision model
# ---------------------------------------------

async def process_image_attachment(attachment, message):
    """
    Process an image attachment using LM Studio's vision model via direct HTTP.
    Returns the description if successful, None otherwise.
    Also archives the image and logs to activity.
    """
    if os.getenv("ENABLE_VISION", "false").lower() != "true":
        return None
    
    if not any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
        return None

    try:
        # Download the image
        original_image_data = None  # Keep original for archiving
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status != 200:
                    return None
                image_data = await resp.read()
                original_image_data = image_data  # Save original

        # Transcode to JPEG for maximum compatibility
        try:
            from io import BytesIO
            with Image.open(BytesIO(image_data)) as im:
                if im.mode in ("P", "RGBA"):
                    im = im.convert("RGB")
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=92, optimize=True)
                image_bytes_for_api = buf.getvalue()
                image_subtype = "jpeg"
        except Exception:
            # Fallback: use original bytes and infer subtype from filename
            image_bytes_for_api = image_data
            image_subtype = "jpeg"
            name = attachment.filename.lower()
            if name.endswith(".png"):
                image_subtype = "png"
            elif name.endswith(".gif"):
                image_subtype = "gif"
            elif name.endswith(".webp"):
                image_subtype = "webp"

        # Encode to base64 (final payload)
        encoded_image = base64.b64encode(image_bytes_for_api).decode("utf-8")

        # Build endpoint from OPENAI_BASE_URL
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        endpoint = f"{base}/chat/completions"

        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        model_name = os.getenv("VISION_MODEL") or os.getenv("LOCAL_CHAT")
        prompt_text = os.getenv("VISION_PROMPT", "What is in this image? Describe it concisely.")
        max_tokens = int(os.getenv("VISION_MAX_TOKENS", "300"))
        temperature = float(os.getenv("VISION_TEMPERATURE", "0.7"))

        # First try: raw base64 ONLY (no mime_type), image first then text
        payload_raw_simple = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encoded_image}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload_raw_simple) as r0:
                if r0.status == 200:
                    out = await r0.json()
                    description = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                    if description:
                        logger.info(f"🖼️ Image from {message.author.display_name}: {description}")
                        
                        # Archive the image and log to activity
                        guild_id = message.guild.id if message.guild else None
                        channel_id = message.channel.id if message.channel else None
                        
                        filename = archive_vision_image(
                            original_image_data,
                            description=description,
                            user_id=message.author.id,
                            username=str(message.author),
                            guild_id=guild_id,
                            channel_id=channel_id,
                            original_url=attachment.url
                        )
                        
                        if filename:
                            archive_sent_message(
                                f"🖼️ Vision: {description}",
                                user_id=message.author.id,
                                username=str(message.author),
                                guild_id=guild_id,
                                channel_id=channel_id,
                                image_filename=filename,
                                event_type="vision"
                            )
                        
                        return description

            # Second try: OpenAI-style input_image with data URI
            payload_input_image = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": {"url": f"data:image/{image_subtype};base64,{encoded_image}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            async with session.post(endpoint, headers=headers, json=payload_input_image) as r1:
                if r1.status == 200:
                    out = await r1.json()
                    description = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                    if description:
                        logger.info(f"🖼️ Image from {message.author.display_name}: {description}")
                        
                        # Archive the image and log to activity
                        guild_id = message.guild.id if message.guild else None
                        channel_id = message.channel.id if message.channel else None
                        
                        filename = archive_vision_image(
                            original_image_data,
                            description=description,
                            user_id=message.author.id,
                            username=str(message.author),
                            guild_id=guild_id,
                            channel_id=channel_id,
                            original_url=attachment.url
                        )
                        
                        if filename:
                            archive_sent_message(
                                f"🖼️ Vision: {description}",
                                user_id=message.author.id,
                                username=str(message.author),
                                guild_id=guild_id,
                                channel_id=channel_id,
                                image_filename=filename,
                                event_type="vision"
                            )
                        
                        return description

                # Third try: data URI (image_url) then raw with mime_type
                payload_data_uri = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/{image_subtype};base64,{encoded_image}"}},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                async with session.post(endpoint, headers=headers, json=payload_data_uri) as r2:
                    if r2.status == 200:
                        out = await r2.json()
                        description = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                        if description:
                            logger.info(f"🖼️ Image from {message.author.display_name}: {description}")
                            
                            # Archive the image and log to activity
                            guild_id = message.guild.id if message.guild else None
                            channel_id = message.channel.id if message.channel else None
                            
                            filename = archive_vision_image(
                                original_image_data,
                                description=description,
                                user_id=message.author.id,
                                username=str(message.author),
                                guild_id=guild_id,
                                channel_id=channel_id,
                                original_url=attachment.url
                            )
                            
                            if filename:
                                archive_sent_message(
                                    f"🖼️ Vision: {description}",
                                    user_id=message.author.id,
                                    username=str(message.author),
                                    guild_id=guild_id,
                                    channel_id=channel_id,
                                    image_filename=filename,
                                    event_type="vision"
                                )
                            
                            return description

                    payload_raw_b64 = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": encoded_image, "mime_type": f"image/{image_subtype}"}},
                                    {"type": "text", "text": prompt_text},
                                ],
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    async with session.post(endpoint, headers=headers, json=payload_raw_b64) as r3:
                        if r3.status == 200:
                            out = await r3.json()
                            description = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                            if description:
                                logger.info(f"🖼️ Image from {message.author.display_name}: {description}")
                                
                                # Archive the image and log to activity
                                guild_id = message.guild.id if message.guild else None
                                channel_id = message.channel.id if message.channel else None
                                
                                filename = archive_vision_image(
                                    original_image_data,
                                    description=description,
                                    user_id=message.author.id,
                                    username=str(message.author),
                                    guild_id=guild_id,
                                    channel_id=channel_id,
                                    original_url=attachment.url
                                )
                                
                                if filename:
                                    archive_sent_message(
                                        f"🖼️ Vision: {description}",
                                        user_id=message.author.id,
                                        username=str(message.author),
                                        guild_id=guild_id,
                                        channel_id=channel_id,
                                        image_filename=filename,
                                        event_type="vision"
                                    )
                                
                                return description

                        # Compact debug only
                        try:
                            e0 = await r0.text()
                        except Exception:
                            e0 = str(r0.status)
                        try:
                            e1 = await r1.text()
                        except Exception:
                            e1 = str(r1.status)
                        try:
                            e2 = await r2.text()
                        except Exception:
                            e2 = str(r2.status)
                        try:
                            e3 = await r3.text()
                        except Exception:
                            e3 = str(r3.status)
                        logger.debug(f"Vision processing failed: HTTP {r0.status}/{r1.status}/{r2.status}/{r3.status} - {e0[:70]} | {e1[:70]} | {e2[:70]} | {e3[:70]}")
                        return None

    except Exception as e:
        logger.debug(f"Vision processing failed: {str(e)[:120]}")
        return None


# ---------------------------------------------
# New Slash Commands: /img2img and /inpaint
# ---------------------------------------------

@bot.tree.command(name="img2img", description="Stable Diffusion 3.5: transform an image with a prompt.")
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
    steps: app_commands.Range[int, 1, 50] = 20,           # Optimized for speed/quality
    guidance: app_commands.Range[float, 0.0, 20.0] = 7.5, # Better for SD 3.5 Medium
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
        await interaction.response.send_message("❌ Attach an image (or have one recently in the channel) and try again.", ephemeral=True)
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
        form.add_field("negative_prompt", os.getenv("SD_NEGATIVE_PROMPT", "") or "")
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
            form2.add_field("negative_prompt", os.getenv("SD_NEGATIVE_PROMPT", "") or "")
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


@bot.tree.command(name="inpaint", description="Stable Diffusion 3.5: inpaint an image with a mask and prompt.")
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
    steps: app_commands.Range[int, 1, 50] = 20,          # Optimized
    guidance: app_commands.Range[float, 0.0, 20.0] = 7.5, # Better for SD 3.5
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
        await interaction.response.send_message("❌ Please upload an image and a mask (white=edit, black=keep) and try again.", ephemeral=True)
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
        form.add_field("negative_prompt", os.getenv("SD_NEGATIVE_PROMPT", "") or "")
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
            form2.add_field("negative_prompt", os.getenv("SD_NEGATIVE_PROMPT", "") or "")
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


@bot.tree.command(name="outpaint", description="Stable Diffusion 3.5: extend an image by 25% in specified directions.")
@app_commands.describe(
    prompt="What should extend into the new areas (e.g., 'extend the landscape, continue the mountains')",
    direction="Which direction(s) to extend the image",
    strength="How aggressively to change the extended areas (0.0-1.0)",
    steps="Inference steps (20-30 recommended)",
    guidance="CFG guidance scale (7.5-12 recommended for SD 3.5)",
)
@app_commands.choices(direction=[
    app_commands.Choice(name="Horizontal (left + right)", value="horizontal"),
    app_commands.Choice(name="Vertical (top + bottom)", value="vertical"),
    app_commands.Choice(name="Both (all directions)", value="both"),
])
async def outpaint_cmd(
    interaction: discord.Interaction,
    prompt: str,
    direction: app_commands.Choice[str],
    strength: app_commands.Range[float, 0.0, 1.0] = 0.8,  # Higher for outpaint
    steps: app_commands.Range[int, 1, 50] = 20,          # Optimized
    guidance: app_commands.Range[float, 0.0, 20.0] = 7.5, # Better for SD 3.5
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

    logger.info(f"🖼️ Slash Command 'outpaint' invoked by {interaction.user} with prompt: '{prompt}', direction: '{direction.value}', strength: '{strength}'")
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
    
    await bot.sd_queue.put({
        'type': 'outpaint',
        'interaction': interaction,
        'prompt': prompt,
        'direction': direction_value,
        'width': width,
        'height': height,
        'seed': -1,  # Random seed
        'strength': strength,
        'steps': steps,
        'guidance': guidance,
    })
    logger.info(f"🖼️ Queued outpaint generation for {interaction.user}: prompt='{prompt}', direction='{direction_value}', strength='{strength}'")


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


@bot.command(name='synccommands', help='Syncs slash commands to Discord (Owner only)')
async def sync_commands(ctx):
    # Check if the user is in OWNER_IDS
    if ctx.author.id not in OWNER_IDS:
        await ctx.send("❌ You don't have permission to use this command.", ephemeral=True)
        logger.warning(f"Unauthorized attempt to sync commands by {ctx.author}")
        return
    
    await ctx.send("🔄 Syncing slash commands...", ephemeral=True)
    
    try:
        # List all registered commands for debugging
        registered_commands = [cmd.name for cmd in bot.tree.get_commands()]
        logger.info(f"Registered commands before sync: {registered_commands}")
        
        # Sync to guild first (instant) if GUILD_ID is set, then globally
        guild_id_str = os.getenv("GUILD_ID")
        if guild_id_str:
            try:
                gid = int(guild_id_str)
                guild_obj = discord.Object(id=gid)
                bot.tree.copy_global_to(guild=guild_obj)
                synced = await bot.tree.sync(guild=guild_obj)
                synced_names = [cmd.name for cmd in synced]
                logger.info(f"Synced commands to guild {gid}: {synced_names}")
                response_msg = f"✅ Synced {len(synced)} commands to guild (instant).\n"
                response_msg += f"Commands: {', '.join(synced_names)}"
                await ctx.send(response_msg, ephemeral=True)
                logger.info(f"Commands synced to guild {gid} by {ctx.author}")
            except (ValueError, AttributeError) as e:
                logger.warning(f"Guild sync failed ({e}), falling back to global sync")
                synced = await bot.tree.sync()
                synced_names = [cmd.name for cmd in synced]
                response_msg = f"✅ Synced {len(synced)} commands globally (may take up to 1 hour).\n"
                response_msg += f"Commands: {', '.join(synced_names)}"
                await ctx.send(response_msg, ephemeral=True)
        else:
            synced = await bot.tree.sync()
            synced_names = [cmd.name for cmd in synced]
            logger.info(f"Synced commands: {synced_names}")
            response_msg = f"✅ Synced {len(synced)} commands globally (may take up to 1 hour).\n"
            response_msg += f"Commands: {', '.join(synced_names)}"
            await ctx.send(response_msg, ephemeral=True)
            logger.info(f"Commands synced globally by {ctx.author}")
        
    except Exception as e:
        error_message = f"❌ Error syncing commands: {str(e)}"
        await ctx.send(error_message, ephemeral=True)
        logger.error(f"Error syncing commands by {ctx.author}: {str(e)}")


@bot.tree.command(name="soupyself", description="View or manage Soupy's self-knowledge document (owner only).")
@app_commands.describe(
    action="What to do: view (default), core (view core summary), archive (view pruned entries), reflect (force reflection now), or reset"
)
@app_commands.choices(action=[
    app_commands.Choice(name="view", value="view"),
    app_commands.Choice(name="core", value="core"),
    app_commands.Choice(name="archive", value="archive"),
    app_commands.Choice(name="reflect", value="reflect"),
    app_commands.Choice(name="reset", value="reset"),
])
async def soupyself_command(
    interaction: discord.Interaction,
    action: Optional[app_commands.Choice[str]] = None,
):
    if interaction.user.id not in OWNER_IDS:
        await interaction.response.send_message("not for you.", ephemeral=True)
        return

    act = (action.value if action else "view")
    guild_id = interaction.guild_id

    if not guild_id:
        await interaction.response.send_message("must be used in a server.", ephemeral=True)
        return

    if not is_self_md_enabled():
        await interaction.response.send_message(
            "self-context is disabled. set `SELF_MD_ENABLED=true` in .env-stable and restart.",
            ephemeral=True,
        )
        return

    if act == "view":
        from soupy_database.self_context import load_self_archive
        content = load_self_md(guild_id)
        pending = pending_interaction_count(guild_id)
        archive = load_self_archive(guild_id)
        if not content:
            await interaction.response.send_message(
                f"no self-document yet for this server. {pending} interaction(s) pending reflection.",
                ephemeral=True,
            )
        else:
            header = (
                f"**SELF.MD** (full={len(content)} chars, archive={len(archive)} chars, "
                f"{pending} interactions pending)\n"
            )
            # Discord has a 2000 char limit per message — split into pages
            max_per_msg = 1900
            first_max = max_per_msg - len(header)
            if len(content) <= first_max:
                await interaction.response.send_message(header + content, ephemeral=True)
            else:
                await interaction.response.send_message(header + content[:first_max], ephemeral=True)
                remaining = content[first_max:]
                while remaining:
                    chunk = remaining[:max_per_msg]
                    remaining = remaining[max_per_msg:]
                    await interaction.followup.send(chunk, ephemeral=True)

    elif act == "core":
        from soupy_database.self_context import load_self_core
        core = load_self_core(guild_id)
        if not core:
            await interaction.response.send_message(
                "no core summary yet. run `/soupyself reflect` first.", ephemeral=True,
            )
        else:
            header = f"**SELF.MD CORE** ({len(core)} chars — always in system prompt)\n"
            max_per_msg = 1900
            first_max = max_per_msg - len(header)
            if len(core) <= first_max:
                await interaction.response.send_message(header + core, ephemeral=True)
            else:
                await interaction.response.send_message(header + core[:first_max], ephemeral=True)
                remaining = core[first_max:]
                while remaining:
                    chunk = remaining[:max_per_msg]
                    remaining = remaining[max_per_msg:]
                    await interaction.followup.send(chunk, ephemeral=True)

    elif act == "archive":
        from soupy_database.self_context import load_self_archive
        archive = load_self_archive(guild_id)
        if not archive:
            await interaction.response.send_message("no archive yet.", ephemeral=True)
        else:
            header = f"**SELF.MD ARCHIVE** ({len(archive)} chars — pruned entries, searchable via RAG)\n"
            max_per_msg = 1900
            first_max = max_per_msg - len(header)
            if len(archive) <= first_max:
                await interaction.response.send_message(header + archive, ephemeral=True)
            else:
                await interaction.response.send_message(header + archive[:first_max], ephemeral=True)
                remaining = archive[first_max:]
                while remaining:
                    chunk = remaining[:max_per_msg]
                    remaining = remaining[max_per_msg:]
                    await interaction.followup.send(chunk, ephemeral=True)

    elif act == "reflect":
        await interaction.response.defer(ephemeral=True, thinking=True)
        pending = pending_interaction_count(guild_id)
        logger.info(
            "🪞 /soupyself reflect invoked by %s for guild %s (%d pending interaction(s))",
            interaction.user, guild_id, pending,
        )
        try:
            from soupy_database.rag import embed_texts_lm_studio
            async with aiohttp.ClientSession() as embed_session:
                result = await self_md_reflect(
                    guild_id=guild_id,
                    llm_func=async_chat_completion,
                    model=os.getenv("LOCAL_CHAT"),
                    embed_func=embed_texts_lm_studio,
                    embed_session=embed_session,
                )
            from soupy_database.self_context import load_self_core
            core = load_self_core(guild_id)
            core_len = len(core) if core else 0
            await interaction.followup.send(
                f"reflection complete ({pending} interactions processed).\n"
                f"full doc: {len(result)} chars | core: {core_len} chars",
                ephemeral=True,
            )
        except Exception as exc:
            await interaction.followup.send(f"reflection failed: {exc}", ephemeral=True)

    elif act == "reset":
        from soupy_database.self_context import save_self_md, save_self_core
        save_self_md(guild_id, "")
        save_self_core(guild_id, "")
        await interaction.response.send_message("self-document and core cleared.", ephemeral=True)
        logger.info("SELF.MD reset by %s for guild %s", interaction.user, guild_id)


@bot.tree.command(name="helpsoupy", description="Displays all available commands.")
async def help_command(interaction: discord.Interaction):
    """
    Sends an embedded help message listing all available slash and prefix commands.
    Excludes owner-only commands.
    """
    logger.info(f"📚 Command 'helpsoupy' invoked by {interaction.user}")

    # Owner-only commands to exclude
    owner_only_commands = {"soupyscan", "reloadenv", "synccommands", "soupyself"}
    
    # Commands to exclude from help (deprecated or not meant for users)
    excluded_commands = {"img2img", "inpaint", "outpaint", "testurl", "soupyscan"}

    # Manual mapping of command usage examples with improved descriptions
    command_usage_examples = {
        "sd": "`description` [size] [seed]",
        "soupysearch": "`query`",
        "soupyimage": "`query`",
        "8ball": "`question`",
        "9ball": "`question`",
        "whattime": "`location`",
        "weather": "`location`",
        "stats": "",
        "status": "",
    }
    
    # Improved command descriptions
    command_descriptions = {
        "sd": "Generate an image using Stable Diffusion. Provide a description of what you want to see, optionally choose a size and seed.",
        "soupysearch": "Search the web using DuckDuckGo and get a comprehensive answer with citations from multiple sources.",
        "soupyimage": "Search for images using DuckDuckGo and get a random image from the top results.",
        "8ball": "Ask the Magic 8-Ball a yes/no question and receive a classic 8-ball response.",
        "9ball": "Ask the mystical 9-ball a question and receive a custom response powered by AI.",
        "whattime": "Get the current time in any city or location around the world.",
        "weather": "Get the current weather conditions for any city or location around the world.",
        "stats": "View server statistics showing the top users for images generated and chat responses. (Admin only)",
        "status": "Check the current status of the bot, Stable Diffusion server, and chat functions.",
    }

    # Collect slash commands (excluding owner-only, excluded commands, and helpsoupy itself)
    slash_commands_list = []
    for cmd in bot.tree.get_commands():
        if cmd.name in owner_only_commands or cmd.name in excluded_commands or cmd.name == "helpsoupy":
            continue
        slash_commands_list.append(cmd)

    # Collect prefix commands (excluding owner-only)
    prefix_commands_list = []
    for cmd in bot.commands:
        if not isinstance(cmd, commands.Group) and cmd.name not in owner_only_commands:
            prefix_commands_list.append(cmd)

    # Build command descriptions
    commands_text = []

    # Add slash commands
    if slash_commands_list:
        commands_text.append("**🔹 Slash Commands:**\n")
        for cmd in sorted(slash_commands_list, key=lambda x: x.name):
            cmd_name = f"/{cmd.name}"
            
            # Use improved description if available, otherwise fall back to command description
            cmd_desc = command_descriptions.get(cmd.name, cmd.description or "No description provided.")
            
            # Get usage from manual mapping
            usage = command_usage_examples.get(cmd.name, "")
            
            if usage:
                commands_text.append(f"**{cmd_name}** {usage}")
            else:
                commands_text.append(f"**{cmd_name}**")
            commands_text.append(f"  → {cmd_desc}\n")
        
        commands_text.append("\n")

    # Add prefix commands
    if prefix_commands_list:
        commands_text.append("**🔸 Prefix Commands:**\n")
        for cmd in sorted(prefix_commands_list, key=lambda x: x.name):
            cmd_name = f"!{cmd.name}"
            cmd_help = cmd.help or "No description provided."
            # Remove "(Owner only)" from help text if present
            cmd_help = cmd_help.replace("(Owner only)", "").strip()
            commands_text.append(f"**{cmd_name}**")
            commands_text.append(f"  → {cmd_help}\n")

    if not commands_text:
        commands_text.append("No commands available.")

    # Split into multiple embeds if needed (Discord embed description limit is 4096 chars)
    full_text = "\n".join(commands_text)
    
    # Create embeds, splitting if necessary
    embeds = []
    
    if len(full_text) <= 4000:  # Leave some buffer
        embed = discord.Embed(
            title="📚 Soupy Help Menu",
            description=f"Here's a list of all available commands:\n\n{full_text}",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        embeds.append(embed)
    else:
        # Split into multiple embeds
        chunks = []
        current_chunk = "Here's a list of all available commands:\n\n"
        
        for line in commands_text:
            if len(current_chunk) + len(line) + 1 > 4000:
                chunks.append(current_chunk)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        if current_chunk:
            chunks.append(current_chunk)
        
        for i, chunk in enumerate(chunks):
            embed = discord.Embed(
                title=f"📚 Soupy Help Menu" + (f" (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else ""),
                description=chunk,
                color=discord.Color.blue(),
                timestamp=datetime.utcnow()
            )
            embeds.append(embed)
    
    # Add footer to last embed
    embeds[-1].set_footer(
        text="Use the commands as shown above to interact with me!",
        icon_url=bot.user.avatar.url if bot.user.avatar else None
    )

    # Send the embed(s) as an ephemeral message
    if len(embeds) == 1:
        await interaction.response.send_message(embed=embeds[0], ephemeral=True)
    else:
        await interaction.response.send_message(embed=embeds[0], ephemeral=True)
        for embed in embeds[1:]:
            await interaction.followup.send(embed=embed, ephemeral=True)
    
    logger.info(f"📚 Sent help menu to {interaction.user}")


# Setup the scan command - all database functionality is in soupy_database module
setup_scan_command(bot, OWNER_IDS)





@bot.tree.command(name="soupystats", description="Server and bot statistics dashboard.")
async def stats_command(interaction: discord.Interaction):
    logger.info(f"Command 'soupystats' invoked by {interaction.user} in server {interaction.guild_id}")
    await interaction.response.defer(ephemeral=True)

    try:
        guild_id = interaction.guild_id
        server_id = str(guild_id)
        import sqlite3

        from soupy_database.database import get_stats as get_db_stats, get_db_path
        db_stats = get_db_stats(guild_id)

        embed = discord.Embed(
            title=f"📊 {interaction.guild.name}",
            color=discord.Color.purple(),
        )

        # ── Overview row ─────────────────────────────────────────
        uptime_str = format_uptime(datetime.utcnow() - bot_start_time) if bot_start_time else "unknown"
        overview_parts = [
            f"⏱ Uptime: **{uptime_str}**",
            f"👥 Members: **{interaction.guild.member_count or '?'}**",
        ]
        if db_stats.get("exists"):
            overview_parts.append(f"💾 Archived: **{db_stats['total_messages']:,}** msgs")
        try:
            db_path = get_db_path(guild_id)
            if os.path.exists(db_path):
                _pconn = sqlite3.connect(db_path, check_same_thread=False)
                _pconn.row_factory = sqlite3.Row
                _pcur = _pconn.cursor()
                _pcur.execute("SELECT COUNT(*) AS c FROM user_profile_summaries")
                prof_count = int(_pcur.fetchone()["c"])
                _pconn.close()
                overview_parts.append(f"👤 Profiles: **{prof_count}**")
        except Exception:
            pass
        embed.add_field(name="Server", value=" · ".join(overview_parts), inline=False)

        # ── Most Active Users ────────────────────────────────────
        if db_stats.get("exists"):
            try:
                _conn = sqlite3.connect(get_db_path(guild_id), check_same_thread=False)
                _conn.row_factory = sqlite3.Row
                _cur = _conn.cursor()

                _cur.execute(
                    """
                    SELECT MAX(COALESCE(NULLIF(TRIM(nickname),''), NULLIF(TRIM(username),''))) AS name,
                           COUNT(*) AS cnt
                    FROM messages GROUP BY user_id ORDER BY cnt DESC LIMIT 10
                    """
                )
                top_senders = _cur.fetchall()
                if top_senders:
                    embed.add_field(
                        name="💬 Most Active Users",
                        value="\n".join(
                            f"{i+1}. **{r['name'] or '?'}** — {r['cnt']:,}"
                            for i, r in enumerate(top_senders)
                        ),
                        inline=True,
                    )

                # ── Most Active Channels ─────────────────────────
                _cur.execute(
                    """
                    SELECT channel_name, COUNT(*) AS cnt
                    FROM messages GROUP BY channel_id ORDER BY cnt DESC LIMIT 5
                    """
                )
                top_channels = _cur.fetchall()
                if top_channels:
                    embed.add_field(
                        name="📁 Most Active Channels",
                        value="\n".join(
                            f"{i+1}. **#{r['channel_name']}** — {r['cnt']:,}"
                            for i, r in enumerate(top_channels)
                        ),
                        inline=True,
                    )

                # ── Recent Activity ──────────────────────────────
                _cur.execute("SELECT MIN(date) AS oldest, MAX(date) AS newest FROM messages")
                date_range = _cur.fetchone()
                _cur.execute("SELECT COUNT(*) AS cnt FROM messages WHERE date >= date('now', '-7 days')")
                week_count = int(_cur.fetchone()["cnt"])
                _cur.execute("SELECT COUNT(DISTINCT user_id) AS cnt FROM messages WHERE date >= date('now', '-7 days')")
                week_users = int(_cur.fetchone()["cnt"])
                _cur.execute("SELECT COUNT(*) AS cnt FROM messages WHERE date >= date('now', '-1 day')")
                day_count = int(_cur.fetchone()["cnt"])

                activity_parts = [f"Last 24h: **{day_count:,}** msgs", f"Last 7d: **{week_count:,}** msgs from **{week_users}** users"]
                if date_range and date_range["oldest"]:
                    activity_parts.append(f"Archive: {date_range['oldest']} → {date_range['newest']}")
                embed.add_field(name="📈 Recent Activity", value="\n".join(activity_parts), inline=False)

                _conn.close()
            except Exception as exc:
                logger.error("soupystats archive query failed: %s", exc, exc_info=True)

        # ── Image Generators ─────────────────────────────────────
        stats_data = await read_user_stats()
        if stats_data:
            users_stats = []
            for uid, data in stats_data.items():
                if "servers" in data and server_id in data["servers"]:
                    ss = data["servers"][server_id]
                    imgs = ss.get("images_generated", 0)
                    if imgs > 0:
                        users_stats.append({"username": data.get("username", "Unknown"), "images_generated": imgs})
            if users_stats:
                top_images = sorted(users_stats, key=lambda x: x["images_generated"], reverse=True)[:10]
                embed.add_field(
                    name="🎨 Top Image Generators",
                    value="\n".join(
                        f"{i+1}. **{u['username']}** — {u['images_generated']:,}"
                        for i, u in enumerate(top_images)
                    ),
                    inline=True,
                )

        embed.set_footer(
            text=f"Requested by {interaction.user}",
            icon_url=interaction.user.avatar.url if interaction.user.avatar else None,
        )
        await interaction.followup.send(embed=embed)
        logger.info(f"Sent soupystats to {interaction.user}")

    except Exception as e:
        logger.error(f"soupystats error: {e}", exc_info=True)
        await interaction.followup.send(f"Error fetching statistics: {e}", ephemeral=True)


async def check_sd_server_status() -> bool:
    """
    Checks if the SD server is online by making a request to its health endpoint.
    Returns True if the server is online, False otherwise.
    """
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{SD_SERVER_URL.rstrip('/')}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "ok"
                return False
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error checking SD server status: {error_detail}")
        logger.error(f"SD server URL: {SD_SERVER_URL}")
        return False

@bot.tree.command(name="status", description="Displays the current status of the bot, SD server, and chat functions.")
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
    
    # Check SD server status
    sd_server_online = await check_sd_server_status()
    sd_status = "🟢 Online" if sd_server_online else "🔴 Offline"
    
    # Check chat functions status
    await check_chat_functions()
    chat_status = "🟢 Online" if chat_functions_online else "🔴 Offline"
    
    # Create an embed message
    embed = discord.Embed(
        title="Bot Status",
        color=discord.Color.blue()
    )
    embed.add_field(name="SD Server", value=sd_status, inline=False)
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
    "This is the dumbest question I've ever heard.",
    "Nah, that's dumb.",
    "Bro, really?",
    "Sure, why not.",
    "Probably, I guess.",
    "I mean... maybe?",
    "That's a you problem.",
    "Ask again when you're smarter.",
    "The answer is 42.",
    "I don't care.",
    "Literally who asked?",
    "That's not how this works.",
    "Google it yourself.",
    "Why are you like this?",
    "Just... no.",
    "Yeah, probably.",
    "Doubt it.",
    "Skill issue.",
    "Touch grass.",
    "Maybe touch some grass first.",
    "I'm just a ball, man.",
    "Can you not?",
    "Six seven six seven."
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
                temperature=float(os.getenv("NINE_BALL_TEMPERATURE", 0.8)),
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



@bot.tree.command(name="testurl", description="Test URL extraction functionality")
@app_commands.describe(text="Text containing URLs to test extraction")
async def test_url_command(interaction: discord.Interaction, text: str):
    """Test command to verify URL extraction is working"""
    logger.info(f"URL test command invoked by {interaction.user} with text: '{text}'")
    
    # Extract URLs
    urls = extract_urls(text)
    
    if not urls:
        await interaction.response.send_message("No URLs found in the provided text.", ephemeral=True)
        return
    
    # Create response message
    response_parts = [f"Found {len(urls)} URL(s):"]
    for i, url in enumerate(urls, 1):
        response_parts.append(f"{i}. {url}")
    
    response_text = "\n".join(response_parts)
    
    # Test content extraction for first URL
    if urls:
        await interaction.response.defer(ephemeral=True)
        
        async with aiohttp.ClientSession() as session:
            content = await extract_url_content(urls[0], session)
            if content:
                response_text += f"\n\nContent preview from first URL:\n{content[:500]}..."
            else:
                response_text += f"\n\nFailed to extract content from first URL: {urls[0]}"
        
        await interaction.followup.send(response_text, ephemeral=True)
    else:
        await interaction.response.send_message(response_text, ephemeral=True)
    
    logger.info(f"URL test completed for {interaction.user}")


@bot.tree.command(name="whattime", description="Fetches and displays the current time in a specified city.")
@app_commands.describe(location="The city for which to fetch the current time.")
async def whattime_command(interaction: discord.Interaction, location: str):
    logger.info(f"Command 'whattime' invoked by {interaction.user} for location: '{location}'")
    
    # Defer the response immediately to avoid interaction timeout
    await interaction.response.defer(ephemeral=False)
    
    try:
        # Use a more descriptive user agent to comply with Nominatim usage policy
        geolocator = Nominatim(
            user_agent="SoupyDiscordBot/1.0 (Discord Bot for time queries; contact: bot owner)",
            timeout=10
        )
        location_obj = geolocator.geocode(location, addressdetails=True, language='en', timeout=10)
        if not location_obj:
            await interaction.followup.send(f"Could not geocode the location: {location}", ephemeral=True)
            logger.error(f"[/whattime Command Error] Could not geocode: {location}")
            return
    
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
            await interaction.followup.send(f"Could not find timezone for the location: {location}", ephemeral=True)
            logger.error(f"[/whattime Command Error] Could not find timezone for: {location}")
            return
    
        timezone = pytz.timezone(timezone_str)
        current_time = datetime.now(timezone).strftime('%I:%M %p on %Y-%m-%d')
        await interaction.followup.send(f"It is currently {current_time} in {location_str}.")
        logger.info(f"Provided time for {interaction.user}: {current_time} in {location_str}")
    
    except AdapterHTTPError as e:
        # Handle specific HTTP errors - parse status code from error message
        error_str = str(e)
        status_code = None
        if '403' in error_str:
            status_code = 403
        elif '503' in error_str:
            status_code = 503
        elif '429' in error_str:
            status_code = 429
        
        if status_code == 403:
            error_msg = "The geocoding service has temporarily blocked requests. Please try again in a few minutes."
            logger.warning(f"[/whattime Command] Nominatim returned 403 (blocked): {e}")
        elif status_code == 503:
            error_msg = "The geocoding service is temporarily unavailable. Please try again later."
            logger.warning(f"[/whattime Command] Nominatim returned 503 (service unavailable): {e}")
        elif status_code == 429:
            error_msg = "The geocoding service is rate-limiting requests. Please try again in a moment."
            logger.warning(f"[/whattime Command] Nominatim returned 429 (rate limited): {e}")
        else:
            error_msg = f"The geocoding service returned an error. Please try again later."
            logger.error(f"[/whattime Command] Nominatim HTTP error: {e}")
        
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except Exception as followup_error:
            logger.error(f"[/whattime Command] Failed to send followup message: {followup_error}")
    
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        error_msg = "The geocoding service timed out or is unavailable. Please try again in a moment."
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except Exception as followup_error:
            logger.error(f"[/whattime Command] Failed to send followup message: {followup_error}")
        logger.error(f"[/whattime Command] Geocoding service error: {e}")
    
    except Exception as e:
        error_msg = "Sorry, I'm unable to process your request at the moment. Please try again later."
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except Exception as followup_error:
            logger.error(f"[/whattime Command] Failed to send followup message: {followup_error}")
        logger.error(f"[/whattime Command Exception] An unexpected error occurred: {e}")


@bot.tree.command(name="weather", description="Fetches and displays the current weather for a specified location.")
@app_commands.describe(location="The city or location for which to fetch the weather.")
async def weather_command(interaction: discord.Interaction, location: str):
    logger.info(f"Command 'weather' invoked by {interaction.user} for location: '{location}'")
    
    # Defer the response immediately to avoid interaction timeout
    await interaction.response.defer(ephemeral=False)
    
    try:
        # Geocode the location to get coordinates
        geolocator = Nominatim(
            user_agent="SoupyDiscordBot/1.0 (Discord Bot for weather queries; contact: bot owner)",
            timeout=10
        )
        location_obj = geolocator.geocode(location, addressdetails=True, language='en', timeout=10)
        if not location_obj:
            await interaction.followup.send(f"Could not find the location: {location}", ephemeral=True)
            logger.error(f"[/weather Command Error] Could not geocode: {location}")
            return
        
        address = location_obj.raw.get('address', {})
        country = address.get('country', 'Unknown country')
        admin_area = address.get('state', address.get('region', address.get('county', '')))
        city = address.get('city', address.get('town', address.get('village', address.get('municipality', location.title()))))
        
        is_country_query = location.strip().lower() == country.lower()
        
        # Build location string using address components, avoiding duplicates
        if is_country_query:
            location_str = country
        else:
            location_parts = []
            
            # Add city (or user input if city not available)
            if city and city.lower() != location.lower():
                location_parts.append(city)
            else:
                location_parts.append(location.title())
            
            # Add admin_area only if it's different from city and not already in parts
            if admin_area:
                current_str = ", ".join(location_parts).lower()
                # Check if admin_area is not already in the location string
                if admin_area.lower() not in current_str:
                    location_parts.append(admin_area)
            
            # Add country only if not already included
            current_str = ", ".join(location_parts).lower()
            if country.lower() not in current_str:
                location_parts.append(country)
            
            location_str = ", ".join(location_parts)
        
        # Fetch weather data from Open-Meteo (free, no API key required)
        lat = location_obj.latitude
        lon = location_obj.longitude
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,wind_direction_10m,cloud_cover,surface_pressure,precipitation_probability&hourly=temperature_2m,precipitation_probability&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,precipitation_probability_max&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch&timezone=auto&forecast_days=7"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    error_msg = f"Weather service returned an error (HTTP {response.status}). Please try again later."
                    await interaction.followup.send(error_msg, ephemeral=True)
                    logger.error(f"[/weather Command] Open-Meteo API error: HTTP {response.status}")
                    return
                
                data = await response.json()
        
        # Parse weather data from Open-Meteo format
        current = data.get('current', {})
        if not current:
            error_msg = "Weather data is not available for this location."
            await interaction.followup.send(error_msg, ephemeral=True)
            logger.error("[/weather Command] No current weather data in response")
            return
        
        temp = current.get('temperature_2m', 0)
        feels_like = current.get('apparent_temperature', temp)
        humidity = current.get('relative_humidity_2m', 0)
        pressure = current.get('surface_pressure', 0)
        wind_speed = current.get('wind_speed_10m', 0)
        wind_direction = current.get('wind_direction_10m', None)
        cloudiness = current.get('cloud_cover', 0)
        weather_code = current.get('weather_code', 0)
        precip_probability = current.get('precipitation_probability', 0)
        
        # Convert weather code to description (WMO Weather interpretation codes)
        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        description = weather_descriptions.get(weather_code, "Unknown")
        
        # Get forecast for today's high/low (from hourly data)
        hourly = data.get('hourly', {})
        hourly_temps = hourly.get('temperature_2m', [])
        temp_max = max(hourly_temps) if hourly_temps else temp
        temp_min = min(hourly_temps) if hourly_temps else temp
        
        # Parse daily forecast data
        daily = data.get('daily', {})
        daily_dates = daily.get('time', [])
        daily_max_temps = daily.get('temperature_2m_max', [])
        daily_min_temps = daily.get('temperature_2m_min', [])
        daily_weather_codes = daily.get('weather_code', [])
        daily_precipitation = daily.get('precipitation_sum', [])
        daily_precip_probability = daily.get('precipitation_probability_max', [])
        
        # Convert wind direction to compass direction
        wind_dir_str = "N/A"
        if wind_direction is not None:
            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            wind_dir_str = directions[int((wind_direction + 11.25) / 22.5) % 16]
        
        # Create embed
        embed = discord.Embed(
            title=f"🌤️ Weather in {location_str}",
            description=f"**{description}**",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="🌡️ Temperature",
            value=f"{temp:.1f}°F (feels like {feels_like:.1f}°F)\n"
                  f"High: {temp_max:.1f}°F | Low: {temp_min:.1f}°F",
            inline=False
        )
        
        embed.add_field(
            name="💨 Wind",
            value=f"{wind_speed:.1f} mph {wind_dir_str}",
            inline=True
        )
        
        embed.add_field(
            name="💧 Humidity",
            value=f"{humidity}%",
            inline=True
        )
        
        embed.add_field(
            name="☁️ Clouds",
            value=f"{cloudiness}%",
            inline=True
        )
        
        embed.add_field(
            name="📊 Pressure",
            value=f"{pressure:.1f} hPa",
            inline=True
        )
        
        embed.add_field(
            name="🌧️ Rain Chance",
            value=f"{precip_probability:.0f}%",
            inline=True
        )
        
        # Add 7-day forecast if available
        if daily_dates and len(daily_dates) > 0:
            forecast_lines = []
            for i in range(min(7, len(daily_dates))):
                date_str = daily_dates[i]
                try:
                    # Parse date and format as "Mon 1" (day of week and day number only)
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    formatted_date = date_obj.strftime('%a %d')
                except:
                    formatted_date = date_str
                
                max_temp = daily_max_temps[i] if i < len(daily_max_temps) else 0
                min_temp = daily_min_temps[i] if i < len(daily_min_temps) else 0
                weather_code_daily = daily_weather_codes[i] if i < len(daily_weather_codes) else 0
                precip = daily_precipitation[i] if i < len(daily_precipitation) else 0
                precip_prob = daily_precip_probability[i] if i < len(daily_precip_probability) else 0
                
                # Get weather emoji/description for forecast
                weather_desc = weather_descriptions.get(weather_code_daily, "Unknown")
                # Use shorter descriptions for forecast
                if "Clear" in weather_desc:
                    emoji = "☀️"
                elif "cloudy" in weather_desc.lower() or "Overcast" in weather_desc:
                    emoji = "☁️"
                elif "rain" in weather_desc.lower() or "drizzle" in weather_desc.lower():
                    emoji = "🌧️"
                elif "snow" in weather_desc.lower():
                    emoji = "❄️"
                elif "Thunderstorm" in weather_desc:
                    emoji = "⛈️"
                elif "Fog" in weather_desc:
                    emoji = "🌫️"
                else:
                    emoji = "🌤️"
                
                # Convert mm to inches if needed (Open-Meteo might return mm even with precipitation_unit=inch)
                # If precipitation is > 10, it's likely in mm (convert: 1 mm = 0.0393701 inches)
                precip_inches = precip
                if precip > 10:
                    precip_inches = precip * 0.0393701
                
                # Format with fixed-width fields for uniform spacing
                # Use proper formatting to ensure alignment
                date_padded = formatted_date.ljust(8)  # "Wed 31 " = 8 chars (no month)
                temp_formatted = f"{max_temp:3.0f}°/{min_temp:3.0f}°F"
                temp_padded = temp_formatted.ljust(12)  # " 58°/ 48°F" = 12 chars
                
                # Format precipitation with fixed width (always same length)
                # Check if it's snow based on weather code
                is_snow = "snow" in weather_desc.lower()
                # Show precipitation type if there's any amount OR if there's a probability > 0
                if precip_inches > 0:
                    # Show actual amount, even if small (e.g., 0.005")
                    if is_snow:
                        precip_str = f"{precip_inches:5.2f}\" snow"
                    else:
                        precip_str = f"{precip_inches:5.2f}\" rain"
                    # Pad to fixed width (12 chars to accommodate "X.XX\" rain" or "X.XX\" snow")
                    precip_str = precip_str.ljust(12)
                elif precip_prob > 0:
                    # Only show "<0.01" when amount is exactly 0 but there's a probability
                    # This handles cases where probability exists but expected accumulation is negligible
                    if is_snow:
                        precip_str = "<0.01\" snow".ljust(12)
                    else:
                        precip_str = "<0.01\" rain".ljust(12)
                else:
                    precip_str = "            "  # 12 spaces to match "X.XX\" rain/snow" width
                
                # Format precipitation probability with fixed width (always 4 chars: "XX%" or "XXX%")
                precip_prob_str = f"{precip_prob:3.0f}%"
                
                forecast_lines.append(f"{emoji} {date_padded}{temp_padded}{precip_str}{precip_prob_str}")
            
            forecast_text = "\n".join(forecast_lines)
            # Use code block for monospace font to ensure proper alignment
            embed.add_field(
                name="📅 7-Day Forecast",
                value=f"```\n{forecast_text}\n```",
                inline=False
            )
        
        # Add timestamp
        embed.set_footer(text=f"Data from Open-Meteo • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        await interaction.followup.send(embed=embed)
        logger.info(f"Provided weather for {interaction.user}: {location_str}")
    
    except AdapterHTTPError as e:
        # Handle geocoding HTTP errors
        error_str = str(e)
        status_code = None
        if '403' in error_str:
            status_code = 403
        elif '503' in error_str:
            status_code = 503
        elif '429' in error_str:
            status_code = 429
        
        if status_code == 403:
            error_msg = "The geocoding service has temporarily blocked requests. Please try again in a few minutes."
            logger.warning(f"[/weather Command] Nominatim returned 403 (blocked): {e}")
        elif status_code == 503:
            error_msg = "The geocoding service is temporarily unavailable. Please try again later."
            logger.warning(f"[/weather Command] Nominatim returned 503 (service unavailable): {e}")
        elif status_code == 429:
            error_msg = "The geocoding service is rate-limiting requests. Please try again in a moment."
            logger.warning(f"[/weather Command] Nominatim returned 429 (rate limited): {e}")
        else:
            error_msg = "The geocoding service returned an error. Please try again later."
            logger.error(f"[/weather Command] Nominatim HTTP error: {e}")
        
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except Exception as followup_error:
            logger.error(f"[/weather Command] Failed to send followup message: {followup_error}")
    
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        error_msg = "The geocoding service timed out or is unavailable. Please try again in a moment."
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except Exception as followup_error:
            logger.error(f"[/weather Command] Failed to send followup message: {followup_error}")
        logger.error(f"[/weather Command] Geocoding service error: {e}")
    
    except aiohttp.ClientError as e:
        error_msg = "Failed to fetch weather data. Please try again later."
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except Exception as followup_error:
            logger.error(f"[/weather Command] Failed to send followup message: {followup_error}")
        logger.error(f"[/weather Command] HTTP client error: {e}")
    
    except Exception as e:
        error_msg = "Sorry, I'm unable to process your request at the moment. Please try again later."
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except Exception as followup_error:
            logger.error(f"[/weather Command] Failed to send followup message: {followup_error}")
        logger.error(f"[/weather Command Exception] An unexpected error occurred: {e}")


"""
---------------------------------------------------------------------------------
Stable Diffusion-Related Functionality
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





# Archive helpers for web dashboard
def _ensure_media_dirs():
    from pathlib import Path
    media = Path("media")
    (media / "images").mkdir(parents=True, exist_ok=True)
    (media / "thumbs").mkdir(parents=True, exist_ok=True)
    return media


def archive_image_bytes(image_bytes: bytes, *, filename: str, prompt: str, user_id: int, username: str,
                        width: int, height: int, seed: int, guild_id: int | None, channel_id: int | None) -> None:
    try:
        import json
        from datetime import datetime, timezone
        from io import BytesIO
        from pathlib import Path
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
        }
        index_path = media / "images" / "index.jsonl"
        with open(index_path, "a", encoding="utf-8") as idx:
            idx.write(json.dumps(meta, ensure_ascii=False) + "\n")
    except Exception as _e:
        logger.debug(f"archive_image_bytes failed: {_e}")


def archive_sent_message(content: str, *, user_id: int, username: str, guild_id: int | None, channel_id: int | None, 
                         image_filename: str | None = None, event_type: str = "message", terminal_output: str | None = None) -> None:
    try:
        import json
        from datetime import datetime, timezone
        from pathlib import Path
        media = _ensure_media_dirs()
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "content": content,
            "user_id": user_id,
            "username": username,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "event_type": event_type,
        }
        if image_filename:
            entry["image_filename"] = image_filename
        if terminal_output:
            entry["terminal_output"] = terminal_output
        log_path = media / "messages.jsonl"
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as _e:
        logger.debug(f"archive_sent_message failed: {_e}")


def archive_vision_image(image_bytes: bytes, *, description: str, user_id: int, username: str, 
                         guild_id: int | None, channel_id: int | None, original_url: str) -> str | None:
    """
    Archives a vision-processed image to the media directory.
    Returns the filename if successful, None otherwise.
    """
    try:
        import json
        from datetime import datetime, timezone
        from io import BytesIO
        from pathlib import Path
        from PIL import Image
        import hashlib

        media = _ensure_media_dirs()
        
        # Create a unique filename using timestamp and hash
        timestamp = datetime.now(timezone.utc)
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(image_bytes).hexdigest()[:8]
        filename = f"vision_{ts_str}_{hash_suffix}.png"
        
        img_path = media / "images" / filename
        thumb_path = media / "thumbs" / filename

        # Save original as PNG
        try:
            im = Image.open(BytesIO(image_bytes)).convert("RGB")
            im.save(img_path, format="PNG")
            
            # Create thumbnail
            im_thumb = im.copy()
            im_thumb.thumbnail((400, 400))
            im_thumb.save(thumb_path, format="PNG")
        except Exception as e:
            logger.debug(f"Failed to save vision image: {e}")
            return None

        # Append to images index
        meta = {
            "ts": timestamp.isoformat(),
            "filename": filename,
            "prompt": f"[Vision] {description[:100]}...",  # Truncate for index
            "user_id": user_id,
            "username": username,
            "width": im.width,
            "height": im.height,
            "seed": 0,  # N/A for vision images
            "guild_id": guild_id,
            "channel_id": channel_id,
            "event_type": "vision",
            "original_url": original_url,
        }
        index_path = media / "images" / "index.jsonl"
        with open(index_path, "a", encoding="utf-8") as idx:
            idx.write(json.dumps(meta, ensure_ascii=False) + "\n")
        
        logger.debug(f"Archived vision image: {filename}")
        return filename
        
    except Exception as _e:
        logger.debug(f"archive_vision_image failed: {_e}")
        return None


@bot.tree.command(name="sd", description="Generates an image using Stable Diffusion.")
@app_commands.describe(
    description="Description of the image to generate",
    size="Size of the image",
    seed="Seed for random generation"
)
@app_commands.choices(size=[
    app_commands.Choice(name=f"Default ({SD_DEFAULT_WIDTH}x{SD_DEFAULT_HEIGHT})", value="default"),
    app_commands.Choice(name=f"Wide ({SD_WIDE_WIDTH}x{SD_WIDE_HEIGHT})", value="wide"),
    app_commands.Choice(name=f"Tall ({SD_TALL_WIDTH}x{SD_TALL_HEIGHT})", value="tall"),
    app_commands.Choice(name=f"Square ({SD_DEFAULT_WIDTH}x{SD_DEFAULT_HEIGHT})", value="square"),
])
async def sd(interaction: discord.Interaction, 
               description: str,
               size: Optional[app_commands.Choice[str]] = None,
               seed: Optional[int] = None):
    size_value = size.value if size else 'default'
    
    logger.info(f"🎨 Slash Command 'sd' invoked by {interaction.user} with description: '{description}', size: '{size_value}', seed: '{seed if seed else 'random'}'")
    # Use defer + followup to avoid Unknown interaction issues
    try:
        if not interaction.response.is_done():
            # IMPORTANT: Do NOT defer ephemerally for /sd.
            # An ephemeral defer causes the generation to appear "only to the user",
            # which is not desired for regular /sd generations.
            await interaction.response.defer(thinking=True)
    except Exception as e:
        logger.debug(f"Defer failed: {e}")
    await bot.sd_queue.put({
        'type': 'sd',
        'interaction': interaction,
        'description': description,
        'size': size_value,
        'seed': seed,
    })
    logger.info(f"🎨 Queued image generation for {interaction.user}: description='{description}', size='{size_value}', seed='{seed if seed else 'random'}'")
    # Avoid sending extra followup ack to prevent 40060/10062 errors if interaction state changes



# -------------------------------------------------------------------------
# Define the button-handling methods BEFORE process_flux_queue()
# -------------------------------------------------------------------------

async def handle_remix(interaction, prompt, width, height, seed, queue_size):
    # Update to include server ID
    await increment_user_stat(interaction.user.id, 'images_generated', interaction.guild_id)
    
    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Remix", queue_size=queue_size)

async def handle_wide(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Wide", queue_size=queue_size)

async def handle_tall(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Tall", queue_size=queue_size)

async def handle_edit(interaction, prompt, width, height, seed, queue_size):
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation
    await generate_sd_image(interaction, prompt, width, height, seed, action_name="Edit", queue_size=queue_size)

async def handle_regenerate_selected(interaction, prompt, width, height, seed, queue_size, thumbnail_index):
    """Handle regenerating a full 1024x1024 image using the selected thumbnail's seed."""
    # Increment the images_generated stat
    await increment_user_stat(interaction.user.id, 'images_generated')
    
    # Proceed with image generation using normal SD settings
    action_label = f"Selected #{thumbnail_index}"
    await generate_sd_image(interaction, prompt, width, height, seed, action_name=action_label, queue_size=queue_size)

async def handle_thumbnail_upscale(interaction, prompt, width, height, thumbnail_data, queue_size, thumbnail_index):
    """Handle regenerating thumbnail with 30 steps then upscaling to full-size image"""
    try:
        logger.info(f"🔲 Regenerating and upscaling thumbnail {thumbnail_index} for {interaction.user}: prompt='{prompt}', seed={thumbnail_data['seed']}")
        
        # Check if we need to send an initial response
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        sd_server_url = SD_SERVER_URL.rstrip('/')
        
        # Use typing context manager for consistent behavior
        async with interaction.channel.typing():
            # Use optimized session with connection pooling and keep-alive
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                keepalive_timeout=30,  # Keep connections alive
                enable_cleanup_closed=True
            )
            # Increased timeout for SD 3.5 Medium on Mac (can take 2-5 minutes)
            timeout = aiohttp.ClientTimeout(total=600, connect=10)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive'}
            ) as session:
                # Start timing the entire process
                total_start_time = time.perf_counter()
                
                # Step 1: Regenerate thumbnail with steps from env at same resolution and seed
                steps = int(os.getenv("SD_STEPS", 20))
                logger.info(f"🔲 Step 1: Regenerating thumbnail {thumbnail_index} with {steps} steps at {thumbnail_data['width']}x{thumbnail_data['height']}")
                regenerate_start_time = time.perf_counter()
                
                regenerate_payload = {
                    "prompt": prompt,
                    "negative_prompt": os.getenv("SD_NEGATIVE_PROMPT", "") or "",
                    "steps": str(int(os.getenv("SD_STEPS", 20))),  # Deprecated in new flow; kept for compatibility
                    "guidance_scale": str(float(os.getenv("SD_GUIDANCE", 7.5))),
                    "width": str(thumbnail_data['width']),
                    "height": str(thumbnail_data['height']),
                    "seed": str(thumbnail_data['seed'])
                }
                
                async with session.post(f"{sd_server_url}/sd", data=regenerate_payload) as regenerate_response:
                    if regenerate_response.status != 200:
                        logger.error(f"🔲 Regeneration failed for thumbnail {thumbnail_index}: HTTP {regenerate_response.status}")
                        await interaction.followup.send(f"❌ Failed to regenerate thumbnail: HTTP {regenerate_response.status}", ephemeral=True)
                        return
                    
                    regenerated_bytes = await regenerate_response.read()
                    regenerate_end_time = time.perf_counter()
                    regenerate_duration = regenerate_end_time - regenerate_start_time
                    logger.info(f"⏱️ Thumbnail regeneration completed in {regenerate_duration:.2f} seconds")
                
                # Step 2: Upscale the regenerated thumbnail
                logger.info(f"🔲 Step 2: Upscaling regenerated thumbnail to {width}x{height}")
                upscale_start_time = time.perf_counter()
                
                # Prepare form data for upscaling
                form = aiohttp.FormData()
                form.add_field('image', regenerated_bytes, filename='regenerated_thumbnail.jpg', content_type='image/jpeg')
                form.add_field('target_width', str(width))
                form.add_field('target_height', str(height))

                async with session.post(f"{sd_server_url}/upscale", data=form) as upscale_response:
                    if upscale_response.status == 200:
                        upscaled_bytes = await upscale_response.read()
                        
                        # End timing the upscaling process
                        upscale_end_time = time.perf_counter()
                        upscale_duration = upscale_end_time - upscale_start_time
                        total_duration = upscale_end_time - total_start_time
                        
                        logger.info(f"⏱️ Upscaling completed in {upscale_duration:.2f} seconds")
                        logger.info(f"⏱️ Total process completed in {total_duration:.2f} seconds for {interaction.user}")

                        # Generate a unique filename
                        random_number = random.randint(100000, 999999)
                        safe_prompt = re.sub(r'\W+', '', prompt[:40]).lower()
                        filename = f"{random_number}_{safe_prompt}_regenerated_upscaled.png"

                        # Create a Discord File object from the upscaled image bytes
                        image_file = discord.File(BytesIO(upscaled_bytes), filename=filename)

                        # Create embed messages
                        description_embed = discord.Embed(
                            description=f"**Prompt:** {prompt}\n**Regenerated & Upscaled from thumbnail {thumbnail_index}**",
                            color=discord.Color.purple()
                        )
                        details_embed = discord.Embed(color=discord.Color.green())

                        queue_total = queue_size + 1
                        details_text = f"🔲 Regen+Upscale Thumbnail {thumbnail_index} ⏱️ {total_duration:.2f}s 📋 {queue_total}"
                        details_embed.description = details_text

                        # Initialize the SDRemixView with current image parameters
                        new_view = SDRemixView(
                            prompt=prompt, width=width, height=height, seed=thumbnail_data['seed']
                        )

                        # When sending the final message, use followup if the initial response was deferred
                        if interaction.response.is_done():
                            await interaction.followup.send(
                                content=f"{interaction.user.mention} 🔲 Regenerated & Upscaled Image:",
                                embeds=[description_embed, details_embed],
                                file=image_file,
                                view=new_view
                            )
                        else:
                            await interaction.channel.send(
                                content=f"{interaction.user.mention} 🔲 Regenerated & Upscaled Image:",
                                embeds=[description_embed, details_embed],
                                file=image_file,
                                view=new_view
                            )
                        logger.info(f"🔲 Regeneration and upscaling completed for {interaction.user}: filename='{filename}', total_duration={total_duration:.2f}s")
                    else:
                        logger.error(f"🔲 Upscaling server error for {interaction.user}: HTTP {upscale_response.status}")
                        try:
                            await interaction.followup.send(
                                f"❌ Upscaling server error: HTTP {upscale_response.status}", ephemeral=True
                            )
                        except Exception as send_error:
                            logger.error(f"❌ Failed to send follow-up message: {send_error}")
        
    except (ClientConnectorError, ClientOSError):
        logger.error(f"🔲 Server is offline or unreachable for {interaction.user}.")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    "❌ The server is currently offline.", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except ServerTimeoutError:
        logger.error(f"🔲 Server request timed out for {interaction.user}.")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    "❌ The server timed out while processing your request. Please try again later.", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")
    except Exception as e:
        logger.error(f"🔲 Unexpected error during regeneration and upscaling for {interaction.user}: {e}")
        if isinstance(interaction, discord.Interaction):
            try:
                await interaction.followup.send(
                    f"❌ An unexpected error occurred during regeneration and upscaling: {e}", ephemeral=True
                )
            except Exception as send_error:
                logger.error(f"❌ Failed to send follow-up message: {send_error}")

async def handle_outpaint(interaction, prompt, direction, width, height, seed, queue_size, strength=0.8, steps=None, guidance=None):
    """Handle outpainting - extend image by 25% in specified directions using inpaint mask to preserve interior"""
    try:
        logger.info(f"🖼️ Starting outpainting for {interaction.user}: prompt='{prompt}', direction='{direction}', size={width}x{height}")
        
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
                logger.warning(f"Failed to send outpaint error via followup; falling back to channel.send: {send_error}")
                await interaction.channel.send(f"{interaction.user.mention} ❌ Please upload an image and try again.")
            return

        # Use typing context manager for consistent behavior
        async with interaction.channel.typing():
            # Use optimized session with connection pooling and keep-alive
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                keepalive_timeout=30,  # Keep connections alive
                enable_cleanup_closed=True
            )
            # Increased timeout for SD 3.5 Medium on Mac (can take 2-5 minutes)
            timeout = aiohttp.ClientTimeout(total=600, connect=10)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive'}
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

                    cv_img = cv2.cvtColor(np.array(original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
                    extended = cv2.copyMakeBorder(
                        cv_img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
                    )
                    # Do NOT blur the entire canvas; keep the original interior pixel-perfect
                    padded_image = Image.fromarray(cv2.cvtColor(extended, cv2.COLOR_BGR2RGB))

                    # Build mask that covers ONLY the new outpainted areas (not the original image)
                    # This ensures the original image remains untouched, preventing visible borders
                    mask_img = Image.new('L', (new_width, new_height), 0)
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
                    protection_mask = Image.new('L', (new_width, new_height), 255)
                    protect_draw = ImageDraw.Draw(protection_mask)
                    # The original image area should be black (protected) in the protection mask
                    protect_draw.rectangle(
                        [paste_x, paste_y, paste_x + original_width - 1, paste_y + original_height - 1],
                        fill=0
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
                        mask_img = Image.fromarray(mask_np).convert('L')
                    except Exception as e:
                        logger.warning(f"Mask blur failed, using hard mask: {e}")

                    padded_png_bytes = BytesIO()
                    padded_image.save(padded_png_bytes, format='PNG')
                    padded_png_bytes.seek(0)
                    padded_raw = padded_png_bytes.getvalue()

                    mask_png_bytes = BytesIO()
                    mask_img.save(mask_png_bytes, format='PNG')
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
                    form.add_field("negative_prompt", os.getenv("SD_NEGATIVE_PROMPT", "") or "")
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
                    form.add_field("negative_prompt", os.getenv("SD_NEGATIVE_PROMPT", "") or "")
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
                            await interaction.followup.send(f"❌ Outpaint server error: HTTP {r.status}", ephemeral=True)
                            return

                # Generate a unique filename
                random_number = random.randint(100000, 999999)
                safe_prompt = re.sub(r'\W+', '', prompt[:40]).lower()
                filename = f"{random_number}_{safe_prompt}_outpaint_{direction}.png"

                # If hybrid used, server already harmonized/blended; send directly
                final_bytes = out_bytes
                if not use_hybrid:
                    # Safeguard: composite the original interior back onto the result
                    try:
                        result_img = Image.open(BytesIO(out_bytes)).convert('RGB')
                        result_img.paste(original_image.convert('RGB'), (paste_x, paste_y))
                        final_io = BytesIO()
                        result_img.save(final_io, format='PNG')
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
                    color=discord.Color.purple()
                )
                details_embed = discord.Embed(color=discord.Color.green())

                queue_total = queue_size + 1
                details_text = f"🖼️ Outpaint {direction.title()} ⏱️ Extended ⏱️ 📋 {queue_total}"
                details_embed.description = details_text

                # Initialize the SDRemixView with new image parameters
                new_view = SDRemixView(
                    prompt=prompt, width=new_width, height=new_height, seed=seed
                )

                # Send the result
                # Component interactions can expire if the queue is long; fall back to a normal channel send.
                try:
                    await interaction.followup.send(
                        content=f"{interaction.user.mention} 🖼️ Outpainted Image:",
                        embeds=[description_embed, details_embed],
                        file=image_file,
                        view=new_view,
                        ephemeral=False
                    )
                except Exception as send_error:
                    logger.error(f"❌ Failed to send outpaint result via followup, falling back to channel.send: {send_error}")
                    await interaction.channel.send(
                        content=f"{interaction.user.mention} 🖼️ Outpainted Image:",
                        embeds=[description_embed, details_embed],
                        file=image_file,
                        view=new_view
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
                        event_type="image_generation"
                    )
                except Exception:
                    pass
                
                logger.info(f"🖼️ Outpainting completed for {interaction.user}: filename='{filename}', direction='{direction}'")

    except Exception as e:
        logger.error(f"🖼️ Error in handle_outpaint for {interaction.user}: {e}")
        # Interaction tokens can expire; always fall back to channel.send on failures.
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Error during outpainting: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f"❌ Error during outpainting: {e}", ephemeral=True)
        except Exception as send_error:
            logger.error(f"❌ Failed to send outpaint error via interaction; falling back to channel.send: {send_error}")
            try:
                await interaction.channel.send(f"{interaction.user.mention} ❌ Error during outpainting: {e}")
            except Exception:
                pass

async def handle_2x2_grid(interaction, prompt, width, height, seed, queue_size):
    """Handle 2x2 grid generation - creates 4 candidate images and combines them"""
    try:
        logger.info(f"🔲 Starting 2x2 grid generation for {interaction.user}: prompt='{prompt}', size={width}x{height}")
        
        # For the new flow, generate four 1024x1024 candidates regardless of the current view size
        thumbnail_width = 1024
        thumbnail_height = 1024
        # Ensure dimensions are multiples of 64 for SD (defensive)
        thumbnail_width = ((thumbnail_width + 63) // 64) * 64
        thumbnail_height = ((thumbnail_height + 63) // 64) * 64
        
        # Generate 4 unique seeds
        thumbnail_seeds = []
        base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        for i in range(4):
            thumbnail_seeds.append(base_seed + i)
        
        logger.info(f"🔲 Generated seeds for thumbnails: {thumbnail_seeds}")
        
        # Generate 4 thumbnail images
        thumbnail_images = []
        thumbnail_data = []  # Store individual thumbnail data for upscaling
        sd_server_url = SD_SERVER_URL.rstrip('/')
        # Use 10 steps for the candidate images
        num_steps = 10
        guidance = float(os.getenv("SD_GUIDANCE", 7.5))
        negative_prompt = os.getenv("SD_NEGATIVE_PROMPT", "") or ""
        
        async with interaction.channel.typing():
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            # Increased timeout for SD 3.5 Medium on Mac (can take 2-5 minutes)
            timeout = aiohttp.ClientTimeout(total=600, connect=10)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive'}
            ) as session:
                for i, thumb_seed in enumerate(thumbnail_seeds):
                    payload = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "steps": str(num_steps),
                        "guidance_scale": str(guidance),
                        "width": str(thumbnail_width),
                        "height": str(thumbnail_height),
                        "seed": str(thumb_seed)
                    }
                    
                    logger.info(f"🔲 Generating thumbnail {i+1}/4 with seed {thumb_seed}")
                    
                    async with session.post(f"{sd_server_url}/sd", data=payload) as response:
                        if response.status == 200:
                            image_bytes = await response.read()
                            thumbnail_images.append(image_bytes)
                            # Store individual thumbnail data for upscaling
                            thumbnail_data.append({
                                'image_bytes': image_bytes,
                                'seed': thumb_seed,
                                'width': thumbnail_width,
                                'height': thumbnail_height
                            })
                            logger.info(f"🔲 Successfully generated thumbnail {i+1}/4")
                        else:
                            logger.error(f"🔲 Failed to generate thumbnail {i+1}/4: HTTP {response.status}")
                            raise Exception(f"Failed to generate thumbnail {i+1}/4: HTTP {response.status}")
        
        # Combine thumbnails into 2x2 grid
        logger.info(f"🔲 Combining {len(thumbnail_images)} thumbnails into 2x2 grid")
        
        # Create the combined image (2x2 grid of 1024x1024 => 2048x2048)
        combined_width = thumbnail_width * 2
        combined_height = thumbnail_height * 2
        combined_image = Image.new('RGB', (combined_width, combined_height))
        
        # Paste each thumbnail into the grid
        positions = [
            (0, 0),  # Top-left
            (thumbnail_width, 0),  # Top-right
            (0, thumbnail_height),  # Bottom-left
            (thumbnail_width, thumbnail_height)  # Bottom-right
        ]
        
        for i, (image_bytes, pos) in enumerate(zip(thumbnail_images, positions)):
            thumbnail_img = Image.open(BytesIO(image_bytes))
            combined_image.paste(thumbnail_img, pos)
            logger.debug(f"🔲 Pasted thumbnail {i+1} at position {pos}")
        
        # Convert combined image to bytes
        combined_bytes = BytesIO()
        combined_image.save(combined_bytes, format='PNG')
        combined_bytes.seek(0)
        
        # Generate filename
        random_number = random.randint(100000, 999999)
        safe_prompt = re.sub(r'\W+', '', prompt[:40]).lower()
        filename = f"{random_number}_{safe_prompt}_2x2grid.png"
        
        # Create Discord file
        image_file = discord.File(combined_bytes, filename=filename)
        
        # Create embeds
        description_embed = discord.Embed(
            description=f"**Prompt:** {prompt}\n**Seeds:** {', '.join(map(str, thumbnail_seeds))}",
            color=discord.Color.purple()
        )
        
        details_embed = discord.Embed(
            description=f"🔲 2x2 Grid ⏱️ 10 steps each 📋 {queue_size + 1}",
            color=discord.Color.green()
        )
        
        # Create thumbnail selection view
        thumbnail_view = ThumbnailSelectionView(
            prompt=prompt,
            width=1024,
            height=1024,
            thumbnail_data=thumbnail_data
        )
        
        # Send the combined image with selection buttons
        await interaction.followup.send(
            content=f"{interaction.user.mention} 🔲 2x2 Thumbnail Grid:",
            embeds=[description_embed, details_embed],
            file=image_file,
            view=thumbnail_view
        )
        
        logger.info(f"🔲 Successfully created 2x2 grid for {interaction.user}: {filename}")
        
        # Increment the images_generated stat
        await increment_user_stat(interaction.user.id, 'images_generated')
        
    except Exception as e:
        logger.error(f"🔲 Error generating 2x2 grid for {interaction.user}: {e}")
        if not interaction.response.is_done():
            await interaction.response.send_message(f"❌ Error generating 2x2 grid: {e}", ephemeral=True)
        else:
            await interaction.followup.send(f"❌ Error generating 2x2 grid: {e}", ephemeral=True)


fancy_instructions = os.getenv("FANCY", "")


async def handle_fancy(interaction, prompt, width, height, seed, queue_size):
    """Handle the 'Fancy' button click."""
    try:
        logger.info(f"'Fancy' button clicked by {interaction.user} for prompt: '{prompt}'")
        
        # If this is a new interaction (not a followup), defer it
        if not interaction.response.is_done():
            await interaction.response.defer()  # Remove thinking=True to make it visible to channel

        # Get the fancy instructions
        fancy_instructions = os.getenv("FANCY", "")
        logger.debug(f"📜 Retrieved 'FANCY' instructions: {fancy_instructions}")

        # Combine instructions with prompt
        combined_instructions = f"{fancy_instructions}\n\nThe prompt you are elaborating on is: {prompt}"
        logger.debug(f"📜 Combined rewriting instructions for {interaction.user}: {combined_instructions}")

        # Start timing for prompt generation
        prompt_start_time = time.perf_counter()

        # Generate fancy prompt
        messages = [
            {"role": "system", "content": combined_instructions},
            {"role": "user", "content": "Please rewrite the above prompt accordingly."}
        ]

        logger.debug(f"📜 Sending the following messages to LLM (Fancy):\n{format_messages(messages)}")

        response = await async_chat_completion(
            model=os.getenv("LOCAL_CHAT"),
            messages=messages,
            temperature=float(os.getenv("FANCY_PROMPT_TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("FANCY_MAX_TOKENS", 150))
        )

        fancy_prompt = response.choices[0].message.content.strip()
        logger.info(f"🪄 Fancy prompt generated for {interaction.user}: '{fancy_prompt}'")

        # Calculate prompt rewriting duration
        prompt_end_time = time.perf_counter()
        prompt_duration = prompt_end_time - prompt_start_time
        logger.info(f"⏱️ Prompt rewriting time for {interaction.user}: {prompt_duration:.2f} seconds")

        # Strip surrounding quotes from the prompt (don't use full clean_response — no truncation or RAG sanitization needed)
        cleaned_prompt = fancy_prompt.strip()
        while (cleaned_prompt.startswith('"') and cleaned_prompt.endswith('"')) or (cleaned_prompt.startswith("'") and cleaned_prompt.endswith("'")):
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
            pre_duration=prompt_duration
        )

        logger.info(f"🪄 Passed cleaned fancy prompt to image generator for {interaction.user}")

    except Exception as e:
        error_msg = f"Error handling fancy button: {str(e)}"
        logger.error(error_msg)
        if not interaction.response.is_done():
            await interaction.response.send_message(error_msg, ephemeral=True)
        else:
            await interaction.followup.send(error_msg, ephemeral=True)
class ThumbnailSelectionView(View):
    def __init__(self, prompt: str, width: int, height: int, thumbnail_data: List[Dict]):
        super().__init__(timeout=None)
        self.prompt = prompt
        self.width = width
        self.height = height
        self.thumbnail_data = thumbnail_data
        logger.debug(f"ThumbnailSelectionView initialized: prompt='{prompt}', {width}x{height}, {len(thumbnail_data)} thumbnails")

    @discord.ui.button(label="1", style=discord.ButtonStyle.primary, custom_id="thumbnail_1_button", row=0)
    @universal_cooldown_check()
    async def thumbnail_1_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_thumbnail_selection(interaction, 0)

    @discord.ui.button(label="2", style=discord.ButtonStyle.primary, custom_id="thumbnail_2_button", row=0)
    @universal_cooldown_check()
    async def thumbnail_2_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_thumbnail_selection(interaction, 1)

    @discord.ui.button(label="3", style=discord.ButtonStyle.primary, custom_id="thumbnail_3_button", row=1)
    @universal_cooldown_check()
    async def thumbnail_3_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_thumbnail_selection(interaction, 2)

    @discord.ui.button(label="4", style=discord.ButtonStyle.primary, custom_id="thumbnail_4_button", row=1)
    @universal_cooldown_check()
    async def thumbnail_4_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._handle_thumbnail_selection(interaction, 3)

    async def _handle_thumbnail_selection(self, interaction: discord.Interaction, thumbnail_index: int):
        """Handle thumbnail selection and regenerate at 1024x1024 using selected seed"""
        logger.info(f"Thumbnail {thumbnail_index + 1} selected by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message(f"🛠️ Generating selected image at 1024x1024 using its seed...", ephemeral=True)
            
            selected_thumbnail = self.thumbnail_data[thumbnail_index]
            queue_size = bot.sd_queue.qsize()
            
            await bot.sd_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'regenerate_selected',
                'prompt': self.prompt,
                'width': 1024,
                'height': 1024,
                'seed': selected_thumbnail['seed'],
                'thumbnail_index': thumbnail_index + 1
            })
            
            logger.info(f"Enqueued regenerate_selected for thumbnail {thumbnail_index + 1} for {interaction.user}: prompt='{self.prompt}', size=1024x1024, seed={selected_thumbnail['seed']}")
            
            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, 'images_generated')
            
        except Exception as e:
            logger.error(f"Error during thumbnail selection for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating selected image.", ephemeral=True)


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

            await bot.sd_queue.put({
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
            queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put({
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

    @discord.ui.button(label="🌱🎲", style=discord.ButtonStyle.primary, custom_id="flux_remix_button", row=0)
    @universal_cooldown_check()
    async def remix_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Remix' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Remixing...", ephemeral=True)
            queue_size = bot.sd_queue.qsize()
            new_seed = random.randint(0, 2**32 - 1)
            await bot.sd_queue.put({
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


    @discord.ui.button(label="🎨 R-Fancy", style=discord.ButtonStyle.danger, custom_id="flux_random_fancy_button", row=1)
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
                (SD_TALL_WIDTH, SD_TALL_HEIGHT)   # Tall
            ]
            width, height = random.choice(dimensions)
            
            # Use LLM-generated prompt (set prompt to None so handle_random generates it)
            prompt = None  # Will be generated in handle_random
            
            queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'random',
                'width': width,
                'height': height,
                'seed': None,  # Random will generate its own seed
                'prompt': prompt  # None for LLM-generated prompt
            })
            logger.info(f"🎨 Enqueued 'R-Fancy' action for {interaction.user} with dimensions {width}x{height}")
            
            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, 'images_generated')
        except Exception as e:
            logger.error(f"🎨 Error queueing fancy random generation for {interaction.user}: {e}")

    @discord.ui.button(label="🔤 R-Keyword", style=discord.ButtonStyle.danger, custom_id="flux_random_keyword_button", row=1)
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
                (SD_TALL_WIDTH, SD_TALL_HEIGHT)   # Tall
            ]
            width, height = random.choice(dimensions)
            
            # Get random terms and use them directly as the prompt
            random_terms = get_random_terms()
            # Flatten the terms from the dictionary into a comma-separated string
            terms_list = []
            for category, terms in random_terms.items():
                # Split by comma in case there are multiple terms in a single category
                split_terms = [term.strip() for term in terms.split(',')]
                terms_list.extend(split_terms)
            prompt = ", ".join(terms_list)
            logger.info(f"🔤 Using only random terms for {interaction.user}: {prompt}")
            
            queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'random',
                'width': width,
                'height': height,
                'seed': None,  # Random will generate its own seed
                'prompt': prompt  # Direct terms-only prompt
            })
            logger.info(f"🔤 Enqueued 'R-Keyword' action for {interaction.user} with dimensions {width}x{height}")
            
            # Increment the images_generated stat
            await increment_user_stat(interaction.user.id, 'images_generated')
        except Exception as e:
            logger.error(f"🔤 Error queueing keyword random generation for {interaction.user}: {e}")

    @discord.ui.button(label="↔️", style=discord.ButtonStyle.primary, custom_id="flux_wide_button", row=1)
    @universal_cooldown_check()
    async def wide_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Wide' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Generating wide version...", ephemeral=True)
            queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'wide',
                'prompt': self.cleaned_prompt,
                'width': SD_WIDE_WIDTH,
                'height': SD_WIDE_HEIGHT,
                'seed': self.seed,
            })
            logger.info(f"Enqueued 'Wide' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={SD_WIDE_WIDTH}x{SD_WIDE_HEIGHT}, seed={self.seed}")
        except Exception as e:
            logger.error(f"Error during wide generation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating wide version.", ephemeral=True)

    @discord.ui.button(label="↕️", style=discord.ButtonStyle.primary, custom_id="flux_tall_button", row=1)
    @universal_cooldown_check()
    async def tall_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Tall' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Generating tall version...", ephemeral=True)
            queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'tall',
                'prompt': self.cleaned_prompt,
                'width': SD_TALL_WIDTH,
                'height': SD_TALL_HEIGHT,
                'seed': self.seed,
            })
            logger.info(f"Enqueued 'Tall' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={SD_TALL_WIDTH}x{SD_TALL_HEIGHT}, seed={self.seed}")
        except Exception as e:
            logger.error(f"Error during tall generation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating tall version.", ephemeral=True)

    @discord.ui.button(label="🔲 2x2", style=discord.ButtonStyle.secondary, custom_id="flux_2x2_button", row=1)
    @universal_cooldown_check()
    async def grid_2x2_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'2x2 Grid' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Generating 2x2 grid (4× 1024×1024 @ 10 steps)...", ephemeral=True)
            queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': '2x2_grid',
                'prompt': self.cleaned_prompt,
                'width': self.width,
                'height': self.height,
                'seed': self.seed,
            })
            logger.info(f"Enqueued '2x2 Grid' action for {interaction.user}: prompt='{self.cleaned_prompt}', size={self.width}x{self.height}, seed={self.seed}")
        except Exception as e:
            logger.error(f"Error during 2x2 grid generation for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error generating 2x2 grid.", ephemeral=True)

    @discord.ui.button(label="⤡ Outpaint", style=discord.ButtonStyle.success, custom_id="outpaint_both_button", row=0)
    @universal_cooldown_check()
    async def outpaint_both_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        logger.info(f"'Outpaint Both' button clicked by {interaction.user} for prompt: '{self.prompt}'")
        try:
            await interaction.response.send_message("🛠️ Extending image in all directions...", ephemeral=True)
            queue_size = bot.sd_queue.qsize()
            await bot.sd_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'outpaint',
                'prompt': self.cleaned_prompt,
                'direction': 'both',
                'width': self.width,
                'height': self.height,
                'seed': self.seed,
                'strength': 0.8,  # Higher for outpaint
            })
            logger.info(f"Enqueued 'Outpaint Both' action for {interaction.user}: prompt='{self.cleaned_prompt}', direction='both'")
        except Exception as e:
            logger.error(f"Error during outpainting in all directions for {interaction.user}: {e}")
            await interaction.followup.send("❌ Error extending image in all directions.", ephemeral=True)


async def generate_sd_image(
    interaction,
    prompt,
    width,
    height,
    seed,
    action_name="SD",
    queue_size=0,
    pre_duration=0,
    selected_terms: Optional[str] = None  # New parameter
):
    try:
        # Check if we need to send an initial response
        if not interaction.response.is_done():
            await interaction.response.defer(thinking=True)

        sd_server_url = SD_SERVER_URL.rstrip('/')  # Ensure no trailing slash
        num_steps = int(os.getenv("SD_STEPS", 20))
        guidance = float(os.getenv("SD_GUIDANCE", 7.5))
        negative_prompt = os.getenv("SD_NEGATIVE_PROMPT", "") or ""
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": str(num_steps),
            "guidance_scale": str(guidance),
            "width": str(width),
            "height": str(height),
            "seed": str(seed)
        }

        # Use typing context manager for consistent behavior
        async with interaction.channel.typing():
            # Use optimized session with connection pooling and keep-alive
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                keepalive_timeout=30,  # Keep connections alive
                enable_cleanup_closed=True
            )
            # Increased timeout for SD 3.5 Medium on Mac (can take 2-5 minutes)
            timeout = aiohttp.ClientTimeout(total=600, connect=10)
            
            async with aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive'}
            ) as session:
                # Start timing the image generation process
                image_start_time = time.perf_counter()

                async with session.post(f"{sd_server_url}/sd", data=payload) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '').lower()
                        
                        # Handle different response formats
                        if content_type.startswith('application/json'):
                            # API might return JSON with base64 image
                            try:
                                data = await response.json()
                                if 'image' in data:
                                    import base64
                                    if isinstance(data['image'], str):
                                        image_bytes = base64.b64decode(data['image'])
                                    else:
                                        image_bytes = data['image']
                                elif 'image_bytes' in data:
                                    image_bytes = data['image_bytes']
                                else:
                                    raise ValueError(f"JSON response missing image data: {list(data.keys())}")
                                logger.debug(f"🖼️ Received image from JSON response: {len(image_bytes)} bytes")
                            except Exception as json_error:
                                logger.error(f"🖼️ Failed to parse JSON response: {json_error}")
                                error_text = await response.text()
                                raise ValueError(f"SD server returned JSON but couldn't parse: {error_text[:200]}")
                        elif content_type.startswith('image/'):
                            # Direct image response
                            image_bytes = await response.read()
                            logger.debug(f"🖼️ Received image bytes: {len(image_bytes)} bytes")
                        else:
                            # Unknown format, try to read as image anyway
                            logger.warning(f"🖼️ Unknown content type '{content_type}', attempting to read as image")
                            image_bytes = await response.read()
                        
                        # Validate we got actual image data
                        if not image_bytes or len(image_bytes) < 100:
                            raise ValueError(f"Received invalid image data: {len(image_bytes) if image_bytes else 0} bytes")
                        
                        # Try to validate it's actually an image by checking magic bytes
                        if not (image_bytes.startswith(b'\x89PNG') or image_bytes.startswith(b'\xff\xd8\xff') or image_bytes.startswith(b'GIF')):
                            logger.warning(f"🖼️ Image data doesn't start with PNG/JPEG/GIF magic bytes, but continuing anyway")
                        
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

                        description_embed = discord.Embed(
                            description=description_content, color=discord.Color.blue()
                        )
                        details_embed = discord.Embed(color=discord.Color.green())

                        queue_total = queue_size + 1
                        details_text = f"🌱 {seed} 🔄 {action_name} ⏱️ {total_duration:.2f}s 📋 {queue_total}"

                        # Change this line to set the description instead of adding a field
                        details_embed.description = details_text

                        # Initialize the SDRemixView with current image parameters
                        new_view = SDRemixView(
                            prompt=prompt, width=width, height=height, seed=seed
                        )

                        # When sending the final message, use followup if the initial response was deferred
                        if interaction.response.is_done():
                            # Force non-ephemeral followup to ensure visibility in channel; fall back to channel.send on error
                            try:
                                await interaction.followup.send(
                                    content=f"{interaction.user.mention} 🖼️ Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view,
                                    ephemeral=False
                                )
                            except Exception as send_error:
                                logger.error(f"❌ Failed to send follow-up message (falling back to channel.send): {send_error}")
                                await interaction.channel.send(
                                    content=f"{interaction.user.mention} 🖼️ Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view
                                )
                        else:
                            msg = await interaction.channel.send(
                                content=f"{interaction.user.mention} 🖼️ Generated Image:",
                                embeds=[description_embed, details_embed],
                                file=image_file,
                                view=new_view
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
                                event_type="image_generation"
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
                    ephemeral=True
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

        if size == 'wide':
            width, height = SD_WIDE_WIDTH, SD_WIDE_HEIGHT
        elif size == 'tall':
            width, height = SD_TALL_WIDTH, SD_TALL_HEIGHT
        elif size == 'square':
            width, height = SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        logger.info(f"Processing request: user={interaction.user}, prompt='{description}', size='{size}', dims={width}x{height}, seed={seed}")
        
        # Update image generation count with server ID
        await increment_user_stat(interaction.user.id, 'images_generated', interaction.guild_id)

        await generate_sd_image(interaction, description, width, height, seed, queue_size=bot.sd_queue.qsize())
    except Exception as e:
        try:
            await interaction.followup.send(f"❌ An error occurred: {str(e)}", ephemeral=True)
        except Exception as send_error:
            logger.error(f"❌ Failed to send follow-up message: {send_error}")
        logger.error(f"Error in process_sd_image: {e}")



def should_randomly_respond(probability=None) -> bool:
    """
    Returns True with the given probability (default from RANDOM_RESPONSE_RATE env, or 5%).
    """
    if probability is None:
        probability = float(os.getenv("RANDOM_RESPONSE_RATE", "0.05"))
    return random.random() < probability


"""
---------------------------------------------------------------------------------
Event Handlers
---------------------------------------------------------------------------------
"""

# Add this BEFORE your bot.event decorators and command definitions
async def load_extensions():
    """Load all extension cogs"""
    current_dir = Path(__file__).parent
    
    # Load extensions
    try:
        await bot.load_extension("soupy_search")
        logger.info("✅ Loaded search extension")
    except Exception as e:
        logger.error(f"❌ Failed to load search extension: {e}")

    try:
        await bot.load_extension("soupy_imagesearch")
        logger.info("🖼️ Loaded image search extension")
    except Exception as e:
        logger.error(f"❌ Failed to load image search extension: {e}")

    try:
        await bot.load_extension("soupy_dailypost")
        logger.info("📰 Loaded daily post extension")
    except Exception as e:
        logger.error(f"❌ Failed to load daily post extension: {e}")

    try:
        await bot.load_extension("soupy_musings")
        logger.info("💭 Loaded musings extension")
    except Exception as e:
        logger.error(f"❌ Failed to load musings extension: {e}")

    try:
        await bot.load_extension("soupy_bluesky")
        logger.info("🦋 Loaded Bluesky engagement extension")
    except Exception as e:
        logger.error(f"❌ Failed to load Bluesky extension: {e}")

# Then your existing on_ready event can use it
@bot.event
async def on_ready():
    global bot_start_time
    bot_start_time = datetime.utcnow()
    bot._timer_state = timer_state  # Expose to cogs for dashboard

    # Load extensions
    await load_extensions()
    
    # Existing on_ready operations
    logger.info(f"Logged in as {bot.user.name}")
    logger.info("Bot is ready for commands!")
    logger.info(f'🔵 Bot ready: {bot.user} (ID: {bot.user.id})')

    # Start background tasks immediately so messages are processed during command sync
    bot.loop.create_task(bot.sd_queue.process_queue())
    bot.loop.create_task(scan_trigger_loop(bot))
    bot.loop.create_task(archive_auto_scan_loop(bot))
    bot.loop.create_task(rag_reindex_loop(bot))
    bot.loop.create_task(_dashboard_status_writer(bot))
    if is_self_md_enabled():
        bot.loop.create_task(_self_md_reflection_loop(bot))

    # Sync slash commands
    # Sync to guild if GUILD_ID is set (faster updates), otherwise sync globally
    # Do NOT sync both - that causes duplicate commands
    try:
        # List all registered commands for debugging
        registered_commands = [cmd.name for cmd in bot.tree.get_commands()]
        logger.info(f"Registered commands: {registered_commands}")
        
        guild_id_str = os.getenv("GUILD_ID")
        if guild_id_str:
            try:
                guild_id = int(guild_id_str)
                guild_obj = discord.Object(id=guild_id)
                logger.info(f"Syncing commands to guild {guild_id} (faster sync, guild-only)...")
                bot.tree.copy_global_to(guild=guild_obj)
                synced = await bot.tree.sync(guild=guild_obj)
                synced_names = [cmd.name for cmd in synced]
                logger.info(f"✅ Synced {len(synced)} commands to guild {guild_id}: {synced_names}")
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not sync to guild {guild_id_str}: {e}")
                logger.info("Falling back to global sync...")
                synced = await bot.tree.sync()
                synced_names = [cmd.name for cmd in synced]
                logger.info(f"✅ Synced {len(synced)} commands globally: {synced_names}")
        else:
            logger.info("No GUILD_ID specified, syncing commands globally (may take up to 1 hour)...")
            synced = await bot.tree.sync()
            synced_names = [cmd.name for cmd in synced]
            logger.info(f"✅ Synced {len(synced)} commands globally: {synced_names}")
    except Exception as e:
        logger.error(f"Error syncing commands: {e}")
    

    # Set the bot start time
    bot_start_time = datetime.utcnow()
    logger.info(f"Bot start time set to {bot_start_time} UTC")


async def _dashboard_status_writer(bot_instance):
    """Write bot status to data/bot_dashboard.json every 15 seconds for the web panel."""
    await bot_instance.wait_until_ready()
    status_path = Path("data") / "bot_dashboard.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    while not bot_instance.is_closed():
        try:
            from datetime import timezone as _tz
            now = datetime.now(_tz.utc)
            uptime_sec = None
            if bot_start_time:
                bst = bot_start_time.replace(tzinfo=_tz.utc) if bot_start_time.tzinfo is None else bot_start_time
                uptime_sec = int((now - bst).total_seconds())

            payload = {
                "model": os.getenv("LOCAL_CHAT", ""),
                "uptime_seconds": uptime_sec,
                "sd_online": sd_server_online,
                "llm_online": chat_functions_online,
                "timers": timer_state,
                "updated_at": now.isoformat(),
            }
            status_path.write_text(json.dumps(payload, default=str) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.debug("dashboard status write failed: %s", exc)
        await asyncio.sleep(15)


async def scan_trigger_loop(bot):
    """Periodically check for scan trigger files and process them."""
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            await process_scan_triggers(bot)
        except Exception as e:
            logger.error(f"Error in scan trigger loop: {e}")
        # Check every 5 seconds
        await asyncio.sleep(5)


async def archive_auto_scan_loop(bot):
    """
    When a guild has archive_scan_interval_minutes > 0, start an incremental scan
    if the last successful scan is older than that interval (or never scanned).
    Successful scans trigger RAG reindex inside run_scan_background.
    """
    await bot.wait_until_ready()
    from datetime import datetime, timezone

    from soupy_database.database import (
        active_scans,
        get_archive_scan_interval_minutes,
        get_db_path,
        get_last_scan_time,
        trigger_scan_programmatic,
    )

    try:
        poll_sec = int(os.getenv("ARCHIVE_AUTO_SCAN_POLL_SECONDS", "45"))
    except ValueError:
        poll_sec = 45

    logger.info(
        "Archive auto-scan scheduler started (poll every %ss; set minutes per server in dashboard)",
        poll_sec,
    )
    timer_state["archive_scan"]["interval"] = f"{poll_sec}s poll"
    timer_state["archive_scan"]["enabled"] = True

    while not bot.is_closed():
        try:
            now = datetime.now(timezone.utc)
            timer_state["archive_scan"]["next_run"] = (now + timedelta(seconds=poll_sec)).isoformat()
            for guild in list(bot.guilds):
                gid = guild.id
                try:
                    minutes = get_archive_scan_interval_minutes(gid)
                    if minutes <= 0:
                        continue
                    if not os.path.exists(get_db_path(gid)):
                        continue
                    task = active_scans.get(gid)
                    if task is not None and not task.done():
                        continue
                    last = get_last_scan_time(gid)
                    if last is not None:
                        elapsed = (now - last).total_seconds()
                        if elapsed < minutes * 60:
                            continue
                    ok, msg = await trigger_scan_programmatic(bot, gid)
                    if ok:
                        timer_state["archive_scan"]["last_run"] = datetime.now(timezone.utc).isoformat()
                        logger.info(
                            "Scheduled archive scan started for guild %s (%s)",
                            gid,
                            guild.name,
                        )
                    else:
                        logger.debug(
                            "Scheduled archive scan not started for guild %s: %s",
                            gid,
                            msg,
                        )
                except Exception as inner_e:
                    logger.warning(
                        "Scheduled archive scan check failed guild=%s: %s",
                        gid,
                        inner_e,
                    )
            timer_state["archive_scan"]["last_run"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.error("archive_auto_scan_loop error: %s", e, exc_info=True)
        await asyncio.sleep(poll_sec)


async def rag_reindex_loop(bot):
    """
    Periodically consolidate single-message RAG chunks (from real-time indexing) into
    proper grouped conversation chunks. Does not re-embed already-grouped chunks.
    Interval set by RAG_REINDEX_INTERVAL_HOURS (default 6). Set to 0 to disable.
    Waits one full interval before the first run so startup isn't hammered.
    """
    await bot.wait_until_ready()
    from soupy_database.rag import index_new_messages
    from soupy_database.database import get_db_path

    try:
        interval_hours = float(os.getenv("RAG_REINDEX_INTERVAL_HOURS", "6"))
    except ValueError:
        interval_hours = 6.0

    if interval_hours <= 0:
        logger.info("RAG consolidation loop disabled (RAG_REINDEX_INTERVAL_HOURS=0)")
        timer_state["rag_reindex"]["enabled"] = False
        return

    interval_sec = interval_hours * 3600
    timer_state["rag_reindex"]["interval"] = f"{interval_hours}h"
    timer_state["rag_reindex"]["enabled"] = True
    timer_state["rag_reindex"]["next_run"] = (datetime.now(timezone.utc) + timedelta(seconds=interval_sec)).isoformat()
    logger.info("RAG incremental index loop started (every %.1fh)", interval_hours)

    while not bot.is_closed():
        await asyncio.sleep(interval_sec)
        timer_state["rag_reindex"]["last_run"] = datetime.now(timezone.utc).isoformat()
        timer_state["rag_reindex"]["next_run"] = (datetime.now(timezone.utc) + timedelta(seconds=interval_sec)).isoformat()
        for guild in list(bot.guilds):
            gid = guild.id
            if not os.path.exists(get_db_path(gid)):
                continue
            try:
                result = await index_new_messages(gid)
                if result.get("new_messages", 0) > 0 or result.get("consolidated", 0) > 0:
                    logger.info(
                        "RAG incremental index guild=%s (%s): %s",
                        gid, guild.name, result,
                    )
            except Exception as exc:
                logger.warning(
                    "RAG incremental index failed guild=%s: %s", gid, exc
                )


async def _self_md_reflection_loop(bot_instance):
    """Periodically reflect on accumulated interactions and update per-guild SELF.MD files.

    Interval set by SELF_MD_REFLECT_INTERVAL_HOURS (default 3).
    Only triggers when at least SELF_MD_MIN_INTERACTIONS notable interactions have
    accumulated for a guild.
    """
    await bot_instance.wait_until_ready()

    try:
        interval_hours = float(os.getenv("SELF_MD_REFLECT_INTERVAL_HOURS", "24"))
    except ValueError:
        interval_hours = 3.0

    try:
        min_interactions = int(os.getenv("SELF_MD_MIN_INTERACTIONS", "3"))
    except ValueError:
        min_interactions = 3

    interval_sec = interval_hours * 3600
    # Short initial delay (10 min) so the first reflection can happen soon after
    # the bot has had a few conversations, then switch to the full interval.
    first_delay = min(600, interval_sec)
    timer_state["self_reflect"]["interval"] = f"{interval_hours}h"
    timer_state["self_reflect"]["enabled"] = True
    timer_state["self_reflect"]["next_run"] = (datetime.now(timezone.utc) + timedelta(seconds=first_delay)).isoformat()
    logger.info("SELF.MD reflection loop started (first check in %.0fs, then every %.1fh, min %d interactions)", first_delay, interval_hours, min_interactions)

    await asyncio.sleep(first_delay)

    while not bot_instance.is_closed():
        timer_state["self_reflect"]["next_run"] = (datetime.now(timezone.utc) + timedelta(seconds=interval_sec)).isoformat()
        for guild in list(bot_instance.guilds):
            gid = guild.id
            count = pending_interaction_count(gid)
            if count < min_interactions:
                continue
            try:
                timer_state["self_reflect"]["last_run"] = datetime.now(timezone.utc).isoformat()
                logger.info(
                    "SELF.MD reflecting for guild %s (%s) — %d interactions pending",
                    gid, guild.name, count,
                )
                from soupy_database.rag import embed_texts_lm_studio
                async with aiohttp.ClientSession() as embed_session:
                    await self_md_reflect(
                        guild_id=gid,
                        llm_func=async_chat_completion,
                        model=os.getenv("LOCAL_CHAT"),
                        embed_func=embed_texts_lm_studio,
                        embed_session=embed_session,
                    )
            except Exception as exc:
                logger.warning("SELF.MD reflection failed guild=%s: %s", gid, exc)

        await asyncio.sleep(interval_sec)


# Regex to capture a bot-like name prefix at the start (short word/name followed by colon)
# Only matches if it's a short prefix (1-20 chars) that looks like a name/username
# Handles optional quotes around the username::
# Usernames can contain letters, numbers, underscores, and up to 2 spaces (max 3 words)
# This prevents matching things like "The time is 3:" while still catching display names
remove_all_before_colon_pattern = re.compile(r'^["\'"]?[a-zA-Z0-9_]+(?:\s[a-zA-Z0-9_]+){0,2}["\'"]?:\s*', re.IGNORECASE)

def remove_all_before_colon(text: str) -> str:
    """
    Removes a bot-like name prefix from the start of the text if present.
    Only removes short prefixes (1-20 chars) that look like usernames/bot names.
    Handles both quoted and unquoted prefixes.
    Example: "soupy: hello there" -> "hello there"
    Example: "Alice: what's up" -> "what's up"
    Example: "'glenn': hello" -> "hello"
    Example: "The time is 3:45 PM" -> "The time is 3:45 PM" (not removed, not at start)
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

async def process_chat_message(message: discord.Message, image_descriptions: list):
    """
    Process a chat message and generate a response.
    This function is called from the queue processor to handle chat messages.
    """
    logger.info(f"📝 Processing queued chat message from {message.author}: '{message.content}'")
    
    async with message.channel.typing():
        try:
            # Get guild-specific behavior
            guild_id = str(message.guild.id) if message.guild else None
            chatgpt_behaviour = await get_guild_behaviour(guild_id)
            
            # Add technical clarifications about message format
            technical_instructions = (
                "\n\nIMPORTANT - Message Format:\n"
                "- Messages from YOU (the assistant) appear without a name prefix\n"
                "- Messages from USERS appear with their nickname/display name prefix (e.g., 'Alice: hello' or 'Bob: what's up')\n"
                "- When responding, do NOT include your name or any prefix - just respond naturally\n"
                "- NEVER start your response with another user's name or prefix\n"
                "- NEVER impersonate other users by prefixing your response with their name\n"
                "- Pay attention to which user said what based on their nickname prefix\n"
                "- You are responding to the conversation, tailored to the specific users involved\n"
                "- When referencing users, use their nicknames as shown in the message prefixes"
            )
            if message.guild and is_rag_enabled():
                technical_instructions += (
                    "\n\nLONGER-TERM SERVER MEMORY:\n"
                    "A user message may begin with \"Below are snippets from earlier messages in this server\". "
                    "Those snippets are retrieved history (past messages, links, topics, personal details about users), "
                    "not live chat. Use them as factual context only.\n\n"
                    "How to use the memory block:\n"
                    "- If the user asks a direct factual question (\"do I own a cat?\", \"where do I live?\", \"what games do I play?\", \"did I say X?\") "
                    "and the answer is in the snippets, ANSWER IT directly using that information, in character. A direct question deserves a direct answer.\n"
                    "- For casual chat, the memory block is background. Drop in ONE relevant detail if it fits naturally. Otherwise ignore it.\n"
                    "- If the snippets contradict your own earlier assistant lines about what people said, trust the snippets.\n"
                    "- The BEHAVIOUR rule about prioritizing recent messages sets weight for what the moment is about and who you are answering — "
                    "it does not mean pretend the memory block does not exist when the user is clearly asking about history.\n"
                    "- If nothing in the block fits the moment, ignore it entirely and just respond to the live thread.\n\n"
                    "How NOT to use the memory block:\n"
                    "- Never recite, list, or summarize what is in it. You are not a search engine — you are a person who remembers things.\n"
                    "- Never say \"according to the archives\", \"based on what I found\", \"the records show\", or anything similar.\n"
                    "- Never output raw metadata: no timestamps, no '---' separators, no 'message_id=', no '#channel-name', no '[Embed Title:', no YYYY-MM-DD dates. If any of this appears in your reply, it is broken.\n"
                    "- Never mention archives, RAG, retrieval, databases, embeddings, or that you were given snippets.\n"
                    "- Never fabricate personal stories or attribute opinions to yourself based on the snippets.\n"
                    "- Do not tell the user to dig through old messages.\n"
                    "- Do not say someone never said much about a topic if the snippets show otherwise.\n\n"
                    "All BEHAVIOUR rules above (lower case, no quotation marks, length caps, one topic, one person, no questions, etc.) still govern HOW you write the answer.\n"
                )

            # Inject SELF.MD (accumulated self-knowledge) between behaviour and technical instructions
            self_md_block = ""
            if message.guild and is_self_md_enabled():
                self_md_block = get_self_md_for_injection(message.guild.id)

            # Create the messages list starting with the appropriate behavior prompt
            messages_for_llm = [{"role": "system", "content": chatgpt_behaviour + self_md_block + technical_instructions}]

            # --- Token budget ---
            # Calculate how many tokens are available for variable content after reserving space for
            # the system message, output tokens, and a safety buffer.
            _n_ctx = int(os.getenv("CONTEXT_WINDOW_TOKENS", "16000"))
            _output_reserve = max_tokens_default
            _safety_buffer = int(os.getenv("CONTEXT_SAFETY_BUFFER_TOKENS", "400"))
            _system_tokens = estimate_tokens(chatgpt_behaviour + self_md_block + technical_instructions) + 4
            _current_msg_tokens = estimate_tokens(
                f"{message.author.display_name}: {message.content or ''}"
            ) + 4
            _available = _n_ctx - _output_reserve - _safety_buffer - _system_tokens - _current_msg_tokens

            # Allocate the available budget across history, RAG, and URL content.
            _history_frac = float(os.getenv("CONTEXT_HISTORY_BUDGET_FRAC", "0.45"))
            _rag_frac     = float(os.getenv("CONTEXT_RAG_BUDGET_FRAC",     "0.35"))
            _url_frac     = float(os.getenv("CONTEXT_URL_BUDGET_FRAC",     "0.15"))

            _history_budget_tokens = max(200, int(_available * _history_frac))
            _rag_budget_tokens     = max(200, int(_available * _rag_frac))
            _url_budget_tokens     = max(100, int(_available * _url_frac))

            # Convert token budgets to character budgets for subsystems that work in chars.
            # Use 3.5 chars/token (same ratio as estimate_tokens, just inverted).
            _rag_budget_chars = int(_rag_budget_tokens * 3.5)
            _url_budget_chars = int(_url_budget_tokens * 3.5)

            logger.debug(
                "📊 Token budget: n_ctx=%d output=%d safety=%d system=%d cur_msg=%d available=%d "
                "→ history=%d rag=%d url=%d tokens",
                _n_ctx, _output_reserve, _safety_buffer, _system_tokens, _current_msg_tokens,
                _available, _history_budget_tokens, _rag_budget_tokens, _url_budget_tokens,
            )
            # --- end token budget ---

            # Recent channel history first: used for multi-turn RAG query and later appended to the LLM.
            recent_messages = await fetch_recent_messages(
                message.channel,
                limit=int(os.getenv("RECENT_MESSAGE_LIMIT", 25)),
                current_message_id=message.id,
            )

            # Trim history to its token budget (drops oldest messages first).
            _history_before = len(recent_messages)
            recent_messages = trim_messages_to_token_budget(recent_messages, _history_budget_tokens)
            if len(recent_messages) < _history_before:
                logger.debug(
                    "📊 History trimmed: %d → %d messages to fit %d-token history budget",
                    _history_before, len(recent_messages), _history_budget_tokens,
                )

            # Fetch RAG here but append later (immediately before the current user turn) so it wins over
            # recent assistant replies that contradict retrieved snippets—BEHAVIOUR often says "last two messages only"
            # and earlier Soupy lines like "never said much" would otherwise override good retrieval.
            rag_context_message = None
            # RAG runs on every guild chat reply when the WebUI master switch is on (no RAG keyword required).
            _rag_raw = (message.content or "").strip()
            if message.guild and is_rag_enabled() and _rag_raw:
                _fp_cur = strip_rag_gate_word(message.content or "").strip()
                _fp_cur = (
                    strip_rag_query_invocations(_fp_cur).strip()
                    if _fp_cur
                    else ""
                )
                if not _fp_cur:
                    _fp_cur = _rag_raw
                rag_query = build_rag_retrieval_query(
                    recent_messages,
                    message.author.display_name,
                    message.content or "",
                )
                if not rag_query.strip():
                    rag_query = strip_rag_gate_word(message.content or "") or (message.content or "")
                try:
                    async with aiohttp.ClientSession() as rag_session:
                        rag_block = await fetch_rag_context_for_query(
                            rag_session,
                            int(message.guild.id),
                            int(message.author.id),
                            rag_query,
                            first_person_hint=_fp_cur,
                            max_chars_override=_rag_budget_chars,
                        )
                    if rag_block:
                        rag_context_message = {
                            "role": "user",
                            "content": (
                                f"{RAG_CONTEXT_MESSAGE_SENTINEL}; they may be incomplete or outdated. "
                                "If your own earlier assistant lines in this thread contradict these snippets about what people said, trust the snippets. "
                                "Use the LONGER-TERM SERVER MEMORY rules in the system prompt for how to use this block.\n\n"
                                + rag_block
                            ),
                        }
                        logger.info(
                            "  → injected %d chars into context (budget was %d chars)",
                            len(rag_block), _rag_budget_chars,
                        )
                except Exception as rag_exc:
                    logger.warning("RAG retrieval skipped: %s", format_error_message(rag_exc))

            # Add conversation history (same fetch as used for RAG query context)
            messages_for_llm.extend(recent_messages)

            # Scan recent history for URLs and inline extracted content (cached, deduped, limited).
            # All URL content (history + current message) shares _url_budget_chars so they can't
            # collectively overflow the token budget.
            try:
                max_history_urls = int(os.getenv("MAX_URLS_FROM_HISTORY", 2))
            except Exception:
                max_history_urls = 2

            _url_chars_used = 0  # running total across history + current-message URLs

            history_url_contents = []
            if max_history_urls > 0 and recent_messages:
                processed_history_urls = set()
                # Iterate from most recent to older to prioritize latest links
                async with aiohttp.ClientSession() as session:
                    for hist_msg in reversed(recent_messages):
                        content_text = hist_msg.get("content", "")
                        if not content_text:
                            continue
                        urls_in_hist = extract_urls(content_text)
                        if not urls_in_hist:
                            continue
                        for url in urls_in_hist:
                            if url in processed_history_urls:
                                continue
                            processed_history_urls.add(url)
                            try:
                                current_time = time.time()
                                # Cache check
                                if url in url_cache:
                                    cached_content, timestamp = url_cache[url]
                                    if current_time - timestamp < URL_CACHE_TTL:
                                        if cached_content:
                                            remaining_url_budget = _url_budget_chars - _url_chars_used
                                            chunk = cached_content[:remaining_url_budget]
                                            if chunk:
                                                history_url_contents.append(chunk)
                                                _url_chars_used += len(chunk)
                                                logger.debug(f"Using cached content for HISTORY URL: {url}")
                                        else:
                                            logger.debug(f"Cached HISTORY URL {url} had no content")
                                        if len(history_url_contents) >= max_history_urls or _url_chars_used >= _url_budget_chars:
                                            break
                                        continue

                                # Fetch new
                                logger.debug(f"Fetching fresh content for HISTORY URL: {url}")
                                content = await extract_url_content(url, session)
                                url_cache[url] = (content, current_time)
                                if content:
                                    remaining_url_budget = _url_budget_chars - _url_chars_used
                                    chunk = content[:remaining_url_budget]
                                    history_url_contents.append(chunk)
                                    _url_chars_used += len(chunk)
                                    logger.debug(f"Successfully processed HISTORY URL {url} ({len(chunk)}/{len(content)} chars kept)")
                                else:
                                    logger.warning(f"Failed to extract content from HISTORY URL {url}")

                                if len(history_url_contents) >= max_history_urls or _url_chars_used >= _url_budget_chars:
                                    break
                            except Exception as url_e:
                                logger.error(f"Error processing HISTORY URL {url}: {format_error_message(url_e)}")
                        if len(history_url_contents) >= max_history_urls or _url_chars_used >= _url_budget_chars:
                            break

            # If we gathered history URL contents, add as a user message before current message
            if history_url_contents:
                joined_history_content = " ".join(history_url_contents)
                messages_for_llm.append({
                    "role": "user",
                    "content": joined_history_content
                })
                logger.debug(f"Added {len(history_url_contents)} HISTORY URL contents into context (pre-current message)")

            # If there are image descriptions, add them to context
            if image_descriptions:
                messages_for_llm.append({
                    "role": "user",
                    "content": "\n".join(image_descriptions)
                })
                logger.debug(f"Added {len(image_descriptions)} image description(s) to context")

            # Process URLs in the CURRENT message, respecting remaining URL budget
            current_message_content = message.content
            urls_in_current_message = extract_urls(current_message_content)
            url_contents_for_current_message = []

            if urls_in_current_message and _url_chars_used < _url_budget_chars:
                logger.debug(f"Processing {len(urls_in_current_message)} URLs in CURRENT message from {message.author.display_name}: {urls_in_current_message}")

                async with aiohttp.ClientSession() as session:
                    for url in urls_in_current_message:
                        if _url_chars_used >= _url_budget_chars:
                            logger.debug(f"URL budget exhausted ({_url_chars_used}/{_url_budget_chars} chars); skipping remaining current-message URLs")
                            break
                        # Check cache first
                        current_time = time.time()
                        if url in url_cache:
                            cached_content, timestamp = url_cache[url]
                            if current_time - timestamp < URL_CACHE_TTL:
                                if cached_content:  # Only add if there was actual content
                                    remaining_url_budget = _url_budget_chars - _url_chars_used
                                    chunk = cached_content[:remaining_url_budget]
                                    url_contents_for_current_message.append(chunk)
                                    _url_chars_used += len(chunk)
                                    logger.debug(f"Using cached content for URL: {url}")
                                else:
                                    logger.debug(f"Cached URL {url} had no content")
                                continue

                        # Fetch and process URL content
                        logger.debug(f"Fetching fresh content for URL: {url}")
                        content = await extract_url_content(url, session)
                        url_cache[url] = (content, current_time)
                        if content:
                            remaining_url_budget = _url_budget_chars - _url_chars_used
                            chunk = content[:remaining_url_budget]
                            url_contents_for_current_message.append(chunk)
                            _url_chars_used += len(chunk)
                            logger.debug(f"Successfully processed URL {url} for CURRENT message ({len(chunk)}/{len(content)} chars kept)")
                        else:
                            logger.warning(f"Failed to extract content from URL {url} in CURRENT message from {message.author.display_name}")

            # Add URL contents to current message
            if url_contents_for_current_message:
                original_length = len(current_message_content)
                current_message_content = f"{current_message_content}\n{' '.join(url_contents_for_current_message)}"
                logger.debug(f"Added {len(url_contents_for_current_message)} URL contents to CURRENT message, length increased from {original_length} to {len(current_message_content)} chars")

            if _url_chars_used > 0:
                logger.info("📊 URL content total: %d chars used of %d-char budget", _url_chars_used, _url_budget_chars)
            
            # Add the current user message with URL content (using display name/nickname).
            # The marker block before the message keeps the trigger findable after the
            # consecutive-same-role merge step below glues this onto any preceding
            # URL content / RAG context / image descriptions in a single user blob.
            user_message = {
                "role": "user",
                "content": (
                    "\n\n---\n"
                    "RESPOND TO THE MESSAGE BELOW. Everything earlier in this user turn "
                    "(recent chat history, link previews, retrieved memory snippets) is "
                    "background context only — do not respond to it directly.\n"
                    "---\n"
                    f"{message.author.display_name}: {current_message_content}"
                )
            }
            if rag_context_message:
                messages_for_llm.append(rag_context_message)
            messages_for_llm.append(user_message)

            # Merge consecutive same-role messages so models with strict
            # user/assistant alternation templates don't reject the prompt.
            merged: list = []
            for msg_item in messages_for_llm:
                if merged and merged[-1]["role"] == msg_item["role"]:
                    merged[-1]["content"] += "\n" + msg_item["content"]
                else:
                    merged.append(dict(msg_item))

            # Ensure first non-system message is "user" — some models (Gemma)
            # require system → user → assistant → user... and reject system → assistant.
            # If the first non-system message is assistant, fold it into the system prompt.
            if len(merged) >= 2 and merged[0]["role"] == "system" and merged[1]["role"] == "assistant":
                merged[0]["content"] += "\n\n[Previous bot response for context]\n" + merged[1]["content"]
                merged.pop(1)
                logger.debug("📜 Folded leading assistant message into system prompt (Gemma compatibility)")

            # If after folding the pattern is system → assistant again (e.g. multiple
            # assistant messages got merged), keep folding until we hit a user.
            while len(merged) >= 2 and merged[0]["role"] == "system" and merged[1]["role"] == "assistant":
                merged[0]["content"] += "\n" + merged[1]["content"]
                merged.pop(1)

            if len(merged) < len(messages_for_llm):
                logger.debug("📜 Merged %d → %d messages (consecutive same-role)",
                             len(messages_for_llm), len(merged))
            messages_for_llm = merged

            # Log what we send to the LLM (summary by default; full dump only when debugging)
            if os.getenv("SOUPY_LOG_FULL_LLM_PROMPT", "").strip() in ("1", "true", "yes"):
                logger.debug("📜 Full LLM prompt:\n%s", format_messages(messages_for_llm))
            else:
                logger.info(
                    "📜 LLM messages (%d): summary follows\n%s",
                    len(messages_for_llm),
                    summarize_messages_for_llm_log(messages_for_llm),
                )

            async with message.channel.typing():
                # Generate candidates with varied temperatures for diversity
                model_name = os.getenv("LOCAL_CHAT")
                temperature = float(os.getenv("CHAT_TEMPERATURE", 0.8))
                num_candidates = int(os.getenv("CHAT_NUM_CANDIDATES", 4))
                
                # Log temperature variations that will be used (capped at 1.0 to avoid hallucinations)
                # Generate temperature variations based on number of candidates
                temp_variations = []
                if num_candidates == 1:
                    temp_variations = [temperature]
                elif num_candidates == 2:
                    temp_variations = [temperature, max(0.3, temperature - 0.15)]
                elif num_candidates == 3:
                    temp_variations = [temperature, max(0.3, temperature - 0.15), min(1.0, temperature + 0.15)]
                else:  # 4 or more
                    base_temps = [
                        temperature,
                        max(0.3, temperature - 0.15),
                        min(1.0, temperature + 0.15),
                        min(1.0, temperature + 0.25),
                    ]
                    # If more than 4 candidates, mirror the first 4 temperatures
                    if num_candidates > 4:
                        for i in range(num_candidates - 4):
                            # Cycle through the base 4 temperature patterns
                            mirrored_temp = base_temps[i % 4]
                            base_temps.append(mirrored_temp)
                    temp_variations = base_temps[:num_candidates]
                
                logger.debug(f"🌡️ Generating {num_candidates} candidates with temperatures: {[f'{t:.2f}' for t in temp_variations]}")
                
                candidates = await generate_parallel_candidates(
                    messages=messages_for_llm,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens_default,
                    num_candidates=num_candidates
                )

                if not candidates:
                    raise RuntimeError("No candidates generated from LLM")

                # Build terminal output for archiving
                terminal_log_lines = []
                terminal_log_lines.append(f"Responding to: {message.author.display_name} (@{message.author.name})")
                terminal_log_lines.append(f"Model: {model_name}")
                terminal_log_lines.append(f"Base Temperature: {temperature:.2f}")
                terminal_log_lines.append(f"Temperature Variations: {[f'{t:.2f}' for t in temp_variations]}")
                terminal_log_lines.append("")
                
                # Add last 5 messages from chat history
                terminal_log_lines.append("Last 5 messages from context:")
                terminal_log_lines.append("-" * 50)
                recent_messages = messages_for_llm[-5:] if len(messages_for_llm) >= 5 else messages_for_llm
                for idx, msg in enumerate(recent_messages, 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    # Truncate very long messages for readability
                    if len(content) > 200:
                        content = content[:200] + "..."
                    terminal_log_lines.append(f"{idx}. [{role}] {content}")
                terminal_log_lines.append("-" * 50)
                terminal_log_lines.append("")
                
                # Log all generated candidates clearly for visibility with their temperatures
                # Generate labels dynamically based on number of candidates
                labels_for_logging = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                if num_candidates > len(labels_for_logging):
                    # If we have more candidates than letters, use numbers
                    labels_for_logging.extend([str(i) for i in range(len(labels_for_logging) + 1, num_candidates + 1)])
                
                for idx, cand in enumerate(candidates):
                    label = labels_for_logging[idx] if idx < len(labels_for_logging) else str(idx)
                    temp_used = temp_variations[idx % len(temp_variations)]
                    logger.debug(f"🧪 Candidate {label} (temp={temp_used:.2f}): {cand}")
                    terminal_log_lines.append(f"Candidate {label} (temp={temp_used:.2f}): {cand}")

                terminal_log_lines.append("")
                if len(candidates) == 1:
                    reply = candidates[0]
                    logger.debug("🧩 Only one candidate generated; using it directly.")
                    terminal_log_lines.append("Only one candidate generated; using it directly.")
                else:
                    # Ask the LLM to choose the most relevant candidate
                    logger.debug("🎯 Judging candidates for relevance and appropriateness...")
                    terminal_log_lines.append("Judging candidates for relevance and appropriateness...")
                    selected_index = await judge_best_of_candidates(messages_for_llm, candidates, model=model_name)
                    reply = candidates[selected_index]
                    label = labels_for_logging[selected_index] if selected_index < len(labels_for_logging) else str(selected_index)
                    temp_used = temp_variations[selected_index % len(temp_variations)]
                    logger.debug(f"🏁 Judge selected candidate {label} (temp={temp_used:.2f}) as most relevant response.")
                    terminal_log_lines.append(f"Judge selected candidate {label} (temp={temp_used:.2f}) as most relevant response.")

            # Collapse all newlines and excess whitespace into a single line.
            reply = " ".join(reply.split())

            logger.info(f"🤖 Generated reply for {message.author}: '{reply}'")

            # Update chat response count with server ID
            await increment_user_stat(message.author.id, 'chat_responses', message.guild.id)

            # Split the bot's entire reply into smaller chunks
            chunks = split_message(reply, max_len=1500)

            terminal_output_text = "\n".join(terminal_log_lines)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Remove everything before the first colon
                cleaned_chunk = remove_all_before_colon(chunk)
                # Now remove any surrounding quotation marks
                cleaned_chunk = clean_response(cleaned_chunk)
                logger.debug(f"✂️ Sending cleaned chunk to {message.channel}: '{cleaned_chunk}'")
                out_msg = await message.channel.send(cleaned_chunk)
                try:
                    # Only include terminal output on the first chunk
                    archive_sent_message(
                        content=cleaned_chunk,
                        user_id=bot.user.id if bot.user else 0,
                        username=str(bot.user) if bot.user else "soupy",
                        guild_id=(message.guild.id if message.guild else None),
                        channel_id=(message.channel.id if message.channel else None),
                        terminal_output=terminal_output_text if chunk_idx == 0 else None,
                    )
                except Exception:
                    pass
                await asyncio.sleep(RATE_LIMIT)
            logger.info(f"✅ Successfully sent reply to {message.author}")

            # Accumulate interaction for SELF.MD reflection
            if message.guild and is_self_md_enabled():
                try:
                    # Include brief conversation context so reflection knows the broader topic
                    _ctx_lines = []
                    for _rm in recent_messages[-6:]:  # last few messages for context
                        _rc = (_rm.get("content") or "")[:200]
                        if _rc:
                            _ctx_lines.append(_rc)
                    _context_hint = "\n".join(_ctx_lines) if _ctx_lines else ""
                    await add_notable_interaction(
                        guild_id=message.guild.id,
                        user_display_name=message.author.display_name,
                        user_message=message.content or "",
                        bot_reply=reply,
                        conversation_context=_context_hint,
                    )
                except Exception:
                    pass  # never let self-context tracking break chat
        except Exception as e:
            logger.error(f"❌ Error generating AI response for {message.author}: {format_error_message(e)}")


async def _index_message_realtime(message: discord.Message) -> None:
    """Store and embed a plain-text message immediately into the guild RAG index."""
    try:
        from soupy_database.rag import index_message_immediate
        msg_dt = message.created_at  # discord.py 2.x: always UTC-aware
        await index_message_immediate(
            guild_id=message.guild.id,
            message_id=message.id,
            message_content=message.content,
            username=str(message.author),
            nickname=message.author.display_name if message.author.display_name != str(message.author) else None,
            user_id=message.author.id,
            channel_id=message.channel.id,
            channel_name=getattr(message.channel, "name", str(message.channel.id)),
            msg_date=msg_dt.strftime("%Y-%m-%d"),
            msg_time=msg_dt.strftime("%H:%M:%S"),
        )
    except Exception as exc:
        logger.debug("Real-time message indexing failed msg=%s: %s", message.id, exc)


@bot.event
async def on_message(message):
    # Clear previous image descriptions at the start of each message
    global image_descriptions, message_image_descriptions
    image_descriptions = []

    # Skip bot messages
    if message.author == bot.user:
        return

    # Process commands first
    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.process_commands(message)
        return

    # Process any image attachments
    if message.attachments:
        for attachment in message.attachments:
            description = await process_image_attachment(attachment, message)
            if description:
                # Format the description to include context
                formatted_desc = f"[Image shared by {message.author.display_name}: {description}]"
                image_descriptions.append(formatted_desc)
                
                # Store persistently for this message
                if message.id not in message_image_descriptions:
                    message_image_descriptions[message.id] = []
                message_image_descriptions[message.id].append(formatted_desc)

    # Real-time RAG indexing for plain text messages (no attachments, no URLs).
    # Image/URL messages are left for the scheduled scan which handles LLM enrichment.
    _excluded_scan_ids = [
        int(c.strip()) for c in os.getenv("SCAN_EXCLUDE_CHANNEL_IDS", "").split(",")
        if c.strip().isdigit()
    ]
    if (
        message.guild
        and message.content
        and not message.attachments
        and not extract_urls(message.content)
        and message.channel.id not in _excluded_scan_ids
    ):
        asyncio.create_task(_index_message_realtime(message))

    # Continue with existing message handling
    should_respond = should_bot_respond_to_message(message) or should_randomly_respond()
    if should_respond:
        logger.info(f"📝 Queuing chat message from {message.author}: '{message.content}'")
        
        # Queue the chat message for processing
        # Note: image_descriptions are captured synchronously above and passed to the queue
        await bot.sd_queue.put({
            'type': 'chat',
            'message': message,
            'image_descriptions': image_descriptions.copy()  # Copy the list to avoid issues
        })
        logger.info(f"📝 Queued chat message for {message.author}: description='{message.content}', queue_size={bot.sd_queue.qsize()}")

    # Process commands again if needed
    await bot.process_commands(message)





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


