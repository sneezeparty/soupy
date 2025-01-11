"""
Soupy - A versatile Discord bot
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
import logging
import discord
import openai
import random
import time
import asyncio
import aiohttp
import io
import requests
import base64
import json
import re
import pysolr
import glob
import datetime
import pytz
import tiktoken
import subprocess
from logging.handlers import RotatingFileHandler
from functools import partial
from collections import defaultdict
from asyncio import Queue
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from openai import OpenAI
from openai import OpenAIError
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from colorama import init, Fore, Style
from discord.ext import commands
from discord.ui import View, Button, Modal, TextInput
from discord import app_commands, TextStyle
from dotenv import load_dotenv

# Initialize colorama for colored console output in the terminal
init(autoreset=True)

# --- Colored Formatter for Console Logging ---
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"

# --- JSON Formatter for File Logging ---
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'line': record.lineno,
            'function': record.funcName,
            'module': record.module,
            'thread': record.threadName,
            'process': record.processName,
        }
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# --- Plaintext Formatter for File Logging ---
class PlaintextFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record)

# --- Initialize the Main Logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all log levels

# --- Console Handler with ColoredFormatter ---
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set to INFO or DEBUG as needed

# Create and set the colored formatter for console
console_formatter = ColoredFormatter('%(asctime)s - [%(levelname)s] - %(message)s')
console_handler.setFormatter(console_formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)



# --- File Handler with JSONFormatter ---
# Ensure the 'logs' directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure RotatingFileHandler for JSON file logging
json_file_handler = RotatingFileHandler(
    'logs/bot.json',          # JSON log file
    maxBytes=10*1024*1024,    # 10 MB per log file
    backupCount=10,            # Keep up to 5 backup log files
    encoding='utf-8'
)
json_file_handler.setLevel(logging.DEBUG)  # Log all levels to the JSON file

# Create and set the JSON formatter for JSON file handler
json_formatter = JSONFormatter()
json_file_handler.setFormatter(json_formatter)

# Add the JSON file handler to the logger
logger.addHandler(json_file_handler)
# --- End of JSON File Handler ---

# --- File Handler with PlaintextFormatter ---
# Configure RotatingFileHandler for plaintext file logging
plaintext_file_handler = RotatingFileHandler(
    'logs/bot.log',           # Plaintext log file
    maxBytes=5*1024*1024,     # 5 MB per log file
    backupCount=5,            # Keep up to 5 backup log files
    encoding='utf-8'
)
plaintext_file_handler.setLevel(logging.DEBUG)  # Log all levels to the plaintext file

# Create and set the plaintext formatter for plaintext file handler
plaintext_formatter = PlaintextFormatter('%(asctime)s - %(levelname)s - %(message)s')
plaintext_file_handler.setFormatter(plaintext_formatter)

# Add the plaintext file handler to the logger
logger.addHandler(plaintext_file_handler)
# --- End of Plaintext File Handler ---

# --- Example Usage ---
if __name__ == "__main__":
    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")


# Load environment variables (like API keys) from .env file
load_dotenv()

# Connect to Solr instance for indexing messages
solr_url = 'http://localhost:8983/solr/soupy'
logger.info(f"Attempting to connect to Solr at {solr_url}")

# Initialize the tokenizer with the appropriate encoding for the model
tokenizer = tiktoken.get_encoding('cl100k_base')  # Use the appropriate encoding for your model

# Initialize a cache to store user messages temporarily
user_message_cache = defaultdict(list)

task_queue = Queue()

# Generate a unique message ID using channel ID and timestamp in ISO 8601 format
def generate_message_id(channel_id, timestamp):
    # Use a consistent ISO 8601 format for timestamps
    formatted_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return f"{channel_id}_{formatted_timestamp}"


# Attempt to connect to the Solr instance and verify connectivity
try:
    solr = pysolr.Solr(solr_url, timeout=10)
    # Optionally, perform a simple query to check if Solr is responding
    solr.ping()
    logger.info("Successfully connected to Solr.")
except Exception as e:
    logger.error(f"Failed to connect to Solr. Error: {e}")


# Display an image in the iTerm2 terminal using the 'imgcat' command -- not currently functional
def display_image_iterm2(image_path):
    subprocess.run(["imgcat", image_path])

# Index all JSON files in the specified directory into the Solr instance for efficient searching
def index_all_json_files(directory):
    json_files = glob.glob(f"{directory}/*.json")
    total_files = len(json_files)
    logger.info(f"Total JSON files to index: {total_files}")

    for idx, json_file_path in enumerate(json_files, start=1):
        channel_id = os.path.splitext(os.path.basename(json_file_path))[0]
        logger.info(f"Processing file {idx} of {total_files}: {json_file_path}")
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            if not data:
                logger.warning(f"No data found in file: {json_file_path}")
                continue

            new_entries = 0
            for entry in data:
                timestamp = datetime.strptime(entry['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
                entry['id'] = generate_message_id(channel_id, timestamp)  # Regenerate 'id' using the timestamp

                try:
                    solr.add([entry])
                    new_entries += 1
                except Exception as e:
                    logger.error(f"Failed to index entry from {json_file_path}. Error: {e}")
            
            solr.commit()
            if new_entries > 0:
                logger.info(f"Indexed {new_entries} entries from file: {json_file_path}")
            else:
                logger.warning(f"No new entries to index from file: {json_file_path}")

# Process and index all JSON files from the specified directory
index_all_json_files("/absolute/directory/of/your/script/")

# Commit changes to make sure data is indexed
solr.commit()

# Initialize the OpenAI client with the API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=openai_api_key)
openai.api_key = openai_api_key


# Initialize Discord bot with specific message intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Set a rate limit for message processing to prevent rapid responses
RATE_LIMIT = 0.25

# Retrieve behavior settings for ChatGPT and transformation from environment variables
chatgpt_behaviour = os.getenv("BEHAVIOUR")
transform_behaviour = os.getenv("TRANSFORM")

# Error formatting
def format_error_message(error):
    error_prefix = "Error: "  # Define a prefix for error messages
    try:
        # Handling OpenAI specific errors
        if isinstance(error, openai.OpenAIError):
            if hasattr(error, 'response') and error.response:
                try:
                    error_json = error.response.json()
                    code = error.response.status_code
                    error_details = error_json.get('error', {})
                    error_message = error_details.get('message', 'An unspecified error occurred.')
                    return f"{error_prefix}An OpenAI API error occurred: Error code {code} - {error_message}"
                except ValueError:
                    return f"{error_prefix}An OpenAI API error occurred, but the details were not in a recognizable format."
            else:
                return f"{error_prefix}An OpenAI API error occurred, but no additional details are available."
        elif hasattr(error, 'response') and error.response is not None:
            try:
                error_json = error.response.json()
                code = error.response.status_code
                return f"{error_prefix}An HTTP error occurred: Error code {code} - {error_json}"
            except ValueError:
                return f"{error_prefix}An error occurred, but the response was not in a recognizable format."
        else:
            return f"{error_prefix}{str(error)}"
    except Exception as e:
        return f"{error_prefix}An unexpected error occurred while formatting the error message: {e}"


def get_nickname(user):
    """
    Safely retrieve the nickname of a user. If the user is a Member, return their nickname
    or username if the nickname is None. If the user is a User, return their username.
    """
    if isinstance(user, discord.Member):
        nickname = user.nick or user.name
        logger.debug(f"User '{user}' is a Member. Nickname: '{nickname}'")
        return nickname
    elif isinstance(user, discord.User):
        nickname = user.name
        logger.debug(f"User '{user}' is a User. Username: '{nickname}'")
        return nickname
    else:
        logger.debug(f"User '{user}' is of an unexpected type. Returning 'Unknown User'")
        return "Unknown User"



# Define mapping from categories to profile fields
CATEGORY_TO_PROFILE_FIELDS = {
    "politics": ["political_party", "opinions_about_politics"],
    "games": ["opinions_about_games", "hobbies", "user_interests"],
    "movies": ["opinions_about_movies", "hobbies", "user_interests"],
    "music": ["opinions_about_music", "hobbies", "user_interests"],
    "television": ["opinions_about_television", "hobbies", "user_interests"],
    "life": ["opinions_about_life", "user_job_career", "user_family_friends", "user_problems"],
    "technology": ["tech_interests", "opinions_about_technology"],
    "sports": ["sports_interests", "opinions_about_sports"],
    "books": ["book_preferences", "opinions_about_books"],
    "art": ["art_interests", "opinions_about_art"],
    "health": ["health_concerns", "health_habits"],
    "science": ["science_interests", "opinions_about_science"],
    "travel": ["travel_preferences", "travel_experiences"],
    "food": ["food_preferences", "opinions_about_food"],
    # Add more mappings as needed
}

# Define synonyms for each category to enhance keyword mapping
SYNONYMS = {
    "politics": [
        "government", "policy", "election", "democracy", "legislation",
        "political system", "state", "administration", "bureaucracy",
        "public affairs", "political affairs", "political science", "governance",
        "political parties", "council", "assembly", "reform", "ideology",
        "diplomacy", "foreign policy", "campaign", "parliament", "congress",
        "senate", "house of representatives", "executive branch", "judiciary",
        "policy-making", "political debate", "political ideology", "political movement"
    ],
    "games": [
        "gaming", "videogames", "esports", "gameplay", "game development",
        "game design", "gamers", "console games", "PC games", "mobile games",
        "game genres", "interactive entertainment", "online gaming",
        "role-playing games", "strategy games", "arcade games", "multiplayer games",
        "indie games", "game mechanics", "virtual reality games", "augmented reality games",
        "game consoles", "gaming industry", "game streaming", "game tournaments"
    ],
    "movies": [
        "films", "cinema", "motion pictures", "blockbusters", "feature films",
        "indie films", "documentaries", "short films", "animated films",
        "movie industry", "filmmaking", "screening", "film festivals",
        "box office", "movie genres", "thrillers", "comedies", "dramas",
        "action films", "horror films", "science fiction movies", "romantic films",
        "biographical films", "historical movies", "fantasy films",
        "musicals", "western films", "film criticism", "movie reviews",
        "film scores", "soundtracks", "cinematography"
    ],
    "music": [
        "songs", "melody", "rhythm", "tunes", "composition", "musical pieces",
        "soundtrack", "orchestration", "beats", "musical genres", "symphony",
        "concerts", "performances", "lyrics", "instruments", "musicians",
        "bands", "albums", "singing", "music production", "recording",
        "music theory", "jazz", "rock", "pop music", "classical music",
        "hip-hop", "electronic music", "folk music", "blues", "country music",
        "R&B", "metal", "reggae", "opera", "music videos", "live performances",
        "music festivals", "music awards", "songwriting", "music education"
    ],
    "television": [
        "TV", "shows", "series", "broadcast", "television programs",
        "telenovelas", "reality shows", "sitcoms", "soap operas",
        "documentaries", "miniseries", "dramas", "comedy shows",
        "talk shows", "news programs", "television channels", "streaming shows",
        "animated series", "variety shows", "game shows", "late-night shows",
        "television specials", "TV ratings", "television networks",
        "television streaming", "premium cable shows", "public broadcasting",
        "television advertising", "episodic content", "serialized dramas",
        "TV pilots", "binge-watching", "television marathons"
    ],
    "life": [
        "existence", "living", "daily life", "lifestyle", "biography",
        "experience", "activities", "routine", "well-being", "health",
        "personal life", "work-life balance", "habits", "relationships",
        "social life", "daily routines", "personal development", "mindfulness",
        "happiness", "stress management", "personal growth", "self-improvement",
        "work", "career", "hobbies", "leisure activities", "family life",
        "friendships", "mental health", "physical health", "nutrition",
        "fitness", "sleep habits", "time management", "life goals",
        "self-care", "emotional well-being", "spirituality", "life skills",
        "travel", "adventure", "personal achievements", "challenges",
        "life lessons", "philosophy of life", "purpose", "values", "beliefs",
        "work", "education", "career development", "financial well-being"
    ],
    "technology": [
        "tech", "computers", "software", "hardware", "gadgets",
        "electronics", "AI", "artificial intelligence", "machine learning",
        "data science", "programming", "coding", "cybersecurity",
        "blockchain", "internet", "mobile technology", "cloud computing",
        "virtual reality", "augmented reality", "robotics", "automation",
        "big data", "IoT", "quantum computing", "networking", "operating systems",
        "web development", "gaming technology", "smart devices", "tech trends",
        "tech industry", "digital transformation", "startups", "innovation",
        "tech news", "software development", "app development", "tech support"
    ],
    "sports": [
        "athletics", "competitions", "teams", "players", "matches",
        "tournaments", "leagues", "coaching", "training", "fitness",
        "games", "esports", "recreational activities", "sports events",
        "sports news", "sporting goods", "sports science", "outdoor activities",
        "indoor sports", "extreme sports", "sports management",
        "sports marketing", "sports analytics", "sportsmanship",
        "fitness training", "sport psychology", "sports medicine",
        "recreational sports", "amateur sports", "professional sports",
        "sports broadcasting", "sports sponsorships", "sports fans",
        "sports culture", "sports equipment"
    ],
    "books": [
        "literature", "novels", "fiction", "non-fiction", "books",
        "publishing", "authors", "reading", "book genres", "e-books",
        "audiobooks", "book clubs", "libraries", "bookstores",
        "book reviews", "best-sellers", "classics", "biographies",
        "memoirs", "science fiction", "fantasy", "mystery", "thrillers",
        "romance", "historical fiction", "children's books", "young adult",
        "poetry", "drama", "essays", "short stories", "graphic novels",
        "self-help books", "cookbooks", "travel books", "educational books",
        "textbooks", "reference books", "technical books", "graphic literature",
        "comic books", "serialized novels", "literary criticism",
        "book awards", "book fairs", "book adaptations", "book signings"
    ],
    "art": [
        "painting", "sculpture", "drawing", "photography", "visual arts",
        "fine arts", "digital art", "street art", "graffiti", "illustration",
        "art exhibitions", "art galleries", "art history", "art education",
        "art techniques", "mixed media", "abstract art", "realism",
        "modern art", "contemporary art", "classic art", "art movements",
        "art criticism", "art restoration", "art collecting",
        "art installations", "performance art", "conceptual art",
        "art therapy", "art supplies", "art workshops", "art projects",
        "art competitions", "art festivals", "art publications",
        "art auctions", "art patrons", "art residencies", "art commissions"
    ],
    "health": [
        "wellness", "medicine", "fitness", "nutrition", "exercise",
        "healthcare", "mental health", "physical health", "diet", "workout",
        "healthcare industry", "medical research", "diseases", "preventive care",
        "health policies", "public health", "health education", "health services",
        "holistic health", "alternative medicine", "medical treatments",
        "health tips", "health news", "health trends", "health technology",
        "health supplements", "chronic illness", "health programs",
        "health initiatives", "health advocacy", "health legislation",
        "health awareness"
    ],
    "science": [
        "research", "experiments", "biology", "chemistry", "physics",
        "astronomy", "geology", "environmental science", "ecology",
        "genetics", "space exploration", "scientific discoveries",
        "science news", "science education", "laboratory work",
        "scientific methods", "scientific theories", "quantum physics",
        "organic chemistry", "microbiology", "neuroscience",
        "biotechnology", "nanotechnology", "climate science",
        "oceanography", "meteorology", "materials science",
        "computer science", "data science", "astrophysics",
        "theoretical physics", "applied sciences", "science technology",
        "science experiments", "science projects", "science communication",
        "science policy"
    ],
    "travel": [
        "journeys", "vacations", "tourism", "destinations", "adventures",
        "traveling", "travel tips", "travel guides", "travel experiences",
        "backpacking", "luxury travel", "budget travel", "cultural travel",
        "ecotourism", "sustainable travel", "road trips", "travel photography",
        "travel planning", "travel blogs", "travel itineraries",
        "travel safety", "travel gear", "travel deals", "travel agencies",
        "travel visas", "travel insurance", "travel trends",
        "travel documentaries", "travel vlogs", "solo travel", "group travel",
        "cruises", "hostels", "hotels", "resorts", "air travel",
        "train travel", "bus travel", "car rentals", "travel apps",
        "travel technology", "travel industry", "travel experiences"
    ],
    "food": [
        "cuisine", "recipes", "cooking", "dining", "restaurants",
        "food culture", "food trends", "food reviews", "gastronomy",
        "food photography", "food blogs", "food festivals", "food recipes",
        "healthy eating", "fast food", "fine dining", "street food",
        "vegetarian", "vegan", "gluten-free", "desserts", "appetizers",
        "main courses", "soups", "salads", "seafood", "meat dishes",
        "dessert recipes", "food nutrition", "food safety",
        "food allergies", "food sustainability", "food sourcing",
        "food production", "food technology", "food industry",
        "home cooking", "cooking techniques", "international cuisines",
        "regional dishes", "food presentation", "food pairings",
        "wine and food", "beverages", "food history", "food education",
        "food science", "food blogging", "food styling"
    ],
    # Add more categories and synonyms as needed
}



def map_keywords_to_categories(expanded_keywords, category_mapping, synonyms):
    relevant_categories = set()
    for keyword_list in expanded_keywords:
        for keyword in keyword_list:
            keyword_lower = keyword.lower()
            for category, syns in synonyms.items():
                if keyword_lower == category.lower() or keyword_lower in syns:
                    relevant_categories.add(category)
    return relevant_categories


# Load all JSON channel histories from the specified directory
def load_channel_histories(directory):
    """Load all JSON channel histories from the specified directory."""
    json_files = glob.glob(f"{directory}/*.json")
    messages = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            messages.extend(data)
    return messages


# Group messages by users based on their usernames
def group_messages_by_user(messages):
    """Group messages by users based on their username."""
    user_messages = {}
    for message in messages:
        username = message.get('username')  # Includes discriminator, e.g., User#1234
        if username:
            key = username
            if key not in user_messages:
                user_messages[key] = []
            user_messages[key].append(message)
    return user_messages


# Load messages for a specific user since a given timestamp
def load_user_messages_since(directory, username, since_timestamp):
    """Load messages for a specific user since a given timestamp."""
    messages = []
    json_files = glob.glob(f"{directory}/*.json")
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for msg in data:
                if msg['username'] == username:
                    msg_timestamp = datetime.strptime(msg['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
                    if msg_timestamp > since_timestamp:
                        messages.append(msg)
    return messages

def normalize_username(username):
    """
    Normalizes the username for safe use in filenames by replacing or removing invalid characters.
    """
    # Replace '#' with '-' and remove any other problematic characters
    safe_username = username.replace('#', '-').replace(' ', '_')
    # Remove any other characters you might consider unsafe
    safe_username = re.sub(r'[<>:"/\\|?*]', '', safe_username)
    return safe_username.lower()


async def reindex_user_messages(username):
    # Skip reindexing for the bot's own messages
    if username == str(bot.user):
        logger.warning(f"Skipping reindexing messages for bot user '{username}'.")
        return

    directory = "/absolute/directory/of/your/script/"
    json_files = glob.glob(f"{directory}/*.json")

    for json_file_path in json_files:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            updated_entries = []
            for entry in data:
                if entry['username'].lower() == username.lower():
                    # Load the latest profile data
                    profile_filename = get_profile_filename(username)
                    if os.path.exists(profile_filename):
                        with open(profile_filename, 'r', encoding='utf-8') as profile_file:
                            try:
                                profile_data = json.load(profile_file)
                            except json.JSONDecodeError:
                                profile_data = load_or_create_profile(profile_filename, username)  # Reset profile
                                logger.error(f"Profile file {profile_filename} is corrupted. Resetting profile.")
                    else:
                        # Profile file does not exist; create a new profile
                        profile_data = load_or_create_profile(profile_filename, username)
                        logger.warning(f"Profile file {profile_filename} does not exist. Creating a new profile.")

                    # Update the entry with latest profile fields
                    entry.update({
                        "nickname": profile_data.get("nicknames", []),
                        "political_party": profile_data.get("political_party", ""),
                        "user_job_career": profile_data.get("user_job_career", ""),
                        "user_family_friends": profile_data.get("user_family_friends", ""),
                        "user_activities": profile_data.get("user_activities", ""),
                        "opinions_about_games": profile_data.get("opinions_about_games", ""),
                        "opinions_about_movies": profile_data.get("opinions_about_movies", ""),
                        "opinions_about_music": profile_data.get("opinions_about_music", ""),
                        "opinions_about_television": profile_data.get("opinions_about_television", ""),
                        "opinions_about_life": profile_data.get("opinions_about_life", ""),
                        "general_opinions": profile_data.get("general_opinions", ""),
                        "opinions_about_politics": profile_data.get("opinions_about_politics", ""),
                        "personality_traits": profile_data.get("personality_traits", ""),
                        "hobbies": profile_data.get("hobbies", ""),
                        "user_interests": profile_data.get("user_interests", ""),
                        "user_problems": profile_data.get("user_problems", ""),
                        "tech_interests": profile_data.get("tech_interests", ""),
                        "opinions_about_technology": profile_data.get("opinions_about_technology", ""),
                        "sports_interests": profile_data.get("sports_interests", ""),
                        "opinions_about_sports": profile_data.get("opinions_about_sports", ""),
                        "book_preferences": profile_data.get("book_preferences", ""),
                        "opinions_about_books": profile_data.get("opinions_about_books", ""),
                        "art_interests": profile_data.get("art_interests", ""),
                        "opinions_about_art": profile_data.get("opinions_about_art", ""),
                        "health_concerns": profile_data.get("health_concerns", ""),
                        "health_habits": profile_data.get("health_habits", ""),
                        "science_interests": profile_data.get("science_interests", ""),
                        "opinions_about_science": profile_data.get("opinions_about_science", ""),
                        "travel_preferences": profile_data.get("travel_preferences", ""),
                        "travel_experiences": profile_data.get("travel_experiences", ""),
                        "food_preferences": profile_data.get("food_preferences", ""),
                        "opinions_about_food": profile_data.get("opinions_about_food", ""),
                    })
                    updated_entries.append(entry)

    if updated_entries:
        # Asynchronously add documents to Solr
        await async_solr_add(solr, updated_entries, commit=True)
        logger.info(f"Reindexed {len(updated_entries)} messages for user '{username}'.")
    else:
        logger.warning(f"No messages found to reindex for user '{username}'.")



# Initialize a thread pool executor (optional: configure max_workers as needed)
executor = asyncio.get_event_loop().run_in_executor

async def async_solr_add(solr_instance, documents, commit=True):
    """
    Asynchronously add documents to Solr.
    
    :param solr_instance: Instance of pysolr.Solr
    :param documents: List of documents to add
    :param commit: Whether to commit after adding
    """
    try:
        # Use a partial function to fix the parameters
        add_partial = partial(solr_instance.add, documents, commit=commit)
        # Run the blocking add operation in the default executor
        await asyncio.get_event_loop().run_in_executor(None, add_partial)
        logger.info(f"Successfully indexed {len(documents)} documents in Solr.")
    except Exception as e:
        logger.error(f"Failed to index documents in Solr: {e}")

# Asynchronously initialize user profiles using existing local JSON channel histories
async def initialize_user_profiles_from_json():
    logger.info("Initializing profiles based on local JSON histories.")
    messages = load_channel_histories('/absolute/directory/of/your/script/')  # Update the directory path
    user_messages = group_messages_by_user(messages)

    for username, user_msgs in user_messages.items():
        # Skip creating a profile for the bot
        if username == str(bot.user):
            logger.warning(f"Skipping profile initialization for bot user '{username}'.")
            continue

        filename = get_profile_filename(username)
        if os.path.exists(filename):
            logger.info(f"Profile already exists for {username}. Skipping creation.")
        else:
            await create_or_update_user_profile("LocalGuild", username, user_msgs)


# Asynchronously create or update a user's profile based on their messages
async def create_or_update_user_profile(guild, username, messages):
    # Skip updating the bot's own profile
    if username == str(bot.user):
        logger.warning(f"Skipping profile update for bot user '{username}'.")
        return
    """Create or update a user's profile using their messages."""
    filename = get_profile_filename(username)

    # Load or create the profile
    profile = load_or_create_profile(filename, username)

    # Analyze messages and update the profile
    await analyze_messages_and_update_profile(messages, profile)

    # Log the current last_updated before updating
    logger.debug(f"Current last_updated for {username}: {profile.get('last_updated')}")

    # Update last_updated timestamp
    if messages:
        latest_msg_timestamp = max(
            datetime.strptime(msg['timestamp'], '%Y-%m-%dT%H:%M:%S.%f') for msg in messages
        )
        profile['last_updated'] = latest_msg_timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]  # Truncated to milliseconds

        # Log the new last_updated after updating
        logger.debug(f"Updated last_updated for {username}: {profile['last_updated']}")
    else:
        logger.info(f"No new messages for {username}; last_updated remains the same.")

    # Clean all string fields in the profile to remove unwanted symbols before saving
    for key, value in profile.items():
        if isinstance(value, str):
            cleaned_value = value.replace('\n', ' ').replace('\r', ' ').strip()
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value)  # Remove excessive whitespace
            profile[key] = cleaned_value
        elif isinstance(value, list):
            # Optionally, clean strings within lists
            profile[key] = [item.replace('\n', ' ').replace('\r', ' ').strip() for item in value if isinstance(item, str)]

    # Save the updated profile to the JSON file
    try:
        with open(filename, "w", encoding='utf-8') as file:
            json.dump(profile, file, indent=4, ensure_ascii=False)
        logger.info(f"Profile saved for {username} in guild '{guild}' at {filename}.")
    except Exception as e:
        logger.error(f"Failed to save profile for {username}. Error: {e}")

    # Reindex all messages for this user to include updated profile data
    await reindex_user_messages(username)


# Schedule and perform periodic updates to user profiles every X minutes, set in .env
async def scheduled_profile_updates():
    while True:
        try:
            logger.info("Running scheduled profile updates")
            profiles_dir = "users/"
            profile_files = glob.glob(f"{profiles_dir}/*.json")
            for profile_file in profile_files:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    username = profile.get('username')

                    if not username:
                        logger.warning(f"Username missing in profile file '{profile_file}'. Skipping.")
                        continue

                    # Skip updating the bot's own profile
                    if username == str(bot.user):
                        logger.warning(f"Skipping scheduled update for bot user '{username}'.")
                        continue

                    last_updated_str = profile.get('last_updated', datetime.min.strftime('%Y-%m-%dT%H:%M:%S.%f'))
                    logger.debug(f"Scheduled update for {username}; last_updated is {last_updated_str}")
                    
                    try:
                        last_updated = datetime.strptime(last_updated_str, '%Y-%m-%dT%H:%M:%S.%f')
                    except ValueError as ve:
                        logger.error(f"Invalid date format for user '{username}': {ve}. Using datetime.min.")
                        last_updated = datetime.min

                    messages = load_user_messages_since('/absolute/directory/of/your/script/', username, last_updated)
                    
                    if messages:
                        logger.info(f"Updating profile for {username} with {len(messages)} new messages")
                        await create_or_update_user_profile("ScheduledUpdate", username, messages)
                        # 'create_or_update_user_profile' saves the profile including 'last_updated'
                    else:
                        logger.warning(f"No new messages for {username}")
            
            # Retrieve and convert the update interval from environment variable
            update_interval_str = os.getenv('UPDATE_INTERVAL_MINUTES', '120')  # Default to 120 minutes if not set
            
            try:
                update_interval = int(update_interval_str)
                if update_interval <= 0:
                    logger.warning(f"UPDATE_INTERVAL_MINUTES is non-positive ({update_interval}). Using default of 120 minutes.")
                    update_interval = 120
            except ValueError:
                logger.warning(f"UPDATE_INTERVAL_MINUTES='{update_interval_str}' is not a valid integer. Using default of 120 minutes.")
                update_interval = 120

            logger.info(f"Scheduled profile updates completed. Next update in {update_interval} minutes.")
        except Exception as e:
            logger.error(f"Error during scheduled profile updates: {e}")
        
        # Wait for the specified number of minutes before the next update cycle
        await asyncio.sleep(update_interval * 60)  # Convert minutes to seconds

# Define all profile fields in a master list
ALL_PROFILE_FIELDS = [
    "username",
    "nicknames",
    "join_date",
    "political_party",
    "user_job_career",
    "user_family_friends",
    "user_activities",
    "opinions_about_games",
    "opinions_about_movies",
    "opinions_about_music",
    "opinions_about_television",
    "opinions_about_life",
    "opinions_about_food",
    "general_opinions",
    "opinions_about_politics",
    "personality_traits",
    "hobbies",
    "user_interests",
    "user_problems",
    "tech_interests",
    "opinions_about_technology",
    "sports_interests",
    "opinions_about_sports",
    "book_preferences",
    "opinions_about_books",
    "art_interests",
    "opinions_about_art",
    "health_concerns",
    "health_habits",
    "science_interests",
    "opinions_about_science",
    "travel_preferences",
    "travel_experiences",
    "food_preferences",
    "opinions_about_food",   
    "last_updated"
]


def load_or_create_profile(filename, username):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding='utf-8') as file:
                profile = json.load(file)
            logger.info(f"Loaded existing profile for {username}.")
        except json.JSONDecodeError:
            # Handle corrupted JSON files by resetting the profile
            profile = create_empty_profile(username)
            logger.error(f"Profile file {filename} is corrupted. Resetting profile for {username}.")
    else:
        profile = create_empty_profile(username)
        logger.warning(f"Created new profile for {username}.")
    
    # Ensure all fields are present
    for field in ALL_PROFILE_FIELDS:
        if field not in profile:
            if field == "nicknames":
                profile[field] = []
            elif field == "last_updated":
                profile[field] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            else:
                profile[field] = ""
            logger.info(f"Added missing field '{field}' to profile for {username}.")
    
    # Clean all string fields in the profile to remove unwanted symbols
    for key, value in profile.items():
        if isinstance(value, str):
            cleaned_value = value.replace('\n', ' ').replace('\r', ' ').strip()
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value)  # Remove excessive whitespace
            profile[key] = cleaned_value
        elif isinstance(value, list):
            # Clean strings within lists
            profile[key] = [item.replace('\n', ' ').replace('\r', ' ').strip() for item in value if isinstance(item, str)]
    
    return profile



def create_empty_profile(username):
    profile = {
        "username": username,
        "nicknames": [],
        "join_date": "Unknown",
        "political_party": "",
        "user_job_career": "",
        "user_family_friends": "",
        "user_activities": "",
        "opinions_about_games": "",
        "opinions_about_movies": "",
        "opinions_about_music": "",
        "opinions_about_television": "",
        "opinions_about_life": "",
        "general_opinions": "",
        "opinions_about_politics": "",
        "personality_traits": "",
        "hobbies": "",
        "user_interests": "",
        "user_problems": "",
        "tech_interests": "",
        "opinions_about_technology": "",
        "sports_interests": "",
        "opinions_about_sports": "",
        "book_preferences": "",
        "opinions_about_books": "",
        "art_interests": "",
        "opinions_about_art": "",
        "health_concerns": "",
        "health_habits": "",
        "science_interests": "",
        "opinions_about_science": "",
        "travel_preferences": "",
        "travel_experiences": "",
        "food_preferences": "",
        "opinions_about_food": "",     
        "last_updated": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    }
    logger.warning(f"Created new profile for {username}.")
    # Ensure 'nicknames' and 'last_updated' fields
    if 'nicknames' not in profile:
        profile['nicknames'] = []
    if 'last_updated' not in profile:
        profile['last_updated'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    
    # Clean all string fields in the profile to remove unwanted symbols
    for key, value in profile.items():
        if isinstance(value, str):
            cleaned_value = value.replace('\n', ' ').replace('\r', ' ').strip()
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value)  # Remove excessive whitespace
            profile[key] = cleaned_value
        elif isinstance(value, list):
            # Clean strings within lists
            profile[key] = [item.replace('\n', ' ').replace('\r', ' ').strip() for item in value if isinstance(item, str)]
    
    return profile


async def analyze_messages_and_update_profile(messages, profile):
    """Analyze messages and update the user's profile accurately by categorizing data correctly."""
    existing_fields = "\n".join([f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in profile.items() if value])
    message_context = "\n".join([f"{msg['timestamp']} - {msg['username']}: {msg['content']}" for msg in messages])

    def chunk_text(text, max_tokens=8000):
        """Chunk text into pieces that are within the token limit."""
        sentences = text.split('\n')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            current_chunk += sentence + '\n'
            try:
                token_count = len(tokenizer.encode(current_chunk))
            except Exception as e:
                logger.error(f"Error encoding text: {e}")
                logger.error(f"Current chunk causing error:\n{current_chunk}")
                raise
            if token_count >= max_tokens:
                # Remove the last sentence to stay within the limit
                current_chunk = current_chunk.rsplit('\n', 1)[0]
                chunks.append(current_chunk)
                current_chunk = sentence + '\n'
        if current_chunk.strip():
            chunks.append(current_chunk)
        return chunks

    message_chunks = chunk_text(message_context, max_tokens=8000)
    full_analysis_result = ""

    async def analyze_chunk_with_retries(chunk, retries=3, delay=1):
        """Analyze a chunk of text with retries on failure."""
        for attempt in range(retries):
            try:
                prompt = f"""
You are tasked with accurately updating user profiles based on analyzed messages. Ensure data is categorized correctly into the designated fields.

Profile Fields and their descriptions:
- political_party: The user's political beliefs, affiliations, or philosophies.
- user_job_career: Information about the user's job or career.
- user_family_friends: Information about the user's family and friends.
- user_activities: Information about specific activities (vacations, habits, goings-on).
- opinions_about_games: The user's opinions about video games or gaming.
- opinions_about_movies: The user's opinions about movies.
- opinions_about_music: The user's opinions about music.
- opinions_about_television: The user's opinions about television shows.
- opinions_about_life: The user's general opinions about life.
- opinions_about_food: The user's opinions about foods and favorite foods.
- general_opinions: General opinions the user holds that don't fit into other categories.
- opinions_about_politics: Specific opinions the user holds about politics.
- personality_traits: Descriptions of the user's specific personality traits, generally and also related to the Big Five Personality Traits.
- hobbies: The user's hobbies.
- user_interests: Topics or subjects the user is interested in.
- user_problems: Difficulties that the user faces, in life, work, or other domains.
- tech_interests: The user's interests in technology, including specific areas like AI, cybersecurity, or software development.
- opinions_about_technology: The user's opinions about technological advancements, gadgets, and digital trends.
- sports_interests: The user's interests in sports, including specific sports, teams, or athletic activities.
- opinions_about_sports: The user's opinions about sports, athletes, and sporting events.
- book_preferences: The user's preferences in books, including genres, authors, and favorite titles.
- opinions_about_books: The user's opinions about literature, specific books, and reading habits.
- art_interests: The user's interests in art, including specific art forms, artists, or art movements.
- opinions_about_art: The user's opinions about art, creativity, and artistic expression.
- health_concerns: The user's health-related concerns, including medical conditions or wellness issues.
- health_habits: The user's health-related habits, such as exercise routines, dietary choices, or wellness practices.
- science_interests: The user's interests in scientific topics, including specific fields like biology, chemistry, or physics.
- opinions_about_science: The user's opinions about scientific advancements, research, and the role of science in society.
- travel_preferences: The user's preferences in travel, including favorite destinations, travel styles, and types of vacations.
- travel_experiences: The user's past travel experiences, including memorable trips, challenges, and cultural interactions.
- food_preferences: The user's preferences in food, including favorite cuisines, dietary restrictions, and culinary interests.
- opinions_about_food: The user's opinions about food, cooking, and culinary trends.
Existing Profile (for context):
{existing_fields}

User Messages:
{chunk}

Instructions:
- Extract new information for each field without deleting relevant existing data.
- Place updates in the correct fields. Cross-check context to avoid errors (e.g., gaming info should be in opinions_about_games, not political_party).
- Provide concise updates with clear, evidence-based placements for each field.
- Do not include sensitive information, such as the names of family members or children, or anything else that a reasonable person would consider to be sensitive information.
- Do not include anything blatantly offensive.
- **Output the updates in JSON format, using the field names provided.**
- **Only include fields where you have new updates. Do not include fields without new updates.**
- **Each field should directly contain a list of strings or a single string. Do not nest dictionaries within fields.**
- **Enclose the JSON output in a code block like so: ```json [Your JSON here]```**
- **Do not include any text before or after the JSON code block.**

Provide updates as JSON:
"""
                response = await async_chat_completion(
                    model=os.getenv("MODEL_CHAT"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an assistant specializing in analyzing user messages to update profiles. "
                                "Your primary task is to extract new insights from the provided messages and update specific profile fields. "
                                "Focus on evidence-based updates and use direct quotes when possible to support your conclusions. "
                                "Avoid making assumptions unless clearly justified by the content. Format your response in JSON as instructed."
                                "Notate if something happened in the past and recognize that it is a historical event rather than something ongoing."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=2500,
                    temperature=0.3  # Lower temperature for more deterministic output
                )
                return response
            except Exception as e:
                logger.error(f"Error analyzing chunk: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to analyze chunk after {retries} attempts.")
                    return None


    # Analyze each chunk sequentially and collect the analysis results
    for idx, chunk in enumerate(message_chunks):
        logger.info(f"Analyzing chunk {idx + 1}/{len(message_chunks)}. Chunk size: {len(chunk)} characters.")
        response = await analyze_chunk_with_retries(chunk)

        if response is not None:
            # Extract and aggregate analysis results from each chunk
            analysis_result = response.choices[0].message.content.strip()
            full_analysis_result += analysis_result + "\n"
            logger.info(f"Chunk {idx + 1} analysis completed. Insights generated.")
            logger.debug(analysis_result)
        else:
            logger.error(f"Skipping chunk {idx + 1} due to repeated failures.")

        # Add a pause between processing chunks to prevent rapid successive requests
        await asyncio.sleep(1)

    # Update the profile with the aggregated analysis result
    await update_profile_with_analysis(profile, full_analysis_result)
    logger.info(f"Profile updated based on the aggregated message analysis for {profile['username']}.")


async def update_profile_with_analysis(profile, analysis_result):
    """
    Update profile fields based on the analysis result.
    Cleans the updates by removing newline characters and unwanted symbols.
    """
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Extract all JSON code blocks
    json_pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(json_pattern, analysis_result, re.DOTALL)
    if matches:
        for idx, json_str in enumerate(matches):
            logger.debug(f"JSON string {idx+1} to parse: {json_str}")
            try:
                updates_nested = json.loads(json_str)
                updates = flatten_dict(updates_nested)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON string {idx+1}: {e}")
                logger.error(f"Assistant's response was: {json_str}")
                continue  # Skip this json_str and proceed to the next one

            for field, new_info in updates.items():
                if new_info:
                    key = field.lower()

                    if key == "nicknames":
                        # Ensure 'nicknames' is a list
                        if "nicknames" not in profile or not isinstance(profile["nicknames"], list):
                            profile["nicknames"] = []
                            logger.info(f"Initialized 'nicknames' as a list for user '{profile['username']}'.")

                        # Convert new_info to a list if it's not already
                        if isinstance(new_info, str):
                            new_nicknames = [new_info]
                        elif isinstance(new_info, list):
                            new_nicknames = new_info
                        else:
                            logger.warning(f"Unexpected data type for field 'nicknames': {type(new_info)}")
                            continue

                        # Append new nicknames, avoiding duplicates
                        for nick in new_nicknames:
                            if nick not in profile["nicknames"]:
                                profile["nicknames"].append(nick)
                                logger.info(f"Added new nickname '{nick}' for user '{profile['username']}'.")
                            else:
                                logger.debug(f"Nickname '{nick}' already exists for user '{profile['username']}'. Skipping.")

                    else:
                        existing_info = profile.get(key, '')
                        # If new_info is a string, make it into a list
                        if isinstance(new_info, str):
                            new_info_list = [new_info]
                        elif isinstance(new_info, list):
                            new_info_list = new_info
                        else:
                            logger.warning(f"Unexpected data type for field '{field}': {type(new_info)}")
                            continue

                        # Remove any items that are "no update" (case insensitive)
                        new_info_list = [info for info in new_info_list if isinstance(info, str) and info.lower() != "no update"]
                        if new_info_list:
                            # Use LLM to merge existing_info and new_info intelligently
                            merged_info = await merge_profile_field(field, existing_info, new_info_list)
                            # Further clean the merged_info to remove unwanted symbols
                            merged_info_clean = merged_info.replace('\n', ' ').replace('\r', ' ').strip()
                            merged_info_clean = re.sub(r'\s+', ' ', merged_info_clean)  # Remove excessive whitespace
                            profile[key] = merged_info_clean
                            logger.info(f"Profile updated for {profile['username']} with new info for field '{field}'.")
                        else:
                            # All entries were "no update"
                            continue
                else:
                    # new_info is None or empty
                    continue
    else:
        # If no code blocks found, try to parse the whole analysis_result
        json_str = analysis_result.strip()
        logger.info(f"No code blocks found. Attempting to parse the entire analysis result.")
        try:
            updates_nested = json.loads(json_str)
            updates = flatten_dict(updates_nested)
            # Process updates as above
            for field, new_info in updates.items():
                if new_info:
                    key = field.lower()

                    if key == "nicknames":
                        # Ensure 'nicknames' is a list
                        if "nicknames" not in profile or not isinstance(profile["nicknames"], list):
                            profile["nicknames"] = []
                            logger.info(f"Initialized 'nicknames' as a list for user '{profile['username']}'.")

                        # Convert new_info to a list if it's not already
                        if isinstance(new_info, str):
                            new_nicknames = [new_info]
                        elif isinstance(new_info, list):
                            new_nicknames = new_info
                        else:
                            logger.warning(f"Unexpected data type for field 'nicknames': {type(new_info)}")
                            continue

                        # Append new nicknames, avoiding duplicates
                        for nick in new_nicknames:
                            if nick not in profile["nicknames"]:
                                profile["nicknames"].append(nick)
                                logger.info(f"Added new nickname '{nick}' for user '{profile['username']}'.")
                            else:
                                logger.debug(f"Nickname '{nick}' already exists for user '{profile['username']}'. Skipping.")

                    else:
                        existing_info = profile.get(key, '')
                        if isinstance(new_info, str):
                            new_info_list = [new_info]
                        elif isinstance(new_info, list):
                            new_info_list = new_info
                        else:
                            logger.warning(f"Unexpected data type for field '{field}': {type(new_info)}")
                            continue

                        new_info_list = [info for info in new_info_list if isinstance(info, str) and info.lower() != "no update"]
                        if new_info_list:
                            merged_info = await merge_profile_field(field, existing_info, new_info_list)
                            merged_info_clean = merged_info.replace('\n', ' ').replace('\r', ' ').strip()
                            merged_info_clean = re.sub(r'\s+', ' ', merged_info_clean)
                            profile[key] = merged_info_clean
                            logger.info(f"Profile updated for {profile['username']} with new info for field '{field}'.")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing analysis result as JSON: {e}")
            logger.error(f"Assistant's response was: {analysis_result}")
            return



async def merge_profile_field(field_name, existing_info, new_updates):
    """
    Use the LLM to intelligently merge existing information with new updates for a profile field.
    Cleans the merged information by removing newline characters and unwanted symbols.
    """
    existing_info_text = existing_info if existing_info else 'None'
    new_updates_text = ' '.join(new_updates)  # Replace newlines with spaces

    # Construct a detailed prompt for the LLM to better merge data
    prompt = f"""
You are an assistant tasked with updating a user's profile field by merging new information with existing data carefully.

Field: {field_name}

Existing Information:
{existing_info_text}

New Updates:
{new_updates_text}

Instructions:
- Integrate new updates with existing information clearly and concisely.
- Avoid deleting valuable old data unless it conflicts with more accurate new data.
- Ensure the merged summary avoids redundancy and remains organized.
- Be sure to keep relevant data, but merge data together when it would be redundant.
- Try to avoid deleting unique insights and unique data.
- Correctly place updates in their respective fields; do not mix categories (e.g., gaming data in politics).
- The final summary should maintain coherence and logical structure without significant truncation.
- Do not include sensitive information, such as the names of family members or children, or anything else that a reasonable person would consider to be sensitive information.
- Do not include anything blatantly offensive.
- Keep track of whether or not events are historical and have already come to pass or have resolved.

Merged Information:
"""

    try:
        response = await async_chat_completion(
            model=os.getenv("MODEL_CHAT"),
            messages=[
                {"role": "system", "content": "You are a highly detailed and precise assistant for merging user profile fields."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500
        )
        merged_info = response.choices[0].message.content.strip()
        
        # Clean the merged_info by replacing newline characters with spaces
        merged_info_clean = merged_info.replace('\n', ' ').replace('\r', ' ').strip()
        
        # Optionally, remove other unwanted symbols or perform further cleaning
        # For example, remove excessive whitespace
        merged_info_clean = re.sub(r'\s+', ' ', merged_info_clean)
        
        logger.debug(f"Merged info for field '{field_name}':\n{merged_info_clean}")
        return merged_info_clean
    except Exception as e:
        logger.error(f"Error merging field '{field_name}': {e}")
        combined_info = existing_info + ' ' + ' '.join(new_updates)  # Replace newlines with spaces
        combined_info = re.sub(r'\s+', ' ', combined_info).strip()  # Remove duplicates and excessive whitespace
        return combined_info


def get_profile_filename(username):
    profiles_dir = "users/"
    if not os.path.exists(profiles_dir):
        os.makedirs(profiles_dir)
        logger.info(f"Created directory for user profiles at {profiles_dir}.")
    safe_username = normalize_username(username)
    filename = f"{profiles_dir}/{safe_username}.json"
    return filename


# Function to determine if the bot should respond to a message
def should_bot_respond_to_message(message):
    channel_ids_str = os.getenv("CHANNEL_IDS")
    if not channel_ids_str:
        return False, False

    allowed_channel_ids = [int(cid) for cid in channel_ids_str.split(',')]
    if message.author == bot.user or "Generated Image" in message.content:
        return False, False

    is_random_response = random.random() < 0.015
    is_mentioned = bot.user in [mention for mention in message.mentions]
    if is_mentioned or is_random_response or message.channel.id in allowed_channel_ids:
        return True, is_random_response
    return False, False

# Fetch recent messages from a channel, excluding those containing 'Generated Image' or '!generate' commands
async def fetch_recent_messages(channel, history_length=15):
    message_history = []
    async for message in channel.history(limit=history_length, oldest_first=False):
        if "Generated Image" in message.content or "!generate" in message.content or "!flux" in message.content:
            continue  # Skip messages containing "Generated Image" or commands
        user_mention = f"{message.author.name}: " if message.author != bot.user else ""
        role = "assistant" if message.author == bot.user else "user"
        message_history.append({"role": role, "content": f"{user_mention}{message.content}"})
    return message_history


# Use OpenAI to extract and expand keywords representing the main topics of a message
async def get_expanded_keywords(message):
    prompt = f"Identify and list up to 8 single keywords that represent the main topics of this message. These keywords should be single words. These keywords should be related to the main topic of the message and they should not be things like 'mention' 'notification' 'message' or 'participant'. The keywords should be lowercase, should not be numbered, and should have no punctuation. Avoid using keywords that are a 'type' of message, such as query, recall, remember, mention, or conversation. Instead, they should be real topics of the message. Expand the keywords that you find by adding other, potentially related keywords. For example, if someone asks about 'democrats' then also add the keyword 'republican' and 'politics'. If someone asks about 'food' maybe include different types of common dishes like 'spaghetti' or even types of cuisine like 'italian' or 'chinese'. Message: '{message}'"
    try:
        response = await async_chat_completion(
            model=os.getenv("MODEL_CHAT"),
            messages=[{"role": "system", "content": chatgpt_behaviour}, {"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.5
            
        )
        topic_keywords_str = response.choices[0].message.content.strip()
        
        # Cleaning up the response
        topic_keywords_str = re.sub(r'Keywords:\n\d+\.\s', '', topic_keywords_str)  # Remove 'Keywords:' and numbering
        expanded_keywords = [kw.strip().split(",") for kw in topic_keywords_str.split("\n") if kw.strip()]

        # Log the cleaned and extracted keywords
        logger.info(f"Extracted keywords: {expanded_keywords}")
        return expanded_keywords
    except Exception as e:
        logger.error(f"Error in getting expanded keywords: {e}")
        return None


# Asynchronously save channel message history to JSON and index new messages in Solr
async def save_channel_history_to_json(channel):
    start_time = time.time()
    filename = f"/absolute/directory/of/your/script/{channel.id}.json"
    existing_data = []
    new_message_count = 0

    # Check if JSON file exists for this specific channel
    file_exists_for_channel = os.path.exists(filename)
    days_to_look_back = 15 if file_exists_for_channel else 500

    try:
        if file_exists_for_channel:
            with open(filename, "r", encoding='utf-8') as file:
                existing_data = json.load(file)
            existing_data_count = len(existing_data)
            logger.info(f"Existing data loaded for channel {channel.name} (ID: {channel.id}) with {existing_data_count} messages.")
        else:
            # Create an empty JSON file if it doesn't exist
            with open(filename, "w", encoding='utf-8') as file:
                json.dump(existing_data, file)
            logger.info(f"Created new JSON file for channel {channel.name} (ID: {channel.id}).")
    except PermissionError as e:
        logger.error(f"PermissionError: Unable to create or read file {filename} for channel {channel.name}. Error: {e}")
        return
    except Exception as e:
        logger.error(f"Error: Unable to create or read file {filename} for channel {channel.name}. Error: {e}")
        return

    # Initialize progress tracking
    average_messages_per_day = 100  # Estimate based on your data
    total_messages = days_to_look_back * average_messages_per_day
    processed_messages = 0

    after_date = datetime.utcnow() - timedelta(days=days_to_look_back)
    async for message in channel.history(limit=None, oldest_first=True, after=after_date):
        # Skip messages sent by the bot itself
        if message.author == bot.user:
            continue

        # Skip messages starting with '!'
        if message.content.startswith('!'):
        #    logger.debug(f"Skipping command message: '{message.content}' from user '{message.author}'.")
           continue

        # **New Addition: Skip messages from webhooks**
        if message.webhook_id:
            logger.info(f"Skipping message from webhook in channel {channel.name} (ID: {channel.id}).")
            continue

        message_id = generate_message_id(channel.id, message.created_at)
        if not any(msg['id'] == message_id for msg in existing_data):
            # Retrieve the current nickname; if none, use the username
            nickname = get_nickname(message.author)  # Updated line

            new_message = {
                "id": message_id,
                "username": str(message.author),  # e.g., "User#1234"
                "nicknames": [nickname] if nickname else [],
                "content": message.content,
                "timestamp": message.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # Truncated to milliseconds
            }
            existing_data.append(new_message)
            new_message_count += 1

            # Update progress bar
            processed_messages += 1


    if new_message_count > 0:
        try:
            with open(filename, "w", encoding='utf-8') as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved {new_message_count} new messages to file {filename} for channel {channel.name}.")
        except PermissionError as e:
            logger.error(f"PermissionError: Unable to write to file {filename} for channel {channel.name}. Error: {e}")
        except Exception as e:
            logger.error(f"Error: Unable to write to file {filename} for channel {channel.name}. Error: {e}")

    end_time = time.time()
    duration = end_time - start_time
    new_or_existing = "new" if new_message_count > 0 else "existing"
    logger.info(f"Completed processing channel {channel.name} (ID: {channel.id}) in guild: {channel.guild.name}. {new_or_existing} messages: {new_message_count if new_message_count > 0 else len(existing_data)}. Duration: {duration:.2f} seconds.")





# Add new messages to the json and update the index.
async def save_message_to_json_and_index_solr(channel_id, username, nickname, content, timestamp):
    # Skip saving the bot's own messages
    if username == str(bot.user):
        logger.warning(f"Skipping saving message for bot user '{username}'.")
        return

    # **New Addition: Skip messages starting with '!' (commands)**
    if content.startswith('!'):
        logger.debug(f"Skipping command message: '{content}' from user '{username}'.")
        return
        
    # Generate a unique message ID using channel ID and timestamp
    message_id = generate_message_id(channel_id, timestamp)

    # Save to channel.json
    filename = f"/absolute/directory/of/your/script/{channel_id}.json"

    # Prepare the data structure for the message with only the intended fields
    data = {
        "id": message_id,
        "username": username,       # Unique Discord username (e.g., User#1234)
        "nicknames": [nickname] if nickname else [],  # List of nicknames used on the server
        "content": content,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # Truncated to milliseconds
    }

    # Check if the message ID already exists in Solr
    existing_message_query = f'id:"{message_id}"'
    try:
        existing = await asyncio.get_event_loop().run_in_executor(None, partial(solr.search, existing_message_query))
        if existing.hits > 0:
            logger.warning(f"Message '{message_id}' by {username} already indexed in Solr. Skipping.")
            return  # Skip indexing if the message already exists
    except Exception as e:
        logger.error(f"Error querying Solr for existing message '{message_id}': {e}")
        # Depending on requirements, you might choose to proceed or skip indexing
        return

    # Append data to the JSON file
    try:
        if os.path.exists(filename):
            with open(filename, "r+", encoding='utf-8') as file:
                try:
                    file_data = json.load(file)
                except json.JSONDecodeError:
                    file_data = []
                    logger.error(f"JSON decode error in file {filename}. Starting with an empty list.")
                # Check if the message ID already exists in the file data
                if not any(msg['id'] == message_id for msg in file_data):
                    file_data.append(data)
                    file.seek(0)
                    file.truncate()  # Clear the file before writing new data
                    try:
                        json.dump(file_data, file, indent=4, ensure_ascii=False)
                        logger.info(f"Successfully saved message '{message_id}' to {filename}.")
                    except TypeError as e:
                        logger.error(f"JSON serialization error: {e}")
        else:
            with open(filename, "w", encoding='utf-8') as file:
                try:
                    json.dump([data], file, indent=4, ensure_ascii=False)
                    logger.info(f"Created new JSON file {filename} with message '{message_id}'.")
                except TypeError as e:
                    logger.error(f"JSON serialization error: {e}")
    except Exception as e:
        logger.error(f"Error writing to file {filename}: {e}")
        return

    # Asynchronously add the new message to Solr
    await async_solr_add(solr, [data], commit=True)
    logger.info(f"Message '{message_id}' by {username} indexed in Solr.")

    # **New Addition: Update username.json with the nickname**
    profile_filename = get_profile_filename(username)
    profile = load_or_create_profile(profile_filename, username)

    if nickname and nickname not in profile["nicknames"]:
        profile["nicknames"].append(nickname)
        logger.info(f"Added new nickname '{nickname}' for user '{username}' in profile.")
        
        # Save the updated profile back to username.json
        try:
            with open(profile_filename, "w", encoding='utf-8') as profile_file:
                json.dump(profile, profile_file, indent=4, ensure_ascii=False)
            logger.info(f"Updated profile saved for user '{username}' at {profile_filename}.")
        except Exception as e:
            logger.error(f"Failed to save updated profile for user '{username}'. Error: {e}")
    else:
        if nickname:
            logger.debug(f"Nickname '{nickname}' already exists for user '{username}'. Skipping append.")
        else:
            logger.debug(f"No nickname to add for user '{username}'.")




# Split a long message into smaller chunks
def split_message(message_content, min_length=1500):
    chunks = []
    remaining = message_content
    while len(remaining) > min_length:
        index = max(remaining.rfind(".", 0, min_length),
                    remaining.rfind("!", 0, min_length),
                    remaining.rfind("?", 0, min_length))
        if index == -1: index = min_length
        chunks.append(remaining[:index + 1])
        remaining = remaining[index + 1:]
    chunks.append(remaining)
    return chunks

# Combine and rank Solr search results with recent chat history for context in responses
def combine_and_rank_results(history, solr_results):
    combined_results = []

    # Add a preface to indicate these are past messages
    preface = {
        "role": "system",
        "content": "## Historical Messages:\n\nBelow are messages from the past that may be relevant to the current conversation."
    }
    combined_results.append(preface)

    # Add Solr results first, sorted by tier
    for tier, results in solr_results.items():
        if results:
            tier_label = f"### {tier} Results:\n"
            for result in results:
                if isinstance(result.get('username'), list) and isinstance(result.get('content'), list):
                    username = result['username'][0]
                    content = result['content'][0]
                    combined_results.append({
                        "role": "user",
                        "content": f"{tier_label}{username}: {content}"
                    })
                else:
                    logger.warning("Unexpected data structure in Solr result")

    # Append the recent chat history after the Solr results, with a label
    if history:
        history_label = ""
        for msg in history:
            # Avoid modifying the original message
            msg_copy = msg.copy()
            msg_copy['content'] = history_label + msg_copy['content']
            combined_results.append(msg_copy)

    return combined_results


# Asynchronous function for getting chat completions from OpenAI's API
async def async_chat_completion(*args, **kwargs):
    response = await asyncio.to_thread(openai.chat.completions.create, *args, **kwargs)
    return response


# Perform tiered Solr searches based on expanded keywords to retrieve relevant past messages
async def perform_tiered_solr_search(message_author, expanded_keywords):
    solr_queries = {
        "Tier 1": [f'content:"{keyword}"' for keyword in expanded_keywords[0]],
        "Tier 2": [f'content:"{related}"' for keyword_list in expanded_keywords[1:] for related in keyword_list if related.strip()],
    }
    solr_results = {}

    for tier, queries in solr_queries.items():
        if queries:  # Only construct and execute the query if there are valid terms
            combined_query = f'username:"{message_author}" AND ({ " OR ".join(queries) })'
            try:
                results = solr.search(combined_query, **{"rows": 10})
                solr_results[tier] = results.docs
            except Exception as e:
                logger.error(f"Error querying Solr for {tier}: {e}")
        else:
            logger.warning(f"No valid queries for {tier}. Skipping Solr query for this tier.")
    return solr_results


# Retrieve a specified number of recent messages from a channel for context
async def fetch_message_history(channel, message_author, expanded_keywords):
    # Fetch recent channel messages
    history = await fetch_recent_messages(channel, history_length=15)
    # Perform Solr search with expanded keywords
    solr_results = await perform_tiered_solr_search(message_author, expanded_keywords)
    # Combine and rank results, ensuring Solr results are prioritized
    combined_results = combine_and_rank_results(history, solr_results)
    return combined_results


# Function to remove redundant messages
def remove_redundant_messages(messages):
    filtered_messages = []
    last_message = None
    for message in messages:
        if message != last_message:
            filtered_messages.append(message)
        else:
            logger.debug(f"Redundant message detected and removed: {message}")
        last_message = message
    return filtered_messages

# Set the image side based on a modifier in the prompt
def parse_image_size(prompt):
    if "--wide" in prompt:
        size = "1792x1024"
        prompt = prompt.replace("--wide", "").strip()
    elif "--square" in prompt:
        size = "1024x1024"
        prompt = prompt.replace("--square", "").strip()
    elif "--tall" in prompt:
        size = "1024x1792"
        prompt = prompt.replace("--tall", "").strip()
    elif "--small" in prompt:
        size = "512x512"
        prompt = prompt.replace("--small", "").strip()
    else:
        size = "1024x1024"
    return prompt, size

# Create unique filenames for generated images
def generate_unique_filename(prompt, extension=".png"):
    # Create a base filename from the prompt
    base_filename = re.sub(r'\W+', '', prompt[:80]).lower()
    # Generate a timestamp string
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Combine base filename and timestamp
    unique_filename = f"{base_filename}_{timestamp}{extension}"
    return unique_filename

# Process an image URL and return a base64 encoded string
async def encode_discord_image(image_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as response:
            if response.status == 200:
                image_data = await response.read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                if max(image.size) > 1000:
                    image.thumbnail((1000, 1000))
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Analyze an image and return the AI's description based on instructions
async def analyze_image(base64_image, instructions):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 400,
        "temperature":0.5
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
                response_json = await response.json()
                if response.status == 200:
                    if 'usage' in response_json:
                        total_tokens = response_json['usage']['total_tokens']
                        logger.info(f"Total Tokens for image analysis: {total_tokens}")
                    else:
                        logger.warning("Token usage information not available for image analysis.")
                    return response_json
                else:
                    error_message = response_json.get("error", {}).get("message", "Unknown error occurred.")
                    logger.error(f"OpenAI API Error: {error_message}")
                    return {"choices": [{"message": {"content": f"Error: {error_message}"}}]}
    except Exception as e:
        logger.error(f"Exception during image analysis: {e}")
        return {"choices": [{"message": {"content": "An error occurred while analyzing the image."}}]}


# Event listener for when the bot is ready
@bot.event
async def on_ready():
    try:
        start_time = time.time()  # Initialize start_time at the beginning of on_ready

        logger.info(f'Logged in as {bot.user.name}')

        # Initialize user profiles using existing JSON channel histories
        logger.warning("Starting initial profile and message processing...")
        await initialize_user_profiles_from_json()  # Ensure profile initialization is awaited
        logger.info("Profile initialization complete.")

        # Start the queue processing as a background task
        asyncio.create_task(process_flux_queue())

        # Process each guild and channel
        for guild_idx, guild in enumerate(bot.guilds, start=1):
            total_guilds = len(bot.guilds)
            logger.info(f"Checking channels in guild: {guild.name} ({guild_idx}/{total_guilds})")

            all_channels = guild.text_channels
            total_channels = len(all_channels)
            
            logger.warning(f"Total channels in guild {guild.name}: {total_channels}")

            processed_channels = set()
            
            for channel in all_channels:
                try:
                    logger.warning(f"Processing channel {channel.name} (ID: {channel.id})")
                    await save_channel_history_to_json(channel)  # Ensure each channel is processed and awaited
                    logger.info(f"Completed processing channel {channel.name} (ID: {channel.id})")
                    processed_channels.add(channel.id)
                except Exception as e:
                    logger.error(f"Failed to save history for channel {channel.name} (ID: {channel.id}): {e}")

        # Start the scheduled profile updates task
        asyncio.create_task(scheduled_profile_updates())

        # Capture end time and calculate total time
        end_time = time.time()  # Track the end time after all initialization is complete
        total_time = end_time - start_time  # Calculate the total load time

        # Convert total_time to minutes and seconds
        minutes, seconds = divmod(total_time, 60)

        logger.info("-----------------------------------------------------")
        logger.info(f"✅ Bot has successfully loaded and is now operational!")
        logger.info(f"⏱️ Total load time: {int(minutes)} minutes and {seconds:.2f} seconds.")
        logger.info(f"🚀 The bot is fully ready and awaiting commands.")
        logger.info("-----------------------------------------------------")

    except Exception as e:
        logger.error(f"Error during bot loading: {e}")




# List of possible Magic 8-Ball responses
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
    "Don’t count on it.",
    "My reply is no.",
    "My sources say no.",
    "Outlook not so good.",
    "Very doubtful.",
    "Absolutely not.",
    "What a stupid question.",
    "Are you stupid?",
    "This is the dumbest question I've ever heard."
]

# Define the !8ball command
@bot.command(
    name='8ball',
    help='Ask the Magic 8-Ball a question and get a response.'
)
async def eight_ball(ctx, *, question: str):
    # Choose a random response from the list
    response = random.choice(magic_8ball_responses)
    # Send the response to the channel
    await ctx.send(f'The 8-Ball says: "{response}"')


# New command to fetch and display the current time in a specified city
# Command to fetch and display the current time in a specified city
@bot.command(
    name='whattime',
    help='Fetches and displays the current time in a specified city.\n\n'
         'Provide the city name as an argument to get the local time there.'
)
async def whattime(ctx, *, location_query: str):
    try:
        geolocator = Nominatim(user_agent="discord_bot_yourbotname")
        location = geolocator.geocode(location_query, addressdetails=True, language='en', timeout=10)
        if not location:
            raise ValueError(f"Could not geocode the location: {location_query}")

        address = location.raw.get('address', {})
        country = address.get('country', 'Unknown country')
        # Attempt to find state, department, or prefecture from the address details
        admin_area = address.get('state', address.get('region', address.get('county', '')))
        
        # Determine if the queried location is effectively the country itself
        is_country_query = location_query.strip().lower() == country.lower()
        
        # Formatting the location string to conditionally include administrative area and country
        location_str = country if is_country_query else f"{location_query.title()}, {country}"
        if admin_area and not is_country_query:
            location_str = f"{location_query.title()}, {admin_area}, {country}"
        
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)
        if not timezone_str:
            raise ValueError(f"Could not find timezone for the location: {location_query}")

        timezone = pytz.timezone(timezone_str)
        # Format the current time to 12-hour format with AM/PM
        current_time = datetime.now(timezone).strftime('%I:%M %p on %Y-%m-%d')
        
        await ctx.send(f"It is currently {current_time} in {location_str}.")

    except ValueError as e:
        await ctx.send(str(e))
        logger.error(f"[!whattime Command Error] {e}")
    except Exception as e:
        await ctx.send("Sorry, I'm unable to process your request at the moment.")
        logger.error(f"[!whattime Command Exception] An error occurred: {e}")

# Command to generate an image based on a text prompt
@bot.command(
    name='generate',
    help='Generates an image using DALL-E 3.\n\n'
         'Options:\n'
         '--wide: Generates a wide image (1920x1024).\n'
         '--tall: Generates a tall image (1024x1920).\n'
         '--seed <number>: Use a specific seed for image generation.\n\n'
         'Default size is 1024x1024. The prompt costs $0.04 per image.'
)
async def generate(ctx, *, prompt: str):
    try:
        prompt, size = parse_image_size(prompt)  # Parse the prompt for size modifiers
        logger.info(f"Creating image based on: {prompt} with size {size}")
        async with ctx.typing():
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            image_file = BytesIO(image_data)
            image_file.seek(0)
            unique_filename = generate_unique_filename(prompt)
            image_discord = discord.File(fp=image_file, filename=unique_filename)

        await ctx.send(f"Generated Image -- every image you generate costs $0.04 so please keep that in mind\nPrompt: {prompt}", file=image_discord)

    except openai.BadRequestError as e:
        # Extracting the relevant error message
        error_message = str(e)
        if 'content_policy_violation' in error_message:
            # Find the start and end of the important message
            start = error_message.find("'message': '") + len("'message': '")
            end = error_message.find("', 'param'")
            important_message = error_message[start:end]

            # Send the extracted message to the channel
            await ctx.send(f"Error: {important_message}")
        logger.error(f"OpenAI BadRequestError: {e}")

    except Exception as e:
        formatted_error = format_error_message(e)
        await ctx.send(f"An error occurred during image generation: {formatted_error}")
        logger.error(formatted_error)

# Define the '!transform' command to modify an attached image based on user instructions
@bot.command(
    name='transform',
    help='Transforms an image based on instructions.\n\n'
         'Attach an image and provide instructions to transform it. The command allows\n'
         'modifying the image description before re-generating the image.\n\n'
         'Options:\n'
         '--wide: Generates a wide image (1920x1024).\n'
         '--tall: Generates a tall image (1024x1920).\n'
         '--seed <number>: Use a specific seed for image generation.\n\n'
         'Default size is 1024x1024.'
)
async def transform(ctx, *, instructions: str):
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]  # Consider the first attachment
        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            async with ctx.typing():  # Show "Bot is typing..." in the channel
                try:
                    logger.info(f"Transforming image: {attachment.filename} with instructions: {instructions}")
                    base64_image = await encode_discord_image(attachment.url)
                    
                    # Parse the instructions for size modifiers
                    instructions, size = parse_image_size(instructions)
                    
                    # Analyze the image and get its description
                    description_result = await analyze_image(base64_image, "Describe this image, you give detailed and accurate descriptions, be specific in whatever ways you can, such as but not limited to colors, species, poses, orientations, objects, and contexts.")
                    
                    if description_result.get('choices'):
                        first_choice = description_result['choices'][0]
                        message_content = first_choice.get('message', {}).get('content', '').strip()
                        if not message_content:
                            logger.warning("The API's response did not contain a description.")
                            await ctx.send("Sorry, I couldn't generate a description for the image.")
                            return
                        original_description = message_content
                    else:
                        logger.error("Unexpected response format or error received from API:")
                        logger.error(json.dumps(description_result, indent=2))
                        await ctx.send("Sorry, I encountered an unexpected issue while processing the image.")
                        return

                    logger.debug(f"Original Description: {original_description}")  # Log original description
                    # Prepare a prompt to integrate the user's instructions into the description
                    prompt = f"Rewrite the following description to incorporate the given transformation.\n\nOriginal Description: {original_description}\n\nTransformation: {instructions}\n\nTransformed Description:"

                    # Use GPT to rewrite the description
                    rewriting_result = await async_chat_completion(
                        model=os.getenv("MODEL_CHAT"),
                        messages=[{"role": "system", "content": transform_behaviour},
                                  {"role": "user", "content": prompt}],
                        max_tokens=450,
                        temperature=0.8
                    )
                    if rewriting_result.choices:
                        modified_description = rewriting_result.choices[0].message.content.strip()
                        logger.info(f"Transformed Description: {modified_description}")
                        
                        # Token usage information
                        if hasattr(rewriting_result, 'usage') and hasattr(rewriting_result.usage, 'total_tokens'):
                            total_tokens = rewriting_result.usage.total_tokens
                            logger.info(f"Total Tokens used for description rewriting: {total_tokens}")
                        else:
                            logger.warning("No token usage information available for description rewriting.")
                        
                    # Generate a new image based on the modified description
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=modified_description,  # This is your transformed description
                        size=size,  # Use the parsed size
                        quality="standard",
                        n=1,
                    )
                    new_image_url = response.data[0].url
                    new_image_data = requests.get(new_image_url).content
                    new_image_file = BytesIO(new_image_data)
                    unique_filename = generate_unique_filename(instructions)
                    new_image_file.seek(0)
                    new_image_discord = discord.File(fp=new_image_file, filename=unique_filename)
                    await ctx.send(f"Transformed Image:\nOriginal instructions: {instructions}", file=new_image_discord)

                except Exception as e:
                    formatted_error = format_error_message(e)
                    await ctx.send(f"An error occurred during the transformation: {formatted_error}")
                    logger.error(formatted_error)
        else:
            await ctx.send("Please attach an image with the !transform command.")
    else:
        await ctx.send("Please attach an image with the !transform command.")

def parse_modifiers(prompt):
    """
    Parses the prompt to extract modifiers and removes them from the prompt.
    Returns the cleaned prompt and a list of modifiers.
    """
    modifiers = []
    possible_modifiers = ['--wide', '--tall', '--small', '--fancy', '--seed']
    # Handle modifiers with values separately (e.g., '--seed 1234')
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
            continue  # Skip since we already handled them
        if mod in prompt:
            modifiers.append((mod, None))
            prompt = prompt.replace(mod, '').strip()
    
    return prompt, modifiers


# Define a global lock and queue for the flux command
flux_lock = asyncio.Lock()
flux_queue = asyncio.Queue()

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
                    await handle_remix(interaction, prompt, width, height, seed)
                elif action == 'rewrite':
                    await handle_rewrite(interaction, prompt, width, height, seed)
                elif action == 'wide':
                    await handle_wide(interaction, prompt, width, height, seed)
                elif action == 'tall':
                    await handle_tall(interaction, prompt, width, height, seed)
                elif action == 'edit':
                    # Get 'view' from the task
                    view = task.get('view')
                    if view is None:
                        logger.error("View not found in task for edit action")
                        await interaction.followup.send("An error occurred: View not found.", ephemeral=True)
                    else:
                        new_prompt = prompt
                        await handle_edit(interaction, new_prompt, width, height, seed, view)
                else:
                    logger.error(f"Unknown button action: {action}")
                    await interaction.followup.send(f"Unknown action: {action}", ephemeral=True)
            else:
                logger.error(f"Unknown task type: {task['type']}")
        except Exception as e:
            if task['type'] == 'flux':
                ctx = task['ctx']
                await ctx.send(f"An error occurred: {str(e)}")
            elif task['type'] == 'button':
                interaction = task['interaction']
                await interaction.followup.send(f"An error occurred: {str(e)}", ephemeral=True)
            logger.error(f"Error processing task: {e}")
        finally:
            flux_queue.task_done()





# Process and generate images using the Flux model, handling various modifiers and iterations
async def process_flux_image(ctx, description: str):
    try:
        logger.info(f"Processing !flux command from {ctx.author.name} in channel {ctx.channel.name}")
        logger.debug(f"Original Prompt: {description}")

        # Save the original description with modifiers
        original_description = description

        # Parse modifiers from the description
        description, modifiers = parse_modifiers(description)
        logger.debug(f"Cleaned Prompt: '{description}'")
        logger.debug(f"Modifiers Found: {modifiers}")

        # Default image dimensions
        width = 1024
        height = 1024
        is_fancy = False

        # Set default values for modifiers
        num_iterations = 1
        seed = -1  # Use -1 to indicate random seed

        # Process modifiers
        for mod, value in modifiers:
            if mod == '--wide':
                width = 1920
                height = 1024
                logger.debug("Modifier: Wide (1920x1024)")
            elif mod == '--tall':
                width = 1024
                height = 1920
                logger.debug("Modifier: Tall (1024x1920)")
            elif mod == '--small':
                width = 512
                height = 512
                logger.debug("Modifier: Small (512x512)")
            elif mod == '--fancy':
                is_fancy = True
                logger.debug("Modifier: Fancy (elaborate the prompt)")
            elif mod == '--seed':
                seed = int(value)
                logger.debug(f"Seed: {seed} (specific)")

        # Generate a random seed if none is provided
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            logger.debug(f"Seed: {seed} (random)")

        # Initialize elaborated_description
        elaborated_description = description

        # If the --fancy flag was detected, use OpenAI to elaborate the description
        if is_fancy:
            prompt = (
                f"This is an image prompt for AI image generation. Turn this simple image prompt into something more detailed, forensic and descriptive in nature, "
                f"that consists of no more than 200 tokens and/or 200 words. The new description should be more creative "
                f"and more detailed than the original description, but it should stay within the original "
                f"prompt's spirit. The new prompt should be CLIP-based and creatively forensic. The prompt you are elaborating on is: {description}"
            )

            # Sending the request to OpenAI's API for elaboration
            response = await async_chat_completion(
                model=os.getenv("MODEL_CHAT"),
                messages=[{"role": "system", "content": prompt}],
                max_tokens=300,
                temperature=0.8
            )

            if response and response.choices:
                # Extract the elaborated prompt from the response
                elaborated_description = response.choices[0].message.content.strip()
                logger.info(f"Elaborated Description: {elaborated_description}")
                description = elaborated_description

        num_steps = 4
        guidance = 3.5

        # Iterate the image generation process based on the number of iterations specified
        for i in range(num_iterations):
            # Generate a random seed for each iteration unless a specific seed is provided
            if seed == -1:
                iteration_seed = random.randint(0, 2**32 - 1)
            else:
                iteration_seed = seed + i  # Slightly modify the specific seed for each iteration

            logger.debug(f"Iteration {i+1}/{num_iterations} Seed: {iteration_seed}")

            # Construct the JSON payload for POST request
            payload = {
                "data": [
                    elaborated_description,  # Use elaborated_description
                    num_steps,
                    guidance,
                    width,
                    height,
                    iteration_seed
                ]
            }

            async with ctx.typing():
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                    async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            image_url = result['data'][0]['url'] if 'data' in result else None
                            duration = result.get('duration', 0)
                            path = result['data'][0].get('path', 'N/A').replace("\\", "/")
                            prompt_used = elaborated_description

                            logger.info("API Response Details:")
                            logger.info(f"Path: {path}")
                            logger.info(f"Duration: {duration:.1f} seconds")
                            logger.info(f"Image URL: {image_url}")
                            logger.info(f"Prompt used: {prompt_used}")

                            if image_url:
                                # Fetch the image from the URL
                                async with session.get(image_url) as image_response:
                                    if image_response.status == 200:
                                        image_bytes = await image_response.read()
                                        random_number = random.randint(100000, 999999)
                                        filename = f"{random_number}_{elaborated_description[:40].replace(' ', '_')}.webp"
                                        image_file = discord.File(io.BytesIO(image_bytes), filename=filename)
                                        queue_size = flux_queue.qsize()
                                        queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                        description_embed = discord.Embed(
                                            description=elaborated_description,
                                            color=discord.Color.blue()
                                        )

                                        details_embed = discord.Embed(
                                            color=discord.Color.green()
                                        )
                                        details_embed.add_field(name="Seed", value=f"{iteration_seed}", inline=True)
                                        details_embed.add_field(name="N", value=f"{i+1} of {num_iterations}", inline=True)
                                        details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                        details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                        # Create a FluxRemixView instance with the original description (with modifiers)
                                        remix_view = FluxRemixView(prompt=elaborated_description, width=width, height=height, seed=iteration_seed)

                                        await ctx.send(
                                            content=f"{ctx.author.mention} Generated Image:",
                                            embeds=[description_embed, details_embed],
                                            file=image_file,
                                            view=remix_view  # Attach the Remix button view
                                        )
                                   
                                    else:
                                        await ctx.send("Failed to fetch the generated image from the provided URL.")
                                        logger.error("Failed to fetch the generated image from the provided URL.")
                            else:
                                await ctx.send("Failed to extract image URL from the response.")
                                logger.error("Failed to extract image URL from the response.")
                        else:
                            await ctx.send(f"Failed to generate image. Status code: {response.status}")
                            logger.error(f"Failed to generate image. Status code: {response.status}")

    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
        logger.error(f"Error in process_flux_image: {str(e)}")


# class EditPromptModal(Modal):
    # """A modal dialog for editing the current prompt."""

    # def __init__(self, view: 'FluxRemixView', message: discord.Message):
        # super().__init__(title="Edit Prompt")
        # self.view = view  # Reference to the parent view to access current dimensions and seed
        # self.message = message  # Reference to the original message

        # # Add an input field for the new prompt, pre-populated with the current prompt
        # self.add_item(
            # TextInput(
                # label="New Prompt",
                # placeholder="Enter the new prompt here...",
                # default=self.view.prompt,  # Pre-populate with the current prompt
                # style=TextStyle.paragraph,  # Use TextStyle.paragraph for multi-line input
                # max_length=2000  # Adjust as needed
            # )
        # )

    # async def on_submit(self, interaction: discord.Interaction):
        # """Handles the submission of the new prompt by enqueuing the action."""
        # new_prompt = self.children[0].value.strip()
        # logger.debug(f"Received new prompt from user: '{new_prompt}'")

        # if not new_prompt:
            # await interaction.response.send_message("Prompt cannot be empty.", ephemeral=True)
            # return

        # # Acknowledge the interaction and show that the bot is processing
        # await interaction.response.defer(thinking=True)
        # logger.debug("Interaction deferred to show typing indicator.")

        # # Enqueue the edit action with the new prompt
        # await flux_queue.put({
            # 'type': 'button',
            # 'interaction': interaction,
            # 'action': 'edit',
            # 'prompt': new_prompt,  # Use the new prompt
            # 'width': self.view.width,
            # 'height': self.view.height,
            # 'seed': self.view.seed
        # })
        # logger.info(f"Edit action enqueued with new prompt: '{new_prompt}'")

        
# class EditPromptModal(discord.ui.Modal):
    # def __init__(self, view, message):
        # super().__init__(title="Edit Prompt")
        # self.view = view
        # self.message = message
        # # Add a text input field with the original prompt
        # self.prompt_input = discord.ui.TextInput(
            # label="Edit your prompt:",
            # default=self.view.prompt,  # Use the original prompt
            # style=discord.TextStyle.long,
            # max_length=2000
        # )
        # self.add_item(self.prompt_input)

    # async def on_submit(self, interaction: discord.Interaction):
        # # Get the new prompt from the text input
        # new_prompt = self.prompt_input.value.strip()
        # logger.info(f"Edit action enqueued with new prompt: '{new_prompt}'")

        # # Enqueue the edit action with necessary info
        # await flux_queue.put({
            # 'type': 'button',
            # 'interaction': interaction,
            # 'action': 'edit',
            # 'prompt': new_prompt,
            # 'width': self.view.width,
            # 'height': self.view.height,
            # 'seed': self.view.seed,
            # 'view': self.view  # Pass the view explicitly
        # })
        

class FluxRemixView(View):
    """A custom Discord UI view for handling image remixing with multiple buttons."""

    def __init__(self, prompt: str, width: int, height: int, seed: int = None, is_fancified: bool = False):
        super().__init__(timeout=None)  # Persistent view
        self.prompt = prompt  # Original or elaborated prompt
        self.width = width
        self.height = height
        self.is_fancified = is_fancified
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)  # Store the seed
        # Clean the prompt by removing modifiers
        self.cleaned_prompt = self.parse_prompt(prompt)
        logger.debug(f"FluxRemixView initialized with cleaned_prompt: '{self.cleaned_prompt}', dimensions: {self.width}x{self.height}, seed: {self.seed}")

    def parse_prompt(self, prompt: str) -> str:
        """
        Removes all modifiers from the prompt and returns the cleaned prompt.
        """
        # Define all possible modifiers
        possible_modifiers = ['--wide', '--tall', '--small', '--fancy', '--seed']
        # Remove modifiers with values
        prompt = re.sub(r'(--seed\s+\d+)', '', prompt)
        # Remove modifiers without values
        for mod in possible_modifiers:
            prompt = prompt.replace(mod, '')
        # Remove extra spaces
        prompt = re.sub(r'\s+', ' ', prompt)
        return prompt.strip()

    def disable_all_buttons(self):
        """Disables all buttons in the view."""
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

    def enable_all_buttons(self):
        """Enables all buttons in the view."""
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = False

    @discord.ui.button(label="Rewrite", emoji="🪄", style=discord.ButtonStyle.success, custom_id="flux_fancy_button")
    async def fancy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Rewrite' button click by enqueuing the action."""
        try:
            # Disable all buttons to prevent multiple clicks
            self.disable_all_buttons()
            await interaction.message.edit(view=self)

            # Acknowledge the interaction and show that the bot is processing
            await interaction.response.defer(thinking=True)

            # Enqueue the rewrite action with necessary info
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'rewrite',
                'prompt': self.cleaned_prompt,
                'width': self.width,
                'height': self.height,
                'seed': self.seed
            })
            logger.info(f"Rewrite action enqueued for prompt: '{self.cleaned_prompt}'")
        except Exception as e:
            await interaction.followup.send("An error occurred during rewrite.", ephemeral=True)
            logger.error(f"Error during rewrite: {e}")

    @discord.ui.button(label="Remix", emoji="🌱", style=discord.ButtonStyle.success, custom_id="flux_remix_button")
    async def remix_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Remix' button click by enqueuing the action."""
        try:
            # Disable all buttons to prevent multiple clicks
            self.disable_all_buttons()
            await interaction.message.edit(view=self)

            # Acknowledge the interaction and show that the bot is processing
            await interaction.response.defer(thinking=True)

            # Enqueue the remix action with necessary info
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'remix',
                'prompt': self.cleaned_prompt,
                'width': self.width,
                'height': self.height,
                'seed': self.seed
            })
            logger.info(f"Remix action enqueued for prompt: '{self.cleaned_prompt}'")
        except Exception as e:
            await interaction.followup.send("An error occurred during remix image generation.", ephemeral=True)
            logger.error(f"Error during remix image generation: {e}")

    @discord.ui.button(label="Wide", emoji="↔️", style=discord.ButtonStyle.danger, custom_id="flux_wide_button")
    async def wide_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Wide' button click by enqueuing the action."""
        try:
            # Disable all buttons to prevent multiple clicks
            self.disable_all_buttons()
            await interaction.message.edit(view=self)

            # Acknowledge the interaction and show that the bot is processing
            await interaction.response.defer(thinking=True)

            # Enqueue the wide action with new dimensions
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'wide',
                'prompt': self.cleaned_prompt,
                'width': 1920,
                'height': 1024,
                'seed': self.seed
            })
            logger.info(f"Wide action enqueued for prompt: '{self.cleaned_prompt}'")
        except Exception as e:
            await interaction.followup.send("An error occurred during wide remix.", ephemeral=True)
            logger.error(f"Error during wide remix: {e}")

    @discord.ui.button(label="Tall", emoji="↕️", style=discord.ButtonStyle.danger, custom_id="flux_tall_button")
    async def tall_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handles the 'Tall' button click by enqueuing the action."""
        try:
            # Disable all buttons to prevent multiple clicks
            self.disable_all_buttons()
            await interaction.message.edit(view=self)

            # Acknowledge the interaction and show that the bot is processing
            await interaction.response.defer(thinking=True)

            # Enqueue the tall action with new dimensions
            await flux_queue.put({
                'type': 'button',
                'interaction': interaction,
                'action': 'tall',
                'prompt': self.cleaned_prompt,
                'width': 1024,
                'height': 1920,
                'seed': self.seed
            })
            logger.info(f"Tall action enqueued for prompt: '{self.cleaned_prompt}'")
        except Exception as e:
            await interaction.followup.send("An error occurred during tall remix.", ephemeral=True)
            logger.error(f"Error during tall remix: {e}")

    # @discord.ui.button(label="Edit", emoji="✏️", style=discord.ButtonStyle.primary, custom_id="flux_edit_button")
    # async def edit_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # """Handles the 'Edit' button click by opening the modal."""
        # try:
            # # Disable all buttons to prevent multiple clicks
            # self.disable_all_buttons()
            # await interaction.message.edit(view=self)

            # # Create and send the modal, passing the original message
            # modal = EditPromptModal(view=self, message=interaction.message)
            # await interaction.response.send_modal(modal)
        # except Exception as e:
            # await interaction.followup.send("An error occurred while opening the edit modal.", ephemeral=True)
            # logger.error(f"Error during edit modal: {e}")

    async def process_fancy_image(self, interaction: discord.Interaction):
        """Processes the 'Rewrite' button action to generate a fancy image."""
        try:
            # Use the OpenAI API to elaborate the prompt
            prompt = (
                f"This is an image prompt for AI image generation. Turn this simple image prompt into something more detailed, forensic, and descriptive in nature, "
                f"that consists of no more than 200 tokens and/or 200 words. The new description should be more creative "
                f"and more detailed than the original description, but it should stay within the original "
                f"prompt's spirit. The new prompt should be CLIP-based and creatively forensic. The prompt you are elaborating on is: {self.cleaned_prompt}"
            )

            # Sending the request to OpenAI's API for elaboration
            response = await async_chat_completion(
                model=os.getenv("MODEL_CHAT"),
                messages=[{"role": "system", "content": prompt}],
                max_tokens=300,
                temperature=0.8
            )

            if response and response.choices:
                # Extract the elaborated prompt from the response
                elaborated_prompt = response.choices[0].message.content.strip()
                logger.info(f"Elaborated Prompt: {elaborated_prompt}")
            else:
                await interaction.followup.send("Failed to elaborate the prompt.", ephemeral=True)
                return

            # Generate a new random seed
            new_seed = random.randint(0, 2**32 - 1)
            logger.debug(f"Generated new random seed for Rewrite: {new_seed}")

            # Define image generation parameters
            num_steps = 4
            guidance = 3.5
            # Construct the JSON payload for the image generation
            payload = {
                "data": [
                    elaborated_prompt,
                    num_steps,
                    guidance,
                    self.width,
                    self.height,
                    new_seed
                ]
            }

            # Send a POST request to the Flux image generation server
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Extract key parts of the response
                        image_url = result['data'][0]['url'] if 'data' in result else None
                        duration = result.get('duration', 0)
                        path = result['data'][0].get('path', 'N/A').replace("\\", "/")
                        prompt_used = elaborated_prompt

                        logger.info("Rewrite Button API Response Details:")
                        logger.info(f"Path: {path}")
                        logger.info(f"Duration: {duration:.1f} seconds")
                        logger.info(f"Image URL: {image_url}")
                        logger.info(f"Prompt used: {prompt_used}")

                        if image_url:
                            # Replace backslashes with forward slashes in the URL
                            image_url = image_url.replace("\\", "/")

                            # Fetch the image from the URL
                            async with session.get(image_url) as image_response:
                                if image_response.status == 200:
                                    image_bytes = await image_response.read()

                                    # Generate a unique filename
                                    random_number = random.randint(100000, 999999)
                                    safe_prompt = re.sub(r'\W+', '', elaborated_prompt[:40]).lower()
                                    filename = f"{random_number}_{safe_prompt}.webp"

                                    # Prepare the image file
                                    image_file = discord.File(BytesIO(image_bytes), filename=filename)

                                    # Construct the queue message
                                    queue_size = flux_queue.qsize() if flux_queue else 0
                                    queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                    # Create embeds for the description and details
                                    description_embed = discord.Embed(
                                        description=elaborated_prompt,
                                        color=discord.Color.blue()
                                    )

                                    details_embed = discord.Embed(
                                        color=discord.Color.green()
                                    )
                                    details_embed.add_field(name="Seed", value=f"{new_seed}", inline=True)
                                    details_embed.add_field(name="N", value="Rewrite", inline=True)
                                    details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                    details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                    # Update the view's prompt
                                    self.prompt = elaborated_prompt
                                    self.cleaned_prompt = self.parse_prompt(elaborated_prompt)
                                    logger.debug(f"Updated view prompt: '{self.cleaned_prompt}'")

                                    # Create a new FluxRemixView for further interactions
                                    new_view = FluxRemixView(prompt=elaborated_prompt, width=self.width, height=self.height, seed=new_seed)

                                    # Send the new image with embeds and the Remix and Rewrite buttons
                                    await interaction.followup.send(
                                        content="Generated Image after Rewrite:",
                                        embeds=[description_embed, details_embed],
                                        file=image_file,
                                        view=new_view  # Attach the new view
                                    )
                                else:
                                    await interaction.followup.send("Failed to fetch the generated image from the provided URL.", ephemeral=True)
                                    logger.error("Failed to fetch the generated image from the provided URL during rewrite.")
                        else:
                            await interaction.followup.send("Failed to extract image URL from the response.", ephemeral=True)
                            logger.error("Failed to extract image URL from the response during rewrite.")
                    else:
                        await interaction.followup.send(f"Failed to generate image. Status code: {response.status}", ephemeral=True)
                        logger.error(f"Failed to generate image during rewrite processing. Status code: {response.status}")
        except Exception as e:
            await interaction.followup.send("An error occurred during rewrite image generation.", ephemeral=True)
            logger.error(f"Error during rewrite image generation: {e}")
        finally:
            # Re-enable all buttons if needed (not necessary here since a new view is created)
            pass

    async def process_edited_prompt(self, interaction: discord.Interaction, new_prompt: str):
        """Processes the edited prompt to regenerate the image."""
        try:
            # Use the edited prompt directly
            edited_prompt = self.parse_prompt(new_prompt)
            logger.info(f"Using edited prompt: {edited_prompt}")

            # Reuse the existing seed
            new_seed = self.seed
            logger.debug(f"Reusing existing seed: {new_seed}")

            # Define image generation parameters
            num_steps = 4
            guidance = 3.5

            # Construct the JSON payload for the image generation
            payload = {
                "data": [
                    edited_prompt,
                    num_steps,
                    guidance,
                    self.width,
                    self.height,
                    new_seed
                ]
            }

            # Send a POST request to the Flux image generation server
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Extract key parts of the response
                        image_url = result['data'][0]['url'] if 'data' in result else None
                        duration = result.get('duration', 0)
                        path = result['data'][0].get('path', 'N/A').replace("\\", "/")
                        prompt_used = edited_prompt

                        logger.info("Edit Button API Response Details:")
                        logger.info(f"Path: {path}")
                        logger.info(f"Duration: {duration:.1f} seconds")
                        logger.info(f"Image URL: {image_url}")
                        logger.info(f"Prompt used: {prompt_used}")

                        if image_url:
                            # Replace backslashes with forward slashes in the URL
                            image_url = image_url.replace("\\", "/")

                            # Fetch the image from the URL
                            async with session.get(image_url) as image_response:
                                if image_response.status == 200:
                                    image_bytes = await image_response.read()

                                    # Generate a unique filename
                                    random_number = random.randint(100000, 999999)
                                    safe_prompt = re.sub(r'\W+', '', edited_prompt[:40]).lower()
                                    filename = f"{random_number}_{safe_prompt}.webp"

                                    # Prepare the image file
                                    image_file = discord.File(BytesIO(image_bytes), filename=filename)

                                    # Construct the queue message based on the current queue size
                                    queue_size = flux_queue.qsize() if flux_queue else 0
                                    queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                    # Create embeds for the description and details
                                    description_embed = discord.Embed(
                                        description=edited_prompt,
                                        color=discord.Color.blue()
                                    )

                                    details_embed = discord.Embed(
                                        color=discord.Color.green()
                                    )
                                    details_embed.add_field(name="Seed", value=f"{new_seed}", inline=True)
                                    details_embed.add_field(name="N", value="Edit", inline=True)
                                    details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                    details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                    # Update the view's prompt
                                    self.prompt = edited_prompt
                                    self.cleaned_prompt = self.parse_prompt(edited_prompt)
                                    logger.debug(f"Updated view prompt: '{self.cleaned_prompt}'")

                                    # Create a new FluxRemixView for further interactions
                                    new_view = FluxRemixView(prompt=edited_prompt, width=self.width, height=self.height, seed=new_seed)

                                    # Send the new image with embeds and the Remix and Rewrite buttons
                                    await interaction.followup.send(
                                        content=f"{interaction.user.mention} Generated Image:",
                                        embeds=[description_embed, details_embed],
                                        file=image_file,
                                        view=new_view  # Attach the new view
                                    )
                                else:
                                    await interaction.followup.send("Failed to fetch the generated image from the provided URL.", ephemeral=True)
                                    logger.error("Failed to fetch the generated image during edit processing.")
                        else:
                            await interaction.followup.send("Failed to extract image URL from the response.", ephemeral=True)
                            logger.error("Failed to extract image URL during edit processing.")
                    else:
                        await interaction.followup.send(f"Failed to generate image. Status code: {response.status}", ephemeral=True)
                        logger.error(f"Failed to generate image during edit processing. Status code: {response.status}")
        except Exception as e:
            await interaction.followup.send("An error occurred while editing the prompt.", ephemeral=True)
            logger.error(f"Error during edit prompt processing: {e}")

async def handle_remix(interaction, prompt, width, height, seed):
    try:
        # Generate a new random seed
        new_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Remix requested with prompt: '{prompt}' and seed: {new_seed}")

        # Define image generation parameters
        num_steps = 4
        guidance = 3.5
        
        # Construct the JSON payload for POST request
        payload = {
            "data": [
                prompt,  # Use the cleaned prompt
                num_steps,
                guidance,
                width,
                height,
                new_seed
            ]
        }

        # Send a POST request to the Flux image generation server
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extract key parts of the response
                    image_url = result['data'][0]['url'] if 'data' in result else None
                    duration = result.get('duration', 0)
                    prompt_used = prompt

                    logger.info("Remix API Response Details:")
                    logger.info(f"Duration: {duration:.1f} seconds")
                    logger.info(f"Image URL: {image_url}")
                    logger.info(f"Prompt used: {prompt_used}")

                    if image_url:
                        # Replace backslashes with forward slashes in the URL
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

                                # Construct the queue message based on the current queue size
                                queue_size = flux_queue.qsize() if flux_queue else 0
                                queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                # Create an embed for the description
                                description_embed = discord.Embed(
                                    description=prompt,
                                    color=discord.Color.blue()
                                )

                                # Create a separate embed for the details
                                details_embed = discord.Embed(
                                    color=discord.Color.green()
                                )
                                details_embed.add_field(name="Seed", value=f"{new_seed}", inline=True)
                                details_embed.add_field(name="N", value="Remix", inline=True)
                                details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                # Create a new FluxRemixView for the new image
                                new_view = FluxRemixView(prompt=prompt, width=width, height=height, seed=new_seed)

                                # Send the new image with embeds and the Remix and Rewrite buttons
                                await interaction.followup.send(
                                    content=f"{interaction.user.mention} Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view  # Attach the Remix button view
                                )
                            else:
                                await interaction.followup.send("Failed to fetch the generated image from the provided URL.", ephemeral=True)
                                logger.error("Failed to fetch the generated image from the provided URL during remix.")
                    else:
                        await interaction.followup.send("Failed to extract image URL from the response.", ephemeral=True)
                        logger.error("Failed to extract image URL from the response during remix.")
                else:
                    await interaction.followup.send(f"Failed to generate image. Status code: {response.status}", ephemeral=True)
                    logger.error(f"Failed to generate remix image. Status code: {response.status}")
    except Exception as e:
        await interaction.followup.send("An error occurred during remix image generation.", ephemeral=True)
        logger.error(f"Error in handle_remix: {e}")


async def handle_rewrite(interaction, prompt, width, height, seed):
    try:
        # Use OpenAI to elaborate the prompt
        elaborate_prompt = (
            f"This is an image prompt for AI image generation. Turn this simple image prompt into something more detailed, forensic and descriptive in nature, "
            f"that consists of no more than 200 tokens and/or 200 words. The new description should be more creative "
            f"and more detailed than the original description, but it should stay within the original "
            f"prompt's spirit. The new prompt should be CLIP-based and creatively forensic. The prompt you are elaborating on is: {prompt}"
        )

        # Send the elaboration request to OpenAI
        response = await async_chat_completion(
            model=os.getenv("MODEL_CHAT"),
            messages=[{"role": "system", "content": elaborate_prompt}],
            max_tokens=300,
            temperature=0.8
        )

        if response and response.choices:
            elaborated_prompt = response.choices[0].message.content.strip()
            logger.info(f"Elaborated Prompt: {elaborated_prompt}")
        else:
            await interaction.followup.send("Failed to elaborate the prompt.", ephemeral=True)
            return

        # Generate a new random seed
        new_seed = random.randint(0, 2**32 - 1)
        logger.debug(f"Generated new random seed for Rewrite: {new_seed}")

        # Define image generation parameters
        num_steps = 4
        guidance = 3.5

        # Construct the JSON payload for POST request
        payload = {
            "data": [
                elaborated_prompt,
                num_steps,
                guidance,
                width,
                height,
                new_seed
            ]
        }

        # Send a POST request to the Flux image generation server
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extract key parts of the response
                    image_url = result['data'][0]['url'] if 'data' in result else None
                    duration = result.get('duration', 0)
                    prompt_used = elaborated_prompt

                    logger.info("Rewrite API Response Details:")
                    logger.info(f"Duration: {duration:.1f} seconds")
                    logger.info(f"Image URL: {image_url}")
                    logger.info(f"Prompt used: {prompt_used}")

                    if image_url:
                        # Replace backslashes with forward slashes in the URL
                        image_url = image_url.replace("\\", "/")

                        # Fetch the image from the URL
                        async with session.get(image_url) as image_response:
                            if image_response.status == 200:
                                image_bytes = await image_response.read()

                                # Generate a unique filename
                                random_number = random.randint(100000, 999999)
                                safe_prompt = re.sub(r'\W+', '', elaborated_prompt[:40]).lower()
                                filename = f"{random_number}_{safe_prompt}.webp"

                                # Prepare the image to be sent to Discord
                                image_file = discord.File(BytesIO(image_bytes), filename=filename)

                                # Construct the queue message based on the current queue size
                                queue_size = flux_queue.qsize() if flux_queue else 0
                                queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                # Create an embed for the description
                                description_embed = discord.Embed(
                                    description=elaborated_prompt,
                                    color=discord.Color.blue()
                                )

                                # Create a separate embed for the details
                                details_embed = discord.Embed(
                                    color=discord.Color.green()
                                )
                                details_embed.add_field(name="Seed", value=f"{new_seed}", inline=True)
                                details_embed.add_field(name="N", value="Rewrite", inline=True)
                                details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                # Create a new FluxRemixView for the new image
                                new_view = FluxRemixView(prompt=elaborated_prompt, width=width, height=height, seed=new_seed)

                                # Send the new image with embeds and the Remix and Rewrite buttons
                                await interaction.followup.send(
                                    content=f"{interaction.user.mention} Generated Image after Rewrite:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view  # Attach the new view
                                )
                            else:
                                await interaction.followup.send("Failed to fetch the generated image from the provided URL.", ephemeral=True)
                                logger.error("Failed to fetch the generated image from the provided URL during rewrite.")
                else:
                    await interaction.followup.send(f"Failed to generate image. Status code: {response.status}", ephemeral=True)
                    logger.error(f"Failed to generate rewrite image. Status code: {response.status}")
    except Exception as e:
        await interaction.followup.send("An error occurred during rewrite image generation.", ephemeral=True)
        logger.error(f"Error in handle_rewrite: {e}")


async def handle_wide(interaction, prompt, width, height, seed):
    try:
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

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extract key parts of the response
                    image_url = result['data'][0]['url'] if 'data' in result else None
                    duration = result.get('duration', 0)
                    prompt_used = prompt

                    logger.info("Wide Remix API Response Details:")
                    logger.info(f"Duration: {duration:.1f} seconds")
                    logger.info(f"Image URL: {image_url}")
                    logger.info(f"Prompt used: {prompt_used}")

                    if image_url:
                        # Replace backslashes with forward slashes in the URL
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

                                # Construct the queue message based on the current queue size
                                queue_size = flux_queue.qsize() if flux_queue else 0
                                queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                # Create an embed for the description
                                description_embed = discord.Embed(
                                    description=prompt,
                                    color=discord.Color.blue()
                                )

                                # Create a separate embed for the details
                                details_embed = discord.Embed(
                                    color=discord.Color.green()
                                )
                                details_embed.add_field(name="Seed", value=f"{seed}", inline=True)
                                details_embed.add_field(name="N", value="Wide", inline=True)
                                details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                # Create a new FluxRemixView for the new image
                                new_view = FluxRemixView(prompt=prompt, width=width, height=height, seed=seed)

                                # Send the new image with embeds and the Remix and Rewrite buttons
                                await interaction.followup.send(
                                    content=f"{interaction.user.mention} Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view  # Attach the Remix button view
                                )
                            else:
                                await interaction.followup.send("Failed to fetch the generated image from the provided URL.", ephemeral=True)
                                logger.error("Failed to fetch the generated image from the provided URL during wide remix.")
                else:
                    await interaction.followup.send(f"Failed to generate image. Status code: {response.status}", ephemeral=True)
                    logger.error(f"Failed to generate wide image. Status code: {response.status}")
    except Exception as e:
        await interaction.followup.send("An error occurred during wide remix image generation.", ephemeral=True)
        logger.error(f"Error in handle_wide: {e}")
            
async def handle_tall(interaction, prompt, width, height, seed):
    try:
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

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extract key parts of the response
                    image_url = result['data'][0]['url'] if 'data' in result else None
                    duration = result.get('duration', 0)
                    prompt_used = prompt

                    logger.info("Tall Remix API Response Details:")
                    logger.info(f"Duration: {duration:.1f} seconds")
                    logger.info(f"Image URL: {image_url}")
                    logger.info(f"Prompt used: {prompt_used}")

                    if image_url:
                        # Replace backslashes with forward slashes in the URL
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

                                # Construct the queue message based on the current queue size
                                queue_size = flux_queue.qsize() if flux_queue else 0
                                queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                # Create an embed for the description
                                description_embed = discord.Embed(
                                    description=prompt,
                                    color=discord.Color.blue()
                                )

                                # Create a separate embed for the details
                                details_embed = discord.Embed(
                                    color=discord.Color.green()
                                )
                                details_embed.add_field(name="Seed", value=f"{seed}", inline=True)
                                details_embed.add_field(name="N", value="Tall", inline=True)
                                details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                # Create a new FluxRemixView for the new image
                                new_view = FluxRemixView(prompt=prompt, width=width, height=height, seed=seed)

                                # Send the new image with embeds and the Remix and Rewrite buttons
                                await interaction.followup.send(
                                    content=f"{interaction.user.mention} Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view  # Attach the Remix button view
                                )
                            else:
                                await interaction.followup.send("Failed to fetch the generated image from the provided URL.", ephemeral=True)
                                logger.error("Failed to fetch the generated image from the provided URL during tall remix.")
                else:
                    await interaction.followup.send(f"Failed to generate image. Status code: {response.status}", ephemeral=True)
                    logger.error(f"Failed to generate tall image. Status code: {response.status}")
    except Exception as e:
        await interaction.followup.send("An error occurred during tall remix image generation.", ephemeral=True)
        logger.error(f"Error in handle_tall: {e}")

async def handle_edit(interaction, new_prompt, width, height, seed, view):
    try:
        # Use the edited prompt directly
        edited_prompt = new_prompt
        logger.info(f"Using edited prompt: '{edited_prompt}'")

        # Define image generation parameters
        num_steps = 4
        guidance = 3.5

        # Parse the prompt to remove modifiers using the View's method
        cleaned_prompt = view.parse_prompt(edited_prompt)
        logger.debug(f"Cleaned Prompt: '{cleaned_prompt}'")

        # Construct the JSON payload for POST request
        payload = {
            "data": [
                cleaned_prompt,
                num_steps,
                guidance,
                width,
                height,
                seed  # Use existing seed
            ]
        }

        # Send a POST request to the Flux image generation server
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post("http://192.168.1.96:7860/api/predict", json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    # Extract key parts of the response
                    image_url = result['data'][0]['url'] if 'data' in result else None
                    duration = result.get('duration', 0)
                    prompt_used = cleaned_prompt

                    logger.info("Edit API Response Details:")
                    logger.info(f"Duration: {duration:.1f} seconds")
                    logger.info(f"Image URL: {image_url}")
                    logger.info(f"Prompt used: {prompt_used}")

                    if image_url:
                        # Replace backslashes with forward slashes in the URL
                        image_url = image_url.replace("\\", "/")

                        # Fetch the image from the URL
                        async with session.get(image_url) as image_response:
                            if image_response.status == 200:
                                image_bytes = await image_response.read()

                                # Generate a unique filename
                                random_number = random.randint(100000, 999999)
                                safe_prompt = re.sub(r'\W+', '', cleaned_prompt[:40]).lower()
                                filename = f"{random_number}_{safe_prompt}.webp"

                                # Prepare the image to be sent to Discord
                                image_file = discord.File(BytesIO(image_bytes), filename=filename)

                                # Construct the queue message based on the current queue size
                                queue_size = flux_queue.qsize() if flux_queue else 0
                                queue_message = f"{queue_size} more in queue" if queue_size > 0 else "Nothing queued"

                                # Create an embed for the description
                                description_embed = discord.Embed(
                                    description=cleaned_prompt,
                                    color=discord.Color.blue()
                                )

                                # Create a separate embed for the details
                                details_embed = discord.Embed(
                                    color=discord.Color.green()
                                )
                                details_embed.add_field(name="Seed", value=f"{seed}", inline=True)
                                details_embed.add_field(name="N", value="Edit", inline=True)
                                details_embed.add_field(name="Queue", value=f"{queue_message}", inline=True)
                                details_embed.add_field(name="Time", value=f"{duration:.1f} seconds", inline=True)

                                # Create a new FluxRemixView for the new image
                                new_view = FluxRemixView(prompt=cleaned_prompt, width=width, height=height, seed=seed)

                                # Send the new image with embeds and the Remix and Rewrite buttons
                                await interaction.followup.send(
                                    content=f"{interaction.user.mention} Generated Image:",
                                    embeds=[description_embed, details_embed],
                                    file=image_file,
                                    view=new_view  # Attach the new view
                                )
                            else:
                                await interaction.followup.send("Failed to fetch the generated image from the provided URL.", ephemeral=True)
                                logger.error("Failed to fetch the generated image from the provided URL during edit.")
                    else:
                        await interaction.followup.send("Failed to extract image URL from the response.", ephemeral=True)
                        logger.error("Failed to extract image URL during edit processing.")
                else:
                    await interaction.followup.send(f"Failed to generate image. Status code: {response.status}", ephemeral=True)
                    logger.error(f"Failed to generate edit image. Status code: {response.status}")
    except Exception as e:
        await interaction.followup.send("An error occurred during edit image generation.", ephemeral=True)
        logger.error(f"Error in handle_edit: {e}")
    finally:
        pass




            
# Checks to see if the flux server is currently online
async def is_flux_server_online(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                return response.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False

# Defines the command for !flux, which generates images using Flux locally.
@bot.command(
    name='flux',
    help='Generates an image using the Flux model.\n\nOptions:\n'
         '--wide: Generates a wide image (1920x1024).\n'
         '--tall: Generates a tall image (1024x1920).\n'
         '--seed <number>: Use a specific seed for image generation.\n'
         '--fancy: Elaborates the prompt to be more creative and detailed.\n\n'
         'Default size is 1024x1024.'
)
async def flux_image(ctx, *, description: str):
    flux_server_url = "http://192.168.1.96:7860"
    
    if not await is_flux_server_online(flux_server_url):
        await ctx.send("The !flux command is currently offline. Suck it.")
        logger.warning("The !flux command is currently offline.")
        return
    
    # Enqueue the task with 'flux' type
    await flux_queue.put({
        'type': 'flux',
        'ctx': ctx,
        'description': description
    })
    logger.info(f"Enqueued flux image generation for: {description}")


# Defines the analyze command, which analyzes an attached image based on user instructions.
@bot.command(
    name='analyze',
    help='Analyzes an attached image based on provided instructions.\n\n'
         'Usage:\n'
         '!analyze [instructions]\n\n'
         'Examples:\n'
         '!analyze Translate the text in this image.\n'
         '!analyze Provide a funny description of this image.\n'
         '!analyze Identify objects and their colors in this image.'
)
async def analyze(ctx, *, instructions: str = None):
    # Check if there is an attachment in the message
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]  # Consider the first attachment
        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            async with ctx.typing():
                try:
                    # Encode the image in base64
                    base64_image = await encode_discord_image(attachment.url)
                    
                    # Use provided instructions or default to a standard description
                    if instructions:
                        analysis_instructions = instructions
                        logger.info(f"Received instructions for analysis: '{analysis_instructions}'")
                    else:
                        analysis_instructions = "Please provide a detailed description of the image."
                        logger.info("No instructions provided. Using default analysis prompt.")
                    
                    # Call the analyze_image function with instructions
                    analysis_result = await analyze_image(base64_image, analysis_instructions)
                    
                    # Check if the analysis result has a content field to send
                    response_text = analysis_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if response_text:
                        await ctx.send(response_text)
                        logger.info("Image analysis completed and response sent.")
                    else:
                        await ctx.send("Sorry, I couldn't analyze the image.")
                        logger.warning("No response text received from image analysis.")
                except Exception as e:
                    error_details = str(e)
                    if hasattr(e, 'response') and e.response is not None:
                        error_details += f" Response: {e.response.text}"
                    formatted_error = format_error_message(error_details)
                    await ctx.send(formatted_error)
                    logger.error(formatted_error)
        else:
            await ctx.send("Please attach a valid image file (PNG, JPG, JPEG, WEBP) with the !analyze command.")
            logger.warning("Invalid image attachment found.")
    else:
        await ctx.send("Please attach an image to analyze using the !analyze command.")
        logger.warning("No image attachment found in the message.")


# The on_message function, which is extremely important to the script's functionality.  
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
        
    if message.content.startswith('!'):
        logger.debug(f"Skipping command message: '{message.content}' from user '{message.author}'.")    

    ctx = await bot.get_context(message)
    if ctx.valid:
        # If the message is a valid command, let the command handler process it
        await bot.process_commands(message)
        return  # Exit the on_message handler to prevent further processing

    if message.content.startswith("!flux"):
        logger.debug(f"Processing !flux command from {message.author}: '{message.content}'")

    # Print received message with the username
    logger.debug(f"Received message from {message.author}: '{message.content}'")

    # Retrieve the current nickname; if none, use the username
    nickname = get_nickname(message.author)  # Updated line
    logger.debug(f"Nickname retrieved: {nickname}")

    # Save the user's message to JSON and Solr
    if message.author != bot.user:
        asyncio.create_task(save_message_to_json_and_index_solr(
            channel_id=message.channel.id,
            username=str(message.author),  
            nickname=nickname,
            content=message.content,
            timestamp=message.created_at
        ))

    # Load user's profile
    username_str = str(message.author)  
    profile_filename = get_profile_filename(username_str)

    if os.path.exists(profile_filename):
        with open(profile_filename, 'r', encoding='utf-8') as f:
            try:
                user_profile = json.load(f)
                logger.info(f"Loaded profile for user '{username_str}'.")
            except json.JSONDecodeError:
                # Handle corrupted profile files by creating a new profile
                logger.error(f"Profile file {profile_filename} is corrupted. Resetting profile for {username_str}.")
                user_profile = load_or_create_profile(profile_filename, username_str)
    else:
        # Create a new profile
        user_profile = load_or_create_profile(profile_filename, username_str)
        logger.warning(f"Created new profile for user '{username_str}'.")

    # Determine if the bot should respond to this particular message
    should_respond, is_random_response = should_bot_respond_to_message(message)

    # If the bot should respond, then proceed with fetching history and Solr search
    if should_respond:
        # Extract and expand keywords
        expanded_keywords = await get_expanded_keywords(message.content)

        if not expanded_keywords:
            logger.error("No keywords extracted. Skipping category mapping.")
            relevant_categories = set()
        else:
            # Call the map_keywords_to_categories function with the synonyms argument
            relevant_categories = map_keywords_to_categories(
                expanded_keywords,
                CATEGORY_TO_PROFILE_FIELDS,
                SYNONYMS
            )
            logger.debug(f"Relevant Categories: {relevant_categories}")

        # Always include 'personality_traits'
        relevant_profile_fields = ["personality_traits"]

        # Add profile fields based on relevant categories
        for category in relevant_categories:
            relevant_profile_fields.extend(CATEGORY_TO_PROFILE_FIELDS.get(category, []))

        # Remove duplicates
        relevant_profile_fields = list(set(relevant_profile_fields))

        # Extract the relevant profile data
        profile_data_to_send = {field: user_profile.get(field, "") for field in relevant_profile_fields}

        # Format the profile data as a readable string
        profile_info = json.dumps(profile_data_to_send, indent=2)

        # Define the system prompt
        system_prompt = {"role": "system", "content": chatgpt_behaviour}

        # Define the user profile message with a clear label
        user_profile_message = {
            "role": "user",
            "content": f"## User Profile Information:\n{profile_info}"
        }

        # Fetch message history and Solr search with expanded keywords
        messages_with_solr = await fetch_message_history(message.channel, username_str, expanded_keywords)

        # Prepare the current user message with a label
        current_user_message = {
            "role": "user",
            "content": f"## Current User Message:\n{username_str}: {message.content}"
        }

        # Define the assistant prompt
        assistant_prompt = {
            "role": "system",
            "content": (
                "What is your reply? Keep the current conversation going, but utilize the chat history "
                "if it seems appropriate and relevant. Be in the moment."
            )
        }

        # Construct the final message list with labeled sections
        messages_for_openai = [
            system_prompt,
            user_profile_message,
        ] + messages_with_solr + [
            current_user_message,
            assistant_prompt
        ]

        # Log the chat history for debugging with section labels
        logger.debug('Chat history for OpenAI:')
        for msg in messages_for_openai:
            role = msg["role"].capitalize()
            content = msg["content"]
            logger.debug(f'\n{role}:\n{content}')

        # Initialize variables to store the response
        airesponse_chunks = []
        response = {}
        openai_api_error_occurred = False

        try:
            async with message.channel.typing():
                # Remove redundant messages from channel and Solr history
                filtered_messages = remove_redundant_messages(messages_for_openai)

                # Set the max_tokens based on the type of response
                max_tokens = int(os.getenv("MAX_TOKENS_RANDOM")) if is_random_response else int(os.getenv("MAX_TOKENS"))

                # Generate a response using the OpenAI API
                response = await async_chat_completion(
                    model=os.getenv("MODEL_CHAT"),
                    messages=filtered_messages,
                    temperature=0.8,
                    top_p=0.75,
                    max_tokens=max_tokens
                )

                # Extract and log the token usage information
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    total_tokens = response.usage.total_tokens
                    logger.info(f"Total Tokens for text response: {total_tokens}")
                else:
                    logger.warning("Token usage information not available.")

                # Split the AI response into manageable chunks
                airesponse = response.choices[0].message.content
                airesponse_chunks = split_message(airesponse)
                # Introduce a delay based on the response length
                total_sleep_time = RATE_LIMIT * len(airesponse_chunks)
                await asyncio.sleep(total_sleep_time)

        except openai.OpenAIError as e:
            # Log and send OpenAI specific errors
            error_msg = f"Error: OpenAI API Error - {e}"
            logger.error(error_msg)
            airesponse = (
                "An error has occurred with your request. Please try again later."
            )
            openai_api_error_occurred = True
            await message.channel.send(airesponse)

        except Exception as e:
            # Log and send unexpected errors
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            airesponse = "An unexpected error has occurred. Please try again later."
            openai_api_error_occurred = True
            await message.channel.send(airesponse)

        # Send the response in chunks to avoid message limits
        if not openai_api_error_occurred:
            for chunk in airesponse_chunks:
                # Remove 'username:' prefix from the beginning of the response
                chunk = re.sub(r'^([^\s:]+(\s+[^\s:]+)?):\s*', '', chunk)

                # Send the response without tagging the user or adding a username prefix
                sent_message = await message.channel.send(chunk)
                logger.debug(f"{bot.user}: {chunk}")

                # Fetch the message from the channel to ensure all attributes are populated
                sent_message = await sent_message.channel.fetch_message(sent_message.id)

                # Correctly retrieve the bot's nickname in the guild
                if sent_message.guild:
                    bot_member = sent_message.guild.me
                    bot_nickname = get_nickname(bot_member)  # Updated line
                else:
                    # In case the message is in a DM (guild is None)
                    bot_nickname = bot.user.name

                # Asynchronously add the sent message to Solr
                # To prevent blocking, call the background indexing function
                asyncio.create_task(save_message_to_json_and_index_solr(
                    channel_id=sent_message.channel.id,
                    username=str(bot.user),    # username
                    nickname=bot_nickname,     # correctly retrieved nickname
                    content=chunk,             # content
                    timestamp=sent_message.created_at  # timestamp
                ))

                await asyncio.sleep(RATE_LIMIT)

    # Cache the user's message, but skip if the author is the bot
    if message.author != bot.user:
        user_message_cache[message.author.id].append({
            "timestamp": message.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],  # Truncated to milliseconds
            "username": username_str,
            "content": message.content
        })

        # If the user has sent 10 messages, update their profile
        if len(user_message_cache[message.author.id]) >= 10:
            await create_or_update_user_profile(
                "LocalGuild",
                username_str,
                user_message_cache[message.author.id]
            )
            # Clear the cache for this user
            user_message_cache[message.author.id] = []

    # Process any commands included in the message
    await bot.process_commands(message)


# Initialize the bot with the Discord token from environment variables
discord_bot_token = os.getenv("DISCORD_TOKEN")
if discord_bot_token is None:
    raise ValueError("No Discord bot token found. Make sure to set the DISCORD_BOT_TOKEN environment variable.")
bot.run(discord_bot_token)


