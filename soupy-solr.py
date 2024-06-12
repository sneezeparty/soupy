import os
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
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from openai import OpenAI
from openai import OpenAIError
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from colorama import init, Fore, Style
from discord.ext import commands
from dotenv import load_dotenv

# Initialize colorama for colored console output in the terminal
init(autoreset=True)

# Load environment variables (like API keys) from .env file
load_dotenv()

# Connect to Solr instance for indexing messages
solr_url = 'http://localhost:8983/solr/soupy'
print(f"Attempting to connect to Solr at {solr_url}")

def generate_message_id(channel_id, timestamp):
    # Use a consistent ISO 8601 format for timestamps
    formatted_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
    return f"{channel_id}_{formatted_timestamp}"

try:
    solr = pysolr.Solr(solr_url, timeout=10)
    # Optionally, perform a simple query to check if Solr is responding
    solr.ping()
    print("Successfully connected to Solr.")
except Exception as e:
    print(f"Failed to connect to Solr. Error: {e}")

def display_image_iterm2(image_path):
    subprocess.run(["imgcat", image_path])

# Function to index all JSON files in a given directory into Solr
def index_all_json_files(directory):
    json_files = glob.glob(f"{directory}/*.json")
    total_files = len(json_files)
    print(f"Total JSON files to index: {total_files}")

    for idx, json_file_path in enumerate(json_files, start=1):
        channel_id = os.path.splitext(os.path.basename(json_file_path))[0]
        print(f"Processing file {idx} of {total_files}: {json_file_path}")
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            if not data:
                print(Fore.YELLOW + f"No data found in file: {json_file_path}" + Style.RESET_ALL)
                continue

            new_entries = 0
            for entry in data:
                timestamp = datetime.strptime(entry['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
                entry['id'] = generate_message_id(channel_id, timestamp)  # Regenerate 'id' using the timestamp

                try:
                    solr.add([entry])
                    new_entries += 1
                except Exception as e:
                    print(Fore.RED + f"Failed to index entry from {json_file_path}. Error: {e}" + Style.RESET_ALL)
            
            solr.commit()
            if new_entries > 0:
                print(Fore.GREEN + f"Indexed {new_entries} entries from file: {json_file_path}" + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + f"No new entries to index from file: {json_file_path}" + Style.RESET_ALL)

# Process and index the data
index_all_json_files("/Users/matthewgilford/git/soupy/combined/")

# Commit changes to make sure data is indexed
solr.commit()

# Initialize the OpenAI client with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("No OpenAI API key found. Set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=openai_api_key)
openai.api_key = openai_api_key

# Initialize Discord bot with specific message intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Set a rate limit for message processing
RATE_LIMIT = 0.25

# Retrieve behavior settings from environment variables
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
                    return Fore.RED + error_prefix + f"An OpenAI API error occurred: Error code {code} - {error_message}" + Style.RESET_ALL
                except ValueError:
                    return Fore.RED + error_prefix + "An OpenAI API error occurred, but the details were not in a recognizable format." + Style.RESET_ALL
            else:
                return Fore.RED + error_prefix + "An OpenAI API error occurred, but no additional details are available." + Style.RESET_ALL
        elif hasattr(error, 'response') and error.response is not None:
            try:
                error_json = error.response.json()
                code = error.response.status_code
                return Fore.RED + error_prefix + f"An HTTP error occurred: Error code {code} - {error_json}" + Style.RESET_ALL
            except ValueError:
                return Fore.RED + error_prefix + "An error occurred, but the response was not in a recognizable format." + Style.RESET_ALL
        else:
            return Fore.RED + error_prefix + str(error) + Style.RESET_ALL
    except Exception as e:
        return Fore.RED + error_prefix + f"An unexpected error occurred while formatting the error message: {e}" + Style.RESET_ALL

# Progress bar for use during JSON updating and creation
def update_progress_bar(current, total):
    bar_length = 50
    progress = current / total
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1}%".format("#" * block + "-" * (bar_length - block), round(progress * 100, 2))
    print(text, end="" if current < total else "\n")

# Function to determine if the bot should respond to a message
def should_bot_respond_to_message(message):
    channel_ids_str = os.getenv("CHANNEL_IDS")
    if not channel_ids_str:
        return False, False

    allowed_channel_ids = [int(cid) for cid in channel_ids_str.split(',')]
    if message.author == bot.user or "Generated Image" in message.content:
        return False, False

    is_random_response = random.random() < 0.01
    is_mentioned = bot.user in [mention for mention in message.mentions]
    if is_mentioned or is_random_response or message.channel.id in allowed_channel_ids:
        return True, is_random_response
    return False, False

async def fetch_recent_messages(channel, history_length=15):
    message_history = []
    async for message in channel.history(limit=history_length, oldest_first=False):
        if "Generated Image" in message.content or "!generate" in message.content:
            continue  # Skip messages containing "Generated Image"
        user_mention = f"{message.author.name}: " if message.author != bot.user else ""
        role = "assistant" if message.author == bot.user else "user"
        message_history.append({"role": role, "content": f"{user_mention}{message.content}"})
    return message_history

async def get_expanded_keywords(message):
    prompt = f"Identify and list up to 8 single keywords that represent the main topics of this message. These keywords should be single words. These keywords should be related to the main topic of the message and they should not be things like 'mention' 'notification' 'message' or 'participant'. The keywords should be lowercase, should not be numbered, and should have no punctuation. Avoid using keywords that are a 'type' of message, such as query, recall, remember, mention, or conversation. Instead, they should be real topics of the message. Expand the keywords that you find by adding other, potentially related keywords. For example, if someone asks about 'democrats' then also add the keyword 'republican' and 'politics'. If someone asks about 'food' maybe include different types of common dishes like 'spaghetti' or even types of cuisine like 'italian' or 'chinese'. Message: '{message}'"
    try:
        response = await async_chat_completion(
            model=os.getenv("MODEL_CHAT"),
            messages=[{"role": "system", "content": chatgpt_behaviour}, {"role": "user", "content": prompt}],
            max_tokens=100
        )
        topic_keywords_str = response.choices[0].message.content.strip()
        
        # Cleaning up the response
        topic_keywords_str = re.sub(r'Keywords:\n\d+\.\s', '', topic_keywords_str)  # Remove 'Keywords:' and numbering
        expanded_keywords = [kw.strip().split(",") for kw in topic_keywords_str.split("\n") if kw.strip()]

        # Print the cleaned and extracted keywords
        print(Fore.GREEN + f"Extracted keywords: {expanded_keywords}" + Style.RESET_ALL)
        return expanded_keywords
    except Exception as e:
        print(Fore.RED + f"Error in getting expanded keywords: {e}" + Style.RESET_ALL)
        return None

async def save_channel_history_to_json(channel):
    start_time = time.time()  # Use time.time() instead of datetime.now()
    filename = f"/Users/matthewgilford/git/soupy/combined/{channel.id}.json"
    existing_data = []
    new_message_count = 0

    # Check if JSON file exists for this specific channel
    file_exists_for_channel = os.path.exists(filename)
    days_to_look_back = 15 if file_exists_for_channel else 365

    try:
        if file_exists_for_channel:
            with open(filename, "r", encoding='utf-8') as file:
                existing_data = json.load(file)
            existing_data_count = len(existing_data)
            print(f"Existing data loaded for channel {Fore.CYAN}{channel.name}{Style.RESET_ALL} (ID: {channel.id}) with {existing_data_count} messages.")
        else:
            # Create an empty JSON file if it doesn't exist
            with open(filename, "w", encoding='utf-8') as file:
                json.dump(existing_data, file)
            print(f"Created new JSON file for channel {Fore.CYAN}{channel.name}{Style.RESET_ALL} (ID: {channel.id}).")

    except PermissionError as e:
        print(Fore.RED + f"PermissionError: Unable to create or read file {filename} for channel {channel.name}. Error: {e}" + Style.RESET_ALL)
        return
    except Exception as e:
        print(Fore.RED + f"Error: Unable to create or read file {filename} for channel {channel.name}. Error: {e}" + Style.RESET_ALL)
        return

    # Initialize progress tracking
    average_messages_per_day = 100  # Estimate based on your data
    total_messages = days_to_look_back * average_messages_per_day
    processed_messages = 0

    after_date = datetime.utcnow() - timedelta(days=days_to_look_back)
    async for message in channel.history(limit=None, oldest_first=True, after=after_date):
        message_id = generate_message_id(channel.id, message.created_at)
        if not any(msg['id'] == message_id for msg in existing_data):
            new_message = {
                "id": message_id,
                "username": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            }
            existing_data.append(new_message)
            new_message_count += 1

            # Update progress bar
            processed_messages += 1
            update_progress_bar(processed_messages, total_messages)

    if new_message_count > 0:
        try:
            with open(filename, "w", encoding='utf-8') as file:
                json.dump(existing_data, file, indent=4, ensure_ascii=False)
            print(Fore.GREEN + f"Successfully saved {new_message_count} new messages to file {filename} for channel {channel.name}." + Style.RESET_ALL)
        except PermissionError as e:
            print(Fore.RED + f"PermissionError: Unable to write to file {filename} for channel {channel.name}. Error: {e}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error: Unable to write to file {filename} for channel {channel.name}. Error: {e}" + Style.RESET_ALL)

    end_time = time.time()
    duration = end_time - start_time
    new_or_existing = "new" if new_message_count > 0 else "existing"
    print(Fore.BLUE + f"Completed processing channel {Fore.CYAN}{channel.name}{Fore.BLUE} (ID: {channel.id}) in guild: {channel.guild.name}. {new_or_existing} messages: {new_message_count if new_message_count > 0 else len(existing_data)}. Duration: {duration:.2f} seconds." + Style.RESET_ALL)

# Add new messages to the json and update the index.
def save_message_to_json_and_index_solr(channel_id, username, content, timestamp):
    filename = f"/Users/matthewgilford/git/soupy/combined/{channel_id}.json"
    message_id = generate_message_id(channel_id, timestamp)

    # Prepare the data structure for the message
    data = {
        "id": message_id,
        "username": username,
        "content": content,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
    }

    # Check if the message ID already exists in Solr
    existing_message_query = f"id:\"{message_id}\""
    search_results = solr.search(existing_message_query)
    if search_results.hits > 0:
        print(Fore.YELLOW + f"Message '{message_id}' by {username} already indexed in Solr. Skipping." + Style.RESET_ALL)
        return  # Skip indexing if the message already exists

    # Append data to the JSON file
    if os.path.exists(filename):
        with open(filename, "r+", encoding='utf-8') as file:
            file_data = json.load(file)
            # Check if the message ID already exists in the file data
            if not any(msg['id'] == message_id for msg in file_data):
                file_data.append(data)
                file.seek(0)
                file.truncate()  # Clear the file before writing new data
                json.dump(file_data, file, indent=4, ensure_ascii=False)
    else:
        with open(filename, "w", encoding='utf-8') as file:
            json.dump([data], file, indent=4, ensure_ascii=False)

    # Index the new message in Solr
    try:
        solr.add([data], commit=True)
        print(Fore.GREEN + f"Message '{message_id}' by {username} indexed in Solr." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Failed to index message '{message_id}' in Solr: {e}" + Style.RESET_ALL)

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

def combine_and_rank_results(history, solr_results):
    combined_results = []
    combined_results.extend(history)
    for tier, results in solr_results.items():
        for result in results:
            if isinstance(result.get('username'), list) and isinstance(result.get('content'), list):
                username = result['username'][0]
                content = result['content'][0]
                combined_results.append({"role": "user", "content": f"{username}: {content}", "tier": tier})
            else:
                print("Warning: Unexpected data structure in Solr result")
    # Rank based on tier (Tier 1 > Tier 2 > history)
    ranked_results = sorted(combined_results, key=lambda x: ("tier" not in x, x.get("tier", ""), x.get("timestamp", "")))
    return ranked_results

# Asynchronous function for getting chat completions from OpenAI's API
async def async_chat_completion(*args, **kwargs):
    response = await asyncio.to_thread(openai.chat.completions.create, *args, **kwargs)
    return response

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
                print(f"Error querying Solr for {tier}: {e}")
        else:
            print(f"No valid queries for {tier}. Skipping Solr query for this tier.")
    return solr_results

# Retrieve a specified number of recent messages from a channel for context
async def fetch_message_history(channel, message_author, expanded_keywords):
    history = await fetch_recent_messages(channel, history_length=15)
    solr_results = await perform_tiered_solr_search(message_author, expanded_keywords)
    combined_results = combine_and_rank_results(history, solr_results)
    return combined_results


    # Fetch messages in reverse order (newest first)
    async for message in channel.history(limit=history_length * 2, oldest_first=False):
        if len(message_history) < history_length and message.content:
            user_mention = f"{message.author.name}: " if message.author != bot.user else ""
            role = "assistant" if message.author == bot.user else "user"
            message_history.insert(0, {"role": role, "content": f"{user_mention}{message.content}"})  # Insert at the beginning

    if topic_words:
        # Construct Solr query using 'OR'
        solr_query = f'username:"{message_author}" AND content:(' + ' OR '.join([f'"{word.strip()}"' for word in topic_words]) + ')'
        print(f"Executing Solr query: {solr_query}")

        try:
            solr_results = solr.search(solr_query, **{"rows": 10})
            # Debugging: Print out raw results
            print(f"Raw Solr Results: {solr_results.docs}")
            print(f"Number of results: {len(solr_results)}")
            for idx, result in enumerate(solr_results):
                print(f"Result {idx + 1}: {result}")
                if isinstance(result.get('username'), list) and isinstance(result.get('content'), list):
                    username = result['username'][0]
                    content = result['content'][0]
                    formatted_result = f"{username}: {content}"
                    message_history.append({"role": "user", "content": formatted_result})
                else:
                    print("Warning: Unexpected data structure in Solr result")
        except Exception as e:
            print(f"Error querying Solr: {e}")

    return message_history


# Function to remove redundant messages
def remove_redundant_messages(messages):
    filtered_messages = []
    last_message = None
    for message in messages:
        if message != last_message:
            filtered_messages.append(message)
        else:
            print(f"Redundant message detected and removed: {message}")
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

# Analyze an image and return the AI's description
async def analyze_image(base64_image, instructions):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [{"type": "text", "text": instructions}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
        "max_tokens": 400
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    if 'usage' in response_json:
        total_tokens = response_json['usage']['total_tokens']
        print(f"Total Tokens for image description: {total_tokens}")
    else:
        print("Token usage information not available for image description.")
    return response_json

# Event listener for when the bot is ready
@bot.event
async def on_ready():
    print(Fore.BLUE + f'Logged in as {bot.user.name}' + Style.RESET_ALL)
    for guild_idx, guild in enumerate(bot.guilds, start=1):
        total_guilds = len(bot.guilds)
        print(Fore.MAGENTA + f"Checking channels in guild: {guild.name} ({guild_idx}/{total_guilds})" + Style.RESET_ALL)

        all_channels = guild.text_channels
        total_channels = len(all_channels)
        
        print(Fore.YELLOW + f"Total channels in guild {guild.name}: {total_channels}" + Style.RESET_ALL)

        # Print out all channels
        for channel in all_channels:
            print(Fore.YELLOW + f"Channel {channel.name} (ID: {channel.id})" + Style.RESET_ALL)

        processed_channels = set()
        
        for channel in all_channels:
            try:
                print(Fore.YELLOW + f"Processing channel {channel.name} (ID: {channel.id})" + Style.RESET_ALL)
                await save_channel_history_to_json(channel)
                print(Fore.GREEN + f"Completed processing channel {channel.name} (ID: {channel.id})" + Style.RESET_ALL)
                processed_channels.add(channel.id)
            except Exception as e:
                print(Fore.RED + f"Failed to save history for channel {channel.name} (ID: {channel.id}): {e}" + Style.RESET_ALL)

        # Identify and print any channels that were not processed
        for channel in all_channels:
            if channel.id not in processed_channels:
                print(Fore.RED + f"Channel {channel.name} (ID: {channel.id}) was not processed." + Style.RESET_ALL)

# New command to fetch and display the current time in a specified city
# New command to fetch and display the current time in a specified city
@bot.command(name='whattime')
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
        print(Fore.RED + f"[!whattime Command Error] {e}" + Style.RESET_ALL)
    except Exception as e:
        await ctx.send("Sorry, I'm unable to process your request at the moment.")
        print(Fore.YELLOW + f"[!whattime Command Exception] An error occurred: {e}" + Style.RESET_ALL)



# Command to generate an image based on a text prompt
@bot.command()
async def generate(ctx, *, prompt: str):
    try:
        prompt, size = parse_image_size(prompt)  # Parse the prompt for size modifiers
        print(Fore.GREEN + f"Creating image based on: {prompt} with size {size}" + Style.RESET_ALL)
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

    except Exception as e:
        formatted_error = format_error_message(e)
        await ctx.send(f"An error occurred during image generation: {formatted_error}")
        print(Fore.RED + formatted_error + Fore.RESET)

@bot.command()
async def transform(ctx, *, instructions: str):
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]  # Consider the first attachment
        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            async with ctx.typing():  # Show "Bot is typing..." in the channel
                try:
                    print(Fore.CYAN + f"Transforming image: {attachment.filename} with instructions: {instructions}" + Fore.RESET)
                    base64_image = await encode_discord_image(attachment.url)
                    
                    # Parse the instructions for size modifiers
                    instructions, size = parse_image_size(instructions)
                    
                    # Analyze the image and get its description
                    description_result = await analyze_image(base64_image, "Describe this image, you give detailed and accurate descriptions, be specific in whatever ways you can, such as but not limited to colors, species, poses, orientations, objects, and contexts.")
                    
                    if description_result.get('choices'):
                        first_choice = description_result['choices'][0]
                        message_content = first_choice.get('message', {}).get('content', '').strip()
                        if not message_content:
                            print("The API's response did not contain a description.")
                            await ctx.send("Sorry, I couldn't generate a description for the image.")
                            return
                        original_description = message_content
                    else:
                        print("Unexpected response format or error received from API:")
                        print(json.dumps(description_result, indent=2))
                        await ctx.send("Sorry, I encountered an unexpected issue while processing the image.")
                        return

                    print(Fore.BLUE + "Original Description: " + original_description + Fore.RESET)  # Log original description
                    # Prepare a prompt to integrate the user's instructions into the description
                    prompt = f"Rewrite the following description to incorporate the given transformation.\n\nOriginal Description: {original_description}\n\nTransformation: {instructions}\n\nTransformed Description:"

                    # Use GPT to rewrite the description
                    rewriting_result = await async_chat_completion(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": transform_behaviour},
                                  {"role": "user", "content": prompt}],
                        max_tokens=450
                    )
                    if rewriting_result.choices:
                        modified_description = rewriting_result.choices[0].message.content.strip()
                        print(Fore.GREEN + "Transformed Description: " + modified_description + Fore.RESET)
                        
                        # Token usage information
                        if hasattr(rewriting_result, 'usage') and hasattr(rewriting_result.usage, 'total_tokens'):
                            total_tokens = rewriting_result.usage.total_tokens
                            print(f"Total Tokens used for description rewriting: {total_tokens}")
                        else:
                            print("No token usage information available for description rewriting.")
                        
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
                    print(Fore.RED + formatted_error + Fore.RESET)
    else:
        await ctx.send("Please attach an image with the !transform command.")

@bot.command()
async def analyze(ctx):
    # Check if there is an attachment in the message
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]  # Consider the first attachment
        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            async with ctx.typing():
                try:
                    # Encode the image in base64
                    base64_image = await encode_discord_image(attachment.url)
                    # Call the analyze_image function
                    instructions = "Please describe the image."  # You can customize this instruction
                    analysis_result = await analyze_image(base64_image, instructions)
                    # Check if the analysis result has a content field to send
                    response_text = analysis_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if response_text:
                        await ctx.send(response_text)
                    else:
                        await ctx.send("Sorry, I couldn't analyze the image.")
                except Exception as e:
                    error_details = str(e)
                    if hasattr(e, 'response') and e.response is not None:
                        error_details += f" Response: {e.response.text}"
                    formatted_error = format_error_message(error_details)
                    await ctx.send(formatted_error)
                    print(Fore.RED + formatted_error + Fore.RESET)
    else:
        await ctx.send("Please attach an image to analyze.")

@bot.event
async def on_message(message):
    # Process any commands that might be part of the message
    await bot.process_commands(message)
   
    # Prevent duplicate processing for commands
    if message.content.startswith(bot.command_prefix):
        return

    # Ignore messages sent by the bot itself
    if message.author == bot.user:
        return

    # Print received message with the username
    print(Fore.CYAN + f"Received message from {message.author.name}: {Fore.YELLOW}'{message.content}'" + Style.RESET_ALL)

    # Log every message (user and bot) in the channel
    save_message_to_json_and_index_solr(message.channel.id, message.author.name, message.content, message.created_at)

    # Determine if the bot should react to this particular message
    should_respond, is_random_response = should_bot_respond_to_message(message)

    # Check if the bot is mentioned in the message
    is_mentioned = bot.user in message.mentions

    # Process image analysis only if the bot is mentioned with an image or in a specific channel
    if message.attachments and (is_mentioned or message.channel.id in [int(cid) for cid in os.getenv("CHANNEL_IDS", "").split(',')]):
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                async with message.channel.typing():
                    try:
                        print(Fore.CYAN + f"Processing image: {attachment.filename}" + Fore.RESET)
                        base64_image = await encode_discord_image(attachment.url)
                        instructions = message.content if message.content else "What’s in this image?"
                        analysis_result = await analyze_image(base64_image, instructions)
                        response_text = analysis_result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if response_text:
                            await message.channel.send(response_text)
                            print(Fore.MAGENTA + "Image analysis result: " + response_text + Fore.RESET)
                        else:
                            response_fail_message = "Sorry, I couldn't analyze the image."
                            await message.channel.send(response_fail_message)
                            print(Fore.YELLOW + response_fail_message + Fore.RESET)
                    except Exception as e:
                        error_details = str(e)
                        if hasattr(e, 'response') and e.response is not None:
                            error_details += f" Response: {e.response.text}"
                        formatted_error = format_error_message(error_details)
                        await message.channel.send(formatted_error)
                        print(Fore.RED + formatted_error + Fore.RESET)
                        
        return  # Return after processing images to avoid duplicate responses

    # If the bot should respond, then proceed with fetching history and Solr search
    if should_respond:
        # Extract and expand keywords
        expanded_keywords = await get_expanded_keywords(message.content)
        # Fetch message history and Solr results with expanded keywords
        messages_with_solr = await fetch_message_history(message.channel, message.author.name, expanded_keywords)

        # Initialize variables to store the response
        airesponse_chunks = []
        response = {}
        openai_api_error_occurred = False

        try:
            async with message.channel.typing():
                # Remove redundant messages from channel and Solr history
                filtered_messages = remove_redundant_messages(messages_with_solr)

                # Prepare the messages for sending to OpenAI
                system_message = {"role": "system", "content": chatgpt_behaviour}
                current_user_message = {"role": "user", "content": f"{message.author.name}: {message.content}"}
                assistant_prompt = {"role": "assistant", "content": "What is your reply? Keep the current conversation going, but utilize the chat history if it seems appropriate and relevant."}

                messages_for_openai = [system_message] + filtered_messages + [current_user_message, assistant_prompt]

                # Log the chat history for debugging
                print('Chat history for OpenAI:')
                for msg in messages_for_openai:
                    print(f'{Fore.GREEN}{msg["role"].capitalize()}: {Fore.YELLOW}{msg["content"]}{Fore.RESET}')

                # Set the max_tokens based on the type of response
                max_tokens = int(os.getenv("MAX_TOKENS_RANDOM")) if is_random_response else int(os.getenv("MAX_TOKENS"))

                # Generate a response using the OpenAI API
                response = await async_chat_completion(model=os.getenv("MODEL_CHAT"), messages=messages_for_openai, temperature=1.5, top_p=0.9, max_tokens=max_tokens)

                # Extract and print the token usage information
                if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                    total_tokens = response.usage.total_tokens
                    print(f"Total Tokens for text response: {total_tokens}")
                else:
                    print("Token usage information not available.")

                # Split the AI response into manageable chunks
                airesponse = response.choices[0].message.content
                airesponse_chunks = split_message(airesponse)
                # Introduce a delay based on the response length
                total_sleep_time = RATE_LIMIT * len(airesponse_chunks)
                await asyncio.sleep(total_sleep_time)

        except openai.OpenAIError as e:
            # Log and send OpenAI specific errors
            error_msg = f"Error: OpenAI API Error - {e}"
            print(Fore.RED + error_msg + Fore.RESET)
            airesponse = f"An error has occurred with your request. Please try again. Error details: {e}"
            openai_api_error_occurred = True
            await message.channel.send(airesponse)

        except Exception as e:
            # Log and send unexpected errors
            error_msg = f"Unexpected error: {e}"
            print(Fore.RED + error_msg + Fore.RESET)
            airesponse = "An unexpected error has occurred."

        # Send the response in chunks to avoid message limits
        if not openai_api_error_occurred:
            for chunk in airesponse_chunks:
                # Remove 'username:' prefix from the beginning of the response
                chunk = re.sub(r'^([^\s:]+(\s+[^\s:]+)?):\s*', '', chunk)

                # Send the response without tagging the user or adding a username prefix
                sent_message = await message.channel.send(chunk)
                print(bot.user, ":", Fore.RED + chunk + Fore.RESET)
                save_message_to_json_and_index_solr(sent_message.channel.id, str(bot.user), chunk, sent_message.created_at)  # Log bot's response
                await asyncio.sleep(RATE_LIMIT)

    # Process any commands included in the message
    await bot.process_commands(message)

# Initialize the bot with the Discord token from environment variables
discord_bot_token = os.getenv("DISCORD_TOKEN")
if discord_bot_token is None:
    raise ValueError("No Discord bot token found. Make sure to set the DISCORD_BOT_TOKEN environment variable.")
bot.run(discord_bot_token)



