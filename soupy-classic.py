# This is an older version of soupy, but it still works.  It includes all image functions and chat functions, but it does not have a permanent memory function.  It does not use Flux.
# Importing necessary libraries
from openai import OpenAI
import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import openai
import random
import time
import asyncio
from colorama import init, Fore
import io
from io import BytesIO
import requests
from PIL import Image
import base64

# Load environment variables (like API keys) from .env file
load_dotenv()

# Initialize colorama for colored console output in the terminal
init(autoreset=True)

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
RATE_LIMIT = 0.5

# Retrieve behavior settings from environment variables
chatgpt_behaviour = os.getenv("BEHAVIOUR")
transform_behaviour = os.getenv("TRANSFORM")

# Error formatting
def format_error_message(error):
    try:
        if isinstance(error, openai.error.OpenAIError):
            # Handling OpenAI specific errors
            return str(error)
        elif hasattr(error, 'response') and error.response is not None:
            # Parsing error response from HTTP requests
            error_json = error.response.json()
            if 'error' in error_json:
                if 'message' in error_json['error']:
                    return f"Error: {error_json['error']['message']}"
                else:
                    return "An OpenAI API error occurred."
            else:
                return "An unknown error occurred."
        else:
            # Returning the error message directly
            return str(error)
    except Exception as e:
        # Fallback for unexpected error formats
        return "An unexpected error occurred in formatting the error."


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

# Asynchronous function to get chat completions from OpenAI
async def async_chat_completion(*args, **kwargs):
    response = await asyncio.to_thread(openai.chat.completions.create, *args, **kwargs)
    return response
    

# Retrieve a specified number of recent messages from a channel for context
async def fetch_message_history(channel):
    history_length = int(os.getenv("HISTORYLENGTH", 15))
    message_history = []
    async for message in channel.history(limit=history_length * 2):
        if len(message_history) < history_length and not message.author.bot and message.content:
            message_history.append({"role": "user" if message.author != bot.user else "assistant", "content": message.content})
    return message_history[::-1]

# Process an image URL and return a base64 encoded string
async def encode_discord_image(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        if max(image.size) > 1000:
            image.thumbnail((1000, 1000))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(Fore.RED + f"Error in encode_discord_image: {e}" + Fore.RESET)
        return None

# Analyze an image and return the AI's description
async def analyze_image(base64_image, instructions):
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{"role": "user", "content": [{"type": "text", "text": instructions}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
        "max_tokens": 300
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
    print(f'Logged in as {bot.user.name}')

# Command to generate an image based on a text prompt
@bot.command()
async def generate(ctx, *, prompt: str):
    try:
        print(f"Creating image based on: {prompt}")
        async with ctx.typing():
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            image_file = BytesIO(image_data)
            image_file.seek(0)
            image_discord = discord.File(fp=image_file, filename='image.png')

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
                    
                    # Analyze the image and get its description
                    description_result = await analyze_image(base64_image, "Describe this image, you give detailed and accurate descriptions, be specific in whatever ways you can, such as but not limited to colors, species, poses, orientations, objects, and contexts.")
                    if 'choices' in description_result and description_result['choices']:
                        original_description = description_result['choices'][0].get('message', {}).get('content', '')
                        if not original_description.strip():
                            raise ValueError("Failed to generate an original description for the image.")
                    else:
                        raise ValueError("Invalid response format from image analysis.")

                    print(Fore.BLUE + "Original Description: " + original_description + Fore.RESET)  # Log original description
                    # Prepare a prompt to integrate the user's instructions into the description
                    prompt = f"Rewrite the following description to incorporate the given transformation.\n\nOriginal Description: {original_description}\n\nTransformation: {instructions}\n\nTransformed Description:"

                    # Use GPT to rewrite the description
                    rewriting_result = await async_chat_completion(
                        model="gpt-4",
                        messages=[{"role": "system", "content": transform_behaviour},
                                  {"role": "user", "content": prompt}],
                        max_tokens=250
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
                        prompt=modified_description,
                        size="1024x1024",
                        quality="standard",
                        n=1,
                    )
                    new_image_url = response.data[0].url
                    new_image_data = requests.get(new_image_url).content
                    new_image_file = BytesIO(new_image_data)
                    new_image_file.seek(0)
                    new_image_discord = discord.File(fp=new_image_file, filename='transformed_image.png')
                    await ctx.send(f"Transformed Image:\nOriginal instructions: {instructions}", file=new_image_discord)

                except Exception as e:
                    formatted_error = format_error_message(e)
                    await ctx.send(f"An error occurred during the transformation: {formatted_error}")
                    print(Fore.RED + formatted_error + Fore.RESET)
    else:
        await ctx.send("Please attach an image with the !transform command.")



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

    # Determine if the bot should react to this particular message
    should_respond, is_random_response = should_bot_respond_to_message(message)

    # Check if the message is in a channel where the bot responds to all messages
    always_respond_channel = message.channel.id in [int(cid) for cid in os.getenv("CHANNEL_IDS", "").split(',')]

    # Check if the bot is mentioned in the message
    is_mentioned = bot.user in message.mentions

    # Process image analysis only if the bot is mentioned with an image or in a specific channel
    if message.attachments and (is_mentioned or always_respond_channel):
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
                        formatted_error = format_error_message(e)
                        await message.channel.send(formatted_error)
                        print(Fore.RED + formatted_error + Fore.RESET)
        return  # Return after processing images to avoid duplicate responses

    # Continue with other responses if the bot should respond
    if should_respond:    
        # Initialize variables to store the response
        airesponse_chunks = []
        response = {}
        openai_api_error_occurred = False

        try:
            # Indicate in the Discord channel that the bot is processing a message
            async with message.channel.typing():
                # Process and respond to text messages
                messages = await fetch_message_history(message.channel)
                messages = [{"role": "system", "content": chatgpt_behaviour}, {"role": "user", "content": "Here is the message history:"}] + messages
                messages += [{"role": "assistant", "content": "What is your reply?"}, {"role": "system", "content": chatgpt_behaviour}]

                # Log the chat history for debugging
                print('Chat history:')
                for msg in messages:
                    print(f'{Fore.GREEN}{msg["role"].capitalize()}: {Fore.YELLOW}{msg["content"]}{Fore.RESET}')

                # Set the max_tokens based on the type of response
                max_tokens = int(os.getenv("MAX_TOKENS_RANDOM")) if is_random_response else int(os.getenv("MAX_TOKENS"))

                # Generate a response using the OpenAI API
                response = await async_chat_completion(model=os.getenv("MODEL_CHAT"), messages=messages, temperature=1.5, top_p=0.9, max_tokens=max_tokens)
                # Print the complete API response for debugging
                #print("Complete API Response:", response)

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
                await message.channel.send(chunk)
                print(bot.user, ":", Fore.RED + chunk + Fore.RESET)
                time.sleep(RATE_LIMIT)

    # Process any commands included in the message
    await bot.process_commands(message)


# Initialize the bot with the Discord token from environment variables
discord_bot_token = os.getenv("DISCORD_TOKEN")
if discord_bot_token is None:
    raise ValueError("No Discord bot token found. Make sure to set the DISCORD_BOT_TOKEN environment variable.")
bot.run(discord_bot_token)
