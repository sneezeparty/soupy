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


# Load environment variables
load_dotenv()

# Initialize the OpenAI client with your API key for both image and chat functionalities
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("No OpenAI API key found. Make sure to set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=openai_api_key)
openai.api_key = openai_api_key

# Initialize Discord intents and bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize colorama for colored console output
init()

# Rate limiting
RATE_LIMIT = 0.5

# Get behavior settings from .env
chatgpt_behaviour = os.getenv("BEHAVIOUR")

# Determine if the bot should respond to the message
def should_bot_respond_to_message(message):
    allowed_channel_ids = [int(channel_id) for channel_id in os.getenv("CHANNEL_IDS").split(',')]
    if message.author == bot.user:
        return False, False
    is_random_response = random.random() < 0.01
    mentioned_users = [user for user in message.mentions if not user.bot]
    if mentioned_users or not (bot.user in message.mentions or is_random_response or message.channel.id in allowed_channel_ids):
        return False, False
    return (bot.user in message.mentions or is_random_response or message.channel.id in allowed_channel_ids), is_random_response

# Split message into chunks
def split_message(message_content, min_length=1500):
    chunks = []
    remaining = message_content
    while len(remaining) > min_length:
        chunk = remaining[:min_length]
        last_punctuation_index = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
        if last_punctuation_index == -1:
            last_punctuation_index = min_length - 1
        chunks.append(chunk[:last_punctuation_index + 1])
        remaining = remaining[last_punctuation_index + 1:]
    chunks.append(remaining)
    return chunks

async def async_chat_completion(*args, **kwargs):
    return await asyncio.to_thread(openai.chat.completions.create, *args, **kwargs)

async def fetch_message_history(channel):
    message_history = []
    async for message in channel.history(limit=int(os.getenv("HISTORYLENGTH"))):
        # Skip messages that contain embeds (likely bot messages or system messages)
        if message.embeds:
            continue
        # Skip messages that start with the command prefix '!generate'
        if message.content.startswith('!generate'):
            continue
        # Assuming that system messages do not come from 'User' accounts (you might need to adjust this)
        role = "system" if message.author.bot else "user"
        message_history.append({"role": role, "content": message.content})
    return message_history[::-1]


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.command()
async def generate(ctx, *, prompt: str):
    async with ctx.typing():
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

    embed = discord.Embed(title="Generated Image", description=f"Prompt: {prompt}")
    embed.set_image(url=image_url)
    await ctx.send(embed=embed)

@bot.event
async def on_message(message):
    await bot.process_commands(message)  # Ensure commands are processed

    if message.author == bot.user or message.embeds:
        return
        
    should_respond, is_random_response = should_bot_respond_to_message(message)
    if should_respond:
        airesponse_chunks = []
        response = {}
        openai_api_error_occurred = False

        try:
            channel = message.channel
            async with channel.typing():
                # Fetch and filter the message history
                messages = await fetch_message_history(channel)
                
                messages = [
                               {"role": "system", "content": chatgpt_behaviour},
                               {"role": "user", "content": "Here is the message history:"}
                           ] + messages
                messages += [{"role": "assistant", "content": "What is your reply?"}, {"role": "system", "content": chatgpt_behaviour}]
                
                        # Printing each chat message on a new line for readability
                print('Chat history:')
                for msg in messages:
                    role = msg['role']
                    content = msg['content'].replace('\n', '\n               ')  # Replace newlines in content to align multi-line messages
                    print(f'  {role.capitalize()}: {content}')
                
                if is_random_response:
                    max_tokens = int(os.getenv("MAX_TOKENS_RANDOM"))
                else:
                    max_tokens = int(os.getenv("MAX_TOKENS"))

                response = await async_chat_completion(
                    model=os.getenv("MODEL_CHAT"),
                    messages=messages,
                    temperature=1.5,
                    top_p=0.9,
                    max_tokens=max_tokens
                )
                airesponse = response.choices[0].message.content
                airesponse_chunks = split_message(airesponse)
                total_sleep_time = RATE_LIMIT * len(airesponse_chunks)
                await asyncio.sleep(total_sleep_time)
        
        except openai.OpenAIError as e:
            print(f"Error: OpenAI API Error - {e}")
            airesponse = f"An error has occurred with your request. Please try again. Error details: {e}"
            openai_api_error_occurred = True
            await message.channel.send(airesponse)

        except Exception as e:
            print(Fore.BLUE + f"Error: {e}" + Fore.RESET)
            airesponse = "Wuh?"

        if not openai_api_error_occurred:
            for chunk in airesponse_chunks:
                await message.channel.send(chunk)
                print(bot.user, ":", Fore.RED + chunk + Fore.RESET)
                time.sleep(RATE_LIMIT)

        if 'usage' in response:
            print("Total Tokens:", Fore.GREEN + str(response["usage"]["total_tokens"]) + Fore.RESET)

# Run the bot with your token
discord_bot_token = os.getenv("DISCORD_TOKEN")
if discord_bot_token is None:
    raise ValueError("No Discord bot token found. Make sure to set the DISCORD_BOT_TOKEN environment variable.")
bot.run(discord_bot_token)
