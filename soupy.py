from dotenv import load_dotenv
import os
import discord
from discord.ext import commands
import openai
import random
import time
import asyncio
from colorama import init, Fore

# constants and settings
load_dotenv()  # Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key
init(convert=True)  # enable colors in windows cmd console
RATE_LIMIT = 0.5  # rate limit in seconds
intents = discord.Intents.default()
intents.messages = True  # This is for general message content outside of direct messages.
intents.message_content = True  # This is for the content of the message, which is now considered privileged.
bot = commands.Bot(command_prefix='!', intents=intents)
chatgpt_behaviour = os.getenv("BEHAVIOUR")
messages = []


# The bot will respond whenever it is @mentioned, and also in whatever channels are specified in .env CHANNEL_IDS.
# It will respond in CHANNEL_IDS channel to every message, even when it is not mentioned.
# Soupy won't respond if someone else is @mentioned in the channel where it responds to all messages.

def should_bot_respond_to_message(message):
    allowed_channel_ids = [int(channel_id) for channel_id in os.environ.get("CHANNEL_IDS").split(',')]
    if message.author == bot.user:
        return False, False
    is_random_response = random.random() < 0.01
    mentioned_users = [user for user in message.mentions if not user.bot]  # Get all non-bot mentioned users
    if mentioned_users or not (
            bot.user in message.mentions or is_random_response or message.channel.id in allowed_channel_ids):
        return False, False

    return (bot.user in message.mentions or is_random_response or
            message.channel.id in allowed_channel_ids), is_random_response


# Split message into multiple chunks of at least min_length characters.
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
    async for message in channel.history(limit=int(os.environ.get("HISTORYLENGTH"))):
        message_history.append({"role": "system", "content": message.content})
    return message_history[::-1]


@bot.event
async def on_message(message):
    global messages
    if message.author == bot.user:
        return
        
    # Fetch message history
    messages = await fetch_message_history(message.channel)

    should_respond, is_random_response = should_bot_respond_to_message(message)
    if should_respond:
        airesponse_chunks = []
        response = {}
        openai_api_error_occurred = False  # Add a flag variable here

        try:
            channel = message.channel
            async with channel.typing():

                messages = [
                               {"role": "system", "content": chatgpt_behaviour},
                               {"role": "user", "content": "Here is the message history:"}
                           ] + messages
                messages += [{"role": "assistant", "content": "What is your reply?"},{"role": "system", "content": chatgpt_behaviour}]

                print(f'chat history: {messages}')
                if is_random_response:
                    max_tokens = int(os.environ.get("MAX_TOKENS_RANDOM"))
                else:
                    max_tokens = int(os.environ.get("MAX_TOKENS"))

                response = await async_chat_completion(
                    model=os.environ.get("MODEL"),
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
            openai_api_error_occurred = True  # Set the flag to True when an error occurs

            # Send the error message to the channel
            await message.channel.send(airesponse)

        except Exception as e:
            print(Fore.BLUE + f"Error: {e}" + Fore.RESET)
            airesponse = "Wuh?"

        # Send the response to the channel if there was no OpenAI API error
        if not openai_api_error_occurred:
            for chunk in airesponse_chunks:
                await message.channel.send(chunk)
                print(bot.user, ":", Fore.RED + chunk + Fore.RESET)
                time.sleep(RATE_LIMIT)

        if 'usage' in response:
            print("Total Tokens:", Fore.GREEN + str(
                response["usage"]["total_tokens"]) + Fore.RESET)


bot.run(os.environ.get("DISCORD_TOKEN"))
