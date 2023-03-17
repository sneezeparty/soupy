from dotenv import load_dotenv
import os
import discord
from discord.ext import commands
import openai
import random
import time
from colorama import init, Fore

# enable colors in windows cmd console
init(convert=True)

# Load environment variables
load_dotenv()

RATE_LIMIT = 1.0  # rate limit in seconds (sleep this many seconds between each OpenAI API request)
openai.api_key = os.environ.get("OPENAI_API_KEY")  # insert your own openai API key here
intents = discord.Intents.default()  # by default, it uses all intents, but you can change this
intents.members = True
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
chatgpt_behaviour = os.environ.get("BEHAVIOUR")  # this is the .env variable to alter the bots "personality"
messages = []


def should_bot_respond_to_message(message):
    if message.author == bot.user:
        return False, False
    is_random_response = random.random() < 0.02  # It will also respond to 2% of every message in every channel that it can access.
    return (bot.user in message.mentions or is_random_response or
            message.channel.id == int(os.environ.get("CHANNEL_ID"))), is_random_response


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


@bot.event
async def on_message(message):
    global messages
    should_respond, is_random_response = should_bot_respond_to_message(message)
    if should_respond:
        try:
            channel = message.channel

            messages = []
            async for msg in channel.history(limit=int(os.environ.get("HISTORYLENGTH"))):
                messages.append({"role": "system", "content": msg.content})
            messages = messages[::-1]

            messages = [
                           {"role": "system", "content": chatgpt_behaviour},
                           {"role": "user", "content": "Here is the message history:"}
                       ] + messages
            messages += [{"role": "system", "content": "What is your reply?"}]

            print(f'chat history: {messages}')
            if is_random_response:
                max_tokens = int(os.environ.get("MAX_TOKENS_RANDOM"))
            else:
                max_tokens = int(os.environ.get("MAX_TOKENS"))

            response = openai.ChatCompletion.create(
                model=os.environ.get("MODEL"),
                messages=messages,
                temperature=1.5,
                top_p=0.9,
                max_tokens=max_tokens
            )
            airesponse = response.choices[0].message.content
        except openai.error.OpenAIError as e:
            print(f"Error: OpenAI API Error - {e}")
            airesponse = "An error has occurred with your request.  Please try again."
        except Exception as e:
            print(Fore.BLUE + f"Error: {e}" + Fore.RESET)
            airesponse = "Wuh?"

        airesponse_chunks = split_message(airesponse)

        for chunk in airesponse_chunks:
            if is_random_response:
                await message.channel.send(chunk)
            else:
                await message.channel.send(f'{message.author.mention} {chunk}')
            print(bot.user, ":", Fore.RED + chunk + Fore.RESET)
            time.sleep(RATE_LIMIT)

        print("Total Tokens:", Fore.GREEN + str(
            response["usage"]["total_tokens"]) + Fore.RESET)  # displays total tokens used in the console


bot.run(os.environ.get("DISCORD_TOKEN"))
