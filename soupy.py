from dotenv import load_dotenv
import os
import discord
from discord.ext import commands
import openai
import random
import time
import asyncio
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

# The bot will respond whenever it is @mentioned, and also in whatever channel is specified in .env CHANNEL_ID.
# It will respond in CHANNEL_ID channel to every message, even when it is not mentioned.



def should_bot_respond_to_message(message):
    if message.author == bot.user:
        return False, False
    is_random_response = random.random() < 0.02
    return (bot.user in message.mentions or is_random_response or
            message.channel.id == int(os.environ.get("CHANNEL_ID"))), is_random_response


# Split message into multiple chunks of at least min_length characters
def split_message(message_content, min_length=1500):
    chunks = []
    remaining = message_content
    while len(remaining) > min_length:
        chunk = remaining[:min_length]
        last_punctuation_index = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
        if last_punctuation_index == -1:
            last_punctuation_index = min_length - 1
        chunks.append(chunk[:last_punctuation_index+1])
        remaining = remaining[last_punctuation_index+1:]
    chunks.append(remaining)
    return chunks


@bot.event
async def on_message(message):
    global messages
    should_respond, is_random_response = should_bot_respond_to_message(message)  # Update this line
    if should_respond:
        try:
            # Get the channel object from the message object
            channel = message.channel

            # Send a typing indicator while generating a response
            async with channel.typing():
                # Retrieve the last n messages from the channel history
                messages = []
                async for message in channel.history(limit=int(os.environ.get("HISTORYLENGTH"))):
                    messages.append({"role": "system", "content": message.content})
                messages = messages[::-1]

                # Append the messages to the chat history
                messages = [
                    {"role": "system", "content": chatgpt_behaviour},
                    {"role": "user", "content": "Here is the message history:"}
                ] + messages
                messages += [{"role": "system", "content": "What is your reply?"}]

                print(f'chat history: {messages}')
                # Update max_tokens assignment based on whether it's a random response
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

                # Split response into multiple messages if it is longer than min_length characters
                airesponse_chunks = split_message(airesponse)

                # Calculate the total time to sleep
                total_sleep_time = RATE_LIMIT * len(airesponse_chunks)

                # Sleep while the typing indicator is active
                await asyncio.sleep(total_sleep_time)

        except openai.error.OpenAIError as e:  # basic error handling
            print(f"Error: OpenAI API Error - {e}")
            airesponse = "An error has occurred with your request.  Please try again."
        except Exception as e:
            print(Fore.BLUE + f"Error: {e}" + Fore.RESET)
            airesponse = "Wuh?"

        # Send each chunked message individually
        for chunk in airesponse_chunks:
            await message.channel.send(chunk)
            print(bot.user, ":", Fore.RED + chunk + Fore.RESET)
            time.sleep(RATE_LIMIT)  # rate limit on OpenAI queries

        print("Total Tokens:", Fore.GREEN + str(
            response["usage"]["total_tokens"]) + Fore.RESET)  # displays total tokens used in the console


bot.run(os.environ.get("DISCORD_TOKEN"))
