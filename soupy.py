from dotenv import load_dotenv
import os
import discord
from discord.ext import commands
import openai
import random
import time
from colorama import init, Fore

# enable colors in windows cmd
init(convert=True)

# Load environment variables
load_dotenv()

# rate limit in seconds (sleep this many seconds between each OpenAI API request)
RATE_LIMIT = 1.0
openai.api_key = os.environ.get("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
chatgpt_behaviour = "You are a Discord chatbot named Soupy Dafoe that reluctantly answers questions with sarcastic and witty responses, and you are preoccupied with soup."
messages = []

def should_bot_respond_to_message(message):
    if message.author == bot.user:
        return False
    return bot.user in message.mentions or (random.random() < 0.02)


@bot.event
async def on_message(message):
    if should_bot_respond_to_message(message):
        try:
            # Get the channel object from the message object
            channel = message.channel

            # Retrieve the last 10 messages from the channel history
            messages = []
            async for message in channel.history(limit=10):
                messages.append({"role": "system", "content": message.content})
            messages = messages[::-1]

            # Append the messages to the chat history
            messages = [
                {"role": "user", "content": chatgpt_behaviour},
                {"role": "system", "content": "Here are the last 10 messages"}
            ] + messages
            messages += [{"role": "system", "content": "What is your reply?"}]

            print(f'invoking openapi with messages: {messages}')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            airesponse = (response.choices[0].message.content)
        except openai.errors.ApiError as e:
            print(f"Error: OpenAI API Error - {e}")
            airesponse = "I'm not feeling very chatty right now. Try again later?"
        except Exception as e:
            print(Fore.BLUE + f"Error: {e}" + Fore.RESET)
            airesponse = "Wuh?"
        await message.channel.send(airesponse)
        print(message.author, ":", message.content)
        print(bot.user, ":", Fore.RED + airesponse + Fore.RESET)
        print("Tokens:", Fore.GREEN + str(
            response["usage"]["total_tokens"]) + Fore.RESET)  # displays total tokens used in the console
        time.sleep(RATE_LIMIT)  # rate limit on OpenAI queries

bot.run(os.environ.get("DISCORD_TOKEN"))
