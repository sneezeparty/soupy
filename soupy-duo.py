from openai import OpenAI
import os
from dotenv import load_dotenv
import discord
from discord.ext import commands

# Load environment variables
load_dotenv()

# Initialize the OpenAI client with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("No OpenAI API key found. Make sure to set the OPENAI_API_KEY environment variable in a .env file.")
client = OpenAI(api_key=openai_api_key)

# Define the bot with the specified intents and command prefix
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.command()
async def generate(ctx, *, prompt: str):
    async with ctx.typing():  # This will show "bot is typing..." in the channel
        # Generate the image using the provided API
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

    # Create an embed to display the image
    embed = discord.Embed(title="Generated Image", description=f"Prompt: {prompt}")
    embed.set_image(url=image_url)
    await ctx.send(embed=embed)

# Run the bot with your token
discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
if discord_bot_token is None:
    raise ValueError("No Discord bot token found. Make sure to set the DISCORD_BOT_TOKEN environment variable in a .env file.")
bot.run(discord_bot_token)
