"""
Image search functionality for Soupy Bot
Provides DuckDuckGo image search capabilities with rate limiting and result processing
"""

import discord
from discord import app_commands
from discord.ext import commands
import logging
from collections import defaultdict
import time
import os
import asyncio
from typing import Optional, List, Dict
from duckduckgo_search import DDGS
import aiohttp
from io import BytesIO
import random

# Configure logging
logger = logging.getLogger(__name__)

class ImageSearchCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.search_rate_limits = defaultdict(list)
        self.MAX_SEARCHES_PER_MINUTE = 10
        self.ddgs = DDGS()
        self.session = aiohttp.ClientSession()

    async def cog_unload(self):
        # Cleanup when cog is unloaded
        if hasattr(self, 'session'):
            await self.session.close()

    async def is_rate_limited(self, user_id: int) -> bool:
        # Check if user has exceeded rate limits
        current_time = time.time()
        search_times = self.search_rate_limits.get(user_id, [])
        
        # Clean up old timestamps
        search_times = [t for t in search_times if current_time - t < 60]
        self.search_rate_limits[user_id] = search_times
        
        if len(search_times) >= self.MAX_SEARCHES_PER_MINUTE:
            return True
        
        self.search_rate_limits[user_id].append(current_time)
        return False

    async def fetch_image(self, url: str) -> Optional[bytes]:
        # Fetch image data from URL with error handling
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.read()
                logger.error(f"Failed to fetch image from {url}: HTTP {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching image from {url}: {e}")
            return None

    @app_commands.command(
        name="soupyimage",
        description="Performs a DuckDuckGo image search and returns a random image from the top 300 results."
    )
    @app_commands.describe(query="The search query for finding images.")
    async def image_search_command(self, interaction: discord.Interaction, query: str):
        # Handle the /soupyimage command
        start_time = time.time()
        logger.info(f"🔍 Image search requested by {interaction.user}: '{query}'")
        
        if await self.is_rate_limited(interaction.user.id):
            await interaction.response.send_message(
                "⚠️ You are searching too quickly. Please wait a moment.",
                ephemeral=True
            )
            return
        
        await interaction.response.defer()
        
        try:
            # Get image search results
            results = list(await asyncio.to_thread(
                self.ddgs.images,
                query,
                max_results=300  # Get top 300 results
            ))
            
            if not results:
                await interaction.followup.send("❌ No images found.", ephemeral=True)
                return
            
            # Select a random image from the results
            selected_index = random.randint(0, len(results) - 1)
            selected_image = results[selected_index]
            image_url = selected_image.get('image')
            title = selected_image.get('title', 'Untitled')
            source_url = selected_image.get('url', '')
            
            if not image_url:
                await interaction.followup.send("❌ Failed to get image URL.", ephemeral=True)
                return
            
            # Fetch the image data
            image_data = await self.fetch_image(image_url)
            if not image_data:
                await interaction.followup.send("❌ Failed to fetch image.", ephemeral=True)
                return
            
            # Create a Discord File object
            image_file = discord.File(BytesIO(image_data), filename="image.png")
            
            # Create an embed for the image with result number
            embed = discord.Embed(
                title=f"🔍 Image Search: {query}",
                description=f"Result #{selected_index + 1} of {len(results)}",
                color=discord.Color.blue()
            )
            
            # Add source information if available
            if source_url:
                embed.add_field(
                    name="Source",
                    value=f"[{title}]({source_url})",
                    inline=False
                )
            
            # Set the image in the embed
            embed.set_image(url="attachment://image.png")
            
            # Add footer with timing information
            elapsed_time = round(time.time() - start_time, 2)
            embed.set_footer(text=f"Search completed in {elapsed_time} seconds")
            
            # Send the image with embed
            await interaction.followup.send(embed=embed, file=image_file)
            logger.info(f"✅ Image search completed for {interaction.user}")
            
        except Exception as e:
            logger.error(f"❌ Error in image search command: {e}")
            await interaction.followup.send(
                f"❌ An error occurred: {str(e)}",
                ephemeral=True
            )

async def setup(bot):
    # Add the cog to the bot
    try:
        await bot.add_cog(ImageSearchCog(bot))
        logger.info("✅ ImageSearchCog loaded successfully")
        
        # Log registered commands
        for command in bot.tree.get_commands():
            if command.name == "soupyimage":
                logger.info(f"✅ /soupyimage command registered successfully")
                return
        logger.warning("⚠️ /soupyimage command not found in command tree")
    except Exception as e:
        logger.error(f"❌ Failed to load ImageSearchCog: {e}")
