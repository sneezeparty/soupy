import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Optional

import discord
from discord.ext import tasks, commands
import pytz

logger = logging.getLogger(__name__)

class Interjector(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.last_interjection = datetime.now(pytz.UTC)
        self.chance_per_check = 0.1
        self.min_time_between = timedelta(hours=5)  # Minimum 6 hours between interjections
        self.behaviour = os.getenv("INTERJECT", "You're a stupid bot.")
        self.timezone = pytz.timezone(os.getenv("TIMEZONE", "UTC"))
        self.interject_check.start()

    def cog_unload(self):
        self.interject_check.cancel()

    async def get_random_response(self) -> Optional[str]:
        """Get a random response using the bot's existing LLM functionality"""
        try:
            # Use the bot's BEHAVIOUR setting but add specific interjection guidance
            system_message = (
                f"{self.behaviour}\n\n"
                "You are currently making an interjection into a random chat channel. "
                "Keep your response brief (1-2 sentences), casual, and natural, as if you're "
                "just popping into the conversation with a random thought or observation or fact "
                "or response to something someone said earlier."
            )
            
            response = await self.bot.async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": "Generate a casual, interjection or observation."}
                ],
                temperature=0.9,
                max_tokens=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating random interjection: {e}")
            return None

    async def get_random_channel(self) -> Optional[discord.TextChannel]:
        """Get a random text channel"""
        available_channels = []
        
        for guild in self.bot.guilds:
            for channel in guild.text_channels:
                if channel.permissions_for(guild.me).send_messages:
                    available_channels.append(channel)
        
        return random.choice(available_channels) if available_channels else None

    @tasks.loop(minutes=15)
    async def interject_check(self):
        """Check if we should interject a message"""
        try:
            now = datetime.now(self.timezone)
            current_hour = now.hour
            
            # Only proceed if enough time has passed since last interjection
            if now - self.last_interjection < self.min_time_between:
                return
            
            # Only run between 6am and 9pm in configured timezone
            if 6 <= current_hour < 22:
                if random.random() < self.chance_per_check:
                    await asyncio.sleep(random.randint(0, 60))
                    channel = await self.get_random_channel()
                    if channel:
                        message = await self.get_random_response()
                        if message:
                            await channel.send(message)
                            self.last_interjection = now
                            logger.info(f"🗣️ Interjected message in {channel.guild.name}/{channel.name}: '{message}'")
        except Exception as e:
            logger.error(f"Error in interject_check: {e}")

    @interject_check.before_loop
    async def before_interject_check(self):
        """Wait until the bot is ready before starting the loop"""
        await self.bot.wait_until_ready()

async def setup(bot):
    """Setup function for loading the cog"""
    await bot.add_cog(Interjector(bot))
