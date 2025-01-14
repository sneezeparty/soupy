import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Optional

import discord
from discord.ext import tasks, commands
import pytz
from openai import OpenAI
from soupy_remastered import client, async_chat_completion

logger = logging.getLogger(__name__)

# Response styles configuration
RESPONSE_STYLES = {
    'normal': {
        'weight': 0.7,
        'instruction': 'Respond in a casual, conversational manner. Keep it brief and natural.'
    },
    'sarcastic': {
        'weight': 0.15,
        'instruction': 'Respond with mild sarcasm and wit. Be slightly edgy but not mean.'
    },
    'philosophical': {
        'weight': 0.05,
        'instruction': 'Respond with a thoughtful or philosophical observation. Keep it accessible.'
    },
    'random_fact': {
        'weight': 0.05,
        'instruction': 'Share an interesting or obscure fact, but make it sound casual and conversational.'
    },
    'pop_culture': {
        'weight': 0.05,
        'instruction': 'Make a reference to movies, TV, music, or gaming, but keep it subtle and natural.'
    }
}

# Initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio")
)


class Interjector(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.last_interjection = datetime.now(pytz.UTC)
        self.chance_per_check = 0.03
        self.min_time_between = timedelta(hours=7)  # Minimum 5 hours between interjections
        self.behaviour = os.getenv("INTERJECT", "You're a helpful Discord chatbot.")
        self.timezone = pytz.timezone(os.getenv("TIMEZONE", "UTC"))
        self.active_hours = {
            'start': 6,  # 6 AM
            'end': 21    # 9 PM
        }
        self.interject_check.start()

    def cog_unload(self):
        self.interject_check.cancel()

    async def get_random_response(self) -> Optional[str]:
        """Get a random response using the LLM functionality with varied styles"""
        try:
            # Select a random response style based on weights
            total_weight = sum(style['weight'] for style in RESPONSE_STYLES.values())
            r = random.uniform(0, total_weight)
            
            cumulative_weight = 0
            selected_style = None
            for style_name, style_info in RESPONSE_STYLES.items():
                cumulative_weight += style_info['weight']
                if r <= cumulative_weight:
                    selected_style = style_info
                    break
            
            # Combine the style instruction with existing behavior and interjection guidance
            system_message = (
                f"{self.behaviour}\n\n"
                f"{selected_style['instruction']}\n\n"
                "You are currently making an interjection into a random chat channel. "
                "Keep your response brief (1-2 sentences), casual, and natural, as if you're "
                "just popping into the conversation with a random thought or observation. "
                "Avoid asking questions unless they're rhetorical. Don't be overly positive "
                "or enthusiastic. Be somewhat sardonic but not mean."
            )
            
            logger.debug(f"🎭 Selected response style: {next((k for k, v in RESPONSE_STYLES.items() if v == selected_style), 'unknown')}")
            
            # Use the global async_chat_completion function
            response = await async_chat_completion(
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
            time_since_last = now - self.last_interjection
            if time_since_last < self.min_time_between:
                logger.debug(f"Skipping interjection check - only {time_since_last.total_seconds()/3600:.1f} hours since last")
                return
            
            # Only run during active hours in configured timezone
            if self.active_hours['start'] <= current_hour < self.active_hours['end']:
                if random.random() < self.chance_per_check:
                    await asyncio.sleep(random.randint(0, 60))
                    channel = await self.get_random_channel()
                    if channel:
                        message = await self.get_random_response()
                        if message:
                            await channel.send(message)
                            self.last_interjection = now
                            logger.info(f"🗣️ Interjected message in {channel.guild.name}/{channel.name}: '{message}'")
                else:
                    logger.debug("Random check failed - no interjection this time")
            else:
                logger.debug(f"Outside active hours (current hour: {current_hour})")
        except Exception as e:
            logger.error(f"Error in interject_check: {e}")

    @interject_check.before_loop
    async def before_interject_check(self):
        """Wait until the bot is ready before starting the loop"""
        await self.bot.wait_until_ready()

async def setup(bot):
    """Setup function for loading the cog"""
    await bot.add_cog(Interjector(bot))
