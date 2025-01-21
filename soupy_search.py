"""
Search functionality for Soupy Bot
Provides DuckDuckGo search capabilities with rate limiting and result processing
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
from openai import OpenAI
import aiohttp
import trafilatura
from bs4 import BeautifulSoup
import json

# Configure logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio")
)

async def async_chat_completion(*args, **kwargs):
    """Wraps the OpenAI chat completion in an async context"""
    return await asyncio.to_thread(client.chat.completions.create, *args, **kwargs)

class SearchCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.search_rate_limits = defaultdict(list)
        self.MAX_SEARCHES_PER_MINUTE = 10
        self.ddgs = DDGS()
        self.session = aiohttp.ClientSession()

    async def cog_unload(self):
        """Cleanup when cog is unloaded"""
        if hasattr(self, 'session'):
            await self.session.close()

    def is_rate_limited(self, user_id: int) -> bool:
        """Check if user has exceeded search rate limit"""
        current_time = time.time()
        search_times = self.search_rate_limits[user_id]
        
        # Remove timestamps older than 60 seconds
        search_times = [t for t in search_times if current_time - t < 60]
        self.search_rate_limits[user_id] = search_times
        
        if len(search_times) >= self.MAX_SEARCHES_PER_MINUTE:
            return True
        
        self.search_rate_limits[user_id].append(current_time)
        return False

    async def fetch_article_content(self, url: str) -> str:
        """Fetch and extract the main content of an article"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    content = trafilatura.extract(html)
                    if content:
                        return content.strip()
                    return ""
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {e}")
            return ""

    async def select_relevant_articles(self, search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Select most relevant articles from search results"""
        try:
            # Format results for LLM
            formatted_results = "\n\n".join([
                f"[{i}] {result.get('title', 'No title')}\n"
                f"URL: {result.get('href', 'No URL')}\n"
                f"Preview: {result.get('body', 'No preview')}"
                for i, result in enumerate(search_results)
            ])

            response = await async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Return ONLY a JSON array of numbers."},
                    {"role": "user", "content": (
                        f"From these search results, select 3-5 most relevant articles. "
                        f"Return ONLY a JSON array of indices (e.g. [0,1,4]). No other text.\n\n"
                        f"{formatted_results}"
                    )}
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=50     # Reduced tokens since we only need numbers
            )

            try:
                selected_indices = json.loads(response.choices[0].message.content.strip())
                if isinstance(selected_indices, list):
                    return [search_results[i] for i in selected_indices if i < len(search_results)]
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON array")
                
            # Fall back to first 3 articles if selection fails
            return search_results[:3]

        except Exception as e:
            logger.error(f"Error selecting relevant articles: {e}")
            return search_results[:3]

    async def generate_final_response(self, query: str, articles: List[Dict[str, str]]) -> str:
        """Generate final response from full article contents"""
        try:
            # Add error handling for article formatting
            if not articles or not isinstance(articles, list):
                logger.error("Invalid articles format")
                return "❌ No valid articles found to cite."
                
            # Format articles with proper JSON structure and include full content
            formatted_articles = []
            for article in articles:
                content = await self.fetch_article_content(article.get('href', ''))
                if article.get('href') and (article.get('body') or content):
                    formatted_articles.append({
                        'title': article.get('title', 'No title'),
                        'source': article.get('source', 'Unknown'),
                        'href': article['href'],
                        'content': content or article.get('body', '')
                    })
            
            if not formatted_articles:
                return "❌ No articles with valid content found."
                
            # Format articles with full content for LLM
            formatted_articles_content = "\n\n".join([
                f"Article {i+1}:\n"
                f"Title: {article['title']}\n"
                f"Source: {article['source']}\n"
                f"URL: {article['href']}\n"
                f"Content: {article['content']}"
                for i, article in enumerate(formatted_articles)
            ])

            response = await async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=[
                    {"role": "system", "content": (
                        f"{os.getenv('BEHAVIOUR_SEARCH')}\n\n"
                        "‼️ ABSOLUTE CITATION REQUIREMENTS ‼️\n"
                        "1. EVERY SINGLE PARAGRAPH must contain an inline citation\n"
                        "2. NO EXCEPTIONS - if you can't cite it, don't say it\n"
                        "3. Format: [exact quote or close paraphrase](URL)\n"
                        "4. Citations must be IN the sentences, not after them\n"
                        "5. Never list sources separately - weave them into text"
                    )},
                    {"role": "user", "content": (
                        f"Search Query: {query}\n\n"
                        f"Create a response where EVERY sentence has an inline citation. Examples:\n\n"
                        f"✅ CORRECT: According to [Science Now](http://www.sciencenow.com/article.html), scientists have discovered a new species in the Amazon.\n"
                        f"❌ WRONG: Scientists discovered a new species. [Source](url)\n"
                        f"✅ CORRECT: [According to NASA](http://www.nasa.com/article/link.html), the mission succeeded, and [SpaceX reports](url2) the landing was perfect.\n"
                        f"❌ WRONG: The mission succeeded and the landing was perfect [Souce](url)\n\n"
                        f"Rules for your response:\n"
                        f"1. Every single paragraph MUST have [text](url) format citations\n"
                        f"2. Citations go INSIDE sentences, not at the end\n"
                        f"3. Use exact URLs from the articles\n"
                        f"4. Stay in character but NEVER compromise on citations\n"
                        f"5. If you can't cite something, don't include it\n"
                        f"6. ***Always cite the source of the information you are using***\n\n"
                        f"Here are the articles to analyze:\n\n"
                        f"{formatted_articles_content}"
                    )}
                ],
                temperature=0.7,
                max_tokens=1600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return "❌ An error occurred while generating the response."

    @app_commands.command(
        name="soupysearch",
        description="Performs a DuckDuckGo search and returns a comprehensive answer based on the results."
    )
    @app_commands.describe(query="The search query.")
    async def search_command(self, interaction: discord.Interaction, query: str):
        """Handle the /soupysearch command"""
        logger.info(f"🔍 Search requested by {interaction.user}: '{query}'")
        
        if self.is_rate_limited(interaction.user.id):
            await interaction.response.send_message(
                "⚠️ You are searching too quickly. Please wait a moment.",
                ephemeral=True
            )
            return
        
        await interaction.response.defer()
        
        try:
            # First tier: Get initial search results
            initial_results = list(await asyncio.to_thread(
                self.ddgs.text,
                query,
                max_results=20
            ))
            
            if not initial_results:
                await interaction.followup.send("❌ No results found.", ephemeral=True)
                return
            
            # Log initial results
            logger.debug(f"Initial search results: {initial_results}")
            
            # Have LLM select most relevant articles
            selected_results = await self.select_relevant_articles(initial_results)
            logger.debug(f"Selected articles: {selected_results}")
            
            # Second tier: Fetch full content for selected articles
            for result in selected_results:
                if result.get('href'):
                    result['full_content'] = await self.fetch_article_content(result['href'])
            
            # Generate final response
            final_response = await self.generate_final_response(query, selected_results)
            logger.debug(f"Final response: {final_response}")
            
            # Split and send response
            MAX_EMBED_LENGTH = 3900
            response_chunks = [final_response[i:i + MAX_EMBED_LENGTH] 
                             for i in range(0, len(final_response), MAX_EMBED_LENGTH)]
            
            for i, chunk in enumerate(response_chunks):
                embed = discord.Embed(
                    title=f"🔍 Search Results for: {query}" + 
                          (f" (Part {i+1}/{len(response_chunks)})" if len(response_chunks) > 1 else ""),
                    description=chunk,
                    color=discord.Color.green()
                )
                
                if i == 0:
                    await interaction.followup.send(
                        content=f"{interaction.user.mention}",
                        embed=embed,
                        allowed_mentions=discord.AllowedMentions(users=True)
                    )
                else:
                    await interaction.followup.send(embed=embed)
                
                if i < len(response_chunks) - 1:
                    await asyncio.sleep(1)
            
            logger.info(f"✅ Search completed for {interaction.user}")
            
        except Exception as e:
            logger.error(f"❌ Error in search command: {e}")
            await interaction.followup.send(
                f"❌ An error occurred: {str(e)}",
                ephemeral=True
            )

async def setup(bot):
    """Setup function for loading the cog"""
    await bot.add_cog(SearchCog(bot))
