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
import re

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

    async def is_rate_limited(self, user_id: int) -> bool:
        """Check if user has exceeded rate limits"""
        current_time = time.time()
        search_times = self.search_rate_limits.get(user_id, [])
        
        # Clean up old timestamps
        search_times = [t for t in search_times if current_time - t < 60]
        self.search_rate_limits[user_id] = search_times
        
        if len(search_times) >= self.MAX_SEARCHES_PER_MINUTE:
            return True
        
        self.search_rate_limits[user_id].append(current_time)
        return False

    async def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch and extract the main content of an article with improved error handling"""
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Try trafilatura first
                    content = trafilatura.extract(html)
                    if content:
                        return content.strip()
                    
                    # Fallback to BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                        element.decompose()
                    
                    # Get main content
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post'))
                    if main_content:
                        return main_content.get_text(strip=True, separator=' ')
                    
                    # Last resort: get body text
                    body = soup.find('body')
                    if body:
                        return body.get_text(strip=True, separator=' ')
                    
                    return None
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {e}")
            return None

    async def select_articles(self, search_results: List[Dict]) -> List[Dict]:
        """Select the 5 most relevant articles using improved selection criteria"""
        try:
            if len(search_results) <= 5:
                return search_results

            # Format articles for evaluation
            formatted_results = []
            for result in search_results:
                # Skip results without required fields
                if not all(key in result for key in ['title', 'body', 'href']):
                    continue
                    
                formatted_results.append({
                    'title': result['title'],
                    'preview': result.get('body', '')[:500],  # Limit preview length
                    'url': result['href']
                })

            if not formatted_results:
                return search_results[:5]

            # Create selection prompt
            prompt = (
                "Select the 5 most informative and relevant articles from these search results. "
                "Consider:\n"
                "1. Relevance to the topic\n"
                "2. Information quality and depth\n"
                "3. Source credibility\n"
                "4. Content uniqueness\n\n"
                "Respond ONLY with the numbers (0-based) of the 5 best articles, separated by spaces.\n\n"
            )
            
            for i, result in enumerate(formatted_results):
                prompt += f"[{i}] {result['title']}\n"
                prompt += f"URL: {result['url']}\n"
                prompt += f"Preview: {result['preview']}\n\n"

            response = await async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that selects the most relevant and informative articles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            # Parse indices and validate
            try:
                indices = [int(idx) for idx in response.choices[0].message.content.strip().split()]
                valid_indices = [i for i in indices if 0 <= i < len(search_results)]
                
                if len(valid_indices) >= 5:
                    return [search_results[i] for i in valid_indices[:5]]
            except:
                logger.warning("Failed to parse article selection response")
                
            return search_results[:5]

        except Exception as e:
            logger.error(f"Error selecting articles: {e}")
            return search_results[:5]

    async def generate_final_response(self, query: str, articles: List[Dict]) -> str:
        """Generate a conversational response with proper citations"""
        try:
            if not articles:
                return "❌ No articles found to analyze."

            # Process articles and extract content
            processed_articles = []
            total_tokens = 0
            MAX_TOKENS_PER_ARTICLE = 1000  # Limit tokens per article

            for article in articles:
                content = await self.fetch_article_content(article.get('href', ''))
                if not content:
                    continue

                # Truncate content to manage token count
                content = content[:MAX_TOKENS_PER_ARTICLE]
                processed_articles.append({
                    'title': article.get('title', 'Untitled'),
                    'source': article.get('source', 'Unknown Source'),
                    'url': article['href'],
                    'content': content
                })

            if not processed_articles:
                return "❌ Could not extract content from any articles."

            # Create system message with Soupy's personality and strong citation requirements
            system_message = (
                f"{os.getenv('BEHAVIOUR_SEARCH', '')}\n\n"
                "CRITICAL INSTRUCTIONS FOR RESPONSE GENERATION:\n"
                "1. You MUST include citations for every piece of information you provide\n"
                "2. Format ALL citations as [Source Name](URL) using Discord markdown\n"
                "3. Citations MUST be naturally integrated into your response\n"
                "4. EVERY paragraph or major point MUST have at least one citation\n"
                "5. Be conversational and engaging while maintaining accuracy\n"
                "6. Keep responses concise but informative\n"
                "7. Use Soupy's sarcastic and witty personality\n"
                "8. Organize the response in a clear, readable format\n"
                "9. DO NOT generate a response without citations\n"
                "10. If you reference multiple sources for a point, cite them all\n"
            )

            # Format content for LLM with emphasis on citation requirements
            content_prompt = (
                f"Search Query: {query}\n\n"
                "IMPORTANT: Your response MUST include citations from the following articles. "
                "Every significant piece of information MUST be backed by at least one citation. "
                "Here are the articles to analyze and cite:\n\n"
            )

            for i, article in enumerate(processed_articles, 1):
                content_prompt += (
                    f"Article {i}:\n"
                    f"Title: {article['title']}\n"
                    f"Source: {article['source']}\n"
                    f"URL: {article['url']}\n"
                    f"Content: {article['content']}\n\n"
                )

            content_prompt += (
                "REMINDER: Format your response as a natural conversation, but ensure EVERY "
                "significant point has a citation in [Source Name](URL) format. DO NOT skip citations.\n\n"
            )

            # Generate response with chunked content if needed
            try:
                response = await async_chat_completion(
                    model=os.getenv("LOCAL_CHAT"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": content_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                return response.choices[0].message.content.strip()

            except Exception as e:
                if "context length" in str(e).lower():
                    # Fallback to shorter content if context length is exceeded
                    logger.warning("Context length exceeded, falling back to shorter content")
                    shortened_articles = []
                    for article in processed_articles:
                        shortened_articles.append({
                            'title': article['title'],
                            'source': article['source'],
                            'url': article['url'],
                            'content': article['content'][:300]  # Use shorter excerpts
                        })

                    content_prompt = (
                        f"Search Query: {query}\n\n"
                        "IMPORTANT: Your response MUST include citations from these articles. "
                        "Every significant piece of information MUST be backed by at least one citation. "
                        "Here are brief excerpts from the articles to analyze and cite:\n\n"
                    )

                    for i, article in enumerate(shortened_articles, 1):
                        content_prompt += (
                            f"Article {i}:\n"
                            f"Title: {article['title']}\n"
                            f"Source: {article['source']}\n"
                            f"URL: {article['url']}\n"
                            f"Excerpt: {article['content']}\n\n"
                        )

                    content_prompt += (
                        "REMINDER: Format your response as a natural conversation, but ensure EVERY "
                        "significant point has a citation in [Source Name](URL) format. DO NOT skip citations.\n\n"
                    )

                    response = await async_chat_completion(
                        model=os.getenv("LOCAL_CHAT"),
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": content_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    return response.choices[0].message.content.strip()
                else:
                    raise

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
        start_time = time.time()
        logger.info(f"🔍 Search requested by {interaction.user}: '{query}'")
        
        if await self.is_rate_limited(interaction.user.id):
            await interaction.response.send_message(
                "⚠️ You are searching too quickly. Please wait a moment.",
                ephemeral=True
            )
            return
        
        await interaction.response.defer()
        
        try:
            # Get initial search results
            initial_results = list(await asyncio.to_thread(
                self.ddgs.text,
                query,
                max_results=10  # Increased from 5 to 10 for better selection
            ))
            
            if not initial_results:
                await interaction.followup.send("❌ No results found.", ephemeral=True)
                return
            
            # Select and process articles
            selected_results = await self.select_articles(initial_results)
            final_response = await self.generate_final_response(query, selected_results)
            
            # Ensure sources are included by appending them
            sources_section = "\n\n**Sources Used:**\n"
            for i, article in enumerate(selected_results, 1):
                title = article.get('title', 'Untitled').strip()
                url = article.get('href', '').strip()
                if url:  # Only include if we have a URL
                    sources_section += f"{i}. [{title}]({url})\n"
            
            # Combine response with sources
            final_response = final_response.strip() + sources_section
            
            # Split response if needed
            MAX_EMBED_LENGTH = 3900
            response_chunks = [final_response[i:i + MAX_EMBED_LENGTH] 
                             for i in range(0, len(final_response), MAX_EMBED_LENGTH)]
            
            elapsed_time = round(time.time() - start_time, 2)
            
            for i, chunk in enumerate(response_chunks):
                embed = discord.Embed(
                    title=f"🔍 Search Results for: {query}" + 
                          (f" (Part {i+1}/{len(response_chunks)})" if len(response_chunks) > 1 else ""),
                    description=chunk,
                    color=discord.Color.green()
                )
                
                if i == 0:
                    embed.set_footer(text=f"Search completed in {elapsed_time} seconds")
                    await interaction.followup.send(embed=embed)
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
