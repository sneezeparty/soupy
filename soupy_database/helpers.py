"""
Helper functions for processing images and URLs during scanning.
"""

import base64
import logging
import os
import re
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image
from typing import Optional, List
from urllib.parse import urlparse
import html2text
import trafilatura

logger = logging.getLogger(__name__)


def extract_urls(text: str) -> List[str]:
    """
    Extracts URLs from text, supporting common URL formats including query parameters and fragments.
    Returns a list of URLs, limited by MAX_URLS_PER_MESSAGE.
    """
    # Improved regex pattern that captures URLs with query parameters, fragments, and various characters
    url_pattern = r'https?://(?:[-\w.~:/?#[\]@!$&\'()*+,;=%]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, text)
    max_urls = int(os.getenv('MAX_URLS_PER_MESSAGE', 3))
    if urls:
        logger.debug("Extracted %s URL(s) from text: %s", len(urls), urls[:max_urls])
    return urls[:max_urls]  # Limit number of URLs processed per message


async def extract_url_content(url: str, session: aiohttp.ClientSession) -> Optional[str]:
    """
    Extracts relevant content from a URL, with safety checks and timeout.
    Returns a concise summary of the content or None if extraction fails.
    """
    try:
        logger.debug(f"Attempting to extract content from URL: {url}")
        
        # Get timeout from env or default to 15 seconds
        timeout = aiohttp.ClientTimeout(total=float(os.getenv('URL_FETCH_TIMEOUT', 15000)) / 1000)
        
        # Add headers to mimic a real browser and avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        async with session.get(url, timeout=timeout, headers=headers) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch URL {url}: HTTP {response.status}")
                return None
                
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if not content_type.startswith('text/html'):
                logger.debug(f"Skipping non-HTML content type: {content_type} for {url}")
                return None
            
            # Get text content
            html = await response.text()
            logger.debug(f"Fetched HTML content from {url}, length: {len(html)} characters")
            
            # Try multiple extraction methods in order of preference
            content = None
            
            # Method 1: Try trafilatura with different configurations
            try:
                content = trafilatura.extract(html, include_links=False, include_images=False, 
                                            include_tables=False, no_fallback=False)
                if content and len(content.strip()) > 50:  # Ensure we got substantial content
                    logger.debug(f"Successfully extracted content using trafilatura: {len(content)} chars")
                else:
                    content = None
            except Exception as e:
                logger.debug(f"Trafilatura extraction failed for {url}: {e}")
                content = None
            
            # Method 2: Fallback to html2text if trafilatura fails
            if not content:
                try:
                    h = html2text.HTML2Text()
                    h.ignore_links = True
                    h.ignore_images = True
                    h.ignore_tables = True
                    h.body_width = 0  # Don't wrap lines
                    content = h.handle(html).strip()
                    if content and len(content.strip()) > 50:
                        logger.debug(f"Successfully extracted content using html2text: {len(content)} chars")
                    else:
                        content = None
                except Exception as e:
                    logger.debug(f"html2text extraction failed for {url}: {e}")
                    content = None
            
            # Method 3: Basic BeautifulSoup extraction as last resort
            if not content:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    # Get text
                    content = soup.get_text()
                    # Clean up whitespace
                    lines = (line.strip() for line in content.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    content = ' '.join(chunk for chunk in chunks if chunk)
                    if content and len(content.strip()) > 50:
                        logger.debug(f"Successfully extracted content using BeautifulSoup: {len(content)} chars")
                    else:
                        content = None
                except Exception as e:
                    logger.debug(f"BeautifulSoup extraction failed for {url}: {e}")
                    content = None
            
            if not content:
                logger.debug(f"No content extracted from {url}")
                return None
            
            # Clean and limit content length
            content = ' '.join(content.split())  # Normalize whitespace
            
            # Filter out common boilerplate content
            content_lower = content.lower()
            boilerplate_patterns = [
                'javascript must be enabled',
                'please enable javascript',
                'cookie policy',
                'privacy policy',
                'terms of service',
                'terms and conditions',
                'all rights reserved',
                'copyright ©',
                'about press copyright contact us creators',
                'advertise developers terms privacy policy',
                'how youtube works',
                'test new features',
                'nfl sunday ticket',
                'this website uses cookies',
                'by using this site, you consent',
                'cookie policy for more information',
            ]
            
            # Check if content is mostly boilerplate
            boilerplate_count = sum(1 for pattern in boilerplate_patterns if pattern in content_lower)
            if boilerplate_count >= 3:
                logger.debug(f"Skipping URL with too much boilerplate content: {url}")
                return None
            
            # Check if content is too short or seems like just navigation/footer
            if len(content.strip()) < 100:
                logger.debug(f"Skipping URL with insufficient content ({len(content)} chars): {url}")
                return None
            
            # Check if content is mostly just links/navigation (lots of common link words)
            link_words = ['click here', 'read more', 'learn more', 'view', 'see more', 'home', 'about', 'contact']
            link_word_count = sum(1 for word in link_words if word in content_lower)
            if link_word_count > 5 and len(content) < 300:
                logger.debug(f"Skipping URL that appears to be mostly navigation: {url}")
                return None
            
            # Get max content length from env or default to 800
            max_content_length = int(os.getenv('URL_MAX_CONTENT_LENGTH', 800))
            if len(content) > max_content_length:
                content = content[:max_content_length-3] + "..."
            
            # Add URL source information
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check if we should include domain info
            include_domain = os.getenv('URL_INCLUDE_DOMAIN', 'true').lower() == 'true'
            if include_domain:
                result = f"[URL content from {domain}: {content}]"
            else:
                result = f"[URL content: {content}]"
            
            logger.info(f"Successfully extracted content from {url}: {len(result)} chars")
            return result
            
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return None


async def describe_image(attachment_url: str, filename: Optional[str] = None) -> Optional[str]:
    """
    Use LLM vision model to describe an image.
    Returns description or None if processing fails.
    Skips GIFs - only processes static images.
    Ported from the working implementation in soupy_remastered_stablediffusion.py
    """
    if os.getenv("ENABLE_VISION", "false").lower() != "true":
        return None
    
    # Skip GIFs - only process static images
    if filename and filename.lower().endswith(".gif"):
        return None
    
    if not any(filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) if filename else True:
        return None
    
    try:
        # Download the image
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment_url) as resp:
                if resp.status != 200:
                    return None
                image_data = await resp.read()
        
        # Transcode to JPEG for maximum compatibility
        try:
            with Image.open(BytesIO(image_data)) as im:
                # Skip animated GIFs
                if getattr(im, 'is_animated', False):
                    return None
                if im.mode in ("P", "RGBA"):
                    im = im.convert("RGB")
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=92, optimize=True)
                image_bytes_for_api = buf.getvalue()
                image_subtype = "jpeg"
        except Exception:
            # Fallback: use original bytes and infer subtype from filename
            image_bytes_for_api = image_data
            image_subtype = "jpeg"
            if filename:
                name = filename.lower()
                if name.endswith(".png"):
                    image_subtype = "png"
                elif name.endswith(".webp"):
                    image_subtype = "webp"
        
        # Encode to base64
        encoded_image = base64.b64encode(image_bytes_for_api).decode("utf-8")
        
        # Build endpoint from OPENAI_BASE_URL
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        endpoint = f"{base}/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        model_name = os.getenv("VISION_MODEL") or os.getenv("LOCAL_CHAT")
        prompt_text = os.getenv("VISION_PROMPT", "What is in this image? Describe it concisely.")
        max_tokens = int(os.getenv("VISION_MAX_TOKENS", "300"))
        temperature = float(os.getenv("VISION_TEMPERATURE", "0.7"))
        
        # First try: raw base64 ONLY (no mime_type), image first then text
        payload_raw_simple = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encoded_image}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload_raw_simple) as r0:
                if r0.status == 200:
                    out = await r0.json()
                    description = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                    if description:
                        return description
            
            # Second try: OpenAI-style input_image with data URI
            payload_input_image = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": {"url": f"data:image/{image_subtype};base64,{encoded_image}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            async with session.post(endpoint, headers=headers, json=payload_input_image) as r1:
                if r1.status == 200:
                    out = await r1.json()
                    description = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                    if description:
                        return description
            
            # Third try: data URI (image_url) then raw with mime_type
            payload_data_uri = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/{image_subtype};base64,{encoded_image}"}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            async with session.post(endpoint, headers=headers, json=payload_data_uri) as r2:
                if r2.status == 200:
                    out = await r2.json()
                    description = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                    if description:
                        return description
        
        return None
        
    except Exception as e:
        logger.debug(f"Image description failed: {e}")
        return None


async def summarize_url(url: str) -> Optional[str]:
    """
    Use LLM to summarize content from a URL.
    Returns summary or None if processing fails.
    Skips URLs that point only to images/GIFs - only processes webpages.
    """
    try:
        from urllib.parse import urlparse
        import trafilatura
        import html2text
        
        # Check if URL points to an image/GIF file directly
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        image_extensions = ['.gif', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.svg']
        if any(path_lower.endswith(ext) for ext in image_extensions):
            logger.debug(f"Skipping URL that points to image/GIF: {url}")
            return None
        
        timeout = aiohttp.ClientTimeout(total=float(os.getenv('URL_FETCH_TIMEOUT', 15000)) / 1000)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Fetch URL content
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout, headers=headers) as response:
                if response.status != 200:
                    logger.debug(f"URL fetch failed: HTTP {response.status} for {url}")
                    return None
                
                content_type = response.headers.get('Content-Type', '').lower()
                # Skip if it's an image/GIF content type
                if any(ct in content_type for ct in ['image/', 'gif']):
                    logger.debug(f"Skipping URL with image content type: {content_type} for {url}")
                    return None
                
                if not content_type.startswith('text/html'):
                    logger.debug(f"Skipping non-HTML content type: {content_type} for {url}")
                    return None
                
                html = await response.text()
                logger.debug(f"Fetched HTML from {url}, length: {len(html)} chars")
        
        # Extract content
        content = None
        try:
            content = trafilatura.extract(html, include_links=False, include_images=False, 
                                        include_tables=False, no_fallback=False)
            if content and len(content.strip()) > 50:
                pass  # Good content
            else:
                content = None
        except Exception:
            pass
        
        if not content:
            try:
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                content = h.handle(html)
            except Exception:
                pass
        
        if not content or len(content.strip()) < 50:
            return None
        
        # Clean and limit content
        content = ' '.join(content.split())
        
        # Filter out common boilerplate content
        content_lower = content.lower()
        boilerplate_patterns = [
            'javascript must be enabled',
            'please enable javascript',
            'cookie policy',
            'privacy policy',
            'terms of service',
            'terms and conditions',
            'all rights reserved',
            'copyright ©',
            'about press copyright contact us creators',
            'advertise developers terms privacy policy',
            'how youtube works',
            'test new features',
            'nfl sunday ticket',
            'this website uses cookies',
            'by using this site, you consent',
            'cookie policy for more information',
        ]
        
        # Check if content is mostly boilerplate
        boilerplate_count = sum(1 for pattern in boilerplate_patterns if pattern in content_lower)
        if boilerplate_count >= 3:
            logger.debug(f"Skipping URL with too much boilerplate content: {url}")
            return None
        
        # Check if content is too short or seems like just navigation/footer
        if len(content.strip()) < 100:
            logger.debug(f"Skipping URL with insufficient content ({len(content)} chars): {url}")
            return None
        
        # Check if content is mostly just links/navigation
        link_words = ['click here', 'read more', 'learn more', 'view', 'see more', 'home', 'about', 'contact']
        link_word_count = sum(1 for word in link_words if word in content_lower)
        if link_word_count > 5 and len(content) < 300:
            logger.debug(f"Skipping URL that appears to be mostly navigation: {url}")
            return None
        
        max_length = int(os.getenv('URL_MAX_CONTENT_LENGTH', 2000))
        if len(content) > max_length:
            content = content[:max_length-3] + "..."
        
        # Use LLM to summarize
        base = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        endpoint = f"{base}/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        model_name = os.getenv("LOCAL_CHAT")
        if not model_name:
            return None
        
        summary_prompt = f"Summarize the following web content concisely in 2-3 sentences:\n\n{content}"
        
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": summary_prompt
            }],
            "max_tokens": 200,
            "temperature": 0.7,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200:
                    out = await r.json()
                    summary = (out.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
                    if summary:
                        return summary
        
        return None
        
    except Exception as e:
        logger.debug(f"URL summarization failed for {url}: {e}")
        return None

