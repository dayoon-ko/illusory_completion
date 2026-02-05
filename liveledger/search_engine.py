"""
Search engine and web browser implementations.

Provides:
- SerperSearchEngine: Google search via Serper API
- JinaBrowser: Web page content extraction via Jina Reader API
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import logging

import requests

logger = logging.getLogger(__name__)

# Constants
DEFAULT_JINA_API_KEY_ENV = "JINA_API_KEY"
DEFAULT_SERPER_API_KEY_ENV = "SERPER_API_KEY"
DEFAULT_TOP_K = 10
DEFAULT_TIMEOUT = 300
MAX_RETRY_ATTEMPTS = 10
INITIAL_BACKOFF = 1  # seconds


# ============================================================================
# SERPER SEARCH ENGINE
# ============================================================================

class SerperSearchEngine:
    """
    Google search interface via Serper API.
    
    Features:
    - Batch search support (multiple queries at once)
    - Automatic retry with exponential backoff
    - Returns both URLs and formatted snippets
    """
    
    def __init__(
        self,
        *,
        serper_api_key: str,
        topk: int = DEFAULT_TOP_K,
        request_timeout: int = DEFAULT_TIMEOUT,
        max_workers: int = 1
    ):
        """
        Initialize search engine.
        
        Args:
            serper_api_key: Serper API key
            topk: Number of results to return per query
            request_timeout: HTTP request timeout in seconds
            max_workers: Number of parallel workers for batch search
        """
        self.serper_api_key = serper_api_key
        self.topk = topk
        self.request_timeout = request_timeout
        self.max_workers = max_workers
    
    def search(self, query: str) -> Tuple[List[str], str]:
        """
        Search for a single query.
        
        Args:
            query: Search query string
            
        Returns:
            Tuple of (urls, formatted_results)
            - urls: List of result URLs
            - formatted_results: Formatted string with titles, URLs, and snippets
        """
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        
        # Retry with exponential backoff
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    json={"q": query, "num": self.topk},
                    timeout=self.request_timeout,
                    verify=False
                )
                response.raise_for_status()
                data = response.json()
                break
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Search attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} failed: {e}")
                
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.error(f"Search failed after {MAX_RETRY_ATTEMPTS} attempts for query: {query}")
                    return [], "No search results available."
                
                # Exponential backoff with cap at 30 seconds
                sleep_time = min(INITIAL_BACKOFF * (2 ** attempt), 30)
                time.sleep(sleep_time)
        else:
            return [], "No search results available."
        
        # Extract organic results
        organic = data.get("organic", [])[:self.topk]
        urls = [r.get("link", "") for r in organic]
        
        # Format results
        doc_str = "\n".join(
            f"[{i+1}] {r.get('title', '')} ({r.get('link', '')})\n"
            f"    {r.get('snippet', '')}"
            for i, r in enumerate(organic)
        )
        
        return urls, doc_str.strip()
    
    def search_batch(self, queries: List[str]) -> str:
        """
        Search for multiple queries in parallel.
        
        Args:
            queries: List of search query strings
            
        Returns:
            Formatted string with all results, grouped by query
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            futures = {
                executor.submit(self.search, q.strip()): q
                for q in queries
                if q.strip()
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                query = futures[future]
                try:
                    _, result = future.result()
                    if not result or len(result) == 0:
                        result = "No helpful search results found."
                    results[query] = result
                    
                except Exception as e:
                    logger.exception(f"Search failed for query '{query}'")
                    results[query] = f"Search failed: {str(e)}"
        
        # Format all results
        return "\n\n".join(
            f"**Query: {q}**\n{r}"
            for q, r in results.items()
        )


# ============================================================================
# JINA BROWSER
# ============================================================================

class JinaBrowser:
    """
    Web page content extractor via Jina Reader API.
    
    Features:
    - Extract clean, readable text from web pages
    - Batch browsing support (multiple URLs at once)
    - Automatic content truncation to manageable size
    """
    
    def __init__(
        self,
        *,
        jina_api_key: str,
        request_timeout: int = DEFAULT_TIMEOUT,
        max_content_length: int = 8000
    ):
        """
        Initialize browser.
        
        Args:
            jina_api_key: Jina API key
            request_timeout: HTTP request timeout in seconds
            max_content_length: Maximum content length to return
        """
        self.jina_api_key = jina_api_key
        self.request_timeout = request_timeout
        self.max_content_length = max_content_length
    
    def browse(self, url: str) -> str:
        """
        Read content from a single URL.
        
        Args:
            url: URL to read
            
        Returns:
            Formatted string with title and content
        """
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.jina_api_key}"
                },
                timeout=self.request_timeout,
                verify=False
            )
            response.raise_for_status()
            
            data = response.json()
            content = data.get("data", {}).get("content", "")
            title = data.get("data", {}).get("title", "")
            
            # Truncate content if too long
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
                logger.info(f"Truncated content from {url} to {self.max_content_length} chars")
            
            return f"**{title}**\n{content}"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to read {url}: {e}")
            return f"Failed to read {url}: {str(e)}"
        
        except Exception as e:
            logger.exception(f"Unexpected error reading {url}")
            return f"Failed to read {url}: {str(e)}"
    
    def browse_batch(self, urls: List[str]) -> str:
        """
        Read content from multiple URLs in parallel.
        
        Args:
            urls: List of URLs to read
            
        Returns:
            Formatted string with all content, separated by URL
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all URLs
            futures = {
                executor.submit(self.browse, u.strip()): u
                for u in urls
                if u.strip()
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                url = futures[future]
                try:
                    results[url] = future.result()
                    
                except Exception as e:
                    logger.exception(f"Browse failed for URL '{url}'")
                    results[url] = f"Failed to read {url}: {str(e)}"
        
        # Format all results
        return "\n\n---\n\n".join(
            f"**URL: {u}**\n{r}"
            for u, r in results.items()
        )