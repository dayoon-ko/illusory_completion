from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import time

DEFAULT_JINA_API_KEY_ENV = "JINA_API_KEY"
DEFAULT_SERPER_API_KEY_ENV = "SERPER_API_KEY"
DEFAULT_TOP_K = 10

class SerperSearchEngine:
    def __init__(self, *, serper_api_key: str, topk: int = DEFAULT_TOP_K,
                 request_timeout: int = 300, max_workers: int = 1):
        self.serper_api_key = serper_api_key
        self.topk = topk
        self.request_timeout = request_timeout
        self.max_workers = max_workers

    def search(self, query: str) -> Tuple[List[str], str]:
        headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
        for i in range(10):
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
            except Exception:
                if i == 9:
                    return [], "No search results available."
                time.sleep(min(1 * (2**i), 30))
        else:
            return [], "No search results available."
        
        organic = data.get("organic", [])[:self.topk]
        urls = [r.get("link", "") for r in organic]
        doc_str = "\n".join(
            f"[{i+1}] {r.get('title', '')} ({r.get('link', '')})\n    {r.get('snippet', '')}"
            for i, r in enumerate(organic)
        )
        return urls, doc_str.strip()

    def search_batch(self, queries: List[str]) -> str:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.search, q.strip()): q for q in queries if q.strip()}
            for future in as_completed(futures):
                query = futures[future]
                try:
                    _, result = future.result()
                    if len(result) == 0:
                        result = "No helpful search results found."
                    results[query] = result
                except Exception:
                    results[query] = "Search failed."
        
        return "\n\n".join(f"**Query: {q}**\n{r}" for q, r in results.items())


class JinaBrowser:
    def __init__(self, *, jina_api_key: str, request_timeout: int = 300):
        self.jina_api_key = jina_api_key
        self.request_timeout = request_timeout

    def browse(self, url: str) -> str:
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "application/json", "Authorization": f"Bearer {self.jina_api_key}"},
                timeout=self.request_timeout,
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("data", {}).get("content", "")[:8000]
            title = data.get("data", {}).get("title", "")
            return f"**{title}**\n{content}"
        except Exception:
            return f"Failed to read {url}"

    def browse_batch(self, urls: List[str]) -> str:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.browse, u.strip()): u for u in urls if u.strip()}
            for future in as_completed(futures):
                url = futures[future]
                try:
                    results[url] = future.result()
                except Exception:
                    results[url] = f"Failed to read {url}"
        
        return "\n\n---\n\n".join(f"**URL: {u}**\n{r}" for u, r in results.items())
