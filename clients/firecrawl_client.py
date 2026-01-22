"""Firecrawl API client wrapper."""

import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class FirecrawlClient:
    """Client for Firecrawl API."""

    api_key: str
    base_url: str = "https://api.firecrawl.dev/v1"
    name: str = field(default="Firecrawl", init=False)

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request with timing."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            if method == "GET":
                resp = requests.get(url, headers=self.headers, timeout=30)
            elif method == "POST":
                resp = requests.post(url, headers=self.headers, json=data, timeout=60)
            else:
                return {"error": f"Unknown method: {method}", "latency_ms": 0}

            latency_ms = (time.time() - start_time) * 1000
            result = resp.json()
            result["latency_ms"] = latency_ms
            return result

        except requests.exceptions.RequestException as e:
            return {"error": str(e), "latency_ms": (time.time() - start_time) * 1000}

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection."""
        # Try to scrape a simple page
        result = self._request("POST", "/scrape", {"url": "https://example.com"})
        if result.get("success") or "data" in result:
            return {
                "connected": True,
                "latency_ms": result.get("latency_ms", 0)
            }
        return {"connected": False, "error": result.get("error", "Unknown error")}

    def search(
        self,
        query: str,
        max_results: int = 10,
        scrape_content: bool = False
    ) -> Dict[str, Any]:
        """Execute a search query."""
        payload = {
            "query": query,
            "limit": max_results,
        }

        if scrape_content:
            payload["scrapeOptions"] = {"formats": ["markdown"]}

        result = self._request("POST", "/search", payload)

        if not result.get("success") and "data" not in result:
            return {
                "provider": self.name,
                "query": query,
                "latency_ms": result.get("latency_ms", 0),
                "error": result.get("error", result.get("message", "Unknown error")),
                "results": [],
                "result_count": 0,
                "credits_used": "N/A"
            }

        # Firecrawl search returns data array directly or nested
        data = result.get("data", result)
        if isinstance(data, dict):
            results = data.get("web", data.get("results", []))
        else:
            results = data if isinstance(data, list) else []

        return {
            "provider": self.name,
            "query": query,
            "latency_ms": result.get("latency_ms", 0),
            "result_count": len(results),
            "credits_used": result.get("creditsUsed", "N/A"),
            "results": [
                {
                    "title": r.get("title", "No title"),
                    "url": r.get("url", ""),
                    "snippet": r.get("description", r.get("markdown", ""))[:200]
                }
                for r in results[:10]
            ],
            "raw": result,
            "error": None
        }

    def crawl(self, url: str, max_pages: int = 1) -> Dict[str, Any]:
        """Scrape a URL and extract content."""
        payload = {
            "url": url,
            "formats": ["markdown"],
        }
        result = self._request("POST", "/scrape", payload)

        if not result.get("success") and "data" not in result:
            return {
                "provider": self.name,
                "url": url,
                "latency_ms": result.get("latency_ms", 0),
                "error": result.get("error", result.get("message", "Unknown error"))
            }

        data = result.get("data", {})
        content = data.get("markdown", data.get("content", ""))

        return {
            "provider": self.name,
            "url": url,
            "latency_ms": result.get("latency_ms", 0),
            "status": "completed",
            "content_length": len(content),
            "content_preview": content[:500] if content else "",
            "credits_used": result.get("creditsUsed", 1),
            "raw": result,
            "error": None
        }
