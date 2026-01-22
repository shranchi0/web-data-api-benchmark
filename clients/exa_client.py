"""Exa.ai API client wrapper."""

import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ExaClient:
    """Client for Exa.ai API."""

    api_key: str
    base_url: str = "https://api.exa.ai"
    name: str = field(default="Exa.ai", init=False)

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
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
        """Test API connection with a minimal search."""
        result = self._request("POST", "/search", {"query": "test", "numResults": 1})
        if "error" not in result or "results" in result:
            return {
                "connected": True,
                "latency_ms": result.get("latency_ms", 0)
            }
        return {"connected": False, "error": result.get("error", "Unknown error")}

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "auto",  # neural, keyword, or auto
        include_highlights: bool = True,
        include_summary: bool = False
    ) -> Dict[str, Any]:
        """Execute a search query with Exa's best features."""
        payload = {
            "query": query,
            "numResults": max_results,
            "type": search_type,
            "contents": {
                "highlights": include_highlights,
                "summary": include_summary
            }
        }

        result = self._request("POST", "/search", payload)

        if "error" in result and "results" not in result:
            return {
                "provider": self.name,
                "query": query,
                "latency_ms": result.get("latency_ms", 0),
                "error": result.get("error", result.get("message", "Unknown error")),
                "results": [],
                "result_count": 0,
                "credits_used": "N/A"
            }

        # Normalize response format
        results = result.get("results", [])
        normalized_results = []

        for r in results[:10]:
            # Get snippet from highlights or summary
            snippet = ""
            highlights = r.get("highlights", [])
            if highlights and isinstance(highlights, list) and len(highlights) > 0:
                snippet = highlights[0][:200]
            elif r.get("summary"):
                snippet = r.get("summary", "")[:200]

            normalized_results.append({
                "title": r.get("title", "No title"),
                "url": r.get("url", ""),
                "snippet": snippet,
                "score": r.get("score"),
                "published_date": r.get("publishedDate")
            })

        return {
            "provider": self.name,
            "query": query,
            "latency_ms": result.get("latency_ms", 0),
            "result_count": len(results),
            "credits_used": result.get("costDollars", "N/A"),
            "results": normalized_results,
            "search_type_used": search_type,
            "raw": result,
            "error": None
        }

    def crawl(self, url: str, max_pages: int = 1) -> Dict[str, Any]:
        """Get contents from a URL using Exa's contents endpoint."""
        payload = {
            "urls": [url],
            "text": True,
            "highlights": True,
            "summary": True
        }
        result = self._request("POST", "/contents", payload)

        if "error" in result and "results" not in result:
            return {
                "provider": self.name,
                "url": url,
                "latency_ms": result.get("latency_ms", 0),
                "error": result.get("error", result.get("message", "Unknown error"))
            }

        contents = result.get("results", [])
        if not contents:
            return {
                "provider": self.name,
                "url": url,
                "latency_ms": result.get("latency_ms", 0),
                "status": "completed",
                "content_length": 0,
                "content_preview": "",
                "error": "No content returned"
            }

        content = contents[0].get("text", "")
        title = contents[0].get("title", "")
        summary = contents[0].get("summary", "")

        return {
            "provider": self.name,
            "url": url,
            "latency_ms": result.get("latency_ms", 0),
            "status": "completed",
            "title": title,
            "content_length": len(content),
            "content_preview": content[:500] if content else "",
            "summary": summary,
            "credits_used": result.get("costDollars", "N/A"),
            "raw": result,
            "error": None
        }
