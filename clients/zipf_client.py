"""Zipf.ai API client wrapper."""

import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ZipfClient:
    """Client for Zipf.ai API."""

    api_key: str
    base_url: str = "https://zipf.ai/api/v1"
    name: str = field(default="Zipf.ai", init=False)

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
            elif method == "DELETE":
                resp = requests.delete(url, headers=self.headers, timeout=30)
            else:
                return {"error": f"Unknown method: {method}", "latency_ms": 0}

            latency_ms = (time.time() - start_time) * 1000

            if resp.status_code == 204:
                return {"success": True, "latency_ms": latency_ms}

            result = resp.json()
            result["latency_ms"] = latency_ms
            return result

        except requests.exceptions.RequestException as e:
            return {"error": str(e), "latency_ms": (time.time() - start_time) * 1000}

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection and return account info."""
        result = self._request("GET", "")
        if "error" not in result and "user" in result:
            return {
                "connected": True,
                "user": result["user"]["name"],
                "credits": result["user"]["credits_balance"],
                "latency_ms": result.get("latency_ms", 0)
            }
        return {"connected": False, "error": result.get("error", "Unknown error")}

    def search(
        self,
        query: str,
        max_results: int = 10,
        interpret_query: bool = True,  # Enable by default for best results
        rerank_results: bool = True,   # Enable by default for best results
        generate_suggestions: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a search query with full feature set."""
        payload = {
            "query": query,
            "max_results": max_results,
            "interpret_query": interpret_query,
            "rerank_results": rerank_results,
            "generate_suggestions": generate_suggestions,
        }
        if session_id:
            payload["session_id"] = session_id

        result = self._request("POST", "/search", payload)

        if "error" in result and "results" not in result:
            # Combine error and message for clarity
            error_msg = result.get("message", result.get("error", "Unknown error"))
            return {
                "provider": self.name,
                "query": query,
                "latency_ms": result.get("latency_ms", 0),
                "error": error_msg,
                "results": [],
                "result_count": 0,
                "credits_used": "N/A"
            }

        # Extract query interpretation if available
        interpretation = result.get("query_interpretation", {})

        # Normalize response format
        return {
            "provider": self.name,
            "query": query,
            "latency_ms": result.get("latency_ms", 0),
            "result_count": len(result.get("results", [])),
            "credits_used": result.get("execution", {}).get("credits_consumed", "N/A"),
            "results": [
                {
                    "title": r.get("title", "No title"),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", r.get("description", ""))[:200] if r.get("snippet") or r.get("description") else "",
                    "score": r.get("score")
                }
                for r in result.get("results", [])[:10]
            ],
            # Zipf-specific features
            "query_interpretation": interpretation,
            "rewritten_query": interpretation.get("rewritten_query"),
            "detected_intent": interpretation.get("intent"),
            "suggestions": result.get("suggestions", []),
            "raw": result,
            "error": None
        }

    def search_with_session(
        self,
        queries: List[str],
        session_name: str = "benchmark_session",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Run multiple searches in a session to test URL deduplication."""
        # Create session
        session_result = self._request("POST", "/sessions", {
            "name": session_name,
            "description": "Benchmark session for URL deduplication test"
        })

        session_data = session_result.get("session", session_result)
        session_id = session_data.get("id")

        if not session_id:
            return {"error": f"Failed to create session: {session_result}"}

        all_results = []
        all_urls = []
        duplicate_count = 0

        for query in queries:
            result = self.search(query, max_results=max_results, session_id=session_id)
            urls = [r["url"] for r in result.get("results", [])]

            # Check for duplicates
            for url in urls:
                if url in all_urls:
                    duplicate_count += 1
                all_urls.append(url)

            all_results.append({
                "query": query,
                "result_count": result.get("result_count", 0),
                "urls": urls
            })

        # Cleanup session
        self._request("DELETE", f"/sessions/{session_id}")

        unique_urls = len(set(all_urls))

        return {
            "session_id": session_id,
            "queries_run": len(queries),
            "total_results": len(all_urls),
            "unique_urls": unique_urls,
            "duplicate_urls": duplicate_count,
            "deduplication_worked": duplicate_count == 0,
            "results_by_query": all_results
        }

    def crawl(self, url: str, max_pages: int = 1) -> Dict[str, Any]:
        """Crawl a URL and extract content."""
        payload = {
            "urls": [url],
            "max_pages": max_pages,
        }
        result = self._request("POST", "/crawls", payload)

        if "error" in result:
            return {
                "provider": self.name,
                "url": url,
                "latency_ms": result.get("latency_ms", 0),
                "error": result.get("error")
            }

        crawl_id = result.get("id")
        if not crawl_id:
            return {
                "provider": self.name,
                "url": url,
                "latency_ms": result.get("latency_ms", 0),
                "error": "No crawl ID returned"
            }

        # Poll for completion
        total_latency = result.get("latency_ms", 0)
        for _ in range(30):  # Max 60 seconds
            time.sleep(2)
            status = self._request("GET", f"/crawls/{crawl_id}")
            total_latency += status.get("latency_ms", 0)

            if status.get("status") in ["completed", "failed"]:
                content = ""
                title = ""
                if status.get("results"):
                    content = status["results"][0].get("content", "")
                    title = status["results"][0].get("title", "")

                return {
                    "provider": self.name,
                    "url": url,
                    "latency_ms": total_latency,
                    "status": status.get("status"),
                    "title": title,
                    "content_length": len(content),
                    "content_preview": content[:500] if content else "",
                    "credits_used": status.get("credits_consumed", "N/A"),
                    "raw": status,
                    "error": status.get("error")
                }

        return {
            "provider": self.name,
            "url": url,
            "latency_ms": total_latency,
            "error": "Crawl timed out"
        }
