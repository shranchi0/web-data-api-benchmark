"""Crawl/scrape benchmark logic."""

from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_crawl_benchmark(
    url: str,
    clients: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run crawl/scrape benchmark across all available clients.

    Args:
        url: URL to crawl/scrape
        clients: Dict of {name: client_instance}

    Returns:
        Dict with results from each provider and summary metrics
    """
    results = {}

    def run_crawl(name: str, client: Any) -> tuple:
        try:
            result = client.crawl(url)
            return name, result
        except Exception as e:
            return name, {
                "provider": name,
                "url": url,
                "error": str(e),
                "latency_ms": 0,
                "content_length": 0
            }

    # Run crawls in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_crawl, name, client): name
            for name, client in clients.items()
        }

        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result

    # Calculate summary metrics
    summary = {
        "url": url,
        "providers_tested": len(results),
        "fastest": None,
        "most_content": None,
        "comparison": []
    }

    valid_results = [(name, r) for name, r in results.items() if not r.get("error")]

    if valid_results:
        # Find fastest
        fastest = min(valid_results, key=lambda x: x[1].get("latency_ms", float("inf")))
        summary["fastest"] = {
            "provider": fastest[0],
            "latency_ms": fastest[1].get("latency_ms", 0)
        }

        # Find most content
        most = max(valid_results, key=lambda x: x[1].get("content_length", 0))
        summary["most_content"] = {
            "provider": most[0],
            "content_length": most[1].get("content_length", 0)
        }

    # Build comparison table data
    for name, result in results.items():
        summary["comparison"].append({
            "provider": name,
            "latency_ms": round(result.get("latency_ms", 0), 1),
            "content_length": result.get("content_length", 0),
            "credits_used": result.get("credits_used", "N/A"),
            "status": result.get("status", "N/A"),
            "error": result.get("error")
        })

    return {
        "results": results,
        "summary": summary
    }


def compare_content_quality(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare content extraction quality between providers.

    Returns dict with quality metrics.
    """
    quality_metrics = {}

    for name, result in results.items():
        if not result.get("error"):
            content = result.get("content_preview", "")
            quality_metrics[name] = {
                "content_length": result.get("content_length", 0),
                "has_content": len(content) > 0,
                "preview_length": len(content),
                # Simple heuristics for content quality
                "has_paragraphs": "\n\n" in content or ". " in content,
                "word_count": len(content.split()) if content else 0
            }

    return {"quality_analysis": quality_metrics}
