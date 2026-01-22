"""Search benchmark logic."""

from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_search_benchmark(
    query: str,
    clients: Dict[str, Any],
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Run search benchmark across all available clients.

    Args:
        query: Search query string
        clients: Dict of {name: client_instance}
        max_results: Maximum results to request

    Returns:
        Dict with results from each provider and summary metrics
    """
    results = {}

    def run_search(name: str, client: Any) -> tuple:
        try:
            result = client.search(query, max_results=max_results)
            return name, result
        except Exception as e:
            return name, {
                "provider": name,
                "query": query,
                "error": str(e),
                "latency_ms": 0,
                "result_count": 0,
                "results": []
            }

    # Run searches in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_search, name, client): name
            for name, client in clients.items()
        }

        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result

    # Calculate summary metrics
    summary = {
        "query": query,
        "providers_tested": len(results),
        "fastest": None,
        "most_results": None,
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

        # Find most results
        most = max(valid_results, key=lambda x: x[1].get("result_count", 0))
        summary["most_results"] = {
            "provider": most[0],
            "count": most[1].get("result_count", 0)
        }

    # Build comparison table data
    for name, result in results.items():
        summary["comparison"].append({
            "provider": name,
            "latency_ms": round(result.get("latency_ms", 0), 1),
            "result_count": result.get("result_count", 0),
            "credits_used": result.get("credits_used", "N/A"),
            "error": result.get("error")
        })

    return {
        "results": results,
        "summary": summary
    }


def compare_result_overlap(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare URL overlap between providers.

    Returns dict with overlap statistics.
    """
    url_sets = {}
    for name, result in results.items():
        if not result.get("error"):
            urls = {r.get("url", "") for r in result.get("results", [])}
            url_sets[name] = urls

    if len(url_sets) < 2:
        return {"overlap_analysis": "Need at least 2 providers to compare"}

    providers = list(url_sets.keys())
    overlaps = {}

    for i, p1 in enumerate(providers):
        for p2 in providers[i + 1:]:
            common = url_sets[p1] & url_sets[p2]
            total = url_sets[p1] | url_sets[p2]
            overlap_pct = (len(common) / len(total) * 100) if total else 0
            overlaps[f"{p1} vs {p2}"] = {
                "common_urls": len(common),
                "overlap_percentage": round(overlap_pct, 1)
            }

    return {"overlap_analysis": overlaps}
