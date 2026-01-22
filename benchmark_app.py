"""
Web Data API Benchmark Tool
============================
Rigorous comparison of web search APIs for AI/agent use cases.
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, ".")

from clients import ZipfClient, ExaClient, FirecrawlClient

st.set_page_config(
    page_title="Web Data API Benchmark",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
# Keys can be pre-filled via query params: ?zipf=xxx&exa=xxx&firecrawl=xxx
if "history" not in st.session_state:
    st.session_state.history = []
if "zipf_key" not in st.session_state:
    st.session_state.zipf_key = st.query_params.get("zipf", "")
if "exa_key" not in st.session_state:
    st.session_state.exa_key = st.query_params.get("exa", "")
if "firecrawl_key" not in st.session_state:
    st.session_state.firecrawl_key = st.query_params.get("firecrawl", "")


def get_active_clients():
    clients = {}
    if st.session_state.zipf_key:
        clients["Zipf.ai"] = ZipfClient(st.session_state.zipf_key)
    if st.session_state.exa_key:
        clients["Exa.ai"] = ExaClient(st.session_state.exa_key)
    if st.session_state.firecrawl_key:
        clients["Firecrawl"] = FirecrawlClient(st.session_state.firecrawl_key)
    return clients


def run_parallel_search(clients, query, max_results=10):
    """Run search across all clients in parallel, return detailed results."""
    results = {}

    def search_provider(name, client):
        start = time.time()
        try:
            result = client.search(query, max_results=max_results)
            result["total_time_ms"] = (time.time() - start) * 1000
            return name, result
        except Exception as e:
            return name, {"error": str(e), "total_time_ms": (time.time() - start) * 1000}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(search_provider, name, client): name
                   for name, client in clients.items()}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result

    return results


def extract_domains(results):
    """Extract unique domains from results."""
    domains = set()
    for r in results:
        url = r.get("url", "")
        if url:
            parts = url.split("/")
            if len(parts) >= 3:
                domains.add(parts[2])
    return domains


def compute_overlap(results_dict):
    """Compute URL overlap between all provider pairs."""
    providers = list(results_dict.keys())
    url_sets = {}

    for name, data in results_dict.items():
        if not data.get("error"):
            url_sets[name] = set(r.get("url", "") for r in data.get("results", []))

    overlaps = {}
    for i, p1 in enumerate(providers):
        for p2 in providers[i+1:]:
            if p1 in url_sets and p2 in url_sets:
                common = url_sets[p1] & url_sets[p2]
                total = url_sets[p1] | url_sets[p2]
                pct = (len(common) / len(total) * 100) if total else 0
                overlaps[f"{p1} ∩ {p2}"] = {
                    "common": len(common),
                    "total_unique": len(total),
                    "overlap_pct": round(pct, 1)
                }
    return overlaps


# Sidebar - minimal
with st.sidebar:
    st.title("API Keys")

    zipf_key = st.text_input("Zipf.ai", value=st.session_state.zipf_key, type="password", key="z")
    exa_key = st.text_input("Exa.ai", value=st.session_state.exa_key, type="password", key="e")
    fc_key = st.text_input("Firecrawl", value=st.session_state.firecrawl_key, type="password", key="f")

    st.session_state.zipf_key = zipf_key
    st.session_state.exa_key = exa_key
    st.session_state.firecrawl_key = fc_key

    st.divider()
    clients = get_active_clients()

    status_text = []
    for name, client in clients.items():
        s = client.test_connection()
        if s.get("connected"):
            status_text.append(f"✓ {name}")
        else:
            status_text.append(f"✗ {name}")

    st.write(" | ".join(status_text))


# Main content
st.title("Web Data API Benchmark")
st.caption("Rigorous side-by-side comparison of Zipf.ai, Exa.ai, and Firecrawl")

clients = get_active_clients()

if len(clients) < 2:
    st.warning("Add at least 2 API keys to run comparisons")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Search Comparison", "Session Test (Key Differentiator)", "Crawl Comparison", "Export Data"])


# ============================================================================
# TAB 1: SEARCH COMPARISON
# ============================================================================
with tab1:
    st.header("Search Comparison")

    # Query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Query", value="", placeholder="Enter search query...")
    with col2:
        max_results = st.number_input("Results", min_value=5, max_value=20, value=10)

    # Preset queries for systematic testing
    st.markdown("**Preset test queries:**")
    presets = {
        "Simple lookup": "python requests library documentation",
        "Complex technical": "implementing retrieval augmented generation with vector databases best practices architecture",
        "Recent events": "latest developments in AI agents and tool use January 2025",
        "Niche B2B": "commercial real estate cap rate calculation methodology",
        "With operators": "site:github.com langchain retrieval -javascript",
    }

    preset_cols = st.columns(len(presets))
    for i, (name, q) in enumerate(presets.items()):
        if preset_cols[i].button(name, key=f"preset_{i}", use_container_width=True):
            st.session_state["query_input"] = q
            st.rerun()

    if "query_input" in st.session_state:
        query = st.session_state.pop("query_input")

    if st.button("Run Comparison", type="primary", disabled=not query):

        with st.spinner("Running searches in parallel..."):
            results = run_parallel_search(clients, query, max_results)

        # Store for export
        st.session_state.history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "search",
            "query": query,
            "max_results": max_results,
            "results": results
        })

        # ============================================================
        # SECTION 1: METRICS OVERVIEW
        # ============================================================
        st.subheader("1. Performance Metrics")

        metrics_data = []
        for name, data in results.items():
            metrics_data.append({
                "Provider": name,
                "Latency (ms)": round(data.get("latency_ms", data.get("total_time_ms", 0))),
                "Results Returned": data.get("result_count", 0),
                "Credits/Cost": data.get("credits_used", "N/A"),
                "Error": data.get("error", "None")
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ============================================================
        # SECTION 2: QUERY PROCESSING
        # ============================================================
        st.subheader("2. Query Processing")
        st.markdown("How each provider interpreted/processed the query:")

        processing_cols = st.columns(len(results))
        for i, (name, data) in enumerate(results.items()):
            with processing_cols[i]:
                st.markdown(f"**{name}**")

                if data.get("error"):
                    st.error(data["error"])
                    continue

                # Zipf-specific: query interpretation
                if name == "Zipf.ai":
                    interp = data.get("query_interpretation", {})
                    if interp:
                        st.markdown("*Query Interpretation:*")
                        if data.get("rewritten_query") and data.get("rewritten_query") != query:
                            st.code(f"Rewritten: {data['rewritten_query']}")
                        if data.get("detected_intent"):
                            st.write(f"Intent: {data['detected_intent']}")

                        # Show metadata if present
                        metadata = interp.get("metadata", {})
                        if metadata:
                            with st.expander("Full interpretation"):
                                st.json(interp)
                    else:
                        st.write("No query interpretation returned")

                # Exa-specific: search type
                elif name == "Exa.ai":
                    st.write(f"Search type: `{data.get('search_type_used', 'auto')}`")
                    st.write("(Embedding-based semantic matching)")

                # Firecrawl
                elif name == "Firecrawl":
                    st.write("Standard keyword search")
                    st.write("(Web search + optional scraping)")

        # ============================================================
        # SECTION 3: RESULTS COMPARISON
        # ============================================================
        st.subheader("3. Results Comparison")

        # Create a combined view
        max_to_show = min(max_results, 10)

        result_cols = st.columns(len(results))
        for col_idx, (name, data) in enumerate(results.items()):
            with result_cols[col_idx]:
                st.markdown(f"**{name}**")

                if data.get("error"):
                    st.error(data["error"])
                    continue

                for j, r in enumerate(data.get("results", [])[:max_to_show], 1):
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "")

                    # Extract domain
                    domain = ""
                    if url:
                        parts = url.split("/")
                        if len(parts) >= 3:
                            domain = parts[2]

                    st.markdown(f"**{j}. {title[:60]}{'...' if len(title) > 60 else ''}**")
                    st.caption(f"🔗 {domain}")
                    if snippet:
                        st.text(snippet[:150] + "..." if len(snippet) > 150 else snippet)
                    st.markdown("---")

        # ============================================================
        # SECTION 4: URL OVERLAP ANALYSIS
        # ============================================================
        st.subheader("4. URL Overlap Analysis")
        st.markdown("Do providers return the same URLs? Low overlap = different indices/algorithms.")

        overlaps = compute_overlap(results)

        if overlaps:
            overlap_data = []
            for pair, data in overlaps.items():
                overlap_data.append({
                    "Comparison": pair,
                    "Common URLs": data["common"],
                    "Total Unique": data["total_unique"],
                    "Overlap %": f"{data['overlap_pct']}%"
                })

            overlap_df = pd.DataFrame(overlap_data)
            st.dataframe(overlap_df, use_container_width=True, hide_index=True)

            # Show actual common URLs
            for name, data in results.items():
                if not data.get("error"):
                    urls = [r.get("url", "") for r in data.get("results", [])]
                    with st.expander(f"URLs from {name} ({len(urls)})"):
                        for url in urls:
                            st.code(url, language=None)

        # ============================================================
        # SECTION 5: DOMAIN DIVERSITY
        # ============================================================
        st.subheader("5. Source Diversity")
        st.markdown("Unique domains in results (more = broader coverage):")

        diversity_data = []
        for name, data in results.items():
            if not data.get("error"):
                domains = extract_domains(data.get("results", []))
                diversity_data.append({
                    "Provider": name,
                    "Unique Domains": len(domains),
                    "Top Domains": ", ".join(list(domains)[:5])
                })

        if diversity_data:
            diversity_df = pd.DataFrame(diversity_data)
            st.dataframe(diversity_df, use_container_width=True, hide_index=True)

        # ============================================================
        # SECTION 6: RAW RESPONSES
        # ============================================================
        st.subheader("6. Raw API Responses")

        for name, data in results.items():
            with st.expander(f"{name} - Full Response"):
                # Remove the raw nested response to avoid duplication
                display_data = {k: v for k, v in data.items() if k != "raw"}
                st.json(display_data)


# ============================================================================
# TAB 2: SESSION TEST - THE KEY DIFFERENTIATOR
# ============================================================================
with tab2:
    st.header("Session Test: URL Deduplication")

    st.markdown("""
    ### Why This Matters

    **Zipf's core claim:** When an AI agent does iterative research (multiple related searches),
    Zipf's sessions remember what URLs have been seen and won't return them again.

    **Exa and Firecrawl:** Every search is independent ("episodic"). If the same URL is relevant
    to multiple queries, it will be returned multiple times, wasting agent context.

    ### Test Protocol

    1. Run 3 related searches on the same topic
    2. Track all URLs returned
    3. Count duplicates (same URL appearing in multiple searches)

    **Expected result if sessions work:** Zero duplicate URLs across searches.
    """)

    st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Research topic", value="large language model optimization techniques")
    with col2:
        results_per_query = st.number_input("Results per query", min_value=5, max_value=15, value=10)

    # Generate related queries
    queries = [
        f"{topic}",
        f"{topic} best practices",
        f"{topic} recent research 2024 2025",
    ]

    st.markdown("**Queries that will be run:**")
    for i, q in enumerate(queries, 1):
        st.code(f"{i}. {q}")

    if st.button("Run Session Test", type="primary"):

        results_container = st.container()

        # Initialize all result variables to avoid undefined errors
        zipf_session_all_urls = []
        zipf_session_unique = set()
        zipf_session_duplicates = 0
        zipf_session_success = False

        zipf_nosession_all_urls = []
        zipf_nosession_unique = set()
        zipf_nosession_duplicates = 0
        zipf_nosession_success = False

        exa_all_urls = []
        exa_unique = set()
        exa_duplicates = 0
        exa_success = False

        fc_all_urls = []
        fc_unique = set()
        fc_duplicates = 0
        fc_success = False

        try:
          with results_container:
            progress_status = st.empty()

            # ============================================================
            # TEST ZIPF WITH SESSIONS
            # ============================================================
            progress_status.info("Step 1/4: Testing Zipf.ai WITH session...")
            st.subheader("1. Zipf.ai WITH Session (should deduplicate)")

            if "Zipf.ai" not in clients:
                st.warning("Zipf API key required")
            else:
                zipf = clients["Zipf.ai"]

                try:
                    # Create session
                    session_result = zipf._request("POST", "/sessions", {
                        "name": f"benchmark_session_{datetime.now().strftime('%H%M%S')}",
                        "description": "Deduplication test"
                    })

                    session_data = session_result.get("session", session_result)
                    session_id = session_data.get("id")

                    if not session_id:
                        st.error(f"Failed to create session: {session_result}")
                    else:
                        zipf_session_results = []

                        for i, q in enumerate(queries):
                            progress_status.info(f"Step 1/4: Zipf session search {i+1}/{len(queries)}...")
                            result = zipf.search(q, max_results=results_per_query, session_id=session_id)
                            urls = [r.get("url", "") for r in result.get("results", [])]
                            zipf_session_results.append({
                                "query": q,
                                "urls": urls,
                                "count": len(urls)
                            })
                            zipf_session_all_urls.extend(urls)

                        # Cleanup session
                        zipf._request("DELETE", f"/sessions/{session_id}")

                        # Analyze
                        zipf_session_unique = set(zipf_session_all_urls)
                        zipf_session_duplicates = len(zipf_session_all_urls) - len(zipf_session_unique)
                        zipf_session_success = True

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total URLs returned", len(zipf_session_all_urls))
                        col2.metric("Unique URLs", len(zipf_session_unique))
                        col3.metric("Duplicates", zipf_session_duplicates)

                        if zipf_session_duplicates == 0:
                            st.success("Session deduplication WORKING - no duplicate URLs")
                        else:
                            st.warning(f"{zipf_session_duplicates} duplicates found")

                        with st.expander("Details by query"):
                            for r in zipf_session_results:
                                st.write(f"**{r['query'][:50]}...** → {r['count']} URLs")
                except Exception as e:
                    st.error(f"Error during Zipf session test: {str(e)}")

            # ============================================================
            # TEST ZIPF WITHOUT SESSIONS (control)
            # ============================================================
            progress_status.info("Step 2/4: Testing Zipf.ai WITHOUT session...")
            st.subheader("2. Zipf.ai WITHOUT Session (control group)")

            if "Zipf.ai" in clients:
                try:
                    zipf_nosession_results = []

                    for i, q in enumerate(queries):
                        progress_status.info(f"Step 2/4: Zipf no-session search {i+1}/{len(queries)}...")
                        result = zipf.search(q, max_results=results_per_query)  # No session_id
                        urls = [r.get("url", "") for r in result.get("results", [])]
                        zipf_nosession_results.append({
                            "query": q,
                            "urls": urls,
                            "count": len(urls)
                        })
                        zipf_nosession_all_urls.extend(urls)

                    zipf_nosession_unique = set(zipf_nosession_all_urls)
                    zipf_nosession_duplicates = len(zipf_nosession_all_urls) - len(zipf_nosession_unique)
                    zipf_nosession_success = True

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total URLs returned", len(zipf_nosession_all_urls))
                    col2.metric("Unique URLs", len(zipf_nosession_unique))
                    col3.metric("Duplicates", zipf_nosession_duplicates)

                    st.info(f"Without sessions: {zipf_nosession_duplicates} duplicate URLs (expected)")
                except Exception as e:
                    st.error(f"Error during Zipf no-session test: {str(e)}")

            # ============================================================
            # TEST EXA (no sessions - should have duplicates)
            # ============================================================
            progress_status.info("Step 3/4: Testing Exa.ai...")
            st.subheader("3. Exa.ai (no session feature)")

            if "Exa.ai" not in clients:
                st.warning("Exa API key required")
            else:
                exa = clients["Exa.ai"]

                try:
                    exa_results = []

                    for i, q in enumerate(queries):
                        progress_status.info(f"Step 3/4: Exa search {i+1}/{len(queries)}...")
                        result = exa.search(q, max_results=results_per_query)
                        urls = [r.get("url", "") for r in result.get("results", [])]
                        exa_results.append({
                            "query": q,
                            "urls": urls,
                            "count": len(urls)
                        })
                        exa_all_urls.extend(urls)

                    exa_unique = set(exa_all_urls)
                    exa_duplicates = len(exa_all_urls) - len(exa_unique)
                    exa_success = True

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total URLs returned", len(exa_all_urls))
                    col2.metric("Unique URLs", len(exa_unique))
                    col3.metric("Duplicates", exa_duplicates)

                    if exa_duplicates > 0:
                        st.info(f"Exa: {exa_duplicates} duplicate URLs (no session feature)")
                    else:
                        st.write("No duplicates")
                except Exception as e:
                    st.error(f"Error during Exa test: {str(e)}")

            # ============================================================
            # TEST FIRECRAWL (no sessions - should have duplicates)
            # ============================================================
            progress_status.info("Step 4/4: Testing Firecrawl...")
            st.subheader("4. Firecrawl (no session feature)")

            if "Firecrawl" not in clients:
                st.warning("Firecrawl API key required")
            else:
                fc = clients["Firecrawl"]

                try:
                    fc_results = []

                    for i, q in enumerate(queries):
                        progress_status.info(f"Step 4/4: Firecrawl search {i+1}/{len(queries)}...")
                        result = fc.search(q, max_results=results_per_query)
                        urls = [r.get("url", "") for r in result.get("results", [])]
                        fc_results.append({
                            "query": q,
                            "urls": urls,
                            "count": len(urls)
                        })
                        fc_all_urls.extend(urls)

                    fc_unique = set(fc_all_urls)
                    fc_duplicates = len(fc_all_urls) - len(fc_unique)
                    fc_success = True

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total URLs returned", len(fc_all_urls))
                    col2.metric("Unique URLs", len(fc_unique))
                    col3.metric("Duplicates", fc_duplicates)

                    if fc_duplicates > 0:
                        st.info(f"Firecrawl: {fc_duplicates} duplicate URLs (no session feature)")
                    else:
                        st.write("No duplicates")
                except Exception as e:
                    st.error(f"Error during Firecrawl test: {str(e)}")

            # ============================================================
            # SUMMARY
            # ============================================================
            progress_status.success("Complete!")
            st.subheader("5. Summary")

            summary_data = []

            if zipf_session_success:
                summary_data.append({
                    "Provider": "Zipf.ai (with session)",
                    "Total URLs": len(zipf_session_all_urls),
                    "Unique URLs": len(zipf_session_unique),
                    "Duplicates": zipf_session_duplicates,
                    "Deduplication": "YES" if zipf_session_duplicates == 0 else "PARTIAL"
                })

            if zipf_nosession_success:
                summary_data.append({
                    "Provider": "Zipf.ai (no session)",
                    "Total URLs": len(zipf_nosession_all_urls),
                    "Unique URLs": len(zipf_nosession_unique),
                    "Duplicates": zipf_nosession_duplicates,
                    "Deduplication": "NO"
                })

            if exa_success:
                summary_data.append({
                    "Provider": "Exa.ai",
                    "Total URLs": len(exa_all_urls),
                    "Unique URLs": len(exa_unique),
                    "Duplicates": exa_duplicates,
                    "Deduplication": "NO (not supported)"
                })

            if fc_success:
                summary_data.append({
                    "Provider": "Firecrawl",
                    "Total URLs": len(fc_all_urls),
                    "Unique URLs": len(fc_unique),
                    "Duplicates": fc_duplicates,
                    "Deduplication": "NO (not supported)"
                })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                st.markdown("""
                ### Interpretation

                - **If Zipf with session has 0 duplicates** and others have duplicates → Session feature is working and valuable
                - **If all have similar duplicates** → Session feature may not be working as claimed
                - **If none have duplicates** → Queries were different enough that there's no natural overlap

                For AI agents doing iterative research, deduplication means:
                - Less wasted context window
                - No re-processing same information
                - More efficient token usage
                """)

        except Exception as e:
            import traceback
            st.error(f"Session test crashed: {str(e)}")
            st.code(traceback.format_exc(), language="python")


# ============================================================================
# TAB 3: CRAWL COMPARISON
# ============================================================================
with tab3:
    st.header("Crawl/Extract Comparison")

    url = st.text_input("URL to crawl", value="", placeholder="https://example.com")

    preset_urls = {
        "Wikipedia": "https://en.wikipedia.org/wiki/Large_language_model",
        "GitHub README": "https://github.com/langchain-ai/langchain",
        "News article": "https://techcrunch.com",
        "Documentation": "https://docs.python.org/3/library/asyncio.html",
    }

    url_cols = st.columns(len(preset_urls))
    for i, (name, u) in enumerate(preset_urls.items()):
        if url_cols[i].button(name, key=f"url_{i}", use_container_width=True):
            st.session_state["url_input"] = u
            st.rerun()

    if "url_input" in st.session_state:
        url = st.session_state.pop("url_input")

    if st.button("Run Crawl Comparison", type="primary", disabled=not url):

        crawl_results = {}

        progress = st.progress(0)
        status = st.empty()

        for i, (name, client) in enumerate(clients.items()):
            status.write(f"Crawling via {name}...")
            progress.progress((i + 1) / len(clients))

            try:
                crawl_results[name] = client.crawl(url)
            except Exception as e:
                crawl_results[name] = {"error": str(e)}

        progress.empty()
        status.empty()

        # Store for export
        st.session_state.history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "crawl",
            "url": url,
            "results": crawl_results
        })

        # Metrics
        st.subheader("1. Extraction Metrics")

        crawl_metrics = []
        for name, data in crawl_results.items():
            crawl_metrics.append({
                "Provider": name,
                "Latency (ms)": round(data.get("latency_ms", 0)),
                "Content Length": data.get("content_length", 0),
                "Status": data.get("status", data.get("error", "Unknown")),
                "Credits/Cost": data.get("credits_used", "N/A"),
            })

        crawl_df = pd.DataFrame(crawl_metrics)
        st.dataframe(crawl_df, use_container_width=True, hide_index=True)

        # Content comparison
        st.subheader("2. Extracted Content")

        content_cols = st.columns(len(crawl_results))
        for i, (name, data) in enumerate(crawl_results.items()):
            with content_cols[i]:
                st.markdown(f"**{name}**")

                if data.get("error"):
                    st.error(data["error"])
                    continue

                st.metric("Characters", f"{data.get('content_length', 0):,}")

                if data.get("title"):
                    st.write(f"**Title:** {data['title']}")

                content = data.get("content_preview", "")
                st.text_area(
                    "Content preview",
                    content,
                    height=300,
                    key=f"content_{name}",
                    disabled=True
                )

        # Raw responses
        st.subheader("3. Raw Responses")
        for name, data in crawl_results.items():
            with st.expander(f"{name} - Full Response"):
                display_data = {k: v for k, v in data.items() if k != "raw"}
                st.json(display_data)


# ============================================================================
# TAB 4: EXPORT
# ============================================================================
with tab4:
    st.header("Export Data")

    if not st.session_state.history:
        st.info("Run some comparisons first. Results will appear here for export.")
    else:
        st.write(f"**{len(st.session_state.history)} comparisons recorded**")

        # Summary table
        summary_data = []
        for entry in st.session_state.history:
            if entry["type"] == "search":
                summary_data.append({
                    "Time": entry["timestamp"][:19],
                    "Type": "Search",
                    "Input": entry["query"][:50] + "...",
                    "Providers": ", ".join(entry["results"].keys())
                })
            else:
                summary_data.append({
                    "Time": entry["timestamp"][:19],
                    "Type": "Crawl",
                    "Input": entry["url"][:50] + "...",
                    "Providers": ", ".join(entry["results"].keys())
                })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Export buttons
        col1, col2 = st.columns(2)

        with col1:
            json_str = json.dumps(st.session_state.history, indent=2, default=str)
            st.download_button(
                "Download JSON",
                json_str,
                file_name=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col2:
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

        # Detailed view
        st.subheader("Detailed Results")

        for i, entry in enumerate(reversed(st.session_state.history)):
            ts = entry["timestamp"][:19].replace("T", " ")

            if entry["type"] == "search":
                label = f"Search: \"{entry['query'][:40]}...\" ({ts})"
            else:
                label = f"Crawl: {entry['url'][:40]}... ({ts})"

            with st.expander(label):
                st.json(entry)


# Footer
st.divider()
st.caption("Web Data API Benchmark Tool | Compare search and crawl APIs side-by-side")
