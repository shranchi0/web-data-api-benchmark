# Web Data API Benchmark

Rigorous side-by-side comparison of web search and crawl APIs for AI/agent use cases.

**Compares:** Zipf.ai, Exa.ai, Firecrawl

## Features

- **Search Comparison**: Same query across all providers with metrics, URL overlap analysis, source diversity
- **Session Test**: Validates Zipf's URL deduplication feature vs episodic competitors
- **Crawl Comparison**: Content extraction quality comparison
- **Export**: Download results as JSON

## Run Locally

```bash
pip install -r requirements.txt
streamlit run benchmark_app.py
```

## Pre-fill API Keys via URL

```
https://your-app.streamlit.app/?zipf=YOUR_KEY&exa=YOUR_KEY&firecrawl=YOUR_KEY
```

## Get API Keys

- [Zipf.ai](https://zipf.ai) - 100 free credits/month
- [Exa.ai](https://exa.ai) - $10 free credits
- [Firecrawl](https://firecrawl.dev) - 500 free credits
