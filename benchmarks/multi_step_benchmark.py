"""Multi-Step Research Workflow Benchmark.

Evaluates session-aware search APIs for multi-query research sessions
where context from earlier queries should inform and improve later results.

Metrics:
- Information Completeness Score (ICS)
- Context Coherence Score (CCS)
- Deduplication Efficiency Ratio (DER)
- Research Velocity Index (RVI)
- Cost Efficiency Score (CES)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class ResearchFact:
    """A fact to verify in research results."""
    id: str
    description: str
    category: str
    weight: float = 1.0


@dataclass
class ResearchScenario:
    """A multi-step research scenario."""
    id: str
    name: str
    goal: str
    intent_type: str
    queries: List[str]
    facts: List[ResearchFact]


# =============================================================================
# RESEARCH SCENARIOS
# =============================================================================

SCENARIOS: Dict[str, ResearchScenario] = {
    "competitive_analysis": ResearchScenario(
        id="competitive_analysis",
        name="SaaS Competitive Analysis",
        goal="Conduct comprehensive competitive analysis of Notion vs. Coda vs. Confluence for enterprise wiki/documentation use cases.",
        intent_type="competitive_analysis",
        queries=[
            "Notion enterprise features pricing 2025",
            "Coda enterprise pricing features comparison",
            "Confluence enterprise pricing Atlassian 2025",
            "Notion vs Coda enterprise features comparison",
            "Notion vs Confluence enterprise wiki comparison",
            "enterprise wiki software market share 2025",
            "Notion enterprise customer reviews complaints",
            "Coda enterprise limitations problems",
            "Confluence enterprise migration challenges",
            "best enterprise wiki software analyst reports",
        ],
        facts=[
            ResearchFact("F1", "Notion Enterprise price per user", "Pricing"),
            ResearchFact("F2", "Coda Enterprise price per user", "Pricing"),
            ResearchFact("F3", "Confluence Enterprise price per user", "Pricing"),
            ResearchFact("F4", "Notion's AI features description", "Features"),
            ResearchFact("F5", "Coda's automation capabilities", "Features"),
            ResearchFact("F6", "Confluence's Atlassian integration depth", "Features"),
            ResearchFact("F7", "Notion's SOC 2 compliance status", "Security"),
            ResearchFact("F8", "Each product's SSO support", "Security"),
            ResearchFact("F9", "Market share or user count for each", "Market"),
            ResearchFact("F10", "Notable enterprise customers for each", "Market"),
            ResearchFact("F11", "G2/Capterra rating for each", "Sentiment"),
            ResearchFact("F12", "Common complaints for Notion", "Sentiment"),
            ResearchFact("F13", "Common complaints for Coda", "Sentiment"),
            ResearchFact("F14", "Common complaints for Confluence", "Sentiment"),
            ResearchFact("F15", "API/integration capabilities comparison", "Features"),
            ResearchFact("F16", "Mobile app quality comparison", "Features"),
            ResearchFact("F17", "Offline support comparison", "Features"),
            ResearchFact("F18", "Version history/backup features", "Features"),
            ResearchFact("F19", "Permission/access control granularity", "Security"),
            ResearchFact("F20", "Data residency options", "Security"),
            ResearchFact("F21", "Recent funding or acquisition news", "Market"),
            ResearchFact("F22", "Product roadmap or recent launches", "Features"),
            ResearchFact("F23", "Implementation/onboarding time estimates", "Operations"),
            ResearchFact("F24", "Customer support quality comparison", "Operations"),
            ResearchFact("F25", "Analyst recommendation (Gartner/Forrester)", "Market"),
        ]
    ),

    "due_diligence": ResearchScenario(
        id="due_diligence",
        name="Startup Due Diligence",
        goal="Conduct due diligence on Anthropic for potential investment, covering financials, team, technology, risks, and competitive position.",
        intent_type="due_diligence",
        queries=[
            "Anthropic funding history valuation 2024 2025",
            "Anthropic founders background Dario Amodei",
            "Anthropic Claude AI capabilities benchmarks",
            "Anthropic revenue ARR business model",
            "Anthropic Amazon Google investment partnership",
            "Anthropic vs OpenAI comparison market position",
            "Anthropic AI safety approach constitutional AI",
            "Anthropic risks challenges lawsuits regulatory",
            "Anthropic hiring growth employee count",
            "Anthropic enterprise customers case studies",
        ],
        facts=[
            ResearchFact("F1", "Total funding raised", "Financials"),
            ResearchFact("F2", "Latest valuation", "Financials"),
            ResearchFact("F3", "Key investors list", "Financials"),
            ResearchFact("F4", "Revenue or ARR estimate", "Financials"),
            ResearchFact("F5", "Dario Amodei background", "Team"),
            ResearchFact("F6", "Daniela Amodei background", "Team"),
            ResearchFact("F7", "Employee count", "Team"),
            ResearchFact("F8", "Claude model versions", "Technology"),
            ResearchFact("F9", "Benchmark performance vs GPT-4", "Technology"),
            ResearchFact("F10", "Constitutional AI explanation", "Technology"),
            ResearchFact("F11", "Amazon investment amount", "Partnerships"),
            ResearchFact("F12", "Google investment amount", "Partnerships"),
            ResearchFact("F13", "Enterprise customer examples", "Traction"),
            ResearchFact("F14", "API pricing structure", "Business Model"),
            ResearchFact("F15", "Main competitors", "Competition"),
            ResearchFact("F16", "Competitive advantages", "Competition"),
            ResearchFact("F17", "Regulatory risks", "Risks"),
            ResearchFact("F18", "Litigation history", "Risks"),
            ResearchFact("F19", "Key person dependency", "Risks"),
            ResearchFact("F20", "Compute/infrastructure costs", "Risks"),
            ResearchFact("F21", "Recent news/developments", "Current"),
            ResearchFact("F22", "Growth trajectory indicators", "Traction"),
        ]
    ),

    "technical_evaluation": ResearchScenario(
        id="technical_evaluation",
        name="Technical Framework Evaluation",
        goal="Evaluate React vs. Vue vs. Svelte for a new enterprise dashboard project, focusing on performance, ecosystem, and enterprise adoption.",
        intent_type="technical_evaluation",
        queries=[
            "React 19 performance benchmarks 2025",
            "Vue 3 performance benchmarks comparison",
            "Svelte 5 performance benchmarks comparison",
            "React vs Vue vs Svelte bundle size comparison",
            "React enterprise adoption Fortune 500",
            "Vue enterprise case studies large scale",
            "Svelte enterprise production examples",
            "React component library ecosystem 2025",
            "Vue component library ecosystem comparison",
            "Svelte component library ecosystem limitations",
            "React developer hiring availability salary",
            "frontend framework learning curve comparison",
        ],
        facts=[
            ResearchFact("F1", "React latest version", "Version"),
            ResearchFact("F2", "Vue latest version", "Version"),
            ResearchFact("F3", "Svelte latest version", "Version"),
            ResearchFact("F4", "React rendering benchmark score", "Performance"),
            ResearchFact("F5", "Vue rendering benchmark score", "Performance"),
            ResearchFact("F6", "Svelte rendering benchmark score", "Performance"),
            ResearchFact("F7", "React typical bundle size", "Performance"),
            ResearchFact("F8", "Vue typical bundle size", "Performance"),
            ResearchFact("F9", "Svelte typical bundle size", "Performance"),
            ResearchFact("F10", "React GitHub stars", "Popularity"),
            ResearchFact("F11", "Vue GitHub stars", "Popularity"),
            ResearchFact("F12", "Svelte GitHub stars", "Popularity"),
            ResearchFact("F13", "React npm weekly downloads", "Adoption"),
            ResearchFact("F14", "Vue npm weekly downloads", "Adoption"),
            ResearchFact("F15", "Svelte npm weekly downloads", "Adoption"),
            ResearchFact("F16", "Notable React enterprise users", "Enterprise"),
            ResearchFact("F17", "Notable Vue enterprise users", "Enterprise"),
            ResearchFact("F18", "Notable Svelte enterprise users", "Enterprise"),
            ResearchFact("F19", "Top React component libraries", "Ecosystem"),
            ResearchFact("F20", "Top Vue component libraries", "Ecosystem"),
            ResearchFact("F21", "Top Svelte component libraries", "Ecosystem"),
            ResearchFact("F22", "React developer salary range", "Hiring"),
            ResearchFact("F23", "Learning curve comparison", "DX"),
            ResearchFact("F24", "TypeScript support quality", "DX"),
        ]
    ),

    "market_research_ev": ResearchScenario(
        id="market_research_ev",
        name="EV Charging Market Research",
        goal="Analyze the US electric vehicle charging infrastructure market for a potential new entrant.",
        intent_type="market_research",
        queries=[
            "US EV charging market size 2025 forecast",
            "EV charging station companies market share US",
            "ChargePoint Tesla Supercharger EVgo comparison",
            "EV charging infrastructure government incentives 2025",
            "EV charging station installation costs economics",
            "EV charging network reliability problems",
            "EV charging station utilization rates data",
            "new EV charging startups funding 2024 2025",
            "EV charging technology trends bidirectional V2G",
            "EV adoption forecast US 2025 2030",
        ],
        facts=[
            ResearchFact("F1", "US EV charging market size 2025", "Market"),
            ResearchFact("F2", "Market growth rate CAGR", "Market"),
            ResearchFact("F3", "ChargePoint market share", "Competition"),
            ResearchFact("F4", "Tesla Supercharger market share", "Competition"),
            ResearchFact("F5", "EVgo market share", "Competition"),
            ResearchFact("F6", "Federal EV charging incentives", "Regulatory"),
            ResearchFact("F7", "State-level incentive examples", "Regulatory"),
            ResearchFact("F8", "Average installation cost Level 2", "Economics"),
            ResearchFact("F9", "Average installation cost DC fast", "Economics"),
            ResearchFact("F10", "Typical utilization rates", "Operations"),
            ResearchFact("F11", "Average revenue per charger", "Economics"),
            ResearchFact("F12", "Major pain points for users", "Problems"),
            ResearchFact("F13", "Major pain points for operators", "Problems"),
            ResearchFact("F14", "Recent startup funding examples", "Competition"),
            ResearchFact("F15", "V2G technology status", "Technology"),
            ResearchFact("F16", "Charging speed improvements", "Technology"),
            ResearchFact("F17", "US EV sales 2024", "Demand"),
            ResearchFact("F18", "US EV sales forecast 2030", "Demand"),
            ResearchFact("F19", "Charger-to-EV ratio current", "Infrastructure"),
            ResearchFact("F20", "Charger-to-EV ratio target", "Infrastructure"),
        ]
    ),

    "academic_research": ResearchScenario(
        id="academic_research",
        name="Academic Literature Review",
        goal="Conduct a literature review on transformer architecture improvements for efficient inference (2023-2025).",
        intent_type="academic_research",
        queries=[
            "transformer efficient inference survey paper 2024",
            "flash attention paper algorithm explanation",
            "speculative decoding transformer inference paper",
            "KV cache optimization transformer papers",
            "mixture of experts efficient inference MoE",
            "quantization LLM inference INT8 INT4 papers",
            "transformer pruning structured unstructured papers",
            "efficient transformer inference benchmark comparison",
            "transformer inference optimization open source implementations",
            "future directions efficient LLM inference research",
        ],
        facts=[
            ResearchFact("F1", "Flash Attention paper citation", "Papers"),
            ResearchFact("F2", "Flash Attention speedup factor", "Results"),
            ResearchFact("F3", "Speculative decoding paper citation", "Papers"),
            ResearchFact("F4", "Speculative decoding speedup factor", "Results"),
            ResearchFact("F5", "Key KV cache optimization papers", "Papers"),
            ResearchFact("F6", "KV cache memory reduction achieved", "Results"),
            ResearchFact("F7", "Mixtral MoE paper citation", "Papers"),
            ResearchFact("F8", "MoE efficiency gains", "Results"),
            ResearchFact("F9", "Leading quantization papers", "Papers"),
            ResearchFact("F10", "INT4 accuracy vs FP16", "Results"),
            ResearchFact("F11", "Pruning papers citations", "Papers"),
            ResearchFact("F12", "Pruning sparsity vs accuracy tradeoff", "Results"),
            ResearchFact("F13", "vLLM implementation details", "Implementation"),
            ResearchFact("F14", "TensorRT-LLM features", "Implementation"),
            ResearchFact("F15", "Benchmark datasets used", "Benchmarks"),
            ResearchFact("F16", "SOTA inference speed results", "Benchmarks"),
            ResearchFact("F17", "Open research questions", "Gaps"),
            ResearchFact("F18", "Hardware-software co-design trends", "Trends"),
            ResearchFact("F19", "Most cited authors in field", "Authors"),
            ResearchFact("F20", "Key conferences for this research", "Venues"),
        ]
    ),

    "news_monitoring": ResearchScenario(
        id="news_monitoring",
        name="AI Regulation News Monitoring",
        goal="Monitor and synthesize recent developments in AI regulation across US, EU, and China.",
        intent_type="news_monitoring",
        queries=[
            "EU AI Act implementation 2025 latest news",
            "US AI regulation executive order Congress 2025",
            "China AI regulation latest rules 2025",
            "AI regulation impact tech companies compliance",
            "OpenAI Anthropic Google AI regulation response",
            "AI regulation lobbying tech industry",
            "AI safety regulation proposals international",
            "AI regulation enforcement actions fines",
        ],
        facts=[
            ResearchFact("F1", "EU AI Act effective date", "EU"),
            ResearchFact("F2", "EU AI Act key requirements", "EU"),
            ResearchFact("F3", "EU AI Act penalties", "EU"),
            ResearchFact("F4", "US executive order status", "US"),
            ResearchFact("F5", "US Congressional AI bills", "US"),
            ResearchFact("F6", "China AI regulation key rules", "China"),
            ResearchFact("F7", "OpenAI regulatory stance", "Companies"),
            ResearchFact("F8", "Anthropic regulatory stance", "Companies"),
            ResearchFact("F9", "Google regulatory stance", "Companies"),
            ResearchFact("F10", "Compliance cost estimates", "Impact"),
            ResearchFact("F11", "International coordination efforts", "Global"),
            ResearchFact("F12", "Recent enforcement actions", "Enforcement"),
            ResearchFact("F13", "Industry lobbying spending", "Politics"),
            ResearchFact("F14", "Key regulatory agencies", "Governance"),
            ResearchFact("F15", "Timeline of upcoming deadlines", "Timeline"),
            ResearchFact("F16", "Expert predictions", "Analysis"),
        ]
    ),

    "lead_generation": ResearchScenario(
        id="lead_generation",
        name="DevTools Lead Generation",
        goal="Identify and research 10 promising DevTools startups for partnership outreach.",
        intent_type="lead_generation",
        queries=[
            "developer tools startups funding 2024 2025",
            "YC developer tools companies recent batches",
            "developer productivity tools Series A B 2024",
            "AI coding assistant startups besides Copilot",
            "developer experience DX platform startups",
            "infrastructure as code startups Terraform alternatives",
            "observability monitoring startups Datadog alternatives",
            "CI CD pipeline startups GitHub Actions alternatives",
            "developer tools startup founders LinkedIn contacts",
            "developer tools startup CTO VP Engineering contacts",
        ],
        facts=[
            ResearchFact("F1", "Company 1 name + description", "Companies"),
            ResearchFact("F2", "Company 2 name + description", "Companies"),
            ResearchFact("F3", "Company 3 name + description", "Companies"),
            ResearchFact("F4", "Company 4 name + description", "Companies"),
            ResearchFact("F5", "Company 5 name + description", "Companies"),
            ResearchFact("F6", "Company 6 name + description", "Companies"),
            ResearchFact("F7", "Company 7 name + description", "Companies"),
            ResearchFact("F8", "Company 8 name + description", "Companies"),
            ResearchFact("F9", "Company 9 name + description", "Companies"),
            ResearchFact("F10", "Company 10 name + description", "Companies"),
            ResearchFact("F11", "Funding amount for company 1", "Financials"),
            ResearchFact("F12", "Funding amount for company 2", "Financials"),
            ResearchFact("F13", "Funding amount for company 3", "Financials"),
            ResearchFact("F14", "Funding amount for company 4", "Financials"),
            ResearchFact("F15", "Funding amount for company 5", "Financials"),
            ResearchFact("F16", "Founder names for company 1", "Contacts"),
            ResearchFact("F17", "Founder names for company 2", "Contacts"),
            ResearchFact("F18", "Founder names for company 3", "Contacts"),
            ResearchFact("F19", "Common investor names", "Investors"),
            ResearchFact("F20", "Product categories represented", "Market"),
        ]
    ),

    "api_security": ResearchScenario(
        id="api_security",
        name="API Security Evaluation",
        goal="Evaluate API security best practices and tools for a fintech platform audit.",
        intent_type="technical_evaluation",
        queries=[
            "API security best practices OWASP 2025",
            "API gateway security features comparison",
            "OAuth 2.0 API security implementation guide",
            "API rate limiting DDoS protection best practices",
            "API security testing tools DAST SAST",
            "fintech API security compliance PCI DSS",
            "API security breaches case studies 2024",
            "API security monitoring logging best practices",
            "zero trust API security architecture",
            "API security audit checklist enterprise",
        ],
        facts=[
            ResearchFact("F1", "OWASP API Top 10 list", "Standards"),
            ResearchFact("F2", "API gateway comparison (Kong, Apigee, AWS)", "Tools"),
            ResearchFact("F3", "OAuth 2.0 best practices", "Auth"),
            ResearchFact("F4", "Rate limiting strategies", "Protection"),
            ResearchFact("F5", "DDoS protection approaches", "Protection"),
            ResearchFact("F6", "SAST tools for API security", "Testing"),
            ResearchFact("F7", "DAST tools for API security", "Testing"),
            ResearchFact("F8", "PCI DSS API requirements", "Compliance"),
            ResearchFact("F9", "Notable API security breaches", "Case Studies"),
            ResearchFact("F10", "Breach root causes", "Case Studies"),
            ResearchFact("F11", "Logging best practices", "Operations"),
            ResearchFact("F12", "Monitoring tools", "Operations"),
            ResearchFact("F13", "Zero trust principles for APIs", "Architecture"),
            ResearchFact("F14", "mTLS implementation", "Architecture"),
            ResearchFact("F15", "Audit checklist items", "Audit"),
            ResearchFact("F16", "Compliance frameworks", "Compliance"),
            ResearchFact("F17", "Security headers to implement", "Standards"),
            ResearchFact("F18", "Input validation best practices", "Standards"),
            ResearchFact("F19", "Secret management solutions", "Tools"),
            ResearchFact("F20", "API security training resources", "Resources"),
        ]
    ),

    "cloud_costs": ResearchScenario(
        id="cloud_costs",
        name="Cloud Cost Comparison",
        goal="Compare AWS vs. Azure vs. GCP pricing and cost optimization for a mid-size SaaS company.",
        intent_type="competitive_analysis",
        queries=[
            "AWS pricing calculator compute storage 2025",
            "Azure pricing calculator comparison AWS",
            "Google Cloud pricing comparison AWS Azure",
            "cloud cost optimization tools comparison",
            "AWS reserved instances savings plans comparison",
            "Azure reserved capacity committed use discounts",
            "GCP committed use discounts sustained use",
            "cloud cost case studies SaaS companies",
            "multi-cloud cost management strategies",
            "cloud cost benchmarks per employee SaaS",
        ],
        facts=[
            ResearchFact("F1", "AWS compute pricing (EC2)", "Pricing"),
            ResearchFact("F2", "Azure compute pricing (VMs)", "Pricing"),
            ResearchFact("F3", "GCP compute pricing (Compute Engine)", "Pricing"),
            ResearchFact("F4", "AWS storage pricing (S3)", "Pricing"),
            ResearchFact("F5", "Azure storage pricing (Blob)", "Pricing"),
            ResearchFact("F6", "GCP storage pricing (Cloud Storage)", "Pricing"),
            ResearchFact("F7", "AWS Reserved Instances savings %", "Discounts"),
            ResearchFact("F8", "Azure Reserved Capacity savings %", "Discounts"),
            ResearchFact("F9", "GCP Committed Use savings %", "Discounts"),
            ResearchFact("F10", "Top cost optimization tools", "Tools"),
            ResearchFact("F11", "FinOps best practices", "Strategy"),
            ResearchFact("F12", "Multi-cloud cost management approaches", "Strategy"),
            ResearchFact("F13", "SaaS company cloud cost benchmarks", "Benchmarks"),
            ResearchFact("F14", "Cost per employee benchmarks", "Benchmarks"),
            ResearchFact("F15", "Egress pricing comparison", "Pricing"),
            ResearchFact("F16", "Data transfer costs between clouds", "Pricing"),
            ResearchFact("F17", "Spot/Preemptible instance savings", "Discounts"),
            ResearchFact("F18", "Cost allocation tagging strategies", "Operations"),
            ResearchFact("F19", "FinOps team structure", "Operations"),
            ResearchFact("F20", "Cost anomaly detection tools", "Tools"),
        ]
    ),

    "sea_fintech": ResearchScenario(
        id="sea_fintech",
        name="Southeast Asia Fintech Market",
        goal="Assess Southeast Asia fintech market for potential expansion, focusing on Indonesia, Vietnam, and Philippines.",
        intent_type="market_research",
        queries=[
            "Southeast Asia fintech market size 2025",
            "Indonesia fintech regulation OJK 2025",
            "Vietnam fintech regulation State Bank",
            "Philippines fintech regulation BSP",
            "top fintech companies Indonesia GoPay OVO",
            "top fintech companies Vietnam MoMo VNPay",
            "top fintech companies Philippines GCash Maya",
            "Southeast Asia fintech funding 2024 2025",
            "digital payments adoption Southeast Asia statistics",
            "fintech market entry barriers Southeast Asia",
        ],
        facts=[
            ResearchFact("F1", "SEA fintech market size 2025", "Market"),
            ResearchFact("F2", "Market growth rate CAGR", "Market"),
            ResearchFact("F3", "Indonesia OJK licensing requirements", "Regulatory"),
            ResearchFact("F4", "Vietnam SBV fintech regulations", "Regulatory"),
            ResearchFact("F5", "Philippines BSP requirements", "Regulatory"),
            ResearchFact("F6", "GoPay market position", "Competition"),
            ResearchFact("F7", "OVO market position", "Competition"),
            ResearchFact("F8", "MoMo market position", "Competition"),
            ResearchFact("F9", "VNPay market position", "Competition"),
            ResearchFact("F10", "GCash market position", "Competition"),
            ResearchFact("F11", "Maya market position", "Competition"),
            ResearchFact("F12", "Recent funding rounds in SEA", "Investment"),
            ResearchFact("F13", "Key investors in SEA fintech", "Investment"),
            ResearchFact("F14", "Digital payment adoption rates", "Adoption"),
            ResearchFact("F15", "Unbanked population statistics", "Opportunity"),
            ResearchFact("F16", "Mobile penetration rates", "Infrastructure"),
            ResearchFact("F17", "Market entry barriers", "Challenges"),
            ResearchFact("F18", "Local partnership requirements", "Challenges"),
            ResearchFact("F19", "Currency/FX considerations", "Operations"),
            ResearchFact("F20", "Success stories of foreign entrants", "Case Studies"),
        ]
    ),
}


# =============================================================================
# EXECUTION MODES
# =============================================================================

@dataclass
class ExecutionMode:
    """Defines how a provider executes the benchmark."""
    name: str
    description: str
    uses_session: bool
    injects_context: bool


EXECUTION_MODES = {
    "zipf_native": ExecutionMode(
        name="Zipf.ai (Native)",
        description="Uses Sessions API with auto-deduplication and context",
        uses_session=True,
        injects_context=False,
    ),
    "exa_native": ExecutionMode(
        name="Exa.ai (Native)",
        description="Stateless queries, no context passing",
        uses_session=False,
        injects_context=False,
    ),
    "exa_enhanced": ExecutionMode(
        name="Exa.ai (Enhanced)",
        description="Manual context injection: previous queries + summaries passed in prompt",
        uses_session=False,
        injects_context=True,
    ),
    "firecrawl_native": ExecutionMode(
        name="Firecrawl (Native)",
        description="Stateless queries, no context passing",
        uses_session=False,
        injects_context=False,
    ),
    "firecrawl_enhanced": ExecutionMode(
        name="Firecrawl (Enhanced)",
        description="Manual context injection: previous queries + summaries passed in prompt",
        uses_session=False,
        injects_context=True,
    ),
}


# =============================================================================
# DATA STRUCTURES FOR RESULTS
# =============================================================================

@dataclass
class QueryResult:
    """Result from a single query in the workflow."""
    step: int
    query: str
    urls: List[str]
    snippets: List[str]
    latency_ms: float
    credits_used: float
    result_count: int
    error: Optional[str] = None


@dataclass
class FactResult:
    """Result of fact verification."""
    fact_id: str
    fact_description: str
    status: str  # FOUND, PARTIAL, NOT_FOUND
    evidence: str
    confidence: str  # HIGH, MEDIUM, LOW
    score: float  # 1.0, 0.5, 0.0


@dataclass
class CoherenceResult:
    """Result of context coherence evaluation."""
    query_number: int
    query: str
    score: float
    reasoning: str


@dataclass
class BenchmarkMetrics:
    """Computed metrics for a benchmark run."""
    # Primary metrics
    information_completeness_score: float  # ICS: 0-100%
    context_coherence_score: float  # CCS: 0-1
    deduplication_efficiency_ratio: float  # DER: 0-1
    research_velocity_index: float  # RVI
    cost_efficiency_score: float  # CES

    # Secondary metrics
    query_refinement_rate: float
    dead_end_rate: float
    source_diversity: float
    unique_domains: int
    total_urls: int
    unique_urls: int
    duplicate_urls: int

    # Raw counts
    facts_found: int
    facts_partial: int
    facts_not_found: int
    total_facts: int


@dataclass
class BenchmarkRun:
    """Complete results from a benchmark run."""
    scenario_id: str
    provider: str
    mode: str
    run_number: int
    timestamp: str
    query_results: List[QueryResult]
    fact_results: List[FactResult]
    coherence_results: List[CoherenceResult]
    metrics: BenchmarkMetrics
    total_latency_ms: float
    total_credits: float


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================

def run_multi_step_benchmark(
    scenario: ResearchScenario,
    client: Any,
    mode: ExecutionMode,
    max_results_per_query: int = 10,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a multi-step research workflow benchmark.

    Args:
        scenario: The research scenario to run
        client: API client (Zipf, Exa, or Firecrawl)
        mode: Execution mode configuration
        max_results_per_query: Max results per query
        session_id: Optional session ID for Zipf

    Returns:
        Dict with query results and raw data for later metric computation
    """
    query_results = []
    all_urls = []
    all_snippets = []
    context_summary = []

    for step, query in enumerate(scenario.queries, 1):
        # For enhanced mode, inject context into query
        actual_query = query
        if mode.injects_context and context_summary:
            context_str = " | ".join(context_summary[-3:])  # Last 3 queries
            actual_query = f"{query} (Context: {context_str})"

        # Execute search
        start_time = time.time()
        try:
            if mode.uses_session and session_id:
                result = client.search(actual_query, max_results=max_results_per_query, session_id=session_id)
            else:
                result = client.search(actual_query, max_results=max_results_per_query)

            latency_ms = (time.time() - start_time) * 1000

            urls = [r.get("url", "") for r in result.get("results", [])]
            snippets = [r.get("snippet", "") for r in result.get("results", [])]

            query_result = QueryResult(
                step=step,
                query=query,
                urls=urls,
                snippets=snippets,
                latency_ms=result.get("latency_ms", latency_ms),
                credits_used=float(result.get("credits_used", 0) or 0),
                result_count=result.get("result_count", len(urls)),
                error=result.get("error"),
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            query_result = QueryResult(
                step=step,
                query=query,
                urls=[],
                snippets=[],
                latency_ms=latency_ms,
                credits_used=0,
                result_count=0,
                error=str(e),
            )

        query_results.append(query_result)
        all_urls.extend(query_result.urls)
        all_snippets.extend(query_result.snippets)

        # Update context summary for enhanced mode
        if query_result.snippets:
            summary = f"Q{step}: {query[:50]}... -> {query_result.result_count} results"
            context_summary.append(summary)

    # Compute basic deduplication stats
    unique_urls = set(all_urls)
    duplicate_count = len(all_urls) - len(unique_urls)

    # Extract unique domains
    domains = set()
    for url in all_urls:
        if url:
            parts = url.split("/")
            if len(parts) >= 3:
                domains.add(parts[2])

    return {
        "scenario_id": scenario.id,
        "scenario_name": scenario.name,
        "provider": client.name,
        "mode": mode.name,
        "query_results": [
            {
                "step": qr.step,
                "query": qr.query,
                "urls": qr.urls,
                "snippets": qr.snippets,
                "latency_ms": qr.latency_ms,
                "credits_used": qr.credits_used,
                "result_count": qr.result_count,
                "error": qr.error,
            }
            for qr in query_results
        ],
        "all_content": "\n\n".join(all_snippets),
        "total_urls": len(all_urls),
        "unique_urls": len(unique_urls),
        "duplicate_urls": duplicate_count,
        "unique_domains": len(domains),
        "domains": list(domains),
        "total_latency_ms": sum(qr.latency_ms for qr in query_results),
        "total_credits": sum(qr.credits_used for qr in query_results),
        "queries_run": len(query_results),
        "dead_ends": sum(1 for qr in query_results if qr.result_count < 2),
    }


def compute_metrics(
    benchmark_result: Dict[str, Any],
    fact_results: Optional[List[Dict[str, Any]]] = None,
    coherence_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute all benchmark metrics from raw results.

    Args:
        benchmark_result: Output from run_multi_step_benchmark
        fact_results: Optional LLM-evaluated fact verification results
        coherence_results: Optional LLM-evaluated coherence scores

    Returns:
        Dict with all computed metrics
    """
    metrics = {}

    # Deduplication Efficiency Ratio (DER)
    total_urls = benchmark_result["total_urls"]
    unique_urls = benchmark_result["unique_urls"]
    metrics["deduplication_efficiency_ratio"] = (
        unique_urls / total_urls if total_urls > 0 else 1.0
    )

    # Source Diversity
    metrics["source_diversity"] = (
        benchmark_result["unique_domains"] / unique_urls if unique_urls > 0 else 0
    )

    # Dead End Rate
    queries_run = benchmark_result["queries_run"]
    metrics["dead_end_rate"] = (
        benchmark_result["dead_ends"] / queries_run if queries_run > 0 else 0
    )

    # Information Completeness Score (ICS) - requires LLM evaluation
    if fact_results:
        found = sum(1 for f in fact_results if f["status"] == "FOUND")
        partial = sum(1 for f in fact_results if f["status"] == "PARTIAL")
        total = len(fact_results)
        metrics["information_completeness_score"] = (
            ((found + partial * 0.5) / total * 100) if total > 0 else 0
        )
        metrics["facts_found"] = found
        metrics["facts_partial"] = partial
        metrics["facts_not_found"] = total - found - partial
        metrics["total_facts"] = total
    else:
        metrics["information_completeness_score"] = None
        metrics["facts_found"] = None
        metrics["facts_partial"] = None
        metrics["facts_not_found"] = None
        metrics["total_facts"] = None

    # Context Coherence Score (CCS) - requires LLM evaluation
    if coherence_results:
        total_weight = 0
        weighted_score = 0
        for i, cr in enumerate(coherence_results, 1):
            weight = 1 + (i - 1) * 0.2  # Later queries weighted more heavily
            weighted_score += cr["score"] * weight
            total_weight += weight
        metrics["context_coherence_score"] = (
            weighted_score / total_weight if total_weight > 0 else 0
        )
    else:
        metrics["context_coherence_score"] = None

    # Research Velocity Index (RVI)
    ics = metrics["information_completeness_score"]
    total_latency_s = benchmark_result["total_latency_ms"] / 1000
    if ics is not None and total_latency_s > 0 and queries_run > 0:
        metrics["research_velocity_index"] = ics / (queries_run * total_latency_s)
    else:
        metrics["research_velocity_index"] = None

    # Cost Efficiency Score (CES)
    total_credits = benchmark_result["total_credits"]
    if ics is not None and total_credits > 0:
        metrics["cost_efficiency_score"] = ics / total_credits
    else:
        metrics["cost_efficiency_score"] = None

    # Add raw stats
    metrics["total_urls"] = total_urls
    metrics["unique_urls"] = unique_urls
    metrics["duplicate_urls"] = benchmark_result["duplicate_urls"]
    metrics["unique_domains"] = benchmark_result["unique_domains"]
    metrics["total_latency_ms"] = benchmark_result["total_latency_ms"]
    metrics["total_credits"] = benchmark_result["total_credits"]
    metrics["queries_run"] = queries_run

    return metrics


def get_scenario(scenario_id: str) -> Optional[ResearchScenario]:
    """Get a scenario by ID."""
    return SCENARIOS.get(scenario_id)


def list_scenarios() -> List[Dict[str, str]]:
    """List all available scenarios."""
    return [
        {
            "id": s.id,
            "name": s.name,
            "goal": s.goal,
            "intent_type": s.intent_type,
            "query_count": len(s.queries),
            "fact_count": len(s.facts),
        }
        for s in SCENARIOS.values()
    ]
