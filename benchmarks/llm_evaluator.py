"""LLM-based evaluation for multi-step research benchmark.

Uses GPT-4 or Claude to:
1. Verify if specific facts were found in research results
2. Score context coherence across query sequence
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM evaluator."""
    provider: str  # "openai" or "anthropic"
    model: str
    api_key: str
    temperature: float = 0.0


def get_openai_client(api_key: str):
    """Get OpenAI client."""
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")


def get_anthropic_client(api_key: str):
    """Get Anthropic client."""
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")


# =============================================================================
# FACT VERIFICATION
# =============================================================================

FACT_VERIFICATION_PROMPT = """You are evaluating whether a research session successfully found specific facts.

Research Goal: {goal}
Required Fact: {fact_description}

Content gathered during research:
{content}

Did the research find this fact?
- FOUND: The fact is clearly stated in the content
- PARTIAL: The fact is implied or partially addressed
- NOT_FOUND: The fact is not present in the content

If FOUND or PARTIAL, quote the relevant excerpt (max 200 chars).

Respond in JSON format:
{{
    "status": "FOUND" | "PARTIAL" | "NOT_FOUND",
    "evidence": "quoted excerpt or N/A",
    "confidence": "HIGH" | "MEDIUM" | "LOW"
}}"""


def verify_fact_openai(
    client,
    model: str,
    goal: str,
    fact_description: str,
    content: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Verify a single fact using OpenAI."""
    prompt = FACT_VERIFICATION_PROMPT.format(
        goal=goal,
        fact_description=fact_description,
        content=content[:15000],  # Limit content length
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "status": result.get("status", "NOT_FOUND"),
            "evidence": result.get("evidence", "N/A"),
            "confidence": result.get("confidence", "LOW"),
        }
    except (json.JSONDecodeError, IndexError):
        return {
            "status": "NOT_FOUND",
            "evidence": "Error parsing LLM response",
            "confidence": "LOW",
        }


def verify_fact_anthropic(
    client,
    model: str,
    goal: str,
    fact_description: str,
    content: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Verify a single fact using Anthropic."""
    prompt = FACT_VERIFICATION_PROMPT.format(
        goal=goal,
        fact_description=fact_description,
        content=content[:15000],
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        # Extract JSON from response
        response_text = response.content[0].text
        # Try to find JSON in response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response_text[start:end])
            return {
                "status": result.get("status", "NOT_FOUND"),
                "evidence": result.get("evidence", "N/A"),
                "confidence": result.get("confidence", "LOW"),
            }
    except (json.JSONDecodeError, IndexError):
        pass

    return {
        "status": "NOT_FOUND",
        "evidence": "Error parsing LLM response",
        "confidence": "LOW",
    }


def verify_facts(
    config: LLMConfig,
    goal: str,
    facts: List[Dict[str, str]],
    content: str,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Verify multiple facts against research content.

    Args:
        config: LLM configuration
        goal: Research goal
        facts: List of {id, description} dicts
        content: Gathered research content
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of verification results
    """
    if config.provider == "openai":
        client = get_openai_client(config.api_key)
        verify_fn = verify_fact_openai
    elif config.provider == "anthropic":
        client = get_anthropic_client(config.api_key)
        verify_fn = verify_fact_anthropic
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

    results = []
    for i, fact in enumerate(facts):
        if progress_callback:
            progress_callback(i + 1, len(facts))

        result = verify_fn(
            client=client,
            model=config.model,
            goal=goal,
            fact_description=fact["description"],
            content=content,
            temperature=config.temperature,
        )
        result["fact_id"] = fact["id"]
        result["fact_description"] = fact["description"]

        # Calculate score
        if result["status"] == "FOUND":
            result["score"] = 1.0
        elif result["status"] == "PARTIAL":
            result["score"] = 0.5
        else:
            result["score"] = 0.0

        results.append(result)

    return results


# =============================================================================
# CONTEXT COHERENCE
# =============================================================================

COHERENCE_PROMPT = """You are evaluating how well search results build upon previous research context.

Research Goal: {goal}

Previous queries and key findings:
{context_summary}

Current query (#{query_number}): {current_query}

Results returned:
{results}

Rate the contextual coherence from 0.0 to 1.0:
- 1.0: Results perfectly complement prior findings, no redundancy, clear progression
- 0.8: Strong coherence, minor redundancy or tangential results
- 0.6: Moderate coherence, some useful new information
- 0.4: Weak coherence, mostly redundant or loosely related
- 0.2: Poor coherence, largely irrelevant to research progression
- 0.0: No coherence, completely off-topic

Respond in JSON format:
{{
    "score": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""


def evaluate_coherence_openai(
    client,
    model: str,
    goal: str,
    query_number: int,
    current_query: str,
    results: str,
    context_summary: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate context coherence using OpenAI."""
    prompt = COHERENCE_PROMPT.format(
        goal=goal,
        query_number=query_number,
        current_query=current_query,
        results=results[:5000],
        context_summary=context_summary,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "score": float(result.get("score", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, IndexError, ValueError):
        return {"score": 0.5, "reasoning": "Error parsing LLM response"}


def evaluate_coherence_anthropic(
    client,
    model: str,
    goal: str,
    query_number: int,
    current_query: str,
    results: str,
    context_summary: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate context coherence using Anthropic."""
    prompt = COHERENCE_PROMPT.format(
        goal=goal,
        query_number=query_number,
        current_query=current_query,
        results=results[:5000],
        context_summary=context_summary,
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        response_text = response.content[0].text
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response_text[start:end])
            return {
                "score": float(result.get("score", 0.5)),
                "reasoning": result.get("reasoning", ""),
            }
    except (json.JSONDecodeError, IndexError, ValueError):
        pass

    return {"score": 0.5, "reasoning": "Error parsing LLM response"}


def evaluate_coherence(
    config: LLMConfig,
    goal: str,
    query_results: List[Dict[str, Any]],
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate context coherence for each query in the sequence.

    Args:
        config: LLM configuration
        goal: Research goal
        query_results: List of query results with 'query', 'snippets', 'step'
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of coherence results
    """
    if config.provider == "openai":
        client = get_openai_client(config.api_key)
        eval_fn = evaluate_coherence_openai
    elif config.provider == "anthropic":
        client = get_anthropic_client(config.api_key)
        eval_fn = evaluate_coherence_anthropic
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

    results = []
    context_summary = []

    for i, qr in enumerate(query_results):
        if progress_callback:
            progress_callback(i + 1, len(query_results))

        # Build context from previous queries
        context_str = "\n".join(context_summary) if context_summary else "This is the first query."

        # Build results string
        results_str = "\n".join(qr.get("snippets", [])[:5])
        if not results_str:
            results_str = f"({qr.get('result_count', 0)} results returned, no snippets available)"

        result = eval_fn(
            client=client,
            model=config.model,
            goal=goal,
            query_number=qr["step"],
            current_query=qr["query"],
            results=results_str,
            context_summary=context_str,
            temperature=config.temperature,
        )

        result["query_number"] = qr["step"]
        result["query"] = qr["query"]
        results.append(result)

        # Update context for next query
        summary = f"Q{qr['step']}: {qr['query'][:50]}... -> {qr.get('result_count', 0)} results"
        if qr.get("snippets"):
            summary += f" (e.g., '{qr['snippets'][0][:100]}...')"
        context_summary.append(summary)

    return results


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def run_full_evaluation(
    config: LLMConfig,
    scenario_goal: str,
    facts: List[Dict[str, str]],
    query_results: List[Dict[str, Any]],
    all_content: str,
    fact_progress_callback: Optional[callable] = None,
    coherence_progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Run complete LLM evaluation (facts + coherence).

    Args:
        config: LLM configuration
        scenario_goal: Research goal
        facts: List of facts to verify
        query_results: List of query results
        all_content: All gathered content
        fact_progress_callback: Progress callback for fact verification
        coherence_progress_callback: Progress callback for coherence evaluation

    Returns:
        Dict with fact_results and coherence_results
    """
    # Verify facts
    fact_results = verify_facts(
        config=config,
        goal=scenario_goal,
        facts=facts,
        content=all_content,
        progress_callback=fact_progress_callback,
    )

    # Evaluate coherence
    coherence_results = evaluate_coherence(
        config=config,
        goal=scenario_goal,
        query_results=query_results,
        progress_callback=coherence_progress_callback,
    )

    return {
        "fact_results": fact_results,
        "coherence_results": coherence_results,
    }


# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def create_openai_config(api_key: str, model: str = "gpt-4o") -> LLMConfig:
    """Create OpenAI configuration."""
    return LLMConfig(
        provider="openai",
        model=model,
        api_key=api_key,
        temperature=0.0,
    )


def create_anthropic_config(api_key: str, model: str = "claude-3-5-sonnet-20241022") -> LLMConfig:
    """Create Anthropic configuration."""
    return LLMConfig(
        provider="anthropic",
        model=model,
        api_key=api_key,
        temperature=0.0,
    )
