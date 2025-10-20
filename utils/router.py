from typing import Literal, Tuple
from services.summary_builders import (
    _extract_year, is_year_query, is_gp_query, is_season_query, build_gp_summary,
    summarize_season_fact, summarize_plain, build_season_summary_text
)
QueryType = Literal["season", "gp", "unknown"]

def route_query(query: str) -> QueryType:
    try:
        if is_year_query(query) and not is_gp_query(query):
            return "season"
        elif is_gp_query(query) and is_year_query(query):
            return "gp"
        else:
            return "unknown"
    except Exception:
        return "unknown"

def build_prompt(query: str) -> Tuple[str, str]:
    """Return (prompt, facts) depending on routed type."""
    qtype = route_query(query)
    if qtype == "season":
        q = query.strip().lower()
        year = _extract_year(q)
        facts = summarize_season_fact(build_season_summary_text(year))
        prompt = f"""You are a concise Formula 1 analyst. Answer the following question: {query}
        Write a compact season summary (6–10 sentences). USE ONLY the facts below. Do not speculate.
ss
        Facts:
        {facts}
        """.strip()
        return prompt, facts
    elif qtype == "gp":
        facts = summarize_plain(build_gp_summary(query))
        prompt = f"""You are a concise Formula 1 analyst. Answer the following question: {query}
        Write a short Grand Prix summary (5–8 sentences). USE ONLY the facts below.
        Do not speculate. End with one sentence about the circuit/context.

        Facts:
        {facts}
        """.strip()
        return prompt, facts
    else:
        prompt = "Cannot determine if the query is about a season or a Grand Prix. Please clarify."
        return prompt, ""
