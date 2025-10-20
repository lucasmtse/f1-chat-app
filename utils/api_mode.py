from typing import Dict, Any
from utils.router import route_query
from services.summary_builders import (
    build_gp_summary, build_season_summary_text, summarize_plain, summarize_season_fact, is_season_query
)


def build_api_response(query: str) -> Dict[str, Any]:
    """Return a machine-friendly JSON with routing + facts strings (no LLM)."""
    qtype = route_query(query)
    if qtype == "season":
        facts = summarize_season_fact(build_season_summary_text(query))
        return {
            "query": query,
            "type": qtype,
            "facts": facts,
        }
    elif qtype == "gp":
        facts = summarize_plain(build_gp_summary(query))
        return {
            "query": query,
            "type": qtype,
            "facts": facts,
        }
    else:
        return {
            "query": query,
            "type": "unknown",
            "message": "Cannot determine if the query is about a season or a Grand Prix."
        }
