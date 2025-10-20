import requests
from typing import Dict, Any, Optional

OPENF1_BASE = "https://api.openf1.org/v1"

def fetch_openf1(endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> list[dict]:
    """Generic helper to call OpenF1 API.
    Example: fetch_openf1("laps", {"session_key": 9479})
    """
    url = f"{OPENF1_BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        # some endpoints may return single object
        return [data]
    return data
