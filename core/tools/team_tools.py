"""
Team-local tools (in-process) for quick extensibility.

Add simple Python functions here and declare them in config under
gateway.tools with type "local" and the module path
e.g., core.tools.team_tools:product_lookup
"""

from typing import Any, Dict, List, Optional


def product_lookup(gtin: Optional[str] = None, tpnb: Optional[str] = None) -> Dict[str, Any]:
    """
    Lookup product details by GTIN or TPNB.

    Replace stub logic with a real implementation (DB/API) as needed.
    """
    if not gtin and not tpnb:
        return {"success": False, "error": "Provide gtin or tpnb"}

    return {
        "success": True,
        "query": {"gtin": gtin, "tpnb": tpnb},
        "product": {
            "id": tpnb or gtin,
            "name": "Stub Product",
            "category": "stub",
            "status": "unknown",
        },
    }


def location_lookup(cluster_id: str) -> Dict[str, Any]:
    """Resolve a location cluster to its stores (stub)."""
    if not cluster_id:
        return {"success": False, "error": "cluster_id required"}

    return {
        "success": True,
        "cluster_id": cluster_id,
        "stores": [
            {"store_id": "1001", "name": "Stub Store A"},
            {"store_id": "1002", "name": "Stub Store B"},
        ],
    }


def entity_normalize(value: str) -> Dict[str, Any]:
    """
    Minimal helper to normalize a business identifier; can be used by agents or tools.
    """
    v = (value or "").strip()
    return {
        "input": value,
        "normalized": v,
        "type": "gtin" if v.isdigit() and 12 <= len(v) <= 14 else "tpnb" if v.isdigit() else "unknown",
    }


