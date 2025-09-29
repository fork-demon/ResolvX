"""
Local mock tools for dev/testing when centralized services are unavailable.

Map these functions in config under gateway.tools with type "local".
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


def splunk_search_mock(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return stubbed Splunk search results."""
    return {
        "success": True,
        "query": query,
        "executed": True,
        "results": [
            {"_time": datetime.utcnow().isoformat() + "Z", "count": 7, "source": "app.log"},
            {"_time": datetime.utcnow().isoformat() + "Z", "count": 3, "source": "api.log"},
        ],
        "context": context or {},
    }


def newrelic_metrics_mock(nrql: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return stubbed New Relic query results."""
    return {
        "success": True,
        "nrql": nrql,
        "executed": True,
        "results": [
            {"timestamp": datetime.utcnow().isoformat() + "Z", "appName": "stub-app", "value": 123},
        ],
        "context": context or {},
    }


def zendesk_poll_queue_mock(team: str, queue_name: str, status: str = "new", limit: int = 10) -> Dict[str, Any]:
    """Return stubbed Zendesk tickets for a queue."""
    tickets: List[Dict[str, Any]] = []
    for i in range(min(limit, 5)):
        tickets.append({
            "id": f"Z-{queue_name}-{i+1}",
            "subject": f"Stub ticket {i+1} in {queue_name}",
            "status": status,
            "priority": "normal",
            "requester_id": f"U-{i+1}",
            "assignee_id": None,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "tags": ["stub", queue_name],
            "description": "This is a stubbed ticket for local testing.",
        })
    return {
        "success": True,
        "team": team,
        "queue": queue_name,
        "status": status,
        "ticket_count": len(tickets),
        "tickets": tickets,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


