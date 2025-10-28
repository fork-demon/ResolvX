"""
Central MCP Gateway HTTP Wrapper
Wraps the MCP stdio server with FastAPI for HTTP/JSON-RPC access
Port: 8081
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os
import json
from datetime import datetime

app = FastAPI(title="Central MCP Gateway", version="1.0.0")


class MCPRequest(BaseModel):
    jsonrpc: str
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None


# Tool definitions
TOOLS = [
    {
        "name": "poll_queue",
        "description": "Poll Zendesk support queue for new tickets. Returns list of tickets from specified queue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "queue_name": {"type": "string", "description": "Queue name (e.g., 'pricing', 'urgent', 'l2_support')", "default": "pricing"},
                "limit": {"type": "integer", "description": "Maximum number of tickets to return", "default": 10},
                "status": {"type": "string", "description": "Filter by ticket status", "enum": ["new", "open", "pending", "solved"]}
            },
            "required": ["queue_name"]
        },
        "server": "central-gateway",
        "capabilities": []
    },
    {
        "name": "get_queue_stats",
        "description": "Get statistics for Zendesk support queues. Returns counts by status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "queue_name": {"type": "string", "description": "Queue name", "default": "pricing"}
            }
        },
        "server": "central-gateway",
        "capabilities": []
    },
    {
        "name": "splunk_search",
        "description": "Search Splunk logs for errors, warnings, or patterns. Use for diagnostic investigation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Splunk SPL query"},
                "time_range": {"type": "string", "description": "Time range (e.g., '-1h', '-24h')", "default": "-1h"},
                "max_results": {"type": "integer", "description": "Maximum results", "default": 100}
            },
            "required": ["query"]
        },
        "server": "central-gateway",
        "capabilities": []
    },
    {
        "name": "newrelic_query",
        "description": "Query New Relic for metrics, traces, or errors. Use for performance analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nrql": {"type": "string", "description": "NRQL query"},
                "account_id": {"type": "string", "description": "New Relic account ID", "default": "1234567"},
                "time_range": {"type": "string", "description": "Time range in minutes", "default": "60"}
            },
            "required": ["nrql"]
        },
        "server": "central-gateway",
        "capabilities": []
    },
    {
        "name": "memory_search",
        "description": "Search historical tickets for similar issues or resolutions. Use for duplicate detection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "similarity_threshold": {"type": "number", "description": "Similarity threshold (0.0-1.0)", "default": 0.9},
                "max_results": {"type": "integer", "description": "Maximum results", "default": 5}
            },
            "required": ["query"]
        },
        "server": "central-gateway",
        "capabilities": []
    },
]


def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return results."""
    
    if tool_name == "poll_queue":
        queue_name = parameters.get("queue_name", "pricing")
        limit = parameters.get("limit", 10)
        status = parameters.get("status")
        
        # Try to read tickets from filesystem
        base_dir = os.getenv("MOCK_TICKETS_DIR", "/data/mock_tickets").rstrip("/")
        tickets: List[Dict[str, Any]] = []
        
        search_paths = [os.path.join(base_dir, queue_name), base_dir]
        for path in search_paths:
            if os.path.isdir(path):
                for fname in sorted(os.listdir(path)):
                    if not fname.lower().endswith(".json"):
                        continue
                    fpath = os.path.join(path, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            ticket = json.load(f)
                            if status and str(ticket.get("status", "")).lower() != str(status).lower():
                                continue
                            tickets.append(ticket)
                            if len(tickets) >= limit:
                                break
                    except Exception:
                        continue
            if len(tickets) >= limit:
                break
        
        # Fallback to stubs if no files
        if not tickets:
            for i in range(min(limit, 3)):
                tickets.append({
                    "id": f"Z-{queue_name}-{i+1}",
                    "subject": f"Stub ticket {i+1} in {queue_name}",
                    "status": status or "new",
                    "priority": "normal",
                    "requester_id": f"U-{i+1}",
                    "assignee_id": None,
                    "created_at": "2025-10-21T00:00:00Z",
                    "updated_at": "2025-10-21T00:00:00Z",
                    "tags": ["stub", queue_name],
                    "description": "This is a stubbed ticket for local testing."
                })
        
        return {"success": True, "tickets": tickets, "count": len(tickets)}
    
    elif tool_name == "get_queue_stats":
        queue_name = parameters.get("queue_name", "pricing")
        return {
            "success": True,
            "queue": queue_name,
            "stats": {"new": 5, "open": 12, "pending": 3, "solved": 45},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    elif tool_name == "splunk_search":
        query = parameters.get("query", "")
        time_range = parameters.get("time_range", "-1h")
        max_results = parameters.get("max_results", 100)
        
        return {
            "success": True,
            "query": query,
            "time_range": time_range,
            "results": [
                {"_time": "2025-10-28T14:30:00Z", "service": "price-api", "level": "ERROR", "message": "Connection timeout to Quote service", "count": 5},
                {"_time": "2025-10-28T14:25:00Z", "service": "price-api", "level": "WARN", "message": "Slow response from Adaptor", "duration_ms": 3500},
                {"_time": "2025-10-28T14:20:00Z", "service": "price-api", "level": "ERROR", "message": "Failed to fetch base price for GTIN 12345678", "count": 2}
            ],
            "result_count": 3
        }
    
    elif tool_name == "newrelic_query":
        nrql = parameters.get("nrql", "")
        account_id = parameters.get("account_id", "1234567")
        time_range = parameters.get("time_range", "60")
        
        return {
            "success": True,
            "nrql": nrql,
            "account_id": account_id,
            "time_range_minutes": time_range,
            "results": [
                {"timestamp": "2025-10-28T14:30:00Z", "appName": "PriceAPI", "average_duration": 245.5, "error_rate": 0.02},
                {"timestamp": "2025-10-28T14:25:00Z", "appName": "PriceAPI", "average_duration": 189.3, "error_rate": 0.01},
                {"timestamp": "2025-10-28T14:20:00Z", "appName": "PriceAPI", "average_duration": 512.8, "error_rate": 0.05}
            ],
            "summary": {"avg_duration_ms": 315.9, "avg_error_rate": 0.027, "samples": 3}
        }
    
    elif tool_name == "memory_search":
        query = parameters.get("query", "")
        threshold = parameters.get("similarity_threshold", 0.9)
        max_results = parameters.get("max_results", 5)
        
        return {
            "success": True,
            "query": query,
            "similarity_threshold": threshold,
            "matches": [
                {
                    "ticket_id": "Z-HIST-12345",
                    "subject": "Price not found for GTIN 12345678",
                    "similarity": 0.95,
                    "status": "solved",
                    "resolution": "Quote service was down. Restarted and prices restored.",
                    "resolved_at": "2025-10-15T10:30:00Z",
                    "resolver": "L2 Support"
                },
                {
                    "ticket_id": "Z-HIST-12290",
                    "subject": "Missing prices for multiple products",
                    "similarity": 0.92,
                    "status": "solved",
                    "resolution": "Database sync issue between Quote and Price API. Ran manual sync.",
                    "resolved_at": "2025-10-10T14:15:00Z",
                    "resolver": "Engineering"
                }
            ],
            "match_count": 2
        }
    
    else:
        return {"error": f"Unknown tool: {tool_name}"}


@app.post("/mcp")
def mcp_endpoint(req: MCPRequest):
    """MCP JSON-RPC endpoint."""
    
    if req.method == "health":
        return {"jsonrpc": "2.0", "id": req.id, "result": {"status": "ok", "gateway": "central"}}
    
    if req.method == "tools/discover":
        return {"jsonrpc": "2.0", "id": req.id, "result": {"tools": TOOLS}}
    
    if req.method == "tools/execute":
        params = req.params or {}
        tool_name = params.get("name") or params.get("tool_name")
        tool_params = params.get("arguments", {}) or params.get("parameters", {})
        result = execute_tool(tool_name, tool_params)
        return {"jsonrpc": "2.0", "id": req.id, "result": {"result": result}}
    
    if req.method == "tools/schema":
        tool_name = req.params.get("tool_name") if req.params else None
        for tool in TOOLS:
            if tool["name"] == tool_name:
                return {"jsonrpc": "2.0", "id": req.id, "result": {"schema": tool["input_schema"]}}
        return {"jsonrpc": "2.0", "id": req.id, "error": {"code": -32602, "message": "Tool not found"}}
    
    return {"jsonrpc": "2.0", "id": req.id, "error": {"code": -32601, "message": "Method not found"}}


@app.get("/health")
def health():
    return {"status": "ok", "gateway": "central", "tools_count": len(TOOLS)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)

