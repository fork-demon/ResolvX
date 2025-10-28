"""
Central MCP Gateway - Zendesk, Splunk, New Relic, Memory tools
Port: 8081
"""
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import json
import os
from typing import Any, Dict, List
from datetime import datetime


app = Server("central-gateway")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools in the central gateway."""
    return [
        Tool(
            name="poll_queue",
            description="Poll Zendesk support queue for new tickets. Returns list of tickets from specified queue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "queue_name": {
                        "type": "string",
                        "description": "Queue name (e.g., 'pricing', 'urgent', 'l2_support')",
                        "default": "pricing"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tickets to return",
                        "default": 10
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by ticket status (new, open, pending, solved)",
                        "enum": ["new", "open", "pending", "solved"]
                    }
                },
                "required": ["queue_name"]
            }
        ),
        Tool(
            name="get_queue_stats",
            description="Get statistics for Zendesk support queues. Returns counts by status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "queue_name": {
                        "type": "string",
                        "description": "Queue name",
                        "default": "pricing"
                    }
                }
            }
        ),
        Tool(
            name="splunk_search",
            description="Search Splunk logs for errors, warnings, or patterns. Use for diagnostic investigation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Splunk SPL query (e.g., 'index=pricing error | stats count by service')"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range (e.g., '-1h', '-24h', '-7d')",
                        "default": "-1h"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="newrelic_query",
            description="Query New Relic for metrics, traces, or errors. Use for performance analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "nrql": {
                        "type": "string",
                        "description": "NRQL query (e.g., 'SELECT average(duration) FROM Transaction WHERE appName = \"PriceAPI\"')"
                    },
                    "account_id": {
                        "type": "string",
                        "description": "New Relic account ID",
                        "default": "1234567"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range in minutes",
                        "default": "60"
                    }
                },
                "required": ["nrql"]
            }
        ),
        Tool(
            name="memory_search",
            description="Search historical tickets for similar issues or resolutions. Use for duplicate detection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (ticket description or keywords)"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Similarity threshold (0.0-1.0)",
                        "default": 0.9
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool and return results."""
    
    if name == "poll_queue":
        queue_name = arguments.get("queue_name", "pricing")
        limit = arguments.get("limit", 10)
        status = arguments.get("status")
        
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
        
        result = {"success": True, "tickets": tickets, "count": len(tickets)}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "get_queue_stats":
        queue_name = arguments.get("queue_name", "pricing")
        result = {
            "success": True,
            "queue": queue_name,
            "stats": {
                "new": 5,
                "open": 12,
                "pending": 3,
                "solved": 45
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "splunk_search":
        query = arguments.get("query", "")
        time_range = arguments.get("time_range", "-1h")
        max_results = arguments.get("max_results", 100)
        
        # Mock Splunk results
        result = {
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "newrelic_query":
        nrql = arguments.get("nrql", "")
        account_id = arguments.get("account_id", "1234567")
        time_range = arguments.get("time_range", "60")
        
        # Mock New Relic results
        result = {
            "success": True,
            "nrql": nrql,
            "account_id": account_id,
            "time_range_minutes": time_range,
            "results": [
                {"timestamp": "2025-10-28T14:30:00Z", "appName": "PriceAPI", "average_duration": 245.5, "error_rate": 0.02},
                {"timestamp": "2025-10-28T14:25:00Z", "appName": "PriceAPI", "average_duration": 189.3, "error_rate": 0.01},
                {"timestamp": "2025-10-28T14:20:00Z", "appName": "PriceAPI", "average_duration": 512.8, "error_rate": 0.05}
            ],
            "summary": {
                "avg_duration_ms": 315.9,
                "avg_error_rate": 0.027,
                "samples": 3
            }
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "memory_search":
        query = arguments.get("query", "")
        threshold = arguments.get("similarity_threshold", 0.9)
        max_results = arguments.get("max_results", 5)
        
        # Mock historical ticket search
        result = {
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server using stdio transport."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

