from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os
import json
import requests


class MCPRequest(BaseModel):
    jsonrpc: str
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None


app = FastAPI(title="Mock MCP Gateway", version="0.1.0")


def price_api_base() -> str:
    return os.getenv("PRICE_API_BASE_URL", "http://price-api-mock:8080").rstrip("/")


def route_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    base = price_api_base()

    if tool_name == "price_minimum_calculate":
        url = f"{base}/pricelifecycle/v1/minimumPrices/calculate"
        resp = requests.post(url, json={
            "gtin": parameters.get("gtin"),
            "locationClusterIds": parameters.get("locationClusterIds", []),
            "qtyContents": {
                "totalQuantity": parameters.get("totalQuantity", 0),
                "quantityUom": parameters.get("quantityUom", "cl"),
            },
            "taxTypeCode": parameters.get("taxTypeCode"),
            "supplierABV": parameters.get("supplierABV"),
        }, timeout=10)
        return resp.json()

    if tool_name == "price_minimum_get":
        url = f"{base}/pricelifecycle/v1/minimumPrices"
        resp = requests.get(url, params={
            "locationClusterId": parameters.get("locationClusterId"),
            "gtin": parameters.get("gtin"),
            "effectiveDateTime": parameters.get("effectiveDateTime"),
        }, timeout=10)
        return resp.json()

    if tool_name == "basket_segment_get":
        url = f"{base}/pricelifecycle/v1/basketSegments"
        resp = requests.get(url, params={
            "tpnb": parameters.get("tpnb"),
            "locationClusterId": parameters.get("locationClusterId"),
            "subClass": parameters.get("subClass"),
        }, timeout=10)
        return resp.json()

    if tool_name == "competitor_prices_get":
        url = f"{base}/pricelifecycle/v1/competitorPrices"
        params: List[tuple] = [("tpnb", parameters.get("tpnb"))]
        for cid in parameters.get("locationClusterIds", []):
            params.append(("locationClusterId", cid))
        resp = requests.get(url, params=params, timeout=10)
        return resp.json()

    if tool_name == "competitor_promotional_prices_get":
        url = f"{base}/pricelifecycle/v1/advisory/promotions/competitor-promotional-prices"
        params: List[tuple] = [("mechanic", parameters.get("mechanic"))]
        for cid in parameters.get("locationClusterIds", []):
            params.append(("locationClusterIds", cid))
        for t in parameters.get("tpnbs", []):
            params.append(("tpnbs", t))
        resp = requests.get(url, params=params, timeout=10)
        return resp.json()

    if tool_name == "promo_effectiveness_get":
        url = f"{base}/pricelifecycle/v1/advisory/promotions/effectiveness"
        params: List[tuple] = [("mechanic", parameters.get("mechanic"))]
        for cid in parameters.get("locationClusterIds", []):
            params.append(("locationClusterIds", cid))
        for t in parameters.get("tpnbs", []):
            params.append(("tpnbs", t))
        resp = requests.get(url, params=params, timeout=10)
        return resp.json()

    if tool_name == "policies_view":
        url = f"{base}/pricelifecycle/v1/policies/view"
        resp = requests.post(url, json={
            "applicableEntities": parameters.get("applicableEntities", []),
            "classifications": parameters.get("classifications", []),
            "clusters": parameters.get("clusters", []),
        }, timeout=10)
        return resp.json()

    if tool_name == "base_prices_get":
        url = f"{base}/pricelifecycle/v5/basePrices"
        params = {
            "locationClusterId": parameters.get("locationClusterId"),
            "tpnb": parameters.get("tpnb"),
        }
        if parameters.get("effectiveDateTime"):
            params["effectiveDateTime"] = parameters["effectiveDateTime"]
        if parameters.get("teamnumber"):
            params["teamnumber"] = parameters["teamnumber"]
        resp = requests.get(url, params=params, timeout=10)
        return resp.json()

    # --- Minimal Zendesk mocks ---
    if tool_name == "poll_queue":
        q = parameters.get("queue_name", "default")
        limit = int(parameters.get("limit", 10) or 10)
        status = parameters.get("status", None)

        # Try to read tickets from filesystem
        base_dir = os.getenv("MOCK_TICKETS_DIR", "/data/mock_tickets").rstrip("/")
        tickets: List[Dict[str, Any]] = []

        # Search order: per-queue dir then base dir
        search_paths = [os.path.join(base_dir, q), base_dir]
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
                    "id": f"Z-{q}-{i+1}",
                    "subject": f"Stub ticket {i+1} in {q}",
                    "status": status or "new",
                    "priority": "normal",
                    "requester_id": f"U-{i+1}",
                    "assignee_id": None,
                    "created_at": "2025-10-21T00:00:00Z",
                    "updated_at": "2025-10-21T00:00:00Z",
                    "tags": ["stub", q],
                    "description": "This is a stubbed ticket for local testing."
                })

        return {"success": True, "tickets": tickets}

    if tool_name == "get_queue_stats":
        return {"success": True, "open": 7, "new": 3, "pending": 2}

    # --- Minimal Splunk/NewRelic mocks ---
    if tool_name == "splunk_search":
        return {"success": True, "query": parameters.get("query", ""), "results": [{"_time": "now", "count": 10}]}

    if tool_name == "newrelic_metrics":
        return {"success": True, "nrql": parameters.get("nrql", ""), "results": [{"timestamp": "now", "value": 123}]}

    return {"error": f"Unknown tool: {tool_name}"}


@app.post("/mcp")
def mcp_endpoint(req: MCPRequest):
    if req.method == "health":
        return {"jsonrpc": "2.0", "id": req.id, "result": {"status": "ok"}}

    if req.method == "tools/discover":
        tools = [
            {
                "name": "price_minimum_calculate",
                "description": "Calculate minimum price for GTIN across clusters",
                "input_schema": {},
                "server": "mock-mcp-gateway",
                "capabilities": [],
            },
            {
                "name": "price_minimum_get",
                "description": "Get minimum price for GTIN and cluster",
                "input_schema": {},
                "server": "mock-mcp-gateway",
                "capabilities": [],
            },
            {"name": "basket_segment_get", "description": "Get basket segment", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "competitor_prices_get", "description": "Get competitor prices", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "competitor_promotional_prices_get", "description": "Get competitor promo prices", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "promo_effectiveness_get", "description": "Get promotion effectiveness", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "policies_view", "description": "View policies", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "base_prices_get", "description": "Get base price", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "poll_queue", "description": "Poll Zendesk queue", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "get_queue_stats", "description": "Zendesk queue stats", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "splunk_search", "description": "Search Splunk logs", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
            {"name": "newrelic_metrics", "description": "Query New Relic metrics", "input_schema": {}, "server": "mock-mcp-gateway", "capabilities": []},
        ]
        return {"jsonrpc": "2.0", "id": req.id, "result": {"tools": tools}}

    if req.method == "tools/execute":
        params = req.params or {}
        tool_name = params.get("tool_name")
        tool_params = params.get("parameters", {})
        result = route_tool(tool_name, tool_params)
        return {"jsonrpc": "2.0", "id": req.id, "result": {"result": result}}

    if req.method == "tools/schema":
        return {"jsonrpc": "2.0", "id": req.id, "result": {"schema": {}}}

    return {"jsonrpc": "2.0", "id": req.id, "error": {"code": -32601, "message": "Method not found"}}


def _main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)


if __name__ == "__main__":
    _main()


