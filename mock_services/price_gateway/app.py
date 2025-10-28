"""
Price Team MCP Gateway HTTP Wrapper
Port: 8082
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os
import requests
import json

app = FastAPI(title="Price Team MCP Gateway", version="1.0.0")


class MCPRequest(BaseModel):
    jsonrpc: str
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None


def price_api_base() -> str:
    """Get Price API base URL from environment."""
    return os.getenv("PRICE_API_BASE_URL", "http://127.0.0.1:8090").rstrip("/")


# Tool definitions
TOOLS = [
    {
        "name": "base_prices_get",
        "description": "Get base price for a product (TPNB) at a location cluster. Core pricing data from Price Lifecycle API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tpnb": {"type": "string", "description": "Product identifier (TPNB)"},
                "locationClusterId": {"type": "string", "description": "Location cluster ID"},
                "effectiveDateTime": {"type": "string", "description": "Effective date-time (ISO format), optional"},
                "teamnumber": {"type": "string", "description": "Team number, optional"}
            },
            "required": ["tpnb", "locationClusterId"]
        },
        "server": "price-team-gateway",
        "capabilities": []
    },
    {
        "name": "competitor_prices_get",
        "description": "Get competitor pricing data for a product across location clusters. Used for price intelligence analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tpnb": {"type": "string", "description": "Product identifier (TPNB)"},
                "locationClusterIds": {"type": "array", "items": {"type": "string"}, "description": "List of location cluster IDs"}
            },
            "required": ["tpnb"]
        },
        "server": "price-team-gateway",
        "capabilities": []
    },
    {
        "name": "product_info_get",
        "description": "Get detailed product information including GTIN, description, category, and attributes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tpnb": {"type": "string", "description": "Product identifier (TPNB)"},
                "gtin": {"type": "string", "description": "Alternative: Global Trade Item Number (GTIN/EAN)"}
            }
        },
        "server": "price-team-gateway",
        "capabilities": []
    },
    {
        "name": "location_info_get",
        "description": "Get location or location cluster information including stores, regions, and cluster mapping.",
        "input_schema": {
            "type": "object",
            "properties": {
                "locationClusterId": {"type": "string", "description": "Location cluster ID"},
                "storeId": {"type": "string", "description": "Alternative: specific store ID"}
            }
        },
        "server": "price-team-gateway",
        "capabilities": []
    },
]


def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool and return results."""
    
    if tool_name == "base_prices_get":
        tpnb = parameters.get("tpnb")
        location_cluster_id = parameters.get("locationClusterId")
        effective_date_time = parameters.get("effectiveDateTime")
        teamnumber = parameters.get("teamnumber")
        
        try:
            base = price_api_base()
            url = f"{base}/pricelifecycle/v5/basePrices"
            params = {
                "tpnb": tpnb,
                "locationClusterId": location_cluster_id
            }
            if effective_date_time:
                params["effectiveDateTime"] = effective_date_time
            if teamnumber:
                params["teamnumber"] = teamnumber
            
            resp = requests.get(url, params=params, timeout=10)
            result = resp.json()
            
        except Exception as e:
            # Fallback mock data
            result = {
                "success": True,
                "tpnb": tpnb,
                "locationClusterId": location_cluster_id,
                "basePrice": 2.50,
                "currency": "GBP",
                "effectiveFrom": "2025-10-28T00:00:00Z",
                "priceType": "BASE",
                "source": "mock_fallback",
                "error": str(e) if os.getenv("DEBUG") else None
            }
        
        return result
    
    elif tool_name == "competitor_prices_get":
        tpnb = parameters.get("tpnb")
        location_cluster_ids = parameters.get("locationClusterIds", [])
        
        try:
            base = price_api_base()
            url = f"{base}/pricelifecycle/v1/competitorPrices"
            params = [("tpnb", tpnb)]
            for cid in location_cluster_ids:
                params.append(("locationClusterId", cid))
            
            resp = requests.get(url, params=params, timeout=10)
            result = resp.json()
            
        except Exception as e:
            # Fallback mock data
            result = {
                "success": True,
                "tpnb": tpnb,
                "locationClusterIds": location_cluster_ids,
                "competitors": [
                    {
                        "competitor": "ASDA",
                        "price": 2.35,
                        "currency": "GBP",
                        "locationClusterId": location_cluster_ids[0] if location_cluster_ids else "LC001",
                        "lastUpdated": "2025-10-28T12:00:00Z"
                    },
                    {
                        "competitor": "Sainsburys",
                        "price": 2.45,
                        "currency": "GBP",
                        "locationClusterId": location_cluster_ids[0] if location_cluster_ids else "LC001",
                        "lastUpdated": "2025-10-28T12:00:00Z"
                    }
                ],
                "source": "mock_fallback",
                "error": str(e) if os.getenv("DEBUG") else None
            }
        
        return result
    
    elif tool_name == "product_info_get":
        tpnb = parameters.get("tpnb")
        gtin = parameters.get("gtin")
        
        # Mock product data
        return {
            "success": True,
            "tpnb": tpnb or "12345678",
            "gtin": gtin or "5000112345678",
            "description": "Tesco Finest Organic Milk 2L",
            "brand": "Tesco Finest",
            "category": "Dairy",
            "subCategory": "Milk",
            "department": "Fresh Food",
            "unitOfMeasure": "litres",
            "packSize": "2.0",
            "attributes": {
                "organic": True,
                "freshness": "fresh",
                "refrigerated": True,
                "shelfLife": "7 days"
            },
            "status": "active",
            "source": "mock_product_master"
        }
    
    elif tool_name == "location_info_get":
        location_cluster_id = parameters.get("locationClusterId")
        store_id = parameters.get("storeId")
        
        # Mock location data
        if location_cluster_id:
            return {
                "success": True,
                "locationClusterId": location_cluster_id,
                "clusterName": f"Cluster {location_cluster_id}",
                "region": "South East",
                "storeCount": 45,
                "stores": [
                    {"storeId": "S001", "name": "Tesco Superstore London Bridge", "postcode": "SE1 9SG"},
                    {"storeId": "S002", "name": "Tesco Express Canary Wharf", "postcode": "E14 5AB"},
                    {"storeId": "S003", "name": "Tesco Extra Stratford", "postcode": "E15 1NG"}
                ],
                "clusterType": "urban",
                "pricingStrategy": "competitive",
                "source": "mock_location_master"
            }
        else:
            return {
                "success": True,
                "storeId": store_id or "S001",
                "name": "Tesco Superstore London Bridge",
                "postcode": "SE1 9SG",
                "region": "South East",
                "locationClusterId": "LC_LONDON_CENTRAL",
                "format": "Superstore",
                "size_sqft": 45000,
                "source": "mock_location_master"
            }
    
    else:
        return {"error": f"Unknown tool: {tool_name}"}


@app.post("/mcp")
def mcp_endpoint(req: MCPRequest):
    """MCP JSON-RPC endpoint."""
    
    if req.method == "health":
        return {"jsonrpc": "2.0", "id": req.id, "result": {"status": "ok", "gateway": "price-team"}}
    
    if req.method == "tools/discover":
        return {"jsonrpc": "2.0", "id": req.id, "result": {"tools": TOOLS}}
    
    if req.method == "tools/execute":
        params = req.params or {}
        tool_name = params.get("tool_name")
        tool_params = params.get("parameters", {})
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
    return {"status": "ok", "gateway": "price-team", "tools_count": len(TOOLS)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)

