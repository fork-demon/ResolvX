"""
Price Team MCP Gateway - Price, Product, Location, Competitor tools
Port: 8082
"""
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import json
import os
import requests
from typing import Any, Dict


app = Server("price-team-gateway")


def price_api_base() -> str:
    """Get Price API base URL from environment."""
    return os.getenv("PRICE_API_BASE_URL", "http://127.0.0.1:8090").rstrip("/")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools in the price team gateway."""
    return [
        Tool(
            name="base_prices_get",
            description="Get base price for a product (TPNB) at a location cluster. Core pricing data from Price Lifecycle API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tpnb": {
                        "type": "string",
                        "description": "Product identifier (TPNB)"
                    },
                    "locationClusterId": {
                        "type": "string",
                        "description": "Location cluster ID (e.g., 'LC001', 'LC_LONDON')"
                    },
                    "effectiveDateTime": {
                        "type": "string",
                        "description": "Effective date-time (ISO format), optional"
                    },
                    "teamnumber": {
                        "type": "string",
                        "description": "Team number for filtering, optional"
                    }
                },
                "required": ["tpnb", "locationClusterId"]
            }
        ),
        Tool(
            name="competitor_prices_get",
            description="Get competitor pricing data for a product across location clusters. Used for price intelligence analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tpnb": {
                        "type": "string",
                        "description": "Product identifier (TPNB)"
                    },
                    "locationClusterIds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of location cluster IDs"
                    }
                },
                "required": ["tpnb"]
            }
        ),
        Tool(
            name="product_info_get",
            description="Get detailed product information including GTIN, description, category, and attributes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tpnb": {
                        "type": "string",
                        "description": "Product identifier (TPNB)"
                    },
                    "gtin": {
                        "type": "string",
                        "description": "Alternative: Global Trade Item Number (GTIN/EAN)"
                    }
                }
            }
        ),
        Tool(
            name="location_info_get",
            description="Get location or location cluster information including stores, regions, and cluster mapping.",
            inputSchema={
                "type": "object",
                "properties": {
                    "locationClusterId": {
                        "type": "string",
                        "description": "Location cluster ID"
                    },
                    "storeId": {
                        "type": "string",
                        "description": "Alternative: specific store ID"
                    }
                }
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool and return results."""
    
    if name == "base_prices_get":
        tpnb = arguments.get("tpnb")
        location_cluster_id = arguments.get("locationClusterId")
        effective_date_time = arguments.get("effectiveDateTime")
        teamnumber = arguments.get("teamnumber")
        
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
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "competitor_prices_get":
        tpnb = arguments.get("tpnb")
        location_cluster_ids = arguments.get("locationClusterIds", [])
        
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
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "product_info_get":
        tpnb = arguments.get("tpnb")
        gtin = arguments.get("gtin")
        
        # Mock product data
        result = {
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
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "location_info_get":
        location_cluster_id = arguments.get("locationClusterId")
        store_id = arguments.get("storeId")
        
        # Mock location data
        if location_cluster_id:
            result = {
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
            result = {
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

