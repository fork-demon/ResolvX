"""
Base MCP Server implementation.

Provides common functionality for all MCP servers including
HTTP endpoints, tool discovery, and execution.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from aiohttp import web, ClientSession
from aiohttp.web import Request, Response

from . import MCPServerBase


class BaseMCPServer(MCPServerBase):
    """Base implementation for MCP servers."""

    def __init__(self, team: str, port: int):
        """
        Initialize base MCP server.
        
        Args:
            team: Team name
            port: Server port
        """
        super().__init__(team, port)
        self.logger = logging.getLogger(f"mcp.{team}")
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "tools_discovered": 0,
        }

    async def discover_tools(self) -> Dict[str, Any]:
        """Discover available tools."""
        self.metrics["requests_total"] += 1
        
        try:
            tools_list = []
            for tool_name, tool_info in self.tools.items():
                tools_list.append({
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters"],
                    "team": self.team,
                })
            
            self.metrics["tools_discovered"] = len(tools_list)
            self.metrics["requests_success"] += 1
            
            return {
                "tools": tools_list,
                "count": len(tools_list),
                "team": self.team,
            }
        except Exception as e:
            self.metrics["requests_error"] += 1
            self.logger.error(f"Tool discovery failed: {e}")
            raise

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool."""
        self.metrics["requests_total"] += 1
        
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            tool_info = self.tools[tool_name]
            result = await tool_info["handler"](parameters)
            
            self.metrics["requests_success"] += 1
            return {
                "tool_name": tool_name,
                "result": result,
                "timestamp": datetime.utcnow().isoformat(),
                "team": self.team,
            }
        except Exception as e:
            self.metrics["requests_error"] += 1
            self.logger.error(f"Tool execution failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": self.health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics,
            "team": self.team,
        }

    def register_tool(self, tool_name: str, description: str, parameters: Dict[str, Any], handler):
        """Register a tool with the server."""
        self.tools[tool_name] = {
            "name": tool_name,
            "description": description,
            "parameters": parameters,
            "handler": handler,
        }

    async def start_server(self):
        """Start the MCP server."""
        app = web.Application()
        
        # Add routes
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/tools", self._handle_tools_discover)
        app.router.add_post("/tools/execute", self._handle_tool_execute)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        
        self.logger.info(f"{self.team.title()} MCP Server running on port {self.port}")
        return runner

    async def _handle_health(self, request: Request) -> Response:
        """Handle health check requests."""
        try:
            health_data = await self.health_check()
            return web.json_response(health_data)
        except Exception as e:
            return web.json_response(
                {"error": f"Health check failed: {e}"},
                status=500
            )

    async def _handle_tools_discover(self, request: Request) -> Response:
        """Handle tool discovery requests."""
        try:
            tools_data = await self.discover_tools()
            return web.json_response(tools_data)
        except Exception as e:
            return web.json_response(
                {"error": f"Tool discovery failed: {e}"},
                status=500
            )

    async def _handle_tool_execute(self, request: Request) -> Response:
        """Handle tool execution requests."""
        try:
            data = await request.json()
            tool_name = data.get("tool_name")
            parameters = data.get("parameters", {})
            
            if not tool_name:
                return web.json_response(
                    {"error": "tool_name is required"},
                    status=400
                )
            
            result = await self.execute_tool(tool_name, parameters)
            return web.json_response(result)
        except Exception as e:
            return web.json_response(
                {"error": f"Tool execution failed: {e}"},
                status=500
            )
