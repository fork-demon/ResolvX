"""
MCP (Model Context Protocol) client for gateway integration.

Provides client implementation for communicating with MCP gateways
and executing tools via the centralized protocol.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx
import websockets
from pydantic import BaseModel

from core.config import Config
from core.exceptions import GatewayError, ToolError
from core.gateway.circuit_breaker import get_circuit_breaker, with_circuit_breaker
from core.observability import get_logger, get_tracer


class MCPRequest(BaseModel):
    """MCP request model."""

    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP response model."""

    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class MCPTool(BaseModel):
    """MCP tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server: str
    capabilities: List[str] = []


class MCPClient:
    """
    Client for communicating with MCP gateways.

    Supports both HTTP and WebSocket connections for tool execution
    and service discovery.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize MCP client.

        Args:
            config: Framework configuration
        """
        self.config = config
        self.logger = get_logger("mcp.client")
        self.tracer = get_tracer("mcp.client")

        # Configuration
        self.gateway_url = self._get_gateway_url()
        self.timeout = self._get_timeout()
        self.retry_attempts = self._get_retry_attempts()
        self.fallback_mode = self._get_fallback_mode()

        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

        # WebSocket connection
        self._ws_connection: Optional[websockets.WebSocketServerProtocol] = None

        # Tool registry
        self._available_tools: Dict[str, MCPTool] = {}

        # Circuit breaker
        self._circuit_breaker = get_circuit_breaker(
            name="mcp_gateway",
            failure_threshold=5,
            success_threshold=3,
            timeout=60,
        )

    async def initialize(self) -> None:
        """Initialize the MCP client."""
        try:
            # Create HTTP client
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )

            # Discover available tools
            await self.discover_tools()

            self.logger.info(f"MCP client initialized with {len(self._available_tools)} tools")

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP client: {e}")
            raise GatewayError(f"MCP client initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the MCP client and cleanup resources."""
        try:
            # Close HTTP client
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

            # Close WebSocket connection
            if self._ws_connection:
                await self._ws_connection.close()
                self._ws_connection = None

            self.logger.info("MCP client closed")

        except Exception as e:
            self.logger.error(f"Error closing MCP client: {e}")

    async def discover_tools(self) -> List[MCPTool]:
        """
        Discover available tools from the MCP gateway.

        Returns:
            List of available tools
        """
        try:
            response = await self._make_request(
                method="tools/discover",
                params={},
            )

            tools = []
            for tool_data in response.get("tools", []):
                tool = MCPTool(**tool_data)
                tools.append(tool)
                self._available_tools[tool.name] = tool

            self.logger.info(f"Discovered {len(tools)} tools from MCP gateway")
            return tools

        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")
            if self.fallback_mode == "error":
                raise GatewayError(f"Tool discovery failed: {e}") from e
            return []

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool via MCP gateway.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            timeout: Optional timeout override

        Returns:
            Tool execution result

        Raises:
            ToolError: If tool execution fails
        """
        if tool_name not in self._available_tools:
            # Try to refresh tool registry
            await self.discover_tools()

            if tool_name not in self._available_tools:
                if self.fallback_mode == "local":
                    return await self._execute_local_fallback(tool_name, parameters)
                else:
                    raise ToolError(f"Tool {tool_name} not found in MCP gateway")

        try:
            with self.tracer.start_as_current_span(f"mcp_tool_{tool_name}") as span:
                # Set input data
                input_data = {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "gateway_url": self.gateway_url,
                    "timeout": timeout
                }
                span.set_input(input_data)
                span.set_attribute("tool_name", tool_name)
                span.set_attribute("gateway_url", self.gateway_url)

                # Execute tool via circuit breaker
                result = await with_circuit_breaker(
                    name="mcp_gateway",
                    func=self._execute_tool_request,
                    tool_name=tool_name,
                    parameters=parameters,
                    timeout=timeout,
                    fallback=self._execute_local_fallback if self.fallback_mode == "local" else None,
                )

                # Set output data
                output_data = {
                    "success": True,
                    "tool_name": tool_name,
                    "result": result
                }
                span.set_output(output_data)

                self.logger.debug(f"Tool {tool_name} executed successfully")
                return result

        except Exception as e:
            self.logger.error(f"Tool execution failed for {tool_name}: {e}")
            # Set error output
            if 'span' in locals():
                span.set_output({"success": False, "error": str(e), "tool_name": tool_name})
            raise ToolError(f"Tool {tool_name} execution failed: {e}") from e

    async def _execute_tool_request(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the actual tool request."""
        response = await self._make_request(
            method="tools/execute",
            params={
                "tool_name": tool_name,
                "parameters": parameters,
            },
            timeout=timeout,
        )

        if "error" in response:
            raise ToolError(f"Tool execution error: {response['error']}")

        return response.get("result", {})

    async def _execute_local_fallback(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute local fallback when gateway is unavailable."""
        self.logger.warning(f"Using local fallback for tool {tool_name}")

        # Simple mock response for fallback
        return {
            "status": "fallback",
            "tool": tool_name,
            "message": "Executed using local fallback - gateway unavailable",
            "parameters": parameters,
        }

    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema or None if not found
        """
        if tool_name in self._available_tools:
            return self._available_tools[tool_name].input_schema

        try:
            response = await self._make_request(
                method="tools/schema",
                params={"tool_name": tool_name},
            )
            return response.get("schema")

        except Exception as e:
            self.logger.error(f"Failed to get schema for tool {tool_name}: {e}")
            return None

    async def list_tools(self) -> List[str]:
        """
        List available tool names.

        Returns:
            List of tool names
        """
        return list(self._available_tools.keys())

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MCP gateway.

        Returns:
            Health check results
        """
        try:
            response = await self._make_request(
                method="health",
                params={},
                timeout=10,
            )

            return {
                "status": "healthy",
                "gateway_url": self.gateway_url,
                "response": response,
                "available_tools": len(self._available_tools),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "gateway_url": self.gateway_url,
                "error": str(e),
                "available_tools": len(self._available_tools),
            }

    async def _make_request(
        self,
        method: str,
        params: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make an MCP request to the gateway.

        Args:
            method: MCP method name
            params: Request parameters
            timeout: Optional timeout override

        Returns:
            Response data

        Raises:
            GatewayError: If request fails
        """
        if not self._http_client:
            raise GatewayError("MCP client not initialized")

        request = MCPRequest(
            id=f"req_{asyncio.current_task().get_name()}_{method}",
            method=method,
            params=params,
        )

        try:
            response = await self._http_client.post(
                urljoin(self.gateway_url, "/mcp"),
                json=request.dict(),
                timeout=timeout or self.timeout,
            )

            response.raise_for_status()

            mcp_response = MCPResponse(**response.json())

            if mcp_response.error:
                raise GatewayError(f"MCP error: {mcp_response.error}")

            return mcp_response.result or {}

        except httpx.HTTPError as e:
            raise GatewayError(f"HTTP error communicating with MCP gateway: {e}") from e
        except Exception as e:
            raise GatewayError(f"Error communicating with MCP gateway: {e}") from e

    def _get_gateway_url(self) -> str:
        """Get MCP gateway URL from configuration."""
        # First try the environment variable
        if self.config and hasattr(self.config, 'central_mcp_gateway_url'):
            return self.config.central_mcp_gateway_url
        
        # Fallback to gateway config
        if self.config and self.config.gateway.mcp_gateway:
            return self.config.gateway.mcp_gateway.get("url", "http://localhost:8081")
        return "http://localhost:8081"

    def _get_timeout(self) -> int:
        """Get timeout from configuration."""
        if self.config and self.config.gateway.mcp_gateway:
            return self.config.gateway.mcp_gateway.get("timeout", 30)
        return 30

    def _get_retry_attempts(self) -> int:
        """Get retry attempts from configuration."""
        if self.config and self.config.gateway.mcp_gateway:
            return self.config.gateway.mcp_gateway.get("retry_attempts", 3)
        return 3

    def _get_fallback_mode(self) -> str:
        """Get fallback mode from configuration."""
        if self.config and self.config.gateway.mcp_gateway:
            return self.config.gateway.mcp_gateway.get("fallback_mode", "local")
        return "local"


class MCPToolProxy:
    """
    Proxy for executing tools through MCP gateway.

    Provides a unified interface for tool execution that handles
    gateway routing, fallbacks, and error handling transparently.
    """

    def __init__(self, mcp_client: MCPClient):
        """
        Initialize tool proxy.

        Args:
            mcp_client: MCP client instance
        """
        self.mcp_client = mcp_client
        self.logger = get_logger("mcp.tool_proxy")

    async def execute(
        self,
        tool_name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters

        Returns:
            Tool execution result
        """
        try:
            result = await self.mcp_client.execute_tool(tool_name, kwargs)

            self.logger.debug(f"Tool {tool_name} executed via MCP gateway")
            return result

        except Exception as e:
            self.logger.error(f"Tool proxy execution failed for {tool_name}: {e}")
            raise

    async def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return await self.mcp_client.list_tools()

    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool."""
        return await self.mcp_client.get_tool_schema(tool_name)

    def __getattr__(self, tool_name: str) -> callable:
        """
        Dynamic tool access via attribute syntax.

        Allows calling tools like: proxy.splunk_search(query="error")
        """

        async def tool_executor(**kwargs: Any) -> Dict[str, Any]:
            return await self.execute(tool_name, **kwargs)

        return tool_executor