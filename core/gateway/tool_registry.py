"""
Tool registry for managing local and remote tools.

Provides centralized registration and discovery of tools available to agents,
including both MCP-based remote tools and local custom tools.
"""

import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from core.config import Config
from core.exceptions import ToolError
from core.gateway.mcp_client import MCPClient, MCPToolProxy
from core.observability import get_logger


class ToolDefinition(BaseModel):
    """Tool definition model."""

    name: str
    description: str
    tool_type: str  # local, mcp, builtin
    module: Optional[str] = None
    server: Optional[str] = None
    capabilities: List[str] = []
    input_schema: Optional[Dict[str, Any]] = None
    rate_limit: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None


class BaseTool:
    """
    Base class for local tools.

    All local tools should inherit from this class and implement
    the execute method.
    """

    name: str = ""
    description: str = ""
    capabilities: List[str] = []

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            Tool execution result

        Raises:
            ToolError: If execution fails
        """
        raise NotImplementedError("Tool must implement execute method")

    def validate_parameters(self, **kwargs: Any) -> bool:
        """
        Validate tool parameters.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool input schema.

        Returns:
            JSON schema for tool parameters
        """
        return {}


class ToolRegistry:
    """
    Registry for managing all available tools.

    Provides unified access to both local and remote tools with
    discovery, validation, and execution capabilities.
    """

    def __init__(self, config: Optional[Any] = None, mcp_client: Optional[MCPClient] = None):
        """
        Initialize tool registry.

        Args:
            config: Framework configuration
            mcp_client: Optional MCP client for remote tools
        """
        self.config = config
        self.mcp_client = mcp_client
        self.logger = get_logger("tool_registry")

        # Tool storage
        self._local_tools: Dict[str, BaseTool] = {}
        self._mcp_tools: Dict[str, ToolDefinition] = {}
        self._tool_definitions: Dict[str, ToolDefinition] = {}

        # Tool proxy for MCP tools
        self._mcp_proxy: Optional[MCPToolProxy] = None
        if mcp_client:
            self._mcp_proxy = MCPToolProxy(mcp_client)

    async def initialize(self) -> None:
        """Initialize the tool registry."""
        try:
            # Load local tools from configuration
            await self._load_local_tools()

            # Discover MCP tools
            if self.mcp_client:
                await self._discover_mcp_tools()

            self.logger.info(
                f"Tool registry initialized with {len(self._local_tools)} local tools "
                f"and {len(self._mcp_tools)} MCP tools"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize tool registry: {e}")
            raise ToolError(f"Tool registry initialization failed: {e}") from e

    async def register_tool(
        self, tool: Union[BaseTool, ToolDefinition], tool_type: str = "local"
    ) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance or definition
            tool_type: Type of tool (local, mcp)

        Raises:
            ToolError: If registration fails
        """
        try:
            if isinstance(tool, BaseTool):
                # Register local tool
                tool_name = tool.name or tool.__class__.__name__.lower()
                self._local_tools[tool_name] = tool

                # Create tool definition
                definition = ToolDefinition(
                    name=tool_name,
                    description=tool.description,
                    tool_type="local",
                    capabilities=tool.capabilities,
                    input_schema=tool.get_schema(),
                )
                self._tool_definitions[tool_name] = definition

                self.logger.debug(f"Registered local tool: {tool_name}")

            elif isinstance(tool, ToolDefinition):
                # Register tool definition
                self._tool_definitions[tool.name] = tool

                if tool.tool_type == "mcp":
                    self._mcp_tools[tool.name] = tool

                self.logger.debug(f"Registered {tool.tool_type} tool: {tool.name}")

        except Exception as e:
            tool_name = getattr(tool, "name", str(tool))
            self.logger.error(f"Failed to register tool {tool_name}: {e}")
            raise ToolError(f"Tool registration failed: {e}") from e

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            **kwargs: Additional execution options

        Returns:
            Tool execution result

        Raises:
            ToolError: If tool not found or execution fails
        """
        if tool_name not in self._tool_definitions:
            raise ToolError(f"Tool {tool_name} not found in registry")

        tool_def = self._tool_definitions[tool_name]

        try:
            if tool_def.tool_type == "local":
                return await self._execute_local_tool(tool_name, parameters)

            elif tool_def.tool_type == "mcp":
                return await self._execute_mcp_tool(tool_name, parameters)

            else:
                raise ToolError(f"Unknown tool type: {tool_def.tool_type}")

        except Exception as e:
            self.logger.error(f"Tool execution failed for {tool_name}: {e}")
            raise ToolError(f"Tool {tool_name} execution failed: {e}") from e

    # Backward compatibility alias
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return await self.execute_tool(tool_name, parameters, **kwargs)

    async def _execute_local_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a local tool."""
        if tool_name not in self._local_tools:
            raise ToolError(f"Local tool {tool_name} not found")

        tool = self._local_tools[tool_name]

        # Validate parameters
        if not tool.validate_parameters(**parameters):
            raise ToolError(f"Invalid parameters for tool {tool_name}")

        # Execute tool
        result = await tool.execute(**parameters)

        return {
            "tool": tool_name,
            "result": result,
            "execution_type": "local",
        }

    async def _execute_mcp_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an MCP tool."""
        if not self._mcp_proxy:
            raise ToolError("MCP client not available")

        result = await self._mcp_proxy.execute(tool_name, **parameters)

        return {
            "tool": tool_name,
            "result": result,
            "execution_type": "mcp",
        }

    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema or None if not found
        """
        if tool_name in self._tool_definitions:
            tool_def = self._tool_definitions[tool_name]

            if tool_def.tool_type == "local" and tool_name in self._local_tools:
                return self._local_tools[tool_name].get_schema()

            elif tool_def.tool_type == "mcp" and self._mcp_proxy:
                return await self._mcp_proxy.get_tool_schema(tool_name)

            return tool_def.input_schema

        return None

    def list_tools(self, tool_type: Optional[str] = None) -> List[str]:
        """
        List available tools.

        Args:
            tool_type: Optional filter by tool type

        Returns:
            List of tool names
        """
        if tool_type:
            return [
                name
                for name, definition in self._tool_definitions.items()
                if definition.tool_type == tool_type
            ]

        return list(self._tool_definitions.keys())

    def get_tool_info(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get information about a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool definition or None if not found
        """
        return self._tool_definitions.get(tool_name)

    def search_tools(self, query: str, capabilities: Optional[List[str]] = None) -> List[str]:
        """
        Search for tools by name, description, or capabilities.

        Args:
            query: Search query
            capabilities: Optional capability filter

        Returns:
            List of matching tool names
        """
        matches = []
        query_lower = query.lower()

        for name, definition in self._tool_definitions.items():
            # Check name and description
            if (
                query_lower in name.lower()
                or query_lower in definition.description.lower()
            ):
                matches.append(name)
                continue

            # Check capabilities
            if capabilities:
                if any(cap in definition.capabilities for cap in capabilities):
                    matches.append(name)

        return matches

    async def _load_local_tools(self) -> None:
        """Load local tools from configuration."""
        tools_cfg: Optional[Dict[str, Any]] = None
        if not self.config:
            return
        # Support both full Config (with .gateway.tools) and GatewayConfig (with .tools)
        if hasattr(self.config, 'tools') and isinstance(self.config.tools, dict):
            tools_cfg = self.config.tools
        elif hasattr(self.config, 'gateway') and getattr(self.config.gateway, 'tools', None):
            tools_cfg = self.config.gateway.tools
        else:
            return

        for tool_name, tool_config in tools_cfg.items():
            if tool_config.get("type") == "local":
                try:
                    await self._load_local_tool(tool_name, tool_config)
                except Exception as e:
                    self.logger.error(f"Failed to load local tool {tool_name}: {e}")

    async def _load_local_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> None:
        """Load a single local tool."""
        module_path = tool_config.get("module")
        if not module_path:
            self.logger.warning(f"No module specified for local tool {tool_name}")
            return

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Find tool class
            tool_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseTool) and obj != BaseTool:
                    tool_class = obj
                    break

            if not tool_class:
                raise ToolError(f"No BaseTool subclass found in module {module_path}")

            # Create tool instance
            tool_instance = tool_class()

            # Override name if specified in config
            if not tool_instance.name:
                tool_instance.name = tool_name

            # Register the tool
            await self.register_tool(tool_instance)

        except Exception as e:
            raise ToolError(f"Failed to load tool from {module_path}: {e}") from e

    async def _discover_mcp_tools(self) -> None:
        """Discover tools from MCP gateway."""
        if not self.mcp_client:
            return

        try:
            mcp_tools = await self.mcp_client.discover_tools()

            for mcp_tool in mcp_tools:
                tool_definition = ToolDefinition(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    tool_type="mcp",
                    server=mcp_tool.server,
                    capabilities=mcp_tool.capabilities,
                    input_schema=mcp_tool.input_schema,
                )

                await self.register_tool(tool_definition)

        except Exception as e:
            self.logger.error(f"Failed to discover MCP tools: {e}")

    async def refresh_mcp_tools(self) -> None:
        """Refresh MCP tools from gateway."""
        if self.mcp_client:
            # Clear existing MCP tools
            self._mcp_tools.clear()

            # Remove MCP tools from definitions
            mcp_tool_names = [
                name
                for name, definition in self._tool_definitions.items()
                if definition.tool_type == "mcp"
            ]

            for tool_name in mcp_tool_names:
                del self._tool_definitions[tool_name]

            # Rediscover MCP tools
            await self._discover_mcp_tools()

            self.logger.info(f"Refreshed {len(self._mcp_tools)} MCP tools")

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name not in self._tool_definitions:
            return False

        tool_def = self._tool_definitions[tool_name]

        # Remove from appropriate storage
        if tool_def.tool_type == "local" and tool_name in self._local_tools:
            del self._local_tools[tool_name]

        if tool_def.tool_type == "mcp" and tool_name in self._mcp_tools:
            del self._mcp_tools[tool_name]

        # Remove from definitions
        del self._tool_definitions[tool_name]

        self.logger.debug(f"Unregistered tool: {tool_name}")
        return True

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on tool registry.

        Returns:
            Health check results
        """
        total_tools = len(self._tool_definitions)
        local_tools = len(self._local_tools)
        mcp_tools = len(self._mcp_tools)

        # Check MCP connectivity
        mcp_healthy = True
        if self.mcp_client:
            mcp_health = await self.mcp_client.health_check()
            mcp_healthy = mcp_health.get("status") == "healthy"

        return {
            "status": "healthy" if mcp_healthy else "degraded",
            "total_tools": total_tools,
            "local_tools": local_tools,
            "mcp_tools": mcp_tools,
            "mcp_gateway_healthy": mcp_healthy,
        }