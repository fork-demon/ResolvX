"""
Base tool classes and ToolProxy for the Golden Agent Framework.

Provides the foundation for tool execution with MCP gateway integration
and local fallback capabilities.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

from pydantic import BaseModel, Field

from core.config import Config
from core.exceptions import ToolError
from core.gateway.mcp_client import MCPClient, MCPToolProxy
from core.observability import get_logger, get_tracer


class ToolExecutionResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    execution_type: str = "unknown"  # local, mcp, fallback
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class BaseTool(ABC):
    """
    Base class for all tools in the framework.
    
    Provides common functionality for tool execution, validation,
    and error handling.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        **kwargs: Any
    ):
        """
        Initialize base tool.
        
        Args:
            name: Tool name
            description: Tool description
            capabilities: List of tool capabilities
            **kwargs: Additional tool-specific parameters
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.logger = get_logger(f"tools.{name}")
        self.tracer = get_tracer(f"tools.{name}")
        
        # Tool configuration
        self.config = kwargs.get("config", {})
        self.rate_limit = kwargs.get("rate_limit", {})
        self.security = kwargs.get("security", {})
        
        # Execution tracking
        self._execution_count = 0
        self._last_execution = None
        self._error_count = 0

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If execution fails
        """
        pass

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
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def get_capabilities(self) -> List[str]:
        """Get tool capabilities."""
        return self.capabilities.copy()

    def is_available(self) -> bool:
        """
        Check if tool is available for execution.
        
        Returns:
            True if available, False otherwise
        """
        return True

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on tool.
        
        Returns:
            Health check results
        """
        try:
            return {
                "status": "healthy",
                "tool_name": self.name,
                "execution_count": self._execution_count,
                "error_count": self._error_count,
                "last_execution": self._last_execution,
                "available": self.is_available(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "tool_name": self.name,
                "error": str(e),
            }

    def _record_execution(self, success: bool, execution_time: float) -> None:
        """Record execution statistics."""
        self._execution_count += 1
        self._last_execution = datetime.utcnow()
        
        if not success:
            self._error_count += 1


class ToolProxy:
    """
    Proxy for executing tools through MCP gateway with local fallbacks.
    
    Provides unified interface for tool execution that handles
    gateway routing, fallbacks, and error handling transparently.
    """

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        local_tools: Optional[Dict[str, BaseTool]] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize tool proxy.
        
        Args:
            mcp_client: MCP client for gateway tools
            local_tools: Dictionary of local tools
            config: Framework configuration
        """
        self.mcp_client = mcp_client
        self.local_tools = local_tools or {}
        self.config = config
        self.logger = get_logger("tools.proxy")
        self.tracer = get_tracer("tools.proxy")
        
        # MCP tool proxy
        self._mcp_proxy = MCPToolProxy(mcp_client) if mcp_client else None
        
        # Tool routing configuration
        self._tool_routing = self._load_tool_routing()
        
        # Execution statistics
        self._execution_stats: Dict[str, Dict[str, int]] = {}

    async def execute(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> ToolExecutionResult:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            **kwargs: Additional execution options
            
        Returns:
            Tool execution result
        """
        start_time = datetime.utcnow()
        parameters = parameters or {}
        
        try:
            with self.tracer.start_as_current_span(f"tool_execute_{tool_name}") as span:
                span.set_attribute("tool_name", tool_name)
                span.set_attribute("parameters", str(parameters))
                
                # Determine execution strategy
                execution_type = self._determine_execution_type(tool_name)
                span.set_attribute("execution_type", execution_type)
                
                # Execute tool based on strategy
                if execution_type == "mcp":
                    result = await self._execute_mcp_tool(tool_name, parameters, **kwargs)
                elif execution_type == "local":
                    result = await self._execute_local_tool(tool_name, parameters, **kwargs)
                else:
                    result = await self._execute_fallback_tool(tool_name, parameters, **kwargs)
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Record statistics
                self._record_execution_stats(tool_name, execution_type, True, execution_time)
                
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time_ms=execution_time,
                    execution_type=execution_type,
                    metadata={"parameters": parameters}
                )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record error statistics
            self._record_execution_stats(tool_name, "error", False, execution_time)
            
            self.logger.error(f"Tool execution failed for {tool_name}: {e}")
            
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                execution_type="error",
                metadata={"parameters": parameters}
            )

    async def _execute_mcp_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        """Execute tool via MCP gateway."""
        if not self._mcp_proxy:
            raise ToolError("MCP client not available")
        
        try:
            result = await self._mcp_proxy.execute(tool_name, **parameters)
            self.logger.debug(f"Executed MCP tool {tool_name} successfully")
            return result
            
        except Exception as e:
            self.logger.warning(f"MCP tool {tool_name} failed, trying fallback: {e}")
            # Try local fallback
            return await self._execute_local_tool(tool_name, parameters, **kwargs)

    async def _execute_local_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        """Execute local tool."""
        if tool_name not in self.local_tools:
            raise ToolError(f"Local tool {tool_name} not found")
        
        tool = self.local_tools[tool_name]
        
        # Validate parameters
        if not tool.validate_parameters(**parameters):
            raise ToolError(f"Invalid parameters for tool {tool_name}")
        
        # Execute tool
        result = await tool.execute(**parameters)
        self.logger.debug(f"Executed local tool {tool_name} successfully")
        return result

    async def _execute_fallback_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs: Any
    ) -> Any:
        """Execute fallback tool implementation."""
        # Create a simple fallback response
        fallback_response = {
            "status": "fallback",
            "tool": tool_name,
            "message": f"Tool {tool_name} executed using fallback implementation",
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.logger.warning(f"Using fallback implementation for tool {tool_name}")
        return fallback_response

    def _determine_execution_type(self, tool_name: str) -> str:
        """Determine how to execute a tool."""
        # Check tool routing configuration
        if tool_name in self._tool_routing:
            routing = self._tool_routing[tool_name]
            
            # Check if MCP is preferred and available
            if routing.get("prefer_mcp", True) and self._mcp_proxy:
                return "mcp"
            
            # Check if local is preferred and available
            if routing.get("prefer_local", False) and tool_name in self.local_tools:
                return "local"
        
        # Default strategy: try MCP first, then local, then fallback
        if self._mcp_proxy:
            return "mcp"
        elif tool_name in self.local_tools:
            return "local"
        else:
            return "fallback"

    def _load_tool_routing(self) -> Dict[str, Dict[str, Any]]:
        """Load tool routing configuration."""
        if not self.config or not self.config.gateway.tools:
            return {}
        
        routing = {}
        for tool_name, tool_config in self.config.gateway.tools.items():
            routing[tool_name] = {
                "prefer_mcp": tool_config.get("type") == "mcp",
                "prefer_local": tool_config.get("type") == "local",
                "fallback_mode": tool_config.get("fallback_mode", "local"),
            }
        
        return routing

    def _record_execution_stats(
        self,
        tool_name: str,
        execution_type: str,
        success: bool,
        execution_time: float
    ) -> None:
        """Record execution statistics."""
        if tool_name not in self._execution_stats:
            self._execution_stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "mcp_executions": 0,
                "local_executions": 0,
                "fallback_executions": 0,
                "total_execution_time": 0.0,
            }
        
        stats = self._execution_stats[tool_name]
        stats["total_executions"] += 1
        stats["total_execution_time"] += execution_time
        
        if success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
        
        if execution_type == "mcp":
            stats["mcp_executions"] += 1
        elif execution_type == "local":
            stats["local_executions"] += 1
        elif execution_type == "fallback":
            stats["fallback_executions"] += 1

    def register_local_tool(self, tool: BaseTool) -> None:
        """Register a local tool."""
        self.local_tools[tool.name] = tool
        self.logger.info(f"Registered local tool: {tool.name}")

    def unregister_local_tool(self, tool_name: str) -> bool:
        """Unregister a local tool."""
        if tool_name in self.local_tools:
            del self.local_tools[tool_name]
            self.logger.info(f"Unregistered local tool: {tool_name}")
            return True
        return False

    def list_available_tools(self) -> List[str]:
        """List all available tools."""
        tools = set()
        
        # Add local tools
        tools.update(self.local_tools.keys())
        
        # Add MCP tools if available
        if self._mcp_proxy:
            try:
                # This would need to be async in real implementation
                pass
            except Exception:
                pass
        
        return list(tools)

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tool."""
        if tool_name in self.local_tools:
            tool = self.local_tools[tool_name]
            return {
                "name": tool.name,
                "description": tool.description,
                "capabilities": tool.capabilities,
                "type": "local",
                "schema": tool.get_schema(),
            }
        
        # Check MCP tools
        if self._mcp_proxy:
            try:
                # This would need to be async in real implementation
                pass
            except Exception:
                pass
        
        return None

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on tool proxy."""
        try:
            # Check MCP client health
            mcp_healthy = True
            if self._mcp_proxy:
                try:
                    mcp_health = await self.mcp_client.health_check()
                    mcp_healthy = mcp_health.get("status") == "healthy"
                except Exception:
                    mcp_healthy = False
            
            # Check local tools health
            local_tools_healthy = True
            for tool in self.local_tools.values():
                try:
                    health = await tool.health_check()
                    if health.get("status") != "healthy":
                        local_tools_healthy = False
                        break
                except Exception:
                    local_tools_healthy = False
                    break
            
            return {
                "status": "healthy" if (mcp_healthy or local_tools_healthy) else "degraded",
                "mcp_healthy": mcp_healthy,
                "local_tools_healthy": local_tools_healthy,
                "total_tools": len(self.local_tools),
                "execution_stats": self._execution_stats,
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self._execution_stats.copy()
