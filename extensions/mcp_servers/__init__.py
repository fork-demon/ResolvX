"""
MCP Servers for Golden Agent Framework.

Provides pluggable MCP servers for different teams and functions.
Each server can be independently deployed and managed.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class MCPServerBase(ABC):
    """Base class for all MCP servers."""

    def __init__(self, team: str, port: int):
        """
        Initialize MCP server.
        
        Args:
            team: Team name (engineering, devops, security, support)
            port: Server port
        """
        self.team = team
        self.port = port
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.health_status = "healthy"

    @abstractmethod
    async def discover_tools(self) -> Dict[str, Any]:
        """Discover available tools."""
        pass

    @abstractmethod
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass


class MCPServerRegistry:
    """Registry for managing MCP servers."""

    def __init__(self):
        """Initialize server registry."""
        self.servers: Dict[str, MCPServerBase] = {}

    def register_server(self, team: str, server: MCPServerBase):
        """Register an MCP server."""
        self.servers[team] = server

    def get_server(self, team: str) -> Optional[MCPServerBase]:
        """Get MCP server by team."""
        return self.servers.get(team)

    def list_servers(self) -> Dict[str, str]:
        """List all registered servers."""
        return {team: server.team for team, server in self.servers.items()}

    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all tools from all servers."""
        all_tools = {}
        for team, server in self.servers.items():
            all_tools.update(server.tools)
        return all_tools


# Global registry instance
server_registry = MCPServerRegistry()


def register_mcp_server(team: str, server: MCPServerBase):
    """Register an MCP server."""
    server_registry.register_server(team, server)


def get_mcp_server(team: str) -> Optional[MCPServerBase]:
    """Get MCP server by team."""
    return server_registry.get_server(team)


def list_mcp_servers() -> Dict[str, str]:
    """List all registered MCP servers."""
    return server_registry.list_servers()


def get_all_mcp_tools() -> Dict[str, Dict[str, Any]]:
    """Get all tools from all MCP servers."""
    return server_registry.get_all_tools()
