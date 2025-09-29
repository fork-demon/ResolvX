"""
Tool system for the Golden Agent Framework.

Provides centralized tool management with MCP gateway integration,
local tool fallbacks, and unified tool execution interface.
"""

from core.tools.base import BaseTool, ToolProxy
from core.tools.central_gateway import CentralGatewayClient
from core.tools.std_tools import (
    SplunkTool,
    ZendeskTool, 
    NewRelicTool,
    VaultTool,
    MetricsTool
)

__all__ = [
    "BaseTool",
    "ToolProxy",
    "CentralGatewayClient",
    "SplunkTool",
    "ZendeskTool",
    "NewRelicTool", 
    "VaultTool",
    "MetricsTool",
]
