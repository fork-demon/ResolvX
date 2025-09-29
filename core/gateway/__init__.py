"""
Gateway integration components for MCP and LLM gateways.

This module provides integration with centralized gateways for:
- Model Context Protocol (MCP) tool access
- LLM gateway for centralized model routing
- Tool proxy pattern for unified tool execution
- Circuit breaker and fallback mechanisms
"""

from core.gateway.mcp_client import MCPClient, MCPToolProxy
from core.gateway.llm_client import LLMGatewayClient
from core.gateway.tool_registry import ToolRegistry
from core.gateway.circuit_breaker import CircuitBreaker

__all__ = [
    "MCPClient",
    "MCPToolProxy",
    "LLMGatewayClient",
    "ToolRegistry",
    "CircuitBreaker",
]