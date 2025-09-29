"""
Central gateway client for MCP and LLM integration.

Provides unified client for communicating with central gateways
and managing tool execution through the MCP protocol.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from core.config import Config
from core.exceptions import GatewayError, ToolError
from core.gateway.mcp_client import MCPClient, MCPToolProxy
from core.gateway.llm_client import LLMGatewayClient, LLMProxy
from core.observability import get_logger, get_tracer


class CentralGatewayClient:
    """
    Unified client for central gateway integration.
    
    Provides access to both MCP and LLM gateways with
    automatic failover and load balancing.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize central gateway client.
        
        Args:
            config: Framework configuration
        """
        self.config = config
        self.logger = get_logger("gateway.central")
        self.tracer = get_tracer("gateway.central")
        
        # Gateway clients
        self.mcp_client: Optional[MCPClient] = None
        self.llm_client: Optional[LLMGatewayClient] = None
        
        # Proxies
        self.mcp_proxy: Optional[MCPToolProxy] = None
        self.llm_proxy: Optional[LLMProxy] = None
        
        # Connection status
        self._mcp_connected = False
        self._llm_connected = False
        self._last_health_check = None

    async def initialize(self) -> None:
        """Initialize the central gateway client."""
        try:
            # Initialize MCP client
            if self.config and self.config.gateway.mcp_gateway:
                self.mcp_client = MCPClient(self.config)
                await self.mcp_client.initialize()
                self.mcp_proxy = MCPToolProxy(self.mcp_client)
                self._mcp_connected = True
                self.logger.info("MCP gateway client initialized")
            
            # Initialize LLM client
            if self.config and self.config.gateway.llm_gateway:
                self.llm_client = LLMGatewayClient(self.config)
                await self.llm_client.initialize()
                self.llm_proxy = LLMProxy(self.llm_client)
                self._llm_connected = True
                self.logger.info("LLM gateway client initialized")
            
            self.logger.info("Central gateway client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize central gateway client: {e}")
            raise GatewayError(f"Central gateway initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the central gateway client."""
        try:
            # Close MCP client
            if self.mcp_client:
                await self.mcp_client.close()
                self.mcp_client = None
                self.mcp_proxy = None
                self._mcp_connected = False
            
            # Close LLM client
            if self.llm_client:
                await self.llm_client.close()
                self.llm_client = None
                self.llm_proxy = None
                self._llm_connected = False
            
            self.logger.info("Central gateway client closed")
            
        except Exception as e:
            self.logger.error(f"Error closing central gateway client: {e}")

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute a tool through the central gateway.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            **kwargs: Additional execution options
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If tool execution fails
        """
        if not self.mcp_proxy:
            raise ToolError("MCP gateway not available")
        
        try:
            with self.tracer.start_as_current_span(f"central_tool_{tool_name}") as span:
                span.set_attribute("tool_name", tool_name)
                span.set_attribute("gateway", "mcp")
                
                result = await self.mcp_proxy.execute(tool_name, **parameters)
                
                self.logger.debug(f"Executed tool {tool_name} via central gateway")
                return result
                
        except Exception as e:
            self.logger.error(f"Tool execution failed for {tool_name}: {e}")
            raise ToolError(f"Tool {tool_name} execution failed: {e}") from e

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using the LLM gateway.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Generated text
            
        Raises:
            GatewayError: If generation fails
        """
        if not self.llm_proxy:
            raise GatewayError("LLM gateway not available")
        
        try:
            with self.tracer.start_as_current_span("central_llm_generate") as span:
                span.set_attribute("model", model or "default")
                span.set_attribute("gateway", "llm")
                
                result = await self.llm_proxy.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    **kwargs
                )
                
                self.logger.debug("Generated text via central LLM gateway")
                return result
                
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise GatewayError(f"Text generation failed: {e}") from e

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Perform chat completion using the LLM gateway.
        
        Args:
            messages: List of chat messages
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Assistant response
            
        Raises:
            GatewayError: If completion fails
        """
        if not self.llm_proxy:
            raise GatewayError("LLM gateway not available")
        
        try:
            with self.tracer.start_as_current_span("central_llm_chat") as span:
                span.set_attribute("model", model or "default")
                span.set_attribute("message_count", len(messages))
                span.set_attribute("gateway", "llm")
                
                result = await self.llm_proxy.chat(
                    messages=messages,
                    model=model,
                    **kwargs
                )
                
                self.logger.debug("Chat completion via central LLM gateway")
                return result
                
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise GatewayError(f"Chat completion failed: {e}") from e

    async def discover_tools(self) -> List[str]:
        """
        Discover available tools from the MCP gateway.
        
        Returns:
            List of available tool names
        """
        if not self.mcp_client:
            return []
        
        try:
            tools = await self.mcp_client.discover_tools()
            tool_names = [tool.name for tool in tools]
            
            self.logger.info(f"Discovered {len(tool_names)} tools from MCP gateway")
            return tool_names
            
        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")
            return []

    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema or None if not found
        """
        if not self.mcp_client:
            return None
        
        try:
            return await self.mcp_client.get_tool_schema(tool_name)
        except Exception as e:
            self.logger.error(f"Failed to get schema for tool {tool_name}: {e}")
            return None

    async def list_models(self) -> List[str]:
        """
        List available models from the LLM gateway.
        
        Returns:
            List of available model names
        """
        if not self.llm_client:
            return []
        
        try:
            models = await self.llm_client.list_models()
            self.logger.info(f"Discovered {len(models)} models from LLM gateway")
            return models
            
        except Exception as e:
            self.logger.error(f"Model discovery failed: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on central gateway.
        
        Returns:
            Health check results
        """
        try:
            current_time = datetime.utcnow()
            
            # Check MCP gateway health
            mcp_health = {"status": "unavailable"}
            if self.mcp_client:
                try:
                    mcp_health = await self.mcp_client.health_check()
                    self._mcp_connected = mcp_health.get("status") == "healthy"
                except Exception as e:
                    mcp_health = {"status": "unhealthy", "error": str(e)}
                    self._mcp_connected = False
            
            # Check LLM gateway health
            llm_health = {"status": "unavailable"}
            if self.llm_client:
                try:
                    llm_health = await self.llm_client.health_check()
                    self._llm_connected = llm_health.get("status") == "healthy"
                except Exception as e:
                    llm_health = {"status": "unhealthy", "error": str(e)}
                    self._llm_connected = False
            
            # Determine overall status
            overall_status = "healthy"
            if not self._mcp_connected and not self._llm_connected:
                overall_status = "unhealthy"
            elif not self._mcp_connected or not self._llm_connected:
                overall_status = "degraded"
            
            self._last_health_check = current_time
            
            return {
                "status": overall_status,
                "mcp_gateway": mcp_health,
                "llm_gateway": llm_health,
                "mcp_connected": self._mcp_connected,
                "llm_connected": self._llm_connected,
                "last_check": current_time.isoformat(),
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "mcp_connected": False,
                "llm_connected": False,
            }

    def is_mcp_available(self) -> bool:
        """Check if MCP gateway is available."""
        return self._mcp_connected and self.mcp_proxy is not None

    def is_llm_available(self) -> bool:
        """Check if LLM gateway is available."""
        return self._llm_connected and self.llm_proxy is not None

    async def reconnect(self) -> None:
        """Attempt to reconnect to gateways."""
        try:
            self.logger.info("Attempting to reconnect to gateways...")
            
            # Reinitialize clients
            await self.initialize()
            
            self.logger.info("Gateway reconnection completed")
            
        except Exception as e:
            self.logger.error(f"Gateway reconnection failed: {e}")
            raise GatewayError(f"Gateway reconnection failed: {e}") from e
