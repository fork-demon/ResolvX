"""
LLM Gateway client for centralized model access.

Provides client implementation for communicating with LLM gateways
for centralized model routing, load balancing, and monitoring.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel

from core.config import Config
from core.exceptions import GatewayError
from core.gateway.circuit_breaker import get_circuit_breaker, with_circuit_breaker
from core.observability import get_logger, get_tracer


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""

    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None


class LLMGatewayClient:
    """
    Client for communicating with LLM gateways.

    Provides centralized access to language models with load balancing,
    fallback mechanisms, and comprehensive monitoring.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LLM gateway client.

        Args:
            config: Framework configuration
        """
        self.config = config
        self.logger = get_logger("llm_gateway.client")
        self.tracer = get_tracer("llm_gateway.client")

        # Configuration
        self.gateway_url = self._get_gateway_url()
        self.default_model = self._get_default_model()
        self.timeout = self._get_timeout()
        self.retry_attempts = self._get_retry_attempts()

        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

        # Available models
        self._available_models: Dict[str, Dict[str, Any]] = {}

        # Circuit breaker
        self._circuit_breaker = get_circuit_breaker(
            name="llm_gateway",
            failure_threshold=3,
            success_threshold=2,
            timeout=30,
        )

    async def initialize(self) -> None:
        """Initialize the LLM gateway client."""
        try:
            # Create HTTP client with authentication
            headers = {}
            auth_config = self._get_auth_config()
            if auth_config.get("type") == "bearer" and auth_config.get("token"):
                headers["Authorization"] = f"Bearer {auth_config['token']}"

            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
                follow_redirects=True,
            )

            # Discover available models
            await self.discover_models()

            self.logger.info(f"LLM gateway client initialized with {len(self._available_models)} models")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM gateway client: {e}")
            raise GatewayError(f"LLM gateway client initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the LLM gateway client and cleanup resources."""
        try:
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None

            self.logger.info("LLM gateway client closed")

        except Exception as e:
            self.logger.error(f"Error closing LLM gateway client: {e}")

    async def discover_models(self) -> List[str]:
        """
        Discover available models from the LLM gateway.

        Returns:
            List of available model names
        """
        try:
            response = await self._make_request(
                method="GET",
                endpoint="/v1/models",
            )

            models = []
            for model_data in response.get("data", []):
                model_id = model_data.get("id")
                if model_id:
                    models.append(model_id)
                    self._available_models[model_id] = model_data

            self.logger.info(f"Discovered {len(models)} models from LLM gateway")
            return models

        except Exception as e:
            self.logger.error(f"Model discovery failed: {e}")
            return []

    async def chat_completion(
        self,
        messages: List[Union[ChatMessage, Dict[str, Any]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ChatCompletionResponse, AsyncIterator[Dict[str, Any]]]:
        """
        Create a chat completion.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to configured default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            Chat completion response or stream iterator

        Raises:
            GatewayError: If completion fails
        """
        # Normalize messages
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized_messages.append(ChatMessage(**msg))
            else:
                normalized_messages.append(msg)

        # Create request
        request = ChatCompletionRequest(
            model=model or self.default_model,
            messages=normalized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        try:
            with self.tracer.start_as_current_span("llm_chat_completion") as span:
                # Set input data
                input_data = {
                    "model": request.model,
                    "message_count": len(messages),
                    "messages": [{"role": msg.role, "content": msg.content} for msg in normalized_messages],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": stream
                }
                span.set_input(input_data)
                span.set_attribute("model", request.model)
                span.set_attribute("message_count", len(messages))
                span.set_attribute("stream", stream)

                if stream:
                    result = await self._stream_completion(request)
                else:
                    result = await self._complete_completion(request)

                # Set output data
                if isinstance(result, ChatCompletionResponse):
                    # Extract the actual content
                    content = ""
                    if result.choices:
                        first_choice = result.choices[0]
                        if isinstance(first_choice, dict):
                            content = first_choice.get("message", {}).get("content", "")
                        else:
                            content = getattr(first_choice.message, "content", "") if hasattr(first_choice, "message") else ""
                    
                    output_data = {
                        "success": True,
                        "model": request.model,
                        "content": content,
                        "response_length": len(content),
                        "usage": result.usage,
                        "finish_reason": result.choices[0].get("finish_reason") if result.choices else None
                    }
                else:
                    output_data = {
                        "success": True,
                        "model": request.model,
                        "stream": True
                    }
                span.set_output(output_data)

                return result

        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            # Set error output
            if 'span' in locals():
                span.set_output({"success": False, "error": str(e), "model": request.model})
            raise GatewayError(f"Chat completion failed: {e}") from e

    async def _complete_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute non-streaming completion."""
        response_data = await with_circuit_breaker(
            name="llm_gateway",
            func=self._make_completion_request,
            request=request,
        )

        return ChatCompletionResponse(**response_data)

    async def _stream_completion(self, request: ChatCompletionRequest) -> AsyncIterator[Dict[str, Any]]:
        """Execute streaming completion."""
        if not self._http_client:
            raise GatewayError("LLM gateway client not initialized")

        try:
            async with self._http_client.stream(
                method="POST",
                url=f"{self.gateway_url}/v1/chat/completions",
                json=request.dict(exclude_none=True),
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPError as e:
            raise GatewayError(f"Streaming completion failed: {e}") from e

    async def _make_completion_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Make the actual completion request."""
        response_data = await self._make_request(
            method="POST",
            endpoint="/v1/chat/completions",
            json_data=request.dict(exclude_none=True),
        )

        return response_data

    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model: Model name

        Returns:
            Model information or None if not found
        """
        if model in self._available_models:
            return self._available_models[model]

        try:
            response = await self._make_request(
                method="GET",
                endpoint=f"/v1/models/{model}",
            )
            return response

        except Exception as e:
            self.logger.error(f"Failed to get model info for {model}: {e}")
            return None

    async def list_models(self) -> List[str]:
        """
        List available model names.

        Returns:
            List of model names
        """
        return list(self._available_models.keys())

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on LLM gateway.

        Returns:
            Health check results
        """
        try:
            response = await self._make_request(
                method="GET",
                endpoint="/health",
                timeout=10,
            )

            return {
                "status": "healthy",
                "gateway_url": self.gateway_url,
                "response": response,
                "available_models": len(self._available_models),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "gateway_url": self.gateway_url,
                "error": str(e),
                "available_models": len(self._available_models),
            }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the LLM gateway.

        Args:
            method: HTTP method
            endpoint: API endpoint
            json_data: Optional JSON data
            timeout: Optional timeout override

        Returns:
            Response data

        Raises:
            GatewayError: If request fails
        """
        if not self._http_client:
            raise GatewayError("LLM gateway client not initialized")

        try:
            response = await self._http_client.request(
                method=method,
                url=f"{self.gateway_url}{endpoint}",
                json=json_data,
                timeout=timeout or self.timeout,
            )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            raise GatewayError(f"HTTP error communicating with LLM gateway: {e}") from e
        except Exception as e:
            raise GatewayError(f"Error communicating with LLM gateway: {e}") from e

    def _get_gateway_url(self) -> str:
        """Get LLM gateway URL from configuration."""
        # self.config is a GatewayConfig when passed from AgentManager
        if self.config and getattr(self.config, 'llm_gateway', None):
            return self.config.llm_gateway.get("url", "http://localhost:8000")
        return "http://localhost:8000"

    def _get_default_model(self) -> str:
        """Get default model from configuration."""
        if self.config and getattr(self.config, 'llm_gateway', None):
            return self.config.llm_gateway.get("default_model", "gpt-4")
        return "gpt-4"

    def _get_timeout(self) -> int:
        """Get timeout from configuration."""
        if self.config and getattr(self.config, 'llm_gateway', None):
            return self.config.llm_gateway.get("timeout", 60)
        return 60

    def _get_retry_attempts(self) -> int:
        """Get retry attempts from configuration."""
        if self.config and getattr(self.config, 'llm_gateway', None):
            return self.config.llm_gateway.get("retry_attempts", 2)
        return 2

    def _get_auth_config(self) -> Dict[str, Any]:
        """Get authentication configuration."""
        if self.config and getattr(self.config, 'llm_gateway', None):
            return self.config.llm_gateway.get("auth", {})
        return {}


class LLMProxy:
    """
    Proxy for LLM operations with simplified interface.

    Provides a high-level interface for common LLM operations
    with automatic model selection and parameter optimization.
    """

    def __init__(self, llm_client: LLMGatewayClient):
        """
        Initialize LLM proxy.

        Args:
            llm_client: LLM gateway client instance
        """
        self.llm_client = llm_client
        self.logger = get_logger("llm_gateway.proxy")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

        messages.append(ChatMessage(role="user", content=prompt))

        response = await self.llm_client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if isinstance(response, ChatCompletionResponse):
            if response.choices:
                return response.choices[0].get("message", {}).get("content", "")

        return ""

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate text from a prompt with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Generated text chunks
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

        messages.append(ChatMessage(role="user", content=prompt))

        stream = await self.llm_client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Perform a chat completion.

        Args:
            messages: List of chat messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Assistant response
        """
        response = await self.llm_client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if isinstance(response, ChatCompletionResponse):
            if response.choices:
                return response.choices[0].get("message", {}).get("content", "")

        return ""