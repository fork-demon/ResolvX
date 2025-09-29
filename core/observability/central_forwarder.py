"""
Central observability forwarder for the Golden Agent Framework.

Provides a centralized way to forward observability data to multiple
backends including LangSmith, LangFuse, and custom endpoints.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from core.exceptions import ObservabilityError
from core.observability import get_logger


class ForwarderConfig(BaseModel):
    """Configuration for central observability forwarder."""
    enabled: bool = True
    backends: List[str] = ["langsmith", "langfuse", "console"]
    batch_size: int = 100
    flush_interval: float = 30.0
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class ObservabilityEvent(BaseModel):
    """Represents an observability event."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str  # agent_execution, tool_call, llm_call, workflow, etc.
    agent_name: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CentralForwarder:
    """
    Central observability forwarder.
    
    Forwards observability data to multiple backends
    including LangSmith, LangFuse, and custom endpoints.
    """

    def __init__(self, config: ForwarderConfig):
        """
        Initialize central forwarder.
        
        Args:
            config: Forwarder configuration
        """
        self.config = config
        self.logger = get_logger("observability.forwarder")
        
        # Initialize backends
        self._backends = {}
        self._event_queue: List[ObservabilityEvent] = []
        self._batch_task: Optional[asyncio.Task] = None
        
        # Initialize if enabled
        if config.enabled:
            self._initialize_backends()

    def _initialize_backends(self):
        """Initialize observability backends."""
        for backend in self.config.backends:
            try:
                if backend == "langsmith":
                    from core.observability.langsmith_tracer import LangSmithTracer, LangSmithConfig
                    backend_config = LangSmithConfig()
                    self._backends[backend] = LangSmithTracer(backend_config)
                elif backend == "langfuse":
                    from core.observability.langfuse_tracer import LangFuseTracer, LangFuseConfig
                    backend_config = LangFuseConfig()
                    self._backends[backend] = LangFuseTracer(backend_config)
                elif backend == "console":
                    from core.observability.console_tracer import ConsoleTracer, ConsoleConfig
                    backend_config = ConsoleConfig()
                    self._backends[backend] = ConsoleTracer(backend_config)
                else:
                    self.logger.warning(f"Unknown backend: {backend}")
                    continue
                
                self.logger.info(f"Initialized {backend} backend")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {backend} backend: {e}")
        
        # Start batch processing task
        if self._backends:
            self._batch_task = asyncio.create_task(self._batch_processor())

    async def forward_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        agent_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Forward an observability event.
        
        Args:
            event_type: Type of event
            data: Event data
            agent_name: Name of the agent
            trace_id: Trace ID
            span_id: Span ID
            parent_span_id: Parent span ID
            tags: Event tags
            metadata: Event metadata
        """
        if not self.config.enabled:
            return
        
        event = ObservabilityEvent(
            event_type=event_type,
            agent_name=agent_name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            data=data,
            tags=tags or {},
            metadata=metadata or {},
        )
        
        self._event_queue.append(event)
        
        # Flush if queue is full
        if len(self._event_queue) >= self.config.batch_size:
            await self._flush_events()

    async def forward_agent_execution(
        self,
        agent_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        status: str = "completed",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ):
        """
        Forward agent execution event.
        
        Args:
            agent_name: Name of the agent
            input_data: Input data
            output_data: Output data
            execution_time: Execution time in seconds
            status: Execution status
            trace_id: Trace ID
            span_id: Span ID
        """
        await self.forward_event(
            event_type="agent_execution",
            data={
                "input_data": input_data,
                "output_data": output_data,
                "execution_time": execution_time,
                "status": status,
            },
            agent_name=agent_name,
            trace_id=trace_id,
            span_id=span_id,
            tags={
                "agent_name": agent_name,
                "execution_type": "agent",
                "status": status,
            },
        )

    async def forward_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        execution_time: float,
        status: str = "completed",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        """
        Forward tool call event.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Tool result
            execution_time: Execution time in seconds
            status: Execution status
            trace_id: Trace ID
            span_id: Span ID
            parent_span_id: Parent span ID
        """
        await self.forward_event(
            event_type="tool_call",
            data={
                "tool_name": tool_name,
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time,
                "status": status,
            },
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            tags={
                "tool_name": tool_name,
                "execution_type": "tool",
                "status": status,
            },
        )

    async def forward_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_used: int,
        execution_time: float,
        status: str = "completed",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        """
        Forward LLM call event.
        
        Args:
            model: LLM model name
            prompt: Input prompt
            response: LLM response
            tokens_used: Number of tokens used
            execution_time: Execution time in seconds
            status: Execution status
            trace_id: Trace ID
            span_id: Span ID
            parent_span_id: Parent span ID
        """
        await self.forward_event(
            event_type="llm_call",
            data={
                "model": model,
                "prompt": prompt,
                "response": response,
                "tokens_used": tokens_used,
                "execution_time": execution_time,
                "status": status,
            },
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            tags={
                "model": model,
                "execution_type": "llm",
                "status": status,
            },
        )

    async def forward_workflow(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]],
        status: str = "completed",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ):
        """
        Forward workflow event.
        
        Args:
            workflow_name: Name of the workflow
            steps: Workflow steps
            status: Workflow status
            trace_id: Trace ID
            span_id: Span ID
        """
        await self.forward_event(
            event_type="workflow",
            data={
                "workflow_name": workflow_name,
                "steps": steps,
                "status": status,
            },
            trace_id=trace_id,
            span_id=span_id,
            tags={
                "workflow_name": workflow_name,
                "execution_type": "workflow",
                "status": status,
            },
        )

    async def forward_metrics(
        self,
        metrics_name: str,
        metrics_data: Dict[str, Any],
        agent_name: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """
        Forward metrics event.
        
        Args:
            metrics_name: Name of the metrics
            metrics_data: Metrics data
            agent_name: Name of the agent
            trace_id: Trace ID
        """
        await self.forward_event(
            event_type="metrics",
            data={
                "metrics_name": metrics_name,
                "metrics_data": metrics_data,
            },
            agent_name=agent_name,
            trace_id=trace_id,
            tags={
                "metrics_name": metrics_name,
                "execution_type": "metrics",
            },
        )

    async def forward_error(
        self,
        error_type: str,
        error_message: str,
        error_details: Dict[str, Any],
        agent_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ):
        """
        Forward error event.
        
        Args:
            error_type: Type of error
            error_message: Error message
            error_details: Error details
            agent_name: Name of the agent
            trace_id: Trace ID
            span_id: Span ID
        """
        await self.forward_event(
            event_type="error",
            data={
                "error_type": error_type,
                "error_message": error_message,
                "error_details": error_details,
            },
            agent_name=agent_name,
            trace_id=trace_id,
            span_id=span_id,
            tags={
                "error_type": error_type,
                "execution_type": "error",
            },
        )

    async def _batch_processor(self):
        """Process events in batches."""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch processor: {e}")

    async def _flush_events(self):
        """Flush queued events to all backends."""
        if not self._event_queue:
            return
        
        events = self._event_queue.copy()
        self._event_queue.clear()
        
        # Forward to all backends
        tasks = []
        for backend_name, backend in self._backends.items():
            task = asyncio.create_task(
                self._forward_to_backend(backend_name, backend, events)
            )
            tasks.append(task)
        
        # Wait for all backends to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.debug(f"Flushed {len(events)} events to {len(self._backends)} backends")

    async def _forward_to_backend(
        self,
        backend_name: str,
        backend: Any,
        events: List[ObservabilityEvent],
    ):
        """Forward events to a specific backend."""
        try:
            for event in events:
                if hasattr(backend, 'forward_event'):
                    await backend.forward_event(event)
                elif hasattr(backend, 'trace_agent_execution') and event.event_type == "agent_execution":
                    await backend.trace_agent_execution(
                        agent_name=event.agent_name or "unknown",
                        input_data=event.data.get("input_data", {}),
                        output_data=event.data.get("output_data", {}),
                        execution_time=event.data.get("execution_time", 0.0),
                        status=event.data.get("status", "completed"),
                    )
                elif hasattr(backend, 'trace_tool_call') and event.event_type == "tool_call":
                    await backend.trace_tool_call(
                        tool_name=event.data.get("tool_name", "unknown"),
                        parameters=event.data.get("parameters", {}),
                        result=event.data.get("result"),
                        execution_time=event.data.get("execution_time", 0.0),
                        status=event.data.get("status", "completed"),
                        parent_span_id=event.parent_span_id,
                    )
                elif hasattr(backend, 'trace_llm_call') and event.event_type == "llm_call":
                    await backend.trace_llm_call(
                        model=event.data.get("model", "unknown"),
                        prompt=event.data.get("prompt", ""),
                        response=event.data.get("response", ""),
                        tokens_used=event.data.get("tokens_used", 0),
                        execution_time=event.data.get("execution_time", 0.0),
                        status=event.data.get("status", "completed"),
                        parent_span_id=event.parent_span_id,
                    )
                elif hasattr(backend, 'trace_workflow') and event.event_type == "workflow":
                    await backend.trace_workflow(
                        workflow_name=event.data.get("workflow_name", "unknown"),
                        steps=event.data.get("steps", []),
                        status=event.data.get("status", "completed"),
                    )
            
            self.logger.debug(f"Successfully forwarded {len(events)} events to {backend_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to forward events to {backend_name}: {e}")

    async def close(self):
        """Close the forwarder and flush remaining events."""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining events
        await self._flush_events()
        
        # Close all backends
        for backend_name, backend in self._backends.items():
            try:
                if hasattr(backend, 'close'):
                    await backend.close()
            except Exception as e:
                self.logger.error(f"Error closing {backend_name} backend: {e}")
        
        self.logger.info("Central forwarder closed")
