"""
LangFuse tracing implementation for the Golden Agent Framework.

Provides comprehensive tracing and monitoring using LangFuse
for agent execution, tool calls, and LLM interactions.
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


class LangFuseConfig(BaseModel):
    """Configuration for LangFuse tracing."""
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "https://cloud.langfuse.com"
    project_name: str = "golden-agents"
    environment: str = "development"
    enabled: bool = True
    batch_size: int = 100
    flush_interval: float = 30.0
    timeout: float = 30.0


class LangFuseSpan(BaseModel):
    """Represents a span in LangFuse tracing."""
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: Optional[str] = None
    name: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "started"  # started, completed, failed
    tags: Dict[str, str] = Field(default_factory=dict)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    input: Optional[Any] = None
    output: Optional[Any] = None
    langfuse_span: Optional[Any] = Field(default=None, exclude=True)  # Reference to actual Langfuse span
    
    model_config = {
        "arbitrary_types_allowed": True
    }


class LangFuseTracer:
    """
    LangFuse tracing implementation.
    
    Provides comprehensive tracing for agent execution,
    tool calls, and LLM interactions.
    """

    def __init__(self, config: LangFuseConfig):
        """
        Initialize LangFuse tracer.
        
        Args:
            config: LangFuse configuration
        """
        self.config = config
        self.logger = get_logger("observability.langfuse")
        
        # Initialize LangFuse client
        self._client = None
        self._spans: Dict[str, LangFuseSpan] = {}
        self._batch_queue: List[Dict[str, Any]] = []
        self._batch_task: Optional[asyncio.Task] = None
        
        # Initialize if enabled
        if config.enabled:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize LangFuse client."""
        try:
            # Import LangFuse client
            from langfuse import Langfuse
            
            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
            )
            
            # Don't start batch processing task here - it will be started when first span is created
            self._batch_task = None
            
            self.logger.info("LangFuse tracer initialized successfully")
            
        except ImportError:
            self.logger.warning("LangFuse not available, tracing disabled")
            self.config.enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize LangFuse tracer: {e}")
            self.config.enabled = False

    async def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new span.
        
        Args:
            name: Span name
            parent_span_id: Parent span ID
            tags: Span tags
            attributes: Span attributes
            
        Returns:
            Span ID
        """
        if not self.config.enabled:
            return str(uuid4())
        
        # Start batch processor if not already started
        if self._batch_task is None:
            try:
                self._batch_task = asyncio.create_task(self._batch_processor())
                self.logger.debug("Started LangFuse batch processor")
            except Exception as e:
                self.logger.warning(f"Failed to start batch processor: {e}")
        
        span = LangFuseSpan(
            name=name,
            parent_span_id=parent_span_id,
            tags=tags or {},
            attributes=attributes or {},
        )
        
        self._spans[span.span_id] = span
        
        # Create actual Langfuse span if client is available
        if self._client:
            try:
                # Prepare metadata from tags and attributes
                metadata = {**span.tags, **span.attributes}
                
                # Start Langfuse span
                langfuse_span = self._client.start_span(
                    name=name,
                    metadata=metadata
                )
                
                # Store the Langfuse span reference
                span.langfuse_span = langfuse_span
                
            except Exception as e:
                self.logger.error(f"Failed to start Langfuse span: {e}")
        
        self.logger.debug(f"Started span: {name} ({span.span_id})")
        return span.span_id

    async def end_span(
        self,
        span_id: str,
        status: str = "completed",
        attributes: Optional[Dict[str, Any]] = None,
        output: Optional[Any] = None,
    ):
        """
        End a span.
        
        Args:
            span_id: Span ID
            status: Span status
            attributes: Additional attributes
            output: Span output data
        """
        if not self.config.enabled or span_id not in self._spans:
            return
        
        span = self._spans[span_id]
        span.end_time = datetime.utcnow()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        if attributes:
            span.attributes.update(attributes)
        
        if output is not None:
            span.output = output
        
        # End the Langfuse span if it exists
        if hasattr(span, 'langfuse_span') and span.langfuse_span:
            try:
                # Update the Langfuse span with final data
                span.langfuse_span.update(
                    output=output,
                    metadata={**span.tags, **span.attributes},
                    status_message=status
                )
                # The span will be automatically ended when it goes out of scope
            except Exception as e:
                self.logger.error(f"Failed to update Langfuse span: {e}")
        
        # Queue for batch processing
        await self._queue_span(span)
        
        self.logger.debug(f"Ended span: {span.name} ({span_id}) - {status}")

    async def add_event(
        self,
        span_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Add an event to a span.
        
        Args:
            span_id: Span ID
            name: Event name
            attributes: Event attributes
        """
        if not self.config.enabled or span_id not in self._spans:
            return
        
        event = {
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        }
        
        self._spans[span_id].events.append(event)
        
        self.logger.debug(f"Added event to span {span_id}: {name}")

    async def add_attributes(
        self,
        span_id: str,
        attributes: Dict[str, Any],
    ):
        """
        Add attributes to a span.
        
        Args:
            span_id: Span ID
            attributes: Attributes to add
        """
        if not self.config.enabled or span_id not in self._spans:
            return
        
        self._spans[span_id].attributes.update(attributes)
        
        self.logger.debug(f"Added attributes to span {span_id}: {list(attributes.keys())}")

    async def add_tags(
        self,
        span_id: str,
        tags: Dict[str, str],
    ):
        """
        Add tags to a span.
        
        Args:
            span_id: Span ID
            tags: Tags to add
        """
        if not self.config.enabled or span_id not in self._spans:
            return
        
        self._spans[span_id].tags.update(tags)
        
        self.logger.debug(f"Added tags to span {span_id}: {list(tags.keys())}")

    async def trace_agent_execution(
        self,
        agent_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        status: str = "completed",
    ) -> str:
        """
        Trace agent execution.
        
        Args:
            agent_name: Name of the agent
            input_data: Input data
            output_data: Output data
            execution_time: Execution time in seconds
            status: Execution status
            
        Returns:
            Span ID
        """
        span_id = await self.start_span(
            name=f"agent_execution_{agent_name}",
            tags={
                "agent_name": agent_name,
                "execution_type": "agent",
            },
            attributes={
                "input_data": input_data,
                "output_data": output_data,
                "execution_time": execution_time,
                "status": status,
            },
        )
        
        await self.add_event(
            span_id,
            "agent_started",
            {"agent_name": agent_name, "input_data": input_data},
        )
        
        await self.add_event(
            span_id,
            "agent_completed",
            {"agent_name": agent_name, "output_data": output_data, "status": status},
        )
        
        await self.end_span(span_id, status)
        return span_id

    async def trace_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        execution_time: float,
        status: str = "completed",
        parent_span_id: Optional[str] = None,
    ) -> str:
        """
        Trace tool call.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Tool result
            execution_time: Execution time in seconds
            status: Execution status
            parent_span_id: Parent span ID
            
        Returns:
            Span ID
        """
        span_id = await self.start_span(
            name=f"tool_call_{tool_name}",
            parent_span_id=parent_span_id,
            tags={
                "tool_name": tool_name,
                "execution_type": "tool",
            },
            attributes={
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time,
                "status": status,
            },
        )
        
        await self.add_event(
            span_id,
            "tool_started",
            {"tool_name": tool_name, "parameters": parameters},
        )
        
        await self.add_event(
            span_id,
            "tool_completed",
            {"tool_name": tool_name, "result": result, "status": status},
        )
        
        await self.end_span(span_id, status)
        return span_id

    async def trace_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_used: int,
        execution_time: float,
        status: str = "completed",
        parent_span_id: Optional[str] = None,
    ) -> str:
        """
        Trace LLM call.
        
        Args:
            model: LLM model name
            prompt: Input prompt
            response: LLM response
            tokens_used: Number of tokens used
            execution_time: Execution time in seconds
            status: Execution status
            parent_span_id: Parent span ID
            
        Returns:
            Span ID
        """
        span_id = await self.start_span(
            name=f"llm_call_{model}",
            parent_span_id=parent_span_id,
            tags={
                "model": model,
                "execution_type": "llm",
            },
            attributes={
                "prompt": prompt,
                "response": response,
                "tokens_used": tokens_used,
                "execution_time": execution_time,
                "status": status,
            },
        )
        
        await self.add_event(
            span_id,
            "llm_started",
            {"model": model, "prompt": prompt},
        )
        
        await self.add_event(
            span_id,
            "llm_completed",
            {"model": model, "response": response, "tokens_used": tokens_used},
        )
        
        await self.end_span(span_id, status)
        return span_id

    async def trace_workflow(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]],
        status: str = "completed",
    ) -> str:
        """
        Trace workflow execution.
        
        Args:
            workflow_name: Name of the workflow
            steps: Workflow steps
            status: Workflow status
            
        Returns:
            Span ID
        """
        span_id = await self.start_span(
            name=f"workflow_{workflow_name}",
            tags={
                "workflow_name": workflow_name,
                "execution_type": "workflow",
            },
            attributes={
                "steps": steps,
                "status": status,
            },
        )
        
        await self.add_event(
            span_id,
            "workflow_started",
            {"workflow_name": workflow_name, "steps": steps},
        )
        
        await self.add_event(
            span_id,
            "workflow_completed",
            {"workflow_name": workflow_name, "status": status},
        )
        
        await self.end_span(span_id, status)
        return span_id

    async def _queue_span(self, span: LangFuseSpan):
        """Queue span for batch processing."""
        if not self.config.enabled:
            return
        
        span_data = {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "start_time": span.start_time.isoformat(),
            "end_time": span.end_time.isoformat() if span.end_time else None,
            "duration_ms": span.duration_ms,
            "status": span.status,
            "tags": span.tags,
            "attributes": span.attributes,
            "events": span.events,
            "metadata": span.metadata,
        }
        
        self._batch_queue.append(span_data)
        
        # Flush if batch is full
        if len(self._batch_queue) >= self.config.batch_size:
            await self._flush_batch()

    async def _batch_processor(self):
        """Process spans in batches."""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch processor: {e}")

    async def _flush_batch(self):
        """Flush queued spans to LangFuse."""
        if not self.config.enabled or not self._batch_queue:
            return
        
        try:
            span_count = len(self._batch_queue)
            
            # Clear queue (spans are already sent via the context manager in start_span/end_span)
            self._batch_queue.clear()
            
            # Flush the Langfuse client to ensure data is sent to the server
            if self._client:
                self._client.flush()
            
            self.logger.debug(f"Flushed {span_count} spans to LangFuse")
            
        except Exception as e:
            self.logger.error(f"Failed to flush spans to LangFuse: {e}")

    async def flush(self):
        """Manually flush queued spans to LangFuse."""
        await self._flush_batch()
        # Also flush the Langfuse client to ensure data is sent to the server
        if self._client:
            self._client.flush()
        self.logger.debug("Manually flushed spans to LangFuse")
    
    async def close(self):
        """Close the tracer and flush remaining spans."""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining spans
        await self._flush_batch()
        
        self.logger.info("LangFuse tracer closed")
