"""
LangSmith tracing implementation for the Golden Agent Framework.

Provides comprehensive tracing and monitoring using LangSmith
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


class LangSmithConfig(BaseModel):
    """Configuration for LangSmith tracing."""
    api_url: str = "https://api.smith.langchain.com"
    api_key: Optional[str] = None
    project_name: str = "golden-agents"
    environment: str = "development"
    enabled: bool = True
    batch_size: int = 100
    flush_interval: float = 30.0
    timeout: float = 30.0


class LangSmithSpan(BaseModel):
    """Represents a span in LangSmith tracing."""
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


class LangSmithTracer:
    """
    LangSmith tracing implementation.
    
    Provides comprehensive tracing for agent execution,
    tool calls, and LLM interactions.
    """

    def __init__(self, config: LangSmithConfig):
        """
        Initialize LangSmith tracer.
        
        Args:
            config: LangSmith configuration
        """
        self.config = config
        self.logger = get_logger("observability.langsmith")
        
        # Initialize LangSmith client
        self._client = None
        self._spans: Dict[str, LangSmithSpan] = {}
        self._batch_queue: List[Dict[str, Any]] = []
        self._batch_task: Optional[asyncio.Task] = None
        
        # Initialize if enabled
        if config.enabled:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize LangSmith client."""
        try:
            # Import LangSmith client
            from langsmith import Client
            
            self._client = Client(
                api_url=self.config.api_url,
                api_key=self.config.api_key,
            )
            
            # Start batch processing task
            self._batch_task = asyncio.create_task(self._batch_processor())
            
            self.logger.info("LangSmith tracer initialized successfully")
            
        except ImportError:
            self.logger.warning("LangSmith not available, tracing disabled")
            self.config.enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize LangSmith tracer: {e}")
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
        
        span = LangSmithSpan(
            name=name,
            parent_span_id=parent_span_id,
            tags=tags or {},
            attributes=attributes or {},
        )
        
        self._spans[span.span_id] = span
        
        self.logger.debug(f"Started span: {name} ({span.span_id})")
        return span.span_id

    async def end_span(
        self,
        span_id: str,
        status: str = "completed",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        End a span.
        
        Args:
            span_id: Span ID
            status: Span status
            attributes: Additional attributes
        """
        if not self.config.enabled or span_id not in self._spans:
            return
        
        span = self._spans[span_id]
        span.end_time = datetime.utcnow()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        if attributes:
            span.attributes.update(attributes)
        
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

    async def _queue_span(self, span: LangSmithSpan):
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
        """Flush queued spans to LangSmith."""
        if not self.config.enabled or not self._batch_queue:
            return
        
        try:
            # Convert spans to LangSmith format
            langsmith_spans = []
            for span_data in self._batch_queue:
                langsmith_span = {
                    "id": span_data["span_id"],
                    "trace_id": span_data["trace_id"],
                    "parent_id": span_data["parent_span_id"],
                    "name": span_data["name"],
                    "start_time": span_data["start_time"],
                    "end_time": span_data["end_time"],
                    "duration_ms": span_data["duration_ms"],
                    "status": span_data["status"],
                    "tags": span_data["tags"],
                    "attributes": span_data["attributes"],
                    "events": span_data["events"],
                    "metadata": span_data["metadata"],
                }
                langsmith_spans.append(langsmith_span)
            
            # Send to LangSmith
            if self._client:
                await self._client.create_spans(
                    project_name=self.config.project_name,
                    spans=langsmith_spans,
                )
            
            # Clear queue
            self._batch_queue.clear()
            
            self.logger.debug(f"Flushed {len(langsmith_spans)} spans to LangSmith")
            
        except Exception as e:
            self.logger.error(f"Failed to flush spans to LangSmith: {e}")

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
        
        self.logger.info("LangSmith tracer closed")
