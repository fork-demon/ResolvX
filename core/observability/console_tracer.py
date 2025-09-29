"""
Console tracing implementation for the Golden Agent Framework.

Provides console-based tracing and logging for local development
and debugging purposes.
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


class ConsoleConfig(BaseModel):
    """Configuration for console tracing."""
    enabled: bool = True
    log_level: str = "INFO"
    include_timestamps: bool = True
    include_metadata: bool = True
    format: str = "structured"  # structured, simple, json
    colorize: bool = True


class ConsoleTracer:
    """
    Console tracing implementation.
    
    Provides console-based tracing for local development
    and debugging purposes.
    """

    def __init__(self, config: ConsoleConfig):
        """
        Initialize console tracer.
        
        Args:
            config: Console configuration
        """
        self.config = config
        self.logger = get_logger("observability.console")
        
        # Set up console logging
        if config.enabled:
            self._setup_console_logging()

    def _setup_console_logging(self):
        """Set up console logging."""
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Set up formatter
        if self.config.format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
        elif self.config.format == "structured":
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:  # simple
            formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))

    async def forward_event(self, event: 'ObservabilityEvent'):
        """
        Forward an observability event to console.
        
        Args:
            event: Observability event
        """
        if not self.config.enabled:
            return
        
        # Format event for console output
        message = self._format_event(event)
        
        # Log based on event type
        if event.event_type == "error":
            self.logger.error(message)
        elif event.event_type == "agent_execution":
            self.logger.info(message)
        elif event.event_type == "tool_call":
            self.logger.debug(message)
        elif event.event_type == "llm_call":
            self.logger.debug(message)
        elif event.event_type == "workflow":
            self.logger.info(message)
        elif event.event_type == "metrics":
            self.logger.info(message)
        else:
            self.logger.info(message)

    def _format_event(self, event: 'ObservabilityEvent') -> str:
        """Format event for console output."""
        if self.config.format == "json":
            return self._format_json(event)
        elif self.config.format == "structured":
            return self._format_structured(event)
        else:  # simple
            return self._format_simple(event)

    def _format_json(self, event: 'ObservabilityEvent') -> str:
        """Format event as JSON."""
        event_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "agent_name": event.agent_name,
            "trace_id": event.trace_id,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "data": event.data,
            "tags": event.tags,
            "metadata": event.metadata,
        }
        
        return json.dumps(event_data, indent=2)

    def _format_structured(self, event: 'ObservabilityEvent') -> str:
        """Format event as structured text."""
        lines = []
        
        # Header
        lines.append(f"=== {event.event_type.upper()} ===")
        
        # Basic info
        if event.agent_name:
            lines.append(f"Agent: {event.agent_name}")
        if event.trace_id:
            lines.append(f"Trace ID: {event.trace_id}")
        if event.span_id:
            lines.append(f"Span ID: {event.span_id}")
        if event.parent_span_id:
            lines.append(f"Parent Span ID: {event.parent_span_id}")
        
        # Data
        if event.data:
            lines.append("Data:")
            for key, value in event.data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"  {key}: {json.dumps(value, indent=4)}")
                else:
                    lines.append(f"  {key}: {value}")
        
        # Tags
        if event.tags:
            lines.append("Tags:")
            for key, value in event.tags.items():
                lines.append(f"  {key}: {value}")
        
        # Metadata
        if event.metadata and self.config.include_metadata:
            lines.append("Metadata:")
            for key, value in event.metadata.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"  {key}: {json.dumps(value, indent=4)}")
                else:
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)

    def _format_simple(self, event: 'ObservabilityEvent') -> str:
        """Format event as simple text."""
        parts = [event.event_type]
        
        if event.agent_name:
            parts.append(f"agent={event.agent_name}")
        if event.trace_id:
            parts.append(f"trace={event.trace_id}")
        if event.span_id:
            parts.append(f"span={event.span_id}")
        
        # Add key data points
        if event.data:
            if "execution_time" in event.data:
                parts.append(f"time={event.data['execution_time']:.2f}s")
            if "status" in event.data:
                parts.append(f"status={event.data['status']}")
            if "tool_name" in event.data:
                parts.append(f"tool={event.data['tool_name']}")
            if "model" in event.data:
                parts.append(f"model={event.data['model']}")
        
        return " | ".join(parts)

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
        span_id = str(uuid4())
        
        # Create event
        from core.observability.central_forwarder import ObservabilityEvent
        event = ObservabilityEvent(
            event_type="agent_execution",
            agent_name=agent_name,
            span_id=span_id,
            data={
                "input_data": input_data,
                "output_data": output_data,
                "execution_time": execution_time,
                "status": status,
            },
            tags={
                "agent_name": agent_name,
                "execution_type": "agent",
                "status": status,
            },
        )
        
        await self.forward_event(event)
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
        span_id = str(uuid4())
        
        # Create event
        from core.observability.central_forwarder import ObservabilityEvent
        event = ObservabilityEvent(
            event_type="tool_call",
            span_id=span_id,
            parent_span_id=parent_span_id,
            data={
                "tool_name": tool_name,
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time,
                "status": status,
            },
            tags={
                "tool_name": tool_name,
                "execution_type": "tool",
                "status": status,
            },
        )
        
        await self.forward_event(event)
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
        span_id = str(uuid4())
        
        # Create event
        from core.observability.central_forwarder import ObservabilityEvent
        event = ObservabilityEvent(
            event_type="llm_call",
            span_id=span_id,
            parent_span_id=parent_span_id,
            data={
                "model": model,
                "prompt": prompt,
                "response": response,
                "tokens_used": tokens_used,
                "execution_time": execution_time,
                "status": status,
            },
            tags={
                "model": model,
                "execution_type": "llm",
                "status": status,
            },
        )
        
        await self.forward_event(event)
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
        span_id = str(uuid4())
        
        # Create event
        from core.observability.central_forwarder import ObservabilityEvent
        event = ObservabilityEvent(
            event_type="workflow",
            span_id=span_id,
            data={
                "workflow_name": workflow_name,
                "steps": steps,
                "status": status,
            },
            tags={
                "workflow_name": workflow_name,
                "execution_type": "workflow",
                "status": status,
            },
        )
        
        await self.forward_event(event)
        return span_id

    async def close(self):
        """Close the console tracer."""
        self.logger.info("Console tracer closed")
