"""
Distributed tracing for the Golden Agent Framework.

Provides tracing capabilities using OpenTelemetry with support for
multiple backends including console, Jaeger, and cloud services.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from core.config import Config
from core.observability.logger import get_logger

# Global tracer configuration
_tracer_configured = False
_tracer_provider: Optional[TracerProvider] = None
_current_config: Optional[Dict[str, Any]] = None


def configure_tracing(config: Optional[Config] = None) -> None:
    """
    Configure distributed tracing for the framework.

    Args:
        config: Framework configuration
    """
    global _tracer_configured, _tracer_provider, _current_config

    if config is None:
        # Use default configuration
        observability_config = {
            "backend": "console",
            "console": {
                "enabled": True,
            }
        }
    else:
        observability_config = config.observability.dict()

    # Don't reconfigure if already configured with the same settings
    if _tracer_configured and _current_config == observability_config:
        return

    backend = observability_config.get("backend", "console")

    # Create resource
    resource = Resource.create({
        "service.name": "golden-agent-framework",
        "service.version": config.version if config else "0.1.0",
    })

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    if backend == "console":
        _configure_console_tracing(observability_config.get("console", {}))
    elif backend == "langsmith":
        _configure_langsmith_tracing(observability_config.get("langsmith", {}))
    elif backend == "langfuse":
        _configure_langfuse_tracing(observability_config.get("langfuse", {}))
    else:
        # Default to console
        _configure_console_tracing({})

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Instrument HTTP libraries
    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

    _tracer_configured = True
    _current_config = observability_config

    logger = get_logger("tracing")
    logger.info("Distributed tracing configured", backend=backend)


def _configure_console_tracing(console_config: Dict[str, Any]) -> None:
    """Configure console-based tracing."""
    if not console_config.get("enabled", True):
        return

    # Add console exporter
    console_exporter = ConsoleSpanExporter()
    console_processor = BatchSpanProcessor(console_exporter)
    _tracer_provider.add_span_processor(console_processor)


def _configure_langsmith_tracing(langsmith_config: Dict[str, Any]) -> None:
    """Configure LangSmith-based tracing."""
    # For now, fall back to console tracing
    # LangSmith integration would be implemented here
    _configure_console_tracing({"enabled": True})


def _configure_langfuse_tracing(langfuse_config: Dict[str, Any]) -> None:
    """Configure Langfuse-based tracing."""
    # For now, fall back to console tracing
    # Langfuse integration would be implemented here
    _configure_console_tracing({"enabled": True})


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer instance.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance
    """
    # Ensure tracing is configured
    if not _tracer_configured:
        configure_tracing()

    return trace.get_tracer(name)


class TracerMixin:
    """Mixin class to add tracing capabilities to other classes."""

    @property
    def tracer(self) -> trace.Tracer:
        """Get tracer for this class."""
        return get_tracer(self.__class__.__module__ + "." + self.__class__.__name__)


@contextmanager
def trace_function(
    tracer: trace.Tracer,
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> Generator[trace.Span, None, None]:
    """
    Context manager for tracing function execution.

    Args:
        tracer: Tracer instance
        span_name: Name of the span
        attributes: Span attributes
        kind: Span kind

    Yields:
        Active span
    """
    with tracer.start_as_current_span(span_name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def trace_agent_execution(
    tracer: trace.Tracer,
    agent_name: str,
    action: str,
    input_data: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
):
    """
    Decorator for tracing agent execution.

    Args:
        tracer: Tracer instance
        agent_name: Name of the agent
        action: Action being performed
        input_data: Input data for the action
        workflow_id: Optional workflow ID

    Returns:
        Context manager for the trace
    """
    attributes = {
        "agent.name": agent_name,
        "agent.action": action,
    }

    if workflow_id:
        attributes["workflow.id"] = workflow_id

    if input_data:
        # Add safe attributes from input data
        for key, value in input_data.items():
            if isinstance(value, (str, int, float, bool)):
                attributes[f"input.{key}"] = value

    return trace_function(
        tracer,
        f"agent.{agent_name}.{action}",
        attributes,
        trace.SpanKind.INTERNAL
    )


def trace_tool_execution(
    tracer: trace.Tracer,
    tool_name: str,
    parameters: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for tracing tool execution.

    Args:
        tracer: Tracer instance
        tool_name: Name of the tool
        parameters: Tool parameters

    Returns:
        Context manager for the trace
    """
    attributes = {
        "tool.name": tool_name,
    }

    if parameters:
        # Add safe attributes from parameters
        for key, value in parameters.items():
            if isinstance(value, (str, int, float, bool)):
                attributes[f"tool.param.{key}"] = value

    return trace_function(
        tracer,
        f"tool.{tool_name}",
        attributes,
        trace.SpanKind.CLIENT
    )


def trace_workflow_execution(
    tracer: trace.Tracer,
    workflow_name: str,
    workflow_id: str,
    phase: Optional[str] = None,
):
    """
    Context manager for tracing workflow execution.

    Args:
        tracer: Tracer instance
        workflow_name: Name of the workflow
        workflow_id: Unique workflow ID
        phase: Current workflow phase

    Returns:
        Context manager for the trace
    """
    attributes = {
        "workflow.name": workflow_name,
        "workflow.id": workflow_id,
    }

    if phase:
        attributes["workflow.phase"] = phase

    span_name = f"workflow.{workflow_name}"
    if phase:
        span_name += f".{phase}"

    return trace_function(
        tracer,
        span_name,
        attributes,
        trace.SpanKind.INTERNAL
    )


def trace_gateway_request(
    tracer: trace.Tracer,
    gateway_type: str,
    operation: str,
    url: Optional[str] = None,
):
    """
    Context manager for tracing gateway requests.

    Args:
        tracer: Tracer instance
        gateway_type: Type of gateway (mcp, llm)
        operation: Operation being performed
        url: Gateway URL

    Returns:
        Context manager for the trace
    """
    attributes = {
        "gateway.type": gateway_type,
        "gateway.operation": operation,
    }

    if url:
        attributes["gateway.url"] = url

    return trace_function(
        tracer,
        f"gateway.{gateway_type}.{operation}",
        attributes,
        trace.SpanKind.CLIENT
    )


class PerformanceTracker:
    """Helper class for tracking performance metrics within traces."""

    def __init__(self, span: trace.Span):
        """
        Initialize performance tracker.

        Args:
            span: Active span to add metrics to
        """
        self.span = span
        self.start_time = time.time()
        self.checkpoints: Dict[str, float] = {}

    def checkpoint(self, name: str) -> None:
        """
        Add a performance checkpoint.

        Args:
            name: Checkpoint name
        """
        current_time = time.time()
        elapsed = (current_time - self.start_time) * 1000  # Convert to milliseconds
        self.checkpoints[name] = elapsed
        self.span.set_attribute(f"perf.{name}_ms", elapsed)

    def finish(self) -> None:
        """Finish performance tracking and add final metrics."""
        total_time = (time.time() - self.start_time) * 1000
        self.span.set_attribute("perf.total_ms", total_time)


def create_performance_tracker(span: trace.Span) -> PerformanceTracker:
    """
    Create a performance tracker for a span.

    Args:
        span: Active span

    Returns:
        Performance tracker instance
    """
    return PerformanceTracker(span)


# Correlation ID utilities

def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID from the active span.

    Returns:
        Correlation ID or None if no active span
    """
    span = trace.get_current_span()
    if span and span.is_recording():
        # Try to get correlation ID from span attributes
        return span.get_span_context().trace_id
    return None


def set_correlation_id(span: trace.Span, correlation_id: str) -> None:
    """
    Set correlation ID on a span.

    Args:
        span: Span to set correlation ID on
        correlation_id: Correlation ID to set
    """
    span.set_attribute("correlation.id", correlation_id)


# Trace data utilities

def add_agent_metadata(span: trace.Span, agent_data: Dict[str, Any]) -> None:
    """
    Add agent metadata to a span.

    Args:
        span: Span to add metadata to
        agent_data: Agent metadata
    """
    for key, value in agent_data.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"agent.{key}", value)


def add_tool_metadata(span: trace.Span, tool_data: Dict[str, Any]) -> None:
    """
    Add tool metadata to a span.

    Args:
        span: Span to add metadata to
        tool_data: Tool metadata
    """
    for key, value in tool_data.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"tool.{key}", value)


def add_workflow_metadata(span: trace.Span, workflow_data: Dict[str, Any]) -> None:
    """
    Add workflow metadata to a span.

    Args:
        span: Span to add metadata to
        workflow_data: Workflow metadata
    """
    for key, value in workflow_data.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"workflow.{key}", value)


# Error handling utilities

def record_error(span: trace.Span, error: Exception, error_type: str = "error") -> None:
    """
    Record an error in a span.

    Args:
        span: Span to record error in
        error: Exception that occurred
        error_type: Type of error
    """
    span.record_exception(error)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
    span.set_attribute("error.type", error_type)
    span.set_attribute("error.message", str(error))


def record_warning(span: trace.Span, message: str, warning_type: str = "warning") -> None:
    """
    Record a warning in a span.

    Args:
        span: Span to record warning in
        message: Warning message
        warning_type: Type of warning
    """
    span.add_event("warning", {
        "warning.type": warning_type,
        "warning.message": message,
    })
    span.set_attribute("warning.occurred", True)