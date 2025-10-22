"""
Metrics collection for the Golden Agent Framework.

Provides metrics collection and monitoring with support for
Prometheus, cloud-based metrics services, and custom backends.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import prometheus_client

from core.config import Config
from core.observability.logger import get_logger

# Global metrics configuration
_metrics_configured = False
_metrics_registry: Optional[prometheus_client.CollectorRegistry] = None
_current_config: Optional[Dict[str, Any]] = None

# Default metrics
_agent_execution_counter: Optional[Counter] = None
_agent_execution_duration: Optional[Histogram] = None
_tool_execution_counter: Optional[Counter] = None
_tool_execution_duration: Optional[Histogram] = None
_workflow_execution_counter: Optional[Counter] = None
_workflow_execution_duration: Optional[Histogram] = None
_memory_operations_counter: Optional[Counter] = None
_gateway_requests_counter: Optional[Counter] = None
_gateway_request_duration: Optional[Histogram] = None
_active_agents_gauge: Optional[Gauge] = None
_framework_info: Optional[Info] = None


def configure_metrics(config: Optional[Config] = None) -> None:
    """
    Configure metrics collection for the framework.

    Args:
        config: Framework configuration
    """
    global _metrics_configured, _metrics_registry, _current_config

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
    if _metrics_configured and _current_config == observability_config:
        return

    backend = observability_config.get("backend", "console")

    # Create metrics registry
    _metrics_registry = prometheus_client.CollectorRegistry()

    if backend == "console":
        _configure_console_metrics(observability_config.get("console", {}))
    elif backend == "langsmith":
        _configure_langsmith_metrics(observability_config.get("langsmith", {}))
    elif backend == "langfuse":
        _configure_langfuse_metrics(observability_config.get("langfuse", {}))
    else:
        # Default to console
        _configure_console_metrics({})

    # Initialize default metrics
    _initialize_default_metrics()

    _metrics_configured = True
    _current_config = observability_config

    logger = get_logger("metrics")
    logger.info("Metrics collection configured", backend=backend)


def _configure_console_metrics(console_config: Dict[str, Any]) -> None:
    """Configure console-based metrics."""
    if not console_config.get("enabled", True):
        return

    # Start Prometheus HTTP server for metrics endpoint
    port = console_config.get("port", 8080)
    try:
        start_http_server(port, registry=_metrics_registry)
        logger = get_logger("metrics")
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger = get_logger("metrics")
        logger.warning(f"Failed to start metrics server: {e}")


def _configure_langsmith_metrics(langsmith_config: Dict[str, Any]) -> None:
    """Configure LangSmith-based metrics."""
    # For now, fall back to console metrics
    # LangSmith integration would be implemented here
    _configure_console_metrics({"enabled": True, "port": 8080})


def _configure_langfuse_metrics(langfuse_config: Dict[str, Any]) -> None:
    """Configure Langfuse-based metrics."""
    # For now, fall back to console metrics
    # Langfuse integration would be implemented here
    _configure_console_metrics({"enabled": True, "port": 8080})


def _initialize_default_metrics() -> None:
    """Initialize default framework metrics."""
    global _agent_execution_counter, _agent_execution_duration
    global _tool_execution_counter, _tool_execution_duration
    global _workflow_execution_counter, _workflow_execution_duration
    global _memory_operations_counter, _gateway_requests_counter
    global _gateway_request_duration, _active_agents_gauge, _framework_info

    # Agent metrics
    _agent_execution_counter = Counter(
        'agent_executions_total',
        'Total number of agent executions',
        ['agent_name', 'status', 'action'],
        registry=_metrics_registry
    )

    _agent_execution_duration = Histogram(
        'agent_execution_duration_seconds',
        'Time spent executing agents',
        ['agent_name', 'action'],
        registry=_metrics_registry
    )

    # Tool metrics
    _tool_execution_counter = Counter(
        'tool_executions_total',
        'Total number of tool executions',
        ['tool_name', 'status', 'tool_type'],
        registry=_metrics_registry
    )

    _tool_execution_duration = Histogram(
        'tool_execution_duration_seconds',
        'Time spent executing tools',
        ['tool_name', 'tool_type'],
        registry=_metrics_registry
    )

    # Workflow metrics
    _workflow_execution_counter = Counter(
        'workflow_executions_total',
        'Total number of workflow executions',
        ['workflow_name', 'status'],
        registry=_metrics_registry
    )

    _workflow_execution_duration = Histogram(
        'workflow_execution_duration_seconds',
        'Time spent executing workflows',
        ['workflow_name'],
        registry=_metrics_registry
    )

    # Memory metrics
    _memory_operations_counter = Counter(
        'memory_operations_total',
        'Total number of memory operations',
        ['operation', 'backend', 'namespace', 'status'],
        registry=_metrics_registry
    )

    # Gateway metrics
    _gateway_requests_counter = Counter(
        'gateway_requests_total',
        'Total number of gateway requests',
        ['gateway_type', 'operation', 'status'],
        registry=_metrics_registry
    )

    _gateway_request_duration = Histogram(
        'gateway_request_duration_seconds',
        'Time spent on gateway requests',
        ['gateway_type', 'operation'],
        registry=_metrics_registry
    )

    # System metrics
    _active_agents_gauge = Gauge(
        'active_agents',
        'Number of currently active agents',
        ['agent_type'],
        registry=_metrics_registry
    )

    # Framework info
    _framework_info = Info(
        'framework_info',
        'Information about the Golden Agent Framework',
        registry=_metrics_registry
    )

    _framework_info.info({
        'version': '0.1.0',
        'python_version': '3.10+',
    })


class MetricsClient:
    """Client for collecting and publishing metrics."""

    def __init__(self):
        """Initialize metrics client."""
        self.logger = get_logger("metrics.client")
        self._custom_counters: Dict[str, Counter] = {}
        self._custom_histograms: Dict[str, Histogram] = {}
        self._custom_gauges: Dict[str, Gauge] = {}

    def record_agent_execution(
        self,
        agent_name: str,
        action: str,
        duration_seconds: float,
        status: str = "success",
        **labels: Any
    ) -> None:
        """
        Record agent execution metrics.

        Args:
            agent_name: Name of the agent
            action: Action performed
            duration_seconds: Execution duration in seconds
            status: Execution status (success, failure, timeout)
            **labels: Additional labels
        """
        if not _metrics_configured:
            return

        try:
            _agent_execution_counter.labels(
                agent_name=agent_name,
                status=status,
                action=action
            ).inc()

            _agent_execution_duration.labels(
                agent_name=agent_name,
                action=action
            ).observe(duration_seconds)

        except Exception as e:
            self.logger.warning(f"Failed to record agent metrics: {e}")

    def record_tool_execution(
        self,
        tool_name: str,
        tool_type: str,
        duration_seconds: float,
        status: str = "success",
        **labels: Any
    ) -> None:
        """
        Record tool execution metrics.

        Args:
            tool_name: Name of the tool
            tool_type: Type of tool (local, mcp)
            duration_seconds: Execution duration in seconds
            status: Execution status
            **labels: Additional labels
        """
        if not _metrics_configured:
            return

        try:
            _tool_execution_counter.labels(
                tool_name=tool_name,
                status=status,
                tool_type=tool_type
            ).inc()

            _tool_execution_duration.labels(
                tool_name=tool_name,
                tool_type=tool_type
            ).observe(duration_seconds)

        except Exception as e:
            self.logger.warning(f"Failed to record tool metrics: {e}")

    def record_workflow_execution(
        self,
        workflow_name: str,
        duration_seconds: float,
        status: str = "success",
        **labels: Any
    ) -> None:
        """
        Record workflow execution metrics.

        Args:
            workflow_name: Name of the workflow
            duration_seconds: Execution duration in seconds
            status: Execution status
            **labels: Additional labels
        """
        if not _metrics_configured:
            return

        try:
            _workflow_execution_counter.labels(
                workflow_name=workflow_name,
                status=status
            ).inc()

            _workflow_execution_duration.labels(
                workflow_name=workflow_name
            ).observe(duration_seconds)

        except Exception as e:
            self.logger.warning(f"Failed to record workflow metrics: {e}")

    def record_memory_operation(
        self,
        operation: str,
        backend: str,
        namespace: str = "default",
        status: str = "success",
        **labels: Any
    ) -> None:
        """
        Record memory operation metrics.

        Args:
            operation: Type of operation (store, retrieve, search, delete)
            backend: Memory backend used
            namespace: Memory namespace
            status: Operation status
            **labels: Additional labels
        """
        if not _metrics_configured:
            return

        try:
            _memory_operations_counter.labels(
                operation=operation,
                backend=backend,
                namespace=namespace,
                status=status
            ).inc()

        except Exception as e:
            self.logger.warning(f"Failed to record memory metrics: {e}")

    def record_gateway_request(
        self,
        gateway_type: str,
        operation: str,
        duration_seconds: float,
        status: str = "success",
        **labels: Any
    ) -> None:
        """
        Record gateway request metrics.

        Args:
            gateway_type: Type of gateway (mcp, llm)
            operation: Operation performed
            duration_seconds: Request duration in seconds
            status: Request status
            **labels: Additional labels
        """
        if not _metrics_configured:
            return

        try:
            _gateway_requests_counter.labels(
                gateway_type=gateway_type,
                operation=operation,
                status=status
            ).inc()

            _gateway_request_duration.labels(
                gateway_type=gateway_type,
                operation=operation
            ).observe(duration_seconds)

        except Exception as e:
            self.logger.warning(f"Failed to record gateway metrics: {e}")

    def set_active_agents(self, agent_type: str, count: int) -> None:
        """
        Set the number of active agents.

        Args:
            agent_type: Type of agent
            count: Number of active agents
        """
        if not _metrics_configured:
            return

        try:
            _active_agents_gauge.labels(agent_type=agent_type).set(count)

        except Exception as e:
            self.logger.warning(f"Failed to set agent gauge: {e}")

    def create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Counter:
        """
        Create a custom counter metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels

        Returns:
            Counter instance
        """
        if name in self._custom_counters:
            return self._custom_counters[name]

        try:
            counter = Counter(
                name,
                description,
                labels or [],
                registry=_metrics_registry
            )
            self._custom_counters[name] = counter
            return counter

        except Exception as e:
            self.logger.error(f"Failed to create counter {name}: {e}")
            # Return a no-op counter
            return Counter(name, description, labels or [])

    def create_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """
        Create a custom histogram metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
            buckets: Histogram buckets

        Returns:
            Histogram instance
        """
        if name in self._custom_histograms:
            return self._custom_histograms[name]

        try:
            histogram = Histogram(
                name,
                description,
                labels or [],
                buckets=buckets,
                registry=_metrics_registry
            )
            self._custom_histograms[name] = histogram
            return histogram

        except Exception as e:
            self.logger.error(f"Failed to create histogram {name}: {e}")
            # Return a no-op histogram
            return Histogram(name, description, labels or [])

    def create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """
        Create a custom gauge metric.

        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels

        Returns:
            Gauge instance
        """
        if name in self._custom_gauges:
            return self._custom_gauges[name]

        try:
            gauge = Gauge(
                name,
                description,
                labels or [],
                registry=_metrics_registry
            )
            self._custom_gauges[name] = gauge
            return gauge

        except Exception as e:
            self.logger.error(f"Failed to create gauge {name}: {e}")
            # Return a no-op gauge
            return Gauge(name, description, labels or [])


class MetricsCollector:
    """Collector for gathering framework metrics."""

    def __init__(self, metrics_client: MetricsClient):
        """
        Initialize metrics collector.

        Args:
            metrics_client: Metrics client instance
        """
        self.metrics_client = metrics_client
        self.logger = get_logger("metrics.collector")
        self._counters: Dict[str, int] = defaultdict(int)

    def start_agent_execution(self, agent_name: str, action: str) -> 'MetricsTimer':
        """
        Start timing an agent execution.

        Args:
            agent_name: Name of the agent
            action: Action being performed

        Returns:
            Metrics timer for the execution
        """
        return MetricsTimer(
            self.metrics_client.record_agent_execution,
            agent_name=agent_name,
            action=action
        )

    def start_tool_execution(self, tool_name: str, tool_type: str) -> 'MetricsTimer':
        """
        Start timing a tool execution.

        Args:
            tool_name: Name of the tool
            tool_type: Type of tool

        Returns:
            Metrics timer for the execution
        """
        return MetricsTimer(
            self.metrics_client.record_tool_execution,
            tool_name=tool_name,
            tool_type=tool_type
        )

    def start_workflow_execution(self, workflow_name: str) -> 'MetricsTimer':
        """
        Start timing a workflow execution.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Metrics timer for the execution
        """
        return MetricsTimer(
            self.metrics_client.record_workflow_execution,
            workflow_name=workflow_name
        )

    def start_gateway_request(self, gateway_type: str, operation: str) -> 'MetricsTimer':
        """
        Start timing a gateway request.

        Args:
            gateway_type: Type of gateway
            operation: Operation being performed

        Returns:
            Metrics timer for the request
        """
        return MetricsTimer(
            self.metrics_client.record_gateway_request,
            gateway_type=gateway_type,
            operation=operation
        )

    def increment_counter(self, name: str, value: int = 1, **labels: Any) -> None:
        """
        Increment a custom counter.

        Args:
            name: Counter name
            value: Value to increment by
            **labels: Counter labels
        """
        self._counters[name] += value


class MetricsTimer:
    """Context manager for timing operations and recording metrics."""

    def __init__(self, record_func: callable, **kwargs: Any):
        """
        Initialize metrics timer.

        Args:
            record_func: Function to call with timing results
            **kwargs: Arguments to pass to record function
        """
        self.record_func = record_func
        self.kwargs = kwargs
        self.start_time: Optional[float] = None
        self.status = "success"

    def __enter__(self) -> 'MetricsTimer':
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record metrics."""
        if self.start_time is None:
            return

        duration = time.time() - self.start_time

        # Set status based on exception
        if exc_type is not None:
            self.status = "failure"

        try:
            self.record_func(
                duration_seconds=duration,
                status=self.status,
                **self.kwargs
            )
        except Exception as e:
            logger = get_logger("metrics.timer")
            logger.warning(f"Failed to record metrics: {e}")

    def set_status(self, status: str) -> None:
        """
        Set the operation status.

        Args:
            status: Status to set
        """
        self.status = status


# Global metrics client instance
_global_metrics_client: Optional[MetricsClient] = None


def get_metrics_client() -> MetricsClient:
    """
    Get the global metrics client instance.

    Returns:
        Metrics client instance
    """
    global _global_metrics_client

    # Ensure metrics are configured
    if not _metrics_configured:
        configure_metrics()

    if _global_metrics_client is None:
        _global_metrics_client = MetricsClient()

    return _global_metrics_client


class MetricsMixin:
    """Mixin class to add metrics capabilities to other classes."""

    @property
    def metrics_client(self) -> MetricsClient:
        """Get metrics client for this class."""
        return get_metrics_client()

    @property
    def metrics_collector(self) -> MetricsCollector:
        """Get metrics collector for this class."""
        return MetricsCollector(self.metrics_client)


# Utility functions for common metrics patterns

def time_agent_execution(agent_name: str, action: str):
    """
    Decorator for timing agent execution.

    Args:
        agent_name: Name of the agent
        action: Action being performed

    Returns:
        Context manager for timing
    """
    metrics_client = get_metrics_client()
    collector = MetricsCollector(metrics_client)
    return collector.start_agent_execution(agent_name, action)


def time_tool_execution(tool_name: str, tool_type: str):
    """
    Decorator for timing tool execution.

    Args:
        tool_name: Name of the tool
        tool_type: Type of tool

    Returns:
        Context manager for timing
    """
    metrics_client = get_metrics_client()
    collector = MetricsCollector(metrics_client)
    return collector.start_tool_execution(tool_name, tool_type)


def time_workflow_execution(workflow_name: str):
    """
    Decorator for timing workflow execution.

    Args:
        workflow_name: Name of the workflow

    Returns:
        Context manager for timing
    """
    metrics_client = get_metrics_client()
    collector = MetricsCollector(metrics_client)
    return collector.start_workflow_execution(workflow_name)


def time_gateway_request(gateway_type: str, operation: str):
    """
    Decorator for timing gateway requests.

    Args:
        gateway_type: Type of gateway
        operation: Operation being performed

    Returns:
        Context manager for timing
    """
    metrics_client = get_metrics_client()
    collector = MetricsCollector(metrics_client)
    return collector.start_gateway_request(gateway_type, operation)