"""
Observability components for the Golden Agent Framework.

This module provides pluggable observability backends including logging,
metrics, tracing, and monitoring for agents and workflows.
"""

from core.observability.logger import get_logger, configure_logging
from core.observability.tracer import get_tracer, configure_tracing
from core.observability.metrics import get_metrics_client, configure_metrics
from core.observability.factory import ObservabilityFactory, create_observability

__all__ = [
    "get_logger",
    "configure_logging",
    "get_tracer",
    "configure_tracing",
    "get_metrics_client",
    "configure_metrics",
    "ObservabilityFactory",
    "create_observability",
]