"""
Golden Agent Framework - Core Module

A pluggable, central-gateway-aware framework for building and deploying AI agents.
"""

__version__ = "0.1.0"

from core.config import Config, load_config
from core.exceptions import (
    AgentFrameworkError,
    ConfigurationError,
    GatewayError,
    MemoryError,
    ObservabilityError,
)

__all__ = [
    "Config",
    "load_config",
    "AgentFrameworkError",
    "ConfigurationError",
    "GatewayError",
    "MemoryError",
    "ObservabilityError",
]