"""
Core exceptions for the Golden Agent Framework.
"""


class AgentFrameworkError(Exception):
    """Base exception for all Golden Agent Framework errors."""

    pass


class ConfigurationError(AgentFrameworkError):
    """Raised when there's an error in configuration."""

    pass


class GatewayError(AgentFrameworkError):
    """Raised when there's an error communicating with gateways."""

    pass


class MemoryError(AgentFrameworkError):
    """Raised when there's an error with memory operations."""

    pass


class ObservabilityError(AgentFrameworkError):
    """Raised when there's an error with observability components."""

    pass


class ToolError(AgentFrameworkError):
    """Raised when there's an error with tool execution."""

    pass


class AgentError(AgentFrameworkError):
    """Raised when there's an error with agent execution."""

    pass


class RAGError(AgentFrameworkError):
    """Raised when there's an error with RAG operations."""

    pass