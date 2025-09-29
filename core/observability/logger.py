"""
Structured logging for the Golden Agent Framework.

Provides structured logging with multiple backends including console,
file, and cloud-based logging services.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from core.config import Config

# Global logger configuration
_logger_configured = False
_current_config: Optional[Dict[str, Any]] = None


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "getMessage"
            }:
                log_entry[key] = value

        return json.dumps(log_entry)


class StructuredFormatter(logging.Formatter):
    """Human-readable structured formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in a structured but readable way."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        log_parts = [
            f"[{timestamp}]",
            f"[{record.levelname:8}]",
            f"[{record.name}]",
            record.getMessage(),
        ]

        # Add exception info if present
        if record.exc_info:
            log_parts.append(f"\n{self.formatException(record.exc_info)}")

        return " ".join(log_parts)


def configure_logging(config: Optional[Config] = None) -> None:
    """
    Configure structured logging for the framework.

    Args:
        config: Framework configuration
    """
    global _logger_configured, _current_config

    if config is None:
        # Use default configuration
        observability_config = {
            "backend": "console",
            "console": {
                "level": "INFO",
                "format": "structured",
                "include_timestamps": True,
            }
        }
    else:
        observability_config = config.observability.dict()

    # Don't reconfigure if already configured with the same settings
    if _logger_configured and _current_config == observability_config:
        return

    backend = observability_config.get("backend", "console")

    if backend == "console":
        _configure_console_logging(observability_config.get("console", {}))
    elif backend == "langsmith":
        _configure_langsmith_logging(observability_config.get("langsmith", {}))
    elif backend == "langfuse":
        _configure_langfuse_logging(observability_config.get("langfuse", {}))
    else:
        # Default to console
        _configure_console_logging({})

    _logger_configured = True
    _current_config = observability_config


def _configure_console_logging(console_config: Dict[str, Any]) -> None:
    """Configure console-based logging."""
    level = console_config.get("level", "INFO").upper()
    format_type = console_config.get("format", "structured")
    include_timestamps = console_config.get("include_timestamps", True)

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO") if include_timestamps else structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
        formatter = JSONFormatter()
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
        formatter = StructuredFormatter()

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level))
    root_logger.addHandler(console_handler)


def _configure_langsmith_logging(langsmith_config: Dict[str, Any]) -> None:
    """Configure LangSmith-based logging."""
    # For now, fall back to console logging
    # LangSmith integration would be implemented here
    _configure_console_logging({
        "level": "DEBUG",
        "format": "json",
        "include_timestamps": True,
    })


def _configure_langfuse_logging(langfuse_config: Dict[str, Any]) -> None:
    """Configure Langfuse-based logging."""
    # For now, fall back to console logging
    # Langfuse integration would be implemented here
    _configure_console_logging({
        "level": "DEBUG",
        "format": "json",
        "include_timestamps": True,
    })


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically module name)

    Returns:
        Structured logger instance
    """
    # Ensure logging is configured
    if not _logger_configured:
        configure_logging()

    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """
    Log a function call with parameters.

    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger("function_calls")
    logger.debug("Function called", function=func_name, parameters=kwargs)


def log_agent_action(
    agent_name: str,
    action: str,
    input_data: Any = None,
    output_data: Any = None,
    duration_ms: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None,
    **metadata: Any
) -> None:
    """
    Log an agent action with structured data.

    Args:
        agent_name: Name of the agent
        action: Action being performed
        input_data: Input data for the action
        output_data: Output data from the action
        duration_ms: Action duration in milliseconds
        success: Whether the action succeeded
        error: Error message if action failed
        **metadata: Additional metadata
    """
    logger = get_logger("agent_actions")

    log_data = {
        "agent": agent_name,
        "action": action,
        "success": success,
        **metadata
    }

    if input_data is not None:
        log_data["input"] = _sanitize_log_data(input_data)

    if output_data is not None:
        log_data["output"] = _sanitize_log_data(output_data)

    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    if error:
        log_data["error"] = error

    if success:
        logger.info("Agent action completed", **log_data)
    else:
        logger.error("Agent action failed", **log_data)


def log_tool_execution(
    tool_name: str,
    parameters: Dict[str, Any],
    result: Any = None,
    duration_ms: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None,
    **metadata: Any
) -> None:
    """
    Log tool execution with structured data.

    Args:
        tool_name: Name of the tool
        parameters: Tool parameters
        result: Tool execution result
        duration_ms: Execution duration in milliseconds
        success: Whether execution succeeded
        error: Error message if execution failed
        **metadata: Additional metadata
    """
    logger = get_logger("tool_execution")

    log_data = {
        "tool": tool_name,
        "parameters": _sanitize_log_data(parameters),
        "success": success,
        **metadata
    }

    if result is not None:
        log_data["result"] = _sanitize_log_data(result)

    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    if error:
        log_data["error"] = error

    if success:
        logger.info("Tool execution completed", **log_data)
    else:
        logger.error("Tool execution failed", **log_data)


def log_workflow_event(
    workflow_id: str,
    event_type: str,
    agent_name: Optional[str] = None,
    phase: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    **metadata: Any
) -> None:
    """
    Log workflow events for tracking multi-agent coordination.

    Args:
        workflow_id: Unique workflow identifier
        event_type: Type of event (started, completed, failed, etc.)
        agent_name: Name of agent if event is agent-specific
        phase: Current workflow phase
        data: Event data
        **metadata: Additional metadata
    """
    logger = get_logger("workflow_events")

    log_data = {
        "workflow_id": workflow_id,
        "event_type": event_type,
        **metadata
    }

    if agent_name:
        log_data["agent"] = agent_name

    if phase:
        log_data["phase"] = phase

    if data:
        log_data["data"] = _sanitize_log_data(data)

    logger.info("Workflow event", **log_data)


def _sanitize_log_data(data: Any, max_length: int = 1000) -> Any:
    """
    Sanitize data for logging by removing sensitive information and limiting size.

    Args:
        data: Data to sanitize
        max_length: Maximum string length

    Returns:
        Sanitized data
    """
    if data is None:
        return None

    if isinstance(data, str):
        # Truncate long strings
        if len(data) > max_length:
            return data[:max_length] + "..."
        return data

    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Skip sensitive keys
            if any(sensitive in key.lower() for sensitive in ["password", "token", "secret", "key", "auth"]):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _sanitize_log_data(value, max_length)
        return sanitized

    if isinstance(data, (list, tuple)):
        return [_sanitize_log_data(item, max_length) for item in data[:10]]  # Limit list size

    if isinstance(data, (int, float, bool)):
        return data

    # For other types, convert to string and truncate
    str_data = str(data)
    if len(str_data) > max_length:
        return str_data[:max_length] + "..."
    return str_data


# Context managers for structured logging

class LogContext:
    """Context manager for adding structured context to logs."""

    def __init__(self, logger: structlog.BoundLogger, **context: Any):
        """
        Initialize log context.

        Args:
            logger: Logger instance
            **context: Context data to add
        """
        self.logger = logger
        self.context = context
        self.bound_logger: Optional[structlog.BoundLogger] = None

    def __enter__(self) -> structlog.BoundLogger:
        """Enter context and return bound logger."""
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass


def log_context(logger: structlog.BoundLogger, **context: Any) -> LogContext:
    """
    Create a log context for adding structured context.

    Args:
        logger: Logger instance
        **context: Context data to add

    Returns:
        Log context manager
    """
    return LogContext(logger, **context)