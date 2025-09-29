"""
Observability factory for creating and configuring observability components.

Provides a factory pattern for setting up logging, tracing, and metrics
collection based on configuration settings.
"""

from typing import Any, Dict, Optional

from core.config import Config, ObservabilityConfig
from core.exceptions import ConfigurationError
from core.observability.logger import configure_logging, get_logger
from core.observability.tracer import configure_tracing, get_tracer
from core.observability.metrics import configure_metrics, get_metrics_client
from core.observability.logger import LoggerMixin
from core.observability.tracer import TracerMixin
from core.observability.metrics import MetricsMixin
from core.observability.central_forwarder import CentralForwarder, ForwarderConfig
from core.observability.langsmith_tracer import LangSmithTracer, LangSmithConfig
from core.observability.langfuse_tracer import LangFuseTracer, LangFuseConfig
from core.observability.console_tracer import ConsoleTracer, ConsoleConfig


class ObservabilityFactory:
    """
    Factory for creating and configuring observability components.

    Provides centralized setup of logging, tracing, and metrics collection
    based on configuration settings.
    """

    def __init__(self):
        """Initialize the observability factory."""
        self.logger = get_logger("observability.factory")

    def configure_all(self, config: Optional[Config] = None) -> None:
        """
        Configure all observability components.

        Args:
            config: Framework configuration
        """
        try:
            # Configure logging first (needed for other components)
            configure_logging(config)

            # Configure tracing
            configure_tracing(config)

            # Configure metrics
            configure_metrics(config)

            self.logger.info("All observability components configured successfully")

        except Exception as e:
            print(f"Failed to configure observability: {e}")  # Use print since logging might not be ready
            raise ConfigurationError(f"Observability configuration failed: {e}") from e

    def configure_from_observability_config(self, observability_config: ObservabilityConfig) -> None:
        """
        Configure observability from observability-specific configuration.

        Args:
            observability_config: Observability configuration
        """
        # Create a minimal config object
        config = Config(observability=observability_config)
        self.configure_all(config)

    def get_observability_components(self, config: Optional[Config] = None) -> Dict[str, Any]:
        """
        Get all configured observability components.

        Args:
            config: Framework configuration

        Returns:
            Dictionary containing logger, tracer, and metrics client
        """
        # Ensure all components are configured
        self.configure_all(config)

        return {
            "logger": get_logger("framework"),
            "tracer": get_tracer("framework"),
            "metrics": get_metrics_client(),
        }

    def create_central_forwarder(self, config: Optional[Config] = None) -> CentralForwarder:
        """
        Create a central observability forwarder.

        Args:
            config: Framework configuration

        Returns:
            Central forwarder instance
        """
        if config and config.observability:
            forwarder_config = ForwarderConfig(
                enabled=config.observability.enabled,
                backends=config.observability.backends or ["console"],
                batch_size=config.observability.batch_size or 100,
                flush_interval=config.observability.flush_interval or 30.0,
                timeout=config.observability.timeout or 30.0,
                retry_attempts=config.observability.retry_attempts or 3,
                retry_delay=config.observability.retry_delay or 1.0,
            )
        else:
            forwarder_config = ForwarderConfig()

        return CentralForwarder(forwarder_config)

    def create_langsmith_tracer(self, config: Optional[Config] = None) -> LangSmithTracer:
        """
        Create a LangSmith tracer.

        Args:
            config: Framework configuration

        Returns:
            LangSmith tracer instance
        """
        if config and config.observability and config.observability.langsmith:
            langsmith_config = LangSmithConfig(
                api_url=config.observability.langsmith.get("api_url", "https://api.smith.langchain.com"),
                api_key=config.observability.langsmith.get("api_key"),
                project_name=config.observability.langsmith.get("project_name", "golden-agents"),
                environment=config.observability.langsmith.get("environment", "development"),
                enabled=config.observability.langsmith.get("enabled", True),
                batch_size=config.observability.langsmith.get("batch_size", 100),
                flush_interval=config.observability.langsmith.get("flush_interval", 30.0),
                timeout=config.observability.langsmith.get("timeout", 30.0),
            )
        else:
            langsmith_config = LangSmithConfig()

        return LangSmithTracer(langsmith_config)

    def create_langfuse_tracer(self, config: Optional[Config] = None) -> LangFuseTracer:
        """
        Create a LangFuse tracer.

        Args:
            config: Framework configuration

        Returns:
            LangFuse tracer instance
        """
        if config and config.observability and config.observability.langfuse:
            langfuse_config = LangFuseConfig(
                public_key=config.observability.langfuse.get("public_key"),
                secret_key=config.observability.langfuse.get("secret_key"),
                host=config.observability.langfuse.get("host", "https://cloud.langfuse.com"),
                project_name=config.observability.langfuse.get("project_name", "golden-agents"),
                environment=config.observability.langfuse.get("environment", "development"),
                enabled=config.observability.langfuse.get("enabled", True),
                batch_size=config.observability.langfuse.get("batch_size", 100),
                flush_interval=config.observability.langfuse.get("flush_interval", 30.0),
                timeout=config.observability.langfuse.get("timeout", 30.0),
            )
        else:
            langfuse_config = LangFuseConfig()

        return LangFuseTracer(langfuse_config)

    def create_console_tracer(self, config: Optional[Config] = None) -> ConsoleTracer:
        """
        Create a console tracer.

        Args:
            config: Framework configuration

        Returns:
            Console tracer instance
        """
        if config and config.observability and config.observability.console:
            console_config = ConsoleConfig(
                enabled=config.observability.console.get("enabled", True),
                log_level=config.observability.console.get("level", "INFO"),
                include_timestamps=config.observability.console.get("include_timestamps", True),
                include_metadata=config.observability.console.get("include_metadata", True),
                format=config.observability.console.get("format", "structured"),
                colorize=config.observability.console.get("colorize", True),
            )
        else:
            console_config = ConsoleConfig()

        return ConsoleTracer(console_config)


class ObservabilityMixin(LoggerMixin, TracerMixin, MetricsMixin):
    """
    Combined mixin class that provides all observability capabilities.

    Inherits from LoggerMixin, TracerMixin, and MetricsMixin to provide
    comprehensive observability features to any class.
    """

    def __init__(self, *args, **kwargs):
        """Initialize observability mixin."""
        super().__init__(*args, **kwargs)

        # Ensure observability is configured
        if not hasattr(self, '_observability_configured'):
            try:
                # Try to configure with default settings
                factory = ObservabilityFactory()
                factory.configure_all(None)
                self._observability_configured = True
            except Exception as e:
                # Log warning but don't fail
                print(f"Warning: Failed to auto-configure observability: {e}")
                self._observability_configured = False

    def configure_observability(self, config: Optional[Config] = None) -> None:
        """
        Configure observability for this instance.

        Args:
            config: Framework configuration
        """
        factory = ObservabilityFactory()
        factory.configure_all(config)
        self._observability_configured = True

    @property
    def is_observability_configured(self) -> bool:
        """Check if observability is configured."""
        return getattr(self, '_observability_configured', False)


# Global factory instance
observability_factory = ObservabilityFactory()


def create_observability(config: Optional[Config] = None) -> Dict[str, Any]:
    """
    Create and configure all observability components.

    Args:
        config: Framework configuration

    Returns:
        Dictionary containing logger, tracer, and metrics client
    """
    return observability_factory.get_observability_components(config)


def configure_observability(config: Optional[Config] = None) -> None:
    """
    Configure all observability components using the global factory.

    Args:
        config: Framework configuration
    """
    observability_factory.configure_all(config)


# Convenience functions for quick setup

def setup_console_observability(
    log_level: str = "INFO",
    metrics_port: int = 8080,
    tracing_enabled: bool = True
) -> Dict[str, Any]:
    """
    Quick setup for console-based observability.

    Args:
        log_level: Logging level
        metrics_port: Port for metrics server
        tracing_enabled: Whether to enable tracing

    Returns:
        Dictionary containing observability components
    """
    from core.config import ObservabilityConfig

    observability_config = ObservabilityConfig(
        backend="console",
        console={
            "level": log_level,
            "format": "structured",
            "include_timestamps": True,
            "port": metrics_port,
            "enabled": tracing_enabled,
        }
    )

    config = Config(observability=observability_config)
    return create_observability(config)


def setup_production_observability(
    backend: str = "langsmith",
    api_key: Optional[str] = None,
    project_name: str = "golden-agents",
    metrics_port: int = 8080
) -> Dict[str, Any]:
    """
    Quick setup for production observability.

    Args:
        backend: Observability backend (langsmith, langfuse)
        api_key: API key for the backend service
        project_name: Project name for organization
        metrics_port: Port for metrics server

    Returns:
        Dictionary containing observability components
    """
    from core.config import ObservabilityConfig

    backend_config = {
        "api_key": api_key,
        "project_name": project_name,
        "port": metrics_port,
        "enabled": True,
    }

    observability_config = ObservabilityConfig(
        backend=backend,
        **{backend: backend_config}
    )

    config = Config(observability=observability_config)
    return create_observability(config)


# Health check utilities

def check_observability_health() -> Dict[str, Any]:
    """
    Check the health of all observability components.

    Returns:
        Health status dictionary
    """
    health_status = {
        "overall_status": "healthy",
        "components": {}
    }

    # Check logging
    try:
        logger = get_logger("health_check")
        logger.debug("Logging health check")
        health_status["components"]["logging"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["logging"] = {"status": "unhealthy", "error": str(e)}
        health_status["overall_status"] = "degraded"

    # Check tracing
    try:
        tracer = get_tracer("health_check")
        with tracer.start_as_current_span("health_check") as span:
            span.set_attribute("check_type", "health")
        health_status["components"]["tracing"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["tracing"] = {"status": "unhealthy", "error": str(e)}
        health_status["overall_status"] = "degraded"

    # Check metrics
    try:
        metrics_client = get_metrics_client()
        metrics_client.create_counter("health_check_counter", "Health check counter").inc()
        health_status["components"]["metrics"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["metrics"] = {"status": "unhealthy", "error": str(e)}
        health_status["overall_status"] = "degraded"

    return health_status


# Development utilities

def setup_development_observability(verbose: bool = True) -> Dict[str, Any]:
    """
    Setup observability for development with verbose logging.

    Args:
        verbose: Whether to enable verbose logging

    Returns:
        Dictionary containing observability components
    """
    log_level = "DEBUG" if verbose else "INFO"

    return setup_console_observability(
        log_level=log_level,
        metrics_port=8080,
        tracing_enabled=True
    )


def setup_testing_observability() -> Dict[str, Any]:
    """
    Setup minimal observability for testing.

    Returns:
        Dictionary containing observability components
    """
    return setup_console_observability(
        log_level="WARNING",  # Minimal logging for tests
        metrics_port=0,  # Don't start metrics server
        tracing_enabled=False  # Disable tracing for tests
    )


# Context manager for temporary observability configuration

class TemporaryObservabilityConfig:
    """Context manager for temporary observability configuration."""

    def __init__(self, config: Config):
        """
        Initialize temporary configuration.

        Args:
            config: Temporary configuration to apply
        """
        self.config = config
        self.original_config: Optional[Dict[str, Any]] = None

    def __enter__(self) -> Dict[str, Any]:
        """Apply temporary configuration."""
        # Store current configuration state (simplified)
        self.original_config = {}

        # Apply new configuration
        configure_observability(self.config)

        return create_observability(self.config)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original configuration."""
        # In a full implementation, we would restore the original configuration
        # For now, we'll just note that this would need to be implemented
        pass


def temporary_observability_config(config: Config) -> TemporaryObservabilityConfig:
    """
    Create a temporary observability configuration context.

    Args:
        config: Temporary configuration to apply

    Returns:
        Context manager for temporary configuration
    """
    return TemporaryObservabilityConfig(config)