"""
Memory factory for creating memory backend instances.

Provides a factory pattern for creating and configuring different
memory backend implementations based on configuration.
"""

from typing import Any, Dict, Optional, Type

from core.config import Config, MemoryConfig
from core.exceptions import ConfigurationError, MemoryError
from core.memory.base import BaseMemory
from core.memory.faiss_memory import FAISSMemory
from core.observability import get_logger


class MemoryFactory:
    """
    Factory for creating memory backend instances.

    Provides centralized creation and configuration of memory backends
    based on configuration settings.
    """

    # Registry of available memory backends
    # FAISS is the production-ready vector memory implementation
    _backends: Dict[str, Type[BaseMemory]] = {
        "faiss": FAISSMemory,
    }

    def __init__(self):
        """Initialize the memory factory."""
        self.logger = get_logger("memory.factory")

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[BaseMemory]) -> None:
        """
        Register a custom memory backend.

        Args:
            name: Backend name
            backend_class: Backend implementation class
        """
        cls._backends[name] = backend_class

    @classmethod
    def list_backends(cls) -> list[str]:
        """
        List available memory backend names.

        Returns:
            List of backend names
        """
        return list(cls._backends.keys())

    def create_memory(
        self,
        backend: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseMemory:
        """
        Create a memory backend instance.

        Args:
            backend: Backend type name
            config: Backend-specific configuration

        Returns:
            Configured memory backend instance

        Raises:
            MemoryError: If backend creation fails
            ConfigurationError: If backend is not supported
        """
        if backend not in self._backends:
            available = ", ".join(self._backends.keys())
            raise ConfigurationError(
                f"Unsupported memory backend '{backend}'. "
                f"Available backends: {available}"
            )

        try:
            backend_class = self._backends[backend]
            instance = backend_class(config=config or {})

            self.logger.info(f"Created {backend} memory backend")
            return instance

        except Exception as e:
            self.logger.error(f"Failed to create {backend} memory backend: {e}")
            raise MemoryError(f"Memory backend creation failed: {e}") from e

    def create_from_config(self, memory_config: MemoryConfig) -> BaseMemory:
        """
        Create a memory backend from configuration.

        Args:
            memory_config: Memory configuration

        Returns:
            Configured memory backend instance

        Raises:
            MemoryError: If backend creation fails
        """
        backend_name = memory_config.backend

        # Get backend-specific configuration
        backend_config = getattr(memory_config, backend_name, {})
        if isinstance(backend_config, dict):
            config = backend_config
        else:
            # Convert Pydantic model to dict
            config = backend_config.dict() if backend_config else {}

        return self.create_memory(backend_name, config)


# Global factory instance
memory_factory = MemoryFactory()


def create_memory(
    backend: str,
    config: Optional[Dict[str, Any]] = None,
) -> BaseMemory:
    """
    Create a memory backend instance using the global factory.

    Args:
        backend: Backend type name
        config: Backend-specific configuration

    Returns:
        Configured memory backend instance

    Raises:
        MemoryError: If backend creation fails
        ConfigurationError: If backend is not supported
    """
    return memory_factory.create_memory(backend, config)


def create_memory_from_config(config: Config) -> BaseMemory:
    """
    Create a memory backend from framework configuration.

    Args:
        config: Framework configuration

    Returns:
        Configured memory backend instance

    Raises:
        MemoryError: If backend creation fails
    """
    return memory_factory.create_from_config(config.memory)


async def initialize_memory_from_config(config: Config) -> BaseMemory:
    """
    Create and initialize a memory backend from configuration.

    Args:
        config: Framework configuration

    Returns:
        Initialized memory backend instance

    Raises:
        MemoryError: If backend creation or initialization fails
    """
    memory = create_memory_from_config(config)
    await memory.initialize()
    return memory