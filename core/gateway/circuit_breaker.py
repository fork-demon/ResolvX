"""
Circuit breaker implementation for gateway resilience.

Provides circuit breaker patterns to handle gateway failures gracefully
and prevent cascading failures in the agent system.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from core.exceptions import GatewayError
from core.observability import get_logger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting against gateway failures.

    Implements the circuit breaker pattern to:
    - Fail fast when services are down
    - Allow automatic recovery testing
    - Prevent cascading failures
    - Provide fallback mechanisms
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: int = 60,
        expected_exception: type = GatewayError,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes needed to close circuit
            timeout: Time in seconds before attempting recovery
            expected_exception: Exception type that triggers circuit opening
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None

        self.logger = get_logger(f"circuit_breaker.{name}")

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            GatewayError: If circuit is open or function fails
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._half_open()
            else:
                raise GatewayError(
                    f"Circuit breaker {self.name} is OPEN. Service unavailable."
                )

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Handle success
            self._on_success()
            return result

        except self.expected_exception as e:
            # Handle expected failure
            self._on_failure()
            raise

        except Exception as e:
            # Handle unexpected failure
            self.logger.warning(f"Unexpected exception in circuit breaker {self.name}: {e}")
            raise

    def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            self.logger.debug(
                f"Circuit breaker {self.name}: Success {self.success_count}/{self.success_threshold}"
            )

            if self.success_count >= self.success_threshold:
                self._close()

    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        self.logger.warning(
            f"Circuit breaker {self.name}: Failure {self.failure_count}/{self.failure_threshold}"
        )

        if self.state == CircuitState.HALF_OPEN:
            self._open()
        elif self.failure_count >= self.failure_threshold:
            self._open()

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return False

        return time.time() - self.last_failure_time >= self.timeout

    def _close(self) -> None:
        """Close the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} CLOSED")

    def _open(self) -> None:
        """Open the circuit breaker."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.logger.warning(f"Circuit breaker {self.name} OPENED")

    def _half_open(self) -> None:
        """Set circuit breaker to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} HALF-OPEN")

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self.state == CircuitState.HALF_OPEN

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "last_failure_time": self.last_failure_time,
            "timeout": self.timeout,
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._close()
        self.logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.

    Provides centralized management of circuit breakers for different
    services and components.
    """

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_logger("circuit_breaker.manager")

    def create_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: int = 60,
        expected_exception: type = GatewayError,
    ) -> CircuitBreaker:
        """
        Create or get a circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening
            success_threshold: Number of successes needed to close
            timeout: Timeout before attempting recovery
            expected_exception: Exception type that triggers opening

        Returns:
            Circuit breaker instance
        """
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=timeout,
                expected_exception=expected_exception,
            )
            self.logger.debug(f"Created circuit breaker: {name}")

        return self._circuit_breakers[name]

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._circuit_breakers.get(name)

    def remove_circuit_breaker(self, name: str) -> bool:
        """Remove a circuit breaker."""
        if name in self._circuit_breakers:
            del self._circuit_breakers[name]
            self.logger.debug(f"Removed circuit breaker: {name}")
            return True
        return False

    def list_circuit_breakers(self) -> Dict[str, Dict[str, Any]]:
        """List all circuit breakers and their stats."""
        return {name: cb.get_stats() for name, cb in self._circuit_breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for circuit_breaker in self._circuit_breakers.values():
            circuit_breaker.reset()
        self.logger.info("All circuit breakers reset")

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all circuit breakers."""
        total = len(self._circuit_breakers)
        open_count = sum(1 for cb in self._circuit_breakers.values() if cb.is_open)
        half_open_count = sum(1 for cb in self._circuit_breakers.values() if cb.is_half_open)
        closed_count = sum(1 for cb in self._circuit_breakers.values() if cb.is_closed)

        health_status = "healthy"
        if open_count > 0:
            health_status = "degraded"
        if open_count >= total / 2:
            health_status = "unhealthy"

        return {
            "overall_status": health_status,
            "total_circuit_breakers": total,
            "closed": closed_count,
            "half_open": half_open_count,
            "open": open_count,
            "circuit_breakers": self.list_circuit_breakers(),
        }


# Global circuit breaker manager instance
circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: int = 60,
    expected_exception: type = GatewayError,
) -> CircuitBreaker:
    """
    Get or create a circuit breaker.

    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        success_threshold: Number of successes needed to close
        timeout: Timeout before attempting recovery
        expected_exception: Exception type that triggers opening

    Returns:
        Circuit breaker instance
    """
    return circuit_breaker_manager.create_circuit_breaker(
        name=name,
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exception=expected_exception,
    )


async def with_circuit_breaker(
    name: str,
    func: Callable,
    *args: Any,
    fallback: Optional[Callable] = None,
    circuit_breaker_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Execute a function with circuit breaker protection.

    Args:
        name: Circuit breaker name
        func: Function to execute
        *args: Function arguments
        fallback: Optional fallback function
        circuit_breaker_config: Optional circuit breaker configuration
        **kwargs: Function keyword arguments

    Returns:
        Function result or fallback result

    Raises:
        GatewayError: If circuit is open and no fallback provided
    """
    config = circuit_breaker_config or {}
    circuit_breaker = get_circuit_breaker(name, **config)

    try:
        return await circuit_breaker.call(func, *args, **kwargs)
    except GatewayError as e:
        if fallback is not None:
            logger = get_logger("circuit_breaker")
            logger.info(f"Circuit breaker {name} failed, using fallback")

            if asyncio.iscoroutinefunction(fallback):
                return await fallback(*args, **kwargs)
            else:
                return fallback(*args, **kwargs)
        else:
            raise