"""
Graph executor for running LangGraph workflows with monitoring and error handling.

Provides execution management including streaming, checkpointing, monitoring,
and human-in-the-loop capabilities.
"""

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from uuid import UUID, uuid4

from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.base import Checkpoint

from core.config import Config
from core.exceptions import AgentError
from core.graph.state import AgentState, WorkflowState
from core.observability import get_logger, get_tracer


class GraphExecutor:
    """
    Executor for running LangGraph workflows with comprehensive monitoring and control.

    Provides execution management, monitoring, checkpointing, and human-in-the-loop
    capabilities for agent workflows.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the graph executor.

        Args:
            config: Framework configuration
        """
        self.config = config
        self.logger = get_logger("graph.executor")
        self.tracer = get_tracer("graph.executor")

        # Execution tracking
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[Dict[str, Any]] = []

        # Human-in-the-loop management
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

    async def execute(
        self,
        graph: CompiledGraph,
        input_data: Union[Dict[str, Any], AgentState, WorkflowState],
        config: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a graph with full monitoring and error handling.

        Args:
            graph: Compiled LangGraph to execute
            input_data: Input data or state
            config: Optional runtime configuration
            execution_id: Optional execution identifier

        Returns:
            Execution results

        Raises:
            AgentError: If execution fails
        """
        execution_id = execution_id or str(uuid4())

        try:
            # Start execution tracking
            execution_info = self._start_execution_tracking(execution_id, input_data)

            with self.tracer.start_as_current_span(f"graph_execution_{execution_id}") as span:
                span.set_attribute("execution_id", execution_id)
                span.set_attribute("graph_type", type(graph).__name__)

                # Execute the graph
                result = await graph.ainvoke(input_data, config=config)

                # Complete execution tracking
                self._complete_execution_tracking(execution_id, result, "completed")

                self.logger.info(f"Graph execution {execution_id} completed successfully")
                return result

        except Exception as e:
            # Handle execution failure
            self._complete_execution_tracking(execution_id, None, "failed", str(e))
            self.logger.error(f"Graph execution {execution_id} failed: {e}")
            raise AgentError(f"Graph execution failed: {e}") from e

    async def stream(
        self,
        graph: CompiledGraph,
        input_data: Union[Dict[str, Any], AgentState, WorkflowState],
        config: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream graph execution results.

        Args:
            graph: Compiled LangGraph to execute
            input_data: Input data or state
            config: Optional runtime configuration
            execution_id: Optional execution identifier

        Yields:
            Streaming execution results

        Raises:
            AgentError: If streaming fails
        """
        execution_id = execution_id or str(uuid4())

        try:
            # Start execution tracking
            execution_info = self._start_execution_tracking(input_data, execution_id)

            with self.tracer.start_as_current_span(f"graph_stream_{execution_id}") as span:
                span.set_attribute("execution_id", execution_id)
                span.set_attribute("streaming", True)

                # Stream the graph execution
                async for chunk in graph.astream(input_data, config=config):
                    # Update execution tracking
                    self._update_execution_tracking(execution_id, chunk)

                    # Yield the chunk
                    yield chunk

                # Complete execution tracking
                self._complete_execution_tracking(execution_id, None, "completed")

        except Exception as e:
            self._complete_execution_tracking(execution_id, None, "failed", str(e))
            self.logger.error(f"Graph streaming {execution_id} failed: {e}")
            raise AgentError(f"Graph streaming failed: {e}") from e

    async def execute_with_approval(
        self,
        graph: CompiledGraph,
        input_data: Union[Dict[str, Any], AgentState, WorkflowState],
        approval_callback: callable,
        config: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a graph with human-in-the-loop approval points.

        Args:
            graph: Compiled LangGraph to execute
            input_data: Input data or state
            approval_callback: Function to call for approvals
            config: Optional runtime configuration
            execution_id: Optional execution identifier

        Returns:
            Execution results

        Raises:
            AgentError: If execution fails
        """
        execution_id = execution_id or str(uuid4())

        try:
            # Enhanced config for approval handling
            enhanced_config = config or {}
            enhanced_config["approval_callback"] = approval_callback
            enhanced_config["execution_id"] = execution_id

            # Execute with approval handling
            result = await self.execute(graph, input_data, enhanced_config, execution_id)

            return result

        except Exception as e:
            self.logger.error(f"Approval-based execution {execution_id} failed: {e}")
            raise

    async def pause_execution(self, execution_id: str) -> bool:
        """
        Pause a running execution.

        Args:
            execution_id: ID of execution to pause

        Returns:
            True if paused successfully, False otherwise
        """
        if execution_id in self._active_executions:
            execution_info = self._active_executions[execution_id]
            execution_info["status"] = "paused"
            execution_info["paused_at"] = datetime.utcnow().isoformat()

            self.logger.info(f"Execution {execution_id} paused")
            return True

        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused execution.

        Args:
            execution_id: ID of execution to resume

        Returns:
            True if resumed successfully, False otherwise
        """
        if execution_id in self._active_executions:
            execution_info = self._active_executions[execution_id]
            if execution_info.get("status") == "paused":
                execution_info["status"] = "running"
                execution_info["resumed_at"] = datetime.utcnow().isoformat()

                self.logger.info(f"Execution {execution_id} resumed")
                return True

        return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running execution.

        Args:
            execution_id: ID of execution to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        if execution_id in self._active_executions:
            self._complete_execution_tracking(execution_id, None, "cancelled")
            self.logger.info(f"Execution {execution_id} cancelled")
            return True

        return False

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of an execution.

        Args:
            execution_id: ID of execution to check

        Returns:
            Execution status information or None if not found
        """
        return self._active_executions.get(execution_id)

    async def list_active_executions(self) -> List[Dict[str, Any]]:
        """
        List all currently active executions.

        Returns:
            List of active execution information
        """
        return list(self._active_executions.values())

    async def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get execution history.

        Args:
            limit: Maximum number of historical executions to return

        Returns:
            List of historical execution information
        """
        return self._execution_history[-limit:] if limit > 0 else self._execution_history

    # Approval management

    async def request_approval(
        self,
        execution_id: str,
        agent_name: str,
        action: str,
        context: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> str:
        """
        Request human approval for an action.

        Args:
            execution_id: ID of the execution requesting approval
            agent_name: Name of the agent requesting approval
            action: Description of the action requiring approval
            context: Additional context for the approval decision
            timeout: Optional timeout in seconds

        Returns:
            Approval request ID
        """
        approval_id = str(uuid4())

        approval_request = {
            "id": approval_id,
            "execution_id": execution_id,
            "agent_name": agent_name,
            "action": action,
            "context": context,
            "requested_at": datetime.utcnow().isoformat(),
            "timeout": timeout,
            "status": "pending",
        }

        self._pending_approvals[approval_id] = approval_request

        self.logger.info(
            f"Approval requested: {approval_id} for agent {agent_name} in execution {execution_id}"
        )

        return approval_id

    async def provide_approval(
        self,
        approval_id: str,
        approved: bool,
        feedback: Optional[str] = None,
        approver: Optional[str] = None,
    ) -> bool:
        """
        Provide approval response.

        Args:
            approval_id: ID of the approval request
            approved: Whether the action is approved
            feedback: Optional feedback message
            approver: Optional approver identifier

        Returns:
            True if approval was processed, False if not found
        """
        if approval_id not in self._pending_approvals:
            return False

        approval_request = self._pending_approvals[approval_id]
        approval_request.update(
            {
                "status": "approved" if approved else "denied",
                "approved": approved,
                "feedback": feedback,
                "approver": approver,
                "responded_at": datetime.utcnow().isoformat(),
            }
        )

        # Remove from pending
        del self._pending_approvals[approval_id]

        self.logger.info(
            f"Approval {approval_id} {'approved' if approved else 'denied'} by {approver}"
        )

        return True

    async def list_pending_approvals(self) -> List[Dict[str, Any]]:
        """
        List all pending approval requests.

        Returns:
            List of pending approval requests
        """
        return list(self._pending_approvals.values())

    # Checkpoint management

    async def save_checkpoint(
        self,
        execution_id: str,
        checkpoint_name: str,
        state: Dict[str, Any],
    ) -> bool:
        """
        Save a checkpoint for an execution.

        Args:
            execution_id: ID of the execution
            checkpoint_name: Name for the checkpoint
            state: Current state to save

        Returns:
            True if checkpoint saved successfully
        """
        try:
            if execution_id in self._active_executions:
                execution_info = self._active_executions[execution_id]
                if "checkpoints" not in execution_info:
                    execution_info["checkpoints"] = {}

                execution_info["checkpoints"][checkpoint_name] = {
                    "state": state,
                    "saved_at": datetime.utcnow().isoformat(),
                }

                self.logger.debug(f"Checkpoint {checkpoint_name} saved for execution {execution_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

        return False

    async def load_checkpoint(
        self,
        execution_id: str,
        checkpoint_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for an execution.

        Args:
            execution_id: ID of the execution
            checkpoint_name: Name of the checkpoint to load

        Returns:
            Checkpoint state or None if not found
        """
        if execution_id in self._active_executions:
            execution_info = self._active_executions[execution_id]
            checkpoints = execution_info.get("checkpoints", {})
            checkpoint = checkpoints.get(checkpoint_name)

            if checkpoint:
                self.logger.debug(f"Checkpoint {checkpoint_name} loaded for execution {execution_id}")
                return checkpoint["state"]

        return None

    # Private helper methods

    def _start_execution_tracking(
        self, execution_id: str, input_data: Union[Dict[str, Any], AgentState, WorkflowState]
    ) -> Dict[str, Any]:
        """Start tracking an execution."""
        execution_info = {
            "id": execution_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "input_data": input_data,
            "steps": [],
            "checkpoints": {},
        }

        self._active_executions[execution_id] = execution_info
        self.logger.debug(f"Started execution tracking for {execution_id}")

        return execution_info

    def _update_execution_tracking(self, execution_id: str, step_data: Dict[str, Any]) -> None:
        """Update execution tracking with step data."""
        if execution_id in self._active_executions:
            execution_info = self._active_executions[execution_id]
            execution_info["steps"].append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": step_data,
                }
            )

    def _complete_execution_tracking(
        self,
        execution_id: str,
        result: Optional[Dict[str, Any]],
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Complete execution tracking."""
        if execution_id in self._active_executions:
            execution_info = self._active_executions[execution_id]
            execution_info.update(
                {
                    "status": status,
                    "completed_at": datetime.utcnow().isoformat(),
                    "result": result,
                    "error": error,
                }
            )

            # Move to history
            self._execution_history.append(execution_info.copy())

            # Remove from active
            del self._active_executions[execution_id]

            self.logger.debug(f"Completed execution tracking for {execution_id} with status {status}")

    async def cleanup_old_executions(self, max_history: int = 1000) -> int:
        """
        Clean up old execution history.

        Args:
            max_history: Maximum number of historical executions to keep

        Returns:
            Number of executions cleaned up
        """
        if len(self._execution_history) > max_history:
            cleanup_count = len(self._execution_history) - max_history
            self._execution_history = self._execution_history[-max_history:]
            self.logger.info(f"Cleaned up {cleanup_count} old execution records")
            return cleanup_count

        return 0