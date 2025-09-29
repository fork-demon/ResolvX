"""
State management for LangGraph agents and workflows.

Provides typed state classes that maintain context across agent interactions
and workflow execution steps.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseState(BaseModel):
    """Base state class for all LangGraph states."""

    # Unique identifier for this state instance
    id: UUID = Field(default_factory=uuid4)

    # Timestamp when state was created
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Timestamp when state was last updated
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Correlation ID for tracing across services
    correlation_id: Optional[str] = None

    # Metadata for storing arbitrary key-value pairs
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.updated_at = datetime.utcnow()

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AgentState(BaseState):
    """State for individual agent execution."""

    # Agent identification
    agent_name: str
    agent_type: str
    agent_version: str = "1.0.0"

    # Execution context
    task_description: str = ""
    current_step: int = 0
    max_steps: int = 10
    status: str = "pending"  # pending, running, completed, failed, paused

    # Input and output data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Tool usage tracking
    tools_used: List[str] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)

    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

    # Confidence and quality metrics
    confidence_score: float = 0.0
    quality_score: Optional[float] = None

    # Human-in-the-loop
    requires_approval: bool = False
    approval_status: Optional[str] = None  # pending, approved, rejected
    human_feedback: Optional[str] = None

    def add_error(self, error: str, error_type: str = "general", **kwargs: Any) -> None:
        """Add an error to the state."""
        self.errors.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "error": error,
                "error_type": error_type,
                "step": self.current_step,
                **kwargs,
            }
        )
        self.update_timestamp()

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Add a tool execution result."""
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)

        self.tool_results[tool_name] = {
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "step": self.current_step,
        }
        self.update_timestamp()

    def add_intermediate_result(self, result: Dict[str, Any]) -> None:
        """Add an intermediate result from processing."""
        self.intermediate_results.append(
            {
                **result,
                "timestamp": datetime.utcnow().isoformat(),
                "step": self.current_step,
            }
        )
        self.update_timestamp()

    def increment_step(self) -> None:
        """Move to the next step in execution."""
        self.current_step += 1
        self.update_timestamp()

    def is_complete(self) -> bool:
        """Check if agent execution is complete."""
        return self.status in {"completed", "failed"}

    def should_retry(self) -> bool:
        """Check if agent should retry execution."""
        return (
            self.status == "failed"
            and self.retry_count < self.max_retries
            and len(self.errors) > 0
        )


class WorkflowState(BaseState):
    """State for multi-agent workflow coordination."""

    # Workflow identification
    workflow_name: str
    workflow_version: str = "1.0.0"

    # Execution tracking
    status: str = "pending"  # pending, running, completed, failed, paused
    current_phase: str = ""
    phases: List[str] = Field(default_factory=list)

    # Agent coordination
    active_agents: List[str] = Field(default_factory=list)
    completed_agents: List[str] = Field(default_factory=list)
    failed_agents: List[str] = Field(default_factory=list)

    # Agent states
    agent_states: Dict[str, AgentState] = Field(default_factory=dict)

    # Workflow data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    final_output: Dict[str, Any] = Field(default_factory=dict)

    # Communication between agents
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    pending_approvals: List[str] = Field(default_factory=list)

    # Workflow metrics
    total_agents: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0

    # Escalation and error handling
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    def add_agent_state(self, agent_name: str, agent_state: AgentState) -> None:
        """Add or update an agent's state."""
        self.agent_states[agent_name] = agent_state
        if agent_name not in self.active_agents:
            self.active_agents.append(agent_name)
        self.update_timestamp()

    def mark_agent_completed(self, agent_name: str) -> None:
        """Mark an agent as completed."""
        if agent_name in self.active_agents:
            self.active_agents.remove(agent_name)
        if agent_name not in self.completed_agents:
            self.completed_agents.append(agent_name)
        self.update_timestamp()

    def mark_agent_failed(self, agent_name: str, error: str) -> None:
        """Mark an agent as failed."""
        if agent_name in self.active_agents:
            self.active_agents.remove(agent_name)
        if agent_name not in self.failed_agents:
            self.failed_agents.append(agent_name)

        self.errors.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": agent_name,
                "error": error,
                "phase": self.current_phase,
            }
        )
        self.update_timestamp()

    def add_message(
        self,
        from_agent: str,
        to_agent: Optional[str],
        message: str,
        message_type: str = "info",
        **kwargs: Any,
    ) -> None:
        """Add a message between agents."""
        self.messages.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "from": from_agent,
                "to": to_agent,
                "message": message,
                "type": message_type,
                "phase": self.current_phase,
                **kwargs,
            }
        )
        self.update_timestamp()

    def request_approval(self, agent_name: str, reason: str) -> None:
        """Request human approval for an agent action."""
        if agent_name not in self.pending_approvals:
            self.pending_approvals.append(agent_name)

        if agent_name in self.agent_states:
            self.agent_states[agent_name].requires_approval = True

        self.add_message(
            from_agent=agent_name,
            to_agent="human",
            message=f"Approval requested: {reason}",
            message_type="approval_request",
            reason=reason,
        )

    def provide_approval(
        self, agent_name: str, approved: bool, feedback: Optional[str] = None
    ) -> None:
        """Provide approval response for an agent."""
        if agent_name in self.pending_approvals:
            self.pending_approvals.remove(agent_name)

        if agent_name in self.agent_states:
            self.agent_states[agent_name].approval_status = (
                "approved" if approved else "rejected"
            )
            if feedback:
                self.agent_states[agent_name].human_feedback = feedback

        self.add_message(
            from_agent="human",
            to_agent=agent_name,
            message=f"Approval {'granted' if approved else 'denied'}",
            message_type="approval_response",
            approved=approved,
            feedback=feedback,
        )

    def update_shared_context(self, key: str, value: Any) -> None:
        """Update shared context accessible to all agents."""
        self.shared_context[key] = value
        self.update_timestamp()

    def calculate_metrics(self) -> None:
        """Calculate workflow metrics from agent states."""
        total_agents = len(self.agent_states)
        completed_agents = len(self.completed_agents)

        if total_agents > 0:
            self.success_rate = completed_agents / total_agents

        # Calculate average confidence from completed agents
        confidence_scores = [
            state.confidence_score
            for state in self.agent_states.values()
            if state.is_complete() and state.confidence_score > 0
        ]

        if confidence_scores:
            self.average_confidence = sum(confidence_scores) / len(confidence_scores)

        self.total_agents = total_agents
        self.update_timestamp()

    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.status in {"completed", "failed"}

    def requires_escalation(self) -> bool:
        """Check if workflow requires human escalation."""
        return (
            self.escalation_required
            or len(self.pending_approvals) > 0
            or len(self.failed_agents) > 0
            or self.average_confidence < 0.5
        )


class ConversationState(BaseState):
    """State for conversational interactions with agents."""

    # Conversation context
    conversation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Message history
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_turn: int = 0

    # Intent and context tracking
    detected_intent: Optional[str] = None
    intent_confidence: float = 0.0
    entities: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    # Multi-turn state
    expecting_response: bool = False
    pending_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)

    def add_message(
        self,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Add a message to the conversation."""
        self.messages.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "role": role,  # user, assistant, system
                "content": content,
                "agent": agent_name,
                "turn": self.current_turn,
                **kwargs,
            }
        )
        self.current_turn += 1
        self.update_timestamp()

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        return self.messages[-count:] if count > 0 else self.messages

    def set_context(self, key: str, value: Any) -> None:
        """Set context information."""
        self.context[key] = value
        self.update_timestamp()

    def add_entity(self, entity_type: str, entity_value: Any) -> None:
        """Add an extracted entity."""
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        self.entities[entity_type].append(entity_value)
        self.update_timestamp()