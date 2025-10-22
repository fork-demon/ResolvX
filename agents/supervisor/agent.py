"""
Supervisor agent for coordinating multi-agent workflows and task delegation.

This agent manages and coordinates other agents, handles task routing,
and orchestrates complex multi-agent workflows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable
from enum import Enum

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from core.config import Config, AgentConfig
from core.exceptions import AgentError, ToolError
from core.graph.base import BaseAgent, BaseCoordinator
from core.graph.state import AgentState, ConversationState
from core.graph.coordinator import SupervisorCoordinator
from core.memory.base import BaseMemory
from core.observability import get_logger, get_tracer, get_metrics_client
from typing import Any as BaseRAG


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """A task to be executed by an agent."""

    id: str = Field(description="Unique task ID")
    type: str = Field(description="Task type")
    description: str = Field(description="Task description")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    assigned_agent: Optional[str] = Field(description="Assigned agent name")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(description="Completion timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Task data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Task metadata")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    timeout: Optional[int] = Field(description="Task timeout in seconds")
    retries: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=3, description="Maximum retries allowed")
    result: Optional[Dict[str, Any]] = Field(description="Task result")
    error: Optional[str] = Field(description="Error message if failed")


class AgentInfo(BaseModel):
    """Information about a managed agent."""

    name: str = Field(description="Agent name")
    type: str = Field(description="Agent type")
    status: str = Field(description="Agent status")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    current_task: Optional[str] = Field(description="Current task ID")
    task_queue_size: int = Field(default=0, description="Number of queued tasks")
    last_heartbeat: datetime = Field(default_factory=datetime.now, description="Last heartbeat")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")


class WorkflowDefinition(BaseModel):
    """Definition of a multi-agent workflow."""

    name: str = Field(description="Workflow name")
    description: str = Field(description="Workflow description")
    steps: List[Dict[str, Any]] = Field(description="Workflow steps")
    triggers: List[str] = Field(default_factory=list, description="Workflow triggers")
    timeout: Optional[int] = Field(description="Workflow timeout in seconds")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Retry policy")
    enabled: bool = Field(default=True, description="Whether workflow is enabled")


class WorkflowExecution(BaseModel):
    """Execution instance of a workflow."""

    id: str = Field(description="Execution ID")
    workflow_name: str = Field(description="Workflow name")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Execution status")
    started_at: datetime = Field(default_factory=datetime.now, description="Start timestamp")
    completed_at: Optional[datetime] = Field(description="Completion timestamp")
    current_step: int = Field(default=0, description="Current step index")
    step_results: List[Dict[str, Any]] = Field(default_factory=list, description="Step results")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    error: Optional[str] = Field(description="Error message if failed")


class SupervisorState(AgentState):
    """State for the supervisor agent."""

    agents: Dict[str, AgentInfo] = Field(default_factory=dict, description="Managed agents")
    tasks: Dict[str, Task] = Field(default_factory=dict, description="All tasks")
    task_queue: List[str] = Field(default_factory=list, description="Task queue (task IDs)")
    workflows: Dict[str, WorkflowDefinition] = Field(default_factory=dict, description="Workflow definitions")
    executions: Dict[str, WorkflowExecution] = Field(default_factory=dict, description="Workflow executions")
    routing_rules: Dict[str, List[str]] = Field(default_factory=dict, description="Task routing rules")
    load_balancing: Dict[str, Any] = Field(default_factory=dict, description="Load balancing configuration")
    performance_history: List[Dict[str, Any]] = Field(default_factory=list, description="Performance history")
    last_health_check: Optional[datetime] = Field(description="Last health check timestamp")


class SupervisorAgent(BaseAgent):
    """
    Agent for supervising and coordinating multi-agent workflows.

    This agent manages other agents, handles task routing and delegation,
    monitors performance, and orchestrates complex workflows.
    """

    def __init__(
        self,
        config: Union[Config, AgentConfig],
        memory: Optional[BaseMemory] = None,
        rag: Optional[BaseRAG] = None,
        **kwargs
    ):
        """Initialize the supervisor agent."""
        # If we get an AgentConfig, wrap it
        if isinstance(config, AgentConfig):
            agent_config = config
            # Create minimal Config wrapper
            from core.config import Config as FullConfig
            full_config = FullConfig()
            super().__init__("supervisor", agent_config, **kwargs)
        else:
            full_config = config
            super().__init__(config, memory, rag, **kwargs)

        self.logger = get_logger("agents.supervisor")
        self.tracer = get_tracer("agents.supervisor")
        self.metrics = get_metrics_client()

        # Supervisor configuration
        self.max_concurrent_tasks = getattr(config, "max_concurrent", 20)
        self.health_check_interval = getattr(config, "health_check_interval", 30)
        self.task_timeout_default = getattr(config, "default_timeout", 300)

        # Agent management
        self._managed_agents: Dict[str, BaseAgent] = {}
        self._agent_tasks: Dict[str, Set[asyncio.Task]] = {}
        self._task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Coordinator (only if full config available)
        self.coordinator = None
        if isinstance(full_config, Config):
            self.coordinator = SupervisorCoordinator(full_config)
    
    def build_graph(self):
        """Build a simple graph for the supervisor agent."""
        from langgraph.graph import StateGraph, END
        graph = StateGraph(AgentState)
        
        # Simple single-node graph for supervisor
        graph.add_node("make_decision", self._make_decision_node)
        graph.set_entry_point("make_decision")
        graph.add_edge("make_decision", END)
        
        return graph.compile()
    
    async def _make_decision_node(self, state: AgentState) -> AgentState:
        """LangGraph node for making decisions."""
        decision = self._make_final_decision(state.input_data)
        state.metadata["decision"] = decision
        state.increment_step()
        return state
    
    async def process(self, state: AgentState) -> Dict[str, Any]:
        """Process enriched ticket data and make final decision."""
        with self.tracer.start_as_current_span("supervisor_process") as span:
            # Set input data
            input_data = {
                "ticket_id": state.input_data.get("ticket_id", "unknown"),
                "triage_severity": state.input_data.get("severity", "unknown"),
                "triage_routing": state.input_data.get("routing_decision", "unknown"),
                "tools_used": state.input_data.get("tools_used", []),
                "enrichment_sources": state.input_data.get("enrichment_sources", {})
            }
            span.set_input(input_data)
            span.set_attribute("ticket_id", input_data["ticket_id"])
            span.set_attribute("severity", input_data["triage_severity"])
            
            try:
                # Make final decision based on enriched data
                decision = self._make_final_decision(state.input_data)
                
                # Set output data
                output_data = {
                    "decision": decision["action"],
                    "reason": decision["reason"],
                    "assigned_to": decision.get("assigned_to"),
                    "escalated": decision.get("escalated", False),
                    "ticket_id": input_data["ticket_id"],
                    "success": True
                }
                span.set_output(output_data)
                span.set_attribute("decision", decision["action"])
                span.set_attribute("escalated", decision.get("escalated", False))
                
                self.logger.info(f"Supervisor decision for {input_data['ticket_id']}: {decision['action']}")
                
                return {
                    "success": True,
                    "decision": decision,
                    "ticket_id": input_data["ticket_id"]
                }
                
            except Exception as e:
                span.record_exception(e)
                span.set_output({"success": False, "error": str(e), "ticket_id": input_data["ticket_id"]})
                self.logger.error(f"Supervisor processing failed: {e}")
                raise AgentError(f"Supervisor processing failed: {e}") from e
    
    def _make_final_decision(self, enriched_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make final decision based on enriched ticket data."""
        severity = enriched_data.get("severity", "medium")
        routing = enriched_data.get("routing_decision", "unknown")
        tools_used = enriched_data.get("tools_used", [])
        
        # Decision logic
        if severity == "critical":
            return {
                "action": "ESCALATE_TO_HUMAN",
                "reason": "Critical severity requires human intervention",
                "escalated": True,
                "assigned_to": "oncall_engineer"
            }
        elif severity == "high" and "splunk_search" in tools_used:
            return {
                "action": "ASSIGN_TO_TEAM",
                "reason": "High severity with error logs found - assign to engineering team",
                "escalated": False,
                "assigned_to": "engineering_team"
            }
        elif "newrelic_metrics" in tools_used:
            return {
                "action": "ASSIGN_TO_TEAM",
                "reason": "Performance issue detected - assign to devops team",
                "escalated": False,
                "assigned_to": "devops_team"
            }
        elif routing in ["security", "infrastructure"]:
            return {
                "action": "ROUTE_TO_SPECIALIST",
                "reason": f"Requires specialist attention: {routing}",
                "escalated": False,
                "assigned_to": f"{routing}_team"
            }
        else:
            return {
                "action": "ADD_COMMENT",
                "reason": "Automated analysis complete - adding findings to ticket",
                "escalated": False,
                "assigned_to": None
            }

    async def initialize(self) -> None:
        """Initialize the supervisor agent."""
        try:
            await super().initialize()

            # Initialize coordinator
            await self.coordinator.initialize()

            # Load routing rules and workflows from config
            await self._load_configuration()

            self.logger.info("Supervisor agent initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize supervisor agent: {e}")
            raise AgentError(f"Supervisor initialization failed: {e}") from e

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel all managed agent tasks
            for agent_name, tasks in self._agent_tasks.items():
                for task in tasks:
                    if not task.done():
                        task.cancel()

            # Wait for tasks to complete
            all_tasks = []
            for tasks in self._agent_tasks.values():
                all_tasks.extend(tasks)

            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)

            # Cleanup managed agents
            for agent in self._managed_agents.values():
                await agent.cleanup()

            # Cleanup coordinator
            await self.coordinator.cleanup()

            await super().cleanup()

        except Exception as e:
            self.logger.error(f"Error during supervisor cleanup: {e}")

    def _create_workflow(self) -> StateGraph:
        """Create the supervisor workflow."""
        workflow = StateGraph(SupervisorState)

        # Add nodes
        workflow.add_node("manage_agents", self._manage_agents)
        workflow.add_node("process_tasks", self._process_tasks)
        workflow.add_node("route_tasks", self._route_tasks)
        workflow.add_node("monitor_execution", self._monitor_execution)
        workflow.add_node("handle_workflows", self._handle_workflows)
        workflow.add_node("perform_health_checks", self._perform_health_checks)
        workflow.add_node("update_metrics", self._update_metrics)

        # Add edges
        workflow.set_entry_point("manage_agents")
        workflow.add_edge("manage_agents", "process_tasks")
        workflow.add_edge("process_tasks", "route_tasks")
        workflow.add_edge("route_tasks", "monitor_execution")
        workflow.add_edge("monitor_execution", "handle_workflows")
        workflow.add_edge("handle_workflows", "perform_health_checks")
        workflow.add_edge("perform_health_checks", "update_metrics")
        workflow.add_edge("update_metrics", "manage_agents")  # Continuous loop

        return workflow

    async def _load_configuration(self) -> None:
        """Load supervisor configuration from config files."""
        try:
            supervisor_config = self.config.agents.get("supervisor", {})

            # Load routing rules
            routing_rules = supervisor_config.get("routing_rules", {})
            if hasattr(self, '_current_state') and self._current_state:
                self._current_state.routing_rules.update(routing_rules)

            # Load workflows
            workflows_config = supervisor_config.get("workflows", [])
            for workflow_data in workflows_config:
                workflow = WorkflowDefinition(**workflow_data)
                if hasattr(self, '_current_state') and self._current_state:
                    self._current_state.workflows[workflow.name] = workflow

            self.logger.info(f"Loaded {len(routing_rules)} routing rules and {len(workflows_config)} workflows")

        except Exception as e:
            self.logger.error(f"Error loading supervisor configuration: {e}")

    async def _manage_agents(self, state: SupervisorState) -> SupervisorState:
        """Manage registered agents and their status."""
        with self.tracer.start_as_current_span("manage_agents"):
            try:
                current_time = datetime.now()

                # Update agent status
                for agent_name, agent in self._managed_agents.items():
                    if agent_name not in state.agents:
                        # Register new agent
                        state.agents[agent_name] = AgentInfo(
                            name=agent_name,
                            type=agent.__class__.__name__,
                            status="active",
                            capabilities=getattr(agent, 'capabilities', []),
                            last_heartbeat=current_time
                        )

                    # Update heartbeat
                    state.agents[agent_name].last_heartbeat = current_time

                    # Update task queue size
                    agent_tasks = self._agent_tasks.get(agent_name, set())
                    state.agents[agent_name].task_queue_size = len(agent_tasks)

                # Check for stale agents
                stale_threshold = timedelta(minutes=5)
                for agent_name, agent_info in state.agents.items():
                    if current_time - agent_info.last_heartbeat > stale_threshold:
                        agent_info.status = "stale"
                        self.logger.warning(f"Agent {agent_name} appears stale")

                return state

            except Exception as e:
                self.logger.error(f"Error managing agents: {e}")
                return state

    async def _process_tasks(self, state: SupervisorState) -> SupervisorState:
        """Process pending tasks in the queue."""
        with self.tracer.start_as_current_span("process_tasks"):
            try:
                # Process pending tasks
                tasks_to_process = []
                for task_id in state.task_queue[:]:
                    task = state.tasks.get(task_id)
                    if not task:
                        state.task_queue.remove(task_id)
                        continue

                    if task.status == TaskStatus.PENDING:
                        # Check dependencies
                        dependencies_met = True
                        for dep_id in task.dependencies:
                            dep_task = state.tasks.get(dep_id)
                            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                                dependencies_met = False
                                break

                        if dependencies_met:
                            tasks_to_process.append(task)

                # Sort by priority
                tasks_to_process.sort(key=lambda t: (
                    t.priority == TaskPriority.CRITICAL,
                    t.priority == TaskPriority.HIGH,
                    t.priority == TaskPriority.MEDIUM,
                    t.priority == TaskPriority.LOW
                ), reverse=True)

                # Update task statuses
                for task in tasks_to_process:
                    task.status = TaskStatus.ASSIGNED
                    task.updated_at = datetime.now()

                return state

            except Exception as e:
                self.logger.error(f"Error processing tasks: {e}")
                return state

    async def _route_tasks(self, state: SupervisorState) -> SupervisorState:
        """Route assigned tasks to appropriate agents."""
        with self.tracer.start_as_current_span("route_tasks"):
            try:
                for task_id in state.task_queue[:]:
                    task = state.tasks.get(task_id)
                    if not task or task.status != TaskStatus.ASSIGNED:
                        continue

                    # Find suitable agent
                    suitable_agent = await self._find_suitable_agent(task, state)
                    if not suitable_agent:
                        self.logger.warning(f"No suitable agent found for task {task_id}")
                        continue

                    # Assign task to agent
                    task.assigned_agent = suitable_agent
                    task.status = TaskStatus.IN_PROGRESS
                    task.updated_at = datetime.now()

                    # Start task execution
                    await self._execute_task(task, suitable_agent)

                    # Remove from queue
                    state.task_queue.remove(task_id)

                return state

            except Exception as e:
                self.logger.error(f"Error routing tasks: {e}")
                return state

    async def _find_suitable_agent(self, task: Task, state: SupervisorState) -> Optional[str]:
        """Find the most suitable agent for a task."""
        try:
            # Check routing rules first
            task_type_rules = state.routing_rules.get(task.type, [])
            if task_type_rules:
                for agent_name in task_type_rules:
                    agent_info = state.agents.get(agent_name)
                    if (agent_info and agent_info.status == "active" and
                        agent_info.task_queue_size < 5):  # Load balancing
                        return agent_name

            # Fallback to capability matching
            for agent_name, agent_info in state.agents.items():
                if (agent_info.status == "active" and
                    task.type in agent_info.capabilities and
                    agent_info.task_queue_size < 5):
                    return agent_name

            # Last resort: any available agent
            for agent_name, agent_info in state.agents.items():
                if (agent_info.status == "active" and
                    agent_info.task_queue_size < 10):
                    return agent_name

            return None

        except Exception as e:
            self.logger.error(f"Error finding suitable agent for task {task.id}: {e}")
            return None

    async def _execute_task(self, task: Task, agent_name: str) -> None:
        """Execute a task on the specified agent."""
        try:
            agent = self._managed_agents.get(agent_name)
            if not agent:
                raise AgentError(f"Agent {agent_name} not found")

            # Create execution task
            async def _run_task():
                async with self._task_semaphore:
                    try:
                        # Execute task on agent
                        result = await agent.execute_task(task.data)

                        # Update task with result
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        task.result = result

                        self.logger.info(f"Task {task.id} completed successfully")

                    except Exception as e:
                        # Handle task failure
                        task.retries += 1
                        if task.retries < task.max_retries:
                            task.status = TaskStatus.PENDING
                            self.logger.warning(f"Task {task.id} failed, retrying ({task.retries}/{task.max_retries})")
                        else:
                            task.status = TaskStatus.FAILED
                            task.error = str(e)
                            self.logger.error(f"Task {task.id} failed permanently: {e}")

                    finally:
                        task.updated_at = datetime.now()

            # Start task
            execution_task = asyncio.create_task(_run_task())

            # Track task
            if agent_name not in self._agent_tasks:
                self._agent_tasks[agent_name] = set()
            self._agent_tasks[agent_name].add(execution_task)

            # Clean up when done
            execution_task.add_done_callback(
                lambda t: self._agent_tasks[agent_name].discard(t)
            )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.now()
            self.logger.error(f"Error executing task {task.id}: {e}")

    async def _monitor_execution(self, state: SupervisorState) -> SupervisorState:
        """Monitor task execution and handle timeouts."""
        with self.tracer.start_as_current_span("monitor_execution"):
            try:
                current_time = datetime.now()

                # Check for task timeouts
                for task in state.tasks.values():
                    if (task.status == TaskStatus.IN_PROGRESS and task.timeout):
                        elapsed = (current_time - task.updated_at).total_seconds()
                        if elapsed > task.timeout:
                            task.status = TaskStatus.FAILED
                            task.error = "Task timeout"
                            task.updated_at = current_time
                            self.logger.warning(f"Task {task.id} timed out")

                return state

            except Exception as e:
                self.logger.error(f"Error monitoring execution: {e}")
                return state

    async def _handle_workflows(self, state: SupervisorState) -> SupervisorState:
        """Handle workflow execution and management."""
        with self.tracer.start_as_current_span("handle_workflows"):
            try:
                # Process active workflow executions
                for execution in state.executions.values():
                    if execution.status != TaskStatus.IN_PROGRESS:
                        continue

                    workflow = state.workflows.get(execution.workflow_name)
                    if not workflow:
                        execution.status = TaskStatus.FAILED
                        execution.error = "Workflow definition not found"
                        continue

                    # Check if current step is complete
                    if execution.current_step < len(workflow.steps):
                        step = workflow.steps[execution.current_step]

                        # Execute step (simplified - would need more complex logic)
                        step_task_id = f"{execution.id}_step_{execution.current_step}"
                        if step_task_id not in state.tasks:
                            # Create task for this step
                            step_task = Task(
                                id=step_task_id,
                                type=step.get("type", "generic"),
                                description=step.get("description", "Workflow step"),
                                data=step.get("data", {}),
                                metadata={"workflow_execution": execution.id}
                            )
                            state.tasks[step_task_id] = step_task
                            state.task_queue.append(step_task_id)
                        else:
                            # Check if step is complete
                            step_task = state.tasks[step_task_id]
                            if step_task.status == TaskStatus.COMPLETED:
                                execution.step_results.append(step_task.result or {})
                                execution.current_step += 1
                            elif step_task.status == TaskStatus.FAILED:
                                execution.status = TaskStatus.FAILED
                                execution.error = f"Step {execution.current_step} failed: {step_task.error}"

                    else:
                        # Workflow complete
                        execution.status = TaskStatus.COMPLETED
                        execution.completed_at = datetime.now()

                return state

            except Exception as e:
                self.logger.error(f"Error handling workflows: {e}")
                return state

    async def _perform_health_checks(self, state: SupervisorState) -> SupervisorState:
        """Perform health checks on managed agents."""
        with self.tracer.start_as_current_span("perform_health_checks"):
            try:
                current_time = datetime.now()

                # Check if it's time for health checks
                if (state.last_health_check and
                    (current_time - state.last_health_check).total_seconds() < self.health_check_interval):
                    return state

                # Perform health checks
                for agent_name, agent in self._managed_agents.items():
                    try:
                        health_result = await agent.health_check()

                        if agent_name in state.agents:
                            state.agents[agent_name].performance_metrics.update(health_result)

                            # Update status based on health
                            if health_result.get("status") == "healthy":
                                state.agents[agent_name].status = "active"
                            else:
                                state.agents[agent_name].status = "degraded"

                    except Exception as e:
                        self.logger.error(f"Health check failed for {agent_name}: {e}")
                        if agent_name in state.agents:
                            state.agents[agent_name].status = "error"

                state.last_health_check = current_time

                return state

            except Exception as e:
                self.logger.error(f"Error performing health checks: {e}")
                return state

    async def _update_metrics(self, state: SupervisorState) -> SupervisorState:
        """Update supervisor and system metrics."""
        with self.tracer.start_as_current_span("update_metrics"):
            try:
                # Count tasks by status
                task_counts = {}
                for task in state.tasks.values():
                    task_counts[task.status.value] = task_counts.get(task.status.value, 0) + 1

                # Update metrics
                for status, count in task_counts.items():
                    self.metrics.gauge("supervisor_tasks_total").labels(status=status).set(count)

                # Agent metrics
                active_agents = sum(1 for info in state.agents.values() if info.status == "active")
                self.metrics.gauge("supervisor_agents_active").set(active_agents)

                # Workflow metrics
                active_workflows = sum(1 for exec in state.executions.values() if exec.status == TaskStatus.IN_PROGRESS)
                self.metrics.gauge("supervisor_workflows_active").set(active_workflows)

                # Store performance history
                performance_snapshot = {
                    "timestamp": datetime.now(),
                    "active_agents": active_agents,
                    "task_counts": task_counts,
                    "active_workflows": active_workflows,
                    "queue_size": len(state.task_queue)
                }
                state.performance_history.append(performance_snapshot)

                # Limit history size
                if len(state.performance_history) > 1000:
                    state.performance_history = state.performance_history[-1000:]

                return state

            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                return state

    # Public API methods

    def register_agent(self, agent: BaseAgent, capabilities: Optional[List[str]] = None) -> None:
        """Register an agent to be managed by the supervisor."""
        try:
            agent_name = agent.__class__.__name__
            self._managed_agents[agent_name] = agent

            if capabilities:
                agent.capabilities = capabilities

            self.logger.info(f"Registered agent: {agent_name}")

        except Exception as e:
            self.logger.error(f"Error registering agent: {e}")
            raise AgentError(f"Failed to register agent: {e}") from e

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister a managed agent."""
        try:
            if agent_name in self._managed_agents:
                del self._managed_agents[agent_name]

                # Cancel any running tasks for this agent
                if agent_name in self._agent_tasks:
                    for task in self._agent_tasks[agent_name]:
                        if not task.done():
                            task.cancel()
                    del self._agent_tasks[agent_name]

            self.logger.info(f"Unregistered agent: {agent_name}")

        except Exception as e:
            self.logger.error(f"Error unregistering agent {agent_name}: {e}")

    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution."""
        try:
            if hasattr(self, '_current_state') and self._current_state:
                self._current_state.tasks[task.id] = task
                self._current_state.task_queue.append(task.id)

            self.logger.info(f"Submitted task: {task.id}")
            return task.id

        except Exception as e:
            self.logger.error(f"Error submitting task {task.id}: {e}")
            raise AgentError(f"Failed to submit task: {e}") from e

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        try:
            if hasattr(self, '_current_state') and self._current_state:
                task = self._current_state.tasks.get(task_id)
                if task:
                    task.status = TaskStatus.CANCELLED
                    task.updated_at = datetime.now()

                    # Remove from queue if pending
                    if task_id in self._current_state.task_queue:
                        self._current_state.task_queue.remove(task_id)

                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get the status of a task."""
        try:
            if hasattr(self, '_current_state') and self._current_state:
                return self._current_state.tasks.get(task_id)
            return None

        except Exception as e:
            self.logger.error(f"Error getting task status {task_id}: {e}")
            return None

    async def start_workflow(self, workflow_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start a workflow execution."""
        try:
            if hasattr(self, '_current_state') and self._current_state:
                workflow = self._current_state.workflows.get(workflow_name)
                if not workflow:
                    raise AgentError(f"Workflow {workflow_name} not found")

                execution_id = f"{workflow_name}_{datetime.now().isoformat()}"
                execution = WorkflowExecution(
                    id=execution_id,
                    workflow_name=workflow_name,
                    context=context or {}
                )

                self._current_state.executions[execution_id] = execution

                self.logger.info(f"Started workflow execution: {execution_id}")
                return execution_id

            raise AgentError("Supervisor not initialized")

        except Exception as e:
            self.logger.error(f"Error starting workflow {workflow_name}: {e}")
            raise AgentError(f"Failed to start workflow: {e}") from e

    async def get_agent_status(self) -> Dict[str, AgentInfo]:
        """Get status of all managed agents."""
        try:
            if hasattr(self, '_current_state') and self._current_state:
                return self._current_state.agents.copy()
            return {}

        except Exception as e:
            self.logger.error(f"Error getting agent status: {e}")
            return {}