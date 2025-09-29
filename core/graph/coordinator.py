"""
Supervisor coordinator for multi-agent orchestration.

Implements the supervisor pattern where a central coordinator makes decisions
about which agents to invoke and how to route work between them.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from langgraph.graph import StateGraph

from core.config import Config
from core.exceptions import AgentError
from core.graph.base import BaseAgent, BaseCoordinator
from core.graph.builder import GraphBuilder
from core.graph.state import AgentState, WorkflowState
from core.observability import get_logger


class SupervisorCoordinator(BaseCoordinator):
    """
    Supervisor coordinator that orchestrates multiple agents using a central decision-making approach.

    The supervisor analyzes incoming tasks, determines which agents are needed,
    and coordinates their execution with proper error handling and escalation.
    """

    def __init__(
        self,
        name: str = "supervisor",
        agents: Optional[List[BaseAgent]] = None,
        config: Optional[Config] = None,
        **kwargs: Any,
    ):
        """
        Initialize the supervisor coordinator.

        Args:
            name: Coordinator name
            agents: List of agents to coordinate
            config: Framework configuration
            **kwargs: Additional parameters
        """
        super().__init__(name, agents or [], **kwargs)
        self.config = config
        self.graph_builder = GraphBuilder(config)

        # Supervisor-specific state
        self._agent_capabilities: Dict[str, List[str]] = {}
        self._agent_load: Dict[str, int] = {}
        self._routing_rules: List[Dict[str, Any]] = []

        # Build the supervisor graph
        self._graph: Optional[StateGraph] = None

    def register_agent_capabilities(self, agent_name: str, capabilities: List[str]) -> None:
        """
        Register what capabilities an agent provides.

        Args:
            agent_name: Name of the agent
            capabilities: List of capability names
        """
        self._agent_capabilities[agent_name] = capabilities
        self.logger.debug(f"Registered capabilities for {agent_name}: {capabilities}")

    def add_routing_rule(
        self,
        condition: str,
        agent: str,
        priority: int = 1,
        description: str = "",
    ) -> None:
        """
        Add a routing rule for task distribution.

        Args:
            condition: Condition for when to apply this rule
            agent: Target agent name
            priority: Rule priority (higher = more important)
            description: Human-readable description
        """
        rule = {
            "condition": condition,
            "agent": agent,
            "priority": priority,
            "description": description,
        }
        self._routing_rules.append(rule)
        # Sort by priority (descending)
        self._routing_rules.sort(key=lambda x: x["priority"], reverse=True)

    async def coordinate(
        self, workflow_input: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate execution across multiple agents.

        Args:
            workflow_input: Input data for the workflow
            config: Optional runtime configuration

        Returns:
            Workflow results
        """
        try:
            # Create workflow state
            workflow_state = WorkflowState(
                workflow_name="supervisor_coordination",
                input_data=workflow_input,
                phases=["analysis", "routing", "execution", "aggregation"],
            )

            self._current_workflow = workflow_state

            # Execute the coordination workflow
            if self._graph is None:
                self._graph = self._build_coordination_graph()

            result = await self._graph.ainvoke(workflow_state, config=config)

            self.logger.info(f"Coordination completed for workflow {workflow_state.id}")
            return result

        except Exception as e:
            self.logger.error(f"Coordination failed: {e}")
            raise AgentError(f"Supervisor coordination failed: {e}") from e

    def _build_coordination_graph(self) -> StateGraph:
        """Build the coordination graph."""
        # Define coordination nodes
        nodes = [
            {"name": "analyze_task", "func": self._analyze_task},
            {"name": "route_agents", "func": self._route_agents},
            {"name": "execute_agents", "func": self._execute_agents},
            {"name": "aggregate_results", "func": self._aggregate_results},
        ]

        # Create sequential graph for coordination
        graph = self.graph_builder.create_sequential_graph(
            nodes=nodes,
            state_class=WorkflowState,
        )

        return self.graph_builder.compile_graph(graph, debug=True)

    async def _analyze_task(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze the incoming task to understand requirements.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with analysis
        """
        try:
            state.current_phase = "analysis"
            state.status = "running"

            # Extract task information
            task_data = state.input_data
            task_type = task_data.get("type", "unknown")
            urgency = task_data.get("urgency", "normal")
            complexity = task_data.get("complexity", "medium")

            # Analyze requirements
            analysis = {
                "task_type": task_type,
                "urgency": urgency,
                "complexity": complexity,
                "estimated_agents_needed": self._estimate_agents_needed(task_data),
                "required_capabilities": self._extract_required_capabilities(task_data),
                "deadline": task_data.get("deadline"),
                "analyzed_at": datetime.utcnow().isoformat(),
            }

            # Update shared context
            state.update_shared_context("task_analysis", analysis)

            self.logger.info(f"Task analysis completed: {analysis}")
            return state

        except Exception as e:
            state.mark_agent_failed("supervisor", f"Task analysis failed: {e}")
            raise

    async def _route_agents(self, state: WorkflowState) -> WorkflowState:
        """
        Determine which agents should handle the task.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with routing decisions
        """
        try:
            state.current_phase = "routing"

            # Get task analysis
            analysis = state.shared_context.get("task_analysis", {})
            required_capabilities = analysis.get("required_capabilities", [])

            # Find suitable agents
            suitable_agents = self._find_suitable_agents(required_capabilities)

            # Apply routing rules
            selected_agents = self._apply_routing_rules(
                task_data=state.input_data,
                suitable_agents=suitable_agents,
                urgency=analysis.get("urgency", "normal"),
            )

            # Create execution plan
            execution_plan = {
                "selected_agents": selected_agents,
                "execution_order": self._determine_execution_order(selected_agents),
                "parallel_groups": self._identify_parallel_groups(selected_agents),
                "fallback_agents": self._identify_fallback_agents(suitable_agents, selected_agents),
            }

            state.update_shared_context("execution_plan", execution_plan)

            self.logger.info(f"Agent routing completed: {execution_plan}")
            return state

        except Exception as e:
            state.mark_agent_failed("supervisor", f"Agent routing failed: {e}")
            raise

    async def _execute_agents(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the selected agents according to the plan.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with execution results
        """
        try:
            state.current_phase = "execution"

            # Get execution plan
            execution_plan = state.shared_context.get("execution_plan", {})
            selected_agents = execution_plan.get("selected_agents", [])
            parallel_groups = execution_plan.get("parallel_groups", [])

            # Execute agents
            agent_results = {}

            for group in parallel_groups:
                # Execute agents in parallel within each group
                group_tasks = []
                for agent_name in group:
                    if agent_name in self.agents:
                        task = self._execute_single_agent(agent_name, state)
                        group_tasks.append(task)

                # Wait for group completion
                if group_tasks:
                    group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

                    # Process results
                    for i, result in enumerate(group_results):
                        agent_name = group[i]
                        if isinstance(result, Exception):
                            state.mark_agent_failed(agent_name, str(result))
                            self.logger.error(f"Agent {agent_name} failed: {result}")
                        else:
                            agent_results[agent_name] = result
                            state.mark_agent_completed(agent_name)
                            self.logger.info(f"Agent {agent_name} completed successfully")

            # Store agent results
            state.update_shared_context("agent_results", agent_results)

            return state

        except Exception as e:
            state.mark_agent_failed("supervisor", f"Agent execution failed: {e}")
            raise

    async def _aggregate_results(self, state: WorkflowState) -> WorkflowState:
        """
        Aggregate results from all executed agents.

        Args:
            state: Current workflow state

        Returns:
            Final workflow state with aggregated results
        """
        try:
            state.current_phase = "aggregation"

            # Get agent results
            agent_results = state.shared_context.get("agent_results", {})

            # Aggregate results based on task type
            aggregated_result = self._aggregate_agent_outputs(
                agent_results,
                state.input_data.get("aggregation_strategy", "merge"),
            )

            # Calculate final metrics
            state.calculate_metrics()

            # Set final output
            state.final_output = {
                "result": aggregated_result,
                "execution_summary": {
                    "total_agents": len(state.agent_states),
                    "successful_agents": len(state.completed_agents),
                    "failed_agents": len(state.failed_agents),
                    "success_rate": state.success_rate,
                    "average_confidence": state.average_confidence,
                },
                "metadata": {
                    "workflow_id": str(state.id),
                    "completed_at": datetime.utcnow().isoformat(),
                    "execution_time": (state.updated_at - state.created_at).total_seconds(),
                },
            }

            state.status = "completed"

            self.logger.info(f"Result aggregation completed for workflow {state.id}")
            return state

        except Exception as e:
            state.mark_agent_failed("supervisor", f"Result aggregation failed: {e}")
            state.status = "failed"
            raise

    async def _execute_single_agent(
        self, agent_name: str, workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """
        Execute a single agent with the workflow context.

        Args:
            agent_name: Name of agent to execute
            workflow_state: Current workflow state

        Returns:
            Agent execution results
        """
        agent = self.agents.get(agent_name)
        if not agent:
            raise AgentError(f"Agent {agent_name} not found")

        # Create agent-specific state
        agent_state = AgentState(
            agent_name=agent_name,
            agent_type=agent.__class__.__name__,
            task_description=workflow_state.input_data.get("description", ""),
            input_data=workflow_state.input_data,
            correlation_id=str(workflow_state.id),
        )

        # Add to workflow state
        workflow_state.add_agent_state(agent_name, agent_state)

        # Execute agent
        result = await agent.invoke(agent_state.dict())

        # Update load tracking
        self._agent_load[agent_name] = self._agent_load.get(agent_name, 0) + 1

        return result

    # Helper methods

    def _estimate_agents_needed(self, task_data: Dict[str, Any]) -> int:
        """Estimate how many agents are needed for a task."""
        complexity = task_data.get("complexity", "medium")
        complexity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return complexity_map.get(complexity, 2)

    def _extract_required_capabilities(self, task_data: Dict[str, Any]) -> List[str]:
        """Extract required capabilities from task data."""
        capabilities = []

        # Extract from explicit capabilities field
        if "required_capabilities" in task_data:
            capabilities.extend(task_data["required_capabilities"])

        # Infer from task type
        task_type = task_data.get("type", "")
        if "incident" in task_type.lower():
            capabilities.extend(["triage", "alerting"])
        elif "monitoring" in task_type.lower():
            capabilities.extend(["metrics", "monitoring"])
        elif "query" in task_type.lower():
            capabilities.extend(["search", "analysis"])

        return list(set(capabilities))  # Remove duplicates

    def _find_suitable_agents(self, required_capabilities: List[str]) -> List[str]:
        """Find agents that have the required capabilities."""
        suitable_agents = []

        for agent_name, capabilities in self._agent_capabilities.items():
            if any(cap in capabilities for cap in required_capabilities):
                suitable_agents.append(agent_name)

        return suitable_agents

    def _apply_routing_rules(
        self,
        task_data: Dict[str, Any],
        suitable_agents: List[str],
        urgency: str,
    ) -> List[str]:
        """Apply routing rules to select specific agents."""
        selected_agents = []

        # Apply rules in priority order
        for rule in self._routing_rules:
            if self._evaluate_condition(rule["condition"], task_data, urgency):
                agent = rule["agent"]
                if agent in suitable_agents and agent not in selected_agents:
                    selected_agents.append(agent)

        # If no rules matched, use all suitable agents
        if not selected_agents:
            selected_agents = suitable_agents

        return selected_agents

    def _evaluate_condition(self, condition: str, task_data: Dict[str, Any], urgency: str) -> bool:
        """Evaluate a routing rule condition."""
        # Simple condition evaluation - could be enhanced with proper expression parser
        if "urgency=critical" in condition and urgency == "critical":
            return True
        elif "type=incident" in condition and task_data.get("type") == "incident":
            return True
        elif "always" in condition:
            return True

        return False

    def _determine_execution_order(self, selected_agents: List[str]) -> List[str]:
        """Determine the order in which agents should execute."""
        # Simple ordering based on agent type priority
        priority_order = ["triage", "poller", "metrics", "supervisor"]

        ordered_agents = []
        for agent_type in priority_order:
            for agent_name in selected_agents:
                if agent_type in agent_name.lower() and agent_name not in ordered_agents:
                    ordered_agents.append(agent_name)

        # Add any remaining agents
        for agent_name in selected_agents:
            if agent_name not in ordered_agents:
                ordered_agents.append(agent_name)

        return ordered_agents

    def _identify_parallel_groups(self, selected_agents: List[str]) -> List[List[str]]:
        """Identify which agents can run in parallel."""
        # For now, run all agents in parallel
        # Could be enhanced with dependency analysis
        return [selected_agents] if selected_agents else []

    def _identify_fallback_agents(
        self, suitable_agents: List[str], selected_agents: List[str]
    ) -> List[str]:
        """Identify fallback agents in case of failures."""
        return [agent for agent in suitable_agents if agent not in selected_agents]

    def _aggregate_agent_outputs(
        self, agent_results: Dict[str, Any], strategy: str = "merge"
    ) -> Dict[str, Any]:
        """Aggregate outputs from multiple agents."""
        if strategy == "merge":
            # Simple merge strategy
            aggregated = {}
            for agent_name, result in agent_results.items():
                if isinstance(result, dict):
                    aggregated.update(result)
                else:
                    aggregated[agent_name] = result
            return aggregated

        elif strategy == "priority":
            # Use result from highest priority agent
            priority_order = ["triage", "supervisor", "metrics", "poller"]
            for agent_type in priority_order:
                for agent_name, result in agent_results.items():
                    if agent_type in agent_name.lower():
                        return result

        # Default: return all results
        return agent_results