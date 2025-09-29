"""
Graph builder for creating and configuring LangGraph instances.

Provides utilities for building complex agent graphs with proper state management,
conditional routing, and error handling.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver

from core.config import Config
from core.exceptions import AgentError
from core.graph.state import AgentState, WorkflowState
from core.observability import get_logger


class GraphBuilder:
    """
    Builder class for creating LangGraph instances with standard patterns.

    Provides convenience methods for common graph patterns used in the
    Golden Agent Framework.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the graph builder.

        Args:
            config: Framework configuration
        """
        self.config = config
        self.logger = get_logger("graph.builder")

    def create_simple_agent_graph(
        self,
        agent_node: Callable,
        state_class: type = AgentState,
        checkpointer: Optional[Any] = None,
    ) -> StateGraph:
        """
        Create a simple single-agent graph.

        Args:
            agent_node: The agent processing function
            state_class: State class to use
            checkpointer: Optional checkpointer for persistence

        Returns:
            Configured StateGraph
        """
        graph = StateGraph(state_class)

        # Add the agent node
        graph.add_node("agent", agent_node)

        # Add conditional routing for retry logic
        graph.add_conditional_edges(
            "agent",
            self._should_retry,
            {
                "retry": "agent",
                "end": END,
            },
        )

        # Set entry point
        graph.set_entry_point("agent")

        return graph

    def create_supervisor_graph(
        self,
        supervisor_node: Callable,
        worker_nodes: Dict[str, Callable],
        state_class: type = WorkflowState,
        checkpointer: Optional[Any] = None,
    ) -> StateGraph:
        """
        Create a supervisor-worker pattern graph.

        Args:
            supervisor_node: The supervisor decision function
            worker_nodes: Dictionary of worker name to function
            state_class: State class to use
            checkpointer: Optional checkpointer for persistence

        Returns:
            Configured StateGraph
        """
        graph = StateGraph(state_class)

        # Add supervisor node
        graph.add_node("supervisor", supervisor_node)

        # Add worker nodes
        for worker_name, worker_func in worker_nodes.items():
            graph.add_node(worker_name, worker_func)

        # Add edges from workers back to supervisor
        for worker_name in worker_nodes:
            graph.add_edge(worker_name, "supervisor")

        # Add conditional routing from supervisor
        routing_map = {name: name for name in worker_nodes}
        routing_map["FINISH"] = END

        graph.add_conditional_edges(
            "supervisor",
            self._supervisor_router,
            routing_map,
        )

        # Set entry point
        graph.set_entry_point("supervisor")

        return graph

    def create_sequential_graph(
        self,
        nodes: List[Dict[str, Union[str, Callable]]],
        state_class: type = WorkflowState,
        checkpointer: Optional[Any] = None,
    ) -> StateGraph:
        """
        Create a sequential processing graph.

        Args:
            nodes: List of node definitions with 'name' and 'func' keys
            state_class: State class to use
            checkpointer: Optional checkpointer for persistence

        Returns:
            Configured StateGraph
        """
        if not nodes:
            raise AgentError("At least one node is required for sequential graph")

        graph = StateGraph(state_class)

        # Add all nodes
        for node_def in nodes:
            name = node_def["name"]
            func = node_def["func"]
            graph.add_node(name, func)

        # Add sequential edges
        for i in range(len(nodes) - 1):
            current_node = nodes[i]["name"]
            next_node = nodes[i + 1]["name"]
            graph.add_edge(current_node, next_node)

        # Set entry and exit points
        graph.set_entry_point(nodes[0]["name"])
        graph.add_edge(nodes[-1]["name"], END)

        return graph

    def create_parallel_graph(
        self,
        parallel_nodes: List[Dict[str, Union[str, Callable]]],
        aggregator_node: Callable,
        state_class: type = WorkflowState,
        checkpointer: Optional[Any] = None,
    ) -> StateGraph:
        """
        Create a parallel processing graph with aggregation.

        Args:
            parallel_nodes: List of nodes to execute in parallel
            aggregator_node: Node to aggregate parallel results
            state_class: State class to use
            checkpointer: Optional checkpointer for persistence

        Returns:
            Configured StateGraph
        """
        graph = StateGraph(state_class)

        # Add aggregator node
        graph.add_node("aggregator", aggregator_node)

        # Add parallel nodes
        for node_def in parallel_nodes:
            name = node_def["name"]
            func = node_def["func"]
            graph.add_node(name, func)

            # Connect parallel nodes to aggregator
            graph.add_edge(name, "aggregator")

        # Set up entry point routing to parallel nodes
        graph.add_node("start", self._parallel_starter)
        graph.set_entry_point("start")

        # Route from start to all parallel nodes
        for node_def in parallel_nodes:
            graph.add_edge("start", node_def["name"])

        # End from aggregator
        graph.add_edge("aggregator", END)

        return graph

    def create_conditional_graph(
        self,
        condition_node: Callable,
        conditional_routes: Dict[str, str],
        node_functions: Dict[str, Callable],
        state_class: type = WorkflowState,
        checkpointer: Optional[Any] = None,
    ) -> StateGraph:
        """
        Create a graph with conditional routing.

        Args:
            condition_node: Function that determines routing
            conditional_routes: Mapping of conditions to node names
            node_functions: Mapping of node names to functions
            state_class: State class to use
            checkpointer: Optional checkpointer for persistence

        Returns:
            Configured StateGraph
        """
        graph = StateGraph(state_class)

        # Add condition node
        graph.add_node("condition", condition_node)

        # Add route nodes
        for node_name, node_func in node_functions.items():
            graph.add_node(node_name, node_func)

        # Add conditional routing
        routing_map = dict(conditional_routes)
        routing_map.setdefault("default", END)

        graph.add_conditional_edges(
            "condition",
            self._extract_routing_decision,
            routing_map,
        )

        # Set entry point
        graph.set_entry_point("condition")

        return graph

    def compile_graph(
        self,
        graph: StateGraph,
        checkpointer: Optional[Any] = None,
        debug: bool = False,
    ) -> CompiledGraph:
        """
        Compile a StateGraph for execution.

        Args:
            graph: StateGraph to compile
            checkpointer: Optional checkpointer for persistence
            debug: Enable debug mode

        Returns:
            Compiled graph ready for execution
        """
        try:
            # Use memory saver if no checkpointer provided
            if checkpointer is None:
                checkpointer = MemorySaver()

            # Compile with checkpointer
            compiled = graph.compile(
                checkpointer=checkpointer,
                debug=debug,
            )

            self.logger.debug("Graph compiled successfully")
            return compiled

        except Exception as e:
            self.logger.error(f"Failed to compile graph: {e}")
            raise AgentError(f"Graph compilation failed: {e}") from e

    # Helper methods for common routing patterns

    def _should_retry(self, state: Dict[str, Any]) -> str:
        """Determine if agent should retry execution."""
        if isinstance(state, dict):
            # Extract state information
            status = state.get("status", "pending")
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 3)
            errors = state.get("errors", [])

            # Retry conditions
            if (
                status == "failed"
                and retry_count < max_retries
                and len(errors) > 0
            ):
                return "retry"

        return "end"

    def _supervisor_router(self, state: Dict[str, Any]) -> str:
        """Route decisions from supervisor node."""
        if isinstance(state, dict):
            # Extract routing decision from state
            next_agent = state.get("next_agent")
            if next_agent and next_agent != "FINISH":
                return next_agent

        return "FINISH"

    def _parallel_starter(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state for parallel processing."""
        if isinstance(state, dict):
            # Mark as ready for parallel processing
            state["parallel_ready"] = True

        return state

    def _extract_routing_decision(self, state: Dict[str, Any]) -> str:
        """Extract routing decision from condition node output."""
        if isinstance(state, dict):
            return state.get("routing_decision", "default")
        return "default"

    # Advanced graph patterns

    def create_human_in_loop_graph(
        self,
        agent_node: Callable,
        approval_node: Callable,
        state_class: type = AgentState,
        checkpointer: Optional[Any] = None,
    ) -> StateGraph:
        """
        Create a graph with human-in-the-loop approval.

        Args:
            agent_node: The main agent processing function
            approval_node: Function to handle approval requests
            state_class: State class to use
            checkpointer: Optional checkpointer for persistence

        Returns:
            Configured StateGraph with approval workflow
        """
        graph = StateGraph(state_class)

        # Add nodes
        graph.add_node("agent", agent_node)
        graph.add_node("approval", approval_node)

        # Add conditional routing for approval
        graph.add_conditional_edges(
            "agent",
            self._needs_approval,
            {
                "approval": "approval",
                "end": END,
            },
        )

        # Route from approval back to agent or end
        graph.add_conditional_edges(
            "approval",
            self._approval_decision,
            {
                "continue": "agent",
                "end": END,
            },
        )

        # Set entry point
        graph.set_entry_point("agent")

        return graph

    def _needs_approval(self, state: Dict[str, Any]) -> str:
        """Check if agent output needs approval."""
        if isinstance(state, dict):
            requires_approval = state.get("requires_approval", False)
            if requires_approval:
                return "approval"
        return "end"

    def _approval_decision(self, state: Dict[str, Any]) -> str:
        """Process approval decision."""
        if isinstance(state, dict):
            approval_status = state.get("approval_status")
            if approval_status == "approved":
                return "continue"
        return "end"

    def create_error_handling_wrapper(
        self,
        base_graph: StateGraph,
        error_handler: Callable,
        max_retries: int = 3,
    ) -> StateGraph:
        """
        Wrap a graph with error handling and retry logic.

        Args:
            base_graph: The base graph to wrap
            error_handler: Function to handle errors
            max_retries: Maximum number of retries

        Returns:
            Graph with error handling
        """
        # This would need more complex implementation
        # to properly wrap an existing graph
        raise NotImplementedError("Error handling wrapper not yet implemented")

    def add_monitoring_nodes(
        self,
        graph: StateGraph,
        monitor_func: Callable,
        positions: List[str],
    ) -> StateGraph:
        """
        Add monitoring nodes to a graph.

        Args:
            graph: Graph to add monitoring to
            monitor_func: Monitoring function
            positions: Node positions to add monitoring

        Returns:
            Graph with monitoring nodes
        """
        # This would require graph modification capabilities
        # that may not be available in LangGraph
        raise NotImplementedError("Monitoring node injection not yet implemented")