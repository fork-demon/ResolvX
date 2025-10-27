"""
Base classes for LangGraph agents and components.

Provides the foundational structure for building agents that work within
the Golden Agent Framework's LangGraph-based orchestration system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from core.config import AgentConfig
from core.exceptions import AgentError
from core.graph.state import AgentState, BaseState, WorkflowState
from core.observability import get_logger
from core.prompts import PromptLoader, PromptTemplate


class BaseAgent(ABC):
    """
    Base class for all LangGraph agents in the Golden Agent Framework.

    Provides common functionality for:
    - State management
    - Tool integration
    - Error handling
    - Observability
    - Configuration management
    """

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique name for this agent
            config: Agent configuration
            tools: List of available tools
            **kwargs: Additional initialization parameters
        """
        self.name = name
        self.config = config
        self.tools = tools or []
        self.logger = get_logger(f"agent.{name}")

        # Internal state
        self._initialized = False
        self._graph: Optional[StateGraph] = None
        
        # Load prompts from configuration
        self.prompts: Dict[str, PromptTemplate] = {}
        self.prompt_loader = PromptLoader()
        self._load_prompts()

        # Initialize the agent
        self.initialize(**kwargs)

    def _load_prompts(self) -> None:
        """Load prompts from configuration."""
        try:
            # Get prompt configuration from agent config
            prompt_config = getattr(self.config, "prompts", {})
            
            if not prompt_config:
                self.logger.debug(f"No prompts configured for agent {self.name}")
                return
            
            # Load all configured prompts
            self.prompts = self.prompt_loader.load_prompts_from_config(prompt_config)
            self.logger.info(f"Loaded {len(self.prompts)} prompts for agent {self.name}")
        
        except Exception as e:
            self.logger.warning(f"Error loading prompts for agent {self.name}: {e}")
            self.prompts = {}
    
    def get_prompt(self, prompt_name: str, **variables) -> str:
        """
        Get a rendered prompt by name.
        
        Args:
            prompt_name: Name of the prompt (e.g., 'system_prompt', 'incident_analysis')
            **variables: Additional variables to inject when rendering
            
        Returns:
            Rendered prompt string
        """
        if prompt_name not in self.prompts:
            self.logger.warning(f"Prompt '{prompt_name}' not found for agent {self.name}")
            return ""
        
        prompt = self.prompts[prompt_name]
        return self.prompt_loader.render_prompt(prompt, **variables)
    
    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt is loaded."""
        return prompt_name in self.prompts

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize agent-specific components.

        Override this method to perform custom initialization logic.
        """
        self._initialized = True
        self.logger.info(f"Agent {self.name} initialized")

    @property
    def is_initialized(self) -> bool:
        """Check if agent is properly initialized."""
        return self._initialized

    @property
    def graph(self) -> StateGraph:
        """Get the LangGraph instance for this agent."""
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph for this agent.

        Returns:
            Configured StateGraph instance
        """
        pass

    @abstractmethod
    async def process(self, state: Union[AgentState, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a request using this agent.

        Args:
            state: Current agent state or input data

        Returns:
            Updated state or output data

        Raises:
            AgentError: If processing fails
        """
        pass

    async def invoke(
        self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the agent with input data.

        Args:
            input_data: Input data for processing
            config: Optional runtime configuration

        Returns:
            Processing results

        Raises:
            AgentError: If invocation fails
        """
        if not self.is_initialized:
            raise AgentError(f"Agent {self.name} is not initialized")

        try:
            # Create agent state if needed
            if isinstance(input_data, dict) and "agent_name" not in input_data:
                state = AgentState(
                    agent_name=self.name,
                    agent_type=self.__class__.__name__,
                    input_data=input_data,
                    max_steps=self.config.max_iterations,
                    max_retries=getattr(self.config, "max_retries", 3),
                )
            else:
                state = input_data

            # Execute using LangGraph
            result = await self.graph.ainvoke(state, config=config)

            self.logger.info(f"Agent {self.name} completed processing")
            return result

        except Exception as e:
            self.logger.error(f"Agent {self.name} failed: {e}")
            raise AgentError(f"Agent {self.name} processing failed: {e}") from e

    async def stream(
        self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ):
        """
        Stream processing results from the agent.

        Args:
            input_data: Input data for processing
            config: Optional runtime configuration

        Yields:
            Intermediate processing results
        """
        if not self.is_initialized:
            raise AgentError(f"Agent {self.name} is not initialized")

        try:
            # Create agent state if needed
            if isinstance(input_data, dict) and "agent_name" not in input_data:
                state = AgentState(
                    agent_name=self.name,
                    agent_type=self.__class__.__name__,
                    input_data=input_data,
                    max_steps=self.config.max_iterations,
                )
            else:
                state = input_data

            # Stream using LangGraph
            async for chunk in self.graph.astream(state, config=config):
                yield chunk

        except Exception as e:
            self.logger.error(f"Agent {self.name} streaming failed: {e}")
            raise AgentError(f"Agent {self.name} streaming failed: {e}") from e

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for this agent.

        Args:
            input_data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation - override in subclasses for specific validation
        required_fields = getattr(self, "required_input_fields", [])
        for field in required_fields:
            if field not in input_data:
                self.logger.warning(f"Missing required field: {field}")
                return False
        return True

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool instance or None if not found
        """
        for tool in self.tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                return tool
        return None

    def add_tool(self, tool: Any) -> None:
        """
        Add a tool to this agent.

        Args:
            tool: Tool instance to add
        """
        if tool not in self.tools:
            self.tools.append(tool)
            self.logger.debug(f"Added tool {getattr(tool, 'name', str(tool))}")

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool by name.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        for i, tool in enumerate(self.tools):
            if hasattr(tool, "name") and tool.name == tool_name:
                del self.tools[i]
                self.logger.debug(f"Removed tool {tool_name}")
                return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on this agent.

        Returns:
            Health check results
        """
        try:
            status = "healthy" if self.is_initialized else "unhealthy"
            return {
                "agent": self.name,
                "status": status,
                "tools_count": len(self.tools),
                "config_valid": self.config is not None,
            }
        except Exception as e:
            return {
                "agent": self.name,
                "status": "error",
                "error": str(e),
            }

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', tools={len(self.tools)})"


class BaseCoordinator(ABC):
    """
    Base class for agent coordinators.

    Coordinators manage the execution and interaction between multiple agents
    in complex workflows.
    """

    def __init__(self, name: str, agents: List[BaseAgent], **kwargs: Any):
        """
        Initialize the coordinator.

        Args:
            name: Coordinator name
            agents: List of agents to coordinate
            **kwargs: Additional initialization parameters
        """
        self.name = name
        self.agents = {agent.name: agent for agent in agents}
        self.logger = get_logger(f"coordinator.{name}")

        # Workflow state
        self._current_workflow: Optional[WorkflowState] = None

    @abstractmethod
    async def coordinate(
        self, workflow_input: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate execution across multiple agents.

        Args:
            workflow_input: Input data for the workflow
            config: Optional configuration

        Returns:
            Workflow results
        """
        pass

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(agent_name)

    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to coordination."""
        self.agents[agent.name] = agent
        self.logger.debug(f"Added agent {agent.name}")

    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent from coordination."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            self.logger.debug(f"Removed agent {agent_name}")
            return True
        return False

    @property
    def current_workflow(self) -> Optional[WorkflowState]:
        """Get the current workflow state."""
        return self._current_workflow

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all coordinated agents.

        Returns:
            Aggregated health check results
        """
        agent_health = {}
        overall_healthy = True

        for agent_name, agent in self.agents.items():
            health = await agent.health_check()
            agent_health[agent_name] = health
            if health.get("status") != "healthy":
                overall_healthy = False

        return {
            "coordinator": self.name,
            "status": "healthy" if overall_healthy else "degraded",
            "agents": agent_health,
            "total_agents": len(self.agents),
        }


class BaseWorkflow(ABC):
    """
    Base class for defining complex multi-agent workflows.

    Workflows define the structure and execution order of agents
    for accomplishing complex tasks.
    """

    def __init__(self, name: str, version: str = "1.0.0", **kwargs: Any):
        """
        Initialize the workflow.

        Args:
            name: Workflow name
            version: Workflow version
            **kwargs: Additional parameters
        """
        self.name = name
        self.version = version
        self.logger = get_logger(f"workflow.{name}")

        # Workflow definition
        self.phases: List[str] = []
        self.phase_agents: Dict[str, List[str]] = {}
        self.dependencies: Dict[str, List[str]] = {}

    @abstractmethod
    async def execute(
        self, input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the workflow.

        Args:
            input_data: Input data for workflow
            config: Optional configuration

        Returns:
            Workflow execution results
        """
        pass

    @abstractmethod
    def define_phases(self) -> List[str]:
        """
        Define the phases of this workflow.

        Returns:
            List of phase names in execution order
        """
        pass

    def add_phase(self, phase_name: str, agents: List[str]) -> None:
        """
        Add a phase to the workflow.

        Args:
            phase_name: Name of the phase
            agents: List of agent names for this phase
        """
        if phase_name not in self.phases:
            self.phases.append(phase_name)
        self.phase_agents[phase_name] = agents

    def add_dependency(self, dependent_phase: str, required_phases: List[str]) -> None:
        """
        Add dependencies between phases.

        Args:
            dependent_phase: Phase that has dependencies
            required_phases: List of phases that must complete first
        """
        self.dependencies[dependent_phase] = required_phases

    def validate_workflow(self) -> bool:
        """
        Validate the workflow definition.

        Returns:
            True if workflow is valid, False otherwise
        """
        # Check for circular dependencies
        visited = set()
        rec_stack = set()

        def has_cycle(phase: str) -> bool:
            visited.add(phase)
            rec_stack.add(phase)

            for dep in self.dependencies.get(phase, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(phase)
            return False

        for phase in self.phases:
            if phase not in visited:
                if has_cycle(phase):
                    self.logger.error(f"Circular dependency detected in workflow {self.name}")
                    return False

        return True