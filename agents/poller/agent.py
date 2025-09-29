"""
Zendesk Poller Agent for monitoring Zendesk queues and processing tickets.

This agent continuously polls specific Zendesk queues for new tickets,
analyzes them, and routes them to appropriate teams or agents for processing.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, TypedDict

from langgraph import StateGraph, END
from pydantic import BaseModel, Field

from core.config import AgentConfig
from core.exceptions import AgentError, ToolError
from core.graph.base import BaseAgent
from core.graph.state import AgentState
from core.memory.base import BaseMemory
from core.observability import get_logger, get_tracer, get_metrics_client
from core.gateway.tool_registry import ToolRegistry
from extensions.rag_backends.base import BaseRAG
from extensions.mcp_servers.zendesk_tools import ZendeskConfig, ZendeskMCPServer


class ZendeskAuth(BaseModel):
    """Authentication and endpoint configuration for Zendesk (team-level)."""

    email: Optional[str] = Field(default=None, description="Service account email (for token auth)")
    api_token: Optional[str] = Field(default=None, description="Zendesk API token")
    username: Optional[str] = Field(default=None, description="Zendesk username (for basic auth)")
    password: Optional[str] = Field(default=None, description="Zendesk password (for basic auth)")
    base_url: Optional[str] = Field(default=None, description="Optional base URL or gateway proxy URL")
    subdomain: Optional[str] = Field(default=None, description="Optional Zendesk subdomain if not using base_url")


class ZendeskQueueConfig(BaseModel):
    """Configuration for a Zendesk queue to poll (per-queue)."""

    queue_name: str = Field(description="Zendesk queue name to poll")
    poll_interval: int = Field(default=30, description="Polling interval in seconds")
    max_tickets_per_poll: int = Field(default=50, description="Maximum tickets to retrieve per poll")
    enabled: bool = Field(default=True, description="Whether polling is enabled")
    priority_threshold: str = Field(default="normal", description="Minimum priority to process")
    auto_assign: bool = Field(default=False, description="Whether to auto-assign tickets")
    assignee_id: Optional[str] = Field(default=None, description="Default assignee ID")
    tags: List[str] = Field(default_factory=list, description="Tags to add to processed tickets")


class ZendeskPollResult(BaseModel):
    """Result of a Zendesk queue polling operation."""

    team: str = Field(description="Team name")
    queue_name: str = Field(description="Queue name")
    timestamp: datetime = Field(description="Poll timestamp")
    success: bool = Field(description="Whether poll was successful")
    ticket_count: int = Field(description="Number of tickets found")
    tickets: List[Dict[str, Any]] = Field(description="List of tickets")
    processing_time: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(description="Error message if failed")
    new_tickets: int = Field(description="Number of new tickets")
    updated_tickets: int = Field(description="Number of updated tickets")


class ZendeskPollerState(AgentState):
    """State for the Zendesk poller agent."""

    queue_configs: List[ZendeskQueueConfig] = Field(default_factory=list)
    poll_results: List[ZendeskPollResult] = Field(default_factory=list)
    last_poll: Optional[datetime] = Field(default=None)
    active_queues: Set[str] = Field(default_factory=set)
    error_count: int = Field(default=0)
    consecutive_errors: int = Field(default=0)
    is_running: bool = Field(default=False)
    poll_interval: int = Field(default=30)
    max_errors: int = Field(default=5)
    backoff_multiplier: float = Field(default=1.5)
    max_backoff: int = Field(default=300)
    processed_tickets: Set[str] = Field(default_factory=set)
    team_assignments: Dict[str, str] = Field(default_factory=dict)
    # Workflow fields
    last_poll_times: Dict[str, datetime] = Field(default_factory=dict, description="Last poll time per queue")
    queues_to_poll: List[str] = Field(default_factory=list, description="Queue names selected for polling")
    tickets: List[Dict[str, Any]] = Field(default_factory=list, description="Tickets collected in this run")
    analyzed_tickets: List[Dict[str, Any]] = Field(default_factory=list, description="Analyzed tickets")
    routing_decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Routing decisions")
    update_results: List[Dict[str, Any]] = Field(default_factory=list, description="Ticket update outcomes")
    final_result: Dict[str, Any] = Field(default_factory=dict, description="Final aggregated result")


class ZendeskPollerAgent(BaseAgent):
    """
    Zendesk Poller Agent for monitoring Zendesk queues and processing tickets.

    This agent continuously polls specific Zendesk queues for new tickets,
    analyzes them, and routes them to appropriate teams or agents for processing.
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[BaseMemory] = None,
        rag: Optional[BaseRAG] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Zendesk poller agent.

        Args:
            config: Agent configuration
            tool_registry: Tool registry for accessing tools
            memory: Memory backend for storing context
            rag: RAG backend for knowledge retrieval
            **kwargs: Additional parameters
        """
        super().__init__("poller", config, **kwargs)

        self.tool_registry = tool_registry
        self.memory = memory
        self.rag = rag
        self.tracer = get_tracer("agent.poller")
        self.metrics = get_metrics_client()
        self.logger = get_logger("agent.poller")

        # Team context and Zendesk configuration
        self.team_name: str = config.get("team", "operations")
        self.zendesk_auth: Optional[ZendeskAuth] = None
        self.zendesk_configs: Dict[str, ZendeskQueueConfig] = {}
        self.zendesk_mcp_server = None
        self.poll_tasks: Dict[str, asyncio.Task] = {}
        
        # Polling configuration
        self.max_concurrent_polls = config.get("max_concurrent_polls", 5)
        self.default_poll_interval = config.get("default_poll_interval", 30)
        self.max_results_history = config.get("max_results_history", 1000)
        self.auto_assign_enabled = config.get("auto_assign_enabled", False)
        
        # Initialize Zendesk MCP server
        self._initialize_zendesk_server()
        # Load optional zendesk config from agent config
        self._load_zendesk_from_config(config)

    def _load_zendesk_from_config(self, config: AgentConfig) -> None:
        """Load Zendesk auth and queues from agent configuration if provided."""
        try:
            zendesk_cfg = config.get("zendesk", {})
            if not zendesk_cfg:
                return
            auth_cfg = zendesk_cfg.get("auth", {})
            self.zendesk_auth = ZendeskAuth(**auth_cfg) if auth_cfg else None
            queues_cfg = zendesk_cfg.get("queues", [])
            for q in queues_cfg:
                qc = ZendeskQueueConfig(
                    queue_name=q.get("name") or q.get("queue_name"),
                    poll_interval=q.get("poll_interval", self.default_poll_interval),
                    max_tickets_per_poll=q.get("max_tickets_per_poll", 50),
                    enabled=q.get("enabled", True),
                    priority_threshold=q.get("priority_threshold", "normal"),
                    auto_assign=q.get("auto_assign", False),
                    assignee_id=q.get("assignee_id"),
                    tags=q.get("tags", []),
                )
                self.add_queue_config(qc)
        except Exception as e:
            self.logger.error(f"Failed to load Zendesk config from agent config: {e}")

    def _initialize_zendesk_server(self):
        """Initialize the Zendesk MCP server."""
        try:
            self.zendesk_mcp_server = ZendeskMCPServer(port=8088)
            self.logger.info("Zendesk MCP server initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Zendesk MCP server: {e}")

    def add_queue_config(self, config: ZendeskQueueConfig):
        """Add a Zendesk queue configuration (keyed by queue name)."""
        self.zendesk_configs[config.queue_name] = config

        # Register team-level auth once with MCP server
        if self.zendesk_mcp_server and self.zendesk_auth:
            # Only register if not already
            team_key = self.team_name
            if team_key not in self.zendesk_mcp_server.zendesk_tools.configs:
                zendesk_config = ZendeskConfig(
                    team_id=team_key,
                    email=self.zendesk_auth.email,
                    api_token=self.zendesk_auth.api_token,
                    username=self.zendesk_auth.username,
                    password=self.zendesk_auth.password,
                    base_url=self.zendesk_auth.base_url,
                    subdomain=self.zendesk_auth.subdomain,
                    poll_interval=config.poll_interval,
                    max_tickets_per_poll=config.max_tickets_per_poll,
                )
                self.zendesk_mcp_server.register_team_config(team_key, zendesk_config)

        self.logger.info(f"Added Zendesk queue config: {config.queue_name} (team: {self.team_name})")

    def build_graph(self) -> StateGraph:
        """Build the LangGraph for the Zendesk poller agent."""
        graph = StateGraph(ZendeskPollerState)

        # Minimal pipeline: poll -> forward
        graph.add_node("check_queues", self._check_queues)
        graph.add_node("poll_queue", self._poll_queue)
        graph.add_node("forward_to_triage", self._forward_to_triage)

        # Edges
        graph.add_edge("check_queues", "poll_queue")
        graph.add_edge("poll_queue", "forward_to_triage")
        graph.add_edge("forward_to_triage", END)

        # Set entry point
        graph.set_entry_point("check_queues")

        return graph.compile()

    async def process(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Process a Zendesk polling request through the workflow."""
        with self.tracer.start_as_current_span("zendesk_poller_process") as span:
            span.set_attribute("team", state.input_data.get("team", "unknown"))

            try:
                # Execute the graph
                result = await self.graph.ainvoke(state.dict())

                span.set_attribute("tickets_forwarded", len(result.get("tickets", [])))
                span.set_attribute("success", result.get("success", False))

                return result

            except Exception as e:
                self.logger.error(f"ZendeskPollerAgent error: {str(e)}")
                span.record_exception(e)
                raise AgentError(f"Zendesk polling failed: {str(e)}")

    async def _check_queues(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Check which queues need to be polled."""
        try:
            current_time = datetime.now()
            queues_to_poll = []

            for queue_name, config in self.zendesk_configs.items():
                if not config.enabled:
                    continue

                # Check if it's time to poll this queue
                last_poll = state.last_poll_times.get(queue_name)
                if last_poll is None or (current_time - last_poll).seconds >= config.poll_interval:
                    queues_to_poll.append(queue_name)

            state.queues_to_poll = queues_to_poll
            state.last_poll = current_time

            self.logger.info(f"Queues to poll: {queues_to_poll}")
            return state.dict()

        except Exception as e:
            self.logger.error(f"Error checking queues: {e}")
            state.queues_to_poll = []
            return state.dict()

    async def _poll_queue(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Poll Zendesk queues for new tickets."""
        try:
            all_tickets = []
            poll_results = []

            for queue_name in state.queues_to_poll:
                config = self.zendesk_configs[queue_name]
                
                # Use MCP client to poll queue
                if self.tool_registry:
                    poll_result = await self.tool_registry.call_tool(
                        "poll_queue",
                        {
                            "team": self.team_name,
                            "queue_name": config.queue_name,
                            "status": "new",
                            "limit": config.max_tickets_per_poll
                        }
                    )
                    
                    if poll_result.get("success"):
                        tickets = poll_result.get("tickets", [])
                        all_tickets.extend(tickets)
                        
                        # Create poll result
                        result = ZendeskPollResult(
                            team=self.team_name,
                            queue_name=config.queue_name,
                            timestamp=datetime.now(),
                            success=True,
                            ticket_count=len(tickets),
                            tickets=tickets,
                            processing_time=0.0,
                            new_tickets=len(tickets),
                            updated_tickets=0
                        )
                        poll_results.append(result)
                        
                        # Update last poll time
                        state.last_poll_times[queue_name] = datetime.now()
                    else:
                        self.logger.error(f"Failed to poll queue {queue_name}: {poll_result.get('error')}")

            state.poll_results = poll_results
            state.tickets = all_tickets

            self.logger.info(f"Polled {len(all_tickets)} tickets from {len(state.queues_to_poll)} queues")
            return state.dict()

        except Exception as e:
            self.logger.error(f"Error polling queues: {e}")
            state.poll_results = []
            state.tickets = []
            return state.dict()

    async def _forward_to_triage(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Forward polled tickets to the triage agent (no decision-making here)."""
        try:
            payload = {
                "source_agent": self.agent_id,
                "team": self.team_name,
                "queues": list(state.queues_to_poll),
                "tickets": state.tickets,
                "polled_at": datetime.now().isoformat(),
            }

            # Store minimal audit log
            if self.memory:
                await self.memory.store(
                    key=f"poller_forward:{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    value=payload,
                )

            # Optionally invoke triage directly if configured
            forward_mode = getattr(self.config, "forward_mode", "return")
            triage_result = None
            if forward_mode == "invoke":
                try:
                    from agents.triage.agent import TriageAgent
                    triage_agent = TriageAgent({
                        "team": self.team_name,
                        "model": getattr(self.config, "model", "{CENTRAL_LLM_GATEWAY_URL}/v1/chat/completions"),
                    }, tool_registry=self.tool_registry, memory=self.memory, rag=self.rag)
                    triage_result = await triage_agent.invoke({
                        "tickets": state.tickets,
                        "context": {"queues": list(state.queues_to_poll)},
                    })
                except Exception as e:
                    self.logger.error(f"Direct triage invocation failed: {e}")

            return {
                "success": True,
                "forwarded_to": "triage",
                "tickets": state.tickets,
                "ticket_count": len(state.tickets),
                "triage_result": triage_result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error forwarding to triage: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_tickets(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Analyze tickets for priority, routing, and processing needs."""
        try:
            analyzed_tickets = []

            for ticket in state.tickets:
                analysis = {
                    "ticket_id": ticket.get("id"),
                    "subject": ticket.get("subject"),
                    "priority": ticket.get("priority", "normal"),
                    "status": ticket.get("status"),
                    "requester_id": ticket.get("requester_id"),
                    "assignee_id": ticket.get("assignee_id"),
                    "tags": ticket.get("tags", []),
                    "created_at": ticket.get("created_at"),
                    "updated_at": ticket.get("updated_at"),
                    "description": ticket.get("description", ""),
                    "analysis": {
                        "urgency": self._assess_urgency(ticket),
                        "category": self._categorize_ticket(ticket),
                        "routing_team": self._determine_routing_team(ticket),
                        "action_required": self._determine_action_required(ticket),
                        "estimated_effort": self._estimate_effort(ticket)
                    }
                }
                analyzed_tickets.append(analysis)

            state.analyzed_tickets = analyzed_tickets

            self.logger.info(f"Analyzed {len(analyzed_tickets)} tickets")
            return state.dict()

        except Exception as e:
            self.logger.error(f"Error analyzing tickets: {e}")
            state.analyzed_tickets = []
            return state.dict()

    async def _route_tickets(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Route tickets to appropriate teams or assignees."""
        try:
            routing_decisions = []

            for ticket in state.analyzed_tickets:
                analysis = ticket["analysis"]
                routing_team = analysis["routing_team"]
                
                # Determine if ticket needs reassignment
                current_assignee = ticket.get("assignee_id")
                target_assignee = self._get_target_assignee(ticket, routing_team)
                
                routing_decision = {
                    "ticket_id": ticket["ticket_id"],
                    "current_team": ticket.get("team", "unknown"),
                    "routing_team": routing_team,
                    "current_assignee": current_assignee,
                    "target_assignee": target_assignee,
                    "needs_reassignment": current_assignee != target_assignee,
                    "priority": analysis["urgency"],
                    "action_required": analysis["action_required"],
                    "estimated_effort": analysis["estimated_effort"]
                }
                routing_decisions.append(routing_decision)

            state.routing_decisions = routing_decisions

            self.logger.info(f"Made routing decisions for {len(routing_decisions)} tickets")
            return state.dict()

        except Exception as e:
            self.logger.error(f"Error routing tickets: {e}")
            state.routing_decisions = []
            return state.dict()

    async def _update_tickets(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Update tickets based on routing decisions."""
        try:
            update_results = []

            for decision in state.routing_decisions:
                ticket_id = decision["ticket_id"]
                team = decision.get("current_team") or self.team_name
                
                # Update ticket if needed
                if decision["needs_reassignment"] and decision["target_assignee"]:
                    if self.tool_registry:
                        assign_result = await self.tool_registry.call_tool(
                            "assign_ticket",
                            {
                                "team": team,
                                "ticket_id": ticket_id,
                                "assignee_id": decision["target_assignee"],
                                "comment": f"Auto-assigned by Poller Agent based on analysis"
                            }
                        )
                        update_results.append(assign_result)

                # Add tags if configured
                # Map by queue name; use tags from the specific queue if available
                # If not available, fall back to a default empty list
                config = None
                if isinstance(self.zendesk_configs, dict):
                    # choose first queue config as default for tags if needed
                    config = next(iter(self.zendesk_configs.values()), None)
                if config and config.tags:
                    if self.tool_registry:
                        tag_result = await self.tool_registry.call_tool(
                            "add_ticket_tags",
                            {
                                "team": self.team_name,
                                "ticket_id": ticket_id,
                                "tags": config.tags
                            }
                        )
                        update_results.append(tag_result)

            state.update_results = update_results

            self.logger.info(f"Updated {len(update_results)} tickets")
            return state.dict()

        except Exception as e:
            self.logger.error(f"Error updating tickets: {e}")
            state.update_results = []
            return state.dict()

    async def _store_results(self, state: ZendeskPollerState) -> Dict[str, Any]:
        """Store polling results in memory and metrics."""
        try:
            # Store results in memory
            if self.memory:
                await self.memory.store(
                    key=f"poller_results:{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    value={
                        "timestamp": datetime.now().isoformat(),
                        "team": state.input_data.get("team", "all"),
                        "tickets_processed": len(state.tickets),
                        "routing_decisions": len(state.routing_decisions),
                        "update_results": len(state.update_results)
                    }
                )

            # Update metrics
            self.metrics.increment("poller.tickets_processed", len(state.tickets))
            self.metrics.increment("poller.routing_decisions", len(state.routing_decisions))
            self.metrics.increment("poller.update_operations", len(state.update_results))

            # Prepare final result
            final_result = {
                "success": True,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "tickets_processed": len(state.tickets),
                "routing_decisions": len(state.routing_decisions),
                "update_operations": len(state.update_results),
                "teams_polled": list(state.queues_to_poll),
                "poll_results": [result.dict() for result in state.poll_results]
            }

            state.final_result = final_result
            return state.dict()

        except Exception as e:
            self.logger.error(f"Error storing results: {e}")
            state.final_result = {
                "success": False,
                "error": str(e),
                "agent_id": self.agent_id
            }
            return state.dict()

    def _assess_urgency(self, ticket: Dict[str, Any]) -> str:
        """Assess ticket urgency based on content and metadata."""
        subject = ticket.get("subject", "").lower()
        description = ticket.get("description", "").lower()
        priority = ticket.get("priority", "normal")
        tags = ticket.get("tags", [])

        # High urgency keywords
        high_urgency_keywords = ["urgent", "critical", "emergency", "down", "outage", "broken", "failed"]
        
        # Check for high urgency indicators
        if priority in ["urgent", "high"]:
            return "high"
        
        if any(keyword in subject or keyword in description for keyword in high_urgency_keywords):
            return "high"
        
        if any(tag.lower() in high_urgency_keywords for tag in tags):
            return "high"
        
        # Medium urgency keywords
        medium_urgency_keywords = ["important", "issue", "problem", "slow", "performance"]
        
        if any(keyword in subject or keyword in description for keyword in medium_urgency_keywords):
            return "medium"
        
        return "low"

    def _categorize_ticket(self, ticket: Dict[str, Any]) -> str:
        """Categorize ticket based on content and tags."""
        subject = ticket.get("subject", "").lower()
        description = ticket.get("description", "").lower()
        tags = ticket.get("tags", [])

        # Category keywords
        categories = {
            "bug": ["bug", "error", "issue", "problem", "broken", "failed"],
            "feature": ["feature", "enhancement", "improvement", "request"],
            "question": ["question", "help", "how", "what", "why"],
            "incident": ["incident", "outage", "down", "critical", "emergency"],
            "maintenance": ["maintenance", "update", "upgrade", "patch"]
        }

        for category, keywords in categories.items():
            if any(keyword in subject or keyword in description for keyword in keywords):
                return category
        
        return "general"

    def _determine_routing_team(self, ticket: Dict[str, Any]) -> str:
        """Determine which team should handle the ticket."""
        subject = ticket.get("subject", "").lower()
        description = ticket.get("description", "").lower()
        tags = ticket.get("tags", [])
        category = self._categorize_ticket(ticket)

        # Team routing rules
        if category == "incident":
            return "operations"
        elif category == "bug":
            return "engineering"
        elif category == "feature":
            return "engineering"
        elif "security" in subject or "security" in description:
            return "security"
        elif "deploy" in subject or "deploy" in description:
            return "devops"
        elif "monitor" in subject or "monitor" in description:
            return "operations"
        else:
            return "operations"  # Default team

    def _determine_action_required(self, ticket: Dict[str, Any]) -> str:
        """Determine what action is required for the ticket."""
        urgency = self._assess_urgency(ticket)
        category = self._categorize_ticket(ticket)

        if urgency == "high" and category == "incident":
            return "immediate_response"
        elif urgency == "high":
            return "urgent_attention"
        elif category == "bug":
            return "investigation"
        elif category == "feature":
            return "planning"
        else:
            return "standard_processing"

    def _estimate_effort(self, ticket: Dict[str, Any]) -> str:
        """Estimate effort required for the ticket."""
        urgency = self._assess_urgency(ticket)
        category = self._categorize_ticket(ticket)
        description_length = len(ticket.get("description", ""))

        if urgency == "high" or category == "incident":
            return "high"
        elif description_length > 500 or category == "feature":
            return "medium"
        else:
            return "low"

    def _get_target_assignee(self, ticket: Dict[str, Any], routing_team: str) -> Optional[str]:
        """Get target assignee for the ticket."""
        config = self.zendesk_configs.get(routing_team)
        
        if config and config.auto_assign and config.assignee_id:
            return config.assignee_id
        
        return None

    async def start_polling(self):
        """Start continuous polling of configured queues."""
        try:
            self.logger.info("Starting Zendesk queue polling")
            
            # Start MCP server
            if self.zendesk_mcp_server:
                await self.zendesk_mcp_server.start()
            
            # Start polling tasks for each configured queue
            for queue_name, config in self.zendesk_configs.items():
                if config.enabled:
                    task = asyncio.create_task(self._poll_loop(queue_name, config))
                    self.poll_tasks[queue_name] = task
            
            self.logger.info(f"Started polling for {len(self.poll_tasks)} queues")
            
        except Exception as e:
            self.logger.error(f"Failed to start polling: {e}")
            raise AgentError(f"Failed to start polling: {e}")

    async def stop_polling(self):
        """Stop continuous polling."""
        try:
            self.logger.info("Stopping Zendesk queue polling")
            
            # Cancel all polling tasks
            for task in self.poll_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self.poll_tasks:
                await asyncio.gather(*self.poll_tasks.values(), return_exceptions=True)
            
            self.poll_tasks.clear()
            self.logger.info("Stopped all polling tasks")
            
        except Exception as e:
            self.logger.error(f"Failed to stop polling: {e}")

    async def _poll_loop(self, queue_name: str, config: ZendeskQueueConfig):
        """Continuous polling loop for a specific team's queue."""
        while True:
            try:
                # Create state for polling
                state = ZendeskPollerState(
                    input_data={"team": self.team_name},
                    queue_configs=[config],
                    last_poll_times={},
                    queues_to_poll=[queue_name]
                )
                
                # Process the polling workflow
                result = await self.process(state)
                
                if result.get("success"):
                    self.logger.info(f"Successfully polled queue {queue_name}")
                else:
                    self.logger.error(f"Failed to poll queue {queue_name}: {result.get('error')}")
                
                # Wait for next poll
                await asyncio.sleep(config.poll_interval)
                
            except asyncio.CancelledError:
                self.logger.info(f"Polling cancelled for queue {queue_name}")
                break
            except Exception as e:
                self.logger.error(f"Error in polling loop for queue {queue_name}: {e}")
                await asyncio.sleep(config.poll_interval)  # Wait before retrying

    async def get_queue_stats(self, team: str) -> Dict[str, Any]:
        """Get statistics for a specific team's queue."""
        try:
            if self.tool_registry:
                stats_result = await self.tool_registry.call_tool(
                    "get_queue_stats",
                    {"team": team}
                )
                return stats_result
            else:
                return {"success": False, "error": "Tool registry not available"}
                
        except Exception as e:
            self.logger.error(f"Error getting queue stats: {e}")
            return {"success": False, "error": str(e)}

    async def run_once(self) -> Dict[str, Any]:
        """Run a single end-to-end polling cycle across all configured queues."""
        state = ZendeskPollerState(
            input_data={"team": self.team_name},
            queue_configs=list(self.zendesk_configs.values()),
            last_poll_times={},
            queues_to_poll=list(self.zendesk_configs.keys()),
        )
        return await self.process(state)