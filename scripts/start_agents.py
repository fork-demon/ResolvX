#!/usr/bin/env python3
"""
Start Golden Agent Framework Agents.

This script starts all configured agents for local development.
"""

import asyncio
import logging
import signal
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import Config, load_config
from core.observability.factory import configure_observability
from core.graph.executor import GraphExecutor
from core.memory.factory import MemoryFactory
from core.gateway.tool_registry import ToolRegistry
from core.gateway.llm_client import LLMGatewayClient
from core.prompts.manager import PromptManager


class AgentManager:
    """Manages all agents in the framework."""

    def __init__(self, config: Config):
        """
        Initialize agent manager.
        
        Args:
            config: Framework configuration
        """
        self.config = config
        self.logger = logging.getLogger("agent.manager")
        self.agents: Dict[str, Any] = {}
        
        # Initialize core components
        self.memory_factory = MemoryFactory()
        # Tool registry is initialized in start_agents to allow per-agent overrides
        self.tool_registry = None
        self.llm_client = LLMGatewayClient(config.gateway)
        self.prompt_manager = PromptManager()
        
        # Initialize graph executor (no-arg constructor)
        self.graph_executor = GraphExecutor()

    async def start_agents(self):
        """Start all configured agents."""
        self.logger.info("Starting Golden Agent Framework...")

        # Initialize default MCP-backed ToolRegistry (central gateway)
        try:
            from core.gateway.mcp_client import MCPClient
            default_mcp_client = MCPClient(self.config)
            await default_mcp_client.initialize()
            self.tool_registry = ToolRegistry(self.config.gateway, mcp_client=default_mcp_client)
            await self.tool_registry.initialize()
        except Exception as e:
            self.logger.warning(f"ToolRegistry initialization failed or MCP unavailable: {e}")
            self.tool_registry = ToolRegistry(self.config.gateway)
        
        # Load agent configurations
        for agent_name, agent_config in self.config.agents.items():
            if agent_config.enabled:
                await self._start_agent(agent_name, agent_config)
        
        self.logger.info(f"Started {len(self.agents)} agents")

    async def _start_agent(self, agent_name: str, agent_config: Any):
        """Start a specific agent."""
        try:
            self.logger.info(f"Starting {agent_name} agent...")
            
            # Create agent instance (this would be implemented based on agent type)
            agent_instance = await self._create_agent_instance(agent_name, agent_config)
            
            if agent_instance:
                self.agents[agent_name] = agent_instance
                self.logger.info(f"Started {agent_name} agent successfully")
            else:
                self.logger.warning(f"Failed to start {agent_name} agent")
                
        except Exception as e:
            self.logger.error(f"Error starting {agent_name} agent: {e}")

    async def _create_agent_instance(self, agent_name: str, agent_config: Any):
        """Create an agent instance based on configuration."""
        # For poller, support scheduled run or continuous polling
        if agent_name == "poller" and agent_config.enabled:
            from agents.poller.agent import ZendeskPollerAgent
            tool_registry = await self._get_tool_registry_for_agent(agent_config)
            # Build minimal config dict for agent
            poller_cfg: Dict[str, Any] = {
                "team": agent_config.team,
                "max_concurrent_polls": getattr(agent_config, "max_concurrent_polls", 5),
                "default_poll_interval": getattr(agent_config, "default_poll_interval", 30),
                "max_results_history": getattr(agent_config, "max_results_history", 1000),
                "auto_assign_enabled": getattr(agent_config, "auto_assign_enabled", False),
                "zendesk": getattr(agent_config, "zendesk", {}),
            }
            agent = ZendeskPollerAgent(poller_cfg, tool_registry=tool_registry, memory=None, rag=None)

            # Check for schedule (cron string). If present, run on schedule; else start continuous polling.
            schedule: Optional[str] = getattr(agent_config, "schedule", None)
            if schedule:
                self.logger.info(f"Scheduling poller with cron: {schedule}")
                await self._schedule_poller(agent, schedule)
                # Trigger an immediate single run at startup
                try:
                    self.logger.info("Triggering initial poller run_once at startup")
                    await agent.run_once()
                except Exception as e:
                    self.logger.warning(f"Initial poller run_once failed: {e}")
            else:
                self.logger.info("Starting poller in continuous mode")
                await agent.start_polling()

            return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # Memory agent: create with in-memory vector backend and run idle (invoked by supervisor/poller)
        if agent_name == "memory" and agent_config.enabled:
            from agents.memory.agent import MemoryAgent
            # Determine backend; default to mock for local dev
            backend = getattr(agent_config, "backend", "mock")
            memory_cfg = getattr(agent_config, "memory_config", None) or {}
            try:
                memory = self.memory_factory.create_memory(backend, memory_cfg)
                await memory.initialize()
            except Exception as e:
                self.logger.warning(f"Failed to init memory backend '{backend}' ({e}); falling back to mock")
                memory = self.memory_factory.create_memory("mock", {})
                await memory.initialize()

            agent = MemoryAgent({
                "namespace_prefix": getattr(agent_config, "namespace_prefix", "tickets"),
                "search_limit": getattr(agent_config, "search_limit", 10),
                "search_threshold": getattr(agent_config, "search_threshold", 0.7),
                "forward_mode": getattr(agent_config, "forward_mode", "return"),
                "embedding_model": getattr(agent_config, "embedding_model", None),
            }, memory=memory)

            # No continuous loop required; supervisor or callers will invoke as needed
            self.agents[agent_name] = agent
            return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # Triage agent: performs CoT analysis and synthesis
        if agent_name == "triage" and agent_config.enabled:
            from agents.triage.agent import TriageAgent
            from core.observability import get_tracer
            
            agent = TriageAgent(config=agent_config, tracer=get_tracer())
            
            # No continuous loop required; invoked by Supervisor
            self.agents[agent_name] = agent
            return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # Supervisor agent: makes final decisions based on Triage synthesis
        if agent_name == "supervisor" and agent_config.enabled:
            from agents.supervisor.agent import SupervisorAgent
            from core.observability import get_tracer
            
            agent = SupervisorAgent(config=agent_config, tracer=get_tracer())
            
            # No continuous loop required; invoked by orchestrator
            self.agents[agent_name] = agent
            return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # Executor agent: executes Supervisor decisions on tickets
        if agent_name == "executor" and agent_config.enabled:
            from agents.executor.agent import TicketExecutorAgent
            from core.observability import get_tracer
            
            # Initialize Zendesk client (or mock for dev)
            zendesk_client = None  # TODO: Initialize real Zendesk client from config
            
            agent = TicketExecutorAgent(
                config=agent_config, 
                zendesk_client=zendesk_client,
                tracer=get_tracer()
            )
            
            # No continuous loop required; invoked by Supervisor
            self.agents[agent_name] = agent
            return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # Default: log and return mock instance
        self.logger.info(f"Creating {agent_name} agent with type: {getattr(agent_config, 'type', 'unknown')}")
        self.logger.info(f"Team: {getattr(agent_config, 'team', 'unknown')}")
        try:
            rp = (agent_config.prompts or {}).get('runtime_prompts', {}) if hasattr(agent_config, 'prompts') else {}
            self.logger.info(f"Prompts: {len(rp)}")
        except Exception:
            self.logger.info("Prompts: 0")

        return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # Note: Splunk and NewRelic are now TOOLS (via MCP gateway), not separate agents
        # They are called by the Triage agent as needed, based on LLM recommendations

    async def _schedule_poller(self, agent, cron_expr: str):
        """Schedule the poller agent using a simple cron-like scheduler (APScheduler if available)."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            scheduler = AsyncIOScheduler()
            trigger = CronTrigger.from_crontab(cron_expr)
            scheduler.add_job(agent.run_once, trigger=trigger, name="zendesk_poller_run_once")
            scheduler.start()
            self.logger.info("APScheduler started for poller")
        except Exception as e:
            self.logger.warning(f"APScheduler unavailable or failed ({e}); falling back to interval scheduler")
            # Fallback: if cron not available, run every 30 minutes by default
            interval_seconds = 30 * 60
            async def loop():
                while True:
                    try:
                        await agent.run_once()
                    except Exception as err:
                        self.logger.error(f"Scheduled poller run failed: {err}")
                    await asyncio.sleep(interval_seconds)
            asyncio.create_task(loop())

    async def _get_tool_registry_for_agent(self, agent_config: Any) -> ToolRegistry:
        """Return a ToolRegistry, honoring per-agent MCP gateway override if present."""
        try:
            agent_mcp = getattr(agent_config, "mcp_gateway", None)
            if agent_mcp and isinstance(agent_mcp, dict) and agent_mcp.get("url"):
                from core.gateway.mcp_client import MCPClient
                mcp_client = MCPClient(self.config)
                # Override gateway URL for this agent
                mcp_client.gateway_url = agent_mcp.get("url")
                await mcp_client.initialize()
                tr = ToolRegistry(self.config.gateway, mcp_client=mcp_client)
                await tr.initialize()
                return tr
        except Exception as e:
            self.logger.warning(f"Per-agent MCP override failed; falling back to default registry: {e}")

        # Fallback to default registry
        return self.tool_registry

    async def stop_agents(self):
        """Stop all agents."""
        self.logger.info("Stopping all agents...")
        
        for agent_name, agent in self.agents.items():
            try:
                self.logger.info(f"Stopping {agent_name} agent...")
                # Stop agent logic would go here
                agent["status"] = "stopped"
            except Exception as e:
                self.logger.error(f"Error stopping {agent_name} agent: {e}")
        
        self.logger.info("All agents stopped")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "total_agents": len(self.agents),
            "running_agents": len([a for a in self.agents.values() if a["status"] == "running"]),
            "agents": self.agents,
        }


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return
    
    # Configure observability (LangSmith or LangFuse based on config)
    try:
        configure_observability(config)
        print(f"Observability configured: backend={getattr(config.observability, 'backend', 'console')}")
    except Exception as e:
        print(f"Warning: Observability configuration failed: {e}")

    # Create agent manager
    agent_manager = AgentManager(config)
    
    # Start agents
    await agent_manager.start_agents()

    # Short verification window before exit (unless running as a service)
    verify_seconds = int(os.getenv("START_AGENTS_VERIFY_SECONDS", "5"))
    if verify_seconds > 0:
        try:
            await asyncio.sleep(verify_seconds)
        except Exception:
            pass

    # Print status
    status = agent_manager.get_agent_status()
    print(f"\nAgent Status:")
    print(f"  Total agents: {status['total_agents']}")
    print(f"  Running agents: {status['running_agents']}")
    print(f"  Agents: {list(status['agents'].keys())}")

    # Exit early after verification window if requested
    if verify_seconds > 0 and os.getenv("START_AGENTS_EXIT_AFTER_VERIFY", "1") == "1":
        return

    # Handle shutdown gracefully
    def signal_handler(signum, frame):
        print("\nShutting down agents...")
        asyncio.create_task(agent_manager.stop_agents())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep agents running
        await asyncio.Future()
    except KeyboardInterrupt:
        print("\nShutting down agents...")
        await agent_manager.stop_agents()


if __name__ == "__main__":
    asyncio.run(main())
