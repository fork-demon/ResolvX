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

from core.config import Config
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
        self.tool_registry = ToolRegistry(config.gateway)
        self.llm_client = LLMGatewayClient(config.gateway)
        self.prompt_manager = PromptManager()
        
        # Initialize graph executor
        self.graph_executor = GraphExecutor(
            memory_factory=self.memory_factory,
            tool_registry=self.tool_registry,
            llm_client=self.llm_client,
            prompt_manager=self.prompt_manager,
        )

    async def start_agents(self):
        """Start all configured agents."""
        self.logger.info("Starting Golden Agent Framework...")
        
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
            # Build minimal config dict for agent
            poller_cfg: Dict[str, Any] = {
                "team": agent_config.team,
                "max_concurrent_polls": getattr(agent_config, "max_concurrent_polls", 5),
                "default_poll_interval": getattr(agent_config, "default_poll_interval", 30),
                "max_results_history": getattr(agent_config, "max_results_history", 1000),
                "auto_assign_enabled": getattr(agent_config, "auto_assign_enabled", False),
                "zendesk": getattr(agent_config, "zendesk", {}),
            }
            agent = ZendeskPollerAgent(poller_cfg, tool_registry=self.tool_registry, memory=None, rag=None)

            # Check for schedule (cron string). If present, run on schedule; else start continuous polling.
            schedule: Optional[str] = getattr(agent_config, "schedule", None)
            if schedule:
                self.logger.info(f"Scheduling poller with cron: {schedule}")
                await self._schedule_poller(agent, schedule)
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

        # Default: log and return mock instance
        self.logger.info(f"Creating {agent_name} agent with type: {agent_config.type}")
        self.logger.info(f"Team: {agent_config.team}")
        self.logger.info(f"Prompts: {len(agent_config.prompts.runtime_prompts) if hasattr(agent_config, 'prompts') else 0}")

        return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # Splunk agent
        if agent_name == "splunk" and agent_config.enabled:
            from agents.splunk.agent import SplunkAgent
            agent = SplunkAgent({
                "endpoint": getattr(agent_config, "endpoint", None),
                "username": getattr(agent_config, "username", None),
                "password": getattr(agent_config, "password", None),
                "knowledge_dir": getattr(agent_config, "knowledge_dir", "knowledge/splunk"),
            })
            self.agents[agent_name] = agent
            return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

        # New Relic agent
        if agent_name == "newrelic" and agent_config.enabled:
            from agents.newrelic.agent import NewRelicAgent
            agent = NewRelicAgent({
                "api_key": getattr(agent_config, "api_key", None),
                "account_id": getattr(agent_config, "account_id", None),
                "region": getattr(agent_config, "region", "US"),
                "knowledge_dir": getattr(agent_config, "knowledge_dir", "knowledge/newrelic"),
            })
            self.agents[agent_name] = agent
            return {"name": agent_name, "type": agent_config.type, "team": agent_config.team, "status": "running"}

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
        config = Config.load()
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
    
    # Print status
    status = agent_manager.get_agent_status()
    print(f"\nAgent Status:")
    print(f"  Total agents: {status['total_agents']}")
    print(f"  Running agents: {status['running_agents']}")
    print(f"  Agents: {list(status['agents'].keys())}")
    
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
