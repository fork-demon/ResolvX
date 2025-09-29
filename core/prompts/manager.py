"""
Prompt manager for loading and managing agent prompts.

Provides centralized prompt management with template processing,
variable substitution, and prompt versioning.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from core.config import Config
from core.exceptions import ConfigurationError
from core.observability import get_logger
from core.prompts.template import PromptTemplate
from core.prompts.loader import PromptLoader


class PromptManager:
    """
    Manages prompts for all agents in the framework.
    
    Provides centralized prompt loading, template processing,
    and prompt versioning capabilities.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize prompt manager.
        
        Args:
            config: Framework configuration
        """
        self.config = config
        self.logger = get_logger("prompts.manager")
        
        # Prompt storage
        self._prompts: Dict[str, Dict[str, PromptTemplate]] = {}
        self._prompt_versions: Dict[str, List[str]] = {}
        
        # Template variables
        self._template_vars: Dict[str, Any] = {}
        
        # Prompt loader
        self._loader = PromptLoader()

    async def initialize(self) -> None:
        """Initialize the prompt manager."""
        try:
            # Load template variables from config
            await self._load_template_variables()
            
            # Load all agent prompts
            await self._load_agent_prompts()
            
            self.logger.info(f"Prompt manager initialized with {len(self._prompts)} agents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize prompt manager: {e}")
            raise ConfigurationError(f"Prompt manager initialization failed: {e}") from e

    async def get_system_prompt(
        self, 
        agent_name: str, 
        version: Optional[str] = None
    ) -> str:
        """
        Get system prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            version: Optional prompt version
            
        Returns:
            System prompt text
            
        Raises:
            ConfigurationError: If prompt not found
        """
        try:
            if agent_name not in self._prompts:
                raise ConfigurationError(f"No prompts found for agent: {agent_name}")
            
            agent_prompts = self._prompts[agent_name]
            
            if "system" not in agent_prompts:
                raise ConfigurationError(f"No system prompt found for agent: {agent_name}")
            
            system_prompt = agent_prompts["system"]
            
            # Apply version if specified
            if version and version in self._prompt_versions.get(agent_name, []):
                system_prompt = await self._get_versioned_prompt(agent_name, "system", version)
            
            # Process template
            return await system_prompt.render(self._template_vars)
            
        except Exception as e:
            self.logger.error(f"Failed to get system prompt for {agent_name}: {e}")
            raise ConfigurationError(f"System prompt retrieval failed: {e}") from e

    async def get_runtime_prompt(
        self, 
        agent_name: str, 
        prompt_name: str,
        version: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Get runtime prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            prompt_name: Name of the runtime prompt
            version: Optional prompt version
            **kwargs: Additional template variables
            
        Returns:
            Runtime prompt text
            
        Raises:
            ConfigurationError: If prompt not found
        """
        try:
            if agent_name not in self._prompts:
                raise ConfigurationError(f"No prompts found for agent: {agent_name}")
            
            agent_prompts = self._prompts[agent_name]
            
            if prompt_name not in agent_prompts:
                raise ConfigurationError(f"No runtime prompt '{prompt_name}' found for agent: {agent_name}")
            
            runtime_prompt = agent_prompts[prompt_name]
            
            # Apply version if specified
            if version and version in self._prompt_versions.get(agent_name, []):
                runtime_prompt = await self._get_versioned_prompt(agent_name, prompt_name, version)
            
            # Merge template variables
            template_vars = {**self._template_vars, **kwargs}
            
            # Process template
            return await runtime_prompt.render(template_vars)
            
        except Exception as e:
            self.logger.error(f"Failed to get runtime prompt {prompt_name} for {agent_name}: {e}")
            raise ConfigurationError(f"Runtime prompt retrieval failed: {e}") from e

    async def list_agent_prompts(self, agent_name: str) -> List[str]:
        """
        List available prompts for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of prompt names
        """
        if agent_name not in self._prompts:
            return []
        
        return list(self._prompts[agent_name].keys())

    async def list_prompt_versions(self, agent_name: str) -> List[str]:
        """
        List available versions for an agent's prompts.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of version strings
        """
        return self._prompt_versions.get(agent_name, [])

    async def add_prompt(
        self,
        agent_name: str,
        prompt_name: str,
        prompt_template: Union[str, PromptTemplate],
        version: Optional[str] = None
    ) -> None:
        """
        Add a new prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            prompt_name: Name of the prompt
            prompt_template: Prompt template or text
            version: Optional version identifier
        """
        try:
            if agent_name not in self._prompts:
                self._prompts[agent_name] = {}
            
            if isinstance(prompt_template, str):
                prompt_template = PromptTemplate(prompt_template)
            
            # Store prompt
            self._prompts[agent_name][prompt_name] = prompt_template
            
            # Track version if specified
            if version:
                if agent_name not in self._prompt_versions:
                    self._prompt_versions[agent_name] = []
                if version not in self._prompt_versions[agent_name]:
                    self._prompt_versions[agent_name].append(version)
            
            self.logger.info(f"Added prompt {prompt_name} for agent {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add prompt {prompt_name} for {agent_name}: {e}")
            raise ConfigurationError(f"Prompt addition failed: {e}") from e

    async def update_template_variables(self, variables: Dict[str, Any]) -> None:
        """
        Update template variables.
        
        Args:
            variables: New template variables
        """
        self._template_vars.update(variables)
        self.logger.debug(f"Updated template variables: {list(variables.keys())}")

    async def reload_agent_prompts(self, agent_name: str) -> None:
        """
        Reload prompts for a specific agent.
        
        Args:
            agent_name: Name of the agent
        """
        try:
            await self._load_agent_prompts(agent_name)
            self.logger.info(f"Reloaded prompts for agent: {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to reload prompts for {agent_name}: {e}")
            raise ConfigurationError(f"Prompt reload failed: {e}") from e

    async def _load_template_variables(self) -> None:
        """Load template variables from configuration."""
        if not self.config:
            return
        
        # Load from global settings
        global_settings = self.config.global_settings or {}
        
        # Load from environment settings
        from core.config import EnvironmentSettings
        env_settings = EnvironmentSettings()
        
        # Merge variables
        self._template_vars = {
            "ORG_NAME": env_settings.org_name,
            "CENTRAL_LLM_GATEWAY_URL": env_settings.central_llm_gateway_url,
            "CENTRAL_MCP_GATEWAY_URL": env_settings.central_mcp_gateway_url,
            "DEFAULT_AGENT_VERSION": env_settings.default_agent_version,
            "DEFAULT_EMBEDDING_MODEL": env_settings.default_embedding_model,
            "DEFAULT_CONFIDENCE_THRESHOLDS": {
                "triage": env_settings.default_confidence_threshold_triage,
                "metrics": env_settings.default_confidence_threshold_metrics,
                "poller": env_settings.default_confidence_threshold_poller,
            },
            **global_settings,
        }

    async def _load_agent_prompts(self, agent_name: Optional[str] = None) -> None:
        """Load prompts for agents."""
        if not self.config:
            return
        
        agents_to_load = [agent_name] if agent_name else list(self.config.agents.keys())
        
        for agent in agents_to_load:
            try:
                await self._load_single_agent_prompts(agent)
            except Exception as e:
                self.logger.error(f"Failed to load prompts for agent {agent}: {e}")

    async def _load_single_agent_prompts(self, agent_name: str) -> None:
        """Load prompts for a single agent."""
        agent_config = self.config.agents.get(agent_name)
        if not agent_config:
            return
        
        # Initialize agent prompts
        if agent_name not in self._prompts:
            self._prompts[agent_name] = {}
        
        # Load system prompt
        if hasattr(agent_config, 'system_prompt') and agent_config.system_prompt:
            system_template = PromptTemplate(agent_config.system_prompt)
            self._prompts[agent_name]["system"] = system_template
        
        # Load runtime prompts
        if hasattr(agent_config, 'runtime_prompts') and agent_config.runtime_prompts:
            for prompt_name, prompt_text in agent_config.runtime_prompts.items():
                runtime_template = PromptTemplate(prompt_text)
                self._prompts[agent_name][prompt_name] = runtime_template
        
        # Load prompts from files if specified
        await self._load_prompt_files(agent_name)

    async def _load_prompt_files(self, agent_name: str) -> None:
        """Load prompts from files for an agent."""
        # Look for prompt files in agent directory
        agent_dir = Path(f"agents/{agent_name}/prompts")
        if not agent_dir.exists():
            return
        
        # Load system prompt file
        system_file = agent_dir / "system.md"
        if system_file.exists():
            system_content = system_file.read_text(encoding="utf-8")
            system_template = PromptTemplate(system_content)
            self._prompts[agent_name]["system"] = system_template
        
        # Load runtime prompt files
        for prompt_file in agent_dir.glob("*.json"):
            if prompt_file.name == "system.md":
                continue
            
            try:
                prompt_data = await self._loader.load_prompt_file(prompt_file)
                for prompt_name, prompt_text in prompt_data.items():
                    runtime_template = PromptTemplate(prompt_text)
                    self._prompts[agent_name][prompt_name] = runtime_template
                    
            except Exception as e:
                self.logger.warning(f"Failed to load prompt file {prompt_file}: {e}")

    async def _get_versioned_prompt(
        self, 
        agent_name: str, 
        prompt_name: str, 
        version: str
    ) -> PromptTemplate:
        """Get a versioned prompt template."""
        # This would implement versioning logic
        # For now, return the current prompt
        return self._prompts[agent_name][prompt_name]

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on prompt manager.
        
        Returns:
            Health check results
        """
        try:
            total_agents = len(self._prompts)
            total_prompts = sum(len(prompts) for prompts in self._prompts.values())
            
            return {
                "status": "healthy",
                "total_agents": total_agents,
                "total_prompts": total_prompts,
                "template_variables": len(self._template_vars),
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
