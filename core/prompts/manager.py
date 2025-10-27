"""
Prompt Manager

High-level interface for managing prompts across agents.
"""
from typing import Dict, Any, Optional
import logging

from core.prompts.loader import PromptLoader, PromptTemplate


class PromptManager:
    """
    Manages prompts for agents.
    
    Provides:
    - Centralized prompt loading
    - Caching and hot reload
    - Variable substitution
    - Prompt validation
    """
    
    _instance: Optional['PromptManager'] = None
    
    def __init__(self, base_dir: str = "prompts"):
        self.loader = PromptLoader(base_dir)
        self.logger = logging.getLogger(__name__)
        self._agent_prompts: Dict[str, Dict[str, PromptTemplate]] = {}
    
    @classmethod
    def get_instance(cls, base_dir: str = "prompts") -> 'PromptManager':
        """Get singleton instance of PromptManager."""
        if cls._instance is None:
            cls._instance = cls(base_dir)
        return cls._instance
    
    def load_agent_prompts(
        self,
        agent_name: str,
        prompt_config: Dict[str, Any]
    ) -> Dict[str, PromptTemplate]:
        """
        Load all prompts for an agent.
        
        Args:
            agent_name: Name of the agent
            prompt_config: Prompt configuration from agent.yaml
            
        Returns:
            Dictionary of prompt templates
        """
        if agent_name in self._agent_prompts:
            return self._agent_prompts[agent_name]
        
        prompts = self.loader.load_prompts_from_config(prompt_config)
        self._agent_prompts[agent_name] = prompts
        
        self.logger.info(f"Loaded {len(prompts)} prompts for agent: {agent_name}")
        return prompts
    
    def get_prompt(
        self,
        agent_name: str,
        prompt_name: str,
        **render_vars
    ) -> str:
        """
        Get a rendered prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            prompt_name: Name of the prompt
            **render_vars: Variables to inject when rendering
            
        Returns:
            Rendered prompt string
        """
        if agent_name not in self._agent_prompts:
            self.logger.warning(f"No prompts loaded for agent: {agent_name}")
            return ""
        
        prompts = self._agent_prompts[agent_name]
        
        if prompt_name not in prompts:
            self.logger.warning(f"Prompt {prompt_name} not found for agent {agent_name}")
            return ""
        
        prompt = prompts[prompt_name]
        return self.loader.render_prompt(prompt, **render_vars)
    
    def reload_agent_prompts(self, agent_name: str):
        """
        Reload all prompts for an agent.
        
        Useful for development and hot reload.
        """
        if agent_name in self._agent_prompts:
            del self._agent_prompts[agent_name]
            self.logger.info(f"Cleared prompts for agent: {agent_name}")
    
    def clear_all_caches(self):
        """Clear all prompt caches."""
        self.loader.clear_cache()
        self._agent_prompts.clear()
        self.logger.info("Cleared all prompt caches")

