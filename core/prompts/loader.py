"""
Prompt Loader

Loads prompts from files with variable substitution.
Framework-based approach using Jinja2 for robust templating.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Represents a loaded prompt template."""
    name: str = Field(description="Prompt name")
    content: str = Field(description="Prompt content")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
    file_path: Optional[str] = Field(None, description="Source file path")
    
    def render(self, **kwargs) -> str:
        """Render the prompt with additional variables."""
        # Merge default variables with provided ones
        context = {**self.variables, **kwargs}
        
        # Use Jinja2 for template rendering
        template = Template(self.content)
        return template.render(**context)


class PromptLoader:
    """
    Loads prompts from files using Jinja2 templating.
    
    Framework-based approach:
    - Uses Jinja2 for robust variable substitution
    - Supports file-based and inline prompts
    - Handles missing files gracefully
    - Provides caching for performance
    """
    
    def __init__(self, base_dir: str = "prompts"):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, PromptTemplate] = {}
        
        # Setup Jinja2 environment
        if self.base_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.base_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
        else:
            self.jinja_env = None
            self.logger.warning(f"Prompts directory not found: {self.base_dir}")
    
    def load_prompt(
        self,
        path: str,
        variables: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> PromptTemplate:
        """
        Load a prompt from a file.
        
        Args:
            path: Path to prompt file (relative to base_dir)
            variables: Variables to inject into the template
            use_cache: Whether to use cached version
            
        Returns:
            PromptTemplate with content and variables
        """
        # Check cache first
        cache_key = f"{path}:{variables}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Prepare variables with defaults
        default_vars = self._get_default_variables()
        all_vars = {**default_vars, **(variables or {})}
        
        # Load from file
        try:
            # Handle if path already includes base_dir prefix
            if path.startswith("prompts/"):
                file_path = Path(path)
            else:
                file_path = self.base_dir / path
            
            if not file_path.exists():
                self.logger.warning(f"Prompt file not found: {file_path}")
                # Return empty prompt as fallback
                return PromptTemplate(
                    name=path,
                    content="",
                    variables=all_vars,
                    file_path=str(file_path)
                )
            
            # Read file content
            content = file_path.read_text(encoding="utf-8")
            
            # Create prompt template
            prompt = PromptTemplate(
                name=path,
                content=content,
                variables=all_vars,
                file_path=str(file_path)
            )
            
            # Cache it
            if use_cache:
                self._cache[cache_key] = prompt
            
            self.logger.debug(f"Loaded prompt: {path}")
            return prompt
        
        except Exception as e:
            self.logger.error(f"Error loading prompt {path}: {e}")
            # Return empty prompt as fallback
            return PromptTemplate(
                name=path,
                content="",
                variables=all_vars
            )
    
    def load_prompts_from_config(self, prompt_config: Dict[str, Any]) -> Dict[str, PromptTemplate]:
        """
        Load all prompts defined in agent config.
        
        Args:
            prompt_config: Prompt configuration from agent.yaml
            
        Returns:
            Dictionary of prompt name -> PromptTemplate
        """
        prompts = {}
        
        if not prompt_config:
            return prompts
        
        # Load system prompt
        if "system_prompt" in prompt_config:
            system_config = prompt_config["system_prompt"]
            if isinstance(system_config, dict):
                prompts["system_prompt"] = self.load_prompt(
                    system_config.get("path", ""),
                    system_config.get("variables", {})
                )
        
        # Load runtime prompts
        if "runtime_prompts" in prompt_config:
            for name, config in prompt_config["runtime_prompts"].items():
                if isinstance(config, dict):
                    prompts[name] = self.load_prompt(
                        config.get("path", ""),
                        config.get("variables", {})
                    )
        
        self.logger.info(f"Loaded {len(prompts)} prompts from config")
        return prompts
    
    def render_prompt(
        self,
        prompt: PromptTemplate,
        **kwargs
    ) -> str:
        """
        Render a prompt template with variables.
        
        Args:
            prompt: The prompt template to render
            **kwargs: Additional variables to inject
            
        Returns:
            Rendered prompt string
        """
        try:
            return prompt.render(**kwargs)
        except Exception as e:
            self.logger.error(f"Error rendering prompt {prompt.name}: {e}")
            return prompt.content  # Return unrendered as fallback
    
    def _get_default_variables(self) -> Dict[str, Any]:
        """Get default variables available to all prompts."""
        return {
            "ORG_NAME": os.getenv("ORG_NAME", "Organization"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
            "CURRENT_TIME": datetime.now().isoformat(),
            "CURRENT_DATE": datetime.now().strftime("%Y-%m-%d"),
        }
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()
        self.logger.debug("Prompt cache cleared")
    
    def reload_prompt(self, path: str, variables: Optional[Dict[str, Any]] = None) -> PromptTemplate:
        """
        Reload a prompt from file (bypassing cache).
        
        Useful for development and testing.
        """
        return self.load_prompt(path, variables, use_cache=False)


def create_prompt_loader(base_dir: str = "prompts") -> PromptLoader:
    """
    Factory function to create a PromptLoader.
    
    Args:
        base_dir: Base directory for prompts
        
    Returns:
        Configured PromptLoader instance
    """
    return PromptLoader(base_dir)

