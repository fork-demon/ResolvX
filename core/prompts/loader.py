"""
Prompt loader for loading prompts from various sources.

Supports loading prompts from files, databases, and other sources
with caching and validation capabilities.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from core.exceptions import ConfigurationError
from core.observability import get_logger


class PromptLoader:
    """
    Loads prompts from various sources.
    
    Supports file-based loading with caching and validation.
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize prompt loader.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self.logger = get_logger("prompts.loader")
        
        # Cache for loaded prompts
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    async def load_prompt_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load prompts from a file.
        
        Args:
            file_path: Path to prompt file
            
        Returns:
            Dictionary of prompt data
            
        Raises:
            ConfigurationError: If file loading fails
        """
        file_path = Path(file_path)
        
        try:
            # Check cache first
            cache_key = str(file_path)
            if self._is_cached(cache_key):
                return self._cache[cache_key]
            
            # Load file based on extension
            if file_path.suffix.lower() == '.json':
                data = await self._load_json_file(file_path)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                data = await self._load_yaml_file(file_path)
            elif file_path.suffix.lower() == '.md':
                data = await self._load_markdown_file(file_path)
            else:
                data = await self._load_text_file(file_path)
            
            # Cache the result
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load prompt file {file_path}: {e}")
            raise ConfigurationError(f"Prompt file loading failed: {e}") from e

    async def load_agent_prompts(self, agent_name: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load all prompts for an agent.
        
        Args:
            agent_name: Name of the agent
            base_path: Base path for prompt files
            
        Returns:
            Dictionary of all agent prompts
        """
        if base_path is None:
            base_path = Path("agents") / agent_name / "prompts"
        
        prompts = {}
        
        try:
            if not base_path.exists():
                self.logger.warning(f"Prompt directory not found: {base_path}")
                return prompts
            
            # Load system prompt
            system_file = base_path / "system.md"
            if system_file.exists():
                system_content = await self._load_markdown_file(system_file)
                prompts["system"] = system_content
            
            # Load runtime prompts
            for prompt_file in base_path.glob("*.json"):
                if prompt_file.name == "system.md":
                    continue
                
                try:
                    prompt_data = await self.load_prompt_file(prompt_file)
                    prompts.update(prompt_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load prompt file {prompt_file}: {e}")
            
            # Load YAML prompt files
            for prompt_file in base_path.glob("*.yaml"):
                try:
                    prompt_data = await self.load_prompt_file(prompt_file)
                    prompts.update(prompt_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load prompt file {prompt_file}: {e}")
            
            self.logger.info(f"Loaded {len(prompts)} prompts for agent {agent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load prompts for agent {agent_name}: {e}")
        
        return prompts

    async def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON prompt file."""
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    async def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML prompt file."""
        with file_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    async def _load_markdown_file(self, file_path: Path) -> str:
        """Load Markdown prompt file."""
        with file_path.open('r', encoding='utf-8') as f:
            return f.read()

    async def _load_text_file(self, file_path: Path) -> str:
        """Load text prompt file."""
        with file_path.open('r', encoding='utf-8') as f:
            return f.read()

    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and not expired."""
        if cache_key not in self._cache:
            return False
        
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[cache_key]
        return datetime.now() - cache_time < timedelta(seconds=self.cache_ttl)

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """
        Clear cache.
        
        Args:
            cache_key: Specific cache key to clear, or None to clear all
        """
        if cache_key:
            self._cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
        
        self.logger.debug(f"Cleared cache for key: {cache_key or 'all'}")

    async def validate_prompt_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        Validate a prompt file.
        
        Args:
            file_path: Path to prompt file
            
        Returns:
            List of validation errors
        """
        errors = []
        file_path = Path(file_path)
        
        try:
            # Check file exists
            if not file_path.exists():
                errors.append(f"File not found: {file_path}")
                return errors
            
            # Check file extension
            if file_path.suffix.lower() not in ['.json', '.yaml', '.yml', '.md', '.txt']:
                errors.append(f"Unsupported file extension: {file_path.suffix}")
            
            # Try to load file
            try:
                await self.load_prompt_file(file_path)
            except Exception as e:
                errors.append(f"File loading error: {e}")
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return errors

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None,
        }
