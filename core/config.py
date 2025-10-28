"""
Configuration management for the Golden Agent Framework.

Provides YAML-based configuration with environment variable substitution,
validation, and hot reloading capabilities.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from core.exceptions import ConfigurationError


class MemoryConfig(BaseModel):
    """Memory backend configuration."""

    backend: str = "faiss"
    faiss: Optional[Dict[str, Any]] = Field(default_factory=dict)
    redis: Optional[Dict[str, Any]] = Field(default_factory=dict)
    pinecone: Optional[Dict[str, Any]] = Field(default_factory=dict)
    mock: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("backend")
    def validate_backend(cls, v: str) -> str:
        """Validate memory backend type."""
        allowed_backends = {"faiss", "redis", "pinecone", "mock"}
        if v not in allowed_backends:
            raise ValueError(f"Backend must be one of {allowed_backends}")
        return v


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    backend: str = "console"
    langsmith: Optional[Dict[str, Any]] = Field(default_factory=dict)
    langfuse: Optional[Dict[str, Any]] = Field(default_factory=dict)
    console: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("backend")
    def validate_backend(cls, v: str) -> str:
        """Validate observability backend type."""
        allowed_backends = {"langsmith", "langfuse", "console"}
        if v not in allowed_backends:
            raise ValueError(f"Backend must be one of {allowed_backends}")
        return v


class RAGConfig(BaseModel):
    """RAG backend configuration."""

    backend: str = "faiss_kb"
    knowledge_dir: Optional[str] = "kb"
    model_name: Optional[str] = "all-MiniLM-L6-v2"
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Legacy backend-specific configs (deprecated)
    llamaindex: Optional[Dict[str, Any]] = Field(default_factory=dict)
    langchain: Optional[Dict[str, Any]] = Field(default_factory=dict)
    local_kb: Optional[Dict[str, Any]] = Field(default_factory=dict)
    mock: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("backend")
    def validate_backend(cls, v: str) -> str:
        """Validate RAG backend type."""
        allowed_backends = {"faiss_kb", "global_rag", "local_kb", "llamaindex", "langchain", "mock"}
        if v not in allowed_backends:
            raise ValueError(f"Backend must be one of {allowed_backends}")
        return v


class GatewayConfig(BaseModel):
    """Gateway configuration for MCP and LLM gateways with multi-gateway support."""

    mcp_gateway: Dict[str, Any] = Field(default_factory=dict)
    additional_mcp_gateways: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Multi-gateway support
    llm_gateway: Dict[str, Any] = Field(default_factory=dict)
    tools: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Individual agent configuration."""

    enabled: bool = True
    model: Optional[str] = None
    type: Optional[str] = None
    team: Optional[str] = None
    system_prompt: str = ""
    runtime_prompts: Dict[str, str] = Field(default_factory=dict)
    # accept nested prompts block from YAML as extra, but keep an optional typed handle
    prompts: Optional[Dict[str, Any]] = None
    tools: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.8
    max_iterations: int = 10
    timeout: int = 300
    custom_settings: Dict[str, Any] = Field(default_factory=dict)

    # allow extra keys from YAML to remain accessible via attribute access
    model_config = SettingsConfigDict(extra='allow')


class Config(BaseModel):
    """Main configuration model."""

    # Core settings
    version: str = "0.1.0"
    environment: str = "local"
    debug: bool = False

    # Component configurations
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)

    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)

    # Global settings
    global_settings: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        env_prefix = "GOLDEN_AGENT_"
        case_sensitive = False


class EnvironmentSettings(BaseSettings):
    """Environment-specific settings loaded from environment variables."""

    # Pydantic v2 settings: allow extra env vars and load from .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')

    # Organization settings
    org_name: str = Field(default="your-org", env="ORG_NAME")

    # Gateway URLs
    central_llm_gateway_url: str = Field(
        default="http://localhost:8000", env="CENTRAL_LLM_GATEWAY_URL"
    )
    central_mcp_gateway_url: str = Field(
        default="http://localhost:8001", env="CENTRAL_MCP_GATEWAY_URL"
    )

    # Default settings
    default_agent_version: str = Field(default="1.0.0", env="DEFAULT_AGENT_VERSION")
    default_embedding_model: str = Field(
        default="text-embedding-ada-002", env="DEFAULT_EMBEDDING_MODEL"
    )

    # Confidence thresholds
    default_confidence_threshold_triage: float = Field(
        default=0.8, env="DEFAULT_CONFIDENCE_THRESHOLD_TRIAGE"
    )
    default_confidence_threshold_metrics: float = Field(
        default=0.7, env="DEFAULT_CONFIDENCE_THRESHOLD_METRICS"
    )
    default_confidence_threshold_poller: float = Field(
        default=0.9, env="DEFAULT_CONFIDENCE_THRESHOLD_POLLER"
    )

    # Note: Pydantic v2 uses model_config; legacy Config removed to avoid conflict


def substitute_variables(
    data: Union[str, Dict[str, Any], list], variables: Dict[str, Any]
) -> Union[str, Dict[str, Any], list]:
    """
    Substitute template variables in configuration data.

    Supports both simple {VAR} and nested {DICT.KEY} syntax.
    """
    if isinstance(data, str):
        # Replace variables using simple substitution
        pattern = r"\{([^}]+)\}"

        def replace_var(match: re.Match[str]) -> str:
            var_path = match.group(1)

            # Handle nested dictionary access (e.g., DEFAULT_CONFIDENCE_THRESHOLDS.triage)
            if "." in var_path:
                keys = var_path.split(".")
                value = variables
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return match.group(0)  # Return original if not found
                return str(value)
            else:
                # Simple variable replacement
                return str(variables.get(var_path, match.group(0)))

        return re.sub(pattern, replace_var, data)

    elif isinstance(data, dict):
        return {k: substitute_variables(v, variables) for k, v in data.items()}

    elif isinstance(data, list):
        return [substitute_variables(item, variables) for item in data]

    else:
        return data


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to configuration file. If None, looks for default locations.

    Returns:
        Validated configuration object.

    Raises:
        ConfigurationError: If configuration is invalid or cannot be loaded.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Load environment settings
    env_settings = EnvironmentSettings()

    # Prepare template variables
    template_vars = {
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
        # Add Langfuse environment variables
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY", "dev"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY", "dev"),
        "LANGFUSE_HOST": os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        "ENVIRONMENT": os.getenv("ENVIRONMENT", "local"),
    }

    # Determine config file path
    if config_path is None:
        # Look for config files in order of preference
        possible_paths = [
            Path("config/agent.yaml"),
            Path("agent.yaml"),
            Path("config.yaml"),
        ]

        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            raise ConfigurationError(
                f"No configuration file found. Looked in: {possible_paths}"
            )
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        # Load YAML content
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raw_config = {}

        # Substitute template variables
        processed_config = substitute_variables(raw_config, template_vars)

        # Validate and create config object
        return Config(**processed_config)

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        config_path: Path where to save the configuration file.

    Raises:
        ConfigurationError: If configuration cannot be saved.
    """
    config_path = Path(config_path)

    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save as YAML
        config_dict = config.dict()

        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
                allow_unicode=True,
            )

    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration: {e}") from e


class ConfigWatcher:
    """Watches configuration files for changes and reloads automatically."""

    def __init__(self, config_path: Union[str, Path], callback: callable):
        """
        Initialize configuration watcher.

        Args:
            config_path: Path to configuration file to watch.
            callback: Function to call when configuration changes.
        """
        self.config_path = Path(config_path)
        self.callback = callback
        self._last_modified = None
        self._update_last_modified()

    def _update_last_modified(self) -> None:
        """Update the last modified timestamp."""
        if self.config_path.exists():
            self._last_modified = self.config_path.stat().st_mtime

    def check_for_changes(self) -> bool:
        """
        Check if configuration file has changed.

        Returns:
            True if file has changed, False otherwise.
        """
        if not self.config_path.exists():
            return False

        current_modified = self.config_path.stat().st_mtime

        if self._last_modified is None or current_modified > self._last_modified:
            self._last_modified = current_modified
            return True

        return False

    async def watch(self, interval: float = 1.0) -> None:
        """
        Watch for configuration changes and call callback when changed.

        Args:
            interval: Check interval in seconds.
        """
        import asyncio

        while True:
            if self.check_for_changes():
                try:
                    new_config = load_config(self.config_path)
                    await self.callback(new_config)
                except Exception as e:
                    # Log error but continue watching
                    print(f"Error reloading config: {e}")

            await asyncio.sleep(interval)