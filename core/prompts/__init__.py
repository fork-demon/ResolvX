"""
Prompt management system for the Golden Agent Framework.

Provides centralized prompt loading, management, and template processing
for all agents in the framework.
"""

from core.prompts.manager import PromptManager
from core.prompts.template import PromptTemplate
from core.prompts.loader import PromptLoader

__all__ = [
    "PromptManager",
    "PromptTemplate", 
    "PromptLoader",
]
