"""
Pluggable memory system for the Golden Agent Framework.

This module provides FAISS-based vector memory for storing and retrieving
agent context, conversation history, and knowledge base information.
"""

from core.memory.base import BaseMemory, MemoryEntry
from core.memory.faiss_memory import FAISSMemory
from core.memory.factory import MemoryFactory, create_memory

__all__ = [
    "BaseMemory",
    "MemoryEntry",
    "FAISSMemory",
    "MemoryFactory",
    "create_memory",
]