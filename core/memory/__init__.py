"""
Pluggable memory system for the Golden Agent Framework.

This module provides various memory backends for storing and retrieving
agent context, conversation history, and knowledge base information.
"""

from core.memory.base import BaseMemory, MemoryEntry
from core.memory.faiss_memory import FAISSMemory
from core.memory.redis_memory import RedisMemory
from core.memory.pinecone_memory import PineconeMemory
from core.memory.mock_memory import MockMemory
from core.memory.factory import MemoryFactory, create_memory

__all__ = [
    "BaseMemory",
    "MemoryEntry",
    "FAISSMemory",
    "RedisMemory",
    "PineconeMemory",
    "MockMemory",
    "MemoryFactory",
    "create_memory",
]