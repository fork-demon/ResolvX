"""
Base memory interface for the Golden Agent Framework.

Defines the common interface that all memory backends must implement
for storing and retrieving agent memories and context.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from core.exceptions import MemoryError


class MemoryEntry(BaseModel):
    """
    Represents a single memory entry.

    Contains the content, metadata, and embedding information for
    a piece of information stored in memory.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Timestamp information
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: Optional[datetime] = None

    # Organization and categorization
    namespace: str = "default"
    category: str = "general"
    tags: List[str] = Field(default_factory=list)

    # Importance and relevance scoring
    importance_score: float = 0.5
    access_count: int = 0

    # Source information
    source: Optional[str] = None
    source_id: Optional[str] = None

    # Expiration (None means no expiration)
    expires_at: Optional[datetime] = None

    def update_access(self) -> None:
        """Update access tracking information."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1
        self.updated_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if this memory entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def add_tag(self, tag: str) -> None:
        """Add a tag to this memory entry."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from this memory entry."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
            return True
        return False

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class SearchResult(BaseModel):
    """
    Represents search results from memory queries.
    """

    entry: MemoryEntry
    score: float
    distance: Optional[float] = None
    relevance: Optional[float] = None

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class BaseMemory(ABC):
    """
    Abstract base class for all memory implementations.

    Defines the interface that all memory backends must implement
    for storing, retrieving, and searching memories.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory backend.

        Args:
            config: Backend-specific configuration
        """
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the memory backend.

        This method should set up any necessary connections,
        indices, or other resources.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the memory backend and cleanup resources.
        """
        pass

    @abstractmethod
    async def store(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        **kwargs: Any
    ) -> str:
        """
        Store a memory entry.

        Args:
            content: The content to store
            embedding: Optional pre-computed embedding
            metadata: Optional metadata dictionary
            namespace: Memory namespace
            **kwargs: Additional storage options

        Returns:
            Unique identifier for the stored memory

        Raises:
            MemoryError: If storage fails
        """
        pass

    @abstractmethod
    async def retrieve(self, memory_id: str, namespace: str = "default") -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory entry by ID.

        Args:
            memory_id: Unique identifier of the memory
            namespace: Memory namespace

        Returns:
            Memory entry or None if not found

        Raises:
            MemoryError: If retrieval fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
        threshold: float = 0.7,
        **kwargs: Any
    ) -> List[SearchResult]:
        """
        Search for similar memories.

        Args:
            query: Search query
            namespace: Memory namespace
            limit: Maximum number of results
            threshold: Similarity threshold
            **kwargs: Additional search options

        Returns:
            List of search results

        Raises:
            MemoryError: If search fails
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str, namespace: str = "default") -> bool:
        """
        Delete a memory entry.

        Args:
            memory_id: Unique identifier of the memory
            namespace: Memory namespace

        Returns:
            True if deleted, False if not found

        Raises:
            MemoryError: If deletion fails
        """
        pass

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        **kwargs: Any
    ) -> bool:
        """
        Update a memory entry.

        Args:
            memory_id: Unique identifier of the memory
            content: Updated content
            embedding: Updated embedding
            metadata: Updated metadata
            namespace: Memory namespace
            **kwargs: Additional update options

        Returns:
            True if updated, False if not found

        Raises:
            MemoryError: If update fails
        """
        pass

    @abstractmethod
    async def list_memories(
        self,
        namespace: str = "default",
        limit: Optional[int] = None,
        offset: int = 0,
        **kwargs: Any
    ) -> List[MemoryEntry]:
        """
        List memory entries.

        Args:
            namespace: Memory namespace
            limit: Maximum number of entries
            offset: Number of entries to skip
            **kwargs: Additional filtering options

        Returns:
            List of memory entries

        Raises:
            MemoryError: If listing fails
        """
        pass

    @abstractmethod
    async def clear_namespace(self, namespace: str) -> int:
        """
        Clear all memories in a namespace.

        Args:
            namespace: Memory namespace to clear

        Returns:
            Number of memories deleted

        Raises:
            MemoryError: If clearing fails
        """
        pass

    @abstractmethod
    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.

        Args:
            namespace: Optional namespace filter

        Returns:
            Dictionary with memory statistics

        Raises:
            MemoryError: If stats retrieval fails
        """
        pass

    # Common utility methods

    async def store_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        namespace: str = "conversations",
        **kwargs: Any
    ) -> List[str]:
        """
        Store a conversation as multiple memory entries.

        Args:
            conversation_id: Unique conversation identifier
            messages: List of conversation messages
            namespace: Memory namespace
            **kwargs: Additional storage options

        Returns:
            List of memory IDs for stored messages
        """
        memory_ids = []

        for i, message in enumerate(messages):
            content = message.get("content", "")
            metadata = {
                "conversation_id": conversation_id,
                "message_index": i,
                "role": message.get("role", "unknown"),
                "timestamp": message.get("timestamp", datetime.utcnow().isoformat()),
                **kwargs.get("metadata", {}),
            }

            memory_id = await self.store(
                content=content,
                metadata=metadata,
                namespace=namespace,
                category="conversation",
                source="chat",
                source_id=conversation_id,
            )
            memory_ids.append(memory_id)

        return memory_ids

    async def search_conversations(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        namespace: str = "conversations",
        limit: int = 10,
        **kwargs: Any
    ) -> List[SearchResult]:
        """
        Search within conversation memories.

        Args:
            query: Search query
            conversation_id: Optional conversation filter
            namespace: Memory namespace
            limit: Maximum number of results
            **kwargs: Additional search options

        Returns:
            List of search results from conversations
        """
        search_kwargs = kwargs.copy()

        if conversation_id:
            search_kwargs["metadata_filter"] = {
                "conversation_id": conversation_id,
                **search_kwargs.get("metadata_filter", {})
            }

        search_kwargs["category_filter"] = "conversation"

        return await self.search(
            query=query,
            namespace=namespace,
            limit=limit,
            **search_kwargs
        )

    async def cleanup_expired(self, namespace: Optional[str] = None) -> int:
        """
        Clean up expired memory entries.

        Args:
            namespace: Optional namespace filter

        Returns:
            Number of expired entries removed
        """
        # Default implementation - subclasses can override for efficiency
        namespaces_to_check = [namespace] if namespace else await self.list_namespaces()
        cleaned_count = 0

        for ns in namespaces_to_check:
            memories = await self.list_memories(namespace=ns)

            for memory in memories:
                if memory.is_expired():
                    if await self.delete(memory.id, ns):
                        cleaned_count += 1

        return cleaned_count

    async def list_namespaces(self) -> List[str]:
        """
        List all available namespaces.

        Returns:
            List of namespace names
        """
        # Default implementation - subclasses should override
        return ["default"]

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the memory backend.

        Returns:
            Health check results
        """
        try:
            # Basic connectivity test
            stats = await self.get_stats()

            return {
                "status": "healthy",
                "backend": self.__class__.__name__,
                "initialized": self._initialized,
                "stats": stats,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": self.__class__.__name__,
                "initialized": self._initialized,
                "error": str(e),
            }

    @property
    def is_initialized(self) -> bool:
        """Check if the memory backend is initialized."""
        return self._initialized