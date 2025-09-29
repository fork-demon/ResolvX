"""
Mock memory implementation for testing and development.

Provides an in-memory implementation of the memory interface for
local development, testing, and scenarios where persistence is not required.
"""

import math
from typing import Any, Dict, List, Optional

from core.exceptions import MemoryError
from core.memory.base import BaseMemory, MemoryEntry, SearchResult
from core.observability import get_logger


class MockMemory(BaseMemory):
    """
    Mock memory implementation using in-memory storage.

    Provides a simple implementation of the memory interface using
    Python dictionaries for testing and development purposes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mock memory backend.

        Args:
            config: Configuration dictionary containing:
                - persist: Whether to persist data to disk (not implemented)
                - initial_data: Initial memory entries to load
                - embedding_dim: Expected embedding dimension
        """
        super().__init__(config)
        self.logger = get_logger("memory.mock")

        # Configuration
        self.persist = self.config.get("persist", False)
        self.initial_data = self.config.get("initial_data", [])
        self.embedding_dim = self.config.get("embedding_dim", 1536)

        # In-memory storage
        self.memory_store: Dict[str, MemoryEntry] = {}
        self.namespace_indices: Dict[str, Dict[str, MemoryEntry]] = {}

    async def initialize(self) -> None:
        """Initialize the mock memory backend."""
        try:
            # Load initial data if provided
            if self.initial_data:
                for data in self.initial_data:
                    await self._load_initial_entry(data)

            self._initialized = True
            self.logger.info(f"Mock memory initialized with {len(self.memory_store)} entries")

        except Exception as e:
            self.logger.error(f"Failed to initialize mock memory: {e}")
            raise MemoryError(f"Mock memory initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the mock memory backend."""
        try:
            # Clear all data
            self.memory_store.clear()
            self.namespace_indices.clear()
            self._initialized = False
            self.logger.info("Mock memory closed")

        except Exception as e:
            self.logger.error(f"Error closing mock memory: {e}")

    async def store(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        **kwargs: Any
    ) -> str:
        """Store a memory entry in mock storage."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            # Validate embedding if provided
            if embedding is not None and len(embedding) != self.embedding_dim:
                raise MemoryError(
                    f"Embedding dimension {len(embedding)} does not match "
                    f"expected dimension {self.embedding_dim}"
                )

            # Create memory entry
            memory_entry = MemoryEntry(
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                namespace=namespace,
                **{k: v for k, v in kwargs.items() if k in MemoryEntry.__fields__}
            )

            # Store in main store
            self.memory_store[memory_entry.id] = memory_entry

            # Store in namespace index
            if namespace not in self.namespace_indices:
                self.namespace_indices[namespace] = {}
            self.namespace_indices[namespace][memory_entry.id] = memory_entry

            self.logger.debug(f"Stored memory {memory_entry.id} in namespace {namespace}")
            return memory_entry.id

        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise MemoryError(f"Memory storage failed: {e}") from e

    async def retrieve(self, memory_id: str, namespace: str = "default") -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            memory_entry = self.memory_store.get(memory_id)

            if memory_entry and memory_entry.namespace == namespace:
                # Update access tracking
                memory_entry.update_access()
                return memory_entry

            return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise MemoryError(f"Memory retrieval failed: {e}") from e

    async def search(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
        threshold: float = 0.7,
        query_embedding: Optional[List[float]] = None,
        **kwargs: Any
    ) -> List[SearchResult]:
        """Search for similar memories using text or vector similarity."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            namespace_memories = self.namespace_indices.get(namespace, {})
            results = []

            for memory_entry in namespace_memories.values():
                score = 0.0

                # If we have embeddings, use vector similarity
                if query_embedding and memory_entry.embedding:
                    if len(query_embedding) == len(memory_entry.embedding):
                        score = self._cosine_similarity(query_embedding, memory_entry.embedding)
                    else:
                        continue  # Skip if embedding dimensions don't match

                # Otherwise, use text similarity
                else:
                    score = self._text_similarity(query, memory_entry.content)

                # Apply filters
                if not self._passes_filters(memory_entry, kwargs):
                    continue

                if score >= threshold:
                    # Update access tracking
                    memory_entry.update_access()

                    result = SearchResult(
                        entry=memory_entry,
                        score=score,
                        distance=1.0 - score,
                        relevance=score,
                    )
                    results.append(result)

            # Sort by score (descending) and limit
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:limit]

            self.logger.debug(f"Found {len(results)} results for search in namespace {namespace}")
            return results

        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            raise MemoryError(f"Memory search failed: {e}") from e

    async def delete(self, memory_id: str, namespace: str = "default") -> bool:
        """Delete a memory entry."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            if memory_id not in self.memory_store:
                return False

            memory_entry = self.memory_store[memory_id]
            if memory_entry.namespace != namespace:
                return False

            # Remove from main store
            del self.memory_store[memory_id]

            # Remove from namespace index
            if namespace in self.namespace_indices:
                self.namespace_indices[namespace].pop(memory_id, None)

            self.logger.debug(f"Deleted memory {memory_id} from namespace {namespace}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise MemoryError(f"Memory deletion failed: {e}") from e

    async def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        **kwargs: Any
    ) -> bool:
        """Update a memory entry."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            if memory_id not in self.memory_store:
                return False

            memory_entry = self.memory_store[memory_id]
            if memory_entry.namespace != namespace:
                return False

            # Update fields
            if content is not None:
                memory_entry.content = content

            if embedding is not None:
                if len(embedding) != self.embedding_dim:
                    raise MemoryError(
                        f"Embedding dimension {len(embedding)} does not match "
                        f"expected dimension {self.embedding_dim}"
                    )
                memory_entry.embedding = embedding

            if metadata is not None:
                memory_entry.metadata.update(metadata)

            # Update other fields
            for key, value in kwargs.items():
                if hasattr(memory_entry, key):
                    setattr(memory_entry, key, value)

            memory_entry.update_timestamp()

            self.logger.debug(f"Updated memory {memory_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update memory {memory_id}: {e}")
            raise MemoryError(f"Memory update failed: {e}") from e

    async def list_memories(
        self,
        namespace: str = "default",
        limit: Optional[int] = None,
        offset: int = 0,
        **kwargs: Any
    ) -> List[MemoryEntry]:
        """List memory entries in a namespace."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            namespace_memories = list(self.namespace_indices.get(namespace, {}).values())

            # Apply filters
            filtered_memories = []
            for memory in namespace_memories:
                if self._passes_filters(memory, kwargs):
                    filtered_memories.append(memory)

            # Sort by creation time (newest first)
            filtered_memories.sort(key=lambda x: x.created_at, reverse=True)

            # Apply pagination
            start_idx = offset
            end_idx = offset + limit if limit else None
            result = filtered_memories[start_idx:end_idx]

            return result

        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            raise MemoryError(f"Memory listing failed: {e}") from e

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all memories in a namespace."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            # Get memories to delete
            namespace_memories = self.namespace_indices.get(namespace, {})
            memory_ids = list(namespace_memories.keys())

            # Remove from main store
            for memory_id in memory_ids:
                self.memory_store.pop(memory_id, None)

            # Clear namespace index
            if namespace in self.namespace_indices:
                del self.namespace_indices[namespace]

            deleted_count = len(memory_ids)
            self.logger.info(f"Cleared {deleted_count} memories from namespace {namespace}")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to clear namespace {namespace}: {e}")
            raise MemoryError(f"Namespace clearing failed: {e}") from e

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self._initialized:
            raise MemoryError("Mock memory not initialized")

        try:
            if namespace:
                # Stats for specific namespace
                namespace_memories = self.namespace_indices.get(namespace, {})
                return {
                    "namespace": namespace,
                    "total_memories": len(namespace_memories),
                }

            else:
                # Global stats
                namespace_stats = {}
                for ns, memories in self.namespace_indices.items():
                    namespace_stats[ns] = {
                        "total_memories": len(memories),
                    }

                return {
                    "total_memories": len(self.memory_store),
                    "total_namespaces": len(self.namespace_indices),
                    "embedding_dimension": self.embedding_dim,
                    "backend": "mock",
                    "namespaces": namespace_stats,
                }

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise MemoryError(f"Stats retrieval failed: {e}") from e

    async def list_namespaces(self) -> List[str]:
        """List all available namespaces."""
        if not self._initialized:
            return []

        return sorted(list(self.namespace_indices.keys()))

    async def cleanup_expired(self, namespace: Optional[str] = None) -> int:
        """Clean up expired memory entries."""
        if not self._initialized:
            return 0

        namespaces_to_check = [namespace] if namespace else self.namespace_indices.keys()
        cleaned_count = 0

        for ns in namespaces_to_check:
            namespace_memories = self.namespace_indices.get(ns, {})
            expired_ids = []

            for memory_id, memory_entry in namespace_memories.items():
                if memory_entry.is_expired():
                    expired_ids.append(memory_id)

            # Remove expired memories
            for memory_id in expired_ids:
                if await self.delete(memory_id, ns):
                    cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} expired memories")

        return cleaned_count

    # Private helper methods

    async def _load_initial_entry(self, data: Dict[str, Any]) -> None:
        """Load an initial memory entry from configuration data."""
        try:
            await self.store(
                content=data.get("content", ""),
                embedding=data.get("embedding"),
                metadata=data.get("metadata", {}),
                namespace=data.get("namespace", "default"),
                category=data.get("category", "general"),
                tags=data.get("tags", []),
                importance_score=data.get("importance_score", 0.5),
                source=data.get("source"),
                source_id=data.get("source_id"),
            )
        except Exception as e:
            self.logger.warning(f"Failed to load initial entry: {e}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap."""
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _passes_filters(self, memory_entry: MemoryEntry, filters: Dict[str, Any]) -> bool:
        """Check if a memory entry passes the given filters."""
        # Category filter
        if filters.get("category_filter") and memory_entry.category != filters["category_filter"]:
            return False

        # Tag filter
        if filters.get("tag_filter") and filters["tag_filter"] not in memory_entry.tags:
            return False

        # Metadata filter
        metadata_filter = filters.get("metadata_filter", {})
        for key, value in metadata_filter.items():
            if memory_entry.metadata.get(key) != value:
                return False

        # Importance score filter
        min_importance = filters.get("min_importance_score")
        if min_importance is not None and memory_entry.importance_score < min_importance:
            return False

        # Source filter
        source_filter = filters.get("source_filter")
        if source_filter and memory_entry.source != source_filter:
            return False

        return True

    # Development and testing utilities

    def add_test_memories(self, count: int = 10, namespace: str = "default") -> List[str]:
        """Add test memories for development and testing."""
        import asyncio

        async def _add_test_memories():
            memory_ids = []
            for i in range(count):
                content = f"Test memory content {i + 1}"
                embedding = [float(j) for j in range(self.embedding_dim)]  # Simple test embedding

                memory_id = await self.store(
                    content=content,
                    embedding=embedding,
                    metadata={"test_index": i, "test_batch": True},
                    namespace=namespace,
                    category="test",
                    tags=[f"test_{i}", "test_data"],
                    importance_score=0.5 + (i % 5) * 0.1,
                )
                memory_ids.append(memory_id)

            return memory_ids

        # Run the async function
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_add_test_memories())

    def get_memory_count(self, namespace: Optional[str] = None) -> int:
        """Get the total number of memories (for testing)."""
        if namespace:
            return len(self.namespace_indices.get(namespace, {}))
        else:
            return len(self.memory_store)