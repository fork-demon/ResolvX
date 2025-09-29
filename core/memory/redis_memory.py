"""
Redis-based memory implementation for the Golden Agent Framework.

Provides distributed memory storage and retrieval using Redis
with vector similarity search capabilities.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from core.memory.base import BaseMemory, MemoryEntry, SearchResult
from core.exceptions import MemoryError
from core.observability import get_logger


class RedisMemory(BaseMemory):
    """
    Redis-based memory implementation.
    
    Uses Redis for distributed memory storage with vector similarity search
    and configurable data persistence.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Redis memory backend.
        
        Args:
            config: Redis-specific configuration
        """
        if not REDIS_AVAILABLE:
            raise MemoryError("Redis library not available. Install with: pip install redis")
        
        super().__init__(config)
        self.logger = get_logger("memory.redis")
        
        # Redis configuration
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 6379)
        self.db = self.config.get("db", 0)
        self.password = self.config.get("password")
        self.key_prefix = self.config.get("key_prefix", "golden_agent:memory")
        self.ttl = self.config.get("ttl", 86400)  # 24 hours default
        
        # Redis client
        self._redis: Optional[aioredis.Redis] = None
        
        # Vector search configuration
        self.vector_dimension = self.config.get("vector_dimension", 1536)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)

    async def initialize(self) -> None:
        """Initialize the Redis memory backend."""
        try:
            # Create Redis connection
            self._redis = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,  # We'll handle encoding manually
            )
            
            # Test connection
            await self._redis.ping()
            
            # Initialize vector search index if supported
            await self._initialize_vector_index()
            
            self._initialized = True
            self.logger.info(f"Redis memory initialized on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis memory: {e}")
            raise MemoryError(f"Redis memory initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the Redis memory backend."""
        try:
            if self._redis:
                await self._redis.close()
                self._redis = None
            
            self._initialized = False
            self.logger.info("Redis memory closed")
            
        except Exception as e:
            self.logger.error(f"Error closing Redis memory: {e}")

    async def store(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "default",
        **kwargs: Any
    ) -> str:
        """Store a memory entry."""
        try:
            # Create memory entry
            memory_entry = MemoryEntry(
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                namespace=namespace,
                **kwargs
            )
            
            # Generate embedding if not provided
            if not memory_entry.embedding:
                memory_entry.embedding = await self._generate_embedding(content)
            
            # Store in Redis
            await self._store_memory_entry(memory_entry)
            
            self.logger.debug(f"Stored memory entry {memory_entry.id}")
            return memory_entry.id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory entry: {e}")
            raise MemoryError(f"Memory storage failed: {e}") from e

    async def retrieve(self, memory_id: str, namespace: str = "default") -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry."""
        try:
            key = f"{self.key_prefix}:entry:{memory_id}"
            data = await self._redis.get(key)
            
            if not data:
                return None
            
            # Deserialize memory entry
            memory_entry = pickle.loads(data)
            
            # Check namespace
            if memory_entry.namespace != namespace:
                return None
            
            # Update access tracking
            memory_entry.update_access()
            
            # Update in Redis
            await self._store_memory_entry(memory_entry)
            
            return memory_entry
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory entry {memory_id}: {e}")
            raise MemoryError(f"Memory retrieval failed: {e}") from e

    async def search(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
        threshold: float = 0.7,
        **kwargs: Any
    ) -> List[SearchResult]:
        """Search for similar memories."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Get all memory entries in namespace
            namespace_key = f"{self.key_prefix}:namespace:{namespace}"
            memory_ids = await self._redis.smembers(namespace_key)
            
            if not memory_ids:
                return []
            
            results = []
            
            # Calculate similarities
            for memory_id_bytes in memory_ids:
                memory_id = memory_id_bytes.decode('utf-8')
                memory_entry = await self.retrieve(memory_id, namespace)
                
                if not memory_entry or not memory_entry.embedding:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(query_embedding, memory_entry.embedding)
                
                if similarity >= threshold:
                    result = SearchResult(
                        entry=memory_entry,
                        score=similarity,
                        relevance=similarity
                    )
                    results.append(result)
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            raise MemoryError(f"Memory search failed: {e}") from e

    async def delete(self, memory_id: str, namespace: str = "default") -> bool:
        """Delete a memory entry."""
        try:
            # Check if memory exists
            memory_entry = await self.retrieve(memory_id, namespace)
            if not memory_entry:
                return False
            
            # Delete from Redis
            key = f"{self.key_prefix}:entry:{memory_id}"
            await self._redis.delete(key)
            
            # Remove from namespace set
            namespace_key = f"{self.key_prefix}:namespace:{namespace}"
            await self._redis.srem(namespace_key, memory_id)
            
            # Remove from vector index if exists
            vector_key = f"{self.key_prefix}:vector:{memory_id}"
            await self._redis.delete(vector_key)
            
            self.logger.debug(f"Deleted memory entry {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory entry {memory_id}: {e}")
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
        try:
            # Retrieve existing entry
            memory_entry = await self.retrieve(memory_id, namespace)
            if not memory_entry:
                return False
            
            # Update fields
            if content is not None:
                memory_entry.content = content
                # Regenerate embedding if content changed
                if not embedding:
                    memory_entry.embedding = await self._generate_embedding(content)
            
            if embedding is not None:
                memory_entry.embedding = embedding
            
            if metadata is not None:
                memory_entry.metadata.update(metadata)
            
            # Update timestamp
            memory_entry.updated_at = datetime.utcnow()
            
            # Store updated entry
            await self._store_memory_entry(memory_entry)
            
            self.logger.debug(f"Updated memory entry {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update memory entry {memory_id}: {e}")
            raise MemoryError(f"Memory update failed: {e}") from e

    async def list_memories(
        self,
        namespace: str = "default",
        limit: Optional[int] = None,
        offset: int = 0,
        **kwargs: Any
    ) -> List[MemoryEntry]:
        """List memory entries."""
        try:
            # Get memory IDs from namespace
            namespace_key = f"{self.key_prefix}:namespace:{namespace}"
            memory_ids = await self._redis.smembers(namespace_key)
            
            if not memory_ids:
                return []
            
            # Convert to list and apply pagination
            memory_id_list = [mid.decode('utf-8') for mid in memory_ids]
            
            if offset > 0:
                memory_id_list = memory_id_list[offset:]
            
            if limit is not None:
                memory_id_list = memory_id_list[:limit]
            
            # Retrieve memory entries
            memories = []
            for memory_id in memory_id_list:
                memory_entry = await self.retrieve(memory_id, namespace)
                if memory_entry:
                    memories.append(memory_entry)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            raise MemoryError(f"Memory listing failed: {e}") from e

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all memories in a namespace."""
        try:
            # Get all memory IDs in namespace
            namespace_key = f"{self.key_prefix}:namespace:{namespace}"
            memory_ids = await self._redis.smembers(namespace_key)
            
            if not memory_ids:
                return 0
            
            # Delete each memory entry
            deleted_count = 0
            for memory_id_bytes in memory_ids:
                memory_id = memory_id_bytes.decode('utf-8')
                
                # Delete entry
                key = f"{self.key_prefix}:entry:{memory_id}"
                await self._redis.delete(key)
                
                # Delete vector
                vector_key = f"{self.key_prefix}:vector:{memory_id}"
                await self._redis.delete(vector_key)
                
                deleted_count += 1
            
            # Clear namespace set
            await self._redis.delete(namespace_key)
            
            self.logger.info(f"Cleared {deleted_count} memories from namespace {namespace}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to clear namespace {namespace}: {e}")
            raise MemoryError(f"Namespace clearing failed: {e}") from e

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            if namespace:
                # Get namespace-specific stats
                namespace_key = f"{self.key_prefix}:namespace:{namespace}"
                memory_ids = await self._redis.smembers(namespace_key)
                namespace_memories = len(memory_ids)
                
                return {
                    "namespace_memories": namespace_memories,
                    "namespace": namespace,
                }
            else:
                # Get global stats
                pattern = f"{self.key_prefix}:namespace:*"
                namespace_keys = await self._redis.keys(pattern)
                
                total_memories = 0
                namespace_counts = {}
                
                for namespace_key in namespace_keys:
                    ns = namespace_key.decode('utf-8').split(':')[-1]
                    memory_ids = await self._redis.smembers(namespace_key)
                    count = len(memory_ids)
                    namespace_counts[ns] = count
                    total_memories += count
                
                return {
                    "total_memories": total_memories,
                    "namespace_counts": namespace_counts,
                    "total_namespaces": len(namespace_keys),
                }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            raise MemoryError(f"Stats retrieval failed: {e}") from e

    async def _store_memory_entry(self, memory_entry: MemoryEntry) -> None:
        """Store a memory entry in Redis."""
        # Serialize memory entry
        data = pickle.dumps(memory_entry)
        
        # Store entry
        key = f"{self.key_prefix}:entry:{memory_entry.id}"
        await self._redis.setex(key, self.ttl, data)
        
        # Add to namespace set
        namespace_key = f"{self.key_prefix}:namespace:{memory_entry.namespace}"
        await self._redis.sadd(namespace_key, memory_entry.id)
        await self._redis.expire(namespace_key, self.ttl)
        
        # Store vector if available
        if memory_entry.embedding:
            vector_key = f"{self.key_prefix}:vector:{memory_entry.id}"
            vector_data = json.dumps(memory_entry.embedding)
            await self._redis.setex(vector_key, self.ttl, vector_data)

    async def _initialize_vector_index(self) -> None:
        """Initialize vector search index."""
        # This would set up Redis vector search if available
        # For now, we'll use simple similarity calculation
        self.logger.info("Using simple vector similarity search")

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Mock embedding generation
        # In production, this would use a real embedding model
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash to embedding vector
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                value = int.from_bytes(chunk, byteorder='big')
                # Normalize to [-1, 1] range
                normalized = (value / (2**32 - 1)) * 2 - 1
                embedding.append(normalized)
        
        # Pad or truncate to required dimension
        while len(embedding) < self.vector_dimension:
            embedding.append(0.0)
        
        return embedding[:self.vector_dimension]

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not NUMPY_AVAILABLE:
            # Simple dot product similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        # Use numpy for more efficient calculation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)