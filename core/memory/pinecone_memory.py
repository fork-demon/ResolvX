"""
Pinecone-based memory implementation for the Golden Agent Framework.

Provides cloud-based vector memory storage and retrieval using Pinecone
for scalable similarity search and clustering.
"""

import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from core.memory.base import BaseMemory, MemoryEntry, SearchResult
from core.exceptions import MemoryError
from core.observability import get_logger


class PineconeMemory(BaseMemory):
    """
    Pinecone-based memory implementation.
    
    Uses Pinecone for cloud-based vector storage with advanced
    similarity search and filtering capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Pinecone memory backend.
        
        Args:
            config: Pinecone-specific configuration
        """
        if not PINECONE_AVAILABLE:
            raise MemoryError("Pinecone library not available. Install with: pip install pinecone-client")
        
        super().__init__(config)
        self.logger = get_logger("memory.pinecone")
        
        # Pinecone configuration
        self.api_key = self.config.get("api_key") or os.getenv("PINECONE_API_KEY")
        self.environment = self.config.get("environment", "us-west1-gcp")
        self.index_name = self.config.get("index_name", "golden-agents")
        self.dimension = self.config.get("dimension", 1536)
        self.metric = self.config.get("metric", "cosine")
        
        if not self.api_key:
            raise MemoryError("Pinecone API key not provided")
        
        # Pinecone client and index
        self._pinecone = None
        self._index = None
        
        # Metadata store for non-vector data
        self._metadata_store: Dict[str, MemoryEntry] = {}

    async def initialize(self) -> None:
        """Initialize the Pinecone memory backend."""
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            self._pinecone = pinecone
            
            # Create or connect to index
            await self._initialize_index()
            
            self._initialized = True
            self.logger.info(f"Pinecone memory initialized with index {self.index_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone memory: {e}")
            raise MemoryError(f"Pinecone memory initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the Pinecone memory backend."""
        try:
            # Pinecone doesn't require explicit closing
            self._pinecone = None
            self._index = None
            
            self._initialized = False
            self.logger.info("Pinecone memory closed")
            
        except Exception as e:
            self.logger.error(f"Error closing Pinecone memory: {e}")

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
            
            # Validate embedding dimension
            if len(memory_entry.embedding) != self.dimension:
                raise MemoryError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(memory_entry.embedding)}")
            
            # Prepare vector for Pinecone
            vector_data = {
                "id": memory_entry.id,
                "values": memory_entry.embedding,
                "metadata": {
                    "content": memory_entry.content,
                    "namespace": memory_entry.namespace,
                    "created_at": memory_entry.created_at.isoformat(),
                    "updated_at": memory_entry.updated_at.isoformat(),
                    "access_count": memory_entry.access_count,
                    "importance_score": memory_entry.importance_score,
                    **memory_entry.metadata
                }
            }
            
            # Add to Pinecone
            self._index.upsert(vectors=[vector_data])
            
            # Store in local metadata store
            self._metadata_store[memory_entry.id] = memory_entry
            
            self.logger.debug(f"Stored memory entry {memory_entry.id} in Pinecone")
            return memory_entry.id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory entry: {e}")
            raise MemoryError(f"Memory storage failed: {e}") from e

    async def retrieve(self, memory_id: str, namespace: str = "default") -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry."""
        try:
            # Check local metadata store first
            if memory_id in self._metadata_store:
                memory_entry = self._metadata_store[memory_id]
                if memory_entry.namespace == namespace:
                    memory_entry.update_access()
                    return memory_entry
            
            # Fetch from Pinecone
            fetch_result = self._index.fetch(ids=[memory_id])
            
            if memory_id not in fetch_result["vectors"]:
                return None
            
            vector_data = fetch_result["vectors"][memory_id]
            metadata = vector_data.get("metadata", {})
            
            # Check namespace
            if metadata.get("namespace") != namespace:
                return None
            
            # Reconstruct memory entry
            memory_entry = MemoryEntry(
                id=memory_id,
                content=metadata.get("content", ""),
                embedding=vector_data.get("values"),
                metadata={k: v for k, v in metadata.items() 
                         if k not in ["content", "namespace", "created_at", "updated_at", 
                                    "access_count", "importance_score"]},
                namespace=namespace,
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.utcnow().isoformat())),
                access_count=metadata.get("access_count", 0),
                importance_score=metadata.get("importance_score", 0.5),
            )
            
            # Update access tracking
            memory_entry.update_access()
            
            # Update in Pinecone
            await self.store(
                content=memory_entry.content,
                embedding=memory_entry.embedding,
                metadata=memory_entry.metadata,
                namespace=memory_entry.namespace,
                document_id=memory_entry.id
            )
            
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
            
            # Search Pinecone
            search_result = self._index.query(
                vector=query_embedding,
                top_k=limit * 2,  # Get more results for filtering
                include_metadata=True,
                filter={"namespace": namespace} if namespace != "default" else None
            )
            
            results = []
            for match in search_result["matches"]:
                # Apply threshold
                if match["score"] < threshold:
                    continue
                
                # Reconstruct memory entry
                memory_entry = await self._reconstruct_memory_entry(match)
                if not memory_entry:
                    continue
                
                # Create search result
                result = SearchResult(
                    entry=memory_entry,
                    score=match["score"],
                    relevance=match["score"]
                )
                results.append(result)
            
            # Sort by score and limit results
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
            
            # Delete from Pinecone
            self._index.delete(ids=[memory_id])
            
            # Remove from local metadata store
            if memory_id in self._metadata_store:
                del self._metadata_store[memory_id]
            
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
            await self.store(
                content=memory_entry.content,
                embedding=memory_entry.embedding,
                metadata=memory_entry.metadata,
                namespace=memory_entry.namespace,
                document_id=memory_entry.id
            )
            
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
            # Get index stats
            stats = self._index.describe_index_stats()
            
            # For listing, we'll use a simple approach
            # In production, you might want to use Pinecone's list functionality
            memories = []
            
            # This is a simplified implementation
            # In practice, you'd need to implement proper pagination
            # using Pinecone's query capabilities
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            raise MemoryError(f"Memory listing failed: {e}") from e

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all memories in a namespace."""
        try:
            # Delete all vectors with namespace filter
            self._index.delete(filter={"namespace": namespace})
            
            # Remove from local metadata store
            to_remove = [
                memory_id for memory_id, entry in self._metadata_store.items()
                if entry.namespace == namespace
            ]
            
            for memory_id in to_remove:
                del self._metadata_store[memory_id]
            
            self.logger.info(f"Cleared memories from namespace {namespace}")
            return len(to_remove)
            
        except Exception as e:
            self.logger.error(f"Failed to clear namespace {namespace}: {e}")
            raise MemoryError(f"Namespace clearing failed: {e}") from e

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            # Get index stats from Pinecone
            stats = self._index.describe_index_stats()
            
            if namespace:
                # Get namespace-specific stats
                namespace_stats = stats.get("namespaces", {}).get(namespace, {})
                return {
                    "namespace_memories": namespace_stats.get("vector_count", 0),
                    "namespace": namespace,
                }
            else:
                # Get global stats
                total_vectors = stats.get("total_vector_count", 0)
                namespaces = stats.get("namespaces", {})
                
                return {
                    "total_memories": total_vectors,
                    "namespace_counts": {
                        ns: ns_stats.get("vector_count", 0)
                        for ns, ns_stats in namespaces.items()
                    },
                    "dimension": stats.get("dimension", self.dimension),
                    "index_fullness": stats.get("index_fullness", 0),
                }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            raise MemoryError(f"Stats retrieval failed: {e}") from e

    async def _initialize_index(self) -> None:
        """Initialize or create Pinecone index."""
        try:
            # Check if index exists
            if self.index_name in pinecone.list_indexes():
                # Connect to existing index
                self._index = pinecone.Index(self.index_name)
                self.logger.info(f"Connected to existing Pinecone index: {self.index_name}")
            else:
                # Create new index
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric
                )
                
                # Wait for index to be ready
                import time
                while not pinecone.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)
                
                # Connect to new index
                self._index = pinecone.Index(self.index_name)
                self.logger.info(f"Created new Pinecone index: {self.index_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone index: {e}")
            raise MemoryError(f"Pinecone index initialization failed: {e}") from e

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
        while len(embedding) < self.dimension:
            embedding.append(0.0)
        
        return embedding[:self.dimension]

    async def _reconstruct_memory_entry(self, match: Dict[str, Any]) -> Optional[MemoryEntry]:
        """Reconstruct MemoryEntry from Pinecone match."""
        try:
            metadata = match.get("metadata", {})
            
            memory_entry = MemoryEntry(
                id=match["id"],
                content=metadata.get("content", ""),
                embedding=match.get("values"),
                metadata={k: v for k, v in metadata.items() 
                         if k not in ["content", "namespace", "created_at", "updated_at", 
                                    "access_count", "importance_score"]},
                namespace=metadata.get("namespace", "default"),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.utcnow().isoformat())),
                access_count=metadata.get("access_count", 0),
                importance_score=metadata.get("importance_score", 0.5),
            )
            
            return memory_entry
            
        except Exception as e:
            self.logger.error(f"Failed to reconstruct memory entry: {e}")
            return None