"""
FAISS-based memory implementation for the Golden Agent Framework.

Provides vector-based memory storage and retrieval using Facebook's FAISS library
for efficient similarity search and clustering.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from core.memory.base import BaseMemory, MemoryEntry, SearchResult
from core.exceptions import MemoryError
from core.observability import get_logger


class FAISSMemory(BaseMemory):
    """
    FAISS-based memory implementation.
    
    Uses Facebook's FAISS library for efficient vector similarity search
    and memory storage with configurable index types.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FAISS memory backend.
        
        Args:
            config: FAISS-specific configuration
        """
        if not FAISS_AVAILABLE:
            raise MemoryError("FAISS library not available. Install with: pip install faiss-cpu")
        
        super().__init__(config)
        self.logger = get_logger("memory.faiss")
        
        # FAISS configuration
        self.index_path = self.config.get("index_path", "./data/faiss_index")
        self.dimension = self.config.get("dimension", 1536)
        self.index_type = self.config.get("index_type", "IndexFlatL2")
        self.metric = self.config.get("metric", "L2")
        
        # FAISS index
        self._index: Optional[faiss.Index] = None
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._metadata_store: Dict[str, MemoryEntry] = {}
        
        # Embedding model
        self._embedding_model = None
        self._embedding_cache: Dict[str, List[float]] = {}

    async def initialize(self) -> None:
        """Initialize the FAISS memory backend."""
        try:
            # Create index directory
            Path(self.index_path).mkdir(parents=True, exist_ok=True)
            
            # Initialize or load FAISS index
            await self._initialize_index()
            
            # Initialize embedding model if specified
            await self._initialize_embedding_model()
            
            self._initialized = True
            self.logger.info(f"FAISS memory initialized with {self.dimension}D vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS memory: {e}")
            raise MemoryError(f"FAISS memory initialization failed: {e}") from e

    async def close(self) -> None:
        """Close the FAISS memory backend."""
        try:
            # Save index and metadata
            await self._save_index()
            
            self._initialized = False
            self.logger.info("FAISS memory closed")
            
        except Exception as e:
            self.logger.error(f"Error closing FAISS memory: {e}")

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
            
            # Add to FAISS index
            vector = np.array([memory_entry.embedding], dtype=np.float32)
            self._index.add(vector)
            
            # Update mappings
            index_id = self._index.ntotal - 1
            self._id_to_index[memory_entry.id] = index_id
            self._index_to_id[index_id] = memory_entry.id
            self._metadata_store[memory_entry.id] = memory_entry
            
            self.logger.debug(f"Stored memory entry {memory_entry.id} at index {index_id}")
            return memory_entry.id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory entry: {e}")
            raise MemoryError(f"Memory storage failed: {e}") from e

    async def retrieve(self, memory_id: str, namespace: str = "default") -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry."""
        try:
            if memory_id not in self._metadata_store:
                return None
            
            memory_entry = self._metadata_store[memory_id]
            
            # Check namespace
            if memory_entry.namespace != namespace:
                return None
            
            # Update access tracking
            memory_entry.update_access()
            
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
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            k = min(limit * 2, self._index.ntotal)  # Get more results for filtering
            distances, indices = self._index.search(query_vector, k)
            
            results = []
            for distance, index in zip(distances[0], indices[0]):
                if index == -1:  # Invalid index
                    continue
                
                memory_id = self._index_to_id.get(index)
                if not memory_id or memory_id not in self._metadata_store:
                    continue
                
                memory_entry = self._metadata_store[memory_id]
                
                # Filter by namespace
                if memory_entry.namespace != namespace:
                    continue
                
                # Calculate similarity score (convert distance to similarity)
                similarity = self._distance_to_similarity(distance)
                
                # Apply threshold
                if similarity < threshold:
                    continue
                
                # Create search result
                result = SearchResult(
                    entry=memory_entry,
                    score=similarity,
                    distance=distance,
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
            if memory_id not in self._metadata_store:
                return False
            
            memory_entry = self._metadata_store[memory_id]
            
            # Check namespace
            if memory_entry.namespace != namespace:
                return False
            
            # Remove from mappings
            if memory_id in self._id_to_index:
                index_id = self._id_to_index[memory_id]
                del self._id_to_index[memory_id]
                del self._index_to_id[index_id]
            
            # Remove from metadata store
            del self._metadata_store[memory_id]
            
            # Note: FAISS doesn't support deletion, so we mark as deleted
            # In a production system, you'd need to rebuild the index periodically
            
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
            if memory_id not in self._metadata_store:
                return False
            
            memory_entry = self._metadata_store[memory_id]
            
            # Check namespace
            if memory_entry.namespace != namespace:
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
            memory_entry.updated_at = memory_entry.updated_at
            
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
            # Filter by namespace
            memories = [
                entry for entry in self._metadata_store.values()
                if entry.namespace == namespace
            ]
            
            # Apply pagination
            if offset > 0:
                memories = memories[offset:]
            
            if limit is not None:
                memories = memories[:limit]
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to list memories: {e}")
            raise MemoryError(f"Memory listing failed: {e}") from e

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all memories in a namespace."""
        try:
            # Find memories in namespace
            to_delete = [
                memory_id for memory_id, entry in self._metadata_store.items()
                if entry.namespace == namespace
            ]
            
            # Delete each memory
            deleted_count = 0
            for memory_id in to_delete:
                if await self.delete(memory_id, namespace):
                    deleted_count += 1
            
            self.logger.info(f"Cleared {deleted_count} memories from namespace {namespace}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to clear namespace {namespace}: {e}")
            raise MemoryError(f"Namespace clearing failed: {e}") from e

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            total_memories = len(self._metadata_store)
            
            if namespace:
                namespace_memories = len([
                    entry for entry in self._metadata_store.values()
                    if entry.namespace == namespace
                ])
            else:
                namespace_memories = total_memories
            
            # Calculate namespace distribution
            namespace_counts = {}
            for entry in self._metadata_store.values():
                ns = entry.namespace
                namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
            
            return {
                "total_memories": total_memories,
                "namespace_memories": namespace_memories,
                "namespace_counts": namespace_counts,
                "index_size": self._index.ntotal if self._index else 0,
                "dimension": self.dimension,
                "index_type": self.index_type,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            raise MemoryError(f"Stats retrieval failed: {e}") from e

    async def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        index_file = Path(self.index_path) / "index.faiss"
        metadata_file = Path(self.index_path) / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            # Load existing index
            self._index = faiss.read_index(str(index_file))
            
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self._id_to_index = data.get("id_to_index", {})
                self._index_to_id = data.get("index_to_id", {})
                self._metadata_store = data.get("metadata_store", {})
            
            self.logger.info(f"Loaded existing FAISS index with {self._index.ntotal} vectors")
        else:
            # Create new index
            if self.index_type == "IndexFlatL2":
                self._index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IndexFlatIP":
                self._index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexIVFFlat":
                quantizer = faiss.IndexFlatL2(self.dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                # Default to IndexFlatL2
                self._index = faiss.IndexFlatL2(self.dimension)
            
            self.logger.info(f"Created new FAISS index: {self.index_type}")

    async def _save_index(self) -> None:
        """Save FAISS index and metadata."""
        try:
            index_file = Path(self.index_path) / "index.faiss"
            metadata_file = Path(self.index_path) / "metadata.pkl"
            
            # Save FAISS index
            faiss.write_index(self._index, str(index_file))
            
            # Save metadata
            metadata_data = {
                "id_to_index": self._id_to_index,
                "index_to_id": self._index_to_id,
                "metadata_store": self._metadata_store,
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata_data, f)
            
            self.logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index: {e}")

    async def _initialize_embedding_model(self) -> None:
        """Initialize embedding model."""
        # This would integrate with actual embedding models
        # For now, we'll use a simple mock implementation
        self.logger.info("Using mock embedding model")

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
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
        
        embedding = embedding[:self.dimension]
        
        # Cache the embedding
        self._embedding_cache[text] = embedding
        
        return embedding

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert FAISS distance to similarity score."""
        if self.metric == "L2":
            # Convert L2 distance to similarity (0-1)
            return 1.0 / (1.0 + distance)
        elif self.metric == "IP":
            # Inner product is already a similarity measure
            return max(0.0, min(1.0, distance))
        else:
            # Default conversion
            return 1.0 / (1.0 + distance)