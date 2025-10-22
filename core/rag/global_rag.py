"""
Global RAG Service: Centralized knowledge base with API support.

Supports two modes:
1. Local mode: Indexes local knowledge files (like LocalKB)
2. API mode: Connects to external RAG service via REST API

This is useful for:
- Enterprise-wide knowledge base (Confluence, SharePoint, etc.)
- Centralized document indexing service
- Shared RAG across multiple teams
"""

from __future__ import annotations

import os
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.observability import get_logger, get_tracer


class GlobalRAG:
    """
    Global RAG service with local and API support.
    
    Can work in two modes:
    - Local: Index local files (similar to LocalKB)
    - API: Connect to external RAG service (e.g., company-wide knowledge base)
    """
    
    def __init__(
        self,
        knowledge_dir: str = "knowledge",
        model_name: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        mode: str = "local"
    ):
        """
        Initialize Global RAG.
        
        Args:
            knowledge_dir: Local knowledge directory (for local mode)
            model_name: Embedding model name (for local mode)
            api_url: External RAG API URL (for API mode)
            api_key: API key for authentication (for API mode)
            mode: "local" or "api"
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.mode = mode
        
        self.logger = get_logger("rag.global")
        self.tracer = get_tracer("rag.global")
        
        # Local mode storage
        self._docs: List[Dict[str, Any]] = []
        self._embedder = None
        self._embeddings: Optional[List[List[float]]] = None
        
        # API mode client
        self._http_client: Optional[httpx.AsyncClient] = None
        
        self.logger.info(f"Initialized Global RAG in {mode} mode")

    def load(self) -> None:
        """Load knowledge base (local mode only)."""
        if self.mode != "local":
            self.logger.info("Skipping load - API mode enabled")
            return
        
        self._docs.clear()
        if not self.knowledge_dir.exists():
            self.logger.warning(f"Knowledge directory not found: {self.knowledge_dir}")
            return
        
        # Load all documents
        for p in self.knowledge_dir.rglob("*"):
            if p.is_file() and p.suffix in {".md", ".txt", ".yaml", ".yml", ".json"}:
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    self._docs.append({
                        "path": str(p),
                        "text": text,
                        "metadata": {"source": p.name}
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to load {p}: {e}")
        
        self.logger.info(f"Loaded {len(self._docs)} documents")
        
        # Initialize embeddings if model specified
        if self.model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
                texts = [d["text"] for d in self._docs]
                self._embeddings = self._embedder.encode(texts, normalize_embeddings=True).tolist()
                self.logger.info(f"Generated embeddings using {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to generate embeddings: {e}")
                self._embedder = None
                self._embeddings = None

    async def initialize(self) -> None:
        """Initialize RAG service (API mode)."""
        if self.mode == "api":
            self._http_client = httpx.AsyncClient(
                base_url=self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                timeout=30.0
            )
            self.logger.info(f"Initialized HTTP client for API: {self.api_url}")
        else:
            # Local mode - load documents
            self.load()

    async def close(self) -> None:
        """Close HTTP client if in API mode."""
        if self._http_client:
            await self._http_client.aclose()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge base (local mode, synchronous).
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of results with path, score, text
        """
        if self.mode == "api":
            raise ValueError("Use async search_async() for API mode")
        
        return self._search_local(query, k)
    
    async def search_async(self, query: str, k: int = 5, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search knowledge base (async, supports both local and API).
        
        Args:
            query: Search query
            k: Number of results
            namespace: Optional namespace filter (API mode)
            
        Returns:
            List of results with path, score, text, metadata
        """
        with self.tracer.start_as_current_span("global_rag_search") as span:
            if hasattr(span, 'set_input'):
                span.set_input({"query": query[:200], "k": k, "mode": self.mode, "namespace": namespace})
            
            try:
                if self.mode == "api":
                    results = await self._search_api(query, k, namespace)
                else:
                    results = self._search_local(query, k)
                
                if hasattr(span, 'set_output'):
                    span.set_output({
                        "results_found": len(results),
                        "top_scores": [r.get("score", 0) for r in results[:3]],
                        "mode": self.mode
                    })
                
                return results
                
            except Exception as e:
                self.logger.error(f"RAG search failed: {e}")
                if hasattr(span, 'set_output'):
                    span.set_output({"success": False, "error": str(e)})
                raise

    def _search_local(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search local knowledge base using embeddings or keyword matching."""
        if not self._docs:
            return []
        
        # Use embeddings if available
        if self._embedder and self._embeddings is not None:
            q_emb = self._embedder.encode([query], normalize_embeddings=True)[0].tolist()
            
            # Cosine similarity
            def cosine_similarity(a, b):
                import math
                dot = sum(x*y for x, y in zip(a, b))
                mag_a = math.sqrt(sum(x*x for x in a)) or 1.0
                mag_b = math.sqrt(sum(x*x for x in b)) or 1.0
                return dot / (mag_a * mag_b)
            
            scored = [(i, cosine_similarity(q_emb, emb)) for i, emb in enumerate(self._embeddings)]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:k]
            
            return [
                {
                    "path": self._docs[i]["path"],
                    "score": score,
                    "text": self._docs[i]["text"],
                    "metadata": self._docs[i].get("metadata", {})
                }
                for i, score in top
            ]
        
        # Fallback: keyword-based scoring
        query_words = set(query.lower().split())
        scored = []
        for i, doc in enumerate(self._docs):
            doc_words = set(doc["text"].lower().split())
            score = len(query_words & doc_words) / (len(query_words) + 1e-6)
            scored.append((i, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        
        return [
            {
                "path": self._docs[i]["path"],
                "score": score,
                "text": self._docs[i]["text"],
                "metadata": self._docs[i].get("metadata", {})
            }
            for i, score in top
        ]

    async def _search_api(self, query: str, k: int = 5, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search external RAG service via API.
        
        Expected API endpoint: POST /search
        Request: {"query": "...", "k": 5, "namespace": "tickets"}
        Response: {
            "results": [
                {"text": "...", "score": 0.85, "metadata": {...}},
                ...
            ]
        }
        """
        if not self._http_client:
            raise ValueError("HTTP client not initialized - call initialize() first")
        
        try:
            response = await self._http_client.post(
                "/search",
                json={
                    "query": query,
                    "k": k,
                    "namespace": namespace,
                    "include_metadata": True
                }
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            self.logger.info(f"API RAG search returned {len(results)} results")
            return results
            
        except httpx.HTTPError as e:
            self.logger.error(f"RAG API request failed: {e}")
            # Fallback to empty results
            return []
        except Exception as e:
            self.logger.error(f"RAG API search failed: {e}")
            return []


# Example usage with both modes:
"""
# Local mode (current):
local_rag = GlobalRAG(knowledge_dir="kb", model_name="all-MiniLM-L6-v2", mode="local")
local_rag.load()
results = local_rag.search("basket segments failure", k=3)

# API mode (for centralized RAG service):
api_rag = GlobalRAG(
    api_url="https://rag-service.tesco.com/api/v1",
    api_key="your-api-key",
    mode="api"
)
await api_rag.initialize()
results = await api_rag.search_async("basket segments failure", k=3, namespace="pricing")

# Hybrid mode (use both):
# 1. Search local KB first (fast, team-specific)
local_results = local_rag.search(query, k=3)

# 2. If not enough results, search global RAG (comprehensive, company-wide)
if not local_results or local_results[0]["score"] < 0.5:
    global_results = await api_rag.search_async(query, k=5)
    results = local_results + global_results
"""
