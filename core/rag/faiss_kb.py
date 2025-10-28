"""
FAISS-based Knowledge Base for runbooks and documentation.

Uses real SentenceTransformer embeddings (same as LocalKB) with FAISS indexing
for fast, persistent, and scalable semantic search.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSKnowledgeBase:
    """
    FAISS-based knowledge base with SentenceTransformer embeddings.
    
    Uses the SAME embedding model and similarity metric as LocalKB,
    but with FAISS for faster search and persistent storage.
    """
    
    def __init__(self, knowledge_dir: str = "kb", model_name: str = "all-MiniLM-L6-v2"):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.knowledge_dir = Path(knowledge_dir)
        self.model_name = model_name
        self.index_path = Path("./data/faiss_kb_index")
        
        # Document storage
        self._docs: List[Dict[str, Any]] = []
        
        # FAISS index
        self._index: Optional[faiss.Index] = None
        self._embedder = None
        self._dimension = 384  # all-MiniLM-L6-v2 dimension
    
    def load(self) -> None:
        """Load knowledge base: either from saved index or by re-indexing documents."""
        # Try to load existing index
        if self._load_from_disk():
            print(f"✓ Loaded existing FAISS index with {len(self._docs)} documents")
            return
        
        # First time: index all documents
        print(f"→ Indexing documents from {self.knowledge_dir}...")
        self._index_documents()
        print(f"✓ Indexed {len(self._docs)} documents")
        
        # Save for next time
        self._save_to_disk()
    
    def _load_from_disk(self) -> bool:
        """Load index and documents from disk."""
        index_file = self.index_path / "faiss.index"
        docs_file = self.index_path / "documents.pkl"
        
        if not (index_file.exists() and docs_file.exists()):
            return False
        
        try:
            # Load FAISS index
            self._index = faiss.read_index(str(index_file))
            
            # Load documents
            with open(docs_file, "rb") as f:
                self._docs = pickle.load(f)
            
            # Initialize embedder (needed for queries)
            self._initialize_embedder()
            
            return True
        except Exception as e:
            print(f"⚠️  Failed to load index: {e}")
            return False
    
    def _save_to_disk(self) -> None:
        """Save index and documents to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path / "faiss.index"))
        
        # Save documents
        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump(self._docs, f)
        
        print(f"✓ Saved FAISS index to {self.index_path}")
    
    def _initialize_embedder(self) -> None:
        """Initialize SentenceTransformer embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
                print(f"✓ Loaded embedding model: {self.model_name}")
            except Exception as e:
                print(f"⚠️  Failed to load embedding model: {e}")
                raise
    
    def _index_documents(self) -> None:
        """Index all markdown documents in knowledge_dir."""
        # Initialize embedder
        self._initialize_embedder()
        
        # Load all markdown files
        self._docs.clear()
        for doc_path in self.knowledge_dir.rglob("*.md"):
            try:
                text = doc_path.read_text(encoding="utf-8", errors="ignore")
                self._docs.append({
                    "path": str(doc_path),
                    "text": text,
                    "name": doc_path.name,
                    "type": self._classify_document(doc_path)
                })
            except Exception as e:
                print(f"⚠️  Failed to read {doc_path}: {e}")
        
        if not self._docs:
            print("⚠️  No documents found!")
            return
        
        # Generate embeddings
        print(f"   Generating embeddings for {len(self._docs)} documents...")
        texts = [d["text"] for d in self._docs]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        # Create FAISS index (Inner Product for normalized embeddings = cosine similarity)
        self._dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        
        # Add embeddings to index
        self._index.add(np.array(embeddings, dtype=np.float32))
        
        print(f"   ✓ Created FAISS index with {self._index.ntotal} vectors (dim: {self._dimension})")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search knowledge base using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of documents with scores (same format as LocalKB)
        """
        if not self._docs or self._index is None:
            return []
        
        # Initialize embedder if needed
        if self._embedder is None:
            self._initialize_embedder()
        
        # Generate query embedding
        query_embedding = self._embedder.encode([query], normalize_embeddings=True)[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        scores, indices = self._index.search(query_embedding, min(k, len(self._docs)))
        
        # Format results (same as LocalKB)
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self._docs):
                doc = self._docs[idx]
                results.append({
                    "path": doc["path"],
                    "score": float(score),  # Convert numpy to Python float
                    "text": doc["text"],
                    "name": doc["name"],
                    "type": doc.get("type", "unknown")
                })
        
        return results
    
    def _classify_document(self, path: Path) -> str:
        """Classify document type based on filename/path."""
        path_str = str(path).lower()
        name = path.name.lower()
        
        if "runbook" in name:
            return "runbook"
        elif "glossary" in name:
            return "glossary"
        elif "incident" in name:
            return "incident_type"
        elif "kb" in path_str or "knowledge" in path_str:
            return "kb_article"
        else:
            return "documentation"
    
    def reindex(self) -> None:
        """Force re-indexing of all documents."""
        print("→ Re-indexing knowledge base...")
        self._index_documents()
        self._save_to_disk()
        print("✓ Re-indexing complete!")
    
    def add_document(self, path: str, content: str) -> None:
        """
        Add a new document to the knowledge base.
        
        This requires re-indexing the entire collection.
        """
        doc_path = Path(path)
        self._docs.append({
            "path": str(doc_path),
            "text": content,
            "name": doc_path.name,
            "type": self._classify_document(doc_path)
        })
        
        # Re-index with new document
        self.reindex()
    
    def __len__(self) -> int:
        """Return number of indexed documents."""
        return len(self._docs)

