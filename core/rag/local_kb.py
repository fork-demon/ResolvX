"""
Local KB Vector Search: indexes markdown files in kb/ and provides semantic search.

Switch rag.backend to "local_kb" and set knowledge_dir to "kb" in config to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional


class LocalKB:
    def __init__(self, knowledge_dir: str = "kb", model_name: Optional[str] = None):
        self.knowledge_dir = Path(knowledge_dir)
        self.model_name = model_name
        self._docs: List[Dict[str, Any]] = []
        self._embedder = None
        self._embeddings: Optional[List[List[float]]] = None

    def load(self) -> None:
        self._docs.clear()
        if not self.knowledge_dir.exists():
            return
        for p in self.knowledge_dir.rglob("*.md"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                self._docs.append({"path": str(p), "text": text})
            except Exception:
                continue
        if self.model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
                self._embeddings = self._embedder.encode([d["text"] for d in self._docs], normalize_embeddings=True).tolist()
            except Exception:
                self._embedder = None
                self._embeddings = None

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._docs:
            return []
        if self._embedder and self._embeddings is not None:
            q = self._embedder.encode([query], normalize_embeddings=True)[0].tolist()
            def cos(a, b):
                import math
                dot = sum(x*y for x, y in zip(a, b))
                ma = math.sqrt(sum(x*x for x in a)) or 1.0
                mb = math.sqrt(sum(x*x for x in b)) or 1.0
                return dot/(ma*mb)
            scored = [(i, cos(q, emb)) for i, emb in enumerate(self._embeddings)]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:k]
            return [{"path": self._docs[i]["path"], "score": s, "text": self._docs[i]["text"]} for i, s in top]
        # Fallback keyword scoring
        qwords = set(query.lower().split())
        scored = []
        for i, d in enumerate(self._docs):
            words = set(d["text"].lower().split())
            score = len(qwords & words) / (len(qwords) + 1e-6)
            scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:k]
        return [{"path": self._docs[i]["path"], "score": s, "text": self._docs[i]["text"]} for i, s in top]


