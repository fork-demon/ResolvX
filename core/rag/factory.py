"""
RAG Factory: Create RAG backend based on configuration.

Supports multiple backends:
1. local_kb: Simple local knowledge base (lightweight, no persistence)
2. faiss_kb: FAISS-based local KB (fast, persistent, production-ready)
3. global_rag: Global RAG service (API mode or local mode)

Usage:
    from core.rag.factory import create_rag_backend
    
    # FAISS (recommended for production)
    rag = create_rag_backend("faiss_kb", knowledge_dir="kb", model_name="all-MiniLM-L6-v2")
    rag.load()
    
    # Global RAG API
    rag = create_rag_backend("global_rag", mode="api", api_url="https://rag.company.com")
    await rag.initialize()
"""

from typing import Any, Dict, Optional, Union
from core.observability import get_logger


def create_rag_backend(
    backend_type: str,
    **kwargs
) -> Any:
    """
    Create RAG backend based on type.
    
    Args:
        backend_type: Type of RAG backend
            - "local_kb": Simple local KB (legacy)
            - "faiss_kb": FAISS-based KB (recommended)
            - "global_rag": Global RAG service (API or local)
        **kwargs: Backend-specific configuration
        
    Returns:
        RAG backend instance
        
    Examples:
        # FAISS KB (best for production)
        rag = create_rag_backend(
            "faiss_kb",
            knowledge_dir="kb",
            model_name="all-MiniLM-L6-v2"
        )
        rag.load()
        results = rag.search(query, k=3)
        
        # Global RAG API
        rag = create_rag_backend(
            "global_rag",
            mode="api",
            api_url="https://rag-service.company.com/api/v1",
            api_key="your-key"
        )
        await rag.initialize()
        results = await rag.search_async(query, k=3)
        
        # Local KB (legacy)
        rag = create_rag_backend(
            "local_kb",
            knowledge_dir="kb",
            model_name="all-MiniLM-L6-v2"
        )
        rag.load()
        results = rag.search(query, k=3)
    """
    logger = get_logger("rag.factory")
    
    if backend_type == "faiss_kb":
        from core.rag.faiss_kb import FAISSKnowledgeBase
        logger.info(f"Creating FAISS KB backend with config: {kwargs}")
        return FAISSKnowledgeBase(**kwargs)
    
    elif backend_type == "global_rag":
        from core.rag.global_rag import GlobalRAG
        mode = kwargs.get("mode", "local")
        logger.info(f"Creating Global RAG backend in {mode} mode with config: {kwargs}")
        return GlobalRAG(**kwargs)
    
    elif backend_type == "local_kb":
        from core.rag.local_kb import LocalKB
        logger.info(f"Creating Local KB backend (legacy) with config: {kwargs}")
        logger.warning("local_kb is deprecated - consider migrating to faiss_kb for better performance")
        return LocalKB(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown RAG backend: {backend_type}. "
            f"Supported backends: faiss_kb (recommended), global_rag, local_kb (deprecated)"
        )


def create_rag_from_config(config: Dict[str, Any]) -> Any:
    """
    Create RAG backend from configuration dictionary.
    
    Args:
        config: RAG configuration
            {
                "backend": "faiss_kb",  # or "global_rag", "local_kb"
                "knowledge_dir": "kb",
                "model_name": "all-MiniLM-L6-v2",
                "config": {
                    # Backend-specific config
                }
            }
    
    Returns:
        RAG backend instance
    """
    backend_type = config.get("backend", "faiss_kb")
    knowledge_dir = config.get("knowledge_dir", "kb")
    model_name = config.get("model_name")
    backend_config = config.get("config", {})
    
    # Build kwargs based on backend type (only pass valid arguments)
    if backend_type == "faiss_kb":
        # FAISS KB only accepts knowledge_dir and model_name
        kwargs = {
            "knowledge_dir": knowledge_dir,
            "model_name": model_name,
        }
    elif backend_type == "global_rag":
        # Global RAG accepts all config
        kwargs = {
            "knowledge_dir": knowledge_dir,
            "model_name": model_name,
            "mode": backend_config.get("mode", "local"),
            "api_url": backend_config.get("api_url"),
            "api_key": backend_config.get("api_key"),
        }
    elif backend_type == "local_kb":
        # Local KB only accepts knowledge_dir and model_name
        kwargs = {
            "knowledge_dir": knowledge_dir,
            "model_name": model_name,
        }
    else:
        kwargs = backend_config
    
    return create_rag_backend(backend_type, **kwargs)


# Convenience functions for common use cases

def create_faiss_rag(knowledge_dir: str = "kb", model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Create FAISS-based RAG (recommended for production)."""
    from core.rag.faiss_kb import FAISSKnowledgeBase
    return FAISSKnowledgeBase(knowledge_dir=knowledge_dir, model_name=model_name)


def create_api_rag(api_url: str, api_key: Optional[str] = None) -> Any:
    """Create Global RAG in API mode (for centralized RAG service)."""
    from core.rag.global_rag import GlobalRAG
    return GlobalRAG(mode="api", api_url=api_url, api_key=api_key)


def create_hybrid_rag(
    local_kb_dir: str = "kb",
    model_name: str = "all-MiniLM-L6-v2",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create hybrid RAG setup with both local FAISS and global API.
    
    Returns:
        Dict with "local" and "global" RAG instances
    """
    result = {
        "local": create_faiss_rag(local_kb_dir, model_name)
    }
    
    if api_url:
        result["global"] = create_api_rag(api_url, api_key)
    
    return result

