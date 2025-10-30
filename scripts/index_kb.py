#!/usr/bin/env python3
"""
Index Knowledge Base

Indexes all markdown files in kb/ folder for RAG search.
Creates FAISS index with embeddings for fast similarity search.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.rag.faiss_kb import FaissKnowledgeBase
from core.observability import get_logger

logger = get_logger("index_kb")


def main():
    """Index knowledge base."""
    try:
        logger.info("Loading configuration...")
        config = load_config("config/agent.yaml")
        
        logger.info("Initializing FAISS knowledge base...")
        kb = FaissKnowledgeBase(config.rag.config)
        
        # Get knowledge directory from config
        kb_dir = config.rag.knowledge_dir
        logger.info(f"Indexing files from {kb_dir}/...")
        
        # Count markdown files
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            logger.error(f"Knowledge directory not found: {kb_dir}")
            logger.info(f"Creating directory: {kb_dir}")
            kb_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Please add markdown (.md) files to {kb_dir}/ and run this script again")
            return
        
        md_files = list(kb_path.glob("*.md"))
        if not md_files:
            logger.warning(f"No markdown files found in {kb_dir}/")
            logger.info("Add .md files to the kb/ folder, then run this script again")
            return
        
        logger.info(f"Found {len(md_files)} markdown files")
        
        # Index files
        logger.info("Generating embeddings and creating FAISS index...")
        kb.index_files(kb_dir)
        
        # Verify index
        index_path = config.rag.config.get("index_path", "./data/faiss_kb_index")
        index_file = Path(index_path) / "index.faiss"
        
        if index_file.exists():
            logger.info(f"✅ Successfully indexed {len(md_files)} runbooks")
            logger.info(f"✅ FAISS index saved to {index_path}")
            
            # Test search
            logger.info("\nTesting search...")
            results = kb.search("basket segment issue", k=1)
            if results:
                logger.info(f"✅ Search test passed: Found '{results[0]['filename']}'")
            else:
                logger.warning("⚠️  Search test returned no results")
        else:
            logger.error(f"❌ Index file not created at {index_file}")
            return
        
        logger.info("\n" + "="*70)
        logger.info("✅ Knowledge Base Indexing Complete!")
        logger.info("="*70)
        logger.info(f"Indexed files: {len(md_files)}")
        logger.info(f"Index location: {index_path}")
        logger.info("\nThe Triage agent will now use this index for RAG search during analysis.")
        
    except Exception as e:
        logger.error(f"Failed to index knowledge base: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

