#!/usr/bin/env python3
"""Test FAISS persistence and duplicate detection."""
import asyncio
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.memory.factory import MemoryFactory


async def main():
    """Test FAISS memory persistence."""
    index_path = "/tmp/test_faiss_pers"
    
    # Clean start
    import shutil
    if Path(index_path).exists():
        shutil.rmtree(index_path)
    
    print("=" * 70)
    print("RUN 1: Store a ticket")
    print("=" * 70)
    
    # Create memory backend
    factory = MemoryFactory()
    memory1 = factory.create_memory("faiss", {
        "dimension": None,
        "index_type": "IndexFlatIP",
        "metric": "IP",
        "index_path": index_path
    })
    await memory1.initialize()
    
    # Store a ticket
    ticket_text = "Basket segments - File drop process failed with timeout"
    embedding1 = [0.1, 0.2, 0.3] * 128  # 384-dim mock embedding
    
    await memory1.store(
        content=ticket_text,
        embedding=embedding1,
        metadata={"ticket_id": "TEST-001"},
        namespace="tickets:operations"
    )
    print(f"  ✓ Stored ticket")
    print(f"  ✓ Embedding (first 5): {embedding1[:5]}")
    
    # Close (to ensure save)
    await memory1.close()
    
    print(f"\n  Files saved: {list(Path(index_path).glob('*'))}")
    
    print("\n" + "=" * 70)
    print("RUN 2: Load index and search for SAME ticket")
    print("=" * 70)
    
    # Create new memory instance (simulates new process)
    memory2 = factory.create_memory("faiss", {
        "dimension": None,
        "index_type": "IndexFlatIP",
        "metric": "IP",
        "index_path": index_path
    })
    await memory2.initialize()
    print(f"  ✓ Loaded index")
    
    # Search for SAME ticket
    results = await memory2.search(
        query=ticket_text,
        namespace="tickets:operations",
        limit=5,
        threshold=0.0,  # Show all results
        query_embedding=embedding1  # SAME embedding
    )
    
    print(f"  ✓ Search results: {len(results)}")
    for i, result in enumerate(results):
        print(f"    Result {i+1}:")
        print(f"      Score: {result.score:.4f}")
        print(f"      Distance: {result.distance:.4f}")
        print(f"      Content: {result.entry.content[:50]}...")
        print(f"      Metadata: {result.entry.metadata}")
    
    if results and results[0].score > 0.95:
        print(f"\n  ✅ PERFECT MATCH! (score={results[0].score:.4f})")
    elif results and results[0].score > 0.7:
        print(f"\n  ✅ DUPLICATE DETECTED! (score={results[0].score:.4f})")
    else:
        print(f"\n  ⚠️  NO DUPLICATE (score={results[0].score if results else 'N/A'})")


if __name__ == "__main__":
    asyncio.run(main())

