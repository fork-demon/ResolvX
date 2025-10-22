#!/usr/bin/env python3
"""
Test Memory Agent Duplicate Detection.

Tests that the Memory Agent can:
1. Store a new ticket
2. Detect duplicate on second submission
3. Return appropriate actions
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import load_config
from core.observability.factory import configure_observability
from core.memory.factory import MemoryFactory
from core.graph.state import AgentState
from agents.memory.agent import MemoryAgent, MemoryAgentState


async def main():
    """Test memory duplicate detection."""
    print("\n" + "="*70)
    print("  Memory Agent - Duplicate Detection Test")
    print("="*70)
    
    # Setup
    config = load_config()
    configure_observability(config)
    
    # Initialize memory backend (using FAISS for better vector search)
    memory_factory = MemoryFactory()
    try:
        memory = memory_factory.create_memory("faiss", {
            "dimension": None,  # Auto-detect from embeddings
            "index_path": "/tmp/faiss_test_index",
            "index_type": "IndexFlatIP",  # Inner Product = Cosine Similarity for normalized vectors
            "metric": "IP"
        })
        await memory.initialize()
        print(f"  ✓ Memory Backend: FAISS (vector search)")
    except Exception as e:
        print(f"  ⚠️  FAISS not available ({e}), using mock")
        memory = memory_factory.create_memory("mock", {})
        await memory.initialize()
        print(f"  ✓ Memory Backend: Mock")
    
    # Create memory agent
    memory_agent = MemoryAgent({
        "namespace_prefix": "tickets",
        "search_limit": 10,
        "search_threshold": 0.9,  # 90% similarity for duplicates
        "duplicate_threshold": 0.95,  # 95% for exact match
        "embedding_model": "all-MiniLM-L6-v2"
    }, memory=memory)
    print(f"  ✓ Memory Agent initialized")
    
    # Test ticket
    ticket = {
        "id": "TEST-001",
        "subject": "Basket segments - File drop process failed",
        "description": "Basket segments feed LHS process failed with timeout error",
        "status": "new",
        "priority": "P3",
        "tags": ["basket-segments", "timeout"],
        "team": "operations"
    }
    
    print(f"\n{'='*70}")
    print("1️⃣  FIRST SUBMISSION: Store new ticket")
    print("="*70)
    
    state1 = MemoryAgentState(
        agent_name="memory",
        agent_type="MemoryAgent",
        input_data={"ticket": ticket, "team": "operations"}
    )
    
    result1 = await memory_agent.process(state1)
    action1 = result1.get("result", {}).get("action", "unknown")
    related1 = len(result1.get("related_tickets", []))
    
    print(f"  ✓ Action: {action1}")
    print(f"  ✓ Related tickets found: {related1}")
    
    if action1 == "stored_current_ticket":
        print(f"  ✅ Ticket stored successfully (no duplicates)")
    else:
        print(f"  ⚠️  Unexpected action: {action1}")
    
    print(f"\n{'='*70}")
    print("2️⃣  SECOND SUBMISSION: Same ticket (should detect duplicate)")
    print("="*70)
    
    state2 = MemoryAgentState(
        agent_name="memory",
        agent_type="MemoryAgent",
        input_data={"ticket": ticket, "team": "operations"}
    )
    
    result2 = await memory_agent.process(state2)
    action2 = result2.get("result", {}).get("action", "unknown")
    related2 = len(result2.get("related_tickets", []))
    decision2 = result2.get("result", {}).get("decision", {})
    
    print(f"  ✓ Action: {action2}")
    print(f"  ✓ Related tickets found: {related2}")
    if related2 > 0:
        top_match = result2.get("related_tickets", [])[0]
        print(f"  ✓ Top match score: {top_match.get('score', 0):.3f}")
        print(f"  ✓ Top match ID: {top_match.get('metadata', {}).get('ticket_id', 'unknown')}")
    
    if decision2:
        print(f"  ✓ Decision: {decision2}")
    
    if "duplicate" in action2:
        print(f"  ✅ Duplicate detected successfully!")
    elif action2 == "forward_to_supervisor":
        print(f"  ✅ Related ticket found, forwarding to supervisor")
    else:
        print(f"  ⚠️  Expected duplicate detection, got: {action2}")
    
    # Test with different ticket
    print(f"\n{'='*70}")
    print("3️⃣  THIRD SUBMISSION: Different ticket (should store as new)")
    print("="*70)
    
    ticket3 = {
        "id": "TEST-002",
        "subject": "Price discrepancy in Large Stores",
        "description": "Customer reported wrong price scanning at checkout",
        "status": "new",
        "priority": "P3",
        "tags": ["pricing", "discrepancy"],
        "team": "operations"
    }
    
    state3 = MemoryAgentState(
        agent_name="memory",
        agent_type="MemoryAgent",
        input_data={"ticket": ticket3, "team": "operations"}
    )
    
    result3 = await memory_agent.process(state3)
    action3 = result3.get("result", {}).get("action", "unknown")
    related3 = len(result3.get("related_tickets", []))
    
    print(f"  ✓ Action: {action3}")
    print(f"  ✓ Related tickets found: {related3}")
    
    if action3 == "stored_current_ticket":
        print(f"  ✅ New ticket stored (no duplicates)")
    else:
        print(f"  ⚠️  Unexpected action: {action3}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"  Test 1 (new ticket):        {action1}")
    print(f"  Test 2 (duplicate):         {action2}")
    print(f"  Test 3 (different ticket):  {action3}")
    print(f"\n{'='*70}")
    
    if action1 == "stored_current_ticket" and "duplicate" in action2 or "forward" in action2:
        print("✅ Memory Agent duplicate detection WORKING!")
    else:
        print("⚠️  Memory Agent needs attention")
    
    print("="*70)
    print(f"\n💡 Check LangFuse at http://localhost:3000")
    print(f"   Look for 'memory_agent_process' traces with actions")


if __name__ == "__main__":
    asyncio.run(main())

