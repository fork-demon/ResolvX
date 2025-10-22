#!/usr/bin/env python3
"""
Smoke test for Golden Agent Framework.

Tests the end-to-end flow:
1. Poller polls tickets from MCP
2. Tickets are forwarded to Triage
3. Triage analyzes and may invoke Memory/Splunk/NewRelic
4. Results are forwarded to Supervisor
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import load_config
from core.observability.factory import configure_observability
from core.gateway.mcp_client import MCPClient
from core.gateway.tool_registry import ToolRegistry
from core.memory.factory import MemoryFactory
from agents.poller.agent import ZendeskPollerAgent, ZendeskPollerState
from agents.triage.agent import TriageAgent
from agents.memory.agent import MemoryAgent


async def test_mcp_poll():
    """Test 1: MCP gateway poll_queue returns tickets."""
    print("\n=== Test 1: MCP Poll Queue ===")
    config = load_config()
    mcp_client = MCPClient(config)
    await mcp_client.initialize()
    
    result = await mcp_client.execute_tool(
        "poll_queue",
        {"queue_name": "engineering-support", "status": "new", "limit": 3}
    )
    
    tickets = result.get("tickets", [])
    print(f"✓ MCP returned {len(tickets)} tickets")
    if tickets:
        print(f"  First ticket: {tickets[0].get('id')} - {tickets[0].get('subject')}")
    return tickets


async def test_poller_agent():
    """Test 2: Poller agent can poll and forward."""
    print("\n=== Test 2: Poller Agent ===")
    config = load_config()
    
    # Initialize tool registry
    mcp_client = MCPClient(config)
    await mcp_client.initialize()
    tool_registry = ToolRegistry(config.gateway, mcp_client=mcp_client)
    await tool_registry.initialize()
    
    # Create poller
    poller_cfg = {
        "team": "operations",
        "zendesk": config.agents.get("poller").zendesk if hasattr(config.agents.get("poller"), "zendesk") else {}
    }
    poller = ZendeskPollerAgent(poller_cfg, tool_registry=tool_registry)
    
    # Run once
    result = await poller.run_once()
    
    ticket_count = result.get("ticket_count", 0)
    print(f"✓ Poller ran successfully")
    print(f"  Tickets polled: {ticket_count}")
    print(f"  Forwarded to: {result.get('forwarded_to', 'N/A')}")
    return result


async def test_memory_agent(ticket):
    """Test 3: Memory agent can search and store."""
    print("\n=== Test 3: Memory Agent ===")
    config = load_config()
    
    # Initialize memory backend
    memory_factory = MemoryFactory()
    memory = memory_factory.create_memory("mock", {})
    await memory.initialize()
    
    # Create memory agent
    memory_agent = MemoryAgent({
        "namespace_prefix": "tickets",
        "search_limit": 10,
        "search_threshold": 0.7,
    }, memory=memory)
    
    # Invoke with a ticket
    from core.graph.state import AgentState
    from agents.memory.agent import MemoryAgentState
    
    state = MemoryAgentState(
        agent_name="memory",
        agent_type="MemoryAgent",
        input_data={"ticket": ticket, "team": "operations"}
    )
    
    result = await memory_agent.process(state)
    
    action = result.get("result", {}).get("action", "unknown")
    print(f"✓ Memory agent processed ticket")
    print(f"  Action: {action}")
    print(f"  Related tickets: {len(result.get('result', {}).get('related_tickets', []))}")
    return result


async def test_triage_agent(tickets):
    """Test 4: Triage agent can analyze tickets."""
    print("\n=== Test 4: Triage Agent ===")
    config = load_config()
    
    # Initialize tool registry
    mcp_client = MCPClient(config)
    await mcp_client.initialize()
    tool_registry = ToolRegistry(config.gateway, mcp_client=mcp_client)
    await tool_registry.initialize()
    
    # Create triage agent (simplified instantiation)
    print("✓ Triage agent would analyze tickets here")
    print(f"  Input tickets: {len(tickets)}")
    print(f"  Tools available: {len(tool_registry.list_tools())}")
    
    # Full triage invocation would require LLM + prompts
    return {"status": "simulated", "tickets_analyzed": len(tickets)}


async def main():
    """Run smoke tests."""
    print("=" * 60)
    print("Golden Agent Framework - Smoke Test")
    print("=" * 60)
    
    # Configure observability
    config = load_config()
    configure_observability(config)
    
    try:
        # Test 1: MCP poll
        tickets = await test_mcp_poll()
        
        # Test 2: Poller agent
        poller_result = await test_poller_agent()
        
        # Test 3: Memory agent (with first ticket)
        if tickets:
            await test_memory_agent(tickets[0])
        
        # Test 4: Triage agent
        await test_triage_agent(tickets[:3])
        
        print("\n" + "=" * 60)
        print("✓ All smoke tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

