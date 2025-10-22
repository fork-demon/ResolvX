#!/usr/bin/env python3
"""
Full Workflow Integration Test
Tests: Poller → Triage → Memory → Supervisor with full tracing
"""

import asyncio
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
from core.graph.state import AgentState
from agents.poller.agent import ZendeskPollerAgent
from agents.triage.agent import TriageAgent
from agents.memory.agent import MemoryAgent, MemoryAgentState


async def main():
    """Run full workflow integration test."""
    print("\n" + "="*70)
    print("  Full Workflow Integration Test")
    print("  Poller → Triage → Memory → Supervisor (with tracing)")
    print("="*70)
    
    # Load config and setup observability
    config = load_config()
    configure_observability(config)
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    # MCP client + tool registry
    mcp_client = MCPClient(config)
    await mcp_client.initialize()
    tool_registry = ToolRegistry(config.gateway, mcp_client=mcp_client)
    await tool_registry.initialize()
    print(f"  ✓ Tool Registry: {len(tool_registry.list_tools())} tools")
    
    # Memory backend
    memory_factory = MemoryFactory()
    memory = memory_factory.create_memory("mock", {})
    await memory.initialize()
    print(f"  ✓ Memory Backend: mock")
    
    # Initialize agents
    print("\n🤖 Initializing agents...")
    
    # 1. Poller Agent
    poller_cfg = {
        "team": "operations",
        "zendesk": config.agents.get("poller").zendesk if hasattr(config.agents.get("poller"), "zendesk") else {}
    }
    poller = ZendeskPollerAgent(poller_cfg, tool_registry=tool_registry)
    print(f"  ✓ Poller Agent initialized")
    
    # 2. Triage Agent
    triage_config = config.agents.get("triage")
    if triage_config:
        triage = TriageAgent(
            config=triage_config,
            tool_registry=tool_registry,
            memory=None,
            rag=None
        )
        # Triage initialize() is sync, not async
        triage.initialize()
        print(f"  ✓ Triage Agent initialized")
    else:
        triage = None
        print(f"  ⚠ Triage Agent config not found, skipping")
    
    # 3. Memory Agent
    memory_agent = MemoryAgent({
        "namespace_prefix": "tickets",
        "search_limit": 10,
        "search_threshold": 0.7,
    }, memory=memory)
    print(f"  ✓ Memory Agent initialized")
    
    # Step 1: Poll tickets
    print(f"\n{'='*70}")
    print("1️⃣  POLLER: Polling tickets")
    print("="*70)
    
    poller_result = await poller.run_once()
    ticket_count = poller_result.get("ticket_count", 0)
    tickets = poller_result.get("tickets", [])
    print(f"  ✓ Polled {ticket_count} tickets")
    
    if not tickets:
        print("  ⚠ No tickets found, ending test")
        return
    
    # Take first ticket for processing
    ticket = tickets[0]
    print(f"  📋 Processing ticket: {ticket.get('id')} - {ticket.get('subject', 'N/A')[:50]}...")
    
    # Step 2: Memory Agent - Check for duplicates
    print(f"\n{'='*70}")
    print("2️⃣  MEMORY: Checking for duplicate tickets")
    print("="*70)
    
    memory_state = MemoryAgentState(
        agent_name="memory",
        agent_type="MemoryAgent",
        input_data={"ticket": ticket, "team": "operations"}
    )
    
    memory_result = await memory_agent.process(memory_state)
    action = memory_result.get("result", {}).get("action", "unknown")
    related_count = len(memory_result.get("result", {}).get("related_tickets", []))
    print(f"  ✓ Memory action: {action}")
    print(f"  ✓ Related tickets: {related_count}")
    
    # Step 3: Triage Agent - Analyze and route
    if triage:
        print(f"\n{'='*70}")
        print("3️⃣  TRIAGE: Analyzing ticket")
        print("="*70)
        
        triage_state = AgentState(
            agent_name="triage",
            agent_type="TriageAgent",
            input_data={
                "incident_id": ticket.get("id"),
                "tickets": [ticket],
                "context": {
                    "memory_result": memory_result,
                    "source": "poller"
                }
            }
        )
        
        triage_result = await triage.process(triage_state)
        severity = triage_result.get("severity", "unknown")
        routing_decision = triage_result.get("routing_decision", "unknown")
        print(f"  ✓ Severity: {severity}")
        print(f"  ✓ Routing: {routing_decision}")
    else:
        print(f"\n{'='*70}")
        print("3️⃣  TRIAGE: Skipped (not configured)")
        print("="*70)
        triage_result = None
    
    # Step 4: Supervisor - Final decision
    print(f"\n{'='*70}")
    print("4️⃣  SUPERVISOR: Making final decision")
    print("="*70)
    
    # Initialize Supervisor agent
    from agents.supervisor.agent import SupervisorAgent
    from core.config import AgentConfig
    supervisor_config = AgentConfig(
        team="operations",
        max_concurrent=20,
        health_check_interval=30,
        default_timeout=300
    )
    supervisor = SupervisorAgent(
        config=supervisor_config,
        memory=None,
        rag=None
    )
    supervisor.initialize()
    print(f"  ✓ Supervisor Agent initialized")
    
    # Prepare enriched data for supervisor
    supervisor_input = {
        "ticket_id": ticket.get("id"),
        "severity": triage_result.get("metadata", {}).get("severity", {}).get("severity", "unknown") if triage_result else "unknown",
        "routing_decision": triage_result.get("metadata", {}).get("routing", {}).get("routing_decision", "unknown") if triage_result else "unknown",
        "tools_used": triage_result.get("metadata", {}).get("analysis", {}).get("tools_used", []) if triage_result else [],
        "enrichment_sources": {
            "llm": bool(triage_result.get("metadata", {}).get("analysis", {}).get("llm_analysis")) if triage_result else False,
            "splunk": bool(triage_result.get("metadata", {}).get("analysis", {}).get("splunk_data")) if triage_result else False,
            "newrelic": bool(triage_result.get("metadata", {}).get("analysis", {}).get("newrelic_data")) if triage_result else False
        },
        "memory_action": action,
        "ticket_data": ticket
    }
    
    supervisor_state = AgentState(
        agent_name="supervisor",
        agent_type="SupervisorAgent",
        input_data=supervisor_input
    )
    
    supervisor_result = await supervisor.process(supervisor_state)
    final_decision = supervisor_result.get("decision", {})
    print(f"  ✓ Decision: {final_decision.get('action', 'unknown')}")
    print(f"  ✓ Reason: {final_decision.get('reason', 'N/A')[:60]}...")
    if final_decision.get('assigned_to'):
        print(f"  ✓ Assigned to: {final_decision.get('assigned_to')}")
    print(f"  ✓ Escalated: {final_decision.get('escalated', False)}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 WORKFLOW SUMMARY")
    print("="*70)
    print(f"  ✅ Poller: {ticket_count} tickets polled")
    print(f"  ✅ Memory: {action} ({related_count} related)")
    if triage_result:
        print(f"  ✅ Triage: {severity} / {routing_decision}")
        tools_used = triage_result.get("metadata", {}).get("analysis", {}).get("tools_used", [])
        if tools_used:
            print(f"      Tools used: {', '.join(tools_used)}")
    else:
        print(f"  ⏭️  Triage: Skipped")
    print(f"  ✅ Supervisor: {final_decision.get('action', 'unknown')}")
    if final_decision.get('assigned_to'):
        print(f"      Assigned to: {final_decision.get('assigned_to')}")
    
    print(f"\n{'='*70}")
    print("✅ Full Workflow Test Complete!")
    print("="*70)
    print(f"\n💡 Check LangFuse at http://localhost:3000 for traces")
    print(f"   You should see complete end-to-end tracing:")
    print(f"   - zendesk_poller_process (with input/output)")
    print(f"   - memory_agent_process (with input/output)")
    if triage_result:
        print(f"   - triage_process (with LLM + tools + input/output)")
        print(f"   - llm_chat_completion (LLM analysis)")
        if tools_used:
            for tool in tools_used:
                print(f"   - mcp_tool_{tool} (enrichment)")
    print(f"   - supervisor_process (with final decision + input/output)")


if __name__ == "__main__":
    asyncio.run(main())

