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
    tool_registry = ToolRegistry(config, mcp_client=mcp_client)  # Pass full config, not just gateway
    await tool_registry.initialize()
    print(f"  ✓ Tool Registry: {len(tool_registry.list_tools())} tools")
    
    # Memory backend (use FAISS from config for better duplicate detection)
    memory_factory = MemoryFactory()
    memory_backend = config.memory.backend
    memory_config = getattr(config.memory, memory_backend, {})
    if hasattr(memory_config, 'dict'):
        memory_config = memory_config.dict()
    memory = memory_factory.create_memory(memory_backend, memory_config)
    await memory.initialize()
    print(f"  ✓ Memory Backend: {memory_backend}")
    
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
    
    # Extract analysis data for supervisor
    analysis = triage_result.get("metadata", {}).get("analysis", {}) if triage_result else {}
    tools_used = analysis.get("tools_used", [])
    synthesis = analysis.get("synthesis", {})
    
    # Prepare enriched data for supervisor
    supervisor_input = {
        "ticket_id": ticket.get("id"),
        "triage_severity": triage_result.get("metadata", {}).get("severity", {}).get("severity", "unknown") if triage_result else "unknown",
        "triage_routing": triage_result.get("metadata", {}).get("routing", {}).get("routing_decision", "unknown") if triage_result else "unknown",
        "tools_used": tools_used,
        "enrichment_sources": {
            "llm": bool(analysis.get("llm_analysis")),
            "memory": related_count > 0,
            "rag": bool(synthesis.get("summary"))
        },
        "analysis": analysis,  # Include full analysis with synthesis
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
    
    # Step 5: Executor - Execute Decision
    print(f"\n{'='*70}")
    print("5️⃣  EXECUTOR: Executing decision")
    print("="*70)
    
    from agents.executor.agent import TicketExecutorAgent
    executor = TicketExecutorAgent(config.agents.get("executor"))
    print(f"  ✓ Executor Agent initialized")
    
    executor_state = AgentState(
        agent_name="executor",
        agent_type="TicketExecutorAgent",
        input_data={
            "ticket_id": ticket.get("id"),
            "decision": final_decision,
            "context": {
                "synthesis": synthesis,
                "tools_used": tools_used,
                "memory_action": action,
                "related_count": related_count,
                "requester_id": ticket.get("requester_id"),
                "ticket_metadata": {
                    "priority": ticket.get("priority"),
                    "created_at": ticket.get("created_at"),
                    "tags": ticket.get("tags", [])
                }
            }
        }
    )
    
    execution_result = await executor.process(executor_state)
    print(f"  ✓ Execution: {'SUCCESS' if execution_result.get('success') else 'FAILED'}")
    print(f"  ✓ Action: {execution_result.get('action', 'N/A')}")
    if execution_result.get('comment_added'):
        print(f"  ✓ Comment added to ticket")
    if execution_result.get('assigned_to'):
        print(f"  ✓ Assigned to: {execution_result.get('assigned_to')}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 WORKFLOW SUMMARY")
    print("="*70)
    print(f"  ✅ Poller: {ticket_count} tickets polled")
    print(f"  ✅ Memory: {action} ({related_count} related)")
    if triage_result:
        if tools_used:
            print(f"  ✅ Triage: {len(tools_used)} tools executed")
            print(f"      → {', '.join(tools_used)}")
        else:
            print(f"  ✅ Triage: Analysis complete (no tools)")
        if synthesis.get('root_cause'):
            print(f"      → Root Cause: {synthesis.get('root_cause')}")
    else:
        print(f"  ⏭️  Triage: Skipped")
    print(f"  ✅ Supervisor: {final_decision.get('action', 'unknown')}")
    if final_decision.get('assigned_to'):
        print(f"      → Assigned to: {final_decision.get('assigned_to')}")
    print(f"  ✅ Executor: {'SUCCESS' if execution_result.get('success') else 'FAILED'}")
    
    print(f"\n{'='*70}")
    print("✅ Full Workflow Test Complete!")
    print("="*70)
    print(f"\n💡 Check LangFuse at http://localhost:3000 for traces")
    print(f"   Search for ticket: {ticket.get('id')}")
    print(f"   You should see 5 complete agent traces:")
    print(f"   1. zendesk_poller_process")
    print(f"   2. memory_agent_process")
    if triage_result:
        print(f"   3. triage_process (with nested CoT spans)")
        print(f"      - rag_knowledge_search")
        print(f"      - cot_entity_extraction → llm_chat_completion")
        print(f"      - cot_plan_creation → llm_chat_completion")
        if tools_used:
            for tool in tools_used:
                print(f"      - mcp_tool_{tool}")
        print(f"      - cot_synthesis")
    print(f"   4. supervisor_process")
    print(f"   5. executor_process")


if __name__ == "__main__":
    asyncio.run(main())

