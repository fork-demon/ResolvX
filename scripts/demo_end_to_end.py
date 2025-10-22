#!/usr/bin/env python3
"""
End-to-End Demo: Golden Agent Framework

Demonstrates complete workflow:
1. Poller polls tickets from MCP
2. Triage uses glossary for entity extraction + RAG for runbook selection + LLM for planning
3. Tools executed (Splunk, NewRelic, Price APIs) via MCP
4. Supervisor makes final decision
5. Full trace captured in LangFuse
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
from core.gateway.llm_client import LLMGatewayClient, ChatMessage
from core.memory.factory import MemoryFactory
from core.rag.local_kb import LocalKB
from agents.poller.agent import ZendeskPollerAgent
from agents.memory.agent import MemoryAgent, MemoryAgentState
import yaml


async def load_glossary():
    """Load business domain glossary."""
    glossary_path = project_root / "resources" / "triage" / "glossary.yaml"
    with open(glossary_path) as f:
        return yaml.safe_load(f)


def extract_entities(ticket, glossary):
    """Extract business entities from ticket using glossary."""
    print(f"\n  📋 Entity Extraction:")
    
    subject = ticket.get("subject", "").lower()
    description = ticket.get("description", "").lower()
    text = f"{subject} {description}"
    
    entities = {}
    
    # Extract GTINs
    entities_dict = glossary.get("entities", {})
    for entity_key, entity in entities_dict.items():
        if entity_key == "GTIN":
            import re
            pattern = entity.get("pattern", r"\b\d{13,14}\b")
            matches = re.findall(pattern, text)
            if matches:
                entities["gtin"] = matches[0]
                print(f"    - GTIN: {matches[0]}")
    
    # Extract TPNBs
    for entity_key, entity in entities_dict.items():
        if entity_key == "TPNB":
            import re
            pattern = entity.get("pattern", r"\b\d{8}\b")
            matches = re.findall(pattern, text)
            if matches:
                entities["tpnb"] = matches[0]
                print(f"    - TPNB: {matches[0]}")
    
    # Extract Location Clusters
    for entity_key, entity in entities_dict.items():
        if entity_key == "LocationCluster":
            for cluster in entity.get("values", []):
                for synonym in cluster.get("synonyms", []):
                    if synonym.lower() in text:
                        entities["location_cluster"] = cluster.get("name")
                        entities["location_cluster_uuid"] = cluster.get("uuid")
                        print(f"    - Location Cluster: {cluster.get('name')} ({cluster.get('uuid')})")
                        break
    
    if not entities:
        print("    - No entities extracted")
    
    return entities


def map_entities_to_tools(entities, glossary):
    """Map extracted entities to candidate tools using glossary."""
    print(f"\n  🔧 Tool Mapping:")
    
    tool_candidates = set()
    
    # Use glossary tool_mapping rules
    tool_mapping = glossary.get("tool_mapping", {})
    
    if entities.get("gtin"):
        tools = tool_mapping.get("GTIN", {}).get("tools", [])
        tool_candidates.update(tools)
        print(f"    - GTIN → {tools}")
    
    if entities.get("tpnb"):
        tools = tool_mapping.get("TPNB", {}).get("tools", [])
        tool_candidates.update(tools)
        print(f"    - TPNB → {tools}")
    
    if entities.get("location_cluster_uuid"):
        tools = tool_mapping.get("LocationCluster", {}).get("tools", [])
        tool_candidates.update(tools)
        print(f"    - LocationCluster → {tools}")
    
    if not tool_candidates:
        print("    - No tools mapped")
    
    return list(tool_candidates)


async def search_kb_runbook(ticket, rag_backend):
    """Search KB for relevant runbook using RAG."""
    print(f"\n  📚 KB Runbook Search (RAG):")
    
    query = f"{ticket.get('subject', '')} {ticket.get('description', '')}"
    
    try:
        results = rag_backend.search(query, k=3)
        
        if results:
            print(f"    - Found {len(results)} relevant KB articles:")
            for i, res in enumerate(results[:2], 1):
                path = Path(res["path"]).name
                print(f"      {i}. Score {res['score']:.2f}: {path}")
            return results[0] if results else None
        else:
            print("    - No KB articles found")
            return None
    except Exception as e:
        print(f"    - RAG search failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def create_llm_plan(ticket, entities, tools, kb_article, llm_client):
    """Use LLM to create execution plan based on ticket, entities, tools, and runbook."""
    print(f"\n  🤖 LLM Planning (Ollama llama3.2):")
    
    # Build context
    kb_context = ""
    if kb_article and isinstance(kb_article, dict):
        kb_text = kb_article.get("text", "")
        kb_context = f"\n\nRelevant Runbook:\n{kb_text[:500]}..."
    
    system_prompt = f"""You are a triage agent analyzing support tickets.
Given a ticket, extracted entities, and available tools, create a diagnostic plan.

Extracted Entities: {json.dumps(entities, indent=2)}
Available Tools: {json.dumps(tools, indent=2)}
{kb_context}

Provide a concise JSON plan with:
- "analysis": Brief issue summary
- "tools_to_call": List of tool names to execute in order
- "parameters": Dict of tool_name -> parameters
- "expected_outcome": What we hope to find
"""
    
    user_prompt = f"""Ticket ID: {ticket.get('id')}
Subject: {ticket.get('subject')}
Description: {ticket.get('description')}
Priority: {ticket.get('priority')}
Source: {ticket.get('source')}

Create a diagnostic plan."""
    
    try:
        # Call LLM
        response = await llm_client.chat_completion(
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ],
            model="llama3.2",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Extract plan
        if response and response.choices:
            content = response.choices[0].get("message", {}).get("content", "")
            print(f"    - LLM Response ({len(content)} chars):")
            print(f"      {content[:200]}...")
            
            # Try to parse JSON from response
            try:
                # Find JSON block in response
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    plan = json.loads(json_match.group(0))
                    print(f"    - Extracted Plan:")
                    print(f"      Tools: {plan.get('tools_to_call', [])}")
                    return plan
            except Exception as e:
                print(f"    - Could not parse JSON plan: {e}")
            
            # Fallback: return raw response
            return {"analysis": content, "tools_to_call": [], "raw": True}
        
        return {"analysis": "LLM returned no response", "tools_to_call": []}
    
    except Exception as e:
        print(f"    - LLM call failed: {e}")
        return {"error": str(e), "tools_to_call": []}


async def execute_tools_from_plan(plan, entities, tool_registry):
    """Execute tools based on LLM plan."""
    print(f"\n  ⚙️  Tool Execution:")
    
    tools_to_call = plan.get("tools_to_call", [])
    parameters = plan.get("parameters", {})
    
    results = {}
    
    for tool_name in tools_to_call:
        try:
            # Get parameters for this tool
            tool_params = parameters.get(tool_name, {})
            
            # Merge with extracted entities if not provided
            if tool_name == "base_prices_get" and not tool_params.get("tpnb"):
                tool_params["tpnb"] = entities.get("tpnb")
                tool_params["locationClusterId"] = entities.get("location_cluster_uuid")
            
            if tool_name == "splunk_search" and not tool_params.get("query"):
                tool_params["query"] = f"index=price* {entities.get('tpnb', '')} error"
            
            if tool_name == "newrelic_metrics" and not tool_params.get("nrql"):
                tool_params["nrql"] = "SELECT average(duration) FROM Transaction SINCE 1 hour ago"
            
            print(f"    - Calling {tool_name}...")
            result = await tool_registry.call_tool(tool_name, tool_params)
            
            # Unwrap nested result if needed
            if isinstance(result, dict) and "result" in result:
                result = result["result"]
                if isinstance(result, dict) and "result" in result:
                    result = result["result"]
            
            results[tool_name] = result
            print(f"      ✓ Success: {str(result)[:100]}...")
        
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            results[tool_name] = {"error": str(e)}
    
    if not tools_to_call:
        print("    - No tools to execute")
    
    return results


async def forward_to_supervisor(ticket, entities, plan, tool_results):
    """Forward enriched ticket data to supervisor for final decision."""
    print(f"\n  👔 Supervisor Decision:")
    
    enriched_ticket = {
        "original_ticket": ticket,
        "extracted_entities": entities,
        "triage_plan": plan,
        "tool_results": tool_results,
        "timestamp": "2025-10-22T06:00:00Z"
    }
    
    # Supervisor would make final decision here
    # For demo, simulate decision logic
    
    has_errors = any("error" in str(r).lower() for r in tool_results.values())
    has_data = any(r for r in tool_results.values() if r and not isinstance(r, dict) or "error" not in r)
    
    if has_errors and not has_data:
        decision = "ESCALATE_TO_HUMAN"
        reason = "Tool execution failed, human investigation required"
    elif has_data:
        decision = "COMMENT_AND_ASSIGN"
        reason = "Diagnostic data collected, assign to pricing team"
    else:
        decision = "REQUEST_MORE_INFO"
        reason = "Insufficient data to make decision"
    
    print(f"    - Decision: {decision}")
    print(f"    - Reason: {reason}")
    print(f"    - Enriched data size: {len(json.dumps(enriched_ticket))} bytes")
    
    return {
        "decision": decision,
        "reason": reason,
        "enriched_ticket": enriched_ticket
    }


async def demo_single_ticket(ticket, config, tool_registry, rag_backend, llm_client, glossary):
    """Run full flow for a single ticket."""
    print(f"\n{'='*70}")
    print(f"🎫 Processing Ticket: {ticket.get('id')}")
    print(f"   Subject: {ticket.get('subject')}")
    print(f"{'='*70}")
    
    # Step 1: Extract entities using glossary
    entities = extract_entities(ticket, glossary)
    
    # Step 2: Map entities to candidate tools
    candidate_tools = map_entities_to_tools(entities, glossary)
    
    # Step 3: Search KB for runbook
    kb_article = await search_kb_runbook(ticket, rag_backend)
    
    # Step 4: Use LLM to create execution plan
    plan = await create_llm_plan(ticket, entities, candidate_tools, kb_article, llm_client)
    
    # Step 5: Execute tools from plan
    tool_results = await execute_tools_from_plan(plan, entities, tool_registry)
    
    # Step 6: Forward to supervisor
    supervisor_decision = await forward_to_supervisor(ticket, entities, plan, tool_results)
    
    return supervisor_decision


async def main():
    """Run end-to-end demo."""
    print("\n" + "="*70)
    print("  Golden Agent Framework - End-to-End Demo")
    print("  Poller → Triage (Glossary + RAG + LLM) → Tools → Supervisor")
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
    
    # LLM client (Ollama)
    llm_client = LLMGatewayClient(config.gateway)
    await llm_client.initialize()
    print(f"  ✓ LLM Gateway: {llm_client.gateway_url} (model: {llm_client.default_model})")
    
    # RAG backend for KB
    kb_cfg = config.rag.get("local_kb", {}) if isinstance(config.rag, dict) else {}
    kb_dir = kb_cfg.get("knowledge_dir", "kb")
    rag_backend = LocalKB(knowledge_dir=kb_dir, model_name="all-MiniLM-L6-v2")
    rag_backend.load()
    print(f"  ✓ RAG Backend: {rag_backend.knowledge_dir} ({len(rag_backend._docs)} docs loaded)")
    
    # Load glossary
    glossary = await load_glossary()
    print(f"  ✓ Glossary: {len(glossary.get('entities', []))} entity types")
    
    # Poller
    poller_cfg = {
        "team": "operations",
        "zendesk": getattr(config.agents.get("poller"), "zendesk", {})
    }
    poller = ZendeskPollerAgent(poller_cfg, tool_registry=tool_registry)
    print(f"  ✓ Poller Agent: {len(poller.zendesk_configs)} queues configured")
    
    # Step 1: Poll tickets
    print(f"\n{'='*70}")
    print("1️⃣  POLLER: Polling tickets from MCP gateway")
    print("="*70)
    
    poller_result = await poller.run_once()
    tickets = poller_result.get("tickets", [])
    
    print(f"\n  ✓ Polled {len(tickets)} tickets")
    if tickets:
        for i, t in enumerate(tickets[:3], 1):
            print(f"    {i}. {t.get('id')}: {t.get('subject')[:60]}...")
    
    # Step 2-6: Process each ticket through triage
    print(f"\n{'='*70}")
    print("2️⃣  TRIAGE: Processing tickets with Glossary + RAG + LLM + Tools")
    print("="*70)
    
    results = []
    for ticket in tickets[:2]:  # Process first 2 tickets for demo
        try:
            result = await demo_single_ticket(
                ticket, config, tool_registry, rag_backend, llm_client, glossary
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Failed to process {ticket.get('id')}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print("="*70)
    print(f"  Total tickets polled: {len(tickets)}")
    print(f"  Tickets processed: {len(results)}")
    print(f"  Decisions:")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r.get('decision', 'N/A')}: {r.get('reason', '')[:50]}...")
    
    print(f"\n{'='*70}")
    print("✅ End-to-End Demo Complete!")
    print("="*70)
    langfuse_cfg = config.observability.get("langfuse", {}) if isinstance(config.observability, dict) else {}
    langfuse_host = langfuse_cfg.get("host", "http://localhost:3000")
    langfuse_key = langfuse_cfg.get("public_key", "dev")
    print(f"\n💡 Check LangFuse at {langfuse_host} for traces")
    print(f"   (Public Key: {langfuse_key})")


if __name__ == "__main__":
    asyncio.run(main())

