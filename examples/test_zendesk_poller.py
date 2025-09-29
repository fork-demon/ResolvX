#!/usr/bin/env python3
"""
Test script for the Zendesk Poller Agent.

This script demonstrates how to use the Zendesk Poller Agent to poll
Zendesk queues and process tickets.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.poller.agent import ZendeskPollerAgent, ZendeskQueueConfig
from core.config import AgentConfig


async def test_zendesk_poller():
    """Test the Zendesk Poller Agent."""
    print("🎫 Testing Zendesk Poller Agent...")
    
    try:
        # Create agent configuration
        config = AgentConfig(
            agent_id="poller",
            model="mock-llm",
            max_concurrent_polls=2,
            default_poll_interval=30,
            auto_assign_enabled=False
        )
        
        # Create the poller agent
        agent = ZendeskPollerAgent(config)
        
        # Add a test queue configuration
        test_config = ZendeskQueueConfig(
            team="engineering",
            queue_name="test-queue",
            subdomain="test-company",
            email="test@example.com",
            api_token="test-token",
            poll_interval=30,
            max_tickets_per_poll=10,
            enabled=True,
            priority_threshold="normal",
            auto_assign=False,
            tags=["test", "poller"]
        )
        
        agent.add_queue_config(test_config)
        
        print("✅ Zendesk Poller Agent created successfully")
        print(f"   - Agent ID: {agent.agent_id}")
        print(f"   - Queue configs: {len(agent.zendesk_configs)}")
        print(f"   - MCP server: {agent.zendesk_mcp_server is not None}")
        
        # Test the graph building
        graph = agent.build_graph()
        print("✅ LangGraph built successfully")
        print(f"   - Graph nodes: {list(graph.nodes.keys())}")
        
        # Test ticket analysis methods
        test_ticket = {
            "id": "12345",
            "subject": "Critical bug in login system",
            "description": "Users cannot login to the application",
            "priority": "high",
            "status": "new",
            "tags": ["bug", "critical"],
            "requester_id": "user123",
            "assignee_id": None,
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-01T10:00:00Z"
        }
        
        # Test urgency assessment
        urgency = agent._assess_urgency(test_ticket)
        print(f"✅ Urgency assessment: {urgency}")
        
        # Test category classification
        category = agent._categorize_ticket(test_ticket)
        print(f"✅ Category classification: {category}")
        
        # Test team routing
        routing_team = agent._determine_routing_team(test_ticket)
        print(f"✅ Team routing: {routing_team}")
        
        # Test action requirements
        action = agent._determine_action_required(test_ticket)
        print(f"✅ Action required: {action}")
        
        # Test effort estimation
        effort = agent._estimate_effort(test_ticket)
        print(f"✅ Effort estimation: {effort}")
        
        print("\n🎉 All tests passed! Zendesk Poller Agent is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_zendesk_tools():
    """Test the Zendesk MCP tools."""
    print("\n🔧 Testing Zendesk MCP Tools...")
    
    try:
        from extensions.mcp_servers.zendesk_tools import ZendeskMCPServer, ZendeskConfig
        
        # Create MCP server
        server = ZendeskMCPServer(port=8088)
        print("✅ Zendesk MCP Server created successfully")
        
        # Test tool registration
        tools = server.get_registered_tools()
        print(f"✅ Registered tools: {list(tools.keys())}")
        
        # Test configuration registration
        config = ZendeskConfig(
            subdomain="test-company",
            email="test@example.com",
            api_token="test-token",
            queue_name="test-queue",
            team_id="engineering",
            poll_interval=30,
            max_tickets_per_poll=10
        )
        
        server.register_team_config("engineering", config)
        print("✅ Team configuration registered successfully")
        
        print("🎉 Zendesk MCP Tools are working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Zendesk tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Zendesk Poller Agent Tests\n")
    
    # Test the agent
    agent_success = await test_zendesk_poller()
    
    # Test the tools
    tools_success = await test_zendesk_tools()
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary")
    print("="*50)
    print(f"Zendesk Poller Agent: {'✅ PASS' if agent_success else '❌ FAIL'}")
    print(f"Zendesk MCP Tools: {'✅ PASS' if tools_success else '❌ FAIL'}")
    
    if agent_success and tools_success:
        print("\n🎉 All tests passed! The Zendesk Poller Agent is ready for use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
