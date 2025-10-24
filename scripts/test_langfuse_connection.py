#!/usr/bin/env python3
"""
Test script to verify LangFuse connection and tracing setup.
Run this script to check if LangFuse is properly configured and working.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.observability.langfuse_tracer import LangFuseTracer, LangFuseConfig
from core.observability import get_logger

async def test_langfuse_connection():
    """Test LangFuse connection and basic tracing."""
    logger = get_logger(__name__)
    
    print("🔍 Testing LangFuse Connection...")
    print("=" * 50)
    
    # Check environment variables
    print("📋 Environment Variables:")
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    host = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
    
    print(f"  LANGFUSE_PUBLIC_KEY: {'✅ Set' if public_key else '❌ Not set'}")
    print(f"  LANGFUSE_SECRET_KEY: {'✅ Set' if secret_key else '❌ Not set'}")
    print(f"  LANGFUSE_HOST: {host}")
    
    if not public_key or not secret_key:
        print("\n❌ Missing API keys! Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        print("   For local setup: export LANGFUSE_PUBLIC_KEY='your-key'")
        print("   For cloud setup: Get keys from https://cloud.langfuse.com")
        return False
    
    try:
        # Initialize tracer
        print("\n🔧 Initializing LangFuse Tracer...")
        config = LangFuseConfig(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            project_name="golden-agents-test",
            environment="test"
        )
        
        tracer = LangFuseTracer(config)
        print("✅ LangFuse tracer initialized successfully")
        
        # Test basic span creation
        print("\n🧪 Testing span creation...")
        
        with tracer.start_as_current_span("test_connection") as span:
            span.set_input({"test": "data", "message": "Hello from test script"})
            span.set_attribute("test_type", "connection_test")
            span.set_attribute("environment", "test")
            
            # Simulate some work
            await asyncio.sleep(0.1)
            
            span.set_output({"status": "success", "message": "Connection test completed"})
            print("✅ Test span created successfully")
        
        # Test flush
        print("\n💾 Testing data flush...")
        await tracer.flush()
        print("✅ Data flushed successfully")
        
        print("\n🎉 LangFuse connection test completed successfully!")
        print(f"   Check your traces at: {host}")
        print("   Look for traces with name 'test_connection'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ LangFuse connection test failed: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if LangFuse is running (for local setup)")
        print("2. Verify API keys are correct")
        print("3. Check network connectivity")
        print("4. Look at the error details above")
        return False

async def test_workflow_tracing():
    """Test tracing with a simple workflow simulation."""
    print("\n🔄 Testing Workflow Tracing...")
    print("=" * 50)
    
    try:
        from core.observability import get_tracer
        
        tracer = get_tracer()
        
        # Simulate a simple agent workflow
        with tracer.start_as_current_span("poller_agent") as poller_span:
            poller_span.set_input({"action": "poll_tickets", "source": "zendesk"})
            poller_span.set_attribute("agent_type", "poller")
            
            # Simulate memory agent
            with tracer.start_as_current_span("memory_agent") as memory_span:
                memory_span.set_input({"action": "check_duplicates", "ticket_id": "TEST-001"})
                memory_span.set_attribute("agent_type", "memory")
                memory_span.set_output({"action": "store_new", "related_tickets": 0})
            
            # Simulate triage agent
            with tracer.start_as_current_span("triage_agent") as triage_span:
                triage_span.set_input({"action": "analyze_incident", "ticket_id": "TEST-001"})
                triage_span.set_attribute("agent_type", "triage")
                triage_span.set_output({"severity": "medium", "routing": "support"})
            
            poller_span.set_output({"tickets_processed": 1, "status": "completed"})
        
        await tracer.flush()
        print("✅ Workflow tracing test completed")
        return True
        
    except Exception as e:
        print(f"❌ Workflow tracing test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 LangFuse Connection and Tracing Test")
    print("=" * 60)
    
    # Test basic connection
    connection_ok = await test_langfuse_connection()
    
    if connection_ok:
        # Test workflow tracing
        workflow_ok = await test_workflow_tracing()
        
        if workflow_ok:
            print("\n🎉 All tests passed! LangFuse is working correctly.")
            print("\n📊 Next steps:")
            print("1. Run the full workflow: python scripts/test_full_workflow.py")
            print("2. Check traces in LangFuse UI")
            print("3. Look for traces with names like 'poller_agent', 'memory_agent', etc.")
        else:
            print("\n⚠️  Connection works but workflow tracing failed")
    else:
        print("\n❌ Basic connection failed. Please fix the setup first.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
