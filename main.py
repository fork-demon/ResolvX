"""
Main entry point for the Golden Agent Framework.

Provides a FastAPI application for running agents and managing the framework.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import Config, EnvironmentSettings
from core.observability import get_logger
from core.graph.executor import GraphExecutor
from core.graph.builder import GraphBuilder
from core.memory.factory import MemoryFactory
from core.gateway.tool_registry import ToolRegistry
from core.gateway.llm_client import LLMGatewayClient
from core.prompts.manager import PromptManager


# Global variables for framework components
config: Config = None
graph_executor: GraphExecutor = None
memory_factory: MemoryFactory = None
tool_registry: ToolRegistry = None
llm_client: LLMGatewayClient = None
prompt_manager: PromptManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config, graph_executor, memory_factory, tool_registry, llm_client, prompt_manager
    
    # Initialize configuration
    config = Config.load()
    
    # Initialize core components
    memory_factory = MemoryFactory()
    tool_registry = ToolRegistry(config.gateway)
    llm_client = LLMGatewayClient(config.gateway)
    prompt_manager = PromptManager()
    
    # Initialize graph executor
    graph_executor = GraphExecutor(
        memory_factory=memory_factory,
        tool_registry=tool_registry,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
    )
    
    # Build agent graphs
    graph_builder = GraphBuilder()
    await graph_builder.build_all_graphs(config.agents)
    
    yield
    
    # Cleanup
    if llm_client:
        await llm_client.close()
    if tool_registry:
        await tool_registry.close()


# Create FastAPI application
app = FastAPI(
    title="Golden Agent Framework",
    description="A pluggable, central-gateway-aware framework for building and deploying AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AgentRequest(BaseModel):
    """Request model for agent execution."""
    agent_name: str
    input_data: Dict[str, Any]
    context: Dict[str, Any] = {}


class AgentResponse(BaseModel):
    """Response model for agent execution."""
    agent_name: str
    output: Dict[str, Any]
    execution_time: float
    status: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    components: Dict[str, str]
    timestamp: str


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {}
    
    # Check MCP Gateway
    try:
        if tool_registry:
            await tool_registry.health_check()
            components["mcp_gateway"] = "healthy"
        else:
            components["mcp_gateway"] = "not_initialized"
    except Exception as e:
        components["mcp_gateway"] = f"unhealthy: {e}"
    
    # Check Memory
    try:
        if memory_factory:
            memory = memory_factory.get_memory()
            await memory.health_check()
            components["memory"] = "healthy"
        else:
            components["memory"] = "not_initialized"
    except Exception as e:
        components["memory"] = f"unhealthy: {e}"
    
    # Check LLM Gateway
    try:
        if llm_client:
            await llm_client.health_check()
            components["llm_gateway"] = "healthy"
        else:
            components["llm_gateway"] = "not_initialized"
    except Exception as e:
        components["llm_gateway"] = f"unhealthy: {e}"
    
    # Overall status
    overall_status = "healthy" if all(
        status == "healthy" for status in components.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        components=components,
        timestamp=asyncio.get_event_loop().time(),
    )


@app.post("/agents/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute an agent with given input."""
    try:
        if not graph_executor:
            raise HTTPException(status_code=503, detail="Graph executor not initialized")
        
        # Execute agent
        start_time = asyncio.get_event_loop().time()
        result = await graph_executor.execute_agent(
            agent_name=request.agent_name,
            input_data=request.input_data,
            context=request.context,
        )
        end_time = asyncio.get_event_loop().time()
        
        return AgentResponse(
            agent_name=request.agent_name,
            output=result,
            execution_time=end_time - start_time,
            status="success",
        )
        
    except Exception as e:
        logger = get_logger("main")
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List available agents."""
    if not config:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    
    agents = []
    for agent_name, agent_config in config.agents.items():
        agents.append({
            "name": agent_name,
            "description": agent_config.description,
            "type": agent_config.type,
            "status": "available",
        })
    
    return {"agents": agents}


@app.get("/tools")
async def list_tools():
    """List available tools."""
    if not tool_registry:
        raise HTTPException(status_code=503, detail="Tool registry not initialized")
    
    tools = await tool_registry.list_tools()
    return {"tools": tools}


@app.get("/memory/stats")
async def memory_stats():
    """Get memory statistics."""
    if not memory_factory:
        raise HTTPException(status_code=503, detail="Memory factory not initialized")
    
    memory = memory_factory.get_memory()
    stats = await memory.get_stats()
    return {"stats": stats}


@app.get("/prompts")
async def list_prompts():
    """List available prompts."""
    if not prompt_manager:
        raise HTTPException(status_code=503, detail="Prompt manager not initialized")
    
    prompts = prompt_manager.list_prompts()
    return {"prompts": prompts}


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
