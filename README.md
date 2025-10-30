# TescoResolveX

**AI-Powered Support Automation Framework** - Multi-agent system for intelligent ticket triage, routing, and resolution with full observability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technical Components](#technical-components)
- [Agentic Pattern](#agentic-pattern)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Project Structure](#project-structure)
- [Observability](#observability)
- [Development](#development)
- [Testing](#testing)

---

## 🎯 Overview

**TescoResolveX** is an enterprise-grade AI agent framework designed to automate support operations by intelligently analyzing, routing, and resolving incidents. The system uses a **multi-agent architecture** powered by Large Language Models (LLMs) with built-in observability, tool orchestration, and knowledge base integration.

### What This Template Does

1. **Polls Support Tickets** from Zendesk queues on a schedule
2. **Detects Duplicates** using FAISS vector memory (similarity search)
3. **Analyzes Incidents** using Chain-of-Thought reasoning with LLMs
4. **Executes Diagnostic Tools** via MCP (Model Context Protocol) gateways
5. **Retrieves Knowledge Base** articles using RAG (Retrieval-Augmented Generation)
6. **Makes Routing Decisions** with a Supervisor agent
7. **Executes Actions** (add comments, assign teams, escalate) via Executor agent
8. **Traces Everything** in LangFuse for debugging and monitoring

---

## ✨ Key Features

### 🤖 Multi-Agent System
- **5 Specialized Agents**: Poller → Memory → Triage → Supervisor → Executor
- **LangGraph-based** state machines with retry logic and error handling
- **Clear separation of concerns**: Analysis vs. Execution

### 🧠 Chain-of-Thought (CoT) Reasoning
- **Step 1**: Entity extraction (GTIN, TPNB, incident type, key terms)
- **Step 2**: RAG knowledge base search for relevant runbooks
- **Step 3**: Execution plan creation with tool selection
- **Step 4**: Automatic parameter mapping (schema-driven, no hallucination)
- **Step 5**: Tool execution via MCP gateways
- **Step 6**: Synthesis - actionable summary with root cause analysis

### 🔧 Tool Orchestration
- **MCP Protocol** for microservice tool discovery
- **Dynamic tool registry** - no hardcoding required
- **Automatic parameter mapping** using JSON schemas
- **13+ tools** including Splunk, Price APIs, SharePoint, NewRelic

### 📚 Knowledge Base (RAG)
- **FAISS-powered** vector search for fast KB retrieval
- **Markdown runbooks** indexed automatically from `kb/` folder
- **Relevance scoring** to fetch the most appropriate guidance

### 🔍 Full Observability
- **LangFuse integration** - trace every agent, LLM call, and tool execution
- **Nested span tracking** - see the complete call hierarchy
- **Input/output logging** - debug with full context
- **Performance metrics** - latency, token usage, success rates

### 🛡️ Production-Ready
- **Circuit breakers** for external service failures
- **Retry logic** with exponential backoff
- **Health checks** for all components
- **Async/await** architecture for high concurrency
- **Configurable timeouts** and resource limits

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          WORKFLOW ORCHESTRATOR                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │  POLLER AGENT     │         │  MEMORY AGENT     │
          │  (Zendesk)        │────────▶│  (FAISS Vector)   │
          └───────────────────┘         └─────────┬─────────┘
                                                  │
                    Duplicate? ──────────────────┘
                         │ No
                    ┌────▼─────────────────────────────────┐
                    │       TRIAGE AGENT (CoT)             │
                    │  ┌─────────────────────────────────┐ │
                    │  │ 1. Entity Extraction (LLM)      │ │
                    │  │ 2. RAG KB Search (FAISS)        │ │
                    │  │ 3. Plan Creation (LLM)          │ │
                    │  │ 4. Parameter Mapping (Schema)   │ │
                    │  │ 5. Tool Execution (MCP)         │ │
                    │  │ 6. Synthesis (LLM)              │ │
                    │  └─────────────────────────────────┘ │
                    └────┬─────────────────────────────────┘
                         │ Analysis + Synthesis
                    ┌────▼─────────────────────────────────┐
                    │    SUPERVISOR AGENT                  │
                    │  - Reviews synthesis                 │
                    │  - Makes routing decision            │
                    │  - Determines escalation             │
                    └────┬─────────────────────────────────┘
                         │ Decision
                    ┌────▼─────────────────────────────────┐
                    │    EXECUTOR AGENT                    │
                    │  - Add comments                      │
                    │  - Assign to teams                   │
                    │  - Escalate to humans                │
                    │  - Update ticket status              │
                    └──────────────────────────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    ┌────▼────┐          ┌────▼────┐
                    │ Zendesk │          │LangFuse │
                    │   API   │          │ Traces  │
                    └─────────┘          └─────────┘
```

### Agent Responsibilities

| Agent | Type | Purpose | Key Actions |
|-------|------|---------|-------------|
| **Poller** | Data Ingestion | Monitor Zendesk queues | Poll tickets, filter by priority |
| **Memory** | Deduplication | Prevent duplicate processing | FAISS similarity search, flag duplicates |
| **Triage** | Analysis | CoT reasoning + tool execution | Entity extraction, RAG search, plan execution, synthesis |
| **Supervisor** | Decision | Final routing and escalation | Review synthesis, assign teams, escalate |
| **Executor** | Action | Modify tickets in Zendesk | Add comments, assign, escalate, tag |

---

## 🔧 Technical Components

### 1. **Core Framework** (`core/`)
- **`config.py`**: YAML-based configuration with env variable substitution
- **`gateway/`**: LLM client, MCP client, tool registry, parameter mapper
- **`memory/`**: FAISS vector store, Redis adapter (optional)
- **`observability/`**: LangFuse, LangSmith, console tracers
- **`graph/`**: LangGraph base agent, state management
- **`rag/`**: FAISS-based knowledge base indexing and search

### 2. **Agents** (`agents/`)
- **`poller/`**: Zendesk ticket polling with queue management
- **`memory/`**: Duplicate detection using vector similarity
- **`triage/`**: Main analysis agent with 6-step CoT reasoning
- **`supervisor/`**: Decision-making agent for routing/escalation
- **`executor/`**: Action execution agent with retry logic

### 3. **MCP Gateways** (`mcp_gateway/`)
- **Engineering tools**: Splunk, NewRelic, Prometheus
- **Pricing tools**: Price Advisory APIs (base prices, competitor prices, basket segments)
- **SharePoint tools**: Document search, file listing, downloads
- **Auto-discovery**: Gateways expose tool schemas via MCP protocol

### 4. **Knowledge Base** (`kb/`)
- Markdown files for runbooks and troubleshooting guides
- Automatically indexed by FAISS on startup
- Queried via semantic search during triage

### 5. **Configuration** (`config/`)
- **`agent.yaml`**: Main configuration for all agents, tools, and observability
- Supports environment variable substitution: `{VAR_NAME}`

---

## 🎭 Agentic Pattern

### Multi-Agent Workflow Pattern
TescoResolveX implements the **Supervisor Pattern** with specialized agents:

1. **Orchestration Layer**: `WorkflowOrchestrator` coordinates agent execution
2. **Agent Layer**: Each agent is a LangGraph state machine with defined transitions
3. **Tool Layer**: MCP gateways provide discoverable tools
4. **Observability Layer**: LangFuse captures all agent interactions

### Chain-of-Thought (CoT) Pattern
The Triage agent uses **multi-step reasoning** inspired by Claude/ChatGPT:

```python
# Step 1: Extract structured entities
entities = llm.extract({
    "gtin": "14-digit product code",
    "tpnb": "9-digit product code",
    "incident_type": "classification",
    "key_terms": ["relevant", "keywords"]
})

# Step 2: Retrieve relevant knowledge
kb_articles = rag.search(entities["key_terms"], k=1)

# Step 3: Create execution plan
plan = llm.create_plan({
    "incident_type": entities["incident_type"],
    "kb_guidance": kb_articles,
    "available_tools": tool_registry.list()
})

# Step 4: Map parameters automatically (deterministic)
payloads = parameter_mapper.map(plan["steps"], entities)

# Step 5: Execute tools
results = await execute_tools(plan["steps"], payloads)

# Step 6: Synthesize findings
synthesis = llm.synthesize({
    "ticket": ticket,
    "entities": entities,
    "tool_results": results,
    "kb_guidance": kb_articles
})
```

### Why This Pattern?

✅ **Prevents hallucination**: Schema-driven parameter mapping (Step 4)  
✅ **Leverages domain knowledge**: RAG retrieves runbooks (Step 2)  
✅ **Explainable**: Each step is traced independently  
✅ **Maintainable**: Adding tools doesn't require prompt changes  
✅ **Scalable**: Agents can run independently or in parallel  

---

## 🚀 Installation

See **[QUICK_INSTALL.md](QUICK_INSTALL.md)** for detailed setup instructions.

### Quick Start

```bash
# 1. Clone repository
git clone <repo-url>
cd TescoResolveX

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# 4. Start Ollama (LLM)
ollama serve

# 5. Start LangFuse (Observability)
cd langfuse && docker-compose up -d

# 6. Initialize knowledge base
python scripts/index_kb.py

# 7. Run full workflow test
python scripts/test_full_workflow.py
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# LLM Gateway
CENTRAL_LLM_GATEWAY_URL=http://localhost:11434

# Observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000

# Zendesk
ZENDESK_SUBDOMAIN=yourcompany
ZENDESK_SERVICE_EMAIL=support@yourcompany.com
ZENDESK_API_TOKEN=your_token

# MCP Gateways
MCP_ENGINEERING_URL=http://localhost:3001
MCP_PRICING_URL=http://localhost:3002
```

### Agent Configuration (`config/agent.yaml`)

```yaml
agents:
  triage:
    enabled: true
    model: "llama3.2"
    team: "operations"
    
  supervisor:
    enabled: true
    team: "operations"
    
  executor:
    enabled: true
    max_retries: 3
    retry_delay: 2.0
    team_mappings:
      engineering_team: "team_eng_001"
      pricing_team: "team_pricing_001"
```

---

## 🏃 Running the System

### 1. Start Infrastructure

```bash
# Terminal 1: Ollama LLM
ollama serve

# Terminal 2: LangFuse
cd langfuse && docker-compose up

# Terminal 3: MCP Gateways (optional for testing)
cd mcp_gateway && python engineering_gateway.py
```

### 2. Run Test Workflow

```bash
# Full end-to-end test (Poller → Memory → Triage → Supervisor → Executor)
python scripts/test_full_workflow.py

# View traces in LangFuse
open http://localhost:3000
```

### 3. Run Production Agents

```bash
# Start all agents
python scripts/start_agents.py

# Or run individual agents
python -m agents.poller.agent    # Polls tickets every 30 min
python -m agents.triage.agent    # Processes from queue
python -m agents.supervisor.agent
```

---

## 📁 Project Structure

```
TescoResolveX/
├── agents/                      # Agent implementations (LangGraph)
│   ├── poller/                  # Zendesk ticket polling
│   ├── memory/                  # FAISS duplicate detection
│   ├── triage/                  # CoT analysis + tool execution
│   ├── supervisor/              # Final decision maker
│   └── executor/                # Action executor (Zendesk API)
├── core/
│   ├── config/                  # Configuration management
│   ├── gateway/
│   │   ├── tool_registry.py     # Multi-gateway tool registry
│   │   ├── mcp_client.py        # MCP protocol client
│   │   ├── llm_client.py        # Ollama LLM client
│   │   └── parameter_mapper.py  # Schema-driven parameter mapping
│   ├── memory/                  # FAISS vector memory
│   ├── observability/           # LangFuse, LangSmith tracers
│   ├── rag/                     # FAISS KB indexing and search
│   └── graph/                   # LangGraph base agent
├── mcp_gateway/                 # MCP tool gateways
│   ├── engineering_gateway.py   # Splunk, NewRelic tools
│   ├── pricing_gateway.py       # Price Advisory APIs
│   └── sharepoint_gateway.py    # SharePoint tools
├── kb/                          # Knowledge base (runbooks)
│   ├── basket_segments_runbook.md
│   ├── price_missing_runbook.md
│   └── file_drop_runbook.md
├── prompts/                     # LLM prompts for agents
│   ├── triage/
│   │   ├── step1_entity_extraction.md
│   │   ├── step2_create_plan.md
│   │   └── step3_synthesize_results.md
│   └── executor/
│       └── comment_templates.md
├── config/
│   └── agent.yaml               # Main configuration
├── scripts/
│   ├── test_full_workflow.py    # End-to-end test
│   ├── start_agents.py          # Production agent launcher
│   └── index_kb.py              # Index knowledge base
├── data/
│   ├── faiss_index/             # FAISS memory indices
│   ├── faiss_kb_index/          # FAISS KB indices
│   └── mock_tickets/            # Test data
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── QUICK_INSTALL.md             # Installation guide
```

---

## 🔍 Observability

### LangFuse Dashboard

All agent executions are traced in **LangFuse** with full observability:

1. **Traces View**: See all workflow executions
2. **Agent Spans**: Nested spans for each agent
3. **LLM Calls**: Token usage, latency, input/output
4. **Tool Executions**: Which tools ran and their results
5. **Error Tracking**: Failed steps with stack traces

**Example Trace Hierarchy**:
```
workflow_execution (15.2s)
├── zendesk_poller_process (0.5s)
├── memory_agent_process (1.2s)
├── triage_process (9.7s)
│   ├── rag_knowledge_search (0.2s)
│   ├── cot_entity_extraction (2.6s)
│   │   └── llm_chat_completion (2.5s)
│   ├── cot_plan_creation (6.1s)
│   │   └── llm_chat_completion (6.0s)
│   ├── mcp_tool_splunk_search (0.3s)
│   ├── mcp_tool_base_prices_get (0.2s)
│   ├── mcp_tool_sharepoint_list_files (0.1s)
│   └── cot_synthesis (3.2s)
│       └── llm_chat_completion (3.1s)
├── supervisor_process (0.5s)
└── executor_process (1.3s)
    ├── _execute_action (0.8s)
    ├── _validate_execution (0.3s)
    └── _should_retry (0.2s)
```

Access: **http://localhost:3000**

---

## 🧪 Testing

### Run Full Workflow Test

```bash
python scripts/test_full_workflow.py
```

This will:
1. Poll mock tickets from `data/mock_tickets/`
2. Check for duplicates in vector memory
3. Run CoT analysis with tool execution
4. Make supervisor decision
5. Execute actions via executor
6. Send traces to LangFuse

### Check LangFuse

Open http://localhost:3000 and search for traces. You should see:
- `zendesk_poller_process`
- `memory_agent_process`
- `triage_process` (with nested CoT spans)
- `supervisor_process`
- `executor_process`

---

## 🛠️ Development

### Adding a New Tool

1. **Add tool to MCP gateway** (e.g., `mcp_gateway/engineering_gateway.py`):
```python
@app.post("/execute")
async def execute_tool(request: ToolRequest):
    if request.tool == "my_new_tool":
        return {"success": True, "data": ...}
```

2. **Add JSON schema**:
```python
TOOL_SCHEMAS["my_new_tool"] = {
    "name": "my_new_tool",
    "description": "What it does",
    "parameters": {
        "param1": {"type": "string", "required": True},
        "param2": {"type": "number", "required": False}
    }
}
```

3. **Update KB runbooks** to reference the tool:
```markdown
## Troubleshooting Steps
1. Check logs: Use `my_new_tool` with param1=...
```

That's it! The tool will automatically:
- Be discovered by the tool registry
- Appear in plan creation prompts
- Have parameters mapped automatically

### Adding a New Runbook

1. Create markdown file in `kb/`:
```bash
echo "# New Issue Type\n## Symptoms\n## Root Causes\n## Steps" > kb/new_issue.md
```

2. Re-index knowledge base:
```bash
python scripts/index_kb.py
```

The runbook will be searchable via RAG during triage.

---

## 📊 Key Metrics

Track these in LangFuse or logs:

- **Ticket Processing Time**: End-to-end latency per ticket
- **Tool Success Rate**: % of successful tool executions
- **LLM Token Usage**: Total tokens per analysis
- **Escalation Rate**: % of tickets escalated to humans
- **Duplicate Detection Rate**: % of duplicates caught
- **Agent Retry Rate**: How often executors retry

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and test: `python scripts/test_full_workflow.py`
4. Commit: `git commit -am 'Add my feature'`
5. Push: `git push origin feature/my-feature`
6. Create Pull Request

---

## 📝 License

[Your License Here]

---

## 🙋 Support

For issues or questions:
- Check **LangFuse traces** first: http://localhost:3000
- Review agent logs in console output
- Open an issue on GitHub

---

## 🎯 Roadmap

- [ ] Add support for Slack notifications
- [ ] Implement auto-resolution for common issues
- [ ] Add batch processing for high ticket volumes
- [ ] Integrate with Jira for advanced workflows
- [ ] Build self-healing capabilities (auto-retry failed actions)

---

**Built with ❤️ for intelligent support automation**
