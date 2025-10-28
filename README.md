# Support Template - Intelligent Agent Framework

A production-ready, multi-agent framework for pricing and competitive intelligence support operations with RAG, LLM integration, and multi-gateway tool orchestration.

---

## 🎯 What It Does

### Agents & Their Roles

1. **Poller Agent** - Polls Zendesk queues, forwards tickets (no analysis)
2. **Memory Agent** - Checks for duplicate tickets, returns resolution/merge/escalate
3. **Triage Agent** - Analyzes tickets using:
   - Historical runbook guidance (RAG)
   - Domain glossary (GTIN, TPNB, location clusters)
   - LLM reasoning (Ollama llama3.2)
   - Diagnostic tools (Splunk, Price API, SharePoint)
4. **Supervisor Agent** - Makes final decisions (comment/assign/escalate/human-in-loop)

### Key Features

- ✅ **Multi-Gateway Architecture**: Central (Zendesk, Splunk, NewRelic, Memory) + Price Team (Price API, Product, Location)
- ✅ **RAG Knowledge Base**: Semantic search over markdown runbooks (`kb/`)
- ✅ **Domain-Aware**: Extracts pricing entities, matches incident types, recommends tools
- ✅ **Prompt-Driven**: Configurable prompts using Jinja2 templates
- ✅ **Evaluation & Guardrails**: Quality assessment, hallucination detection, content filtering
- ✅ **Full Observability**: LangFuse tracing with input/output capture
- ✅ **ReAct Pattern**: Reasoning-Action-Observation loop for decision-making

---

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```bash
# Core Settings
ORG_NAME="Price Intelligence Team"
ENVIRONMENT="local"

# Gateways
CENTRAL_MCP_GATEWAY_URL="http://localhost:8083"
CENTRAL_LLM_GATEWAY_URL="http://localhost:11434"

# Observability (LangFuse)
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="http://localhost:3000"

# Mock Data
MOCK_TICKETS_DIR="/path/to/TescoResolveX/data/mock_tickets"

# SharePoint (for file processing verification)
SHAREPOINT_SITE_URL="https://your-tenant.sharepoint.com/sites/YourSite"
SHAREPOINT_PROCESS_FOLDER="/Shared Documents/CSV_Uploads/Process"
SHAREPOINT_ARCHIVE_FOLDER="/Shared Documents/CSV_Uploads/Archive"
```

### 3. Start Mock Services

**Terminal 1 - Central Gateway (Zendesk, Splunk, NewRelic, Memory):**
```bash
cd mock_services/central_gateway
python app.py
# Runs on http://localhost:8083
```

**Terminal 2 - Price Gateway (Price API, Product, Location, Competitor):**
```bash
cd mock_services/price_gateway
bash start.sh
# Runs on http://localhost:8082 (gateway) + http://localhost:8090 (backend)
```

**Terminal 3 - LLM (Ollama):**
```bash
ollama serve
# Runs on http://localhost:11434
# Load model: ollama pull llama3.2
```

**Terminal 4 - LangFuse (Observability):**
```bash
docker compose -f docker-compose.langfuse.yml up -d
# Access at http://localhost:3000
```

### 4. Run Agents

**Start all agents:**
```bash
python scripts/start_agents.py
```

**Run test workflow:**
```bash
python scripts/test_full_workflow.py
```

---

## 📁 Repository Structure

```
TescoResolveX/
├── agents/                     # Agent implementations
│   ├── poller/                 # Zendesk ticket poller
│   ├── triage/                 # Incident analysis & tool orchestration
│   ├── memory/                 # Duplicate detection (FAISS)
│   └── supervisor/             # Final decision maker
│
├── config/
│   ├── agent.yaml              # Main configuration (agents, tools, prompts)
│   └── environments/           # Environment-specific overrides
│
├── core/
│   ├── gateway/                # Tool registry, MCP clients, LLM client
│   ├── graph/                  # LangGraph base classes
│   ├── memory/                 # FAISS vector memory
│   ├── observability/          # LangFuse tracer, metrics
│   ├── rag/                    # Knowledge base (FAISS, Local KB)
│   ├── domain/                 # Glossary & incident type loaders
│   ├── prompts/                # Jinja2 prompt loader
│   ├── tools/                  # SharePoint tools
│   └── evaluation/             # Guardrails, hallucination checker, quality assessor
│
├── prompts/                    # Agent prompt templates (Jinja2)
│   ├── triage/
│   │   ├── system_prompt.md
│   │   ├── incident_analysis.md
│   │   └── routing_decision.md
│   └── supervisor/
│       └── system_prompt.md
│
├── kb/                         # Knowledge base (Markdown runbooks)
│   ├── domain/                 # Glossary & incident types
│   │   ├── glossary.md
│   │   └── incident_types.md
│   └── runbooks/               # Resolution procedures
│
├── data/
│   ├── mock_tickets/           # Test tickets (ALERT-001 to ALERT-005)
│   ├── faiss_memory_index/     # FAISS persistent index
│   └── faiss_kb_index/         # Knowledge base index
│
├── mock_services/              # Local mock gateways
│   ├── central_gateway/        # MCP gateway (port 8083)
│   │   ├── app.py
│   │   └── server.py
│   └── price_gateway/          # Price team MCP gateway (port 8082)
│       ├── app.py
│       ├── server.py
│       ├── start.sh
│       └── backend/            # Internal price API (port 8090)
│           └── app.py
│
├── scripts/
│   ├── start_agents.py         # Start all agents
│   └── test_full_workflow.py   # End-to-end workflow test
│
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧪 Testing & Demo

### Mock Tickets

Sample tickets in `data/mock_tickets/`:
- `ALERT-001`: Basket segments - File drop process failed
- `ALERT-002`: Competitor promotional price - File drop process failed
- `ALERT-003`: Investigate unexpected price move
- `ALERT-004`: Price-Cmd-Api Memory Usage too high
- `ALERT-005`: Price and promotions: Scanning at wrong price

### Test Complete Flow

```bash
python scripts/test_full_workflow.py
```

**What it does**:
1. Poller fetches tickets from Central Gateway
2. Memory checks for duplicates (FAISS similarity search)
3. Triage:
   - Searches knowledge base (RAG) for similar incidents
   - Sends incident + runbook context to LLM (Ollama llama3.2)
   - LLM recommends diagnostic tools
   - Executes tools (splunk_search, base_prices_get, sharepoint_list_files)
   - Evaluates response (guardrails, hallucination check, quality assessment)
4. Supervisor makes final decision (ADD_COMMENT/ASSIGN/ESCALATE)

**Expected output**:
```
✅ Poller: 0 tickets polled
✅ Memory: stored_current_ticket (0 related)
✅ Triage: Tools used: splunk_search, base_prices_get
✅ Supervisor: ADD_COMMENT
```

### Verify in LangFuse

1. Open http://localhost:3000
2. Look for traces:
   - `zendesk_poller_process`
   - `memory_agent_process`
   - `triage_process` (with RAG + LLM + tools)
   - `llm_chat_completion` (30s duration)
   - `mcp_tool_splunk_search`
   - `mcp_tool_base_prices_get`
   - `supervisor_process`

---

## ⚙️ Configuration

### Agent Configuration (`config/agent.yaml`)

**Multi-Gateway Setup:**
```yaml
gateway:
  # Central Gateway (Operational tools)
  mcp_gateway:
    url: "http://localhost:8083"
  
  # Additional gateways
  additional_mcp_gateways:
    pricing_team:
      url: "http://localhost:8082"
      enabled: true
  
  # LLM Gateway (Ollama)
  llm_gateway:
    url: "http://localhost:11434"
    default_model: "llama3.2"
```

**Agent Prompts:**
```yaml
agents:
  triage:
    prompts:
      system_prompt: "prompts/triage/system_prompt.md"
      incident_analysis: "prompts/triage/incident_analysis.md"
      routing_decision: "prompts/triage/routing_decision.md"
    resources:
      knowledge_base:
        - kb/domain/glossary.md
        - kb/domain/incident_types.md
```

**RAG Configuration:**
```yaml
rag:
  backend: "faiss_kb"  # Recommended (persistent, fast)
  knowledge_dir: "kb"
  model_name: "all-MiniLM-L6-v2"
  config:
    index_path: "./data/faiss_kb_index"
    dimension: 384
```

**Memory Configuration:**
```yaml
memory:
  backend: "faiss"
  faiss:
    index_path: "./data/faiss_memory_index"
    dimension: 384
    similarity_threshold: 0.7  # 70% for duplicate detection
```

---

## 🏗️ Architecture Highlights

### Multi-Gateway Tool Routing

- **Tool Registry** discovers tools from multiple MCP gateways
- **Dynamic Routing**: Tools routed to correct gateway based on source
- **Duplicate Handling**: First registered gateway wins
- **Example**:
  ```
  poll_queue        → Central Gateway (8083)
  splunk_search     → Central Gateway (8083)
  base_prices_get   → Price Gateway (8082)
  product_info_get  → Price Gateway (8082)
  ```

### RAG Pipeline

1. **Ingestion**: Markdown files in `kb/` indexed with SentenceTransformer
2. **Search**: Semantic search using FAISS (cosine similarity)
3. **Context Building**: Top 2 results (2000 chars each) → LLM prompt
4. **Prompt Structure**:
   ```
   ## Incident Details
   ...
   
   ## 📚 Historical Context & Runbook Guidance
   [3488 chars of runbook content]
   
   ## Your Task
   Use historical context to determine severity and recommend tools
   ```

### Evaluation Pipeline

1. **Guardrails**: Safety checks, content filtering, policy enforcement
2. **Hallucination Checker**: Detects factual errors, fabricated citations, speculative claims
3. **Quality Assessor**: Evaluates completeness, relevance, clarity, actionability, technical accuracy

---

## 🔧 Development

### Adding New Tools

**Local Tool (SharePoint example):**
```python
# core/tools/sharepoint_tool.py
class SharePointListFilesTool(BaseTool):
    name = "sharepoint_list_files"
    description = "List files in SharePoint folder"
    
    def execute(self, folder: str = "process") -> Dict[str, Any]:
        # Implementation
```

Register in `core/gateway/tool_registry.py`:
```python
def _load_local_tools(self):
    from core.tools.sharepoint_tool import SharePointListFilesTool
    self.register_tool(SharePointListFilesTool())
```

**MCP Tool:**
Add to `mock_services/central_gateway/server.py`:
```python
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "your_new_tool":
        # Implementation
```

### Adding New Agents

1. Create `agents/your_agent/agent.py`
2. Extend `BaseAgent` from `core/graph/base.py`
3. Implement `process()` method
4. Add configuration to `config/agent.yaml`
5. Add prompts to `prompts/your_agent/`

---

## 📊 Observability

### LangFuse Tracing

All agents automatically traced with:
- **Input**: Ticket data, context, parameters
- **Output**: Decisions, tool results, reasoning
- **Attributes**: Agent name, version, tools used, duration
- **Metadata**: Ticket ID, severity, confidence, entities

### Metrics

- Agent execution time
- Tool call latency
- LLM token usage
- RAG search performance
- Memory operations

---

## 🔐 Security & Best Practices

- ✅ Environment variables for sensitive data (never hardcode)
- ✅ Evaluation pipeline for LLM outputs
- ✅ Circuit breakers for external services
- ✅ Retry logic with exponential backoff
- ✅ FAISS in-memory vector store (no external DB needed)
- ✅ Local-first development (all services mockable)

---

## 🆘 Troubleshooting

### Issue: LLM not responding
**Fix**: Ensure Ollama is running and model is loaded
```bash
ollama serve
ollama pull llama3.2
```

### Issue: Traces not appearing in LangFuse
**Fix**: Check LangFuse connection and credentials
```bash
curl http://localhost:3000/api/public/health
```

### Issue: Mock tickets not loading
**Fix**: Set `MOCK_TICKETS_DIR` environment variable
```bash
export MOCK_TICKETS_DIR=/absolute/path/to/data/mock_tickets
```

### Issue: Tools not found
**Fix**: Verify gateways are running
```bash
# Central Gateway
curl http://localhost:8083/health

# Price Gateway
curl http://localhost:8082/health
```

---

## 📝 License

Proprietary - Price Intelligence Team

---

## 🤝 Contributing

For questions or improvements, contact the Price Intelligence Team.
