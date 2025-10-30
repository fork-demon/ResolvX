# Quick Installation Guide

Complete setup guide for **TescoResolveX** - from zero to running the full multi-agent workflow in ~15 minutes.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.9+** installed
- **Git** for cloning the repository
- **Docker & Docker Compose** for LangFuse (observability)
- **Ollama** for local LLM inference
- **4GB+ RAM** available
- **macOS, Linux, or WSL2** (Windows users: use WSL2)

---

## 🚀 Installation Steps

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd TescoResolveX
```

---

### Step 2: Install Python Dependencies

#### Option A: Using pip

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
conda create -n resolvex python=3.9
conda activate resolvex
pip install -r requirements.txt
```

**Dependencies Installed:**
- `langgraph` - Agent workflow framework
- `langfuse` - Observability and tracing
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Embeddings for RAG
- `pydantic` - Data validation
- `httpx` - Async HTTP client
- `pyyaml` - Configuration parsing
- `fastapi` - For MCP gateways
- `uvicorn` - ASGI server

---

### Step 3: Install and Start Ollama (LLM)

**Ollama** provides local LLM inference (no API keys needed).

#### macOS/Linux

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model (llama3.2 - 3.2B parameters, fast)
ollama pull llama3.2

# Start Ollama server (runs in background)
ollama serve
```

#### Verify Ollama is running

```bash
curl http://localhost:11434/api/tags
```

Expected output: List of installed models including `llama3.2`

---

### Step 4: Install and Start LangFuse (Observability)

**LangFuse** provides full observability - trace every agent, LLM call, and tool execution.

#### Using Docker Compose (Recommended)

```bash
# Navigate to langfuse directory
cd langfuse

# Start LangFuse (Postgres + LangFuse server)
docker-compose up -d

# Wait for services to be ready (~30 seconds)
sleep 30

# Check if LangFuse is running
curl http://localhost:3000/api/public/health
```

Expected output: `{"status":"ok"}`

#### Access LangFuse Web UI

1. Open browser: **http://localhost:3000**
2. Sign up for a free account (local instance, no data leaves your machine)
3. Create a project (e.g., "TescoResolveX")
4. Copy **Public Key** and **Secret Key** from settings

---

### Step 5: Configure Environment Variables

Create `.env` file with your credentials:

```bash
# Copy template
cp .env.example .env

# Edit .env
nano .env  # or vim, code, etc.
```

**Required Environment Variables:**

```bash
# === LLM Gateway (Ollama) ===
CENTRAL_LLM_GATEWAY_URL=http://localhost:11434

# === LangFuse Observability ===
# Copy these from LangFuse UI (Settings → API Keys)
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_HOST=http://localhost:3000

# === Organization Settings ===
ORG_NAME=TescoResolveX
ENVIRONMENT=local

# === Embeddings Model ===
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# === Zendesk (Optional for testing) ===
# Use mock data if you don't have Zendesk credentials
ZENDESK_SUBDOMAIN=yourcompany
ZENDESK_SERVICE_EMAIL=support@yourcompany.com
ZENDESK_API_TOKEN=your_zendesk_api_token

# === MCP Gateways (Local for testing) ===
MCP_ENGINEERING_URL=http://localhost:3001
MCP_PRICING_URL=http://localhost:3002
MCP_SHAREPOINT_URL=http://localhost:3003

# === SharePoint (Optional) ===
SHAREPOINT_TENANT_ID=your-tenant-id
SHAREPOINT_CLIENT_ID=your-client-id
SHAREPOINT_CLIENT_SECRET=your-client-secret
SHAREPOINT_PROCESS_FOLDER=Shared Documents/Uploads
SHAREPOINT_ARCHIVE_FOLDER=Shared Documents/Archive
```

**Save and close the file.**

---

### Step 6: Initialize Knowledge Base

The knowledge base contains runbooks and troubleshooting guides. Index them for RAG search:

```bash
# Index all markdown files in kb/ folder
python scripts/index_kb.py
```

Expected output:
```
✅ Indexed 5 runbooks
✅ FAISS index saved to ./data/faiss_kb_index
```

**What this does:**
- Reads all `.md` files in `kb/` folder
- Generates embeddings using `all-MiniLM-L6-v2`
- Creates FAISS index for fast similarity search
- Saves index to `data/faiss_kb_index/`

---

### Step 7: Start MCP Gateways (Optional for Full Testing)

MCP gateways expose tools (Splunk, Price APIs, SharePoint) via the Model Context Protocol.

#### Terminal 1: Engineering Gateway (Splunk, NewRelic)

```bash
cd mcp_gateway
python engineering_gateway.py
```

Expected output:
```
INFO: Started server on http://0.0.0.0:3001
```

#### Terminal 2: Pricing Gateway (Price Advisory APIs)

```bash
cd mcp_gateway
python pricing_gateway.py
```

Expected output:
```
INFO: Started server on http://0.0.0.0:3002
```

#### Terminal 3: SharePoint Gateway

```bash
cd mcp_gateway
python sharepoint_gateway.py
```

Expected output:
```
INFO: Started server on http://0.0.0.0:3003
```

**Note**: For quick testing, you can skip starting gateways. The test script uses mock data by default.

---

### Step 8: Verify Configuration

Run the configuration check script:

```bash
python -c "
from core.config import load_config
config = load_config('config/agent.yaml')
print(f'✅ Config loaded: {len(config.agents)} agents configured')
print(f'✅ Observability: {config.observability.backend}')
print(f'✅ RAG backend: {config.rag.backend}')
"
```

Expected output:
```
✅ Config loaded: 5 agents configured
✅ Observability: langfuse
✅ RAG backend: faiss_kb
```

---

## 🏃 Running the System

### Option 1: Run Full Workflow Test (Recommended for First Run)

This runs the complete end-to-end workflow with mock data:

```bash
# Run full workflow test
python scripts/test_full_workflow.py
```

**What this does:**
1. **Polls** mock tickets from `data/mock_tickets/`
2. **Checks** for duplicates using FAISS memory
3. **Analyzes** tickets with Triage agent (6-step CoT)
   - Extracts entities (GTIN, TPNB, incident type)
   - Searches knowledge base for relevant runbooks
   - Creates execution plan with tool selection
   - Executes tools (splunk_search, base_prices_get, etc.)
   - Synthesizes findings into actionable summary
4. **Routes** with Supervisor agent
   - Reviews synthesis
   - Makes escalation decision
   - Assigns to appropriate team
5. **Executes** with Executor agent
   - Adds comment to ticket (mock)
   - Assigns to team (mock)
   - Escalates if needed (mock)
6. **Traces** everything to LangFuse

**Expected output:**

```
======================================================================
1️⃣  POLLER: Polling Zendesk
======================================================================
  ✓ Tool Registry: 13 tools
  ✓ Polled 1 tickets

======================================================================
2️⃣  MEMORY: Checking for duplicates
======================================================================
  ✓ Action: stored_current_ticket (0 related tickets)

======================================================================
3️⃣  TRIAGE: Analyzing ticket
======================================================================
  ✓ CoT Step 1: Entity Extraction
     - incident_type: price_not_found
     - TPNB: 12345678
     - Key terms: [basket, segment, price]
  
  ✓ CoT Step 2: RAG KB Search
     - Found: Basket Segments Runbook
  
  ✓ CoT Step 3: Execution Plan
     - 3 steps: splunk_search, base_prices_get, sharepoint_list_files
  
  ✓ CoT Step 4-5: Tool Execution
     - splunk_search: ✅ Found 10 results
     - base_prices_get: ✅ Price data retrieved
     - sharepoint_list_files: ✅ 3 files listed
  
  ✓ CoT Step 6: Synthesis
     - Root Cause: Timeout errors in file drop process
     - Confidence: medium
     - Recommended Actions: [Check network, verify file format, restart service]

======================================================================
4️⃣  SUPERVISOR: Making decision
======================================================================
  ✓ Decision: ESCALATE_TO_HUMAN
  ✓ Reason: Escalation recommended by analysis
  ✓ Assigned to: oncall_engineer

======================================================================
5️⃣  EXECUTOR: Executing decision
======================================================================
  ✓ Execution: SUCCESS
  ✓ Action: ESCALATE_TO_HUMAN
  ✓ Assigned to: user_oncall_001

======================================================================
✅ Full Workflow Test Complete!
======================================================================

💡 Check LangFuse at http://localhost:3000 for traces
```

---

### Option 2: View Traces in LangFuse

1. Open browser: **http://localhost:3000**
2. Navigate to **Traces**
3. Search for recent traces (last few minutes)
4. Click on a trace to see the full execution hierarchy:

```
workflow_execution (15.2s)
├── zendesk_poller_process (0.5s)
├── memory_agent_process (1.2s)
├── triage_process (9.7s)
│   ├── rag_knowledge_search (0.2s)
│   ├── cot_entity_extraction (2.6s)
│   │   └── llm_chat_completion (2.5s) ← View LLM input/output
│   ├── cot_plan_creation (6.1s)
│   │   └── llm_chat_completion (6.0s) ← View execution plan
│   ├── mcp_tool_splunk_search (0.3s) ← View tool results
│   ├── mcp_tool_base_prices_get (0.2s)
│   ├── mcp_tool_sharepoint_list_files (0.1s)
│   └── cot_synthesis (3.2s)
│       └── llm_chat_completion (3.1s) ← View synthesis
├── supervisor_process (0.5s)
└── executor_process (1.3s)
```

**Click on any span to see:**
- Input data
- Output data
- Timing (latency)
- Metadata (model, temperature, tokens)
- Errors (if any)

---

### Option 3: Run Individual Agents (Production Mode)

For production deployment, run agents as separate processes:

#### Start All Agents

```bash
# Start all agents in separate terminals
python scripts/start_agents.py
```

This starts:
- **Poller Agent**: Polls Zendesk every 30 minutes
- **Memory Agent**: Listens for duplicate checks
- **Triage Agent**: Processes tickets from queue
- **Supervisor Agent**: Makes routing decisions
- **Executor Agent**: Executes actions on tickets

#### Or Start Individually

```bash
# Terminal 1: Poller (runs on schedule)
python -m agents.poller.agent

# Terminal 2: Triage
python -m agents.triage.agent

# Terminal 3: Supervisor
python -m agents.supervisor.agent

# Terminal 4: Executor
python -m agents.executor.agent
```

---

## 🔧 Configuration Customization

### Update Team Mappings

Edit `config/agent.yaml` to add your Zendesk team IDs:

```yaml
agents:
  executor:
    team_mappings:
      engineering_team: "team_eng_001"        # Replace with your team ID
      devops_team: "team_devops_001"
      security_team: "team_sec_001"
      pricing_team: "team_pricing_001"
      operations_team: "team_ops_001"
      oncall_engineer: "user_oncall_001"      # Replace with user ID
```

### Update Comment Templates

Customize comment templates in `prompts/executor/comment_templates.md`:

```markdown
## Template: Root Cause Analysis

**Root Cause Identified:**
{{root_cause}}

**Impact:**
{{impact}}

**Recommended Actions:**
{{actions}}

---
🤖 Automated by TescoResolveX
```

### Add Custom Runbooks

Add new troubleshooting guides to `kb/`:

```bash
# Create new runbook
cat > kb/my_issue_runbook.md << 'EOF'
# My Issue Type

## Symptoms
- Symptom 1
- Symptom 2

## Root Causes
- Cause 1
- Cause 2

## Troubleshooting Steps
1. Check logs: Use `splunk_search` with query="my_issue"
2. Verify data: Use `base_prices_get` to check pricing data
3. Escalate if unresolved

## Resolution
- Action to take

EOF

# Re-index knowledge base
python scripts/index_kb.py
```

---

## 🧪 Testing

### Run Tests

```bash
# Full workflow test
python scripts/test_full_workflow.py

# Test with different ticket
export TEST_TICKET_ID=ALERT-002
python scripts/test_full_workflow.py
```

### Test Individual Components

```bash
# Test RAG search
python -c "
from core.rag.faiss_kb import FaissKnowledgeBase
kb = FaissKnowledgeBase({'index_path': './data/faiss_kb_index'})
results = kb.search('basket segment issue', k=1)
print(results)
"

# Test LLM client
python -c "
from core.gateway.llm_client import LLMGatewayClient, ChatMessage
import asyncio
client = LLMGatewayClient()
asyncio.run(client.initialize())
response = asyncio.run(client.chat_completion(
    messages=[ChatMessage(role='user', content='Hello')],
    model='llama3.2'
))
print(response)
"
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'core'"

**Solution:**
```bash
# Ensure you're in the project root
cd TescoResolveX

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

---

### Issue: "Connection refused to localhost:11434"

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

---

### Issue: "LangFuse not initialized"

**Solution:**
```bash
# Check if LangFuse is running
docker ps | grep langfuse

# If not, start it
cd langfuse && docker-compose up -d

# Check logs
docker-compose logs -f langfuse

# Verify keys in .env match LangFuse UI
grep LANGFUSE .env
```

---

### Issue: "FAISS index not found"

**Solution:**
```bash
# Index knowledge base
python scripts/index_kb.py

# Verify index exists
ls -lh data/faiss_kb_index/
```

---

### Issue: "Tool execution failed"

**Solution:**
```bash
# Check if MCP gateways are running
curl http://localhost:3001/health  # Engineering
curl http://localhost:3002/health  # Pricing
curl http://localhost:3003/health  # SharePoint

# Start missing gateways
cd mcp_gateway
python engineering_gateway.py &
python pricing_gateway.py &
python sharepoint_gateway.py &
```

---

### Issue: "Synthesis LLM call fails with 400"

**Solution:**
This was caused by passing URL as model name. **Already fixed** in the code.

Verify fix:
```bash
grep 'model="llama3.2"' agents/triage/agent.py
```

Expected: `model="llama3.2",  # Fixed: use model name, not URL`

---

## 📊 Monitoring

### Check Agent Health

```bash
# Check if agents are running
ps aux | grep "agents\."

# Check logs (if using systemd)
journalctl -u resolvex-poller -f
journalctl -u resolvex-triage -f
```

### View Metrics in LangFuse

1. Open **http://localhost:3000**
2. Navigate to **Metrics**
3. View:
   - Total traces
   - Average latency
   - Token usage
   - Success rate
   - Error rate

---

## 🚀 Next Steps

1. ✅ **Test with Real Tickets**: Update Zendesk credentials in `.env`
2. ✅ **Add Custom Tools**: Create new MCP gateway tools
3. ✅ **Customize Runbooks**: Add company-specific troubleshooting guides
4. ✅ **Deploy to Production**: Set up systemd services or Kubernetes
5. ✅ **Set Up Alerts**: Configure LangFuse alerts for errors
6. ✅ **Scale**: Run multiple agent instances behind a load balancer

---

## 📚 Additional Resources

- **README.md**: Detailed architecture and technical components
- **LangFuse Docs**: https://langfuse.com/docs
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Ollama Docs**: https://ollama.com/docs
- **MCP Protocol**: https://modelcontextprotocol.io

---

## ✅ Installation Complete!

You should now have:
- ✅ Python environment with all dependencies
- ✅ Ollama running with llama3.2 model
- ✅ LangFuse observability platform
- ✅ Knowledge base indexed
- ✅ Configuration set up
- ✅ Full workflow tested

**View your first traces at http://localhost:3000** 🎉

---

**Questions or issues? Check the troubleshooting section above or review LangFuse traces for detailed debugging.**
