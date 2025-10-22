# Golden Agent Framework (Minimal Support Agents)

Prompt-driven, central-gateway-aware agents for support workflows. Minimal, clean, and extensible with MCP or local tools.

## Key Capabilities

- Agents: Poller (Zendesk), Triage, Memory, Splunk, New Relic, Supervisor
- Tools via Centralized MCP Gateway or Local Python functions
- Observability: switchable LangSmith/LangFuse/Console via config
- Local KB RAG: drop ~20 Markdown files into `kb/` and search with embeddings
- Domain Glossary: YAML-driven entity hints (GTIN, TPNB, location clusters)
- Local Mocks for offline testing

## Documentation

- Architecture: `docs/architecture.md`
- Agents: `docs/agents.md`
- Demo Walkthrough: `docs/demo_walkthrough.md`
- Q&A / Design Patterns: `docs/qna_design.md`

## Quick Start (Local)

1) Install (uv)
```bash
# install uv if needed
curl -Ls https://astral.sh/uv/install.sh | sh

# create and activate venv (optional but recommended)
uv venv
source .venv/bin/activate

# install project deps from pyproject
uv pip install -e .
```

Alternative (pip):
```bash
pip install -r requirements.txt
```

2) Set env (example)
```bash
export ORG_NAME="YourOrganization"
export ENVIRONMENT="local"
export CENTRAL_MCP_GATEWAY_URL="http://localhost:8081"
export CENTRAL_LLM_GATEWAY_URL="http://localhost:8082"
export DEFAULT_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export LANGSMITH_API_KEY="..."   # if using LangSmith
# or LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY if using LangFuse
```

3) Start agents
```bash
python scripts/start_agents.py
```

## Configuration Highlights (`config/agent.yaml`)

### Gateways and Tools
Agents never import tools directly; they call by name via the ToolRegistry.
```yaml
gateway:
  mcp_gateway:
    url: "{CENTRAL_MCP_GATEWAY_URL}"
  llm_gateway:
    url: "{CENTRAL_LLM_GATEWAY_URL}"
  tools:
    # MCP tools (prod pattern)
    splunk_search:
      type: "mcp"
      server: "splunk-mcp-server"
    newrelic_metrics:
      type: "mcp"
      server: "newrelic-mcp-server"

    # Local team tools (in-process)
    product_lookup:
      type: "local"
      module: "core.tools.team_tools:product_lookup"
    location_lookup:
      type: "local"
      module: "core.tools.team_tools:location_lookup"

    # Optional local mocks (offline dev)
    splunk_search_mock:
      type: "local"
      module: "core.mocks.tools:splunk_search_mock"
    newrelic_metrics_mock:
      type: "local"
      module: "core.mocks.tools:newrelic_metrics_mock"
    zendesk_poll_queue_mock:
      type: "local"
      module: "core.mocks.tools:zendesk_poll_queue_mock"
```

To test offline, point canonical names to mocks in a local override (e.g., map `splunk_search` → `core.mocks.tools:splunk_search_mock`).

### Agents (minimal roles)
- Poller: polls Zendesk and forwards tickets; no decisions
- Triage: analyzes tickets, chooses tools (Splunk/NewRelic/Memory/APIs), forwards enriched data; no final actions
- Memory: checks duplicates; returns resolution/merge/escalation; stores new tickets; daily cache
- Splunk/New Relic: select query templates; optionally execute if configured or use mocks
- Supervisor: final decisions (comment/assign/escalate), may use RAG

### Observability
Switch backend via config.
```yaml
observability:
  backend: "langsmith"   # langsmith | langfuse | console
  langsmith:
    api_key: "{LANGSMITH_API_KEY}"
    project_name: "{ORG_NAME}-agents"
    environment: "{ENVIRONMENT}"
  langfuse:
    public_key: "{LANGFUSE_PUBLIC_KEY}"
    secret_key: "{LANGFUSE_SECRET_KEY}"
    host: "https://cloud.langfuse.com"
    project_name: "{ORG_NAME}-agents"
    environment: "{ENVIRONMENT}"
    enabled: false
```
Startup auto-configures observability: `scripts/start_agents.py`.

### Local KB RAG
Use in-repo Markdown knowledge base.
```yaml
rag:
  backend: "local_kb"
  local_kb:
    knowledge_dir: "kb"
    model_name: "all-MiniLM-L6-v2"
```
Drop articles in `kb/`; they’ll be indexed for semantic search (with keyword fallback).

### Domain Glossary
Provide business entities and patterns used by Triage.
```yaml
agents:
  triage:
    resources:
      knowledge_base:
        - type: "file"
          path: "resources/triage/incident_types.yaml"
        - type: "file"
          path: "resources/triage/glossary.yaml"
```
Glossary drives entity extraction (GTIN/TPNB/location clusters) and tool suggestions.

## Dev/Testing Aids

- Local mocks: `core/mocks/tools.py` (Splunk/New Relic/Zendesk).
- Team local tools: `core/tools/team_tools.py` (add functions, declare in `gateway.tools`).
- Optional local Zendesk MCP (dev-only): `extensions/mcp_servers/zendesk_tools.py` kept for testing, but agents should still call via ToolRegistry.

### Mock Tickets for Demos
Sample tickets for presentations are under `data/mock_tickets/`:
- `001_login_errors.json`
- `002_price_mismatch.json`
- `003_location_outage.json`

## Repository Structure (minimal)
```
agents/                # poller, triage, memory, splunk, newrelic, supervisor
config/                # agent + gateway + env
core/
  gateway/             # tool registry, mcp/llm clients
  graph/               # base, executor, state
  memory/              # faiss/redis/pinecone/mock
  observability/       # langsmith/langfuse/console
  prompts/             # loader/manager/template
  rag/                 # local_kb
  tools/               # std_tools, team_tools
  mocks/               # local tool mocks
extensions/
  mcp_servers/         # (dev) zendesk MCP tools
kb/                    # drop markdown KB docs here
prompts/               # per-agent system prompts
resources/triage/      # incident_types.yaml, glossary.yaml
scripts/start_agents.py
```

## Notes
- Keep agents thin; all external actions go through tools declared in config.
- Prefer central MCP for production; use local tools/mocks for dev.
- The Poller does not analyze; Triage does not finalize; Supervisor finalizes.
