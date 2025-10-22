## Demo Walkthrough (Local, with Tracing)

This walkthrough shows an end-to-end flow using local mocks and optional tracing.

### 1) Install and Setup
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e .

export ORG_NAME="DemoOrg"
export ENVIRONMENT="local"
export CENTRAL_MCP_GATEWAY_URL="http://localhost:8081"   # if using MCP
export CENTRAL_LLM_GATEWAY_URL="http://localhost:8082"
export DEFAULT_EMBEDDING_MODEL="all-MiniLM-L6-v2"
# Choose ONE tracing backend
export LANGSMITH_API_KEY="..."            # if using LangSmith
# OR
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
```

### 2) Configure Tools to Use Local Mocks
Update your environment override (e.g., `config/environments/local.yaml`) to map canonical tool names to local mocks:
```yaml
gateway:
  tools:
    splunk_search:
      type: "local"
      module: "core.mocks.tools:splunk_search_mock"
    newrelic_metrics:
      type: "local"
      module: "core.mocks.tools:newrelic_metrics_mock"
    zendesk_poll_queue:
      type: "local"
      module: "core.mocks.tools:zendesk_poll_queue_mock"
```

### 3) (Optional) Add KB Articles and Glossary
- Drop ~20 markdown files under `kb/` to power Global RAG.
- Ensure `resources/triage/glossary.yaml` contains your business entities.

### 4) Start Agents
```bash
python scripts/start_agents.py
```
Expected:
- Poller runs on its schedule (or fallback interval), fetches mock Zendesk tickets.
- Triage receives tickets, extracts entities using the glossary, selects tools.
- Splunk/New Relic tools return structured mock results.
- Memory agent finds related tickets (or stores new ones) using in-memory vectors.
- Supervisor receives enriched payload and logs final decisions.

### 5) Tracing and Observability
- LangSmith: verify runs in the project named `{ORG_NAME}-agents`.
- LangFuse: verify traces/events under the configured project/environment.
- Console: logs include agent start, tool calls, and data flow.

### 6) Inspecting Outputs
- Agent logs: stdout from `scripts/start_agents.py`.
- Tool returns: included in logs; adjust logging level for details.
- Memory behavior: confirm daily reset logic and similarity search.

### 7) Mock Ticket Examples
Sample inputs for local storytelling (not required by the mocks to run):
- `data/mock_tickets/001_login_errors.json`: surge of 401/403 in `app:auth`.
- `data/mock_tickets/002_price_mismatch.json`: GTIN/TPNB price discrepancy.
- `data/mock_tickets/003_location_outage.json`: location cluster partial outage.

Use these to frame the narrative when presenting the flow (Poller → Triage → Tools → Supervisor) and how glossary+RAG influence tool selection.


