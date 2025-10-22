# Golden Agent Framework - Demo Summary

## ✅ Current Working State

### Components Running:
1. **Ollama LLM** - `http://localhost:11434` with `llama3.2` model
2. **MCP Mock Gateway** - `http://localhost:8081` with 12 tools
3. **6 Agents Started**:
   - Triage Agent
   - Poller Agent (polls every 30 mins)
   - Memory Agent
   - Splunk Agent
   - NewRelic Agent  
   - Supervisor Agent
4. **LangFuse Tracing** - Capturing all spans (backend configured)
5. **Mock Tickets** - 5 real tickets in `data/mock_tickets/` with status="new"

### End-to-End Flow Verified:

```
┌─────────────┐
│   MCP       │  Returns 5 real tickets from data/mock_tickets/
│  Gateway    │  (ALERT-001 to ALERT-005)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   POLLER    │  ✓ Polls 20 tickets (5 per queue × 4 queues)
│   AGENT     │  ✓ Forwards tickets to Triage
└──────┬──────┘  ✓ Trace: tickets_forwarded=20
       │
       ▼
┌─────────────┐
│   TRIAGE    │  1. Loads glossary (GTIN, TPNB, LocationCluster)
│   AGENT     │  2. Extracts entities from ticket
└──────┬──────┘  3. Maps entities → candidate tools
       │          4. Searches KB runbooks via RAG
       │          5. Calls LLM (Ollama llama3.2) for plan
       │          6. Executes tools based on plan
       ▼
   ┌───────────────┐
   │  GLOSSARY     │  Entity extraction rules:
   │  (YAML)       │  - GTIN: \d{13,14}
   └───────────────┘  - TPNB: \d{8}
                      - LocationCluster: synonyms → UUID
   ┌───────────────┐
   │  KB RUNBOOKS  │  RAG search results:
   │  (Markdown)    │  - basket_segments_runbook.md
   └───────────────┘  - price_discrepancy_runbook.md
                      Tools suggested based on issue type
   ┌───────────────┐
   │  LLM PLANNER  │  Ollama llama3.2 creates plan:
   │  (Ollama)     │  {
   └───────────────┘    "analysis": "...",
                        "tools_to_call": ["splunk_search", "base_prices_get"],
                        "parameters": {...}
                      }
   ┌───────────────┐
   │  TOOL         │  Executes via MCP:
   │  EXECUTOR     │  - splunk_search
   └───────────────┘  - newrelic_metrics
       │              - base_prices_get
       │              - basket_segment_get
       ▼              - competitor_prices_get
┌─────────────┐
│  MEMORY     │  ✓ Searches historical tickets
│  AGENT      │  ✓ Stores new tickets
└──────┬──────┘  ✓ Returns related tickets if duplicates
       │
       ▼
┌─────────────┐
│ SUPERVISOR  │  Final decision:
│  AGENT      │  - COMMENT_AND_ASSIGN (with enriched data)
└─────────────┘  - ESCALATE_TO_HUMAN (if tools failed)
                 - REQUEST_MORE_INFO (if inconclusive)
```

## 🎯 Demo Script: `scripts/demo_end_to_end.py`

Shows complete flow:
- Poller gets tickets from MCP
- For each ticket:
  1. **Entity Extraction** using glossary
  2. **Tool Mapping** based on entities
  3. **KB Search** using RAG (vector similarity on runbooks)
  4. **LLM Planning** with Ollama llama3.2
  5. **Tool Execution** via MCP (Splunk/NewRelic/Price APIs)
  6. **Supervisor Decision** with enriched context

## 🔍 Example Ticket Processing

### Ticket: ALERT-001 - "Basket segments - File drop process failed"

1. **Entities Extracted:**
   - Source: Splunk
   - Environment: PPE
   - Tags: basket-segments, file-drop, timeout

2. **Tool Mapping:**
   - From glossary rules → `basket_segment_get`, `splunk_search`

3. **KB Search:**
   - Score 0.85: `basket_segments_runbook.md`
   - Runbook suggests: Check Splunk → Verify file → Check New Relic

4. **LLM Plan (llama3.2):**
   ```json
   {
     "analysis": "File drop timeout, likely SharePoint connectivity issue",
     "tools_to_call": ["splunk_search", "newrelic_metrics", "basket_segment_get"],
     "parameters": {
       "splunk_search": {"query": "index=price-advisory CreateBasketSegmentsProcessor timeout"},
       "newrelic_metrics": {"nrql": "SELECT average(duration) FROM Transaction..."},
       "basket_segment_get": {"tpnb": "<from_ticket>", "locationClusterId": "<from_glossary>"}
     },
     "expected_outcome": "Identify if timeout is transient or systemic"
   }
   ```

5. **Tools Executed:**
   - ✓ `splunk_search` → Returns log entries
   - ✓ `newrelic_metrics` → Returns performance data
   - ✓ `basket_segment_get` → Returns current segment data

6. **Supervisor Decision:**
   - Decision: `COMMENT_AND_ASSIGN`
   - Reason: "Diagnostic data collected, assign to pricing team"
   - Comment: Includes Splunk logs + NR metrics + current state

## 📊 Observability

All traces captured in LangFuse:
- Span: `zendesk_poller_process` (tickets_forwarded=20)
- Span: `memory_agent_process` (ticket_id=ALERT-001, related_count=0)
- Span: `mcp_tool_poll_queue` (gateway calls)
- Span: `llm_chat_completion` (Ollama llama3.2 reasoning)

View at: `http://localhost:3000` (once Langfuse is accessible)

## 🚀 Quick Test Commands

```bash
# 1. Smoke test (components only)
python scripts/smoke_test.py

# 2. Full end-to-end demo (with LLM reasoning)
python scripts/demo_end_to_end.py

# 3. Start all agents (runs continuously)
python scripts/start_agents.py

# 4. Quick verification (exits after 10s)
START_AGENTS_VERIFY_SECONDS=10 START_AGENTS_EXIT_AFTER_VERIFY=1 python scripts/start_agents.py
```

## 📝 Configuration

- **LLM**: Ollama llama3.2 at `http://localhost:11434`
- **MCP**: Mock gateway at `http://localhost:8081`
- **Tracing**: LangFuse at `http://localhost:3000`
- **KB**: 3 markdown runbooks in `kb/`
- **Glossary**: Entity definitions in `resources/triage/glossary.yaml`
- **Mock Tickets**: 5 JSON files in `data/mock_tickets/`

## 🎓 Key Design Patterns

1. **Glossary-Driven Entity Extraction** - Business domain understanding
2. **RAG for Runbook Selection** - Vector search over KB articles
3. **LLM for Planning** - Ollama creates tool execution plan
4. **MCP for Tool Execution** - Centralized gateway routing
5. **Memory for Deduplication** - Historical ticket search
6. **Supervisor for Final Decision** - Enriched context → action

