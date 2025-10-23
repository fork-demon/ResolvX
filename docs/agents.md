## Agents and Responsibilities

Each agent is prompt-driven and single-purpose. Agents rely on the ToolRegistry and RAG; they do not call external SDKs directly.

### Poller (Zendesk)
- Purpose: Poll queues and forward tickets without analysis.
- Entry/Exit: Input none (scheduled); Output raw tickets to Triage.
- Key Config: `schedule`, `zendesk.auth`, `zendesk.queues[*]`.
- Prompt: `prompts/poller/system_prompt.md` (minimal, poll-and-forward).

### Triage
- Purpose: Analyze tickets, extract entities, select tools, enrich context.
- Tools: Splunk/New Relic/Memory/Product/Location via `gateway.tools`.
- Output: Enriched payload to Supervisor; no final decisions.
- Prompt: `prompts/triage/system_prompt.md` (uses glossary and mapping to tools).
- Resources: `resources/triage/incident_types.yaml`, `resources/triage/glossary.yaml`.
- **Evaluation**: Integrated guardrails, hallucination detection, and quality assessment.

### Memory
- Purpose: De-duplicate and contextualize with historical tickets.
- Behavior: If duplicate resolved → return resolution; in-progress → merge; closed-unresolved → escalate; else store.
- Backend: FAISS vector search with SentenceTransformer embeddings.
- Config: `memory.faiss` (auto-detects embedding dimensions, persistent storage).
- Prompt: `prompts/memory/system_prompt.md`.

### Splunk & New Relic (Tools, not Agents)
- Purpose: Now implemented as MCP tools, not standalone agents.
- Usage: Called by Triage agent via LLM-driven tool selection.
- Tools: `splunk_search`, `newrelic_metrics` via `gateway.tools`.
- Integration: Triage agent uses RAG + glossary to select appropriate tools.

### Supervisor
- Purpose: Final decision-maker (comment/assign/escalate; human-in-the-loop).
- Can query Global RAG for KB context.
- Prompt: `prompts/supervisor/system_prompt.md`.

## Tooling Layer and RAG
- ToolRegistry: resolves tool names to MCP servers or local functions (`gateway.tools`).
- Global RAG: `core/rag/local_kb.py` over `kb/` markdown.
- Team Tools: `core/tools/team_tools.py` for product/location/entity utilities.

## Observability
- Configure LangSmith or LangFuse in `observability` settings.
- Startup hook: `scripts/start_agents.py` configures tracing/metrics automatically.


