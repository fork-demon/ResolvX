## Architecture Overview

This repository implements a minimal, prompt-driven, tool-centric support agent system. Agents are thin, with clear single responsibilities, and rely on a Tooling Layer (via a gateway/registry) and RAG services for context.

### Core Roles and Flow
- Poller → Triage → (Splunk/New Relic/Memory/APIs) → Supervisor
- Only the Supervisor finalizes actions.

### Components
- Agents
  - Poller: polls Zendesk and forwards tickets.
  - Triage: analyzes tickets, chooses tools, enriches context, forwards.
  - Splunk/New Relic: select and run query templates; return structured results.
  - Memory: deduplicates tickets, returns related context, stores new items.
  - Supervisor: final decision-maker; can query Global RAG.
- Tooling Layer (Gateway)
  - Central MCP tools (production) or Local Python tools (development).
  - Declared in `config/agent.yaml` under `gateway.tools`.
- RAG
  - Global RAG: `core/rag/local_kb.py` over `kb/` markdown; embedding + keyword fallback.
  - Memory agent: in-memory vector DB with daily reset cache for historical tickets.
- Observability
  - Configurable: LangSmith, LangFuse, or Console via `observability` in config.

### Design Principles
- Prompt-driven behavior per agent type (system/runtime prompts).
- ToolProxy via `ToolRegistry`: agents never import tools directly.
- Gateway pattern for LLM/MCP endpoints.
- Separation of concerns: analysis (Triage) vs. actuation (Supervisor).
- Extensibility: drop-in team tools and KB docs without code changes.

### Data Flow (E2E)
1. Poller reads Zendesk (or mock) for new tickets; forwards raw tickets.
2. Triage parses entities (uses `resources/triage/glossary.yaml`), selects tools:
   - Splunk: choose SPL template, execute (or mock), return structured logs.
   - New Relic: choose NRQL template, execute (or mock), return metrics.
   - Memory: search related tickets; return resolution/merge/escalation guidance.
   - Product/Location APIs: enrich with domain data via team tools.
3. Triage forwards enriched payload to Supervisor.
4. Supervisor uses Global RAG if needed and finalizes: comment/assign/escalate.

### Configuration Surfaces
- `config/agent.yaml` (and env overrides): agents, tools, observability, RAG.
- `prompts/<agent>/system_prompt.md`: role instructions.
- `kb/`: local knowledge base markdown for Global RAG.
- `resources/triage/glossary.yaml`: domain entities and patterns.

### MCP vs Local Tools
- Prefer MCP for production (network-isolated, auditable, team-owned services).
- Use Local tools for development or lightweight integrations.
- Switch by editing `gateway.tools` mapping without touching agent code.


