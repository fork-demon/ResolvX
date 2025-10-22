## Q&A and Design Patterns Cheat Sheet

### What design patterns are used?
- ToolProxy / Gateway: Agents call tools by name via `ToolRegistry`; swap MCP/local without code changes.
- Separation of Concerns: Poller only polls; Triage analyzes; Supervisor finalizes.
- Strategy via Prompts: Behavior driven by per-agent prompts and config, not hard-coded logic.
- Adapter: Local Python functions and MCP servers present a uniform tool interface.
- Pipeline (LangGraph): StateGraph composes nodes per agent for clear flow.

### Why MCP instead of direct SDK calls?
- Security isolation, centralized auth, auditability, and team ownership of integrations.
- Enables reuse by multiple agents and languages.

### How does RAG fit in?
- Global RAG over `kb/` provides contextual KB retrieval for all agents (esp. Supervisor).
- Memory agent uses vectors for historical ticket similarity and daily reset for freshness.

### How do agents learn business entities (GTIN, TPNB, clusters)?
- YAML glossary (`resources/triage/glossary.yaml`) defines patterns/synonyms and tool hints.
- Triage uses the glossary to extract entities and map to Product/Location tools.

### How is tracing configured?
- `observability.backend`: `langsmith` | `langfuse` | `console`.
- Startup configures the chosen backend; environment variables provide credentials.

### How do I run a local demo offline?
- Map canonical tools to local mocks in the env override and start agents.
- Drop KB markdown into `kb/` for richer narratives.

### How do I add a team tool?
1. Implement a function in `core/tools/team_tools.py`.
2. Declare it in `gateway.tools` with `type: local` and `module: path:func`.
3. Reference by name in prompts/config (agents discover via ToolRegistry).

### How do I add an MCP tool?
- Point `gateway.tools.<name>` to `{ type: mcp, server: <mcp-server-name> }`.
- Run or register the MCP server externally; the ToolRegistry will route calls.


