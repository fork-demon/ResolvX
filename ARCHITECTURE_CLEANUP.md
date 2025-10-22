# Architecture Cleanup - Splunk/NewRelic

## What Was Changed

### ❌ Removed: Splunk and NewRelic as Separate Agents
**Reason**: They were redundant - Splunk and NewRelic should be **TOOLS**, not agents.

**Files Removed**:
- `agents/splunk/agent.py` ❌
- `agents/newrelic/agent.py` ❌
- `prompts/splunk/system_prompt.md` ❌
- `prompts/newrelic/system_prompt.md` ❌

### ✅ What Replaced Them: Tools via MCP Gateway

**Current Implementation**:
- ✅ `mcp_tool_splunk_search` - MCP tool for Splunk queries
- ✅ `mcp_tool_newrelic_metrics` - MCP tool for NewRelic metrics

These tools are:
- **Called by the Triage agent** based on LLM recommendations
- **Not standalone agents** - they're diagnostic tools
- **Fully traced** with input/output in Langfuse

---

## Correct Architecture

### Before (Incorrect):
```
┌─────────────┐
│ Triage      │
│ Agent       │
└─────────────┘

┌─────────────┐     ┌─────────────┐
│ Splunk      │     │ NewRelic    │
│ Agent ❌    │     │ Agent ❌    │
└─────────────┘     └─────────────┘
```
**Problem**: Splunk/NewRelic agents just execute queries - they don't make decisions or route tickets. They're tools, not agents.

### After (Correct):
```
┌─────────────────────────────────────┐
│ Triage Agent                        │
│  (Intelligent Decision Maker)       │
│                                     │
│  ├─ RAG: Search knowledge base     │
│  ├─ LLM: Analyze + recommend tools │
│  └─ Execute recommended tools:     │
│      ├─ splunk_search ✓            │
│      ├─ newrelic_metrics ✓         │
│      ├─ base_prices_get ✓          │
│      └─ sharepoint_* ✓             │
└─────────────────────────────────────┘
```
**Benefit**: Triage agent intelligently decides when to use Splunk/NewRelic based on ticket analysis, not as separate always-running agents.

---

## Agent vs Tool Guidelines

### Use AGENTS for:
- **Decision-making** (Triage, Supervisor)
- **Orchestration** (Supervisor coordinates workflow)
- **Continuous processes** (Poller monitors queues)
- **State management** (Memory stores and retrieves context)

### Use TOOLS for:
- **Data retrieval** (Splunk logs, NewRelic metrics)
- **API calls** (Price APIs, Product APIs)
- **File operations** (SharePoint upload/download)
- **Diagnostics** (Log search, metric queries)

---

## Current Agent Architecture

### ✅ Correct Agents (4):

1. **Poller Agent** 🎫
   - **Role**: Monitor ticket queues
   - **Action**: Poll and forward tickets
   - **Continuous**: Yes (runs on schedule)

2. **Memory Agent** 🧠
   - **Role**: Manage ticket history
   - **Action**: Search for duplicates, store new tickets
   - **Continuous**: No (invoked as needed)

3. **Triage Agent** 📋
   - **Role**: Intelligent ticket analysis
   - **Action**: Analyze, enrich with tools, route
   - **Continuous**: No (invoked per ticket)
   - **Uses Tools**: Splunk, NewRelic, Price APIs, SharePoint (LLM-driven)

4. **Supervisor Agent** 👔
   - **Role**: Final decision maker
   - **Action**: Escalate, assign, route, comment
   - **Continuous**: No (invoked per ticket)

### ✅ Tools Used by Agents (16):

**MCP Tools** (12):
- poll_queue
- get_queue_stats
- splunk_search ⭐
- newrelic_metrics ⭐
- base_prices_get
- price_minimum_get
- price_minimum_calculate
- basket_segment_get
- competitor_prices_get
- competitor_promotional_prices_get
- promo_effectiveness_get
- policies_view

**Local Tools** (4):
- sharepoint_list_files
- sharepoint_download_file
- sharepoint_upload_file
- sharepoint_search_documents

---

## Benefits of This Architecture

### 1. **Cleaner Separation of Concerns**
- Agents make decisions
- Tools provide data
- No overlap or confusion

### 2. **LLM-Driven Intelligence**
- Triage agent's LLM decides when to use Splunk/NewRelic
- Not hardcoded - adapts to each ticket
- Can use multiple tools in combination

### 3. **Better Tracing**
- Tool calls are child spans of Triage agent
- Clear parent-child hierarchy
- Easy to see which tools were used for each ticket

### 4. **Easier to Maintain**
- Fewer agents to manage
- Tools are simpler than agents
- Add new tools easily without creating full agents

### 5. **More Flexible**
- Can call Splunk/NewRelic conditionally
- Can call them multiple times if needed
- Can combine with other tools (SharePoint, Price APIs)

---

## Migration Summary

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Splunk | ❌ Agent | ✅ Tool (MCP) | Cleaner |
| NewRelic | ❌ Agent | ✅ Tool (MCP) | Cleaner |
| SharePoint | N/A | ✅ Tool (Local) | 🆕 Added |
| Triage | Agent | ✅ Enhanced Agent | Uses tools |
| Poller | Agent | ✅ Agent | No change |
| Memory | Agent | ✅ Agent | No change |
| Supervisor | Agent | ✅ Agent | No change |

---

## ✅ Summary

**What we have now:**
- 4 intelligent agents (Poller, Memory, Triage, Supervisor)
- 16 tools (12 MCP + 4 Local SharePoint)
- LLM-driven tool selection
- Complete end-to-end tracing
- Clean architecture with proper separation

**Splunk and NewRelic are now properly categorized as TOOLS, not agents!** 🎯

