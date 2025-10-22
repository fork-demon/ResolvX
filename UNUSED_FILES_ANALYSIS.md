# Unused Files Analysis

## Summary

Analysis of codebase to identify unused/redundant files that can be safely removed for a cleaner minimal example repository.

---

## 🗑️ DEFINITELY UNUSED - Safe to Remove

### 1. Tools Modules (Unused)
- ❌ `core/tools/std_tools.py` - Not imported anywhere (0 references)
- ❌ `core/tools/team_tools.py` - Not imported anywhere (0 references)
- ❌ `core/mocks/tools.py` - Not imported anywhere (0 references)

**Reason**: These were placeholder files that are no longer needed. We have:
- MCP tools (via gateway)
- SharePoint tools (local tools)
- No need for std_tools or team_tools

### 2. RAG Module (Unused)
- ❌ `core/rag/global_rag.py` - Not imported anywhere (0 references)

**Reason**: We're using `core/rag/local_kb.py` (LocalKB) for RAG functionality. The global_rag module is redundant.

---

## ⚠️ PROBABLY UNUSED - Consider Removing

### 1. Evaluation Modules (Only imported in __init__.py)
- ⚠️ `core/evaluation/guardrails.py` - Only imported in `core/evaluation/__init__.py`
- ⚠️ `core/evaluation/hallucination_checker.py` - Only imported in `core/evaluation/__init__.py`
- ⚠️ `core/evaluation/quality_assessor.py` - Only imported in `core/evaluation/__init__.py`

**Status**: Defined but never actually used by any agent or workflow
**Recommendation**: Remove for minimal repo, or add a demo showing how to use them

### 2. Observability Tracers (Partially Used)
- ⚠️ `core/observability/langsmith_tracer.py` - Only 2 imports (in factory + __init__)
- ⚠️ `core/observability/console_tracer.py` - Only 2 imports (in factory + __init__)

**Status**: Alternative tracing backends that aren't actively used (we use Langfuse)
**Recommendation**: Keep if you want multi-backend support, remove if focusing only on Langfuse

### 3. Memory Backends (Defined but Not Used in Demo)
- ⚠️ `core/memory/pinecone_memory.py` - Only imported in factory/__init__
- ⚠️ `core/memory/redis_memory.py` - Only imported in factory/__init__
- ⚠️ `core/memory/faiss_memory.py` - Only imported in factory/__init__

**Status**: Alternative memory backends (we use mock_memory for demo)
**Recommendation**: Keep if you want production options, remove if focusing on minimal example

### 4. Tools Base Classes (Only for Type Definitions)
- ⚠️ `core/tools/base.py` - Only 1 import (might just be type hints)
- ⚠️ `core/tools/central_gateway.py` - Only 1 import

**Status**: May be type definitions or base classes
**Recommendation**: Check if actually needed, remove if not

---

## ✅ KEEP - Actually Used

### Graph Modules (Used)
- ✅ `core/graph/builder.py` - 3 references
- ✅ `core/graph/executor.py` - 3 references  
- ✅ `core/graph/coordinator.py` - 2 references (used by SupervisorAgent)

**Reason**: Core infrastructure for agent graphs

### Observability
- ✅ `core/observability/central_forwarder.py` - 5 references
- ✅ `core/observability/langfuse_tracer.py` - Actively used
- ✅ `core/observability/tracer.py` - Actively used
- ✅ `core/observability/metrics.py` - Actively used

**Reason**: Core observability infrastructure

---

## 📁 Other Files to Review

### Documentation (Keep)
- ✅ `docs/` - Keep all documentation
- ✅ `README.md` - Keep
- ✅ `*.md` files in root - Keep for reference

### Configuration (Keep)
- ✅ `config/` - Keep all
- ✅ `.env` - Keep
- ✅ `pyproject.toml`, `requirements.txt` - Keep

### Mock Data (Keep)
- ✅ `data/mock_tickets/` - Keep for testing
- ✅ `mock_services/` - Keep for local development

### KB (Keep)
- ✅ `kb/` - Keep all knowledge articles

### Scripts (Keep)
- ✅ `scripts/test_full_workflow.py` - Active test script
- ✅ `scripts/smoke_test.py` - Active test script
- ✅ `scripts/start_agents.py` - Used to start agents
- ✅ `scripts/demo_end_to_end.py` - Demo script

### Examples (Keep or Remove)
- ⚠️ `examples/test_zendesk_poller.py` - Standalone example
**Recommendation**: Keep if useful, remove if redundant with scripts/

---

## 🎯 Recommended Actions

### HIGH PRIORITY - Remove Immediately:
```bash
# Completely unused
rm -f core/tools/std_tools.py
rm -f core/tools/team_tools.py
rm -f core/mocks/tools.py
rm -f core/rag/global_rag.py
```

### MEDIUM PRIORITY - Consider Removing:
```bash
# Evaluation modules (if not planning to use)
rm -rf core/evaluation/

# Alternative tracing backends (if only using Langfuse)
rm -f core/observability/langsmith_tracer.py
rm -f core/observability/console_tracer.py

# Alternative memory backends (if only using mock for demo)
rm -f core/memory/pinecone_memory.py
rm -f core/memory/redis_memory.py  
rm -f core/memory/faiss_memory.py

# Standalone examples (if redundant)
rm -rf examples/
```

### LOW PRIORITY - Keep for Now:
- Graph modules (builder, executor, coordinator) - Used
- Observability core (factory, tracer, metrics) - Used
- Memory factory - Used
- Main FastAPI app - May be useful

---

## After Cleanup Checklist

After removing files, verify:

1. ✅ Run test: `python scripts/test_full_workflow.py`
2. ✅ Run smoke test: `python scripts/smoke_test.py`
3. ✅ Check imports: No import errors
4. ✅ Verify agents still work
5. ✅ Verify tools still work
6. ✅ Verify tracing still works

---

## Current State

**Before any cleanup:**
- Agents: 6 → 4 (removed Splunk/NewRelic agents) ✅ Done
- Tools: 12 → 16 (added SharePoint tools) ✅ Done
- Unused evaluation modules: Present
- Unused tool files: Present
- Alternative backends: Present but not used

**Recommended after cleanup:**
- Agents: 4 (clean)
- Tools: 16 (MCP + Local SharePoint)
- Evaluation: Removed (or documented with examples)
- Tools modules: Only what's used
- Backends: Only mock_memory + option to add others

This would result in a **clean, minimal, production-ready** example repository focused on the working features.

