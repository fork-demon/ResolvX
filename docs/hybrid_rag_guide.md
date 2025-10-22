# Hybrid RAG Guide: Local KB + Global RAG

## Overview

The framework supports both local and global RAG for flexible knowledge retrieval:

- **LocalKB** (`core/rag/local_kb.py`): Team-specific, fast, local knowledge base
- **GlobalRAG** (`core/rag/global_rag.py`): Company-wide, centralized RAG service via API

## When to Use Which

### Use LocalKB When:
- ✅ Team-specific runbooks and procedures
- ✅ Fast response needed (no network latency)
- ✅ Sensitive information that shouldn't leave the team
- ✅ Frequent updates to knowledge base
- ✅ Local development and testing

### Use GlobalRAG When:
- ✅ Company-wide knowledge base (Confluence, SharePoint, etc.)
- ✅ Cross-team shared documentation
- ✅ Large-scale document corpus (millions of docs)
- ✅ Centralized updates and governance
- ✅ Advanced features (re-ranking, multi-modal, etc.)

### Use Both (Hybrid):
- ✅ Search local first (fast, team-specific)
- ✅ Fall back to global if local doesn't have good results
- ✅ Combine results from both sources

---

## Configuration

### Local Mode (Current)
```yaml
# config/agent.yaml
agents:
  triage:
    rag:
      type: "local"
      knowledge_dir: "kb"
      model_name: "all-MiniLM-L6-v2"
```

### API Mode (Global RAG Service)
```yaml
# config/agent.yaml
agents:
  triage:
    rag:
      type: "global"
      api_url: "https://rag-service.tesco.com/api/v1"
      api_key: "${RAG_API_KEY}"
      namespace: "pricing"  # Optional: filter by namespace
```

### Hybrid Mode (Both)
```yaml
# config/agent.yaml
agents:
  triage:
    rag:
      type: "hybrid"
      local:
        knowledge_dir: "kb"
        model_name: "all-MiniLM-L6-v2"
      global:
        api_url: "https://rag-service.tesco.com/api/v1"
        api_key: "${RAG_API_KEY}"
        namespace: "pricing"
      fallback_threshold: 0.5  # Use global if local score < 0.5
```

---

## Implementation Example

### Simple: Local Only (Current)
```python
from core.rag.local_kb import LocalKB

# Initialize
rag = LocalKB(knowledge_dir="kb", model_name="all-MiniLM-L6-v2")
rag.load()

# Search
results = rag.search("basket segments failure", k=3)
```

### Simple: Global API Only
```python
from core.rag.global_rag import GlobalRAG

# Initialize
rag = GlobalRAG(
    api_url="https://rag-service.tesco.com/api/v1",
    api_key="your-api-key",
    mode="api"
)
await rag.initialize()

# Search
results = await rag.search_async(
    "basket segments failure",
    k=3,
    namespace="pricing"
)
```

### Advanced: Hybrid Approach
```python
from core.rag.local_kb import LocalKB
from core.rag.global_rag import GlobalRAG

# Initialize both
local_rag = LocalKB(knowledge_dir="kb", model_name="all-MiniLM-L6-v2")
local_rag.load()

global_rag = GlobalRAG(
    api_url="https://rag-service.tesco.com/api/v1",
    api_key="your-api-key",
    mode="api"
)
await global_rag.initialize()

# Hybrid search
def hybrid_search(query: str, k: int = 5):
    # 1. Search local first (fast)
    local_results = local_rag.search(query, k=3)
    
    # 2. Check if local has good results
    if local_results and local_results[0]["score"] > 0.7:
        # Local has good match - use it
        return local_results
    
    # 3. Search global for comprehensive results
    global_results = await global_rag.search_async(query, k=5, namespace="pricing")
    
    # 4. Combine and deduplicate
    combined = local_results + global_results
    # Sort by score
    combined.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return combined[:k]

# Use hybrid search
results = await hybrid_search("basket segments failure", k=5)
```

---

## Integration with Triage Agent

### Option 1: Local Only (Current)
```python
# agents/triage/agent.py
triage = TriageAgent(
    config=config,
    tool_registry=tool_registry,
    rag=LocalKB(knowledge_dir="kb", model_name="all-MiniLM-L6-v2")
)
```

### Option 2: Global API
```python
# agents/triage/agent.py
global_rag = GlobalRAG(
    api_url="https://rag-service.tesco.com/api/v1",
    api_key=os.getenv("RAG_API_KEY"),
    mode="api"
)
await global_rag.initialize()

triage = TriageAgent(
    config=config,
    tool_registry=tool_registry,
    rag=global_rag
)
```

### Option 3: Hybrid (Best of Both)
```python
# Create a hybrid RAG wrapper
class HybridRAG:
    def __init__(self, local_rag, global_rag):
        self.local = local_rag
        self.global_ = global_rag
    
    def search(self, query: str, k: int = 5):
        # Try local first
        local_results = self.local.search(query, k=3)
        
        if local_results and local_results[0]["score"] > 0.7:
            return local_results
        
        # Fall back to global (would need to be sync or make this async)
        return local_results  # or await self.global_.search_async(query, k)

# Use in agent
hybrid = HybridRAG(
    local_rag=LocalKB(knowledge_dir="kb"),
    global_rag=GlobalRAG(api_url="...", mode="api")
)

triage = TriageAgent(config=config, rag=hybrid)
```

---

## API Specification

If you're building a centralized RAG service, here's the expected API:

### Endpoint: `POST /search`

**Request**:
```json
{
  "query": "basket segments file drop failure",
  "k": 5,
  "namespace": "pricing",
  "include_metadata": true,
  "filters": {
    "team": "operations",
    "document_type": "runbook"
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "text": "# Basket Segments File Drop Failure\n\n## Issue Description...",
      "score": 0.89,
      "metadata": {
        "source": "confluence",
        "page_id": "12345",
        "title": "Basket Segments Troubleshooting",
        "last_updated": "2025-10-20T14:30:00Z",
        "team": "operations"
      }
    },
    {
      "text": "...",
      "score": 0.76,
      "metadata": {...}
    }
  ],
  "query_time_ms": 45,
  "total_documents_searched": 10000
}
```

---

## Benefits of Hybrid Approach

### 1. **Performance**
- Local KB: Fast (no network)
- Global RAG: Comprehensive but slower

### 2. **Coverage**
- Local: Team-specific, up-to-date
- Global: Company-wide, all teams

### 3. **Resilience**
- If global API is down, local still works
- If local has no results, global provides backup

### 4. **Cost Optimization**
- Search local first (free)
- Only hit global API when needed (may have API costs)

---

## Current Implementation

**Active**: LocalKB (local mode)
```python
# agents/triage/agent.py (current)
self.rag = LocalKB(knowledge_dir="kb", model_name="all-MiniLM-L6-v2")
```

**Available**: GlobalRAG (API mode ready)
```python
# Can switch to global RAG by changing initialization
global_rag = GlobalRAG(
    api_url=config.global_rag_url,
    api_key=config.global_rag_key,
    mode="api"
)
```

---

## Recommendation

✅ **Keep `global_rag.py`** because:

1. **Production use case**: Many enterprises have centralized RAG services
2. **Extensibility**: Shows how to integrate with external APIs
3. **Hybrid support**: Can use both local + global for best results
4. **Example code**: Demonstrates proper API integration with tracing

It's a **valuable addition** that shows the framework can work with enterprise RAG services (Confluence, SharePoint APIs, custom knowledge bases, etc.)!

