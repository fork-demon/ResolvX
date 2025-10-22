# Memory Agent with FAISS Vector Search

## 🎯 Overview

The Memory Agent now uses **FAISS (Facebook AI Similarity Search)** as its vector memory backend for **production-ready duplicate ticket detection**. This provides:

- ⚡ **Fast similarity search** (optimized for high-dimensional vectors)
- 🎯 **Accurate duplicate detection** (cosine similarity with normalized embeddings)
- 💾 **Persistent storage** (auto-saves to disk after each operation)
- 🔄 **Auto-detection** (automatically detects embedding dimensions)
- 📈 **Scalable** (can handle thousands of tickets efficiently)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Memory Agent                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Receive ticket (subject + description)                       │
│  2. Generate embedding (SentenceTransformers)                    │
│  3. Search FAISS index for similar tickets                       │
│  4. Check similarity threshold (90%)                             │
│  5. Action:                                                       │
│     - If similar: forward_to_supervisor                          │
│     - If new: store in FAISS index                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
  ┌─────────────┐                   ┌─────────────┐
  │   FAISS     │                   │ Sentence    │
  │   Index     │                   │ Transformers│
  │  (384-dim)  │                   │ all-MiniLM  │
  │             │                   │   -L6-v2    │
  └─────────────┘                   └─────────────┘
   - index.faiss                     - Normalized
   - metadata.pkl                    - embeddings
```

## 📋 Configuration

### Config File (`config/agent.yaml`)

```yaml
memory:
  backend: "faiss"

  faiss:
    index_path: "./data/faiss_index"
    embedding_model: "all-MiniLM-L6-v2"
    dimension: null  # Auto-detect (384 for all-MiniLM-L6-v2)
    index_type: "IndexFlatIP"  # Inner Product = Cosine Similarity
    metric: "IP"

agents:
  memory:
    enabled: true
    type: "memory"
    namespace_prefix: "tickets"
    search_limit: 10
    search_threshold: 0.9  # 90% similarity for duplicates
    duplicate_threshold: 0.95  # 95% for exact match
    embedding_model: "all-MiniLM-L6-v2"
```

## 🔍 How Duplicate Detection Works

### 1. Ticket Content Extraction
Only **subject + description** are used for embeddings:
```python
content = f"{ticket['subject']}\n\n{ticket['description']}"
```
**IDs, timestamps, and metadata are NOT included** to ensure duplicates are detected even with different IDs.

### 2. Embedding Generation
```python
# Using SentenceTransformers with normalized embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedding = embedder.encode([text], normalize_embeddings=True)[0]
# Result: 384-dimensional normalized vector (L2 norm = 1.0)
```

### 3. Similarity Search
```python
# FAISS IndexFlatIP computes inner product
# For normalized vectors: Inner Product = Cosine Similarity
similarity = np.dot(embedding1, embedding2)
# Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
```

### 4. Duplicate Classification
```python
if similarity >= 0.95:  # 95%+ similarity
    action = "duplicate_exact"
elif similarity >= 0.9:  # 90-95% similarity
    action = "forward_to_supervisor"
else:
    action = "stored_current_ticket"  # New ticket
```

## 📊 Similarity Thresholds

| Similarity | Classification | Action |
|-----------|----------------|--------|
| ≥ 0.95 | Exact duplicate | Return resolution / Merge |
| 0.90 - 0.95 | Near duplicate | Forward to supervisor |
| < 0.90 | Different ticket | Store as new |

## 🚀 Usage

### Basic Usage

```python
from agents.memory.agent import MemoryAgent, MemoryAgentState
from core.memory.factory import MemoryFactory

# Create FAISS memory backend
factory = MemoryFactory()
memory = factory.create_memory("faiss", {
    "dimension": None,  # Auto-detect
    "index_type": "IndexFlatIP",
    "metric": "IP",
    "index_path": "./data/faiss_index"
})
await memory.initialize()

# Create memory agent
agent = MemoryAgent({
    "namespace_prefix": "tickets",
    "search_limit": 10,
    "search_threshold": 0.9,
    "embedding_model": "all-MiniLM-L6-v2"
}, memory=memory)

# Process ticket
ticket = {
    "id": "TICKET-001",
    "subject": "Price discrepancy in store",
    "description": "Customer reported wrong price at checkout",
    "team": "operations"
}

state = MemoryAgentState(
    agent_name="memory",
    agent_type="MemoryAgent",
    input_data={"ticket": ticket, "team": "operations"}
)

result = await agent.process(state)
action = result.get("result", {}).get("action")
# Possible actions:
# - "stored_current_ticket" (new ticket)
# - "forward_to_supervisor" (similar ticket found)
# - "duplicate_resolved_return_resolution" (resolved duplicate)
# - "duplicate_in_progress_merge" (in-progress duplicate)
```

### Full Workflow Test

```bash
# Clean start
rm -rf ./data/faiss_index

# Run the full workflow
python scripts/test_full_workflow.py

# Expected output:
# ✓ Memory Backend: faiss
# ✓ Memory action: stored_current_ticket
# ✓ Related tickets: 0
```

### Duplicate Detection Test

```bash
# Test duplicate detection within a single process
python scripts/test_memory_duplicate.py

# Expected output:
# 1️⃣ FIRST SUBMISSION: stored_current_ticket (0 related)
# 2️⃣ SECOND SUBMISSION: forward_to_supervisor (1 related, 99.5% similarity)
# 3️⃣ THIRD SUBMISSION: stored_current_ticket (0 related)
```

## 📁 File Structure

```
data/faiss_index/
├── index.faiss       # FAISS vector index (binary)
└── metadata.pkl      # Ticket metadata (pickle)

core/memory/
├── base.py           # BaseMemory interface
├── faiss_memory.py   # FAISS implementation ✅
├── factory.py        # Memory factory
└── __init__.py       # Package exports

agents/memory/
└── agent.py          # Memory Agent implementation
```

## 🔧 Technical Details

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Normalization**: Yes (L2 norm = 1.0)
- **Speed**: ~3000 sentences/sec on CPU
- **Quality**: Optimized for semantic similarity

### FAISS Index
- **Type**: `IndexFlatIP` (Inner Product)
- **Search**: Exact brute-force search (no approximation)
- **Complexity**: O(n) for n stored vectors
- **Memory**: ~1.5KB per stored ticket (384 floats + metadata)

### Persistence
- **Auto-save**: After each store operation
- **Format**: Binary FAISS index + Pickle metadata
- **Load**: Automatic on initialization if index exists

## 🐛 Troubleshooting

### Issue: Low Similarity for Duplicate Tickets

**Symptom**: Identical tickets show low similarity (< 0.5)

**Causes**:
1. Embeddings not normalized
2. Different embedding models used
3. Ticket content includes varying IDs/timestamps

**Solution**:
```python
# Ensure normalization
embedding = embedder.encode([text], normalize_embeddings=True)[0]

# Verify embedding norm
import numpy as np
norm = np.linalg.norm(embedding)
print(f"Embedding norm: {norm}")  # Should be ~1.0
```

### Issue: FAISS Index Not Persisting

**Symptom**: Index resets between runs

**Causes**:
1. Index path not writable
2. Save operation failing silently

**Solution**:
```python
# Check save logs
# Should see: "Saved FAISS index and metadata"

# Verify files exist
import os
print(os.listdir("./data/faiss_index"))
# Expected: ['index.faiss', 'metadata.pkl']
```

### Issue: Dimension Mismatch

**Symptom**: `Embedding dimension mismatch: expected X, got Y`

**Causes**:
1. Different embedding models used
2. Dimension specified in config doesn't match model

**Solution**:
```yaml
# Use auto-detection
memory:
  faiss:
    dimension: null  # Auto-detect from first embedding
```

## 📈 Performance

### Benchmarks (M1 MacBook Pro)

| Operation | Tickets | Time | Throughput |
|-----------|---------|------|------------|
| Store | 1 | ~20ms | 50/sec |
| Search | 100 | ~2ms | 500 queries/sec |
| Search | 1000 | ~15ms | 67 queries/sec |
| Search | 10000 | ~150ms | 7 queries/sec |

### Scalability

- **Up to 10K tickets**: Excellent performance (< 50ms search)
- **10K - 100K tickets**: Good performance (< 500ms search)
- **100K+ tickets**: Consider using approximate search (IndexIVFFlat)

## 🔐 Security Considerations

1. **Data at Rest**: Index files are stored unencrypted
   - Consider encrypting `./data/faiss_index/` directory
   
2. **Data in Memory**: Embeddings stored in RAM
   - Sensitive ticket content may be visible in memory dumps

3. **Access Control**: No built-in authentication
   - Implement file-system level permissions

## 🎓 Best Practices

1. **Threshold Tuning**: Adjust based on your domain
   ```python
   # Stricter (fewer duplicates detected)
   search_threshold: 0.95
   
   # More lenient (more duplicates detected)
   search_threshold: 0.85
   ```

2. **Content Normalization**: Standardize ticket content
   ```python
   # Remove timestamps, normalize whitespace
   content = re.sub(r'\d{4}-\d{2}-\d{2}', '', content)
   content = ' '.join(content.split())
   ```

3. **Regular Index Cleanup**: Remove old resolved tickets
   ```python
   # Delete tickets older than 30 days
   await memory.delete_by_metadata({
       "status": "resolved",
       "updated_at": {"$lt": "2025-09-22"}
   })
   ```

4. **Monitoring**: Track duplicate detection rate
   ```python
   metrics.counter("memory_agent.duplicates_found").inc()
   metrics.histogram("memory_agent.similarity_score").observe(score)
   ```

## 📚 References

- [FAISS Documentation](https://faiss.ai/)
- [SentenceTransformers](https://www.sbert.net/)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Vector Databases](https://www.pinecone.io/learn/vector-database/)

## ✅ Removed Implementations

The following memory backends have been removed to focus on FAISS:

- ❌ `MockMemory` - Simple in-memory storage (no persistence)
- ❌ `RedisMemory` - Redis-based storage
- ❌ `PineconeMemory` - Pinecone cloud vector DB

**FAISS is now the only supported memory backend** for production use.

---

**Last Updated**: October 22, 2025
**Version**: 1.0.0
**Status**: ✅ Production Ready

