# Domain Knowledge

This directory contains domain-specific knowledge that is **loaded directly at agent initialization**, not retrieved via RAG search.

## Files

### `glossary.md`
Contains business domain entity definitions including:
- **GTIN** (Global Trade Item Number) - Product identifiers
- **TPNB** (Tesco Product Number - Base) - Internal product IDs
- **LocationCluster** - Store groupings
- **Tool Selection Guide** - Mapping of business intents to API tools
- **Location Cluster Mappings** - UUID mappings for clusters

**Usage**: Used by Triage agent for:
- Entity extraction from tickets
- Zero-padding normalization
- Tool parameter preparation
- Intent-to-tool mapping

### `incident_types.md`
Contains incident type definitions including:
- **Severity levels** (Critical, High, Medium, Low)
- **Handling procedures** (step-by-step guides)
- **Escalation matrix** (who to notify and when)
- **SLA response times**
- **Required tools** for each incident type

**Usage**: Used by Triage agent for:
- Incident classification
- Severity determination
- Escalation decisions
- Tool selection
- Team routing

## Architecture

```
kb/
├── domain/                     # Domain knowledge (loaded at startup)
│   ├── glossary.md            # Entity definitions + tool mappings
│   ├── incident_types.md      # Incident classifications + procedures
│   └── README.md              # This file
│
└── *.md                       # Runbooks (retrieved via RAG search)
    ├── price_discrepancy_runbook.md
    ├── performance_issues_runbook.md
    └── ...
```

## Loading Mechanism

### Domain Knowledge (This Directory)
```python
from core.domain import load_domain_knowledge

# Loaded once at agent initialization
knowledge = load_domain_knowledge()

# Fast in-memory lookups
gtin_info = knowledge.get_entity_info("GTIN")
incident_info = knowledge.get_incident_type("security_breach")
tool_info = knowledge.get_tool_for_intent("price_minimum_get")
```

### Runbooks (Parent Directory)
```python
from core.rag.local_kb import LocalKB

# Loaded at startup, searched on-demand
rag = LocalKB(knowledge_dir="kb")
rag.load()

# Semantic search for relevant runbooks
results = rag.search("How to investigate price discrepancy?", k=3)
```

## Why Separate?

1. **Performance**: Domain knowledge needs fast, exact lookups (O(1) dict access)
2. **Precision**: Entity definitions must be exact, not fuzzy semantic matches
3. **Reliability**: Tool mappings should be deterministic, not probabilistic
4. **Flexibility**: Runbooks benefit from semantic search and context retrieval

## Adding New Domain Knowledge

### New Entity
Add to `glossary.md` under "Entity Definitions":
```markdown
### MyEntity
- **Type**: Entity Type
- **Description**: What it represents
- **Patterns**: `\b\d{4,6}\b`
- **Examples**: `1234`
- **Synonyms**: other names
```

### New Incident Type
Add to `incident_types.md`:
```markdown
### My New Incident
- **Severity**: High
- **Escalation Level**: High
- **Handling Team**: Engineering
- **SLA Response**: 1 hour
- **SLA Resolution**: 8 hours

**Description**: Brief description

**Procedures**:
1. Step one
2. Step two

**Tools**: tool1, tool2
```

### New Tool Mapping
Add to `glossary.md` under "Tool Selection Guide":
```markdown
#### my_new_tool
- **Purpose**: What it does
- **Required Parameters**:
  - `param1` (what it means)
  - `param2` (what it means)
```

## Testing

Test domain knowledge loading:
```bash
python scripts/test_domain_knowledge.py
```

Test Triage agent with domain knowledge:
```bash
python scripts/test_triage_with_domain.py
```

