# LangFuse Trace Structure with Evaluation

## Visual Trace Structure

```
📊 LangFuse Trace: "Full Workflow Test"
├── 🎯 zendesk_poller_process
│   ├── input: {"tickets": [{"id": "ALERT-001", "subject": "Server down"}]}
│   └── output: {"success": true, "tickets": [...]}
│
├── 🧠 memory_agent_process
│   ├── input: {"ticket": {"id": "ALERT-001", "subject": "Server down"}}
│   ├── output: {"action": "stored_current_ticket", "related_tickets": 0}
│   └── attributes: {"memory_backend": "faiss", "similarity_threshold": 0.9}
│
├── 🔍 triage_process
│   ├── input: {"tickets": [{"id": "ALERT-001", "subject": "Server down"}]}
│   │
│   ├── 📝 llm_chat_completion
│   │   ├── input: {"messages": [{"role": "system", "content": "..."}]}
│   │   └── output: {"content": "{\"severity\": \"critical\", \"tools_to_use\": [\"splunk_search\"]}"}
│   │
│   ├── 🔒 GUARDRAILS_EVALUATION ⭐ NEW!
│   │   ├── input: {"content": "LLM response", "content_type": "json"}
│   │   ├── output: {"passed": true, "violations": [], "severity": "low"}
│   │   └── attributes: {"evaluation_type": "guardrails", "pii_detected": false}
│   │
│   ├── 🧠 HALLUCINATION_EVALUATION ⭐ NEW!
│   │   ├── input: {"response": "LLM response", "context": {...}}
│   │   ├── output: {"has_hallucination": false, "confidence": 0.0}
│   │   └── attributes: {"evaluation_type": "hallucination", "hallucination_types": []}
│   │
│   ├── 📊 QUALITY_EVALUATION ⭐ NEW!
│   │   ├── input: {"response": "LLM response", "context": {...}}
│   │   ├── output: {"overall_score": 0.85, "dimension_scores": {...}}
│   │   └── attributes: {"evaluation_type": "quality", "strengths": [...], "weaknesses": [...]}
│   │
│   ├── 🔧 mcp_tool_splunk_search
│   │   ├── input: {"query": "index=* ALERT-001 error OR failed"}
│   │   └── output: {"results": [{"timestamp": "...", "message": "..."}]}
│   │
│   └── output: {
│       "analysis": {"severity": "critical", "tools_used": ["splunk_search"]},
│       "evaluation_results": {
│         "guardrails": {"passed": true, "violations": []},
│         "hallucination": {"has_hallucination": false, "confidence": 0.0},
│         "quality": {"overall_score": 0.85, "dimension_scores": {...}}
│       }
│   }
│
└── 🎯 supervisor_process
    ├── input: {"enriched_data": {"analysis": {...}, "evaluation_results": {...}}}
    └── output: {"decision": "ADD_COMMENT", "reason": "Automated analysis complete"}
```

## Key Evaluation Features in LangFuse

### 1. **Nested Spans**
- Each evaluation component creates its own span
- Nested under the main `triage_process` span
- Easy to identify and filter

### 2. **Rich Input/Output Data**
- **Input**: Shows what was evaluated (LLM response, context)
- **Output**: Shows evaluation results (scores, violations, suggestions)
- **Attributes**: Metadata about evaluation type and results

### 3. **Searchable and Filterable**
- Filter by evaluation type: `evaluation_type:guardrails`
- Search for violations: `violations:["pii_detected_email"]`
- Find quality issues: `overall_score:<0.7`

### 4. **Metrics Integration**
- Counter metrics for violations and detections
- Histogram metrics for quality scores
- Dashboard-ready data

### 5. **Alerting Capabilities**
- Alert on guardrail violations
- Alert on hallucination detections
- Alert on quality degradation

## How to View in LangFuse UI

1. **Open LangFuse**: http://localhost:3000
2. **Go to Traces**: Click "Traces" in sidebar
3. **Find Recent Trace**: Look for traces with `triage_process`
4. **Expand Triage Process**: Click to see nested spans
5. **View Evaluation Spans**: Look for spans with evaluation icons:
   - 🔒 Guardrails evaluation
   - 🧠 Hallucination evaluation  
   - 📊 Quality evaluation
6. **Inspect Details**: Click each span to see input/output data
7. **Check Attributes**: View metadata about evaluation results

## Example Evaluation Data You'll See

### Guardrails Span
```json
{
  "name": "guardrails_evaluation",
  "input": {
    "content": "{\"severity\": \"critical\", \"user_email\": \"john@company.com\"}",
    "content_type": "json",
    "context": {"ticket_id": "ALERT-001"}
  },
  "output": {
    "passed": false,
    "severity": "medium", 
    "message": "Found 1 violations",
    "violations": ["pii_detected_email"],
    "suggestions": ["Remove or redact personally identifiable information"]
  },
  "attributes": {
    "evaluation_type": "guardrails",
    "pii_detected": true,
    "safety_violations": 0
  }
}
```

### Quality Assessment Span
```json
{
  "name": "quality_evaluation",
  "input": {
    "response": "{\"severity\": \"critical\", \"tools_to_use\": [\"splunk_search\"]}",
    "context": {"ticket": {...}, "tools_recommended": [...]},
    "expected_format": "json"
  },
  "output": {
    "overall_score": 0.85,
    "dimension_scores": {
      "relevance": 0.9,
      "accuracy": 0.95,
      "completeness": 0.8,
      "clarity": 0.85,
      "coherence": 0.8,
      "usefulness": 0.9
    },
    "strengths": ["Strong accuracy", "Good relevance"],
    "weaknesses": ["Could be more complete"],
    "suggestions": ["Provide more comprehensive coverage"]
  },
  "attributes": {
    "evaluation_type": "quality",
    "quality_level": "high"
  }
}
```

This gives you complete visibility into how your agents are performing and where improvements are needed!
