# LangFuse Evaluation Tracing Guide

## Overview

This guide explains how evaluation components (guardrails, hallucination detection, quality assessment) appear and work in LangFuse tracing.

## How Evaluation Appears in LangFuse

### 1. **Trace Structure with Evaluation**

When you run the full workflow, you'll see this trace structure in LangFuse:

```
📊 LangFuse Trace Structure:
├── zendesk_poller_process
│   ├── input: {"tickets": [...]}
│   └── output: {"success": true, "tickets": [...]}
│
├── memory_agent_process  
│   ├── input: {"ticket": {...}}
│   ├── output: {"action": "stored_current_ticket", "related_tickets": 0}
│   └── attributes: {"memory_backend": "faiss", "similarity_threshold": 0.9}
│
├── triage_process
│   ├── input: {"tickets": [...]}
│   ├── llm_chat_completion
│   │   ├── input: {"messages": [...]}
│   │   └── output: {"content": "{\"severity\": \"critical\"...}"}
│   │
│   ├── 🔒 GUARDRAILS_EVALUATION (NEW!)
│   │   ├── input: {"content": "LLM response", "content_type": "json"}
│   │   ├── output: {"passed": true/false, "violations": [...], "severity": "..."}
│   │   └── attributes: {"evaluation_type": "guardrails", "pii_detected": false}
│   │
│   ├── 🧠 HALLUCINATION_EVALUATION (NEW!)
│   │   ├── input: {"response": "LLM response", "context": {...}}
│   │   ├── output: {"has_hallucination": true/false, "confidence": 0.8}
│   │   └── attributes: {"evaluation_type": "hallucination", "hallucination_types": [...]}
│   │
│   ├── 📊 QUALITY_EVALUATION (NEW!)
│   │   ├── input: {"response": "LLM response", "context": {...}}
│   │   ├── output: {"overall_score": 0.85, "dimension_scores": {...}}
│   │   └── attributes: {"evaluation_type": "quality", "strengths": [...], "weaknesses": [...]}
│   │
│   ├── mcp_tool_splunk_search
│   │   ├── input: {"query": "..."}
│   │   └── output: {"results": [...]}
│   │
│   └── output: {"analysis": {...}, "evaluation_results": {...}}
│
└── supervisor_process
    ├── input: {"enriched_data": {...}}
    └── output: {"decision": "ADD_COMMENT", "reason": "..."}
```

### 2. **Evaluation Spans in LangFuse**

Each evaluation component creates its own span with detailed information:

#### **Guardrails Span**
```json
{
  "name": "guardrails_evaluation",
  "input": {
    "content": "{\"severity\": \"critical\", \"category\": \"infrastructure\"}",
    "content_type": "json",
    "context": {"ticket_id": "ALERT-001", "source": "monitoring"}
  },
  "output": {
    "passed": true,
    "severity": "low",
    "message": "Content passed guardrails",
    "violations": [],
    "suggestions": []
  },
  "attributes": {
    "evaluation_type": "guardrails",
    "pii_detected": false,
    "safety_violations": 0,
    "policy_violations": 0
  }
}
```

#### **Hallucination Detection Span**
```json
{
  "name": "hallucination_evaluation", 
  "input": {
    "response": "{\"severity\": \"critical\", \"solution\": \"Restart quantum computer\"}",
    "context": {"ticket": {...}, "runbook_guidance": "..."},
    "sources": [{"text": "...", "score": 0.95}]
  },
  "output": {
    "has_hallucination": true,
    "confidence": 0.8,
    "hallucination_types": ["factual_error"],
    "detected_claims": ["quantum computer restart"],
    "suggestions": ["Verify factual claims against reliable sources"]
  },
  "attributes": {
    "evaluation_type": "hallucination",
    "factual_errors": 1,
    "impossible_claims": 0
  }
}
```

#### **Quality Assessment Span**
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
    "quality_level": "high",
    "improvement_areas": ["completeness"]
  }
}
```

### 3. **Metrics and Monitoring**

Evaluation results are also tracked as metrics:

```python
# Guardrails metrics
self.metrics.create_counter(
    "guardrail_violations_total",
    "Total guardrail violations"
).inc(1)

# Hallucination metrics  
self.metrics.create_counter(
    "hallucination_detections_total", 
    "Total hallucination detections"
).inc(1)

# Quality metrics
self.metrics.create_histogram(
    "quality_score",
    "Quality assessment scores"
).observe(quality_result.overall_score)
```

### 4. **LangFuse UI Views**

In the LangFuse UI, you'll see:

#### **Trace View**
- **Main trace**: Shows the overall workflow
- **Evaluation spans**: Nested under `triage_process`
- **Input/Output**: Full evaluation data for each component
- **Attributes**: Metadata about evaluation results

#### **Observations View**
- **Guardrails observations**: PII violations, safety issues
- **Hallucination observations**: Factual errors, inconsistencies  
- **Quality observations**: Score breakdowns, improvement suggestions

#### **Metrics Dashboard**
- **Guardrail violations**: Count of PII/safety violations
- **Hallucination rate**: Percentage of responses with hallucinations
- **Quality scores**: Average quality scores over time
- **Evaluation latency**: Time spent on evaluation

### 5. **Filtering and Search**

You can filter traces by evaluation results:

```sql
-- Find traces with guardrail violations
SELECT * FROM traces 
WHERE attributes->>'evaluation_type' = 'guardrails' 
AND output->>'passed' = 'false'

-- Find traces with hallucinations
SELECT * FROM traces 
WHERE attributes->>'evaluation_type' = 'hallucination'
AND output->>'has_hallucination' = 'true'

-- Find low quality responses
SELECT * FROM traces 
WHERE attributes->>'evaluation_type' = 'quality'
AND (output->>'overall_score')::float < 0.7
```

### 6. **Alerting and Monitoring**

Set up alerts based on evaluation results:

```yaml
# Example alerting rules
alerts:
  guardrail_violations:
    condition: "guardrail_violations_total > 5"
    message: "High number of guardrail violations detected"
    
  hallucination_rate:
    condition: "hallucination_detections_total / total_responses > 0.1"
    message: "Hallucination rate above 10%"
    
  quality_degradation:
    condition: "avg(quality_score) < 0.7"
    message: "Average quality score below threshold"
```

### 7. **Evaluation Configuration**

Control evaluation behavior via configuration:

```yaml
# config/agent.yaml
agents:
  triage:
    guardrails:
      enable_pii_detection: true
      enable_safety_checks: true
      pii_patterns: ["email", "phone", "ssn"]
      safety_patterns: ["malicious", "harmful"]
    
    hallucination:
      enable_fact_checking: true
      confidence_threshold: 0.7
      knowledge_base_threshold: 0.8
    
    quality:
      enable_quality_assessment: true
      min_score_threshold: 0.6
      expected_format: "json"
```

### 8. **Debugging and Analysis**

Use LangFuse to debug evaluation issues:

1. **Trace Analysis**: See exactly where evaluations fail
2. **Input/Output Inspection**: Understand what triggered violations
3. **Pattern Recognition**: Identify common evaluation failures
4. **Performance Monitoring**: Track evaluation latency
5. **A/B Testing**: Compare evaluation results across different configurations

## Best Practices

1. **Monitor Evaluation Metrics**: Set up dashboards for key evaluation metrics
2. **Alert on Failures**: Get notified when evaluation thresholds are exceeded
3. **Regular Review**: Periodically review evaluation results to improve rules
4. **Configuration Tuning**: Adjust thresholds based on real-world performance
5. **Documentation**: Keep evaluation rules and thresholds documented

## Example: Viewing Evaluation in LangFuse

1. **Open LangFuse UI**: http://localhost:3000
2. **Navigate to Traces**: Click on "Traces" in the sidebar
3. **Find Triage Trace**: Look for traces with `triage_process`
4. **Expand Evaluation Spans**: Click on guardrails, hallucination, quality spans
5. **Review Results**: Check input/output data and attributes
6. **Set Up Alerts**: Configure monitoring for evaluation metrics

This comprehensive evaluation tracing gives you full visibility into how your agents are performing and where improvements are needed!
