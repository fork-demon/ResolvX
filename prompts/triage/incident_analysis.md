# Incident Analysis Prompt

## Incident Details
- **ID**: {INCIDENT_ID}
- **Time**: {REPORTED_TIME}
- **Reporter**: {REPORTER}
- **Description**: {INCIDENT_DESCRIPTION}
- **Affected Systems**: {AFFECTED_SYSTEMS}

---

## 📚 Historical Context & Runbook Guidance

{HISTORICAL_INCIDENTS}

---

## Your Task

Based on the incident details and historical context above:

1. **Determine severity** (Critical/High/Medium/Low) based on pricing impact
2. **Identify root cause** using historical patterns
3. **Recommend diagnostic tools** to gather evidence:
   - **Pricing issues**: `base_prices_get`, `competitor_prices_get`, `splunk_search`
   - **File processing failures**: `splunk_search`, `sharepoint_list_files` (check archive folder)
   - **Product issues**: `base_prices_get`, `splunk_search`

## Required JSON Output

```json
{
  "severity_level": "Critical|High|Medium|Low",
  "root_cause": "Brief root cause based on historical context",
  "tools_to_use": ["tool1", "tool2"],
  "tool_reasoning": "Why these tools based on runbook guidance",
  "confidence_level": 0.0-1.0,
  "reasoning": "Key insights from historical context"
}
```

**CRITICAL**: 
- **USE the historical context** above to inform your analysis
- **ALWAYS include 1-3 tools** in `tools_to_use` array
- **Reference runbook patterns** in your reasoning
