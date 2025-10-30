# Synthesize Diagnostic Results

**Ticket**: {{ TICKET_ID }} - {{ TITLE }}

## Context
**Incident Type**: {{ ENTITIES }}
**Tools Executed**: {{ TOOL_RESULTS }}

## Runbook Guidance
{{ RUNBOOK_GUIDANCE }}

---

## Task
Analyze tool results and provide actionable summary.

**Return ONLY this JSON (no markdown, no code blocks):**
```json
{
  "summary": "What tools found (1-2 sentences)",
  "root_cause": "Likely cause or 'unknown'",
  "recommended_actions": [
    {"action": "Specific step", "priority": "high|medium|low"}
  ],
  "escalation_needed": true|false,
  "confidence": "high|medium|low"
}
```

**Rules:**
- Base on actual tool results only
- Be concise and specific
- If unclear, say "unknown"

