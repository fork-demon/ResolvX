# Executor Comment Templates

These templates are used by the Ticket Executor Agent to format comments added to Zendesk tickets. They are **NOT LLM prompts** - just string templates for consistent formatting.

## Template Variables

All templates support the following variables:
- `{{ticket_id}}` - Zendesk ticket ID
- `{{summary}}` - Synthesis summary from Triage
- `{{root_cause}}` - Identified root cause
- `{{confidence}}` - Confidence level (high/medium/low)
- `{{recommended_actions}}` - List of recommended actions
- `{{reason}}` - Decision reason from Supervisor
- `{{assigned_to}}` - Team or person assigned
- `{{bot_signature}}` - Signature (configured in agent.yaml)

---

## 1. Simple Comment Template

Used for: `ADD_COMMENT` action

```markdown
**Automated Analysis**

{{reason}}

{{bot_signature}}
```

**Example Output:**
```
**Automated Analysis**

Automated analysis complete. No critical issues found.

---
🤖 Automated by TescoResolveX
```

---

## 2. Detailed Comment with Actions Template

Used for: `ADD_COMMENT_WITH_ACTIONS` action

```markdown
**🤖 Automated Analysis Complete**

**Summary**: {{summary}}

**Root Cause**: `{{root_cause}}`

**Confidence**: {{confidence}}

**Recommended Actions**:
{{#recommended_actions}}
{{index}}. {{priority_emoji}} [{{priority}}] {{action}}
{{/recommended_actions}}

{{bot_signature}}
```

**Example Output:**
```
**🤖 Automated Analysis Complete**

**Summary**: Basket segment API appears to be down or returning errors.

**Root Cause**: `price_advisory_api_error`

**Confidence**: HIGH

**Recommended Actions**:
1. 🔴 [HIGH] Check Price Advisory API health status
2. 🔴 [HIGH] Verify basket segment data availability
3. 🟡 [MEDIUM] Review recent deployment logs
4. 🟡 [MEDIUM] Contact pricing team if issue persists

---
🤖 Automated by TescoResolveX
```

---

## 3. Team Assignment Template

Used for: `ASSIGN_TO_TEAM` action

```markdown
**🎯 Ticket Assigned**

**Assigned to**: {{team_name}}

**Reason**: {{reason}}

**Context**: {{summary}}

{{bot_signature}}
```

**Example Output:**
```
**🎯 Ticket Assigned**

**Assigned to**: Pricing Team

**Reason**: Issue related to price advisory API which is managed by pricing team

**Context**: Basket segment API appears to be down or returning errors.

---
🤖 Automated by TescoResolveX
```

---

## 4. Human Escalation Template

Used for: `ESCALATE_TO_HUMAN` action

```markdown
**🚨 ESCALATED TO HUMAN**

**Reason**: {{reason}}

**Root Cause**: `{{root_cause}}`

**Analysis**: {{summary}}

⚠️ **This requires immediate human attention.**

{{bot_signature}}
```

**Example Output:**
```
**🚨 ESCALATED TO HUMAN**

**Reason**: High confidence issue affecting multiple customers, potential revenue impact

**Root Cause**: `price_advisory_api_critical_failure`

**Analysis**: Basket segment API completely unresponsive. Multiple customers affected across all regions.

⚠️ **This requires immediate human attention.**

---
🤖 Automated by TescoResolveX
```

---

## 5. Request More Info Template

Used for: `REQUEST_MORE_INFO` action

```markdown
**📋 Additional Information Needed**

Our automated analysis was inconclusive. To help us resolve your issue faster, please provide:

- Specific error messages or screenshots
- Steps to reproduce the issue
- Affected product GTINs or TPNBs (if applicable)
- Timeframe when the issue occurred

**Current Understanding**: {{summary}}

{{bot_signature}}
```

**Example Output:**
```
**📋 Additional Information Needed**

Our automated analysis was inconclusive. To help us resolve your issue faster, please provide:

- Specific error messages or screenshots
- Steps to reproduce the issue
- Affected product GTINs or TPNBs (if applicable)
- Timeframe when the issue occurred

**Current Understanding**: Unable to determine root cause from provided information. Need more details about the error.

---
🤖 Automated by TescoResolveX
```

---

## Priority Emoji Mapping

The executor uses these emojis to indicate priority:

| Priority | Emoji | Use Case |
|----------|-------|----------|
| HIGH     | 🔴    | Urgent, requires immediate action |
| MEDIUM   | 🟡    | Important, should be addressed soon |
| LOW      | 🟢    | Nice to have, can be addressed later |

---

## Status Icon Mapping

The executor uses these icons to indicate action type:

| Action Type | Icon | Meaning |
|-------------|------|---------|
| Analysis    | 🤖   | Automated analysis complete |
| Assignment  | 🎯   | Ticket assigned to team |
| Escalation  | 🚨   | Escalated to human |
| Info Request| 📋   | More information needed |
| Success     | ✅   | Action completed successfully |
| Warning     | ⚠️    | Requires attention |

---

## Customization

To customize templates:

1. Edit `agents/executor/agent.py`
2. Modify the comment formatting in execution methods:
   - `_add_comment()` - Line ~245
   - `_add_comment_with_actions()` - Line ~270
   - `_assign_to_team()` - Line ~315
   - `_escalate_to_human()` - Line ~345
   - `_request_more_info()` - Line ~375

3. Or override via configuration in `config/agent.yaml`:

```yaml
executor:
  comment_templates:
    simple: "prompts/executor/simple_comment.md"
    detailed: "prompts/executor/detailed_comment.md"
    assignment: "prompts/executor/assignment_comment.md"
    escalation: "prompts/executor/escalation_comment.md"
    info_request: "prompts/executor/info_request_comment.md"
```

---

## Best Practices

1. **Keep it concise**: Zendesk users skim comments
2. **Use visual indicators**: Emojis help with quick scanning
3. **Be specific**: Include actionable information
4. **Maintain consistency**: Use the same format across all tickets
5. **Include bot signature**: Makes it clear this is automated

---

*These are formatting templates, not LLM prompts. The Executor agent does NOT use an LLM - it only formats and posts comments based on Supervisor decisions.*

