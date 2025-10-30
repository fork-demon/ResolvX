# Step 2: Create Execution Plan from KB Articles

Create a diagnostic execution plan based on the incident details, extracted entities, and knowledge base guidance below.

## Incident Details
- **Ticket ID**: {{ TICKET_ID }}
- **Title**: {{ TICKET_TITLE }}
- **Description**: {{ TICKET_DESCRIPTION }}

## Extracted Entities
- **Incident Type**: {{ INCIDENT_TYPE }}
- **GTIN**: {% if ENTITIES.gtin %}{{ ENTITIES.gtin }}{% else %}Not found{% endif %}
- **TPNB**: {% if ENTITIES.tpnb %}{{ ENTITIES.tpnb }}{% else %}Not found{% endif %}
- **Locations**: {% if ENTITIES.locations %}{{ ENTITIES.locations|join(", ") }}{% else %}None{% endif %}
- **Key Terms**: {% if ENTITIES.key_terms %}{{ ENTITIES.key_terms|join(", ") }}{% else %}None{% endif %}
- **Classification Reason**: {{ ENTITIES.classification_reason }}

## Knowledge Base Guidance

{{ KB_ARTICLES }}

{{ AVAILABLE_TOOLS }}

## Your Task

**READ the KB articles above carefully** and extract the recommended diagnostic steps.

**Process**:
1. KB articles mention tool names → find matching tool in "Available Tools" section
2. Extract recommended tools (max 3 tools)
3. For each tool, identify what it needs (parameters will be formed in next step)

## Output Schema

Return **ONLY** a JSON object with this structure:

| Field | Type | Description |
|-------|------|-------------|
| `plan_type` | string | "diagnostic" or "enrichment" |
| `steps` | array | List of execution steps (max 3) |
| `steps[].step` | number | Step number (1, 2, 3) |
| `steps[].tool` | string | Tool name from Available Tools list (exact match) |
| `steps[].reason` | string | Why this tool is needed (from KB article) |
| `routing` | object | Routing information |
| `routing.forward_to_team` | string or null | Team name or null |
| `routing.reason` | string | Routing reason |
| `summary` | string | Brief plan summary |

**IMPORTANT**: 
- Use exact tool names from "Available Tools" section above
- Do NOT include `parameters` field - that will be formed in next step
- Do NOT include example parameter values
- Just list the tools and reasons

