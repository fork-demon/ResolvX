# Step 1: Entity Extraction & Incident Classification

Analyze this incident and extract key information:

**Ticket**: {{ TICKET_ID }}
**Description**: {{ DESCRIPTION }}

## Task

Extract information from the ticket description above:

1. **Classify incident type**: Analyze the description and classify as one of: `price_not_found`, `incorrect_price`, `file_processing_failed`, `product_issue`, or `unknown`

2. **Extract entities** if mentioned in the description:
   - **GTIN**: 14-digit product code (starts with 0 or 5)
   - **TPNB**: 9-digit Tesco product number
   - **Locations**: Store names or location identifiers
   - **Key terms**: Important words for search (extract 3-5 relevant terms from description)

## Output Schema

Return **ONLY** a JSON object with these fields:

| Field | Type | Description | When to populate |
|-------|------|-------------|------------------|
| `gtin` | string or null | 14-digit product code | If found in description |
| `tpnb` | string or null | 9-digit Tesco product number | If found in description |
| `locations` | array of strings | Store/location names | If mentioned in description |
| `incident_type` | string (required) | One of: `price_not_found`, `incorrect_price`, `file_processing_failed`, `product_issue`, `unknown` | Always classify based on keywords |
| `key_terms` | array of strings | Important words for KB search | Extract 3-5 relevant terms |
| `classification_reason` | string | Why you chose this incident_type | Brief explanation |

**CRITICAL**: 
- Do NOT use example values like "05000123456789 or null"
- Extract ACTUAL values from the ticket description
- If no GTIN found, use `null` (not a string)
- If no TPNB found, use `null` (not a string)
- key_terms should be words FROM the description, not generic examples
