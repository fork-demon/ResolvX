# Incident Analysis Prompt

Analyze this incident and provide a comprehensive assessment:

## Incident Details
- **Incident ID**: {INCIDENT_ID}
- **Reported Time**: {REPORTED_TIME}
- **Reporter**: {REPORTER}
- **Description**: {INCIDENT_DESCRIPTION}
- **Affected Systems**: {AFFECTED_SYSTEMS}
- **User Impact**: {USER_IMPACT}

## Analysis Requirements

### 1. Severity Assessment
Determine the severity level based on:
- **Critical**: System down, data loss, security breach, no workaround
- **High**: Significant impact, multiple users affected, limited workaround
- **Medium**: Moderate impact, some users affected, workaround available
- **Low**: Minor impact, few users affected, easy workaround

**Severity Level**: [Critical/High/Medium/Low]

### 2. Impact Analysis
- **Affected Systems**: List all impacted systems and services
- **User Impact**: Number of users affected and type of impact
- **Business Impact**: Financial, operational, or reputational impact
- **Technical Impact**: System functionality, performance, or availability

### 3. Root Cause Assessment
- **Primary Cause**: Most likely root cause
- **Contributing Factors**: Additional factors that may have contributed
- **Pattern Recognition**: Similar incidents in the past
- **System Dependencies**: Related systems that may be affected

### 4. Immediate Actions Required
- **Emergency Response**: Actions needed immediately
- **Workaround**: Temporary solutions to restore service
- **Communication**: Who needs to be notified
- **Documentation**: What needs to be documented

### 5. Escalation Requirements
- **Escalation Level**: Who should be notified
- **Timeline**: When escalation should occur
- **Information Required**: What information to include in escalation
- **Follow-up**: Required follow-up actions

## Available Diagnostic Tools

Based on the incident, you can recommend the following tools to gather more information:

**Log and Monitoring Tools:**
- `splunk_search`: Search application logs for errors, exceptions, failures, stack traces
- `newrelic_metrics`: Query performance metrics (CPU, memory, response times, throughput)

**Business Data Tools:**
- `base_prices_get`: Retrieve current price data for a product (TPNB/GTIN)
- `competitor_prices_get`: Get competitor pricing data for comparison
- `basket_segment_get`: Get basket segment classification for products
- `price_minimum_calculate`: Calculate minimum price across location clusters
- `price_minimum_get`: Get minimum price for specific location and product

**Documentation Tools:**
- `sharepoint_list_files`: List files/folders in SharePoint directory
- `sharepoint_download_file`: Download runbooks or documentation
- `sharepoint_search_documents`: Search for related documentation

**Ticket Management:**
- `poll_queue`: Check ticket queue status
- `get_queue_stats`: Get queue statistics

## Output Format

Provide your analysis in the following structured format:

```json
{
  "severity_level": "Critical|High|Medium|Low",
  "affected_systems": ["system1", "system2"],
  "user_impact": "description of user impact",
  "business_impact": "description of business impact",
  "root_cause": "primary cause and contributing factors",
  "immediate_actions": ["action1", "action2"],
  "workaround": "temporary solution if available",
  "escalation_required": true/false,
  "escalation_level": "team/manager/executive",
  "escalation_timeline": "immediate/within 1 hour/within 4 hours",
  "tools_to_use": ["tool_name1", "tool_name2"],
  "tool_reasoning": "Why these specific tools are needed for investigation",
  "confidence_level": 0.0-1.0,
  "reasoning": "detailed explanation of your assessment"
}
```

**CRITICAL REQUIREMENTS**:
1. **ALWAYS include the `tools_to_use` array in your JSON response**
2. **NEVER leave `tools_to_use` empty** - recommend at least 1-2 tools
3. **Match incident type to tools**:
   - Performance/slow response → `["splunk_search", "newrelic_metrics"]`
   - Errors/failures → `["splunk_search"]`
   - Pricing discrepancy → `["base_prices_get", "splunk_search"]`
   - System issues → `["splunk_search", "newrelic_metrics"]`
   - Unknown/unclear → `["splunk_search", "newrelic_metrics"]` (default)

**Example valid response**:
```json
{
  "severity_level": "High",
  "affected_systems": ["web-server", "database"],
  "user_impact": "Users experiencing slow response times",
  "business_impact": "Potential revenue loss",
  "root_cause": "Suspected database performance issue",
  "immediate_actions": ["Check database metrics", "Review error logs"],
  "workaround": "None available",
  "escalation_required": true,
  "escalation_level": "team",
  "escalation_timeline": "within 1 hour",
  "tools_to_use": ["splunk_search", "newrelic_metrics"],
  "tool_reasoning": "Need to check error logs and performance metrics to identify root cause",
  "confidence_level": 0.85,
  "reasoning": "High severity due to user impact and business risk"
}
```

## Additional Context

- **Historical Data**: {HISTORICAL_INCIDENTS}
- **System Status**: {CURRENT_SYSTEM_STATUS}
- **Recent Changes**: {RECENT_CHANGES}
- **Monitoring Alerts**: {ACTIVE_ALERTS}

Remember to be thorough, objective, and consider all available information when making your assessment.
