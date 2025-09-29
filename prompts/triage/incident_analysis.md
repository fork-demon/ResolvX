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
  "confidence_level": 0.0-1.0,
  "reasoning": "detailed explanation of your assessment"
}
```

## Additional Context

- **Historical Data**: {HISTORICAL_INCIDENTS}
- **System Status**: {CURRENT_SYSTEM_STATUS}
- **Recent Changes**: {RECENT_CHANGES}
- **Monitoring Alerts**: {ACTIVE_ALERTS}

Remember to be thorough, objective, and consider all available information when making your assessment.
