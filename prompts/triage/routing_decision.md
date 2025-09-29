# Routing Decision Prompt

Based on the incident analysis, determine the optimal routing and handling approach:

## Incident Context
- **Incident ID**: {INCIDENT_ID}
- **Severity Level**: {SEVERITY_LEVEL}
- **Affected Systems**: {AFFECTED_SYSTEMS}
- **User Impact**: {USER_IMPACT}
- **Root Cause**: {ROOT_CAUSE}

## Routing Decision Framework

### 1. Team Assignment
Determine which team should handle this incident:

**Engineering Team** - Assign when:
- Code-related issues
- Application bugs
- Feature problems
- Performance issues
- Integration problems

**DevOps Team** - Assign when:
- Infrastructure issues
- Deployment problems
- Configuration issues
- Environment problems
- Monitoring/alerting issues

**Security Team** - Assign when:
- Security incidents
- Access control issues
- Data breaches
- Compliance violations
- Authentication problems

**Support Team** - Assign when:
- User-facing issues
- Documentation problems
- Training needs
- General inquiries
- Non-technical issues

### 2. Urgency Level
Determine the required urgency:

**Immediate** - Within 15 minutes:
- Critical severity
- System down
- Security breach
- Data loss

**High** - Within 1 hour:
- High severity
- Significant impact
- Multiple users affected
- No workaround

**Medium** - Within 4 hours:
- Medium severity
- Limited impact
- Workaround available
- Some users affected

**Low** - Within 24 hours:
- Low severity
- Minor impact
- Easy workaround
- Few users affected

### 3. Special Handling Requirements
Identify any special considerations:

- **Escalation Required**: Immediate escalation needed
- **Stakeholder Notification**: Specific stakeholders to notify
- **Communication Plan**: Special communication requirements
- **Documentation**: Special documentation needs
- **Follow-up**: Required follow-up actions
- **Resources**: Special resources or expertise needed

### 4. Routing Rationale
Provide clear reasoning for your routing decision:

- **Team Selection**: Why this team is best suited
- **Urgency Level**: Why this urgency level is appropriate
- **Special Requirements**: Why special handling is needed
- **Risk Assessment**: Potential risks if not handled properly
- **Success Criteria**: How success will be measured

## Output Format

Provide your routing decision in the following structured format:

```json
{
  "assigned_team": "Engineering|DevOps|Security|Support",
  "urgency_level": "Immediate|High|Medium|Low",
  "special_handling": {
    "escalation_required": true/false,
    "stakeholder_notification": ["stakeholder1", "stakeholder2"],
    "communication_plan": "description of communication requirements",
    "documentation": "special documentation needs",
    "follow_up": "required follow-up actions",
    "resources": "special resources needed"
  },
  "routing_rationale": {
    "team_selection": "explanation of team selection",
    "urgency_level": "explanation of urgency level",
    "special_requirements": "explanation of special requirements",
    "risk_assessment": "potential risks and mitigation",
    "success_criteria": "how success will be measured"
  },
  "confidence_level": 0.0-1.0,
  "alternative_routing": "alternative routing if primary not available"
}
```

## Additional Context

- **Team Availability**: {TEAM_AVAILABILITY}
- **Current Workload**: {CURRENT_WORKLOAD}
- **Escalation Matrix**: {ESCALATION_MATRIX}
- **SLA Requirements**: {SLA_REQUIREMENTS}

Remember to consider team expertise, availability, and workload when making routing decisions.
