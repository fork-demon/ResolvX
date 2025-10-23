# Evaluation and Guardrails Integration Guide

## Overview

This guide shows how to integrate the evaluation system (guardrails, hallucination detection, and quality assessment) into the Golden Agent Framework workflow.

## Available Components

### 1. Guardrails (`core.evaluation.guardrails`)
- **Content filtering**: PII detection, safety checks, policy enforcement
- **Action validation**: Tool usage validation, parameter checking
- **Risk assessment**: Severity levels (LOW, MEDIUM, HIGH, CRITICAL)

### 2. Hallucination Checker (`core.evaluation.hallucination_checker`)
- **Fact verification**: Cross-reference with knowledge base
- **Consistency checks**: Internal consistency validation
- **Source attribution**: Verify claims against sources

### 3. Quality Assessor (`core.evaluation.quality_assessor`)
- **Response quality**: Completeness, accuracy, relevance
- **Performance metrics**: Response time, resource usage
- **User satisfaction**: Feedback integration

## Integration Points

### 1. Triage Agent Integration
```python
# In agents/triage/agent.py
from core.evaluation import Guardrails, HallucinationChecker, QualityAssessor

class TriageAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # Initialize evaluation components
        self.guardrails = Guardrails(config.get("guardrails", {}))
        self.hallucination_checker = HallucinationChecker(config.get("hallucination", {}))
        self.quality_assessor = QualityAssessor(config.get("quality", {}))
    
    async def _analyze_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        # Get LLM analysis
        llm_response = await self._get_llm_analysis(incident)
        
        # Apply guardrails
        guardrail_result = await self.guardrails.check_content(
            content=llm_response,
            content_type="json",
            context={"incident": incident}
        )
        
        if not guardrail_result.passed:
            self.logger.warning(f"Guardrail violation: {guardrail_result.message}")
            # Handle violation (escalate, modify response, etc.)
        
        # Check for hallucinations
        hallucination_result = await self.hallucination_checker.check_response(
            response=llm_response,
            context=incident,
            knowledge_base=self.rag_service
        )
        
        if not hallucination_result.passed:
            self.logger.warning(f"Hallucination detected: {hallucination_result.message}")
            # Handle hallucination (re-analyze, escalate, etc.)
        
        # Assess quality
        quality_result = await self.quality_assessor.assess_response(
            response=llm_response,
            context=incident,
            expected_quality="high"
        )
        
        return {
            "analysis": llm_response,
            "guardrails": guardrail_result.dict(),
            "hallucination": hallucination_result.dict(),
            "quality": quality_result.dict()
        }
```

### 2. Supervisor Agent Integration
```python
# In agents/supervisor/agent.py
class SupervisorAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.guardrails = Guardrails(config.get("guardrails", {}))
    
    async def _make_decision(self, enriched_data: Dict[str, Any]) -> Dict[str, Any]:
        # Make decision
        decision = await self._analyze_and_decide(enriched_data)
        
        # Validate decision against guardrails
        guardrail_result = await self.guardrails.check_action(
            action=decision.get("action"),
            parameters=decision.get("parameters", {}),
            context=enriched_data
        )
        
        if not guardrail_result.passed:
            # Escalate to human or modify decision
            if guardrail_result.severity == GuardrailSeverity.CRITICAL:
                decision["action"] = "ESCALATE_TO_HUMAN"
                decision["reason"] = f"Critical guardrail violation: {guardrail_result.message}"
            else:
                # Apply suggestions to fix decision
                decision = self._apply_guardrail_suggestions(decision, guardrail_result)
        
        return {
            "decision": decision,
            "guardrails": guardrail_result.dict()
        }
```

### 3. Memory Agent Integration
```python
# In agents/memory/agent.py
class MemoryAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.guardrails = Guardrails(config.get("guardrails", {}))
    
    async def _forward_or_store(self, ticket: Dict[str, Any], related_tickets: List[Dict]) -> Dict[str, Any]:
        # Check ticket content for PII and safety
        guardrail_result = await self.guardrails.check_content(
            content=f"{ticket.get('subject', '')}\n{ticket.get('description', '')}",
            content_type="ticket",
            context={"ticket_id": ticket.get("id")}
        )
        
        if not guardrail_result.passed:
            # Handle PII or safety violations
            if "pii_detected" in guardrail_result.violations:
                # Redact PII before storing
                ticket = self._redact_pii(ticket, guardrail_result.violations)
            elif "safety_violation" in guardrail_result.violations:
                # Escalate to human review
                return {"action": "escalate_to_human", "reason": "Safety violation detected"}
        
        # Continue with normal processing
        return await super()._forward_or_store(ticket, related_tickets)
```

## Configuration

### Agent Configuration
```yaml
# config/agent.yaml
agents:
  triage:
    guardrails:
      enable_pii_detection: true
      enable_safety_checks: true
      enable_policy_enforcement: true
      require_human_approval: false
      pii_patterns:
        - "email"
        - "phone"
        - "ssn"
      safety_patterns:
        - "malicious"
        - "harmful"
    
    hallucination:
      enable_fact_checking: true
      enable_consistency_checks: true
      knowledge_base_threshold: 0.8
      cross_reference_sources: true
    
    quality:
      enable_quality_assessment: true
      expected_quality: "high"
      performance_tracking: true
      user_feedback_integration: true

  supervisor:
    guardrails:
      enable_action_validation: true
      critical_actions_require_approval: true
      escalation_threshold: "high"
    
  memory:
    guardrails:
      enable_content_filtering: true
      pii_redaction: true
      safety_escalation: true
```

## Implementation Steps

### Step 1: Add Evaluation to Base Agent
```python
# In core/graph/base.py
from core.evaluation import Guardrails, HallucinationChecker, QualityAssessor

class BaseAgent:
    def __init__(self, config: AgentConfig):
        # ... existing code ...
        
        # Initialize evaluation components if configured
        if config.get("guardrails"):
            self.guardrails = Guardrails(config.get("guardrails"))
        if config.get("hallucination"):
            self.hallucination_checker = HallucinationChecker(config.get("hallucination"))
        if config.get("quality"):
            self.quality_assessor = QualityAssessor(config.get("quality"))
```

### Step 2: Add Evaluation Hooks
```python
# In core/graph/base.py
async def _evaluate_response(self, response: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate agent response using guardrails and quality checks."""
    evaluation_results = {}
    
    if hasattr(self, 'guardrails'):
        guardrail_result = await self.guardrails.check_content(
            content=str(response),
            content_type="response",
            context=context
        )
        evaluation_results["guardrails"] = guardrail_result.dict()
    
    if hasattr(self, 'hallucination_checker'):
        hallucination_result = await self.hallucination_checker.check_response(
            response=str(response),
            context=context
        )
        evaluation_results["hallucination"] = hallucination_result.dict()
    
    if hasattr(self, 'quality_assessor'):
        quality_result = await self.quality_assessor.assess_response(
            response=str(response),
            context=context
        )
        evaluation_results["quality"] = quality_result.dict()
    
    return evaluation_results
```

### Step 3: Update Agent Workflows
```python
# In agents/triage/agent.py
async def _analyze_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
    # Get LLM analysis
    analysis = await self._get_llm_analysis(incident)
    
    # Evaluate the analysis
    evaluation = await self._evaluate_response(analysis, {"incident": incident})
    
    # Handle evaluation results
    if not evaluation.get("guardrails", {}).get("passed", True):
        self.logger.warning("Guardrail violation in analysis")
        # Handle violation
    
    return {
        "analysis": analysis,
        "evaluation": evaluation
    }
```

## Monitoring and Alerting

### Evaluation Metrics
```python
# In core/observability/metrics.py
def track_evaluation_metrics(evaluation_results: Dict[str, Any]):
    """Track evaluation metrics for monitoring."""
    if "guardrails" in evaluation_results:
        guardrails = evaluation_results["guardrails"]
        metrics_client.create_counter(
            "guardrail_violations_total",
            "Total guardrail violations"
        ).inc(1 if not guardrails.get("passed") else 0)
        
        metrics_client.create_histogram(
            "guardrail_severity",
            "Guardrail violation severity"
        ).observe(guardrails.get("severity", "low"))
```

## Best Practices

1. **Gradual Rollout**: Start with guardrails only, then add hallucination detection and quality assessment
2. **Configuration**: Make evaluation components configurable per agent
3. **Monitoring**: Track evaluation metrics and violations
4. **Human Oversight**: Always have human escalation paths for critical violations
5. **Performance**: Consider evaluation overhead and optimize as needed

## Example Integration

See `scripts/test_evaluation_integration.py` for a complete example of how to integrate evaluation components into the agent workflow.
