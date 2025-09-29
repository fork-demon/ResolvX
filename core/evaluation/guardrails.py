"""
Guardrails system for the Golden Agent Framework.

Provides safety checks, content filtering, and policy enforcement
for agent responses and actions.
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from core.exceptions import GuardrailError
from core.observability import get_logger


class GuardrailSeverity(str, Enum):
    """Guardrail severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    passed: bool
    severity: GuardrailSeverity
    message: str
    violations: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Guardrails:
    """
    Guardrails system for content and action validation.
    
    Provides comprehensive safety checks including content filtering,
    policy enforcement, and risk assessment.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize guardrails system.
        
        Args:
            config: Guardrails configuration
        """
        self.config = config or {}
        self.logger = get_logger("evaluation.guardrails")
        
        # Guardrail rules
        self._content_rules = self._load_content_rules()
        self._action_rules = self._load_action_rules()
        self._pii_patterns = self._load_pii_patterns()
        self._safety_patterns = self._load_safety_patterns()
        
        # Configuration
        self.enable_pii_detection = self.config.get("enable_pii_detection", True)
        self.enable_safety_checks = self.config.get("enable_safety_checks", True)
        self.enable_policy_enforcement = self.config.get("enable_policy_enforcement", True)
        self.require_human_approval = self.config.get("require_human_approval", False)

    async def check_content(
        self,
        content: str,
        content_type: str = "text",
        context: Optional[Dict[str, Any]] = None
    ) -> GuardrailResult:
        """
        Check content against guardrails.
        
        Args:
            content: Content to check
            content_type: Type of content (text, code, json, etc.)
            context: Additional context for checking
            
        Returns:
            Guardrail result
        """
        try:
            violations = []
            suggestions = []
            severity = GuardrailSeverity.LOW
            
            # Check PII if enabled
            if self.enable_pii_detection:
                pii_result = await self._check_pii(content)
                if not pii_result["passed"]:
                    violations.extend(pii_result["violations"])
                    severity = max(severity, pii_result["severity"])
            
            # Check safety patterns
            if self.enable_safety_checks:
                safety_result = await self._check_safety(content)
                if not safety_result["passed"]:
                    violations.extend(safety_result["violations"])
                    severity = max(severity, safety_result["severity"])
            
            # Check content-specific rules
            content_result = await self._check_content_rules(content, content_type)
            if not content_result["passed"]:
                violations.extend(content_result["violations"])
                severity = max(severity, content_result["severity"])
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(violations, content_type)
            
            # Determine if passed
            passed = len(violations) == 0 or severity == GuardrailSeverity.LOW
            
            return GuardrailResult(
                passed=passed,
                severity=severity,
                message=f"Found {len(violations)} violations" if violations else "Content passed guardrails",
                violations=violations,
                suggestions=suggestions,
                metadata={
                    "content_type": content_type,
                    "content_length": len(content),
                    "context": context,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Guardrail check failed: {e}")
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.HIGH,
                message=f"Guardrail check failed: {e}",
                violations=["guardrail_error"],
            )

    async def check_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> GuardrailResult:
        """
        Check action against guardrails.
        
        Args:
            action: Action to perform
            parameters: Action parameters
            context: Additional context
            
        Returns:
            Guardrail result
        """
        try:
            violations = []
            suggestions = []
            severity = GuardrailSeverity.LOW
            
            # Check action-specific rules
            action_result = await self._check_action_rules(action, parameters)
            if not action_result["passed"]:
                violations.extend(action_result["violations"])
                severity = max(severity, action_result["severity"])
            
            # Check for destructive actions
            if self._is_destructive_action(action, parameters):
                if self.require_human_approval:
                    violations.append("destructive_action_requires_approval")
                    severity = max(severity, GuardrailSeverity.CRITICAL)
                else:
                    violations.append("destructive_action_detected")
                    severity = max(severity, GuardrailSeverity.HIGH)
            
            # Check parameter safety
            param_result = await self._check_parameter_safety(parameters)
            if not param_result["passed"]:
                violations.extend(param_result["violations"])
                severity = max(severity, param_result["severity"])
            
            # Generate suggestions
            suggestions = await self._generate_action_suggestions(violations, action)
            
            # Determine if passed
            passed = len(violations) == 0 or severity == GuardrailSeverity.LOW
            
            return GuardrailResult(
                passed=passed,
                severity=severity,
                message=f"Found {len(violations)} action violations" if violations else "Action passed guardrails",
                violations=violations,
                suggestions=suggestions,
                metadata={
                    "action": action,
                    "parameters": parameters,
                    "context": context,
                }
            )
            
        except Exception as e:
            self.logger.error(f"Action guardrail check failed: {e}")
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.HIGH,
                message=f"Action guardrail check failed: {e}",
                violations=["guardrail_error"],
            )

    async def _check_pii(self, content: str) -> Dict[str, Any]:
        """Check for personally identifiable information."""
        violations = []
        severity = GuardrailSeverity.LOW
        
        for pattern_name, pattern in self._pii_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                violations.append(f"pii_detected_{pattern_name}")
                if pattern_name in ["ssn", "credit_card", "passport"]:
                    severity = GuardrailSeverity.CRITICAL
                elif pattern_name in ["email", "phone"]:
                    severity = GuardrailSeverity.MEDIUM
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "severity": severity,
        }

    async def _check_safety(self, content: str) -> Dict[str, Any]:
        """Check for safety violations."""
        violations = []
        severity = GuardrailSeverity.LOW
        
        for pattern_name, pattern in self._safety_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(f"safety_violation_{pattern_name}")
                if pattern_name in ["violence", "hate_speech", "self_harm"]:
                    severity = GuardrailSeverity.CRITICAL
                elif pattern_name in ["inappropriate", "offensive"]:
                    severity = GuardrailSeverity.HIGH
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "severity": severity,
        }

    async def _check_content_rules(self, content: str, content_type: str) -> Dict[str, Any]:
        """Check content-specific rules."""
        violations = []
        severity = GuardrailSeverity.LOW
        
        # Check content length
        max_length = self._content_rules.get("max_length", 10000)
        if len(content) > max_length:
            violations.append("content_too_long")
            severity = GuardrailSeverity.MEDIUM
        
        # Check for code injection
        if content_type == "text" and self._contains_code_injection(content):
            violations.append("potential_code_injection")
            severity = GuardrailSeverity.HIGH
        
        # Check for SQL injection
        if self._contains_sql_injection(content):
            violations.append("potential_sql_injection")
            severity = GuardrailSeverity.HIGH
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "severity": severity,
        }

    async def _check_action_rules(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check action-specific rules."""
        violations = []
        severity = GuardrailSeverity.LOW
        
        # Check if action is allowed
        if action not in self._action_rules.get("allowed_actions", []):
            violations.append("action_not_allowed")
            severity = GuardrailSeverity.HIGH
        
        # Check parameter requirements
        required_params = self._action_rules.get("required_parameters", {}).get(action, [])
        for param in required_params:
            if param not in parameters:
                violations.append(f"missing_required_parameter_{param}")
                severity = GuardrailSeverity.MEDIUM
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "severity": severity,
        }

    async def _check_parameter_safety(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check parameter safety."""
        violations = []
        severity = GuardrailSeverity.LOW
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # Check for dangerous patterns in string parameters
                if self._contains_dangerous_patterns(value):
                    violations.append(f"dangerous_parameter_{key}")
                    severity = GuardrailSeverity.HIGH
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "severity": severity,
        }

    def _is_destructive_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Check if action is potentially destructive."""
        destructive_actions = [
            "delete", "remove", "destroy", "drop", "truncate",
            "shutdown", "restart", "kill", "terminate",
            "format", "wipe", "clear_all"
        ]
        
        action_lower = action.lower()
        return any(destructive in action_lower for destructive in destructive_actions)

    def _contains_code_injection(self, content: str) -> bool:
        """Check for potential code injection."""
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in dangerous_patterns)

    def _contains_sql_injection(self, content: str) -> bool:
        """Check for potential SQL injection."""
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+set',
            r';\s*drop',
            r';\s*delete',
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in sql_patterns)

    def _contains_dangerous_patterns(self, value: str) -> bool:
        """Check for dangerous patterns in parameter values."""
        dangerous_patterns = [
            r'rm\s+-rf',
            r'format\s+c:',
            r'del\s+/s',
            r'shutdown',
            r'reboot',
        ]
        
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in dangerous_patterns)

    async def _generate_suggestions(self, violations: List[str], content_type: str) -> List[str]:
        """Generate suggestions for fixing violations."""
        suggestions = []
        
        for violation in violations:
            if "pii_detected" in violation:
                suggestions.append("Remove or redact personally identifiable information")
            elif "safety_violation" in violation:
                suggestions.append("Review content for inappropriate or harmful material")
            elif "content_too_long" in violation:
                suggestions.append("Consider breaking content into smaller chunks")
            elif "code_injection" in violation:
                suggestions.append("Sanitize input to prevent code injection")
            elif "sql_injection" in violation:
                suggestions.append("Use parameterized queries to prevent SQL injection")
        
        return suggestions

    async def _generate_action_suggestions(self, violations: List[str], action: str) -> List[str]:
        """Generate suggestions for fixing action violations."""
        suggestions = []
        
        for violation in violations:
            if "destructive_action" in violation:
                suggestions.append("Consider using a safer alternative or add confirmation")
            elif "action_not_allowed" in violation:
                suggestions.append("Check if this action is permitted in your configuration")
            elif "missing_required_parameter" in violation:
                suggestions.append("Provide all required parameters for this action")
            elif "dangerous_parameter" in violation:
                suggestions.append("Review parameter values for safety")
        
        return suggestions

    def _load_content_rules(self) -> Dict[str, Any]:
        """Load content validation rules."""
        return {
            "max_length": 10000,
            "allowed_html_tags": ["p", "br", "strong", "em", "ul", "ol", "li"],
            "forbidden_patterns": [],
        }

    def _load_action_rules(self) -> Dict[str, Any]:
        """Load action validation rules."""
        return {
            "allowed_actions": [
                "read", "search", "query", "analyze", "generate",
                "create", "update", "list", "get", "fetch"
            ],
            "required_parameters": {
                "create": ["name", "type"],
                "update": ["id"],
                "delete": ["id"],
            },
        }

    def _load_pii_patterns(self) -> Dict[str, str]:
        """Load PII detection patterns."""
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "passport": r'\b[A-Z]{2}\d{6,9}\b',
        }

    def _load_safety_patterns(self) -> Dict[str, str]:
        """Load safety violation patterns."""
        return {
            "violence": r'\b(kill|murder|violence|attack|harm)\b',
            "hate_speech": r'\b(hate|racist|sexist|discriminat)\b',
            "self_harm": r'\b(suicide|self.harm|hurt.myself)\b',
            "inappropriate": r'\b(explicit|adult|nsfw)\b',
            "offensive": r'\b(stupid|idiot|moron|dumb)\b',
        }
