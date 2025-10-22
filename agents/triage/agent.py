"""
Triage agent implementation for intelligent incident routing and prioritization.

This agent analyzes incoming incidents, alerts, and requests to determine
severity levels, priority, and appropriate routing decisions.
"""

import time
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from core.config import AgentConfig
from core.exceptions import AgentError
from core.graph.base import BaseAgent
from core.graph.state import AgentState
from core.gateway.tool_registry import ToolRegistry
from core.observability import get_logger, get_tracer, get_metrics_client
from core.memory.base import BaseMemory
from core.rag.local_kb import LocalKB


class TriageAgent(BaseAgent):
    """
    Triage agent for intelligent incident routing and prioritization.

    Analyzes incoming incidents, alerts, and requests to:
    - Determine severity and priority levels
    - Route tasks to appropriate teams or agents
    - Escalate critical issues immediately
    - Learn from historical patterns
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[BaseMemory] = None,
        rag: Optional[Any] = None,
        **kwargs: Any,
    ):
        """
        Initialize the triage agent.

        Args:
            config: Agent configuration
            tool_registry: Tool registry for accessing tools
            memory: Memory backend for storing context
            rag: RAG backend for knowledge retrieval
            **kwargs: Additional parameters
        """
        super().__init__("triage", config, **kwargs)

        self.tool_registry = tool_registry
        self.memory = memory
        self.rag = rag
        # Auto-load LocalKB if configured externally passed as None
        try:
            if self.rag is None:
                from core.config import Config
                # This agent doesn't have direct config object; rely on defaults
                kb = LocalKB(knowledge_dir="kb", model_name="all-MiniLM-L6-v2")
                kb.load()
                self.rag = kb
        except Exception:
            pass
        self.tracer = get_tracer("agent.triage")
        self.metrics = get_metrics_client()

        # Triage-specific configuration
        self.severity_keywords = {
            "critical": ["down", "outage", "failure", "crash", "emergency", "urgent"],
            "high": ["error", "issue", "problem", "bug", "broken"],
            "medium": ["warning", "slow", "performance", "degraded"],
            "low": ["question", "request", "enhancement", "improvement"],
        }

        self.routing_rules = {
            "security": ["security", "breach", "vulnerability", "hack", "malware"],
            "infrastructure": ["server", "network", "database", "connectivity"],
            "application": ["app", "software", "code", "deployment", "feature"],
            "support": ["user", "customer", "help", "question", "how to"],
        }

    def build_graph(self) -> StateGraph:
        """Build the LangGraph for the triage agent."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("analyze_incident", self._analyze_incident)
        graph.add_node("determine_severity", self._determine_severity)
        graph.add_node("route_incident", self._route_incident)
        graph.add_node("check_escalation", self._check_escalation)
        graph.add_node("store_decision", self._store_decision)

        # Add edges
        graph.add_edge("analyze_incident", "determine_severity")
        graph.add_edge("determine_severity", "route_incident")
        graph.add_edge("route_incident", "check_escalation")
        graph.add_edge("check_escalation", "store_decision")
        graph.add_edge("store_decision", END)

        # Set entry point
        graph.set_entry_point("analyze_incident")

        return graph

    async def process(self, state: AgentState) -> Dict[str, Any]:
        """Process an incident through the triage workflow."""
        with self.tracer.start_as_current_span("triage_process") as span:
            span.set_attribute("incident_id", state.input_data.get("incident_id", "unknown"))

            try:
                # Execute the graph
                result = await self.graph.ainvoke(state.dict())

                span.set_attribute("severity", result.get("severity", "unknown"))
                span.set_attribute("routing_decision", result.get("routing_decision", "unknown"))

                return result

            except Exception as e:
                span.record_exception(e)
                self.logger.error(f"Triage processing failed: {e}")
                raise AgentError(f"Triage processing failed: {e}") from e

    async def _analyze_incident(self, state: AgentState) -> AgentState:
        """Analyze the incoming incident to extract key information."""
        try:
            incident_data = state.input_data

            # Extract basic information
            title = incident_data.get("title", "")
            description = incident_data.get("description", "")
            source = incident_data.get("source", "unknown")
            timestamp = incident_data.get("timestamp", time.time())

            # Perform text analysis
            combined_text = f"{title} {description}".lower()

            # Extract keywords and entities
            keywords = self._extract_keywords(combined_text)
            entities = self._extract_entities(combined_text)

            # Check for KB guidance if RAG is available
            similar_incidents = []
            if self.rag:
                try:
                    # LocalKB has synchronous search; support both patterns
                    if hasattr(self.rag, "search"):
                        similar_incidents = self.rag.search(combined_text, k=3)
                    else:
                        rag_result = await self.rag.search_and_generate(
                            query=combined_text,
                            max_results=3,
                            namespace="incidents"
                        )
                        similar_incidents = rag_result.sources
                except Exception as e:
                    self.logger.warning(f"RAG search failed: {e}")

            # Store analysis results
            analysis = {
                "keywords": keywords,
                "entities": entities,
                "similar_incidents": similar_incidents,
                "text_length": len(combined_text),
                "source": source,
                "analyzed_at": time.time(),
            }

            state.add_intermediate_result({
                "step": "analyze_incident",
                "analysis": analysis,
            })

            # Update state
            state.metadata["analysis"] = analysis
            state.increment_step()

            self.logger.info(f"Analyzed incident: {len(keywords)} keywords, {len(entities)} entities")
            return state

        except Exception as e:
            state.add_error(f"Incident analysis failed: {e}", "analysis_error")
            raise

    async def _determine_severity(self, state: AgentState) -> AgentState:
        """Determine the severity level of the incident."""
        try:
            analysis = state.metadata.get("analysis", {})
            keywords = analysis.get("keywords", [])

            # Calculate severity scores
            severity_scores = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }

            # Score based on keywords
            for keyword in keywords:
                for severity, severity_keywords in self.severity_keywords.items():
                    if any(sk in keyword for sk in severity_keywords):
                        severity_scores[severity] += 1

            # Score based on time sensitivity
            incident_data = state.input_data
            if incident_data.get("urgent", False):
                severity_scores["critical"] += 2

            # Score based on similar incidents
            similar_incidents = analysis.get("similar_incidents", [])
            for incident in similar_incidents:
                if incident.get("score", 0) > 0.8:  # High similarity
                    # Inherit severity from similar incident
                    similar_severity = incident.get("metadata", {}).get("severity", "medium")
                    severity_scores[similar_severity] += 1

            # Determine final severity
            severity = max(severity_scores, key=severity_scores.get)
            confidence = severity_scores[severity] / (sum(severity_scores.values()) + 1)

            # Store severity decision
            severity_decision = {
                "severity": severity,
                "confidence": confidence,
                "scores": severity_scores,
                "reasoning": self._generate_severity_reasoning(severity, severity_scores),
            }

            state.add_intermediate_result({
                "step": "determine_severity",
                "severity_decision": severity_decision,
            })

            state.metadata["severity_decision"] = severity_decision
            state.confidence_score = confidence
            state.increment_step()

            self.logger.info(f"Determined severity: {severity} (confidence: {confidence:.2f})")
            return state

        except Exception as e:
            state.add_error(f"Severity determination failed: {e}", "severity_error")
            raise

    async def _route_incident(self, state: AgentState) -> AgentState:
        """Determine routing decisions for the incident."""
        try:
            analysis = state.metadata.get("analysis", {})
            severity_decision = state.metadata.get("severity_decision", {})

            keywords = analysis.get("keywords", [])
            severity = severity_decision.get("severity", "medium")

            # Determine routing based on keywords and content
            routing_scores = {
                "security": 0,
                "infrastructure": 0,
                "application": 0,
                "support": 0,
            }

            # Score based on routing keywords
            for keyword in keywords:
                for team, team_keywords in self.routing_rules.items():
                    if any(tk in keyword for tk in team_keywords):
                        routing_scores[team] += 1

            # Adjust scoring based on severity
            severity_multipliers = {
                "critical": 2.0,
                "high": 1.5,
                "medium": 1.0,
                "low": 0.8,
            }

            multiplier = severity_multipliers.get(severity, 1.0)
            for team in routing_scores:
                routing_scores[team] *= multiplier

            # Determine primary and secondary routing
            sorted_teams = sorted(routing_scores.items(), key=lambda x: x[1], reverse=True)
            primary_team = sorted_teams[0][0] if sorted_teams[0][1] > 0 else "support"
            secondary_team = sorted_teams[1][0] if len(sorted_teams) > 1 and sorted_teams[1][1] > 0 else None

            # Generate routing decision
            routing_decision = {
                "primary_team": primary_team,
                "secondary_team": secondary_team,
                "routing_scores": routing_scores,
                "urgency_level": self._calculate_urgency(severity),
                "estimated_effort": self._estimate_effort(analysis, severity),
                "reasoning": self._generate_routing_reasoning(primary_team, routing_scores),
            }

            state.add_intermediate_result({
                "step": "route_incident",
                "routing_decision": routing_decision,
            })

            state.metadata["routing_decision"] = routing_decision
            state.increment_step()

            self.logger.info(f"Routed to: {primary_team} (secondary: {secondary_team})")
            return state

        except Exception as e:
            state.add_error(f"Routing decision failed: {e}", "routing_error")
            raise

    async def _check_escalation(self, state: AgentState) -> AgentState:
        """Check if the incident requires escalation."""
        try:
            severity_decision = state.metadata.get("severity_decision", {})
            routing_decision = state.metadata.get("routing_decision", {})

            severity = severity_decision.get("severity", "medium")
            confidence = severity_decision.get("confidence", 0.5)
            urgency_level = routing_decision.get("urgency_level", 1)

            # Escalation criteria
            requires_escalation = False
            escalation_reasons = []

            # Critical severity always escalates
            if severity == "critical":
                requires_escalation = True
                escalation_reasons.append("Critical severity level")

            # Low confidence requires human review
            if confidence < 0.3:
                requires_escalation = True
                escalation_reasons.append(f"Low confidence score: {confidence:.2f}")

            # High urgency escalates
            if urgency_level >= 4:
                requires_escalation = True
                escalation_reasons.append(f"High urgency level: {urgency_level}")

            # Check for security incidents
            if routing_decision.get("primary_team") == "security":
                requires_escalation = True
                escalation_reasons.append("Security incident detected")

            escalation_decision = {
                "requires_escalation": requires_escalation,
                "escalation_reasons": escalation_reasons,
                "escalation_level": self._determine_escalation_level(severity, urgency_level),
                "escalation_timeout": self._calculate_escalation_timeout(severity),
            }

            state.add_intermediate_result({
                "step": "check_escalation",
                "escalation_decision": escalation_decision,
            })

            state.metadata["escalation_decision"] = escalation_decision

            # Set approval requirement if escalation is needed
            if requires_escalation:
                state.requires_approval = True

            state.increment_step()

            self.logger.info(f"Escalation required: {requires_escalation}")
            return state

        except Exception as e:
            state.add_error(f"Escalation check failed: {e}", "escalation_error")
            raise

    async def _store_decision(self, state: AgentState) -> AgentState:
        """Store the triage decision for future learning."""
        try:
            # Compile final decision
            final_decision = {
                "incident_id": state.input_data.get("incident_id"),
                "severity": state.metadata.get("severity_decision", {}).get("severity"),
                "routing": state.metadata.get("routing_decision", {}).get("primary_team"),
                "escalation": state.metadata.get("escalation_decision", {}).get("requires_escalation"),
                "confidence": state.confidence_score,
                "timestamp": time.time(),
                "agent_version": "1.0.0",
            }

            # Store in memory if available
            if self.memory:
                try:
                    await self.memory.store(
                        content=f"Triage decision for incident: {final_decision}",
                        metadata=final_decision,
                        namespace="triage_decisions",
                        category="decision",
                        tags=["triage", final_decision.get("severity", "unknown")],
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store decision in memory: {e}")

            # Store in RAG if available for future learning
            if self.rag:
                try:
                    incident_summary = (
                        f"Incident: {state.input_data.get('title', '')} "
                        f"Severity: {final_decision.get('severity')} "
                        f"Routed to: {final_decision.get('routing')}"
                    )

                    await self.rag.add_document(
                        content=incident_summary,
                        metadata=final_decision,
                        namespace="incidents",
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store decision in RAG: {e}")

            # Set final output
            state.output_data = {
                "decision": final_decision,
                "analysis": state.metadata.get("analysis"),
                "severity_decision": state.metadata.get("severity_decision"),
                "routing_decision": state.metadata.get("routing_decision"),
                "escalation_decision": state.metadata.get("escalation_decision"),
            }

            state.status = "completed"
            state.increment_step()

            # Record metrics
            self.metrics.record_agent_execution(
                agent_name="triage",
                action="triage_decision",
                duration_seconds=(time.time() - state.created_at.timestamp()),
                status="success",
                severity=final_decision.get("severity"),
                routing=final_decision.get("routing"),
            )

            self.logger.info(f"Triage decision stored: {final_decision}")
            return state

        except Exception as e:
            state.add_error(f"Decision storage failed: {e}", "storage_error")
            raise

    # Helper methods

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.split()
        keywords = []

        # Filter out common words and extract meaningful terms
        stopwords = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        for word in words:
            word = word.strip().lower()
            if len(word) > 3 and word not in stopwords:
                keywords.append(word)

        return list(set(keywords))  # Remove duplicates

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities like IP addresses, URLs, etc."""
        import re

        entities = []

        # IP addresses
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        for match in re.finditer(ip_pattern, text):
            entities.append({"type": "ip_address", "value": match.group()})

        # URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            entities.append({"type": "url", "value": match.group()})

        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({"type": "email", "value": match.group()})

        return entities

    def _generate_severity_reasoning(self, severity: str, scores: Dict[str, int]) -> str:
        """Generate human-readable reasoning for severity decision."""
        return f"Determined as {severity} based on keyword analysis. Scores: {scores}"

    def _generate_routing_reasoning(self, team: str, scores: Dict[str, int]) -> str:
        """Generate human-readable reasoning for routing decision."""
        return f"Routed to {team} team based on content analysis. Scores: {scores}"

    def _calculate_urgency(self, severity: str) -> int:
        """Calculate urgency level (1-5) based on severity."""
        urgency_map = {
            "critical": 5,
            "high": 4,
            "medium": 2,
            "low": 1,
        }
        return urgency_map.get(severity, 2)

    def _estimate_effort(self, analysis: Dict[str, Any], severity: str) -> str:
        """Estimate effort required to resolve the incident."""
        # Simple heuristic - could be enhanced with ML
        keyword_count = len(analysis.get("keywords", []))

        if severity == "critical":
            return "high"
        elif severity == "high" or keyword_count > 10:
            return "medium"
        else:
            return "low"

    def _determine_escalation_level(self, severity: str, urgency: int) -> int:
        """Determine escalation level (1-3)."""
        if severity == "critical" or urgency >= 5:
            return 3  # Immediate escalation
        elif severity == "high" or urgency >= 4:
            return 2  # Manager escalation
        else:
            return 1  # Team lead escalation

    def _calculate_escalation_timeout(self, severity: str) -> int:
        """Calculate escalation timeout in minutes."""
        timeout_map = {
            "critical": 5,   # 5 minutes
            "high": 30,      # 30 minutes
            "medium": 120,   # 2 hours
            "low": 480,      # 8 hours
        }
        return timeout_map.get(severity, 120)