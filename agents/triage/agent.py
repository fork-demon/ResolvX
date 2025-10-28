"""
Triage agent implementation for intelligent incident routing and prioritization.

This agent analyzes incoming incidents, alerts, and requests to determine
severity levels, priority, and appropriate routing decisions.
"""

import time
from pathlib import Path
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
from core.evaluation import Guardrails, HallucinationChecker, QualityAssessor
from core.domain import load_domain_knowledge, DomainKnowledge


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
        
        # Load domain knowledge (glossary, incident types) at initialization
        self.domain_knowledge: DomainKnowledge = load_domain_knowledge()
        self.logger.info(f"Loaded domain knowledge: {len(self.domain_knowledge.entities)} entities, "
                        f"{len(self.domain_knowledge.incident_types)} incident types, "
                        f"{len(self.domain_knowledge.location_clusters)} location clusters")
        
        # Auto-load RAG backend for runbooks if not provided
        try:
            if self.rag is None:
                from core.rag.factory import create_rag_from_config
                from core.config import load_config
                
                # Load RAG config from agent config
                config = load_config()
                
                # Get RAG config (it's at top level, not under resources)
                if hasattr(config, 'rag') and config.rag:
                    rag_config = {
                        "backend": config.rag.backend,
                        "knowledge_dir": config.rag.knowledge_dir,
                        "model_name": config.rag.model_name,
                        "config": config.rag.config if hasattr(config.rag, 'config') else {}
                    }
                else:
                    # Default to FAISS if not specified
                    rag_config = {
                        "backend": "faiss_kb",
                        "knowledge_dir": "kb",
                        "model_name": "all-MiniLM-L6-v2"
                    }
                
                self.logger.info(f"Initializing RAG backend: {rag_config['backend']}")
                self.rag = create_rag_from_config(rag_config)
                
                # Load/initialize based on backend type
                if hasattr(self.rag, 'load'):
                    self.rag.load()
                    self.logger.info(f"✓ Loaded RAG backend: {rag_config['backend']}")
                elif hasattr(self.rag, 'initialize'):
                    import asyncio
                    asyncio.create_task(self.rag.initialize())
                    self.logger.info(f"✓ Initialized async RAG backend: {rag_config['backend']}")
        except Exception as e:
            self.logger.error(f"Failed to load RAG backend: {e}")
            self.rag = None
        self.tracer = get_tracer("agent.triage")
        self.metrics = get_metrics_client()
        
        # Initialize evaluation components
        self.guardrails = Guardrails(getattr(config, "guardrails", {}) if hasattr(config, "guardrails") else {})
        self.hallucination_checker = HallucinationChecker(getattr(config, "hallucination", {}) if hasattr(config, "hallucination") else {})
        self.quality_assessor = QualityAssessor(getattr(config, "quality", {}) if hasattr(config, "quality") else {})

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

        # Compile and return the graph
        return graph.compile()

    async def process(self, state: AgentState) -> Dict[str, Any]:
        """Process an incident through the triage workflow."""
        with self.tracer.start_as_current_span("triage_process") as span:
            # Set input data
            input_data = {
                "incident_id": state.input_data.get("incident_id", "unknown"),
                "tickets": state.input_data.get("tickets", []),
                "context": state.input_data.get("context", {}),
                "ticket_count": len(state.input_data.get("tickets", []))
            }
            span.set_input(input_data)
            span.set_attribute("incident_id", state.input_data.get("incident_id", "unknown"))

            try:
                # Execute the graph
                result = await self.graph.ainvoke(state.dict())

                # Extract enriched data from result
                analysis = result.get("metadata", {}).get("analysis", {})
                severity_data = result.get("metadata", {}).get("severity", {})
                routing_data = result.get("metadata", {}).get("routing", {})
                
                # Set output data with enrichment details
                output_data = {
                    "severity": severity_data.get("severity", "unknown"),
                    "routing_decision": routing_data.get("routing_decision", "unknown"),
                    "tickets_processed": len(state.input_data.get("tickets", [])),
                    "success": result.get("success", True),
                    "decisions": result.get("decisions", []),
                    "llm_used": bool(analysis.get("llm_analysis")),
                    "tools_used": analysis.get("tools_used", []),
                    "enrichment_sources": {
                        "llm": bool(analysis.get("llm_analysis")),
                        "rag": bool(analysis.get("similar_incidents")),
                        "splunk": bool(analysis.get("splunk_data")),
                        "newrelic": bool(analysis.get("newrelic_data"))
                    }
                }
                span.set_output(output_data)
                span.set_attribute("severity", severity_data.get("severity", "unknown"))
                span.set_attribute("routing_decision", routing_data.get("routing_decision", "unknown"))
                span.set_attribute("tools_used_count", len(analysis.get("tools_used", [])))

                return result

            except Exception as e:
                span.record_exception(e)
                # Set error output
                span.set_output({"success": False, "error": str(e), "incident_id": state.input_data.get("incident_id", "unknown")})
                self.logger.error(f"Triage processing failed: {e}")
                raise AgentError(f"Triage processing failed: {e}") from e

    async def _analyze_incident(self, state: AgentState) -> AgentState:
        """Analyze the incoming incident using LLM and tools."""
        try:
            # Get tickets from input
            tickets = state.input_data.get("tickets", [])
            if not tickets:
                self.logger.warning("No tickets to analyze")
                return state
            
            ticket = tickets[0]  # Process first ticket

            # Extract basic information
            title = ticket.get("subject", "")
            description = ticket.get("description", "")
            source = ticket.get("source", "unknown")
            ticket_id = ticket.get("id", "unknown")

            # Perform text analysis
            combined_text = f"{title} {description}".lower()

            # Extract keywords and entities
            keywords = self._extract_keywords(combined_text)
            entities = self._extract_entities(combined_text)

            # STEP 1: Check KB/RAG for runbook guidance FIRST
            similar_incidents = []
            runbook_guidance = ""
            if self.rag:
                with self.tracer.start_as_current_span("rag_knowledge_search") as rag_span:
                    # Set RAG search input
                    rag_input = {
                        "ticket_id": ticket_id,
                        "query": combined_text[:200],
                        "k": 3,
                        "search_type": "runbook_lookup"
                    }
                    rag_span.set_input(rag_input)
                    
                    try:
                        if hasattr(self.rag, "search"):
                            similar_incidents = self.rag.search(combined_text, k=3)
                            # Extract runbook guidance from top result
                            if similar_incidents:
                                for incident in similar_incidents[:2]:  # Top 2 results
                                    incident_text = incident.get('text', '')[:2000]  # Increased from 1000 to 2000
                                    incident_path = Path(incident.get('path', 'unknown')).name
                                    runbook_guidance += f"\n### Relevant Runbook: {incident_path}\n{incident_text}\n"
                            
                            # Set RAG search output
                            rag_output = {
                                "results_found": len(similar_incidents),
                                "top_scores": [inc.get("score", 0) for inc in similar_incidents[:3]],
                                "runbook_names": [Path(inc.get("path", "unknown")).name for inc in similar_incidents[:3]],
                                "success": True,
                                "runbook_guidance_length": len(runbook_guidance)
                            }
                            rag_span.set_output(rag_output)
                            rag_span.set_attribute("results_found", len(similar_incidents))
                            rag_span.set_attribute("runbook_guidance_chars", len(runbook_guidance))
                            
                            self.logger.info(f"RAG search found {len(similar_incidents)} similar incidents")
                            self.logger.info(f"Runbook guidance: {len(runbook_guidance)} chars extracted for LLM context")
                    except Exception as e:
                        rag_span.set_output({"success": False, "error": str(e)})
                        rag_span.record_exception(e)
                        self.logger.warning(f"RAG search failed: {e}")
            
            # STEP 2: Use LLM with runbook context to decide which tools to use
            llm_analysis = None
            tools_to_call = []
            if self.tool_registry:
                try:
                    # Get LLM client
                    from core.gateway.llm_client import LLMGatewayClient, ChatMessage
                    from core.config import load_config
                    config = load_config()
                    llm_client = LLMGatewayClient(config.gateway)
                    await llm_client.initialize()
                    
                    # Use loaded prompt or fallback to default
                    if self.has_prompt("system_prompt"):
                        system_prompt = self.get_prompt("system_prompt")
                    else:
                        # Fallback if prompt not loaded
                        system_prompt = """You are an intelligent triage agent analyzing support tickets.
Based on the ticket details and runbook guidance, recommend which diagnostic tools to use.

Available tools:
- splunk_search: Search application logs for errors, exceptions, failures
- newrelic_metrics: Query performance metrics, memory, CPU, response times
- base_prices_get: Retrieve current price data for a product
- competitor_prices_get: Get competitor pricing data
- basket_segment_get: Get basket segment classification
- sharepoint_list_files: List files/folders in SharePoint directory
- sharepoint_download_file: Download runbooks or documentation from SharePoint
- sharepoint_upload_file: Upload files to SharePoint (e.g., analysis reports)
- sharepoint_search_documents: Search SharePoint for related documentation

CRITICAL: Your response MUST be ONLY valid JSON in a code block. Do not include explanations outside the JSON.

Format:
```json
{
  "severity": "critical|high|medium|low",
  "category": "infrastructure|application|security|support",
  "tools_to_use": ["tool_name1", "tool_name2"],
  "reasoning": "Brief explanation of why these tools"
}
```

Always recommend at least 1-2 diagnostic tools based on the incident type."""
                    
                    # Build user prompt with incident analysis template if available
                    if self.has_prompt("incident_analysis"):
                        historical_context = runbook_guidance if runbook_guidance else "No historical data available"
                        user_prompt = self.get_prompt(
                            "incident_analysis",
                            INCIDENT_ID=ticket_id,
                            REPORTED_TIME=str(ticket.get("created_at", "")),
                            REPORTER=ticket.get("requester_id", "unknown"),
                            INCIDENT_DESCRIPTION=description[:500],
                            AFFECTED_SYSTEMS=source,
                            USER_IMPACT="Unknown",
                            HISTORICAL_INCIDENTS=historical_context,
                            CURRENT_SYSTEM_STATUS="Unknown",
                            RECENT_CHANGES="Unknown",
                            ACTIVE_ALERTS="None"
                        )
                        self.logger.info(f"LLM context includes {len(historical_context)} chars of runbook guidance")
                    else:
                        # Fallback user prompt
                        user_prompt = f"""Ticket ID: {ticket_id}
Subject: {title}
Description: {description[:500]}
Source: {source}

{runbook_guidance if runbook_guidance else "No runbook guidance available."}

Analyze this ticket and recommend which diagnostic tools to use."""
                    
                    messages = [
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt)
                    ]
                    
                    response = await llm_client.chat_completion(
                        messages=messages,
                        model="llama3.2",
                        temperature=0.3,
                        max_tokens=1500
                    )
                    
                    # Extract content from response (ChatCompletionResponse object)
                    llm_analysis = ""
                    if hasattr(response, "choices") and response.choices:
                        # Extract from ChatCompletionResponse
                        first_choice = response.choices[0]
                        if isinstance(first_choice, dict):
                            llm_analysis = first_choice.get("message", {}).get("content", "")
                        else:
                            llm_analysis = getattr(first_choice.message, "content", "") if hasattr(first_choice, "message") else ""
                    elif isinstance(response, dict):
                        llm_analysis = response.get("content", "")
                    else:
                        llm_analysis = str(response)
                    
                    self.logger.info(f"Extracted LLM content ({len(llm_analysis)} chars)")
                    
                    # Parse LLM response to extract tools to call
                    import json
                    import re
                    
                    # Log the actual LLM response for debugging
                    self.logger.info(f"LLM raw response (first 300 chars): {llm_analysis[:300]}")
                    
                    parsing_method = "none"
                    try:
                        # Try multiple JSON extraction patterns
                        # Pattern 1: Look for JSON in code blocks
                        json_match = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', llm_analysis, re.DOTALL)
                        if json_match:
                            parsing_method = "json_code_block"
                        else:
                            # Pattern 2: Look for standalone JSON object
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_analysis, re.DOTALL)
                            if json_match:
                                parsing_method = "json_inline"
                        
                        if json_match:
                            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
                            llm_json = json.loads(json_str)
                            tools_to_call = llm_json.get("tools_to_use", llm_json.get("tools_to_call", []))
                            self.logger.info(f"✓ LLM recommends tools via {parsing_method}: {tools_to_call}")
                        else:
                            # Fallback: keyword extraction from LLM response
                            parsing_method = "keyword_fallback"
                            if "splunk" in llm_analysis.lower():
                                tools_to_call.append("splunk_search")
                            if "newrelic" in llm_analysis.lower() or "metrics" in llm_analysis.lower():
                                tools_to_call.append("newrelic_metrics")
                            if "price" in llm_analysis.lower() and "api" in llm_analysis.lower():
                                tools_to_call.append("base_prices_get")
                            self.logger.info(f"⚠ Extracted tools from LLM keywords ({parsing_method}): {tools_to_call}")
                    except Exception as e:
                        parsing_method = "keyword_fallback_error"
                        self.logger.warning(f"Failed to parse LLM tool recommendations: {e}")
                        # Fallback to keyword-based extraction
                        if "splunk" in llm_analysis.lower():
                            tools_to_call.append("splunk_search")
                        if "newrelic" in llm_analysis.lower():
                            tools_to_call.append("newrelic_metrics")
                        self.logger.info(f"⚠ Extracted tools from LLM keywords (error fallback): {tools_to_call}")
                    
                    self.logger.info(f"LLM analysis completed for {ticket_id} (parsing: {parsing_method})")
                    
                    # EVALUATION: Apply guardrails and quality checks to LLM analysis
                    if llm_analysis:
                        try:
                            # Create evaluation spans for proper tracing
                            with self.tracer.start_as_current_span("guardrails_evaluation") as guardrail_span:
                                guardrail_span.set_input({
                                    "content": llm_analysis,
                                    "content_type": "json",
                                    "ticket_id": ticket_id,
                                    "source": source
                                })
                                
                                # Check content against guardrails
                                guardrail_result = await self.guardrails.check_content(
                                    content=llm_analysis,
                                    content_type="json",
                                    context={"ticket_id": ticket_id, "source": source}
                                )
                                
                                guardrail_span.set_output(guardrail_result.dict())
                                guardrail_span.set_attribute("evaluation_type", "guardrails")
                                guardrail_span.set_attribute("passed", guardrail_result.passed)
                                guardrail_span.set_attribute("violations", guardrail_result.violations)
                                
                                if not guardrail_result.passed:
                                    self.logger.warning(f"Guardrail violation in LLM analysis: {guardrail_result.message}")
                                    # Log violations for monitoring
                                    self.metrics.create_counter(
                                        "guardrail_violations_total",
                                        "Total guardrail violations"
                                    ).inc(1)
                            
                            # Check for hallucinations
                            with self.tracer.start_as_current_span("hallucination_evaluation") as hallucination_span:
                                hallucination_span.set_input({
                                    "response": llm_analysis,
                                    "context": {"ticket": ticket, "runbook_guidance": runbook_guidance},
                                    "sources": similar_incidents if similar_incidents else None
                                })
                                
                                hallucination_result = await self.hallucination_checker.check_response(
                                    response=llm_analysis,
                                    context={"ticket": ticket, "runbook_guidance": runbook_guidance},
                                    sources=similar_incidents if similar_incidents else None
                                )
                                
                                hallucination_span.set_output(hallucination_result.dict())
                                hallucination_span.set_attribute("evaluation_type", "hallucination")
                                hallucination_span.set_attribute("has_hallucination", hallucination_result.has_hallucination)
                                hallucination_span.set_attribute("hallucination_types", hallucination_result.hallucination_types)
                                
                                if hallucination_result.has_hallucination:
                                    self.logger.warning(f"Hallucination detected in LLM analysis: {hallucination_result.hallucination_types}")
                                    # Log hallucinations for monitoring
                                    self.metrics.create_counter(
                                        "hallucination_detections_total",
                                        "Total hallucination detections"
                                    ).inc(1)
                            
                            # Assess quality
                            with self.tracer.start_as_current_span("quality_evaluation") as quality_span:
                                quality_span.set_input({
                                    "response": llm_analysis,
                                    "context": {"ticket": ticket, "tools_recommended": tools_to_call},
                                    "expected_format": "json"
                                })
                                
                                quality_result = await self.quality_assessor.assess_response(
                                    response=llm_analysis,
                                    context={"ticket": ticket, "tools_recommended": tools_to_call},
                                    expected_format="json"
                                )
                                
                                quality_span.set_output(quality_result.dict())
                                quality_span.set_attribute("evaluation_type", "quality")
                                quality_span.set_attribute("overall_score", quality_result.overall_score)
                                quality_span.set_attribute("strengths", quality_result.strengths)
                                quality_span.set_attribute("weaknesses", quality_result.weaknesses)
                            
                            # Store evaluation results in state
                            state.evaluation_results = {
                                "guardrails": guardrail_result.dict(),
                                "hallucination": hallucination_result.dict(),
                                "quality": quality_result.dict()
                            }
                            
                            self.logger.info(f"Evaluation completed for {ticket_id}: "
                                            f"guardrails={guardrail_result.passed}, "
                                            f"hallucination={not hallucination_result.has_hallucination}, "
                                            f"quality={quality_result.overall_score}")
                            
                        except Exception as e:
                            self.logger.error(f"Evaluation failed for {ticket_id}: {e}")
                            # Continue with analysis even if evaluation fails
                    
                except Exception as e:
                    self.logger.warning(f"LLM analysis failed: {e}")
            
            # STEP 3: Execute tools recommended by LLM (DYNAMIC - works with any tool!)
            tool_results = {}
            
            if self.tool_registry and tools_to_call:
                self.logger.info(f"Executing {len(tools_to_call)} tools recommended by LLM: {tools_to_call}")
                
                for tool_name in tools_to_call:
                    try:
                        # Dynamically build tool parameters based on tool type and ticket context
                        tool_params = self._build_tool_parameters(
                            tool_name, 
                            ticket_id, 
                            title, 
                            description, 
                            combined_text
                        )
                        
                        # Execute the tool
                        result = await self.tool_registry.call_tool(tool_name, tool_params)
                        tool_results[tool_name] = result
                        
                        self.logger.info(f"✓ Tool '{tool_name}' completed successfully for {ticket_id}")
                        
                    except Exception as e:
                        self.logger.warning(f"✗ Tool '{tool_name}' failed: {e}")
                        tool_results[tool_name] = {"error": str(e), "success": False}

            # Store analysis results (DYNAMIC - works with any tool results)
            analysis = {
                "ticket_id": ticket_id,
                "keywords": keywords,
                "entities": entities,
                "llm_analysis": llm_analysis,
                "llm_tool_recommendations": tools_to_call,
                "similar_incidents": similar_incidents,
                "runbook_guidance_found": bool(runbook_guidance),
                "tool_results": tool_results,  # Dynamic: stores ALL tool results
                "text_length": len(combined_text),
                "source": source,
                "analyzed_at": time.time(),
                "tools_used": [tool for tool, result in tool_results.items() if result and not result.get("error")]
            }
            
            state.add_intermediate_result({
                "step": "analyze_incident",
                "analysis": analysis,
            })

            # Update state
            state.metadata["analysis"] = analysis
            state.increment_step()

            self.logger.info(f"Analyzed incident {ticket_id}: {len(keywords)} keywords, {len(entities)} entities, {len(analysis['tools_used'])} tools used")
            return state

        except Exception as e:
            state.add_error(f"Incident analysis failed: {e}", "analysis_error")
            raise

    async def _determine_severity(self, state: AgentState) -> AgentState:
        """Determine the severity level using LLM analysis and heuristics."""
        try:
            analysis = state.metadata.get("analysis", {})
            keywords = analysis.get("keywords", [])
            llm_analysis = analysis.get("llm_analysis", "")
            
            # Try to extract severity from LLM analysis
            severity = "medium"  # default
            if llm_analysis:
                import json
                import re
                
                # Try to parse JSON from LLM response
                try:
                    # Extract JSON block if present
                    json_match = re.search(r'\{[^{}]*\}', llm_analysis, re.DOTALL)
                    if json_match:
                        llm_json = json.loads(json_match.group(0))
                        severity = llm_json.get("severity", "medium").lower()
                        self.logger.info(f"LLM determined severity: {severity}")
                except:
                    # Fallback to keyword matching in LLM response
                    llm_lower = llm_analysis.lower()
                    if "critical" in llm_lower:
                        severity = "critical"
                    elif "high" in llm_lower:
                        severity = "high"
                    elif "low" in llm_lower:
                        severity = "low"
            
            # Calculate severity scores as fallback/validation
            severity_scores = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }

            # Score based on keywords
            for keyword in keywords:
                for sev, severity_keywords in self.severity_keywords.items():
                    if any(sk in keyword for sk in severity_keywords):
                        severity_scores[sev] += 1

            # Score based on tools that found issues
            if analysis.get("splunk_data"):
                severity_scores["high"] += 2  # Errors found in logs
            if analysis.get("newrelic_data"):
                severity_scores["medium"] += 1  # Performance metrics available

            # Score based on time sensitivity
            tickets = state.input_data.get("tickets", [])
            if tickets and tickets[0].get("priority", "").startswith("P1"):
                severity_scores["critical"] += 3
            elif tickets and tickets[0].get("priority", "").startswith("P2"):
                severity_scores["high"] += 2

            # Use heuristic if LLM didn't provide clear answer
            if severity == "medium" and sum(severity_scores.values()) > 0:
                severity = max(severity_scores, key=severity_scores.get)
            
            confidence = severity_scores.get(severity, 0) / (sum(severity_scores.values()) + 1)

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
    
    def _build_tool_parameters(self, tool_name: str, ticket_id: str, title: str, description: str, combined_text: str) -> dict:
        """
        Dynamically build parameters for any tool based on ticket context.
        This makes the system extensible - LLM can recommend ANY tool and we'll attempt to call it.
        """
        import re
        
        # Default parameters that work for most tools
        params = {}
        
        # Tool-specific parameter builders
        if tool_name == "splunk_search":
            params = {
                "query": f"index=* {ticket_id} error OR failed OR exception",
                "earliest_time": "-1h",
                "latest_time": "now"
            }
        
        elif tool_name == "newrelic_metrics":
            params = {
                "nrql": "SELECT average(duration), average(memoryUsagePercent), count(*) FROM Transaction WHERE appName LIKE '%price%' SINCE 1 hour ago",
                "account_id": "default"
            }
        
        elif tool_name == "base_prices_get":
            # Try to extract TPNB (8-digit number) from ticket
            tpnb_match = re.search(r'\b\d{8}\b', combined_text)
            if tpnb_match:
                params = {
                    "tpnb": tpnb_match.group(0),
                    "locationClusterId": "default"
                }
            else:
                params = {"tpnb": "00000000", "locationClusterId": "default"}
        
        elif tool_name == "competitor_prices_get":
            tpnb_match = re.search(r'\b\d{8}\b', combined_text)
            if tpnb_match:
                params = {"tpnb": tpnb_match.group(0), "locationClusterIds": ["default"]}
            else:
                params = {"tpnb": "00000000", "locationClusterIds": ["default"]}
        
        elif tool_name == "basket_segment_get":
            tpnb_match = re.search(r'\b\d{8}\b', combined_text)
            params = {"tpnb": tpnb_match.group(0) if tpnb_match else "00000000"}
        
        elif tool_name == "sharepoint_search_documents":
            params = {
                "query": f"{title} {description[:100]}",
                "max_results": 5
            }
        
        elif tool_name == "sharepoint_list_files":
            params = {
                "folder_path": "Shared Documents/Runbooks",
                "recursive": False
            }
        
        elif tool_name == "sharepoint_download_file":
            params = {
                "file_path": "Shared Documents/Runbooks/default.md"
            }
        
        elif tool_name == "poll_queue":
            params = {
                "queue_name": "default",
                "limit": 10
            }
        
        elif tool_name == "get_queue_stats":
            params = {}
        
        else:
            # Generic fallback for unknown tools - pass ticket context
            params = {
                "ticket_id": ticket_id,
                "query": f"{title} {description[:200]}",
                "context": "triage_analysis"
            }
            self.logger.info(f"Using generic parameters for unknown tool: {tool_name}")
        
        return params