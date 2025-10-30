"""
Triage agent implementation for intelligent incident routing and prioritization.

This agent analyzes incoming incidents, alerts, and requests to determine
severity levels, priority, and appropriate routing decisions.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from core.config import AgentConfig
from core.exceptions import AgentError
from core.graph.base import BaseAgent
from core.graph.state import AgentState
from core.gateway.tool_registry import ToolRegistry
from core.gateway.llm_client import LLMGatewayClient, ChatMessage
from core.gateway.parameter_mapper import ParameterMapper
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
        
        # Initialize parameter mapper for automatic tool parameter formation (deterministic, no LLM)
        self.parameter_mapper = ParameterMapper()
        
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
        """Build the LangGraph for the triage agent - simplified to analysis only."""
        graph = StateGraph(AgentState)

        # Simplified: Just analyze and store - supervisor handles routing/escalation
        graph.add_node("analyze_incident", self._analyze_incident)
        graph.add_node("store_decision", self._store_decision)

        # Simple linear flow
        graph.add_edge("analyze_incident", "store_decision")
        graph.add_edge("store_decision", END)

        graph.set_entry_point("analyze_incident")
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

                # Extract analysis data - simplified (supervisor handles routing/escalation)
                analysis = result.get("metadata", {}).get("analysis", {})
                
                # Simplified output - just analysis results
                output_data = {
                    "tickets_processed": len(state.input_data.get("tickets", [])),
                    "success": result.get("success", True),
                    "analysis": analysis,
                    "tools_used": analysis.get("tools_used", []),
                    "entities_extracted": analysis.get("entities", {}),
                    "execution_plan": analysis.get("llm_analysis", {}).get("plan", {}) if isinstance(analysis.get("llm_analysis"), dict) else {}
                }
                span.set_output(output_data)
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
                        "k": 1,
                        "search_type": "runbook_lookup"
                    }
                    rag_span.set_input(rag_input)
                    
                    try:
                        if hasattr(self.rag, "search"):
                            similar_incidents = self.rag.search(combined_text, k=1)
                            # Extract runbook guidance from most relevant result
                            if similar_incidents:
                                incident = similar_incidents[0]  # Single most relevant result
                                incident_text = incident.get('text', '')[:2000]
                                incident_path = Path(incident.get('path', 'unknown')).name
                                runbook_guidance += f"\n### Relevant Runbook: {incident_path}\n{incident_text}\n"
                            
                            # Set RAG search output
                            rag_output = {
                                "results_found": len(similar_incidents),
                                "top_score": similar_incidents[0].get("score", 0) if similar_incidents else 0,
                                "runbook_name": Path(similar_incidents[0].get("path", "unknown")).name if similar_incidents else "none",
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

            # STEP 2: Chain-of-Thought Analysis with LLM
            # Break down into smaller, focused steps for better reasoning
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
                    
                    # ========================================
                    # CHAIN-OF-THOUGHT APPROACH
                    # ========================================
                    
                    # STEP 2.1: Entity Extraction - CLEAN (just extract entities, no tools/incident types)
                    self.logger.info(f"CoT Step 1: Extracting entities from {ticket_id}")
                    
                    with self.tracer.start_as_current_span("cot_entity_extraction") as entity_span:
                        entity_span.set_input({
                            "ticket_id": ticket_id,
                            "title": title,
                            "description": description[:500]
                        })
                        entities_result = await self._cot_extract_entities(
                            llm_client, ticket_id, title, description
                        )
                        entity_span.set_output(entities_result)
                        entity_span.set_attribute("step", "entity_extraction")
                        entity_span.set_attribute("incident_type", entities_result.get("incident_type", "unknown"))
                    self.logger.info(f"Extracted entities: {entities_result}")
                    
                    # STEP 2.2: RAG Lookup with extracted entities
                    self.logger.info(f"CoT Step 2: Fetching most relevant KB article")
                    # Enhance search query with extracted incident type
                    incident_type = entities_result.get("incident_type", "")
                    enhanced_query = f"{incident_type} {combined_text[:200]}" if incident_type else combined_text[:200]
                    
                    if not similar_incidents and self.rag and hasattr(self.rag, "search"):
                        similar_incidents = self.rag.search(enhanced_query, k=1)
                        if similar_incidents:
                            # Fetch only the single most relevant runbook
                            incident = similar_incidents[0]
                            incident_text = incident.get('text', '')[:2000]
                            incident_path = Path(incident.get('path', 'unknown')).name
                            runbook_guidance += f"\n### KB Article: {incident_path}\n{incident_text}\n"
                            self.logger.info(f"Found most relevant KB article: {incident_path}")
                    
                    # STEP 2.3: Create Execution Plan
                    self.logger.info(f"CoT Step 3: Creating execution plan")
                    
                    # Get all available tools dynamically for plan creation
                    plan_tools = []
                    if self.tool_registry:
                        all_tools = self.tool_registry.list_tools()
                        for tool_name in all_tools:
                            tool_def = self.tool_registry.get_tool_info(tool_name)
                            if tool_def:
                                params = []
                                if tool_def.input_schema and isinstance(tool_def.input_schema, dict):
                                    properties = tool_def.input_schema.get("properties", {})
                                    params = list(properties.keys())[:5]
                                
                                plan_tools.append({
                                    "name": tool_name,
                                    "description": tool_def.description or "No description",
                                    "parameters": params
                                })
                    
                    with self.tracer.start_as_current_span("cot_plan_creation") as plan_span:
                        plan_span.set_input({
                            "ticket_id": ticket_id,
                            "ticket_title": title,
                            "ticket_description": description[:200],
                            "entities": entities_result,
                            "kb_articles_length": len(runbook_guidance),
                            "available_tools_count": len(plan_tools)
                        })
                        execution_plan = await self._cot_create_plan(
                            llm_client, ticket_id, title, description, entities_result, runbook_guidance, plan_tools
                        )
                        plan_span.set_output(execution_plan)
                        plan_span.set_attribute("step", "plan_creation")
                        plan_span.set_attribute("plan_type", execution_plan.get("plan_type", "unknown"))
                        plan_span.set_attribute("steps_count", len(execution_plan.get("steps", [])))
                    self.logger.info(f"Execution plan: {execution_plan.get('plan_type')} with {len(execution_plan.get('steps', []))} steps")
                    
                    # STEP 2.4: Form tool payloads (NEW CoT STEP)
                    tools_from_plan = []
                    for step in execution_plan.get("steps", []):
                        tool_name = step.get("tool")
                        if tool_name and tool_name != "null":
                            tools_from_plan.append(tool_name)
                    
                    self.logger.info(f"Step 4: Automatically mapping parameters for {len(tools_from_plan)} tools using JSON Schema (no LLM)")
                    
                    with self.tracer.start_as_current_span("auto_parameter_mapping") as payload_span:
                        payload_span.set_input({
                            "ticket_id": ticket_id,
                            "tools_count": len(tools_from_plan),
                            "tools": tools_from_plan
                        })
                        
                        tool_payloads = []
                        ticket_context = {"ticket_id": ticket_id, "title": title, "description": description}
                        
                        for tool_name in tools_from_plan:
                            try:
                                tool_def = self.tool_registry.get_tool_info(tool_name)
                                if not tool_def or not tool_def.input_schema:
                                    self.logger.warning(f"Tool {tool_name} has no schema")
                                    continue
                                
                                parameters = self.parameter_mapper.map_parameters(
                                    tool_name=tool_name,
                                    tool_schema=tool_def.input_schema,
                                    entities=entities_result,
                                    ticket_context=ticket_context
                                )
                                
                                tool_payloads.append({
                                    "step": len(tool_payloads) + 1,
                                    "tool": tool_name,
                                    "parameters": parameters
                                })
                                
                                self.logger.info(f"✓ Mapped {tool_name}: {list(parameters.keys())}")
                            except Exception as e:
                                self.logger.error(f"Failed to map parameters for {tool_name}: {e}")
                                tool_payloads.append({
                                    "step": len(tool_payloads) + 1,
                                    "tool": tool_name,
                                    "parameters": {}
                                })
                        
                        payloads_result = {"tool_payloads": tool_payloads}
                        payload_span.set_output(payloads_result)
                        payload_span.set_attribute("step", "auto_parameter_mapping")
                        payload_span.set_attribute("payloads_count", len(tool_payloads))
                    self.logger.info(f"✓ Automatically formed payloads for {len(tool_payloads)} tools (schema-based)")
                    
                    # STEP 2.5: Execute the plan with formed payloads
                    self.logger.info(f"CoT Step 5: Executing tools with formed payloads")
                    tools_to_call = tools_from_plan  # Use tools from plan
                    
                    # Store the execution plan and payloads in llm_analysis for downstream use
                    llm_analysis = {
                        "entities": entities_result,
                        "plan": execution_plan,
                        "tools_recommended": tools_to_call,
                        "payloads": payloads_result,  # Include formed payloads
                        "routing": execution_plan.get("routing", {})
                    }
                    
                    # === EVALUATION: Apply guardrails and quality checks to execution plan ===
                    plan_summary = execution_plan.get("summary", "")
                    if plan_summary:
                        try:
                            # Guardrails check
                            with self.tracer.start_as_current_span("guardrails_evaluation") as guardrail_span:
                                guardrail_span.set_input({
                                    "plan": execution_plan,
                                    "ticket_id": ticket_id
                                })
                                
                                guardrail_result = await self.guardrails.check_content(
                                    content=str(execution_plan),
                                    content_type="json",
                                    context={"ticket_id": ticket_id, "source": source}
                                )
                                
                                guardrail_span.set_output(guardrail_result.dict())
                                guardrail_span.set_attribute("evaluation_type", "guardrails")
                                guardrail_span.set_attribute("passed", guardrail_result.passed)
                            
                            # Quality assessment
                            with self.tracer.start_as_current_span("quality_evaluation") as quality_span:
                                quality_span.set_input({
                                    "plan": execution_plan,
                                    "tools_count": len(tools_to_call)
                                })
                                
                                quality_result = await self.quality_assessor.assess_response(
                                    response=str(execution_plan),
                                    context={"ticket": ticket, "tools_recommended": tools_to_call},
                                    expected_format="json"
                                )
                                
                                quality_span.set_output(quality_result.dict())
                                quality_span.set_attribute("evaluation_type", "quality")
                                quality_span.set_attribute("overall_score", quality_result.overall_score)
                            
                            self.logger.info(f"Evaluation completed: guardrails={guardrail_result.passed}, quality={quality_result.overall_score}")
                            
                        except Exception as e:
                            self.logger.warning(f"Evaluation failed: {e}")
                    
## === CHAIN-OF-THOUGHT COMPLETE ===
                    # Chain-of-thought analysis is complete with:
                    # - Step 1: Entities extracted (clean, no tools/incident types)
                    # - Step 2: KB articles retrieved via RAG
                    # - Step 3: Execution plan created (just tool names, no parameters)
                    # - Step 4: Tool payloads formed (based on tool schemas + entities)
                    # - Step 5: Tools executed with formed payloads
                    
                except Exception as e:
                    self.logger.error(f"Chain-of-Thought analysis failed: {e}")
                    # Fallback: empty analysis
                    llm_analysis = {"error": str(e)}
                    tools_to_call = []
            
            # STEP 3: Execute tools using formed payloads from CoT step
            tool_results = {}
            
            # Get payloads from CoT step
            tool_payloads_map = {}
            if isinstance(llm_analysis, dict) and "payloads" in llm_analysis:
                for payload in llm_analysis.get("payloads", {}).get("tool_payloads", []):
                    tool_name = payload.get("tool")
                    if tool_name:
                        tool_payloads_map[tool_name] = payload.get("parameters", {})
            
            if self.tool_registry and tools_to_call:
                self.logger.info(f"Executing {len(tools_to_call)} tools with formed payloads: {tools_to_call}")
                
                for tool_name in tools_to_call:
                    try:
                        # Use payload from CoT step, fallback to dynamic building if not available
                        if tool_name in tool_payloads_map:
                            tool_params = tool_payloads_map[tool_name]
                            self.logger.info(f"Using CoT-formed payload for {tool_name}: {tool_params}")
                        else:
                            # Fallback: build parameters dynamically (backward compatibility)
                            entities_for_tools = llm_analysis.get("entities", {}) if isinstance(llm_analysis, dict) else {}
                            tool_params = self._build_tool_parameters(
                                tool_name, ticket_id, title, description, combined_text, entities_for_tools
                            )
                            self.logger.warning(f"Fallback: Dynamically built parameters for {tool_name}")
                        
                        # Execute the tool
                        result = await self.tool_registry.call_tool(tool_name, tool_params)
                        tool_results[tool_name] = result
                        
                        self.logger.info(f"✓ Tool '{tool_name}' completed successfully for {ticket_id}")
                        
                    except Exception as e:
                        self.logger.warning(f"✗ Tool '{tool_name}' failed: {e}")
                        tool_results[tool_name] = {"error": str(e), "success": False}
            
            # STEP 2.6: Synthesize Results - Final CoT step matching Claude/ChatGPT pattern
            self.logger.info(f"CoT Step 6: Synthesizing results into actionable recommendations")
            synthesis_result = {}
            
            if tool_results:  # Only synthesize if we have tool results
                with self.tracer.start_as_current_span("cot_synthesis") as synthesis_span:
                    synthesis_span.set_input({
                        "ticket_id": ticket_id,
                        "tools_executed": list(tool_results.keys()),
                        "results_count": len(tool_results)
                    })
                    
                    synthesis_result = await self._cot_synthesize_results(
                        llm_client=llm_client,
                        ticket_id=ticket_id,
                        title=title,
                        description=description,
                        entities=entities_result,
                        execution_plan=execution_plan,
                        runbook_guidance=runbook_guidance,
                        tool_results=tool_results
                    )
                    
                    synthesis_span.set_output(synthesis_result)
                    synthesis_span.set_attribute("step", "synthesis")
                    synthesis_span.set_attribute("root_cause", synthesis_result.get("root_cause", "unknown"))
                    synthesis_span.set_attribute("escalation_needed", synthesis_result.get("escalation_needed", False))
                    synthesis_span.set_attribute("confidence", synthesis_result.get("confidence", "unknown"))
                
                self.logger.info(f"✓ Synthesis complete: {synthesis_result.get('summary', '')[:100]}")
            else:
                self.logger.warning("No tool results to synthesize, skipping synthesis step")

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
                "synthesis": synthesis_result,  # NEW: Human-readable synthesis with recommendations
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

    # REMOVED: _determine_severity - supervisor handles severity/routing now
    def _determine_severity_removed(self, state: AgentState) -> AgentState:
        """Determine the severity level using LLM analysis and heuristics."""
        try:
            analysis = state.metadata.get("analysis", {})
            keywords = analysis.get("keywords", [])
            llm_analysis = analysis.get("llm_analysis", "")
            
            # Try to extract severity from LLM analysis
            severity = "medium"  # default
            if llm_analysis:
                # Handle both dict (CoT) and string (legacy) formats
                if isinstance(llm_analysis, dict):
                    # Extract from CoT execution plan
                    plan = llm_analysis.get("plan", {})
                    # For now, use heuristic - will extract severity from plan in future
                    self.logger.info(f"Using heuristic severity for CoT plan")
                elif isinstance(llm_analysis, str):
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
        """Store the analysis results - simplified (supervisor handles routing/escalation)."""
        try:
            analysis = state.metadata.get("analysis", {})
            
            # Simple result - just analysis data
            triage_result = {
                "incident_id": state.input_data.get("incident_id"),
                "entities": analysis.get("entities", {}),
                "tools_used": analysis.get("tools_used", []),
                "execution_plan": analysis.get("llm_analysis", {}).get("plan", {}) if isinstance(analysis.get("llm_analysis"), dict) else {},
                "timestamp": time.time(),
            }

            # Store in memory if available
            if self.memory:
                try:
                    await self.memory.store(
                        content=f"Triage analysis for incident: {state.input_data.get('incident_id')}",
                        metadata=triage_result,
                        namespace="triage_analysis",
                        category="analysis",
                        tags=["triage"],
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to store analysis in memory: {e}")

            # Set final output - simplified
            state.output_data = {
                "analysis": analysis,
                "triage_result": triage_result,
            }

            state.status = "completed"
            state.increment_step()

            self.logger.info(f"Triage analysis stored for incident {state.input_data.get('incident_id')}")
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
    
    def _build_tool_parameters(self, tool_name: str, ticket_id: str, title: str, description: str, combined_text: str, entities: dict = None) -> dict:
        """
        Dynamically build parameters for any tool based on ticket context.
        Uses extracted entities when available (from CoT Step 1), falls back to regex.
        
        Args:
            entities: Extracted entities {gtin, tpnb, key_terms, incident_type, locations}
        """
        import re
        
        entities = entities or {}
        params = {}
        
        # Tool-specific parameter builders
        if tool_name == "splunk_search":
            # Use key_terms from entity extraction for better query
            key_terms = entities.get("key_terms", [])
            incident_type = entities.get("incident_type", "")
            
            if key_terms:
                query = " OR ".join(key_terms[:3])
                self.logger.info(f"Splunk: using key_terms [{query}]")
            elif incident_type:
                query = incident_type.replace("_", " ")
                self.logger.info(f"Splunk: using incident_type [{query}]")
            else:
                query = title[:50]
                self.logger.warning(f"Splunk: fallback to title [{query}]")
            
            params = {
                "query": query,
                "time_range": "-2h",
                "index": "price-advisory-*"
            }
        
        elif tool_name == "splunk_search_old_logic":
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
            # Use extracted TPNB/GTIN if available
            gtin = entities.get("gtin")
            tpnb = entities.get("tpnb")
            
            if gtin:
                params = {"gtin": gtin, "location": "online"}
                self.logger.info(f"base_prices_get: using extracted GTIN [{gtin}]")
            elif tpnb:
                params = {"tpnb": tpnb, "locationClusterId": "default"}
                self.logger.info(f"base_prices_get: using extracted TPNB [{tpnb}]")
            else:
                # Fallback: regex
                tpnb_match = re.search(r'\b\d{8}\b', combined_text)
                tpnb = tpnb_match.group(0) if tpnb_match else "00000000"
                params = {"tpnb": tpnb, "locationClusterId": "default"}
                self.logger.warning(f"base_prices_get: fallback regex TPNB [{tpnb}]")
        
        elif tool_name == "competitor_prices_get":
            # Use extracted TPNB
            tpnb = entities.get("tpnb")
            if not tpnb:
                tpnb_match = re.search(r'\b\d{8}\b', combined_text)
                tpnb = tpnb_match.group(0) if tpnb_match else "00000000"
                self.logger.warning(f"competitor_prices_get: fallback regex TPNB [{tpnb}]")
            else:
                self.logger.info(f"competitor_prices_get: using extracted TPNB [{tpnb}]")
            params = {"tpnb": tpnb, "locationClusterIds": ["default"]}
        
        elif tool_name == "basket_segment_get":
            # Use extracted TPNB
            tpnb = entities.get("tpnb")
            if not tpnb:
                tpnb_match = re.search(r'\b\d{8}\b', combined_text)
                tpnb = tpnb_match.group(0) if tpnb_match else "00000000"
                self.logger.warning(f"basket_segment_get: fallback regex TPNB [{tpnb}]")
            else:
                self.logger.info(f"basket_segment_get: using extracted TPNB [{tpnb}]")
            params = {"tpnb": tpnb}
        
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
    
    async def _cot_extract_entities(self, llm_client, ticket_id: str, title: str, description: str) -> dict:
        """
        Chain-of-Thought Step 1: Extract entities from ticket - CLEAN version
        Just extracts entities (GTIN, TPNB, locations, incident_type, key_terms)
        No tools, no incident type suggestions - just pure entity extraction
        """
        try:
            # Load entity extraction prompt - clean, no tools or incident types
            if self.has_prompt("step1_entity_extraction"):
                extraction_prompt = self.get_prompt(
                    "step1_entity_extraction",
                    TICKET_ID=ticket_id,
                    DESCRIPTION=f"{title}\n\n{description[:500]}"
                )
            else:
                # Fallback prompt
                extraction_prompt = f"""Analyze this incident and extract information:

Ticket: {ticket_id}
Title: {title}
Description: {description[:500]}

Extract ACTUAL values from the description above:

1. GTIN: 14-digit number starting with 0 or 5 (or null if not found)
2. TPNB: 9-digit number (or null if not found)
3. Locations: Store or location names mentioned (or empty array)
4. Incident type: Classify based on keywords
   - "file", "drop", "CSV" → file_processing_failed
   - "price missing", "NOF" → price_not_found
   - "wrong price" → incorrect_price
   - "product", "inactive" → product_issue
   - Otherwise → unknown
5. Key terms: 3-5 important words FROM the description for KB search
6. Classification reason: Why you chose this type

Return ONLY valid JSON with these exact fields:
{{
  "gtin": null,
  "tpnb": null,
  "locations": [],
  "incident_type": "...",
  "key_terms": [...],
  "classification_reason": "..."
}}

CRITICAL: Extract REAL values from the ticket, not examples!"""
            
            messages = [
                ChatMessage(role="system", content="You are a data extraction AI. Output only valid JSON."),
                ChatMessage(role="user", content=extraction_prompt)
            ]
            
            response = await llm_client.chat_completion(
                messages=messages,
                model="llama3.2",
                temperature=0.1,
                max_tokens=400
            )
            
            # Parse JSON response - handle both dict and object formats
            if hasattr(response, "choices") and response.choices:
                # ChatCompletionResponse object
                first_choice = response.choices[0]
                if isinstance(first_choice, dict):
                    content = first_choice.get("message", {}).get("content", "")
                else:
                    content = getattr(first_choice.message, "content", "") if hasattr(first_choice, "message") else ""
            elif isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                content = str(response)
            
            self.logger.info(f"Entity extraction LLM response ({len(content)} chars): {content[:200]}")
            entities = self._parse_json_response(content)
            
            if entities:
                # Log classification reason if provided
                if "classification_reason" in entities:
                    self.logger.info(f"Incident classified as '{entities.get('incident_type')}': {entities.get('classification_reason')}")
                return entities
            else:
                return {
                    "gtin": None,
                    "tpnb": None,
                    "locations": [],
                    "incident_type": "unknown",
                    "key_terms": [],
                    "classification_reason": "Failed to extract from LLM response"
                }
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return {
                "gtin": None,
                "tpnb": None,
                "locations": [],
                "incident_type": "unknown",
                "key_terms": []
            }
    
    async def _cot_create_plan(self, llm_client, ticket_id: str, title: str, description: str, entities: dict, kb_articles: str, available_tools: list = None) -> dict:
        """
        Chain-of-Thought Step 3: Create execution plan based on KB articles, ticket details, and extracted entities
        
        Args:
            available_tools: List of all available tools in the system (dynamic, scalable)
        """
        available_tools = available_tools or []
        
        try:
            # Format tools list for prompt (scalable - no hardcoding)
            tools_list_text = ""
            if available_tools:
                tools_list_text = "\n## Available Tools\n\n"
                tools_list_text += "The following tools are available. Match KB article recommendations to tool names:\n\n"
                for tool in available_tools[:30]:  # Limit to 30 to avoid token bloat
                    tools_list_text += f"- **{tool['name']}**: {tool.get('description', 'No description')}\n"
                    if tool.get('parameters'):
                        tools_list_text += f"  Parameters: {', '.join(tool['parameters'][:5])}\n"
            
            # Load plan creation prompt with actual values (not placeholders)
            if self.has_prompt("step2_create_plan"):
                plan_prompt = self.get_prompt(
                    "step2_create_plan",
                    TICKET_ID=ticket_id,
                    TICKET_TITLE=title,
                    TICKET_DESCRIPTION=description[:500],
                    INCIDENT_TYPE=entities.get("incident_type", "unknown"),
                    ENTITIES=entities,  # Pass dict directly, let template format it
                    KB_ARTICLES=kb_articles if kb_articles else "No KB articles found",
                    AVAILABLE_TOOLS=tools_list_text
                )
            else:
                # Fallback prompt
                plan_prompt = f"""Create execution plan for this incident:

Type: {entities.get('incident_type')}
Entities: {entities}

KB Articles:
{kb_articles if kb_articles else "No KB articles found"}

Create a plan with specific steps. Output JSON only:
{{
  "plan_type": "diagnostic|enrichment",
  "steps": [
    {{
      "step": 1,
      "action": "check_price_api|check_splunk|check_sharepoint|add_comment",
      "tool": "base_prices_get|splunk_search|sharepoint_list_files|null",
      "parameters": {{}},
      "reason": "why"
    }}
  ],
  "routing": {{
    "forward_to_team": "Quote Team|Adaptor Team|Product Team|null",
    "reason": "why"
  }},
  "summary": "plan summary"
}}

Rules:
- Max 3 steps
- Use KB recommendations
- If no KB match → add_comment
- Route only if KB specifies"""
            
            messages = [
                ChatMessage(role="system", content="You are a planning AI. Create execution plans from KB articles. Output only valid JSON."),
                ChatMessage(role="user", content=plan_prompt)
            ]
            
            response = await llm_client.chat_completion(
                messages=messages,
                model="llama3.2",
                temperature=0.2,
                max_tokens=600
            )
            
            # Parse JSON response - handle both dict and object formats
            if hasattr(response, "choices") and response.choices:
                # ChatCompletionResponse object
                first_choice = response.choices[0]
                if isinstance(first_choice, dict):
                    content = first_choice.get("message", {}).get("content", "")
                else:
                    content = getattr(first_choice.message, "content", "") if hasattr(first_choice, "message") else ""
            elif isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                content = str(response)
            
            self.logger.info(f"Plan creation LLM response ({len(content)} chars): {content[:200]}")
            plan = self._parse_json_response(content)
            
            return plan if plan else {
                "plan_type": "enrichment",
                "steps": [{
                    "step": 1,
                    "action": "add_comment",
                    "tool": None,
                    "parameters": {},
                    "reason": "No clear plan from KB, need manual investigation"
                }],
                "routing": {"forward_to_team": None, "reason": "No routing specified"},
                "summary": "Manual investigation required"
            }
            
        except Exception as e:
            self.logger.error(f"Plan creation failed: {e}")
            return {
                "plan_type": "enrichment",
                "steps": [{
                    "step": 1,
                    "action": "add_comment",
                    "tool": None,
                    "parameters": {},
                    "reason": f"Error creating plan: {e}"
                }],
                "routing": {"forward_to_team": None, "reason": "Error occurred"},
                "summary": "Error in plan creation"
            }
    
    # REMOVED: _cot_form_payloads() method - replaced with automatic parameter mapping
    # Parameter mapping is now handled by ParameterMapper class (deterministic, schema-based)
    # This avoids LLM hallucination risks and follows industry best practices
    
    async def _cot_synthesize_results(
        self,
        llm_client,
        ticket_id: str,
        title: str,
        description: str,
        entities: dict,
        execution_plan: dict,
        runbook_guidance: str,
        tool_results: dict
    ) -> dict:
        """
        CoT Step 6: Synthesize tool results into actionable recommendations.
        
        This is the final step that matches Claude/ChatGPT's synthesis pattern.
        Takes all diagnostic data and produces human-readable summary with
        actionable recommendations.
        """
        try:
            # Build synthesis prompt
            synthesis_prompt_template = self.prompts.get("step3_synthesize_results", "")
            if not synthesis_prompt_template:
                self.logger.warning("Synthesis prompt not loaded, skipping synthesis")
                return {
                    "summary": "Diagnostic tools executed successfully",
                    "root_cause": "unknown",
                    "recommended_actions": [],
                    "escalation_needed": False,
                    "confidence": "low"
                }
            
            # Format tool results for readability (keep concise for context window)
            formatted_results = []
            for tool_name, result in tool_results.items():
                if isinstance(result, dict):
                    # Extract only key information from tool results
                    if "data" in result:
                        result_summary = f"Success: {len(result.get('data', []))} items found"
                    elif "error" in result:
                        result_summary = f"Error: {result.get('error', 'Unknown error')}"
                    elif "success" in result:
                        result_summary = "Success" if result["success"] else "Failed"
                    else:
                        result_summary = json.dumps(result, indent=2)[:200]
                else:
                    result_summary = str(result)[:200]
                formatted_results.append(f"**{tool_name}**: {result_summary}")
            
            tool_results_text = "\n".join(formatted_results)
            
            # Keep entities and plan concise
            entities_summary = {
                "incident_type": entities.get("incident_type", "unknown"),
                "key_terms": entities.get("key_terms", [])[:3],  # Top 3 terms only
                "gtin": entities.get("gtin"),
                "tpnb": entities.get("tpnb")
            }
            
            plan_summary = {
                "plan_type": execution_plan.get("plan_type", "unknown"),
                "steps": [{"tool": s.get("tool"), "reason": s.get("reason", "")[:50]} 
                         for s in execution_plan.get("steps", [])[:3]]  # Max 3 steps
            }
            
            # Render synthesis prompt (optimized for context window)
            synthesis_input = synthesis_prompt_template.render(
                TICKET_ID=ticket_id,
                TITLE=title,
                DESCRIPTION=description[:300],  # Reduced from 500
                ENTITIES=json.dumps(entities_summary, indent=2),
                EXECUTION_PLAN=json.dumps(plan_summary, indent=2),
                RUNBOOK_GUIDANCE=runbook_guidance[:600],  # Reduced from 1000
                TOOL_RESULTS=tool_results_text
            )
            
            # Call LLM for synthesis
            messages = [
                ChatMessage(role="system", content="You are a diagnostic synthesis AI. Analyze tool results and create actionable summaries. Output only valid JSON."),
                ChatMessage(role="user", content=synthesis_input)
            ]
            
            self.logger.debug(f"Calling LLM for synthesis with {len(synthesis_input)} chars")
            
            response = await llm_client.chat_completion(
                model="llama3.2",  # Fixed: use model name, not URL
                messages=messages,
                temperature=0.2,  # Lower temp for more focused synthesis
                max_tokens=800  # Reduced from 1000 for faster response
            )
            
            # Extract synthesis result (handle both dict and object formats)
            if isinstance(response, dict):
                # Dict format from LLM gateway
                choices = response.get('choices', [])
                if choices:
                    synthesis_text = choices[0].get('message', {}).get('content', '')
                else:
                    synthesis_text = response.get('content', '')
            elif hasattr(response, 'choices') and response.choices:
                # Object format (ChatCompletionResponse)
                first_choice = response.choices[0]
                if isinstance(first_choice, dict):
                    synthesis_text = first_choice.get('message', {}).get('content', '')
                else:
                    synthesis_text = getattr(first_choice.message, 'content', '') if hasattr(first_choice, 'message') else ''
            else:
                synthesis_text = str(response)
            
            self.logger.info(f"Synthesis LLM response ({len(synthesis_text)} chars): {synthesis_text[:200]}")
            
            # Parse JSON response
            synthesis_result = self._parse_json_response(synthesis_text)
            
            if not synthesis_result:
                self.logger.warning("Failed to parse synthesis JSON, using fallback")
                return {
                    "summary": synthesis_text[:150] if synthesis_text else "Synthesis failed",
                    "root_cause": "unknown",
                    "recommended_actions": [],
                    "escalation_needed": False,
                    "confidence": "low"
                }
            
            # Validate and set defaults for required fields
            if "summary" not in synthesis_result:
                synthesis_result["summary"] = "Diagnostic tools executed"
            if "root_cause" not in synthesis_result:
                synthesis_result["root_cause"] = "unknown"
            if "recommended_actions" not in synthesis_result:
                synthesis_result["recommended_actions"] = []
            if "escalation_needed" not in synthesis_result:
                synthesis_result["escalation_needed"] = False
            if "confidence" not in synthesis_result:
                synthesis_result["confidence"] = "medium"
            
            self.logger.info(f"✓ Synthesis: {synthesis_result.get('summary', '')[:80]} (confidence: {synthesis_result.get('confidence')})")
            
            return synthesis_result
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            return {
                "summary": f"Error during synthesis: {str(e)[:100]}",
                "root_cause": "unknown",
                "recommended_actions": [],
                "escalation_needed": False,
                "confidence": "low",
                "error": str(e)
            }
    
    def _parse_json_response(self, content: str) -> Optional[dict]:
        """
        Parse JSON from LLM response, handling code blocks and inline JSON.
        """
        import json
        import re
        
        if not content:
            return None
        
        try:
            # Try multiple JSON extraction patterns
            # Pattern 1: Look for JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*(\{[^`]+\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Pattern 2: Look for standalone JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Pattern 3: Try parsing the whole content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing JSON: {e}")
            return None