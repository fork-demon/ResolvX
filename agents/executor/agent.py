"""
Ticket Executor Agent

Executes Supervisor decisions by making actual Zendesk API calls.
Extends BaseAgent for consistency, retry logic, and observability.
"""

from typing import Any, Dict, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from core.config import AgentConfig
from core.exceptions import AgentError
from core.graph.base import BaseAgent
from core.graph.state import AgentState
from core.observability import get_logger, get_tracer


class TicketExecutorAgent(BaseAgent):
    """
    Agent that executes Supervisor decisions on tickets.
    
    Responsibilities:
    - Add comments to tickets
    - Assign tickets to teams/users
    - Update ticket status
    - Apply tags
    - Request more information
    - Escalate to humans
    
    Features:
    - Built-in retry logic for API failures
    - State management via LangGraph
    - Consistent observability with other agents
    """
    
    def __init__(
        self,
        config: AgentConfig,
        zendesk_client=None,
        **kwargs
    ):
        """
        Initialize the executor agent.
        
        Args:
            config: Agent configuration
            zendesk_client: Zendesk API client (optional, can be mock)
            **kwargs: Additional arguments
        """
        super().__init__("executor", config, **kwargs)
        
        # Initialize tracer for observability
        self.tracer = get_tracer("agent.executor")
        
        self.zendesk = zendesk_client
        self.max_retries = getattr(config, "max_retries", 3)
        
        # Team ID mappings (from config or defaults)
        self.team_mappings = getattr(config, "team_mappings", {
            "engineering_team": "team_eng_001",
            "devops_team": "team_devops_001",
            "security_team": "team_sec_001",
            "infrastructure_team": "team_infra_001",
            "pricing_team": "team_pricing_001",
            "operations_team": "team_ops_001",
            "oncall_engineer": "user_oncall_001"
        })
        
        # Default settings
        self.default_tags = getattr(config, "default_tags", ["automated_analysis", "resolveX"])
        self.bot_signature = getattr(config, "bot_signature", "\n\n---\n🤖 Automated by TescoResolveX")
    
    def build_graph(self):
        """
        Build a simple LangGraph for ticket execution with retry logic.
        
        Graph: execute_action → validate_execution → (retry if failed or end)
        """
        graph = StateGraph(AgentState)
        
        # Nodes
        graph.add_node("execute_action", self._execute_action)
        graph.add_node("validate_execution", self._validate_execution)
        
        # Flow
        graph.set_entry_point("execute_action")
        graph.add_edge("execute_action", "validate_execution")
        
        # Conditional: retry on failure or end
        graph.add_conditional_edges(
            "validate_execution",
            self._should_retry,
            {
                "retry": "execute_action",
                "end": END
            }
        )
        
        return graph.compile()
    
    async def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process a Supervisor decision and execute it on a ticket.
        
        Args:
            state: AgentState with:
                - input_data["ticket_id"]: Zendesk ticket ID
                - input_data["decision"]: Decision dict from Supervisor
                - input_data["context"]: Additional context (optional)
        
        Returns:
            Execution result with success status
        """
        with self.tracer.start_as_current_span("executor_process") as span:
            ticket_id = state.input_data.get("ticket_id", "unknown")
            decision = state.input_data.get("decision", {})
            action = decision.get("action", "UNKNOWN")
            
            span.set_input({
                "ticket_id": ticket_id,
                "action": action,
                "assigned_to": decision.get("assigned_to")
            })
            span.set_attribute("ticket_id", ticket_id)
            span.set_attribute("action", action)
            
            try:
                # Initialize retry counter in metadata
                state.metadata["retry_count"] = 0
                state.metadata["max_retries"] = self.max_retries
                
                # Run the graph
                result_state = await self.graph.ainvoke(state)
                
                # Extract result (result_state is a dict from LangGraph)
                if isinstance(result_state, dict):
                    metadata = result_state.get("metadata", {})
                    result = metadata.get("execution_result", {})
                    retry_count = metadata.get("retry_count", 0)
                else:
                    result = result_state.metadata.get("execution_result", {})
                    retry_count = result_state.metadata.get("retry_count", 0)
                
                span.set_output(result)
                span.set_attribute("success", result.get("success", False))
                span.set_attribute("retries_used", retry_count)
                
                self.logger.info(
                    f"Executor completed for {ticket_id}: {action} - "
                    f"success={result.get('success')} (retries: {retry_count})"
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Executor failed for {ticket_id}: {e}")
                span.record_exception(e)
                raise AgentError(f"Executor processing failed: {e}") from e
    
    async def _execute_action(self, state: AgentState) -> AgentState:
        """LangGraph node: Execute the action based on decision type."""
        ticket_id = state.input_data.get("ticket_id", "unknown")
        decision = state.input_data.get("decision", {})
        context = state.input_data.get("context", {})
        action = decision.get("action", "UNKNOWN")
        
        # Increment retry count at the start of each attempt
        retry_count = state.metadata.get("retry_count", 0)
        state.metadata["retry_count"] = retry_count + 1
        
        self.logger.info(f"Executing action '{action}' on ticket {ticket_id} (attempt {state.metadata['retry_count']})")
        
        try:
            # Route to appropriate executor method
            if action == "ADD_COMMENT":
                result = await self._add_comment(ticket_id, decision, context)
            
            elif action == "ADD_COMMENT_WITH_ACTIONS":
                result = await self._add_comment_with_actions(ticket_id, decision, context)
            
            elif action == "ASSIGN_TO_TEAM":
                result = await self._assign_to_team(ticket_id, decision, context)
            
            elif action == "ESCALATE_TO_HUMAN":
                result = await self._escalate_to_human(ticket_id, decision, context)
            
            elif action == "REQUEST_MORE_INFO":
                result = await self._request_more_info(ticket_id, decision, context)
            
            elif action == "ROUTE_TO_SPECIALIST":
                result = await self._route_to_specialist(ticket_id, decision, context)
            
            else:
                self.logger.warning(f"Unknown action '{action}' for ticket {ticket_id}")
                result = {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "action": action
                }
            
            # Store result in state
            state.metadata["execution_result"] = result
            state.metadata["last_error"] = None if result.get("success") else result.get("error")
            
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            state.metadata["execution_result"] = {
                "success": False,
                "error": str(e),
                "action": action
            }
            state.metadata["last_error"] = str(e)
        
        state.increment_step()
        return state
    
    async def _validate_execution(self, state: AgentState) -> AgentState:
        """LangGraph node: Validate execution result."""
        result = state.metadata.get("execution_result", {})
        success = result.get("success", False)
        
        if success:
            self.logger.info(f"✓ Execution validated successfully")
            state.metadata["validation_passed"] = True
        else:
            error = state.metadata.get("last_error", "Unknown error")
            self.logger.warning(f"✗ Execution failed: {error}")
            state.metadata["validation_passed"] = False
        
        state.increment_step()
        return state
    
    def _should_retry(self, state: AgentState) -> str:
        """Conditional edge: Determine if we should retry."""
        validation_passed = state.metadata.get("validation_passed", False)
        retry_count = state.metadata.get("retry_count", 0)
        max_retries = state.metadata.get("max_retries", self.max_retries)
        
        if validation_passed:
            self.logger.info("✅ Execution successful, ending workflow")
            return "end"
        
        # Check if we've already exceeded max retries
        if retry_count >= max_retries:
            self.logger.error(f"❌ Max retries ({max_retries}) reached, giving up")
            return "end"
        
        self.logger.info(f"⚠️ Retrying execution (attempt {retry_count}/{max_retries})")
        return "retry"
    
    # ===== Execution Methods (same as before) =====
    
    async def _add_comment(
        self, 
        ticket_id: str, 
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a simple comment to the ticket."""
        reason = decision.get("reason", "Automated analysis complete")
        
        comment = f"**Automated Analysis**\n\n{reason}{self.bot_signature}"
        
        self.logger.info(f"Adding comment to ticket {ticket_id}")
        
        # Call Zendesk API (or mock)
        if self.zendesk:
            await self.zendesk.add_comment(ticket_id, comment, public=True)
        else:
            self.logger.debug(f"[MOCK] Would add comment to {ticket_id}: {comment[:100]}...")
        
        return {
            "success": True,
            "action": "ADD_COMMENT",
            "ticket_id": ticket_id,
            "comment_added": True
        }
    
    async def _add_comment_with_actions(
        self, 
        ticket_id: str, 
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a detailed comment with root cause and recommended actions."""
        summary = decision.get("synthesis_summary", "")
        root_cause = decision.get("root_cause", "unknown")
        confidence = decision.get("confidence", "medium")
        recommended_actions = decision.get("recommended_actions", [])
        
        # Build formatted comment
        comment_parts = ["**🤖 Automated Analysis Complete**\n"]
        
        if summary:
            comment_parts.append(f"**Summary**: {summary}\n")
        
        if root_cause and root_cause != "unknown":
            comment_parts.append(f"**Root Cause**: `{root_cause}`")
        
        comment_parts.append(f"**Confidence**: {confidence.upper()}\n")
        
        if recommended_actions:
            comment_parts.append("**Recommended Actions**:")
            for i, action in enumerate(recommended_actions[:5], 1):  # Max 5 actions
                priority = action.get("priority", "medium").upper()
                action_text = action.get("action", "")
                priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
                comment_parts.append(f"{i}. {priority_emoji} [{priority}] {action_text}")
        
        comment_parts.append(self.bot_signature)
        comment = "\n".join(comment_parts)
        
        self.logger.info(f"Adding detailed comment with actions to ticket {ticket_id}")
        
        # Call Zendesk API
        if self.zendesk:
            await self.zendesk.add_comment(ticket_id, comment, public=True)
            # Also add tags based on root cause
            if root_cause != "unknown":
                await self.zendesk.add_tags(ticket_id, [f"root_cause_{root_cause}", f"confidence_{confidence}"])
        else:
            self.logger.debug(f"[MOCK] Would add detailed comment to {ticket_id}")
        
        return {
            "success": True,
            "action": "ADD_COMMENT_WITH_ACTIONS",
            "ticket_id": ticket_id,
            "comment_added": True,
            "root_cause": root_cause,
            "actions_count": len(recommended_actions)
        }
    
    async def _assign_to_team(
        self, 
        ticket_id: str, 
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assign ticket to a team."""
        team_name = decision.get("assigned_to", "operations_team")
        reason = decision.get("reason", "Routing based on analysis")
        summary = decision.get("synthesis_summary", "")
        
        # Get team ID from mapping
        team_id = self.team_mappings.get(team_name, "team_ops_001")
        
        # Build assignment comment
        comment = f"**🎯 Ticket Assigned**\n\n"
        comment += f"**Assigned to**: {team_name.replace('_', ' ').title()}\n"
        comment += f"**Reason**: {reason}\n"
        if summary:
            comment += f"\n**Context**: {summary}\n"
        comment += self.bot_signature
        
        self.logger.info(f"Assigning ticket {ticket_id} to {team_name}")
        
        # Call Zendesk API
        if self.zendesk:
            await self.zendesk.update_ticket(ticket_id, {
                "assignee_id": team_id,
                "group_id": team_id,
                "status": "open"
            })
            await self.zendesk.add_comment(ticket_id, comment, public=False)
            await self.zendesk.add_tags(ticket_id, [f"assigned_{team_name}", "auto_routed"])
        else:
            self.logger.debug(f"[MOCK] Would assign {ticket_id} to {team_name} ({team_id})")
        
        return {
            "success": True,
            "action": "ASSIGN_TO_TEAM",
            "ticket_id": ticket_id,
            "assigned_to": team_name,
            "team_id": team_id
        }
    
    async def _escalate_to_human(
        self, 
        ticket_id: str, 
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Escalate ticket to human (oncall engineer)."""
        reason = decision.get("reason", "Escalation required")
        summary = decision.get("synthesis_summary", "")
        root_cause = decision.get("root_cause", "unknown")
        
        # Get oncall engineer ID
        oncall_id = self.team_mappings.get("oncall_engineer", "user_oncall_001")
        
        # Build escalation comment
        comment = f"**🚨 ESCALATED TO HUMAN**\n\n"
        comment += f"**Reason**: {reason}\n"
        if root_cause != "unknown":
            comment += f"**Root Cause**: `{root_cause}`\n"
        if summary:
            comment += f"\n**Analysis**: {summary}\n"
        comment += "\n⚠️ **This requires immediate human attention.**"
        comment += self.bot_signature
        
        self.logger.warning(f"Escalating ticket {ticket_id} to human ({oncall_id})")
        
        # Call Zendesk API
        if self.zendesk:
            await self.zendesk.update_ticket(ticket_id, {
                "assignee_id": oncall_id,
                "priority": "urgent",
                "status": "open"
            })
            await self.zendesk.add_comment(ticket_id, comment, public=False)
            await self.zendesk.add_tags(ticket_id, ["escalated", "human_review_required"])
            # Could also send notification (Slack, PagerDuty, etc.)
        else:
            self.logger.debug(f"[MOCK] Would escalate {ticket_id} to {oncall_id}")
        
        return {
            "success": True,
            "action": "ESCALATE_TO_HUMAN",
            "ticket_id": ticket_id,
            "escalated": True,
            "assigned_to": oncall_id
        }
    
    async def _request_more_info(
        self, 
        ticket_id: str, 
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Request more information from the ticket requester."""
        reason = decision.get("reason", "Analysis inconclusive")
        summary = decision.get("synthesis_summary", "")
        
        # Build request comment
        comment = f"**📋 Additional Information Needed**\n\n"
        comment += f"Our automated analysis was inconclusive. To help us resolve your issue faster, please provide:\n\n"
        comment += "- Specific error messages or screenshots\n"
        comment += "- Steps to reproduce the issue\n"
        comment += "- Affected product GTINs or TPNBs (if applicable)\n"
        comment += "- Timeframe when the issue occurred\n"
        
        if summary:
            comment += f"\n**Current Understanding**: {summary}\n"
        
        comment += self.bot_signature
        
        self.logger.info(f"Requesting more info for ticket {ticket_id}")
        
        # Call Zendesk API
        if self.zendesk:
            await self.zendesk.update_ticket(ticket_id, {
                "status": "pending",
                "assignee_id": context.get("requester_id") if context else None
            })
            await self.zendesk.add_comment(ticket_id, comment, public=True)
            await self.zendesk.add_tags(ticket_id, ["more_info_needed", "pending_user"])
        else:
            self.logger.debug(f"[MOCK] Would request more info for {ticket_id}")
        
        return {
            "success": True,
            "action": "REQUEST_MORE_INFO",
            "ticket_id": ticket_id,
            "status_changed": "pending"
        }
    
    async def _route_to_specialist(
        self, 
        ticket_id: str, 
        decision: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route to a specialist team."""
        # Similar to _assign_to_team but with specialist flag
        return await self._assign_to_team(ticket_id, decision, context)

