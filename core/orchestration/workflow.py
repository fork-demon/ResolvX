"""
Complete Workflow Orchestrator

Orchestrates the full ticket automation flow:
Poller → Memory → Triage → Supervisor → Executor
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from core.config import AgentConfig
from core.observability import get_logger, get_tracer
from core.graph.state import AgentState


class TicketWorkflowOrchestrator:
    """
    Orchestrates the complete ticket automation workflow.
    
    Flow:
    1. Poller fetches new tickets
    2. Memory checks for duplicates
    3. Triage analyzes and synthesizes
    4. Supervisor makes decision
    5. Executor takes action
    """
    
    def __init__(
        self,
        poller_agent,
        memory_agent,
        triage_agent,
        supervisor_agent,
        executor_agent,
        tracer=None
    ):
        """
        Initialize workflow orchestrator.
        
        Args:
            poller_agent: Poller agent instance
            memory_agent: Memory agent instance
            triage_agent: Triage agent instance
            supervisor_agent: Supervisor agent instance
            executor_agent: Executor agent instance
            tracer: Optional tracer for observability
        """
        self.poller = poller_agent
        self.memory = memory_agent
        self.triage = triage_agent
        self.supervisor = supervisor_agent
        self.executor = executor_agent
        
        self.tracer = tracer or get_tracer()
        self.logger = get_logger("workflow_orchestrator")
    
    async def process_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single ticket through the complete workflow.
        
        Args:
            ticket: Ticket data from Poller
        
        Returns:
            Workflow result with outcomes from each agent
        """
        ticket_id = ticket.get('id', 'unknown')
        
        with self.tracer.start_as_current_span("workflow_process_ticket") as span:
            span.set_attribute("ticket_id", ticket_id)
            span.set_input({"ticket_id": ticket_id})
            
            workflow_result = {
                "ticket_id": ticket_id,
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "stages": {}
            }
            
            try:
                # Stage 1: Memory Check
                self.logger.info(f"[{ticket_id}] Stage 1: Memory - Check for duplicates")
                memory_result = await self._check_duplicate(ticket)
                workflow_result["stages"]["memory"] = memory_result
                
                if memory_result.get("is_duplicate"):
                    self.logger.info(f"[{ticket_id}] ⚠️  Duplicate detected, skipping further processing")
                    workflow_result["skipped_reason"] = "duplicate"
                    workflow_result["success"] = True
                    span.set_attribute("skipped", True)
                    span.set_attribute("skip_reason", "duplicate")
                    return workflow_result
                
                # Stage 2: Triage Analysis
                self.logger.info(f"[{ticket_id}] Stage 2: Triage - Analyze & Synthesize")
                triage_result = await self._analyze_ticket(ticket)
                workflow_result["stages"]["triage"] = triage_result
                
                # Stage 3: Supervisor Decision
                self.logger.info(f"[{ticket_id}] Stage 3: Supervisor - Make Decision")
                decision = await self._make_decision(ticket, triage_result)
                workflow_result["stages"]["supervisor"] = decision
                
                # Stage 4: Executor Action
                self.logger.info(f"[{ticket_id}] Stage 4: Executor - Take Action")
                execution_result = await self._execute_action(ticket, decision, triage_result)
                workflow_result["stages"]["executor"] = execution_result
                
                workflow_result["success"] = execution_result.get("success", False)
                
                if workflow_result["success"]:
                    self.logger.info(f"[{ticket_id}] ✅ Workflow completed successfully")
                else:
                    self.logger.warning(f"[{ticket_id}] ⚠️  Workflow completed with errors")
                
                span.set_attribute("success", workflow_result["success"])
                span.set_output(workflow_result)
                
                return workflow_result
                
            except Exception as e:
                self.logger.error(f"[{ticket_id}] ❌ Workflow failed: {e}", exc_info=True)
                workflow_result["error"] = str(e)
                span.record_exception(e)
                span.set_attribute("success", False)
                return workflow_result
    
    async def _check_duplicate(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Check for duplicate tickets using Memory agent."""
        ticket_id = ticket.get('id', 'unknown')
        
        try:
            state = AgentState(
                agent_name="memory",
                input_data={
                    "ticket_id": ticket_id,
                    "description": ticket.get('description', ''),
                    "title": ticket.get('subject', '')
                }
            )
            
            result = await self.memory.process(state)
            
            return {
                "success": True,
                "is_duplicate": result.get('is_duplicate', False),
                "similar_ticket_id": result.get('similar_ticket_id'),
                "similarity_score": result.get('similarity_score', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"[{ticket_id}] Memory check failed: {e}")
            # Don't fail workflow on memory check failure
            return {
                "success": False,
                "is_duplicate": False,
                "error": str(e)
            }
    
    async def _analyze_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Analyze ticket using Triage agent."""
        ticket_id = ticket.get('id', 'unknown')
        
        state = AgentState(
            agent_name="triage",
            input_data={
                "ticket_id": ticket_id,
                "title": ticket.get('subject', ''),
                "description": ticket.get('description', ''),
                "severity": ticket.get('priority', 'medium'),
                "metadata": {
                    "requester_id": ticket.get('requester_id'),
                    "created_at": ticket.get('created_at'),
                    "tags": ticket.get('tags', [])
                }
            }
        )
        
        result = await self.triage.process(state)
        
        return {
            "success": True,
            "entities": result.get('entities', {}),
            "plan": result.get('plan', {}),
            "tool_results": result.get('tool_results', []),
            "synthesis": result.get('synthesis', {})
        }
    
    async def _make_decision(
        self, 
        ticket: Dict[str, Any], 
        triage_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 3: Make final decision using Supervisor agent."""
        ticket_id = ticket.get('id', 'unknown')
        
        state = AgentState(
            agent_name="supervisor",
            input_data={
                "ticket_id": ticket_id,
                "triage_result": triage_result,
                "synthesis": triage_result.get('synthesis', {}),
                "ticket_metadata": {
                    "priority": ticket.get('priority', 'medium'),
                    "requester_id": ticket.get('requester_id'),
                    "created_at": ticket.get('created_at'),
                    "tags": ticket.get('tags', [])
                }
            }
        )
        
        result = await self.supervisor.process(state)
        
        return {
            "success": True,
            "action": result.get('action', 'UNKNOWN'),
            "assigned_to": result.get('assigned_to'),
            "reason": result.get('reason', ''),
            "confidence": result.get('confidence', 'medium'),
            "synthesis_summary": result.get('synthesis_summary', ''),
            "root_cause": result.get('root_cause', 'unknown'),
            "recommended_actions": result.get('recommended_actions', [])
        }
    
    async def _execute_action(
        self,
        ticket: Dict[str, Any],
        decision: Dict[str, Any],
        triage_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 4: Execute decision using Executor agent."""
        ticket_id = ticket.get('id', 'unknown')
        
        state = AgentState(
            agent_name="executor",
            input_data={
                "ticket_id": ticket_id,
                "decision": decision,
                "context": {
                    "synthesis": triage_result.get('synthesis', {}),
                    "requester_id": ticket.get('requester_id'),
                    "ticket_metadata": {
                        "priority": ticket.get('priority'),
                        "created_at": ticket.get('created_at'),
                        "tags": ticket.get('tags', [])
                    }
                }
            }
        )
        
        result = await self.executor.process(state)
        
        return result
    
    async def process_batch(
        self, 
        tickets: list[Dict[str, Any]], 
        max_concurrent: int = 5
    ) -> list[Dict[str, Any]]:
        """
        Process multiple tickets concurrently.
        
        Args:
            tickets: List of tickets to process
            max_concurrent: Maximum concurrent workflows
        
        Returns:
            List of workflow results
        """
        self.logger.info(f"Processing batch of {len(tickets)} tickets (max_concurrent={max_concurrent})")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(ticket):
            async with semaphore:
                return await self.process_ticket(ticket)
        
        # Process all tickets concurrently (up to max_concurrent at a time)
        results = await asyncio.gather(
            *[process_with_semaphore(ticket) for ticket in tickets],
            return_exceptions=True
        )
        
        # Count successes and failures
        successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failures = len(results) - successes
        
        self.logger.info(f"Batch complete: {successes} succeeded, {failures} failed")
        
        return results
    
    async def run_continuous(self, poll_interval: int = 300):
        """
        Run continuous workflow (poll → process → repeat).
        
        Args:
            poll_interval: Seconds between polls
        """
        self.logger.info(f"Starting continuous workflow (poll_interval={poll_interval}s)")
        
        while True:
            try:
                # Fetch new tickets
                self.logger.info("Polling for new tickets...")
                
                poller_state = AgentState(
                    agent_name="poller",
                    input_data={"queue": "support_queue"}
                )
                
                poll_result = await self.poller.process(poller_state)
                tickets = poll_result.get('tickets', [])
                
                if tickets:
                    self.logger.info(f"Found {len(tickets)} new tickets")
                    await self.process_batch(tickets)
                else:
                    self.logger.info("No new tickets found")
                
            except Exception as e:
                self.logger.error(f"Continuous workflow error: {e}", exc_info=True)
            
            # Wait before next poll
            self.logger.info(f"Waiting {poll_interval}s before next poll...")
            await asyncio.sleep(poll_interval)


def create_workflow_orchestrator(
    config: AgentConfig,
    agents: Dict[str, Any]
) -> TicketWorkflowOrchestrator:
    """
    Factory function to create a workflow orchestrator from agent instances.
    
    Args:
        config: Agent configuration
        agents: Dictionary of agent instances
    
    Returns:
        TicketWorkflowOrchestrator instance
    """
    required_agents = ["poller", "memory", "triage", "supervisor", "executor"]
    
    for agent_name in required_agents:
        if agent_name not in agents:
            raise ValueError(f"Missing required agent: {agent_name}")
    
    return TicketWorkflowOrchestrator(
        poller_agent=agents["poller"],
        memory_agent=agents["memory"],
        triage_agent=agents["triage"],
        supervisor_agent=agents["supervisor"],
        executor_agent=agents["executor"],
        tracer=get_tracer()
    )

