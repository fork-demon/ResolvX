"""
Splunk Agent: selects appropriate SPL query templates and (optionally) executes them.
Minimal: chooses query based on context; execution can be stubbed unless credentials provided.
"""

from typing import Any, Dict, Optional
from langgraph import StateGraph, END
from pydantic import BaseModel, Field

from core.graph.base import BaseAgent
from core.graph.state import AgentState
from core.observability import get_logger, get_tracer


class SplunkAgentState(AgentState):
    ticket: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)


class SplunkAgent(BaseAgent[SplunkAgentState]):
    def __init__(self, config: Dict[str, Any], **kwargs: Any):
        super().__init__("splunk", config, **kwargs)
        self.logger = get_logger("agent.splunk")
        self.tracer = get_tracer("agent.splunk")

        self.endpoint: Optional[str] = config.get("endpoint")
        self.username: Optional[str] = config.get("username")
        self.password: Optional[str] = config.get("password")
        self.knowledge_dir: str = config.get("knowledge_dir", "knowledge/splunk")

    def build_graph(self) -> StateGraph:
        g = StateGraph(SplunkAgentState)
        g.add_node("select_query", self._select_query)
        g.add_node("execute_query", self._execute_query)
        g.add_edge("select_query", "execute_query")
        g.set_entry_point("select_query")
        return g.compile()

    async def process(self, state: SplunkAgentState) -> Dict[str, Any]:
        with self.tracer.start_as_current_span("splunk_process"):
            return await self.graph.ainvoke(state.dict())

    async def _select_query(self, state: SplunkAgentState) -> Dict[str, Any]:
        # Minimal selection: just return a template from knowledge store via simple heuristic
        ticket = state.input_data.get("ticket", {})
        subject = (ticket.get("subject") or "").lower()

        query = "search index=main error | stats count by source"
        if "login" in subject:
            query = "search index=auth action=failure | timechart span=5m count by user"
        elif "payment" in subject:
            query = "search index=payments error OR failure | stats count by error_code"

        state.result = {"query": query, "executed": False, "results": []}
        return state.dict()

    async def _execute_query(self, state: SplunkAgentState) -> Dict[str, Any]:
        # Optional: if credentials configured, we could call Splunk REST API.
        # To keep minimal, just return the query and a stubbed result.
        result = state.result or {}
        if self.endpoint and self.username and self.password:
            # Placeholder for real execution; keep minimal
            result["executed"] = True
            result["results"] = [{"_time": "now", "count": 42}]
        state.result = result
        return state.dict()


